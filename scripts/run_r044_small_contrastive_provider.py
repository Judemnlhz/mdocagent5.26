#!/usr/bin/env python3
"""R044 small contrastive provider run for prompt-visible artifact diagnosis.

This runner consumes R043 prompt previews and runs only the 22 R042 focus cases:
4 binary-divergent records plus 18 answer-text-different/binary-same records.

It is a diagnostic attribution run only. It is not full QA, not an official
MMLongBench result, and it does not report an official score.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any

DEFAULT_R043_ROOT = "outputs/heldout/r043_contrastive_prompt_exposure"
DEFAULT_R042_CASES = "outputs/heldout/r042_r040_manual_attribution/manual_attribution_cases.jsonl"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r044_small_contrastive_provider"
CONDITIONS = [
    "original_pages_only",
    "page_rerank_only",
    "original_pages_plus_artifact_snippets",
    "artifact_snippets_only",
]
EXPECTED_MAPPING = {
    "original_pages_only": {
        "retrieval_source_run": "top4_original_only",
        "prompt_contains_page_text": True,
        "prompt_contains_artifacts": False,
    },
    "page_rerank_only": {
        "retrieval_source_run": "top4_artifact_only",
        "prompt_contains_page_text": True,
        "prompt_contains_artifacts": False,
    },
    "original_pages_plus_artifact_snippets": {
        "retrieval_source_run": "top4_original_only",
        "prompt_contains_page_text": True,
        "prompt_contains_artifacts": True,
    },
    "artifact_snippets_only": {
        "retrieval_source_run": "top4_artifact_only",
        "prompt_contains_page_text": False,
        "prompt_contains_artifacts": True,
    },
}
FORBIDDEN_PREDICTION_KEYS = {
    "answer",
    "answers",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r043-root", default=DEFAULT_R043_ROOT)
    parser.add_argument("--r042-cases", default=DEFAULT_R042_CASES)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--provider-note", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    predictions_path = output_root / "predictions" / "r044_predictions.jsonl"
    gate_json = output_root / "r044_execution_gate.json"
    gate_md = output_root / "r044_execution_gate.md"
    report_json = output_root / "r044_diagnostic_attribution_report.json"
    report_md = output_root / "r044_diagnostic_attribution_report.md"
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "output_root": str(output_root),
                    "conditions": CONDITIONS,
                    "target_policy": "R042 4 binary-divergent + 18 answer-text-different/binary-same cases",
                    "predictions_path": str(predictions_path),
                    "not_full_qa": True,
                    "not_official_score": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    r043_root = Path(args.r043_root)
    r043_manifest = read_json(r043_root / "r043_prompt_exposure_manifest.json")
    r043_gate = read_json(r043_root / "r043_prompt_exposure_gate.json")
    focus_cases = read_jsonl(Path(args.r042_cases))
    target_ids = sorted({int(case["record_id"]) for case in focus_cases})
    focus_case_by_id = {int(case["record_id"]): case for case in focus_cases}
    records = read_json(Path(args.records))
    previews = load_target_previews(r043_root, target_ids)
    mapping_audit = audit_r043_mapping(r043_manifest, r043_gate, previews)
    existing = load_existing_predictions(predictions_path)
    predictions = run_predictions(args, previews, existing, predictions_path)
    write_jsonl(predictions_path, predictions)
    gate = build_gate(args, mapping_audit, target_ids, previews, predictions)
    report = build_report(args, target_ids, records, focus_case_by_id, predictions, gate, mapping_audit)
    write_json(gate_json, gate)
    write_gate_markdown(gate_md, gate)
    write_json(report_json, report)
    write_report_markdown(report_md, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "gate_passed": gate["gate_passed"],
                "num_target_records": len(target_ids),
                "num_predictions": len(predictions),
                "conditions": CONDITIONS,
                "report_md": str(report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_target_previews(r043_root: Path, target_ids: list[int]) -> dict[str, dict[int, dict[str, Any]]]:
    target_set = set(target_ids)
    previews: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = r043_root / "prompt_previews" / f"{condition}.jsonl"
        rows = {}
        for row in read_jsonl(path):
            record_id = int(row["record_id"])
            if record_id in target_set:
                rows[record_id] = row
        previews[condition] = rows
    return previews


def audit_r043_mapping(
    r043_manifest: dict[str, Any],
    r043_gate: dict[str, Any],
    previews: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    condition_checks = {}
    for condition in CONDITIONS:
        expected = EXPECTED_MAPPING[condition]
        rows = list(previews[condition].values())
        manifest_row = (r043_manifest.get("conditions") or {}).get(condition) or {}
        condition_checks[condition] = {
            "retrieval_source_run_matches": manifest_row.get("retrieval_source_run") == expected["retrieval_source_run"]
            and all(row.get("retrieval_source_run") == expected["retrieval_source_run"] for row in rows),
            "page_text_exposure_matches": all(row["exposure"]["prompt_contains_page_text"] == expected["prompt_contains_page_text"] for row in rows),
            "artifact_exposure_matches": all(row["exposure"]["prompt_contains_artifacts"] == expected["prompt_contains_artifacts"] for row in rows),
            "num_focus_rows": len(rows),
        }
    flat = [value for row in condition_checks.values() for key, value in row.items() if key != "num_focus_rows"]
    return {
        "r043_gate_passed": bool(r043_gate.get("gate_passed")),
        "mapping_confirmed": bool(r043_gate.get("gate_passed")) and all(flat),
        "expected_mapping": EXPECTED_MAPPING,
        "condition_checks": condition_checks,
    }


def run_predictions(
    args: argparse.Namespace,
    previews: dict[str, dict[int, dict[str, Any]]],
    existing: dict[tuple[str, int, str], dict[str, Any]],
    predictions_path: Path,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)
    output: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for condition in CONDITIONS:
        for record_id in sorted(previews[condition]):
            preview = previews[condition][record_id]
            prompt_hash = preview["prompt_preview_sha256"]
            key = (condition, record_id, prompt_hash)
            if key in existing:
                if key not in seen_keys:
                    output.append(existing[key])
                    seen_keys.add(key)
                continue
            response_text = call_provider(client, args, preview["prompt_preview"])
            row = {
                "schema_version": "r044_prediction_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "condition": condition,
                "record_id": record_id,
                "doc_id": preview["doc_id"],
                "question": preview["question"],
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "prompt_preview_sha256": prompt_hash,
                "retrieval_source_run": preview["retrieval_source_run"],
                "exposure": preview["exposure"],
                "prediction_text": response_text,
                "not_official_score": True,
            }
            output.append(row)
            seen_keys.add(key)
            write_jsonl(predictions_path, output)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    return output


def call_provider(client: OpenAI, args: argparse.Namespace, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(max(int(args.max_retries), 1)):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(int(args.max_retries), 1):
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Provider call failed after {args.max_retries} attempts: {last_error}")


def build_gate(
    args: argparse.Namespace,
    mapping_audit: dict[str, Any],
    target_ids: list[int],
    previews: dict[str, dict[int, dict[str, Any]]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_keys = {(row["condition"], int(row["record_id"])) for row in predictions}
    expected_keys = {(condition, record_id) for condition in CONDITIONS for record_id in target_ids}
    preview_hash = {
        (condition, record_id): previews[condition][record_id]["prompt_preview_sha256"]
        for condition in CONDITIONS
        for record_id in target_ids
    }
    checks = {
        "r043_condition_mapping_confirmed": mapping_audit["mapping_confirmed"],
        "target_records_exactly_22": len(target_ids) == 22,
        "four_conditions_present": sorted({row["condition"] for row in predictions}) == sorted(CONDITIONS),
        "predictions_complete_4x22": len(predictions) == 88 and prediction_keys == expected_keys,
        "prompt_hashes_match_r043": all(row["prompt_preview_sha256"] == preview_hash[(row["condition"], int(row["record_id"]))] for row in predictions),
        "prediction_records_have_no_gold_fields": all(not forbidden_keys(row) for row in predictions),
        "not_full_qa": True,
        "not_official_score": all(row.get("not_official_score") is True for row in predictions),
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r044_execution_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r044_small_contrastive_gate_pass" if not hard_failures else "r044_small_contrastive_gate_fail",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_target_records": len(target_ids),
        "num_predictions": len(predictions),
        "conditions": CONDITIONS,
        "mapping_audit": mapping_audit,
        "model": args.model,
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(
    args: argparse.Namespace,
    target_ids: list[int],
    records: list[dict[str, Any]],
    focus_case_by_id: dict[int, dict[str, Any]],
    predictions: list[dict[str, Any]],
    gate: dict[str, Any],
    mapping_audit: dict[str, Any],
) -> dict[str, Any]:
    by_record: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        by_record[int(row["record_id"])][row["condition"]] = row
    per_record = []
    transition_counts: Counter[str] = Counter()
    condition_match_counts: Counter[str] = Counter()
    for record_id in target_ids:
        source = records[record_id]
        focus = focus_case_by_id[record_id]
        condition_rows = by_record[record_id]
        diagnostics = {
            condition: answer_diagnostic(condition_rows[condition]["prediction_text"], source.get("answer"), source.get("answer_format"))
            for condition in CONDITIONS
        }
        for condition, diag in diagnostics.items():
            if diag["diagnostic_gold_match"]:
                condition_match_counts[condition] += 1
        transitions = transition_labels(diagnostics)
        transition_counts.update(transitions)
        per_record.append(
            {
                "record_id": record_id,
                "case_type": focus["case_type"],
                "r042_label": focus["manual_attribution"]["label"],
                "question": source.get("question"),
                "gold_answer_for_posthoc_diagnostic_only": source.get("answer"),
                "answer_format": source.get("answer_format"),
                "condition_predictions": {condition: condition_rows[condition]["prediction_text"] for condition in CONDITIONS},
                "diagnostic_gold_match": {condition: diagnostics[condition]["diagnostic_gold_match"] for condition in CONDITIONS},
                "match_level": {condition: diagnostics[condition]["match_level"] for condition in CONDITIONS},
                "transition_labels": transitions,
            }
        )
    return {
        "schema_version": "r044_diagnostic_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r044_small_contrastive_diagnostic_complete" if gate["gate_passed"] else "r044_small_contrastive_diagnostic_needs_review",
        "scope": {
            "small_contrastive_provider_run": True,
            "target_records_only": 22,
            "no_full_qa": True,
            "not_official_mmlongbench_result": True,
            "no_official_score_reported": True,
            "gold_used_only_for_posthoc_diagnostic_attribution": True,
        },
        "inputs": {
            "r043_root": args.r043_root,
            "r042_cases": args.r042_cases,
            "records": args.records,
        },
        "model": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "provider_note": args.provider_note.strip(),
        },
        "gate": gate,
        "mapping_audit": mapping_audit,
        "num_target_records": len(target_ids),
        "num_predictions": len(predictions),
        "condition_diagnostic_match_counts_not_official_scores": dict(sorted(condition_match_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "per_record": per_record,
        "interpretation": interpret(transition_counts, condition_match_counts),
        "recommended_next": [
            "Manually inspect records where artifact snippets improve over original pages and where snippet-only fails despite plus succeeding.",
            "Do not promote these counts to official scores; they are prompt-exposure diagnostics on 22 selected cases.",
            "If the diagnostic pattern is coherent, implement a controlled R045 with fixed prompt templates and explicit support/citation rubric before any full-data run.",
        ],
    }


def transition_labels(diagnostics: dict[str, dict[str, Any]]) -> list[str]:
    labels = []
    original = diagnostics["original_pages_only"]["diagnostic_gold_match"]
    rerank = diagnostics["page_rerank_only"]["diagnostic_gold_match"]
    plus = diagnostics["original_pages_plus_artifact_snippets"]["diagnostic_gold_match"]
    snippet = diagnostics["artifact_snippets_only"]["diagnostic_gold_match"]
    if rerank and not original:
        labels.append("page_rerank_gain_vs_original")
    if original and not rerank:
        labels.append("page_rerank_loss_vs_original")
    if plus and not original:
        labels.append("artifact_injection_gain_vs_original")
    if original and not plus:
        labels.append("artifact_injection_loss_vs_original")
    if snippet:
        labels.append("snippet_only_sufficient")
    if plus and not snippet:
        labels.append("page_text_plus_artifacts_needed")
    if snippet and not plus:
        labels.append("snippet_only_beats_plus")
    if not any([original, rerank, plus, snippet]):
        labels.append("all_conditions_miss")
    return labels


def interpret(transition_counts: Counter[str], condition_match_counts: Counter[str]) -> dict[str, Any]:
    return {
        "bottom_line": "R044 is a 22-case diagnostic contrast over prompt-visible exposure; counts are not official scores.",
        "artifact_injection_gain_count": int(transition_counts.get("artifact_injection_gain_vs_original", 0)),
        "artifact_injection_loss_count": int(transition_counts.get("artifact_injection_loss_vs_original", 0)),
        "page_rerank_gain_count": int(transition_counts.get("page_rerank_gain_vs_original", 0)),
        "page_rerank_loss_count": int(transition_counts.get("page_rerank_loss_vs_original", 0)),
        "snippet_only_sufficient_count": int(transition_counts.get("snippet_only_sufficient", 0)),
        "condition_match_counts_not_official_scores": dict(sorted(condition_match_counts.items())),
    }


def answer_diagnostic(answer: str, gold: Any, answer_format: Any) -> dict[str, Any]:
    answer_text = "" if answer is None else str(answer)
    gold_text = "" if gold is None else str(gold)
    answer_norm = normalize(answer_text)
    gold_norm = normalize(gold_text)
    list_answer = parse_list_like(answer_text)
    list_gold = parse_list_like(gold_text)
    numeric_answer = first_number(answer_text)
    numeric_gold = first_number(gold_text)
    if answer_norm == gold_norm:
        match_level = "normalized_exact"
    elif list_answer is not None and list_gold is not None and [normalize(x) for x in list_answer] == [normalize(x) for x in list_gold]:
        match_level = "list_exact"
    elif numeric_answer is not None and numeric_gold is not None and abs(numeric_answer - numeric_gold) <= 1e-6:
        match_level = "numeric_exact"
    elif gold_norm == "not answerable" and any(phrase in answer_norm for phrase in ["not answerable", "not listed", "not provided", "cannot be determined", "no information"]):
        match_level = "unanswerable_equivalent"
    else:
        match_level = "mismatch"
    return {
        "match_level": match_level,
        "diagnostic_gold_match": match_level != "mismatch",
        "answer_format": answer_format,
    }


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R044 Small Contrastive Execution Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- 22 R042 focus records only.",
        "- Four R043 conditions only.",
        "- Diagnostic attribution only; not full QA and not an official score.",
        "",
        "## Checks",
    ]
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## R043 Mapping Confirmation"])
    for condition, row in gate["mapping_audit"]["condition_checks"].items():
        expected = gate["mapping_audit"]["expected_mapping"][condition]
        lines.append(
            f"- `{condition}`: expected retrieval `{expected['retrieval_source_run']}`, "
            f"expected page_text={expected['prompt_contains_page_text']}, "
            f"expected artifacts={expected['prompt_contains_artifacts']}; "
            f"mapping checks retrieval={row['retrieval_source_run_matches']}, "
            f"page_text={row['page_text_exposure_matches']}, "
            f"artifacts={row['artifact_exposure_matches']}, focus_rows={row['num_focus_rows']}"
        )
    write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R044 Small Contrastive Diagnostic Attribution",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- 22 selected R042 focus records only.",
        "- Diagnostic attribution only.",
        "- Not full QA and not an official MMLongBench result.",
        "- Counts below are diagnostic gold-match counts, not official scores.",
        "",
        "## Diagnostic Counts",
        f"- model: `{report['model']['model']}`",
        f"- temperature: `{report['model']['temperature']}`",
        f"- max_tokens: `{report['model']['max_tokens']}`",
        f"- provider note: {report['model']['provider_note'] or 'none'}",
        f"- predictions: {report['num_predictions']}",
        f"- target records: {report['num_target_records']}",
        f"- condition diagnostic match counts: `{json.dumps(report['condition_diagnostic_match_counts_not_official_scores'], sort_keys=True)}`",
        f"- transition counts: `{json.dumps(report['transition_counts'], sort_keys=True)}`",
        "",
        "## Interpretation",
        f"- {report['interpretation']['bottom_line']}",
        f"- artifact injection gains vs original: {report['interpretation']['artifact_injection_gain_count']}",
        f"- artifact injection losses vs original: {report['interpretation']['artifact_injection_loss_count']}",
        f"- page rerank gains vs original: {report['interpretation']['page_rerank_gain_count']}",
        f"- page rerank losses vs original: {report['interpretation']['page_rerank_loss_count']}",
        f"- snippet-only sufficient cases: {report['interpretation']['snippet_only_sufficient_count']}",
        "",
        "## Recommended Next",
    ]
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def load_existing_predictions(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = read_jsonl(path)
    return {
        (str(row["condition"]), int(row["record_id"]), str(row["prompt_preview_sha256"])): row
        for row in rows
    }


def forbidden_keys(value: Any) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in FORBIDDEN_PREDICTION_KEYS:
                found.append(str(key))
            found.extend(forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(forbidden_keys(item))
    return found


def parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_list_like(value: str) -> list[str] | None:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return None


def first_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
