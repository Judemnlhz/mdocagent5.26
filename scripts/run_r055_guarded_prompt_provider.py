#!/usr/bin/env python3
"""R055 tiny guarded prompt provider diagnostic.

Runs only records 384, 508, and 569 from R054 guarded prompt previews.
This is a provider diagnostic for guard-prompt refusal behavior only: it can
show whether the model follows the guard and avoids artifact-noise traps, but
it cannot show that artifacts improve retrieval or QA.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any

DEFAULT_R054_PREVIEWS = "outputs/heldout/r054_guarded_selector_repair/r054_guarded_prompt_previews.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r055_guarded_prompt_provider_diagnostic"
TARGET_RECORD_IDS = [384, 508, 569]
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
REFUSAL_PATTERNS = [
    "not answerable",
    "cannot be determined",
    "can't be determined",
    "insufficient",
    "not provided",
    "not visible",
    "not listed",
    "does not provide",
    "no evidence",
    "missing",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r054-previews", default=DEFAULT_R054_PREVIEWS)
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
    predictions_path = output_root / "predictions" / "r055_predictions.jsonl"
    gate_json = output_root / "r055_guarded_prompt_provider_gate.json"
    gate_md = output_root / "r055_guarded_prompt_provider_gate.md"
    report_json = output_root / "r055_guarded_prompt_provider_report.json"
    report_md = output_root / "r055_guarded_prompt_provider_report.md"
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "target_record_ids": TARGET_RECORD_IDS,
            "provider_calls_planned": len(TARGET_RECORD_IDS),
            "proves_only": "guard prompt refusal/noise-avoidance behavior on 3 cases",
            "does_not_prove": "artifact positive lift, retrieval improvement, full QA, or official MMLongBench score",
            "not_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    previews = load_target_previews(Path(args.r054_previews))
    existing = load_existing_predictions(predictions_path)
    predictions = run_predictions(args, previews, existing, predictions_path)
    write_jsonl(predictions_path, predictions)
    diagnostics = build_diagnostics(predictions)
    gate = build_gate(args, previews, predictions, diagnostics)
    report = build_report(args, previews, predictions, diagnostics, gate)
    write_json(gate_json, gate)
    write_gate_markdown(gate_md, gate)
    write_json(report_json, report)
    write_report_markdown(report_md, report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_predictions": len(predictions),
        "report_md": str(report_md),
        "not_full_qa": True,
        "not_official_score": True,
        "does_not_prove_artifact_lift": True,
    }, ensure_ascii=False, indent=2))


def load_target_previews(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if int(row["record_id"]) in TARGET_RECORD_IDS]
    rows = sorted(rows, key=lambda row: TARGET_RECORD_IDS.index(int(row["record_id"])))
    found = [int(row["record_id"]) for row in rows]
    if found != TARGET_RECORD_IDS:
        raise ValueError(f"Expected target records {TARGET_RECORD_IDS}, found {found}")
    return rows


def run_predictions(
    args: argparse.Namespace,
    previews: list[dict[str, Any]],
    existing: dict[tuple[int, str], dict[str, Any]],
    predictions_path: Path,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)
    output: list[dict[str, Any]] = []
    for preview in previews:
        record_id = int(preview["record_id"])
        prompt_hash = str(preview["prompt_preview_sha256"])
        key = (record_id, prompt_hash)
        if key in existing:
            output.append(existing[key])
            continue
        response_text = call_provider(client, args, preview["prompt_preview"])
        row = {
            "schema_version": "r055_guarded_prompt_prediction_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "record_id": record_id,
            "doc_id": preview["doc_id"],
            "question": preview["question"],
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_preview_sha256": prompt_hash,
            "r054_guard_decision": preview["guard_decision"],
            "r054_answer_policy": preview["answer_policy"],
            "r054_selected_artifact_count": preview["selected_artifact_count"],
            "prediction_text": response_text,
            "not_full_qa": True,
            "not_official_score": True,
            "does_not_prove_artifact_lift": True,
        }
        output.append(row)
        write_jsonl(predictions_path, output)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    return output


def call_provider(client: Any, args: argparse.Namespace, prompt: str) -> str:
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


def build_diagnostics(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in predictions:
        record_id = int(row["record_id"])
        text = normalize(row["prediction_text"])
        refusal_like = any(pattern in text for pattern in REFUSAL_PATTERNS)
        if record_id == 384:
            passed = refusal_like and "strategic planning services team" not in text
            expected = "Refuse May 2018 metadata lookup or state visible metadata mismatch; avoid Strategic Planning Services Team false positive."
            failure_modes = []
            if "strategic planning services team" in text:
                failure_modes.append("artifact_noise_false_positive")
            if not refusal_like:
                failure_modes.append("no_refusal_or_metadata_mismatch")
        elif record_id == 508:
            no_ar03 = "ar03" in text and any(pattern in text for pattern in ["not listed", "not provided", "not explicitly", "cannot be determined", "not answerable", "no evidence"])
            noisy_market = any(term in text for term in ["little rock", "northern arkansas"]) and not no_ar03
            passed = (refusal_like or no_ar03) and not noisy_market
            expected = "Refuse or state exact AR03 geographic market is absent; avoid assigning AR01/AR02 market names to AR03."
            failure_modes = []
            if noisy_market:
                failure_modes.append("wrong_market_from_neighbor_code")
            if not (refusal_like or no_ar03):
                failure_modes.append("no_exact_code_absence_refusal")
        elif record_id == 569:
            mentions_children_missing = "children" in text and any(pattern in text for pattern in ["missing", "not provided", "no data", "does not provide", "cannot", "insufficient"])
            numeric_answer = bool(re.search(r"final answer\s*[:\-]?\s*[-+]?\d+(?:\.\d+)?\s*%?", text))
            passed = (refusal_like or mentions_children_missing) and not numeric_answer
            expected = "Refuse calculation because operands are incomplete, especially children operand; do not output a computed percentage."
            failure_modes = []
            if numeric_answer:
                failure_modes.append("computed_from_incomplete_operands")
            if not (refusal_like or mentions_children_missing):
                failure_modes.append("no_operand_incomplete_refusal")
        else:
            raise ValueError(f"Unexpected record_id: {record_id}")
        rows.append({
            "record_id": record_id,
            "expected_guard_behavior": expected,
            "guard_behavior_passed": bool(passed),
            "refusal_like": bool(refusal_like),
            "failure_modes": failure_modes,
        })
    return rows


def build_gate(
    args: argparse.Namespace,
    previews: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "target_records_exactly_384_508_569": [int(row["record_id"]) for row in previews] == TARGET_RECORD_IDS,
        "provider_predictions_exactly_3": len(predictions) == 3 and sorted(int(row["record_id"]) for row in predictions) == TARGET_RECORD_IDS,
        "prompt_hashes_match_r054": all(
            pred["prompt_preview_sha256"] == preview["prompt_preview_sha256"]
            for pred, preview in zip(sorted(predictions, key=lambda r: TARGET_RECORD_IDS.index(int(r["record_id"]))), previews)
        ),
        "uses_r054_zero_artifact_guarded_prompts": all(int(preview["selected_artifact_count"]) == 0 for preview in previews),
        "prediction_records_have_no_gold_fields": all(not forbidden_keys(row) for row in predictions),
        "all_guard_behaviors_passed": all(row["guard_behavior_passed"] for row in diagnostics),
        "scope_limited_to_guard_prompt_refusal_noise_avoidance": True,
        "does_not_claim_artifact_positive_lift": True,
        "not_full_qa": True,
        "not_official_score": all(row.get("not_official_score") is True for row in predictions),
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r055_guarded_prompt_provider_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r055_guarded_prompt_provider_gate_pass" if not hard_failures else "r055_guarded_prompt_provider_gate_needs_review",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "target_record_ids": TARGET_RECORD_IDS,
        "num_predictions": len(predictions),
        "model": args.model,
        "not_full_qa": True,
        "not_official_score": True,
        "does_not_prove_artifact_lift": True,
    }


def build_report(
    args: argparse.Namespace,
    previews: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    diag_by_id = {int(row["record_id"]): row for row in diagnostics}
    pred_by_id = {int(row["record_id"]): row for row in predictions}
    preview_by_id = {int(row["record_id"]): row for row in previews}
    per_record = []
    for record_id in TARGET_RECORD_IDS:
        preview = preview_by_id[record_id]
        pred = pred_by_id[record_id]
        diag = diag_by_id[record_id]
        per_record.append({
            "record_id": record_id,
            "question": preview["question"],
            "r054_guard_decision": preview["guard_decision"],
            "r054_answer_policy": preview["answer_policy"],
            "r054_selected_artifact_count": preview["selected_artifact_count"],
            "prediction_text": pred["prediction_text"],
            "guard_behavior": diag,
        })
    pass_count = sum(1 for row in diagnostics if row["guard_behavior_passed"])
    return {
        "schema_version": "r055_guarded_prompt_provider_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r055_guarded_prompt_provider_complete" if gate["gate_passed"] else "r055_guarded_prompt_provider_needs_manual_review",
        "scope": {
            "tiny_provider_diagnostic": True,
            "target_records_only": TARGET_RECORD_IDS,
            "provider_calls": len(predictions),
            "uses_r054_guarded_prompt_previews": True,
            "tests_guard_prompt_refusal_noise_avoidance_only": True,
            "does_not_prove_artifact_positive_lift": True,
            "does_not_compare_retrieval_conditions": True,
            "no_full_qa": True,
            "not_official_mmlongbench_result": True,
            "no_official_score_reported": True,
        },
        "inputs": {"r054_previews": args.r054_previews},
        "model": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "provider_note": args.provider_note.strip(),
        },
        "gate": gate,
        "guard_behavior_pass_count_not_score": pass_count,
        "guard_behavior_total": len(diagnostics),
        "diagnostic_counts_not_scores": dict(Counter("pass" if row["guard_behavior_passed"] else "needs_review" for row in diagnostics)),
        "per_record": per_record,
        "interpretation": {
            "bottom_line": "R055 only tests whether the R054 guard prompts make the provider refuse or avoid artifact-noise traps on 3 manually accepted cases.",
            "artifact_lift_claim": "unsupported_by_this_run",
            "retrieval_improvement_claim": "unsupported_by_this_run",
            "official_score_claim": "unsupported_by_this_run",
        },
        "recommended_next": [
            "If R055 passes, keep the guard as a candidate prompt-control component for later controlled diagnostics.",
            "Do not claim artifact-aware retrieval improves from R055; it has zero retrieval-condition comparison and only three guarded prompts.",
            "Before any broader run, decide whether to integrate these guards into the selector/prompt path or run another tiny contrastive diagnostic with explicit positive evidence cases.",
        ],
    }


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R055 Guarded Prompt Provider Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- Records 384, 508, and 569 only.",
        "- Provider diagnostic on R054 guarded prompt previews only.",
        "- Tests refusal/noise-avoidance behavior only.",
        "- Does not prove artifact positive lift, retrieval improvement, full QA, or official score.",
        "",
        "## Checks",
    ]
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        for item in gate["hard_failures"]:
            lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R055 Guarded Prompt Provider Diagnostic",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- 3 records only: 384, 508, 569.",
        "- Provider diagnostic on R054 guarded prompts only.",
        "- Can only show whether guard prompts make the model refuse/avoid artifact noise.",
        "- Cannot show artifact positive lift, retrieval improvement, full QA, or an official MMLongBench score.",
        "",
        "## Diagnostic Counts",
        f"- model: `{report['model']['model']}`",
        f"- predictions: {report['scope']['provider_calls']}",
        f"- guard behavior pass count, not score: {report['guard_behavior_pass_count_not_score']} / {report['guard_behavior_total']}",
        f"- diagnostic counts, not scores: `{json.dumps(report['diagnostic_counts_not_scores'], sort_keys=True)}`",
        "",
        "## Per Record",
    ]
    for row in report["per_record"]:
        behavior = row["guard_behavior"]
        lines.append(
            f"- {row['record_id']}: guard=`{row['r054_guard_decision']}`, "
            f"passed={behavior['guard_behavior_passed']}, failures=`{behavior['failure_modes']}`"
        )
    lines.extend(["", "## Interpretation"])
    for key, value in report["interpretation"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def load_existing_predictions(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    return {(int(row["record_id"]), str(row["prompt_preview_sha256"])): row for row in read_jsonl(path)}


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


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


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