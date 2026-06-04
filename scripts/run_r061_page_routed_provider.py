#!/usr/bin/env python3
"""R061 tiny page-routed provider diagnostic.

Runs only records 223 and 227 from R060 page/artifact routing prompt previews.
This checks whether the provider follows page-only routing when artifacts are
dimension-guarded. It does not run full QA, official evaluation, or report a
score, and it does not make an artifact-lift claim.
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

DEFAULT_R060_PREVIEWS = "outputs/heldout/r060_page_artifact_routing_audit/r060_routing_prompt_previews.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r061_page_routed_provider_diagnostic"
TARGET_RECORD_IDS = [223, 227]
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
REJECTED_ARTIFACT_PATTERNS = [
    "atomicizer_",
    "artifact id:",
    "artifact ids:",
    "artifact_id",
    "selected artifact",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r060-previews", default=DEFAULT_R060_PREVIEWS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--provider-note", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-page-contexts", type=int, default=4)
    parser.add_argument("--max-page-chars", type=int, default=700)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    predictions_path = output_root / "predictions" / "r061_predictions.jsonl"
    gate_json = output_root / "r061_page_routed_provider_gate.json"
    gate_md = output_root / "r061_page_routed_provider_gate.md"
    report_json = output_root / "r061_page_routed_provider_report.json"
    report_md = output_root / "r061_page_routed_provider_report.md"
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "target_record_ids": TARGET_RECORD_IDS,
            "provider_calls_planned": len(TARGET_RECORD_IDS),
            "proves_only": "provider adherence to R060 page-only routing on 2 prompts",
            "does_not_prove": "artifact positive lift, retrieval improvement, full QA, or official MMLongBench score",
            "not_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    previews = load_target_previews(Path(args.r060_previews))
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
        "page_only_routing_pass_count_not_score": report["page_only_routing_pass_count_not_score"],
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
    for row in rows:
        if row.get("guard_decision") != "artifact_dimension_support_guard":
            raise ValueError(f"R060 preview is not dimension-guarded: {row['record_id']}")
        if int(row.get("selected_artifact_count") or 0) != 0:
            raise ValueError(f"R060 preview selected artifacts unexpectedly: {row['record_id']}")
        if row.get("answer_policy") != "use_page_evidence_or_refuse":
            raise ValueError(f"R060 preview has unexpected answer policy: {row['record_id']}")
        if not row.get("routing_checks", {}).get("passed"):
            raise ValueError(f"R060 routing checks did not pass: {row['record_id']}")
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
        provider_prompt = build_provider_prompt(preview, args)
        response_text = call_provider(client, args, provider_prompt)
        row = {
            "schema_version": "r061_page_routed_prediction_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "record_id": record_id,
            "doc_id": preview["doc_id"],
            "question": preview["question"],
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "r060_prompt_preview_sha256": prompt_hash,
            "prompt_preview_sha256": prompt_hash,
            "provider_prompt_sha256": sha256(provider_prompt),
            "provider_prompt_chars": len(provider_prompt),
            "provider_prompt_mode": "r060_derived_compact_page_routing_prompt",
            "max_page_contexts": args.max_page_contexts,
            "max_page_chars": args.max_page_chars,
            "r060_guard_decision": preview["guard_decision"],
            "r060_answer_policy": preview["answer_policy"],
            "r060_selected_artifact_count": preview["selected_artifact_count"],
            "r060_visible_page_support_sufficient": preview["support_audit"]["visible_support_sufficient"],
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


def build_provider_prompt(preview: dict[str, Any], args: argparse.Namespace) -> str:
    page_contexts = list(preview.get("page_contexts") or [])
    selected_contexts = rank_page_contexts(page_contexts, str(preview.get("question") or ""))[: max(int(args.max_page_contexts), 1)]
    lines = [
        "[R061 compact provider diagnostic derived from R060 prompt preview]",
        "Use only the visible page evidence below.",
        "The R060 selector rejected all artifact snippets with artifact_dimension_support_guard.",
        "Do not cite rejected artifact ids. Do not use artifact evidence.",
        "If the page evidence fully supports the answer, answer from cited page ids only.",
        "If the page evidence does not fully support the answer, say Not answerable and cite what is missing.",
        f"Question: {preview['question']}",
        "",
        "[Guard decision]",
        f"decision={preview['guard_decision']}; answer_policy={preview['answer_policy']}; selected_artifact_count={preview['selected_artifact_count']}; reasons={preview.get('guard_reasons', [])}",
        "",
        "[Page evidence]",
    ]
    for ctx in selected_contexts:
        text = " ".join(str(ctx.get("text_preview") or "").split())[: max(int(args.max_page_chars), 120)]
        lines.append(f"Page {ctx['page_index']} ({'present' if ctx.get('exists') else 'missing'}): {text}")
    lines.extend([
        "",
        "[Selected artifact evidence]",
        "None. Artifact snippets were rejected by the guard and must not be cited.",
        "",
        "[Required response format]",
        "Page evidence: cite page ids or state none.",
        "Artifact evidence: none.",
        "Guard check: state that artifacts were rejected and not used.",
        "Unsupported-answer check: say whether page evidence fully supports the answer.",
        "Final answer: answer or Not answerable.",
    ])
    return "\n".join(lines).strip() + "\n"


def rank_page_contexts(page_contexts: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    q_tokens = content_tokens(question)

    def score(ctx: dict[str, Any]) -> tuple[int, int, int]:
        text = str(ctx.get("text_preview") or "")
        overlap = len(q_tokens & content_tokens(text))
        # Favor pages that contain numeric/table-like evidence when overlap ties.
        numeric_hits = len(re.findall(r"\d+(?:\.\d+)?\s*%?", text))
        return (-overlap, -numeric_hits, int(ctx.get("page_index") or 0))

    return sorted(page_contexts, key=score)


def content_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "are", "what", "which", "with", "from", "into", "this", "that",
        "them", "their", "please", "write", "answer", "format", "center", "project", "survey",
    }
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", normalize(text)) if len(token) > 2 and token not in stop}


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
        text = normalize(row["prediction_text"])
        refusal_like = any(pattern in text for pattern in REFUSAL_PATTERNS)
        page_citation_like = bool(re.search(r"\bpage\s+\d+\b", text)) or "#p" in text
        rejected_artifact_citation = any(pattern in text for pattern in REJECTED_ARTIFACT_PATTERNS)
        artifact_none = artifact_evidence_none(text)
        page_answer_routed = page_citation_like and not rejected_artifact_citation
        refusal_routed = refusal_like and not rejected_artifact_citation
        routing_behavior_passed = (page_answer_routed or refusal_routed) and (artifact_none or "artifact evidence" not in text)
        failure_modes = []
        if not page_citation_like and not refusal_like:
            failure_modes.append("missing_page_citation")
        if rejected_artifact_citation:
            failure_modes.append("cited_or_invoked_rejected_artifact")
        if not artifact_none and "artifact evidence" in text:
            failure_modes.append("artifact_evidence_not_none")
        rows.append({
            "record_id": int(row["record_id"]),
            "expected_routing_behavior": "Use cited page evidence only, do not cite rejected artifacts, and mark artifact evidence as none if mentioned.",
            "routing_behavior_passed": bool(routing_behavior_passed),
            "page_citation_like": bool(page_citation_like),
            "page_answer_routed": bool(page_answer_routed),
            "refusal_routed": bool(refusal_routed),
            "artifact_evidence_none": bool(artifact_none),
            "rejected_artifact_citation": bool(rejected_artifact_citation),
            "refusal_like": bool(refusal_like),
            "failure_modes": failure_modes,
        })
    return rows


def artifact_evidence_none(text: str) -> bool:
    if "artifact evidence" not in text:
        return True
    patterns = [
        r"artifact evidence\s*[:\-]\s*(none|no|not used|state none)",
        r"artifact evidence\s*[:\-]\s*no selected artifact",
        r"artifact evidence\s*[:\-]\s*no artifact evidence",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def build_gate(
    args: argparse.Namespace,
    previews: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    sorted_predictions = sorted(predictions, key=lambda row: TARGET_RECORD_IDS.index(int(row["record_id"])))
    checks = {
        "target_records_exactly_223_227": [int(row["record_id"]) for row in previews] == TARGET_RECORD_IDS,
        "provider_predictions_exactly_2": len(predictions) == 2 and sorted(int(row["record_id"]) for row in predictions) == TARGET_RECORD_IDS,
        "prompt_hashes_match_r060": all(
            pred["prompt_preview_sha256"] == preview["prompt_preview_sha256"]
            for pred, preview in zip(sorted_predictions, previews)
        ),
        "uses_r060_page_routed_zero_artifact_prompts": all(
            preview["guard_decision"] == "artifact_dimension_support_guard"
            and int(preview["selected_artifact_count"]) == 0
            and preview["answer_policy"] == "use_page_evidence_or_refuse"
            for preview in previews
        ),
        "prediction_records_have_no_gold_fields": all(not forbidden_keys(row) for row in predictions),
        "all_routing_behaviors_passed": all(row["routing_behavior_passed"] for row in diagnostics),
        "scope_limited_to_page_routed_provider_diagnostic": True,
        "does_not_claim_artifact_positive_lift": True,
        "not_full_qa": True,
        "not_official_score": all(row.get("not_official_score") is True for row in predictions),
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r061_page_routed_provider_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r061_page_routed_provider_gate_pass" if not hard_failures else "r061_page_routed_provider_needs_prompt_fix",
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
            "r060_guard_decision": preview["guard_decision"],
            "r060_answer_policy": preview["answer_policy"],
            "r060_selected_artifact_count": preview["selected_artifact_count"],
            "r060_visible_page_support_sufficient": preview["support_audit"]["visible_support_sufficient"],
            "prediction_text": pred["prediction_text"],
            "routing_behavior": diag,
        })
    pass_count = sum(1 for row in diagnostics if row["routing_behavior_passed"])
    return {
        "schema_version": "r061_page_routed_provider_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r061_page_routed_provider_complete" if gate["gate_passed"] else "r061_page_routed_provider_needs_prompt_fix",
        "scope": {
            "tiny_provider_diagnostic": True,
            "target_records_only": TARGET_RECORD_IDS,
            "provider_calls": len(predictions),
            "uses_r060_page_routed_prompt_previews": True,
            "tests_page_only_routing_behavior_only": True,
            "does_not_prove_artifact_positive_lift": True,
            "does_not_compare_retrieval_conditions": True,
            "no_full_qa": True,
            "not_official_mmlongbench_result": True,
            "no_official_score_reported": True,
        },
        "inputs": {"r060_previews": args.r060_previews},
        "model": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "provider_note": args.provider_note.strip(),
        },
        "provider_prompt": {
            "mode": "r060_derived_compact_page_routing_prompt",
            "why_compact": "Full R060 prompt previews are long; R061 uses bounded public page contexts derived from the R060 previews for the provider diagnostic.",
            "max_page_contexts": args.max_page_contexts,
            "max_page_chars": args.max_page_chars,
            "chars_by_record": {str(row["record_id"]): row.get("provider_prompt_chars") for row in predictions},
            "sha256_by_record": {str(row["record_id"]): row.get("provider_prompt_sha256") for row in predictions},
        },
        "gate": gate,
        "page_only_routing_pass_count_not_score": pass_count,
        "page_only_routing_total": len(diagnostics),
        "diagnostic_counts_not_scores": dict(Counter("pass" if row["routing_behavior_passed"] else "needs_review" for row in diagnostics)),
        "per_record": per_record,
        "interpretation": {
            "bottom_line": "R061 only tests whether the provider follows R060 page-only routing on 2 prompts.",
            "artifact_lift_claim": "unsupported_by_this_run",
            "retrieval_improvement_claim": "unsupported_by_this_run",
            "official_score_claim": "unsupported_by_this_run",
        },
        "recommended_next": [
            "If R061 passes, keep the page/artifact routing prompt as a candidate provider-facing scaffold.",
            "Do not claim artifact-aware retrieval improves from R061; it has no retrieval-condition comparison and only two page-routed prompts.",
            "If R061 fails, repair the prompt response schema before any broader provider run.",
        ],
    }


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R061 Page-Routed Provider Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- Records 223 and 227 only.",
        "- Provider diagnostic on R060 page-routed prompts only.",
        "- Tests page-only routing behavior only.",
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
        "# R061 Page-Routed Provider Diagnostic",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- 2 records only: 223 and 227.",
        "- Provider diagnostic on R060 page-routed prompts only.",
        "- Can only show whether the provider follows page-only routing.",
        "- Cannot show artifact positive lift, retrieval improvement, full QA, or an official MMLongBench score.",
        "",
        "## Diagnostic Counts",
        f"- model: `{report['model']['model']}`",
        f"- predictions: {report['scope']['provider_calls']}",
        f"- provider prompt mode: `{report['provider_prompt']['mode']}`",
        f"- page-only routing pass count, not score: {report['page_only_routing_pass_count_not_score']} / {report['page_only_routing_total']}",
        f"- diagnostic counts, not scores: `{json.dumps(report['diagnostic_counts_not_scores'], sort_keys=True)}`",
        "",
        "## Per Record",
    ]
    for row in report["per_record"]:
        behavior = row["routing_behavior"]
        lines.append(
            f"- {row['record_id']}: guard=`{row['r060_guard_decision']}`, "
            f"passed={behavior['routing_behavior_passed']}, failures=`{behavior['failure_modes']}`"
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


def sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
