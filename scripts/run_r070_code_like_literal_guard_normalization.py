#!/usr/bin/env python3
"""R070 no-provider code-like literal guard normalization audit.

R070 verifies the parser/selector repair that separates actionable exact codes
(AR01, CA03, CA19, AR03) from temporal/metric code-like literals (FY2015,
FY2018, Q3, AP50, F1). Temporal/metric literals must not trigger the
exact_code_absence_guard; actionable exact codes must still use strict exact-code
selection/absence behavior. This runner uses public questions, public retrieved
page text, and public artifacts only. It does not call providers, generate
predictions, run QA/eval, or report an official score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053
from mdocnexus.integration.guarded_prompt import (
    CODE_PATTERN,
    actionable_exact_codes,
    build_question_profile,
    forbidden_public_fields,
    score_guarded_artifact,
    select_guarded_artifacts,
    temporal_metric_code_like_literals,
)

DEFAULT_OUTPUT_ROOT = "outputs/heldout/r070_code_like_literal_guard_normalization"
ACTIONABLE_TARGETS = ["AR01", "CA03", "CA19", "AR03"]
TEMPORAL_METRIC_TARGETS = ["FY2015", "FY2018", "Q3", "AP50", "F1"]
MAX_EXAMPLES = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=r053.DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=r053.DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=r053.DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0, help="Optional debug cap; 0 means all records.")
    parser.add_argument("--max-page-chars", type=int, default=2200)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "records": args.records,
            "artifacts": args.artifacts,
            "repair": "code_like_literal_guard_normalization",
            "no_provider_calls": True,
            "no_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit)
    report = build_report(args, audit, gate)
    r053.write_json(output_root / "r070_code_like_literal_guard_summary.json", audit["summary"])
    r053.write_jsonl(output_root / "r070_code_like_literal_guard_records.jsonl", audit["records"])
    r053.write_json(output_root / "r070_code_like_literal_guard_gate.json", gate)
    write_gate_markdown(output_root / "r070_code_like_literal_guard_gate.md", gate)
    r053.write_json(output_root / "r070_code_like_literal_guard_report.json", report)
    write_report_markdown(output_root / "r070_code_like_literal_guard_report.md", report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "records_scanned": audit["summary"]["records_scanned"],
        "temporal_metric_records": audit["summary"]["temporal_metric_records"],
        "temporal_metric_exact_code_guard_count": audit["summary"]["temporal_metric_exact_code_guard_count"],
        "actionable_exact_code_records": audit["summary"]["actionable_exact_code_records"],
        "actionable_strict_guard_records": audit["summary"]["actionable_strict_guard_records"],
        "report_md": str(output_root / "r070_code_like_literal_guard_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
        "not_official_score": True,
    }, ensure_ascii=False, indent=2))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    records = r053.read_json(Path(args.records))
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    rows = []
    for record_id, record in enumerate(records):
        row = audit_record(record_id, record, artifacts_by_page, Path(args.extract_path), args)
        if row["code_like_literals"]:
            rows.append(row)
    summary = summarize(rows, len(records), args)
    public_payload = {"summary": summary, "records": rows}
    summary["forbidden_gold_fields_present"] = forbidden_public_fields(public_payload)
    return {
        "summary": summary,
        "records": rows,
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
        "not_full_qa": True,
        "not_official_score": True,
    }


def audit_record(record_id: int, record: Mapping[str, Any], artifacts_by_page: Mapping[tuple[str, int], list[dict[str, Any]]], extract_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or "")
    question = str(record.get("question") or "")
    profile = build_question_profile(question)
    pages = retrieval_pages(record, args.top_k)
    page_contexts = [r053.load_page_context(extract_path, doc_id, page, args.max_page_chars) for page in pages]
    current_artifacts = [artifact for page in pages for artifact in artifacts_by_page.get((doc_id, page), [])]
    selection = selector_summary(current_artifacts, question, profile, pages, page_contexts, args)
    code_like_literals = [str(item) for item in profile.get("code_like_literals") or CODE_PATTERN.findall(question)]
    actionable = actionable_exact_codes(code_like_literals)
    temporal_metric = temporal_metric_code_like_literals(code_like_literals)
    classification = "actionable_exact_code" if actionable else "temporal_metric_literal" if temporal_metric else "other_code_like"
    strict_guard_ok = True
    if actionable:
        strict_guard_ok = selection["guard_decision"] in {"exact_code_absence_guard", "exact_code_key_value_selection"}
    if temporal_metric and not actionable:
        strict_guard_ok = selection["guard_decision"] != "exact_code_absence_guard" and not profile.get("requires_exact_code_selection")
    return {
        "schema_version": "r070_code_like_literal_guard_record_v1",
        "record_id": record_id,
        "doc_id": doc_id,
        "question": question,
        "retrieval_pages": pages,
        "code_like_literals": sorted(set(code_like_literals)),
        "actionable_exact_codes": actionable,
        "temporal_metric_literals": temporal_metric,
        "profile_codes": list(profile.get("codes") or []),
        "requires_exact_code_selection": bool(profile.get("requires_exact_code_selection")),
        "classification": classification,
        "candidate_artifact_count": len(current_artifacts),
        "selector": selection,
        "guard_normalization_ok": strict_guard_ok,
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
    }


def retrieval_pages(record: Mapping[str, Any], top_k: int) -> list[int]:
    values = []
    for key in ["text-top-10-question", "image-top-10-question"]:
        raw = record.get(key)
        if isinstance(raw, list):
            values.extend(raw[:top_k])
    return r053.unique_ints(values)


def selector_summary(artifacts: list[Mapping[str, Any]], question: str, profile: Mapping[str, Any], pages: list[int], page_contexts: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    scored = []
    page_set = set(pages)
    for artifact in artifacts:
        try:
            page = int(artifact.get("page_index"))
        except (TypeError, ValueError):
            page = -1
        scored.append(score_guarded_artifact(artifact, question, profile, page, artifact_pages=list(page_set), original_pages=list(page_set), max_chars=args.max_artifact_chars))
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)
    selected = selection.get("selected_artifacts") or []
    return {
        "candidate_count": len(scored),
        "positive_candidate_count": selection.get("positive_candidate_count"),
        "guard_decision": selection.get("guard_decision"),
        "answer_policy": selection.get("answer_policy"),
        "selected_artifact_count": len(selected),
        "selected_artifact_ids": [row.get("artifact_id") for row in selected[: args.max_artifacts]],
        "selected_exact_code_matches": sorted({code for row in selected for code in row.get("exact_code_matches", [])}),
    }


def summarize(rows: list[dict[str, Any]], total_records_seen: int, args: argparse.Namespace) -> dict[str, Any]:
    classification_counts = Counter(row["classification"] for row in rows)
    guard_counts = Counter(row["selector"].get("guard_decision") for row in rows)
    temporal_rows = [row for row in rows if row["temporal_metric_literals"] and not row["actionable_exact_codes"]]
    actionable_rows = [row for row in rows if row["actionable_exact_codes"]]
    target_presence = {
        "actionable_targets": {code: any(code in row["actionable_exact_codes"] for row in rows) for code in ACTIONABLE_TARGETS},
        "temporal_metric_targets": {code: any(code in row["temporal_metric_literals"] for row in rows) for code in TEMPORAL_METRIC_TARGETS},
    }
    examples = {
        "temporal_metric": compact_examples(temporal_rows),
        "actionable_exact_code": compact_examples(actionable_rows),
        "normalization_failures": compact_examples([row for row in rows if not row["guard_normalization_ok"]]),
    }
    return {
        "schema_version": "r070_code_like_literal_guard_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records_scanned": total_records_seen,
        "code_like_records": len(rows),
        "top_k": args.top_k,
        "input_records": args.records,
        "input_artifacts": args.artifacts,
        "input_extract_path": args.extract_path,
        "classification_counts": dict(sorted(classification_counts.items())),
        "selector_guard_counts": dict(sorted((str(k), v) for k, v in guard_counts.items())),
        "temporal_metric_records": len(temporal_rows),
        "temporal_metric_exact_code_guard_count": sum(1 for row in temporal_rows if row["selector"].get("guard_decision") == "exact_code_absence_guard" or row["requires_exact_code_selection"]),
        "actionable_exact_code_records": len(actionable_rows),
        "actionable_strict_guard_records": sum(1 for row in actionable_rows if row["selector"].get("guard_decision") in {"exact_code_absence_guard", "exact_code_key_value_selection"}),
        "target_presence": target_presence,
        "examples": examples,
        "boundary": {
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_use_answer_or_evidence_pages": True,
        },
    }


def compact_examples(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows[:MAX_EXAMPLES]:
        output.append({
            "record_id": row.get("record_id"),
            "doc_id": row.get("doc_id"),
            "question": str(row.get("question") or "")[:180],
            "code_like_literals": row.get("code_like_literals"),
            "actionable_exact_codes": row.get("actionable_exact_codes"),
            "temporal_metric_literals": row.get("temporal_metric_literals"),
            "requires_exact_code_selection": row.get("requires_exact_code_selection"),
            "selector_guard": row.get("selector", {}).get("guard_decision"),
            "guard_normalization_ok": row.get("guard_normalization_ok"),
        })
    return output


def build_gate(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "not_official_score": True,
        "records_scanned_positive": summary.get("records_scanned", 0) > 0,
        "code_like_records_present": summary.get("code_like_records", 0) > 0,
        "temporal_metric_literals_do_not_trigger_exact_code_guard": summary.get("temporal_metric_exact_code_guard_count", 0) == 0,
        "actionable_exact_codes_keep_strict_guard": summary.get("actionable_exact_code_records", 0) == summary.get("actionable_strict_guard_records", -1),
        "actionable_targets_seen": all(summary.get("target_presence", {}).get("actionable_targets", {}).get(code) for code in ACTIONABLE_TARGETS),
        "temporal_metric_targets_seen": all(summary.get("target_presence", {}).get("temporal_metric_targets", {}).get(code) for code in TEMPORAL_METRIC_TARGETS),
        "no_gold_fields_in_public_outputs": not summary.get("forbidden_gold_fields_present"),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r053.DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r070_code_like_literal_guard_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r070_code_like_literal_guard_normalization_complete" if not hard_failures else "r070_code_like_literal_guard_normalization_invalid",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(args: argparse.Namespace, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    return {
        "schema_version": "r070_code_like_literal_guard_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": gate["decision"],
        "scope": summary["boundary"],
        "summary": summary,
        "gate": gate,
        "recommended_next": recommendations(summary),
    }


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    if summary.get("temporal_metric_exact_code_guard_count", 0) == 0:
        rows.append("Keep temporal/metric code-like literals out of exact-code absence guard; route them through normal numeric/table support checks.")
    if summary.get("actionable_exact_code_records", 0) == summary.get("actionable_strict_guard_records", -1):
        rows.append("Keep actionable exact codes on strict exact-code selection/absence behavior; do not infer from nearby code families.")
    rows.append("Next no-provider step should rebuild or replay bounded Stage 2 artifacts for positive actionable code/name cases before any provider QA.")
    return rows


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = [
        "# R070 Code-Like Literal Guard Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Public question/profile/selector normalization audit only.",
        "- Not an official score.",
        "",
        "## Checks",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# R070 Code-Like Literal Guard Normalization",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Uses public questions, public retrieved page text, and public artifacts only.",
        "- Does not use answers, evidence pages, official scoring, or artifact-lift claims.",
        "",
        "## Summary",
        f"- records scanned: {summary['records_scanned']}",
        f"- code-like records: {summary['code_like_records']}",
        f"- temporal/metric records: {summary['temporal_metric_records']}",
        f"- temporal/metric exact-code guard count: {summary['temporal_metric_exact_code_guard_count']}",
        f"- actionable exact-code records: {summary['actionable_exact_code_records']}",
        f"- actionable strict-guard records: {summary['actionable_strict_guard_records']}",
        "",
        "## Classification Counts",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("classification_counts", {}).items())
    lines.extend(["", "## Recommended Next"])
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
