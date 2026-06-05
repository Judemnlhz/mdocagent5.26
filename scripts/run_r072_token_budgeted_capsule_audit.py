#!/usr/bin/env python3
"""R072 no-provider token-budgeted evidence capsule audit.

R072 validates whether the R071 Evidence Skill Registry can render compact,
auditable evidence capsules before any provider or QA run. It compares token
estimates for raw retrieved page context, flat artifact context, capsule without
trace, and capsule with guard trace. It does not call providers, generate
predictions, evaluate QA, or report an official score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053
import run_r071_evidence_skill_graph_registry_gate as r071
from mdocnexus.integration.evidence_skill_registry import (
    estimate_tokens,
    flat_artifact_context,
    raw_page_context,
    render_evidence_capsule,
)
from mdocnexus.integration.guarded_prompt import build_question_profile, forbidden_public_fields, select_guarded_artifacts

DEFAULT_OUTPUT_ROOT = "outputs/heldout/r072_token_budgeted_capsule_audit"
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
    parser.add_argument("--capsule-units", type=int, default=4)
    parser.add_argument("--flat-artifact-units", type=int, default=8)
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
            "stage": "r072_token_budgeted_capsule_audit",
            "no_provider_calls": True,
            "no_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit)
    report = build_report(audit, gate)
    r053.write_json(output_root / "r072_token_budgeted_capsule_summary.json", audit["summary"])
    r053.write_jsonl(output_root / "r072_token_budgeted_capsule_records.jsonl", audit["records"])
    r053.write_json(output_root / "r072_token_budgeted_capsule_gate.json", gate)
    write_gate_markdown(output_root / "r072_token_budgeted_capsule_gate.md", gate)
    r053.write_json(output_root / "r072_token_budgeted_capsule_report.json", report)
    write_report_markdown(output_root / "r072_token_budgeted_capsule_report.md", report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "records_scanned": audit["summary"]["records_scanned"],
        "mean_raw_tokens": audit["summary"]["token_stats"]["raw_page"]["mean"],
        "mean_capsule_tokens": audit["summary"]["token_stats"]["capsule_with_guard"]["mean"],
        "mean_capsule_compression_ratio": audit["summary"]["compression_stats"]["capsule_with_guard_vs_raw"]["mean"],
        "report_md": str(output_root / "r072_token_budgeted_capsule_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
        "not_official_score": True,
    }, ensure_ascii=False, indent=2))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    records = r053.read_json(Path(args.records))
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    rows = [audit_record(i, record, artifacts_by_page, Path(args.extract_path), args) for i, record in enumerate(records)]
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
    pages = r071.retrieval_pages(record, args.top_k)
    page_contexts = [r053.load_page_context(extract_path, doc_id, page, args.max_page_chars) for page in pages]
    current_artifacts = [artifact for page in pages for artifact in artifacts_by_page.get((doc_id, page), [])]
    scored = r071.score_artifacts(current_artifacts, question, profile, pages, args)
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)
    raw_text = raw_page_context(page_contexts, max_chars_per_page=args.max_page_chars)
    flat_text = flat_artifact_context(scored, max_units=args.flat_artifact_units, max_chars=args.max_artifact_chars)
    capsule_plain = render_evidence_capsule(question, profile, selection, scored, max_units=args.capsule_units, include_guard_trace=False, max_chars=args.max_artifact_chars)
    capsule_guard = render_evidence_capsule(question, profile, selection, scored, max_units=args.capsule_units, include_guard_trace=True, max_chars=args.max_artifact_chars)
    raw_tokens = estimate_tokens(raw_text)
    flat_tokens = estimate_tokens(flat_text)
    capsule_plain_tokens = int(capsule_plain["token_estimate"])
    capsule_guard_tokens = int(capsule_guard["token_estimate"])
    return {
        "schema_version": "r072_token_budgeted_capsule_record_v1",
        "record_id": record_id,
        "doc_id": doc_id,
        "question": question,
        "retrieval_pages": pages,
        "candidate_artifact_count": len(scored),
        "selected_artifact_count": len(selection.get("selected_artifacts") or []),
        "activated_skill_names": capsule_guard["activated_skill_names"],
        "guard_decision": selection.get("guard_decision"),
        "missing_requirements": capsule_guard["missing_requirements"],
        "token_counts": {
            "raw_page": raw_tokens,
            "flat_artifact": flat_tokens,
            "capsule_plain": capsule_plain_tokens,
            "capsule_with_guard": capsule_guard_tokens,
        },
        "compression_ratios": {
            "flat_artifact_vs_raw": ratio(flat_tokens, raw_tokens),
            "capsule_plain_vs_raw": ratio(capsule_plain_tokens, raw_tokens),
            "capsule_with_guard_vs_raw": ratio(capsule_guard_tokens, raw_tokens),
        },
        "capsule_unit_count": capsule_guard["unit_count"],
        "capsule_unit_source": capsule_guard["unit_source"],
        "capsule_selected_artifact_ids": capsule_guard["selected_artifact_ids"],
        "capsule_preview": capsule_guard["text"][:720],
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
    }


def summarize(rows: list[dict[str, Any]], total_records_seen: int, args: argparse.Namespace) -> dict[str, Any]:
    token_keys = ["raw_page", "flat_artifact", "capsule_plain", "capsule_with_guard"]
    ratio_keys = ["flat_artifact_vs_raw", "capsule_plain_vs_raw", "capsule_with_guard_vs_raw"]
    token_stats = {key: number_stats([row["token_counts"][key] for row in rows]) for key in token_keys}
    compression_stats = {key: number_stats([row["compression_ratios"][key] for row in rows]) for key in ratio_keys}
    skill_counts = Counter(skill for row in rows for skill in row.get("activated_skill_names") or [])
    guard_counts = Counter(str(row.get("guard_decision") or "") for row in rows)
    better_than_raw = sum(1 for row in rows if row["token_counts"]["capsule_with_guard"] < row["token_counts"]["raw_page"])
    return {
        "schema_version": "r072_token_budgeted_capsule_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records_scanned": total_records_seen,
        "top_k": args.top_k,
        "capsule_units": args.capsule_units,
        "flat_artifact_units": args.flat_artifact_units,
        "input_records": args.records,
        "input_artifacts": args.artifacts,
        "input_extract_path": args.extract_path,
        "token_stats": token_stats,
        "compression_stats": compression_stats,
        "capsule_with_guard_lower_than_raw_count": better_than_raw,
        "capsule_with_guard_lower_than_raw_rate": round(better_than_raw / len(rows), 6) if rows else 0.0,
        "activated_skill_counts": dict(sorted(skill_counts.items())),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "examples": compact_examples(rows),
        "boundary": {
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_use_answer_or_evidence_pages": True,
        },
    }


def number_stats(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    nums = [float(value) for value in values]
    return {
        "mean": round(statistics.fmean(nums), 6),
        "median": round(statistics.median(nums), 6),
        "min": round(min(nums), 6),
        "max": round(max(nums), 6),
    }


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0 if numerator > 0 else 0.0
    return round(float(numerator) / float(denominator), 6)


def compact_examples(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: row.get("compression_ratios", {}).get("capsule_with_guard_vs_raw", 1.0))[:MAX_EXAMPLES]
    return [{
        "record_id": row.get("record_id"),
        "doc_id": row.get("doc_id"),
        "question": str(row.get("question") or "")[:180],
        "activated_skill_names": row.get("activated_skill_names"),
        "guard_decision": row.get("guard_decision"),
        "token_counts": row.get("token_counts"),
        "compression_ratios": row.get("compression_ratios"),
        "capsule_preview": row.get("capsule_preview"),
    } for row in selected]


def build_gate(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    token_stats = summary["token_stats"]
    compression = summary["compression_stats"]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "not_official_score": True,
        "records_scanned_positive": summary.get("records_scanned", 0) > 0,
        "capsule_with_guard_mean_lower_than_raw": token_stats["capsule_with_guard"]["mean"] < token_stats["raw_page"]["mean"],
        "capsule_plain_mean_not_more_than_guarded": token_stats["capsule_plain"]["mean"] <= token_stats["capsule_with_guard"]["mean"],
        "capsule_with_guard_mean_ratio_below_one": compression["capsule_with_guard_vs_raw"]["mean"] < 1.0,
        "capsule_lower_than_raw_for_majority": summary.get("capsule_with_guard_lower_than_raw_rate", 0.0) >= 0.5,
        "no_gold_fields_in_public_outputs": not summary.get("forbidden_gold_fields_present"),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r053.DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r072_token_budgeted_capsule_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r072_token_budgeted_capsule_audit_complete" if not hard_failures else "r072_token_budgeted_capsule_audit_invalid",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    return {
        "schema_version": "r072_token_budgeted_capsule_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": gate["decision"],
        "scope": summary["boundary"],
        "summary": summary,
        "gate": gate,
        "recommended_next": recommendations(summary),
    }


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    ratio_mean = summary.get("compression_stats", {}).get("capsule_with_guard_vs_raw", {}).get("mean", 1.0)
    rows.append(f"Use the R071 registry renderer for R073 cross-dataset reuse; current guarded capsule mean/raw token ratio is {ratio_mean}.")
    rows.append("Keep capsule rendering deterministic and bounded; do not add a second capsule abstraction.")
    rows.append("Do not run provider QA until R073 confirms cross-dataset schema and token behavior.")
    return rows


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = ["# R072 Token-Budgeted Capsule Gate", "", f"Decision: `{gate['decision']}`", f"Gate passed: {gate['gate_passed']}", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Deterministic capsule/token audit only.", "- Not an official score.", "", "## Checks"]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    token = summary["token_stats"]
    compression = summary["compression_stats"]
    lines = ["# R072 Token-Budgeted Evidence Capsule Audit", "", f"Decision: `{report['decision']}`", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Uses public questions, public retrieved page text, and public artifacts only.", "- Does not use answers, evidence pages, official scoring, or artifact-lift claims.", "", "## Summary", f"- records scanned: {summary['records_scanned']}", f"- mean raw page tokens: {token['raw_page']['mean']}", f"- mean flat artifact tokens: {token['flat_artifact']['mean']}", f"- mean capsule tokens without trace: {token['capsule_plain']['mean']}", f"- mean capsule tokens with guard trace: {token['capsule_with_guard']['mean']}", f"- mean guarded capsule/raw ratio: {compression['capsule_with_guard_vs_raw']['mean']}", f"- guarded capsule lower than raw rate: {summary['capsule_with_guard_lower_than_raw_rate']}", "", "## Activated Skill Counts"]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("activated_skill_counts", {}).items())
    lines.extend(["", "## Recommended Next"])
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
