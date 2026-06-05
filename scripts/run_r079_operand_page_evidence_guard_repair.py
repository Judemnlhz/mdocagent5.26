#!/usr/bin/env python3
"""R079 no-provider operand guard/page-evidence repair gate.

R079 verifies the bounded repair after R077: when selected artifacts do not
cover all computation operands, but retrieved page text visibly covers the
required operands, the evidence layer routes back to page evidence instead of
forcing ``operand_completeness_guard`` refusal.
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
import run_r074_mmlb_evidence_prompt_integration_gate as r074
from mdocnexus.integration.guarded_prompt import build_question_profile, forbidden_public_fields

DEFAULT_OUTPUT_ROOT = "outputs/heldout/r079_operand_page_evidence_guard_repair"
TARGET_RECORD_IDS = [1035]
EXACT_CODE_STRICT_GUARDS = {"exact_code_absence_guard", "exact_code_key_value_selection"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", default=r074.DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--records", default=r074.DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=r074.DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=r074.DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="mmlb-MDocAgent-r079-operand-page-evidence-repair")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-page-chars", type=int, default=1600)
    parser.add_argument("--max-artifact-chars", type=int, default=280)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--capsule-units", type=int, default=4)
    parser.add_argument("--target-record-ids", default=",".join(str(item) for item in TARGET_RECORD_IDS))
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "repair": "operand_page_evidence_guard_repair",
            "target_record_ids": parse_record_ids(args.target_record_ids),
            "no_provider_calls": True,
            "no_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    audit = r074.build_audit(args)
    repair_records = build_repair_records(audit["audit_records"])
    summary = summarize(args, audit["audit_records"], repair_records)
    public_payload = {"summary": summary, "repair_records": repair_records, "retrieval_records": audit["retrieval_records"]}
    summary["forbidden_gold_fields_present"] = forbidden_public_fields(public_payload)
    gate = build_gate(args, summary)
    report = build_report(summary, gate)

    write_r074_compatible_outputs(output_root, audit)
    r053.write_json(output_root / "r079_operand_page_evidence_guard_summary.json", summary)
    r053.write_jsonl(output_root / "r079_operand_page_evidence_guard_records.jsonl", repair_records)
    r053.write_json(output_root / "r079_operand_page_evidence_guard_gate.json", gate)
    write_gate_markdown(output_root / "r079_operand_page_evidence_guard_gate.md", gate)
    r053.write_json(output_root / "r079_operand_page_evidence_guard_report.json", report)
    write_report_markdown(output_root / "r079_operand_page_evidence_guard_report.md", report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "records_scanned": summary["records_scanned"],
        "computation_operand_records": summary["computation_operand_records"],
        "operand_page_evidence_route_records": summary["operand_page_evidence_route_records"],
        "operand_completeness_guard_records": summary["operand_completeness_guard_records"],
        "target_record_checks": summary["target_record_checks"],
        "report_md": str(output_root / "r079_operand_page_evidence_guard_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
        "not_official_score": True,
    }, ensure_ascii=False, indent=2))


def parse_record_ids(raw: str) -> list[int]:
    ids = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return sorted(set(ids))


def build_repair_records(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        profile = build_question_profile(str(row.get("question") or ""))
        guard = str(row.get("guard_decision") or "")
        include = bool(profile.get("is_computation_question") and profile.get("required_operands"))
        include = include or guard in {"operand_page_evidence_route", "operand_completeness_guard"}
        if not include:
            continue
        missing = list(row.get("missing_requirements") or [])
        output.append({
            "schema_version": "r079_operand_page_evidence_guard_record_v1",
            "record_id": int(row.get("record_id") or 0),
            "doc_id": row.get("doc_id"),
            "question": row.get("question"),
            "required_operands": list(profile.get("required_operands") or []),
            "guard_decision": guard,
            "answer_policy": row.get("answer_policy"),
            "selected_artifact_count": int(row.get("selected_artifact_count") or 0),
            "missing_requirements": missing,
            "operand_missing_requirements": [item for item in missing if str(item).startswith("operand:")],
            "prompt_mode": row.get("prompt_mode"),
            "comparison_bucket": row.get("comparison_bucket"),
            "no_provider_calls": True,
            "not_prediction_or_eval": True,
        })
    return output


def summarize(args: argparse.Namespace, rows: list[Mapping[str, Any]], repair_records: list[Mapping[str, Any]]) -> dict[str, Any]:
    target_ids = parse_record_ids(args.target_record_ids)
    row_by_id = {int(row["record_id"]): row for row in rows}
    repair_by_id = {int(row["record_id"]): row for row in repair_records}
    guard_counts = Counter(str(row.get("guard_decision") or "") for row in rows)
    prompt_mode_counts = Counter(str(row.get("prompt_mode") or "") for row in rows)
    exact_code_rows = []
    for row in rows:
        profile = build_question_profile(str(row.get("question") or ""))
        if profile.get("requires_exact_code_selection"):
            exact_code_rows.append(row)
    target_checks = {}
    for record_id in target_ids:
        row = repair_by_id.get(record_id) or row_by_id.get(record_id)
        operand_missing = row.get("operand_missing_requirements", []) if row else []
        target_checks[str(record_id)] = {
            "present": row is not None,
            "guard_decision": row.get("guard_decision") if row else None,
            "prompt_mode": row.get("prompt_mode") if row else None,
            "selected_artifact_count": row.get("selected_artifact_count") if row else None,
            "operand_missing_requirements": operand_missing,
            "route_ok": bool(row and row.get("guard_decision") == "operand_page_evidence_route" and not operand_missing),
        }
    return {
        "schema_version": "r079_operand_page_evidence_guard_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records_scanned": len(rows),
        "repair_records": len(repair_records),
        "target_record_ids": target_ids,
        "target_record_checks": target_checks,
        "computation_operand_records": len(repair_records),
        "operand_page_evidence_route_records": sum(1 for row in repair_records if row.get("guard_decision") == "operand_page_evidence_route"),
        "operand_completeness_guard_records": sum(1 for row in repair_records if row.get("guard_decision") == "operand_completeness_guard"),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "prompt_mode_counts": dict(sorted(prompt_mode_counts.items())),
        "exact_code_records": len(exact_code_rows),
        "exact_code_strict_guard_records": sum(1 for row in exact_code_rows if str(row.get("guard_decision") or "") in EXACT_CODE_STRICT_GUARDS),
        "boundary": {
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_use_answer_or_evidence_pages": True,
        },
    }


def build_gate(args: argparse.Namespace, summary: Mapping[str, Any]) -> dict[str, Any]:
    target_checks = summary.get("target_record_checks") or {}
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "not_official_score": True,
        "records_scanned_positive": summary.get("records_scanned", 0) > 0,
        "target_records_present": all(item.get("present") for item in target_checks.values()),
        "target_records_route_to_page_evidence": all(item.get("route_ok") for item in target_checks.values()),
        "operand_page_evidence_route_present": summary.get("operand_page_evidence_route_records", 0) > 0,
        "exact_code_guards_remain_strict": summary.get("exact_code_records", 0) == summary.get("exact_code_strict_guard_records", -1),
        "no_gold_fields_in_public_outputs": not summary.get("forbidden_gold_fields_present"),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r074.DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r079_operand_page_evidence_guard_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r079_operand_page_evidence_guard_repair_complete" if not hard_failures else "r079_operand_page_evidence_guard_repair_invalid",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(summary: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "r079_operand_page_evidence_guard_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": gate["decision"],
        "scope": summary["boundary"],
        "summary": summary,
        "gate": gate,
        "recommended_next": [
            "Run one bounded paired provider diagnostic with the R079 prompt root and force-include record 1035.",
            "If paired delta >= 0 and no new systematic hurt appears, stop guard repair and enter bounded MDocAgent QA.",
            "If the result is only small-positive or flat, frame the method contribution around token efficiency, evidence auditability, and guarded citation faithfulness with bounded/partial QA claims.",
        ],
    }


def write_r074_compatible_outputs(output_root: Path, audit: Mapping[str, Any]) -> None:
    r053.write_json(output_root / "r074_mmlb_evidence_layer_top4_retrieval.json", audit["retrieval_records"])
    r053.write_jsonl(output_root / "r074_mmlb_evidence_prompt_records.jsonl", audit["audit_records"])
    r053.write_json(output_root / "r074_mmlb_evidence_prompt_summary.json", audit["summary"])


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = ["# R079 Operand Page-Evidence Guard Gate", "", f"Decision: `{gate['decision']}`", f"Gate passed: {gate['gate_passed']}", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Public question/profile/retrieval/artifact audit only.", "- Not an official score.", "", "## Checks"]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# R079 Operand Page-Evidence Guard Repair",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Repairs only the conflict where incomplete artifact operands override complete visible page evidence.",
        "- Does not weaken exact-code strict guards.",
        "",
        "## Summary",
        f"- records scanned: {summary['records_scanned']}",
        f"- computation operand records: {summary['computation_operand_records']}",
        f"- operand page-evidence route records: {summary['operand_page_evidence_route_records']}",
        f"- operand completeness guard records: {summary['operand_completeness_guard_records']}",
        f"- exact-code strict guard records: {summary['exact_code_strict_guard_records']} / {summary['exact_code_records']}",
        f"- target record checks: {summary['target_record_checks']}",
        "",
        "## Recommended Next",
    ]
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
