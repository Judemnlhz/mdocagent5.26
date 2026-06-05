#!/usr/bin/env python3
"""R071 no-provider Evidence Skill Graph registry design gate.

R071 freezes a lightweight, dataset-agnostic evidence-layer interface before any
capsule, provider, or QA work. It audits the Evidence Skill Registry contract,
replays deterministic skill activation traces on public records/artifacts, and
checks that the design remains bounded: no large skill tree, no global graph, no
provider calls, no predictions, no evaluation, and no official score.
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
from mdocnexus.integration.evidence_skill_registry import (
    DOCUMENT_EDGE_TYPES,
    EVIDENCE_UNIT_TYPES,
    REGISTRY,
    activated_skills,
    build_skill_trace,
    registry_contract,
    validate_registry_contract,
)
from mdocnexus.integration.guarded_prompt import (
    build_question_profile,
    forbidden_public_fields,
    score_guarded_artifact,
    select_guarded_artifacts,
)

DEFAULT_OUTPUT_ROOT = "outputs/heldout/r071_evidence_skill_graph_registry_gate"
MAX_EXAMPLES = 20

CONTROL_QUESTIONS = {
    "exact_code_lookup": "According to this document, what's the geographic market name for EPS Code AR03?",
    "key_value_lookup": "What is the main policy name?",
    "table_numeric_lookup": "What is the value for total assets in 2018?",
    "numeric_computation": "What is the percentage difference between older age group with STEM degree and children with the same status?",
    "figure_caption_grounding": "Which figure shows both RAPTOR retrieved nodes and questions?",
    "text_span_grounding": "Describe the policy context.",
}


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
            "stage": "r071_evidence_skill_graph_registry_gate",
            "no_provider_calls": True,
            "no_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit)
    report = build_report(audit, gate)
    r053.write_json(output_root / "r071_evidence_skill_registry_summary.json", audit["summary"])
    r053.write_jsonl(output_root / "r071_evidence_skill_registry_records.jsonl", audit["records"])
    r053.write_json(output_root / "r071_evidence_skill_registry_gate.json", gate)
    write_gate_markdown(output_root / "r071_evidence_skill_registry_gate.md", gate)
    r053.write_json(output_root / "r071_evidence_skill_registry_report.json", report)
    write_report_markdown(output_root / "r071_evidence_skill_registry_report.md", report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "records_scanned": audit["summary"]["records_scanned"],
        "registry_skill_count": audit["summary"]["registry_skill_count"],
        "evidence_unit_type_count": audit["summary"]["evidence_unit_type_count"],
        "document_edge_type_count": audit["summary"]["document_edge_type_count"],
        "activated_skill_names": audit["summary"]["activated_skill_names"],
        "report_md": str(output_root / "r071_evidence_skill_registry_report.md"),
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
        rows.append(audit_record(record_id, record, artifacts_by_page, Path(args.extract_path), args))
    contract = registry_contract()
    controls = build_control_traces()
    summary = summarize(rows, len(records), contract, controls, args)
    public_payload = {"summary": summary, "records": rows, "registry_contract": contract, "control_traces": controls}
    summary["forbidden_gold_fields_present"] = forbidden_public_fields(public_payload)
    return {
        "summary": summary,
        "records": rows,
        "registry_contract": contract,
        "control_traces": controls,
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
    scored = score_artifacts(current_artifacts, question, profile, pages, args)
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)
    trace = build_skill_trace(profile, question, selection, scored)
    selected = selection.get("selected_artifacts") or []
    return {
        "schema_version": "r071_evidence_skill_registry_record_v1",
        "record_id": record_id,
        "doc_id": doc_id,
        "question": question,
        "retrieval_pages": pages,
        "activated_skill_names": trace["activated_skill_names"],
        "guard_decision": selection.get("guard_decision"),
        "selected_artifact_count": len(selected),
        "candidate_artifact_count": len(scored),
        "skill_trace": trace,
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


def score_artifacts(artifacts: list[Mapping[str, Any]], question: str, profile: Mapping[str, Any], pages: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    scored = []
    page_set = set(pages)
    for artifact in artifacts:
        try:
            page = int(artifact.get("page_index"))
        except (TypeError, ValueError):
            page = -1
        scored.append(score_guarded_artifact(artifact, question, profile, page, artifact_pages=list(page_set), original_pages=list(page_set), max_chars=args.max_artifact_chars))
    return scored


def build_control_traces() -> list[dict[str, Any]]:
    rows = []
    for expected_skill, question in CONTROL_QUESTIONS.items():
        profile = build_question_profile(question)
        selection = select_guarded_artifacts([], [], profile)
        trace = build_skill_trace(profile, question, selection, [])
        rows.append({
            "expected_skill": expected_skill,
            "question": question,
            "activated_skill_names": trace["activated_skill_names"],
            "passed": expected_skill in trace["activated_skill_names"],
            "guard_decision": trace["guard_decision"],
        })
    return rows


def summarize(rows: list[dict[str, Any]], total_records_seen: int, contract: Mapping[str, Any], controls: list[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    skill_counts = Counter(skill for row in rows for skill in row.get("activated_skill_names") or [])
    guard_counts = Counter(str(row.get("guard_decision") or "") for row in rows)
    contract_failures = validate_registry_contract(contract)
    all_registry_skills = sorted(skill.name for skill in REGISTRY)
    control_passed = all(bool(row.get("passed")) for row in controls)
    examples = compact_examples(rows)
    return {
        "schema_version": "r071_evidence_skill_registry_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records_scanned": total_records_seen,
        "top_k": args.top_k,
        "input_records": args.records,
        "input_artifacts": args.artifacts,
        "input_extract_path": args.extract_path,
        "registry_skill_count": len(all_registry_skills),
        "registry_skill_names": all_registry_skills,
        "evidence_unit_type_count": len(EVIDENCE_UNIT_TYPES),
        "evidence_unit_types": list(EVIDENCE_UNIT_TYPES),
        "document_edge_type_count": len(DOCUMENT_EDGE_TYPES),
        "document_edge_types": list(DOCUMENT_EDGE_TYPES),
        "contract_failures": contract_failures,
        "control_activation_passed": control_passed,
        "control_traces": controls,
        "activated_skill_names": sorted(skill_counts),
        "activated_skill_counts": dict(sorted(skill_counts.items())),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "examples": examples,
        "boundary": {
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_use_answer_or_evidence_pages": True,
            "not_large_skill_tree": True,
            "not_global_knowledge_graph": True,
        },
    }


def compact_examples(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows[:MAX_EXAMPLES]:
        output.append({
            "record_id": row.get("record_id"),
            "doc_id": row.get("doc_id"),
            "question": str(row.get("question") or "")[:180],
            "activated_skill_names": row.get("activated_skill_names"),
            "guard_decision": row.get("guard_decision"),
            "candidate_artifact_count": row.get("candidate_artifact_count"),
            "selected_artifact_count": row.get("selected_artifact_count"),
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
        "registry_contract_valid": not summary.get("contract_failures"),
        "registry_skill_count_bounded": 0 < summary.get("registry_skill_count", 0) <= 6,
        "evidence_unit_type_count_bounded": 0 < summary.get("evidence_unit_type_count", 0) <= 6,
        "document_edge_type_count_bounded": 0 < summary.get("document_edge_type_count", 0) <= 8,
        "control_activation_passed": summary.get("control_activation_passed") is True,
        "dataset_records_activate_registry_skills": bool(summary.get("activated_skill_names")),
        "no_gold_fields_in_public_outputs": not summary.get("forbidden_gold_fields_present"),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r053.DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r071_evidence_skill_registry_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r071_evidence_skill_registry_gate_complete" if not hard_failures else "r071_evidence_skill_registry_gate_invalid",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    return {
        "schema_version": "r071_evidence_skill_registry_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": gate["decision"],
        "scope": summary["boundary"],
        "summary": summary,
        "gate": gate,
        "recommended_next": recommendations(summary),
    }


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    if not summary.get("contract_failures"):
        rows.append("Keep the registry bounded and dataset-agnostic; do not add dataset-named skills or a large skill tree.")
    if summary.get("control_activation_passed"):
        rows.append("Proceed to R072 token-budgeted capsule renderer using this registry as the only skill dispatch interface.")
    rows.append("Do not run provider QA until R072/R073 no-provider capsule and cross-dataset audits pass.")
    return rows


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = [
        "# R071 Evidence Skill Registry Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Lightweight registry/schema/trace gate only.",
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
        "# R071 Evidence Skill Graph Registry Gate",
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
        f"- registry skills: {summary['registry_skill_count']} `{summary['registry_skill_names']}`",
        f"- evidence unit types: {summary['evidence_unit_type_count']} `{summary['evidence_unit_types']}`",
        f"- document edge types: {summary['document_edge_type_count']} `{summary['document_edge_types']}`",
        f"- contract failures: `{summary['contract_failures']}`",
        f"- control activation passed: {summary['control_activation_passed']}",
        "",
        "## Activated Skill Counts",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("activated_skill_counts", {}).items())
    lines.extend(["", "## Guard Decision Counts"])
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("guard_decision_counts", {}).items())
    lines.extend(["", "## Recommended Next"])
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
