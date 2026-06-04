#!/usr/bin/env python3
"""R059 no-provider selector dimension-support repair gate.

R059 repairs the R058 finding that token/key overlap was being treated as
answer-supporting artifact evidence. It does not call providers, run
prediction, run evaluation, run full QA, or report a score.

The gate checks two sides:
1. R058 positive-signal records 69/223/224/227 are now rejected by the
   artifact-dimension support guard instead of being selected on loose overlap;
2. synthetic positive controls with full question-dimension coverage are still
   retained, proving the repair is not an all-clear/all-refusal strategy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053

from mdocnexus.integration.guarded_prompt import (
    audit_selected_artifact_support,
    build_question_profile,
    forbidden_public_fields,
    render_guarded_prompt,
    score_guarded_artifact,
    select_guarded_artifacts,
    sha256,
)

DEFAULT_R040_ROOT = r053.DEFAULT_R040_ROOT
DEFAULT_R039_RECORD_IDS = r053.DEFAULT_R039_RECORD_IDS
DEFAULT_RECORDS = r053.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r053.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r059_selector_dimension_repair"
DEFAULT_TARGET_RECORD_IDS = "69,223,224,227"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--target-record-ids", default=DEFAULT_TARGET_RECORD_IDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=1600)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    record_ids = parse_record_ids(args.target_record_ids)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "target_record_ids": record_ids,
            "no_provider_calls": True,
            "no_prediction_or_eval": True,
            "no_full_qa": True,
            "audit_focus": "selector dimension-support repair, not score",
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    records = r053.read_json(Path(args.records))
    frozen_record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(frozen_record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))

    previews = build_repaired_previews(args, record_ids, records, offsets, run_records, artifacts_by_page)
    controls = build_positive_controls()
    gate = build_gate(args, record_ids, previews, controls)
    report = build_report(args, previews, controls, gate)

    r053.write_jsonl(output_root / "r059_selector_repair_previews.jsonl", previews)
    r053.write_jsonl(output_root / "r059_positive_control_previews.jsonl", controls)
    r053.write_jsonl(output_root / "r059_selector_repair_compact_index.jsonl", build_compact_index(previews, controls))
    r053.write_json(output_root / "r059_selector_repair_gate.json", gate)
    write_gate_markdown(output_root / "r059_selector_repair_gate.md", gate)
    r053.write_json(output_root / "r059_selector_repair_report.json", report)
    write_report_markdown(output_root / "r059_selector_repair_report.md", report)

    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_target_cases": len(previews),
        "dimension_guarded_records": gate["dimension_guarded_records"],
        "positive_control_retained": gate["positive_control_retained"],
        "report_md": str(output_root / "r059_selector_repair_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, ensure_ascii=False, indent=2))


def build_repaired_previews(
    args: argparse.Namespace,
    record_ids: list[int],
    records: list[dict[str, Any]],
    offsets: dict[int, int],
    run_records: dict[str, list[dict[str, Any]]],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for record_id in record_ids:
        if record_id not in offsets:
            raise ValueError(f"target record_id is not in R039 frozen subset: {record_id}")
        source = records[record_id]
        doc_id = str(source["doc_id"])
        question = str(source["question"])
        offset = offsets[record_id]
        original_record = run_records["top4_original_only"][offset]
        artifact_record = run_records["top4_artifact_only"][offset]
        original_pages = r053.combined_pages(original_record)
        artifact_pages = r053.combined_pages(artifact_record)
        candidate_pages = r053.unique_ints(artifact_pages + original_pages)
        profile = build_question_profile(question)
        page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        candidates = []
        for page in candidate_pages:
            for artifact in artifacts_by_page.get((doc_id, page), []):
                candidates.append(
                    score_guarded_artifact(
                        artifact,
                        question,
                        profile,
                        page,
                        artifact_pages=artifact_pages,
                        original_pages=original_pages,
                        max_chars=args.max_artifact_chars,
                    )
                )
        selection = select_guarded_artifacts(candidates, page_contexts, profile, max_artifacts=args.max_artifacts)
        support = audit_selected_artifact_support(selection.get("selected_artifacts") or [], page_contexts, profile)
        prompt = render_guarded_prompt(question, page_contexts, selection, profile, condition_label="R059 condition: selector_dimension_support_repair")
        public_payload = {
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "question_profile": profile,
            "retrieval_pages": {"artifact": artifact_pages, "original": original_pages, "candidate_union": candidate_pages},
            "selection": selection,
            "support_audit": support,
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
        }
        rows.append({
            "schema_version": "r059_selector_repair_preview_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "question_profile": profile,
            "retrieval_pages": {
                "top4_artifact_only_combined": artifact_pages,
                "top4_original_only_combined": original_pages,
                "candidate_union": candidate_pages,
            },
            "selection_policy": {
                "name": "guarded_selector_dimension_support_v1",
                "module": "mdocnexus.integration.guarded_prompt",
                "uses_gold_fields": False,
                "requires_question_dimension_coverage": True,
                "guards_positive_signal_only_artifacts": True,
                "not_provider_run": True,
            },
            "candidate_artifact_count": len(candidates),
            "positive_candidate_count": selection["positive_candidate_count"],
            "selected_artifact_count": len(selection["selected_artifacts"]),
            "selected_artifacts": selection["selected_artifacts"],
            "guard_decision": selection["guard_decision"],
            "guard_reasons": selection["guard_reasons"],
            "answer_policy": selection["answer_policy"],
            "support_audit": support,
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
            "prompt_preview_sha256": sha256(prompt),
            "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        })
    return rows


def build_positive_controls() -> list[dict[str, Any]]:
    controls = [
        {
            "control_id": "raptor_full_dimension_control",
            "question": "In figure 4, which nodes are retrieved by RAPTOR for both questions?",
            "artifact": {
                "artifact_id": "fig4_nodes_control",
                "artifact_type": "caption",
                "content": "Figure 4 shows the RAPTOR retrieved nodes for both questions.",
                "normalized_content": {
                    "metric_name": "Figure 4",
                    "value_text": "RAPTOR retrieved nodes for both questions",
                },
                "source_anchored": True,
            },
        },
        {
            "control_id": "higher_income_2013_full_dimension_control",
            "question": (
                "Among the Higher-income seniors, what are the percentage of them go online, "
                "has smartphone phone, and own a tablet computer in the Pew Research Center's "
                "Internet Project July 18-September 30, 2013 tracking survey?"
            ),
            "artifact": {
                "artifact_id": "higher_income_2013_control",
                "artifact_type": "table",
                "content": "2013 Higher-income seniors go online: 80%; smartphone: 50%; tablet computer: 30%",
                "normalized_content": {
                    "row_label": "Higher-income seniors",
                    "column_label": "2013 go online smartphone tablet computer",
                    "value_text": "80%; 50%; 30%",
                },
                "source_anchored": True,
            },
        },
    ]
    rows = []
    for item in controls:
        profile = build_question_profile(item["question"])
        scored = score_guarded_artifact(item["artifact"], item["question"], profile, 1)
        selection = select_guarded_artifacts([scored], [], profile)
        support = audit_selected_artifact_support(selection.get("selected_artifacts") or [], [], profile)
        rows.append({
            "schema_version": "r059_positive_control_preview_v1",
            "control_id": item["control_id"],
            "question": item["question"],
            "guard_decision": selection["guard_decision"],
            "guard_reasons": selection["guard_reasons"],
            "selected_artifact_count": len(selection["selected_artifacts"]),
            "selected_artifact_ids": [artifact["artifact_id"] for artifact in selection["selected_artifacts"]],
            "support_audit": support,
            "forbidden_gold_fields_present": forbidden_public_fields({"question": item["question"], "selection": selection, "support": support}),
        })
    return rows


def build_gate(args: argparse.Namespace, record_ids: list[int], previews: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    dimension_guarded_records = [row["record_id"] for row in previews if row["guard_decision"] == "artifact_dimension_support_guard"]
    positive_control_retained = [row["control_id"] for row in controls if row["guard_decision"] == "token_key_value_selection" and row["support_audit"]["artifact_support_sufficient"]]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_records_match_r058_failures": sorted(record_ids) == sorted(row["record_id"] for row in previews),
        "all_target_records_have_positive_candidates": all(row["positive_candidate_count"] > 0 for row in previews),
        "all_target_records_dimension_guarded": sorted(dimension_guarded_records) == sorted(record_ids),
        "all_target_records_select_zero_artifacts": all(row["selected_artifact_count"] == 0 for row in previews),
        "positive_controls_retained": len(positive_control_retained) == len(controls),
        "positive_controls_have_supporting_artifacts": all(row["support_audit"]["artifact_support_sufficient"] for row in controls),
        "no_gold_fields_in_public_previews": all(not row["forbidden_gold_fields_present"] for row in previews + controls),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "not_provider_run": True,
        "not_artifact_lift_claim": True,
        "not_official_score": True,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r059_selector_repair_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r059_selector_repair_gate_pass" if not hard_failures else "r059_selector_repair_gate_fail",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_target_cases": len(previews),
        "num_positive_controls": len(controls),
        "dimension_guarded_records": dimension_guarded_records,
        "positive_control_retained": positive_control_retained,
        "guard_decision_by_record": {str(row["record_id"]): row["guard_decision"] for row in previews},
        "guard_reasons_by_record": {str(row["record_id"]): row["guard_reasons"] for row in previews},
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, previews: list[dict[str, Any]], controls: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    guard_counts = Counter(row["guard_decision"] for row in previews)
    return {
        "schema_version": "r059_selector_repair_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r059_selector_repair_complete" if gate["gate_passed"] else "r059_selector_repair_needs_fix",
        "scope": {
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_prove_artifact_positive_lift": True,
            "selector_repair_gate_only": True,
        },
        "inputs": {
            "records": args.records,
            "r040_root": args.r040_root,
            "r039_record_ids": args.r039_record_ids,
            "artifacts": args.artifacts,
            "target_record_ids": args.target_record_ids,
        },
        "num_target_cases": len(previews),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "dimension_guarded_records": gate["dimension_guarded_records"],
        "positive_control_retained": gate["positive_control_retained"],
        "per_record_summary": build_per_record_summary(previews),
        "positive_controls": controls,
        "gate": gate,
        "recommended_next": [
            "Do not run provider QA from R059.",
            "Review R059 prompts manually if needed, then decide whether the repaired selector should replace the prior token/key-value selector.",
            "Next no-provider gate should audit whether page evidence and artifact evidence are routed separately for cases where page evidence is sufficient but artifact evidence is not.",
            "Keep claims limited: R059 repairs selector safety; it does not prove artifact-aware retrieval lift.",
        ],
    }


def build_per_record_summary(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in previews:
        rows.append({
            "record_id": row["record_id"],
            "question": row["question"],
            "positive_candidate_count": row["positive_candidate_count"],
            "guard_decision": row["guard_decision"],
            "guard_reasons": row["guard_reasons"],
            "selected_artifact_count": row["selected_artifact_count"],
        })
    return rows


def build_compact_index(previews: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [{
        "schema_version": "r059_selector_repair_compact_index_v1",
        "record_id": row["record_id"],
        "guard_decision": row["guard_decision"],
        "positive_candidate_count": row["positive_candidate_count"],
        "selected_artifact_count": row["selected_artifact_count"],
        "guard_reasons": row["guard_reasons"],
        "prompt_preview_sha256": row["prompt_preview_sha256"],
    } for row in previews]
    rows.extend({
        "schema_version": "r059_selector_repair_compact_index_v1",
        "control_id": row["control_id"],
        "guard_decision": row["guard_decision"],
        "selected_artifact_count": row["selected_artifact_count"],
        "selected_artifact_ids": row["selected_artifact_ids"],
        "artifact_support_sufficient": row["support_audit"]["artifact_support_sufficient"],
    } for row in controls)
    return rows


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R059 Selector Dimension Repair Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Selector repair gate only.",
        "- Guards token/key overlap artifacts that do not cover question dimensions.",
        "- Not an official score and not an artifact-lift claim.",
        "",
        "## Checks",
    ]
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        for item in gate["hard_failures"]:
            lines.append(f"- {item}")
    lines.extend([
        "",
        "## Summary",
        f"- dimension guarded records: `{gate['dimension_guarded_records']}`",
        f"- positive controls retained: `{gate['positive_control_retained']}`",
    ])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R059 Selector Dimension-Support Repair",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Repairs R058's selector issue: positive signal is not answer-supporting artifact evidence.",
        "- Not an official score and not evidence of artifact positive lift.",
        "",
        "## Summary",
        f"- target cases: {report['num_target_cases']}",
        f"- guard decisions: `{json.dumps(report['guard_decision_counts'], sort_keys=True)}`",
        f"- dimension guarded records: `{report['dimension_guarded_records']}`",
        f"- positive controls retained: `{report['positive_control_retained']}`",
        "",
        "## Per-Record Repair",
    ]
    for row in report["per_record_summary"]:
        lines.extend([
            f"### Record {row['record_id']}",
            f"- positive candidates before guard: {row['positive_candidate_count']}",
            f"- guard decision: `{row['guard_decision']}`",
            f"- selected artifacts after guard: {row['selected_artifact_count']}",
            f"- guard reasons: `{row['guard_reasons']}`",
        ])
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


def parse_record_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
