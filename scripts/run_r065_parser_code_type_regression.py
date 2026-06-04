#!/usr/bin/env python3
"""R065 no-provider parser code-type normalization regression.

R065 verifies the R065 parser post-normalization repair: EPS/code-like questions
must route to table/code lookup with exact-code selection, while true document
metadata lookup remains metadata-routed. It reuses R063 parser outputs and does
not call providers, run prediction/evaluation, run full QA, or report a score.
"""

from __future__ import annotations

import argparse
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

from mdocnexus.integration.evidence_demand_parser import normalize_evidence_demand, merge_evidence_demand_profile
from mdocnexus.integration.guarded_prompt import (
    audit_selected_artifact_support,
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
DEFAULT_R063_COMPARISONS = "outputs/heldout/r063_llm_evidence_demand_parser/r063_selector_comparisons.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r065_parser_code_type_regression"
TARGET_RECORD_IDS = [508, 384]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r063-comparisons", default=DEFAULT_R063_COMPARISONS)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=1600)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "target_record_ids": TARGET_RECORD_IDS,
            "no_provider_calls": True,
            "no_prediction_or_eval": True,
            "no_full_qa": True,
            "regression_focus": "508 code/table normalization; 384 metadata control",
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    records = r053.read_json(Path(args.records))
    frozen_record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(frozen_record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    r063_rows = load_r063_rows(Path(args.r063_comparisons), TARGET_RECORD_IDS)

    rows = build_regressions(args, records, offsets, run_records, artifacts_by_page, r063_rows)
    gate = build_gate(args, rows)
    report = build_report(args, rows, gate)

    r053.write_jsonl(output_root / "r065_parser_code_type_regressions.jsonl", rows)
    r053.write_json(output_root / "r065_parser_code_type_gate.json", gate)
    write_gate_markdown(output_root / "r065_parser_code_type_gate.md", gate)
    r053.write_json(output_root / "r065_parser_code_type_report.json", report)
    write_report_markdown(output_root / "r065_parser_code_type_report.md", report)

    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_records": len(rows),
        "report_md": str(output_root / "r065_parser_code_type_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, ensure_ascii=False, indent=2))


def load_r063_rows(path: Path, record_ids: list[int]) -> dict[int, dict[str, Any]]:
    rows = {int(row["record_id"]): row for row in r053.read_jsonl(path) if int(row["record_id"]) in record_ids}
    missing = sorted(set(record_ids) - set(rows))
    if missing:
        raise ValueError(f"R063 comparison rows missing records: {missing}")
    return rows


def build_regressions(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    offsets: dict[int, int],
    run_records: dict[str, list[dict[str, Any]]],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    r063_rows: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record_id in TARGET_RECORD_IDS:
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
        page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        r063 = r063_rows[record_id]
        raw_demand = r063.get("parsed_evidence_demand") or {}
        normalized_demand = normalize_evidence_demand(raw_demand)
        profile = merge_evidence_demand_profile(question, raw_demand)
        candidates = []
        for page in candidate_pages:
            for artifact in artifacts_by_page.get((doc_id, page), []):
                candidates.append(score_guarded_artifact(
                    artifact,
                    question,
                    profile,
                    page,
                    artifact_pages=artifact_pages,
                    original_pages=original_pages,
                    max_chars=args.max_artifact_chars,
                ))
        selection = select_guarded_artifacts(candidates, page_contexts, profile, max_artifacts=args.max_artifacts)
        support = audit_selected_artifact_support(selection.get("selected_artifacts") or [], page_contexts, profile)
        prompt = render_guarded_prompt(question, page_contexts, selection, profile, condition_label="R065 condition: parser_code_type_regression")
        checks = record_checks(record_id, normalized_demand, profile, selection)
        public_payload = {
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "normalized_demand": normalized_demand,
            "profile": profile,
            "selection": selection,
            "support": support,
            "prompt_preview": prompt,
        }
        rows.append({
            "schema_version": "r065_parser_code_type_regression_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "r063_answer_type_before_repair": raw_demand.get("answer_type"),
            "normalized_answer_type_after_repair": normalized_demand.get("answer_type"),
            "normalized_demand": normalized_demand,
            "profile_flags": {
                "codes": profile.get("codes"),
                "requires_exact_code_selection": profile.get("requires_exact_code_selection"),
                "is_document_metadata_lookup": profile.get("is_document_metadata_lookup"),
                "is_numeric_or_table_question": profile.get("is_numeric_or_table_question"),
            },
            "selector_replay": {
                "guard_decision": selection.get("guard_decision"),
                "guard_reasons": selection.get("guard_reasons"),
                "answer_policy": selection.get("answer_policy"),
                "selected_artifact_count": len(selection.get("selected_artifacts") or []),
                "positive_candidate_count": selection.get("positive_candidate_count"),
                "candidate_artifact_count": len(candidates),
            },
            "support_audit": support,
            "prompt_preview_sha256": sha256(prompt),
            "checks": checks,
            "passed": all(checks.values()),
            "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
            "no_provider_calls": True,
            "not_prediction_or_eval": True,
            "not_full_qa": True,
            "not_official_score": True,
        })
    return rows


def record_checks(record_id: int, demand: Mapping[str, Any], profile: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, bool]:
    if record_id == 508:
        return {
            "code_pattern_forces_table_lookup": demand.get("answer_type") == "table_lookup",
            "code_pattern_forces_exact_code_selection": demand.get("requires_exact_code_selection") is True and profile.get("requires_exact_code_selection") is True,
            "code_pattern_clears_metadata_route": demand.get("is_document_metadata_lookup") is False and profile.get("is_document_metadata_lookup") is False,
            "selector_routes_to_exact_code_absence": selection.get("guard_decision") == "exact_code_absence_guard",
            "selector_selects_zero_artifacts_when_code_absent": len(selection.get("selected_artifacts") or []) == 0,
        }
    if record_id == 384:
        return {
            "date_metadata_stays_metadata_lookup": demand.get("answer_type") == "metadata_lookup",
            "date_metadata_does_not_force_exact_code": demand.get("requires_exact_code_selection") is False and profile.get("requires_exact_code_selection") is False,
            "date_metadata_keeps_metadata_route": demand.get("is_document_metadata_lookup") is True and profile.get("is_document_metadata_lookup") is True,
            "selector_keeps_metadata_refusal_guard": selection.get("guard_decision") == "document_metadata_refusal_guard",
        }
    raise ValueError(f"Unexpected R065 record: {record_id}")


def build_gate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {int(row["record_id"]): row for row in rows}
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_records_exactly_508_384": sorted(by_id) == sorted(TARGET_RECORD_IDS),
        "record_508_passed": by_id.get(508, {}).get("passed") is True,
        "record_384_control_passed": by_id.get(384, {}).get("passed") is True,
        "no_gold_fields_in_outputs": all(not row.get("forbidden_gold_fields_present") for row in rows),
        "does_not_claim_artifact_lift": True,
        "not_official_score": True,
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r065_parser_code_type_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r065_parser_code_type_gate_pass" if not hard_failures else "r065_parser_code_type_needs_fix",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "target_record_ids": TARGET_RECORD_IDS,
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, rows: list[dict[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "r065_parser_code_type_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r065_parser_code_type_regression_complete" if gate["gate_passed"] else "r065_parser_code_type_regression_needs_fix",
        "scope": {
            "target_records_only": TARGET_RECORD_IDS,
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_mmlongbench_result": True,
            "does_not_prove_artifact_positive_lift": True,
            "parser_post_normalization_regression_only": True,
        },
        "inputs": {
            "r063_comparisons": args.r063_comparisons,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
        },
        "gate": dict(gate),
        "per_record": [compact_record(row) for row in rows],
        "recommended_next": [
            "Keep the R065 parser code-type normalization in the default-off parser scaffold.",
            "Do not run full QA from R065; next inspect or repair artifact key/value extraction for the missing AR03 evidence.",
            "Rerun no-provider mismatch/coverage audit after artifact extraction changes.",
        ],
    }


def compact_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "before_answer_type": row["r063_answer_type_before_repair"],
        "after_answer_type": row["normalized_answer_type_after_repair"],
        "profile_flags": row["profile_flags"],
        "guard_decision": row["selector_replay"]["guard_decision"],
        "passed": row["passed"],
        "failed_checks": [key for key, value in row["checks"].items() if not value],
    }


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = [
        "# R065 Parser Code-Type Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Regression for parser post-normalization only.",
        "- Checks 508 code/table routing and 384 metadata control.",
        "",
        "## Checks",
    ]
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        for item in gate["hard_failures"]:
            lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# R065 Parser Code-Type Normalization Regression",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Does not prove artifact lift or official MMLongBench performance.",
        "",
        "## Per Record",
    ]
    for row in report["per_record"]:
        lines.append(
            f"- {row['record_id']}: before=`{row['before_answer_type']}`, after=`{row['after_answer_type']}`, "
            f"guard=`{row['guard_decision']}`, passed={row['passed']}, failed=`{row['failed_checks']}`"
        )
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
