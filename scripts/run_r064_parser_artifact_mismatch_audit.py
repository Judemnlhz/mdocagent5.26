#!/usr/bin/env python3
"""R064 no-provider parser/artifact mismatch audit.

R064 audits where R063 failed to recover supporting artifacts. It reads the
R063 parser outputs and selector comparisons, rebuilds the same public artifact
candidates, and attributes the break between parser evidence requirements,
artifact snippets, page context, and deterministic selector guards. It does not
call providers, run prediction, run evaluation, run full QA, or report a score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053

from mdocnexus.integration.evidence_demand_parser import merge_evidence_demand_profile
from mdocnexus.integration.guarded_prompt import (
    artifact_evidence_text,
    build_question_profile,
    dimension_checks,
    extract_numeric_values,
    forbidden_public_fields,
    normalize,
    score_guarded_artifact,
    select_guarded_artifacts,
)

DEFAULT_R040_ROOT = r053.DEFAULT_R040_ROOT
DEFAULT_R039_RECORD_IDS = r053.DEFAULT_R039_RECORD_IDS
DEFAULT_RECORDS = r053.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r053.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_R063_COMPARISONS = "outputs/heldout/r063_llm_evidence_demand_parser/r063_selector_comparisons.jsonl"
DEFAULT_R063_GATE = "outputs/heldout/r063_llm_evidence_demand_parser/r063_llm_evidence_demand_gate.json"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r064_parser_artifact_mismatch_audit"
DEFAULT_TARGET_RECORD_IDS = "384,508,569,69,223,224,227"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r063-comparisons", default=DEFAULT_R063_COMPARISONS)
    parser.add_argument("--r063-gate", default=DEFAULT_R063_GATE)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--target-record-ids", default=DEFAULT_TARGET_RECORD_IDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=1600)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--max-artifacts", type=int, default=8)
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
            "audit_focus": "parser requirement vs artifact/page content mismatch attribution",
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    records = r053.read_json(Path(args.records))
    frozen_record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(frozen_record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    r063_rows = load_r063_rows(Path(args.r063_comparisons), record_ids)
    r063_gate = r053.read_json(Path(args.r063_gate))

    audits = build_mismatch_audits(args, record_ids, records, offsets, run_records, artifacts_by_page, r063_rows)
    gate = build_gate(args, record_ids, r063_gate, audits)
    report = build_report(args, audits, gate)

    r053.write_jsonl(output_root / "r064_mismatch_audits.jsonl", audits)
    r053.write_jsonl(output_root / "r064_mismatch_compact_index.jsonl", build_compact_index(audits))
    r053.write_json(output_root / "r064_parser_artifact_mismatch_gate.json", gate)
    write_gate_markdown(output_root / "r064_parser_artifact_mismatch_gate.md", gate)
    r053.write_json(output_root / "r064_parser_artifact_mismatch_report.json", report)
    write_report_markdown(output_root / "r064_parser_artifact_mismatch_report.md", report)

    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_records": len(audits),
        "root_cause_counts": report["summary"]["root_cause_counts"],
        "report_md": str(output_root / "r064_parser_artifact_mismatch_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, ensure_ascii=False, indent=2))


def load_r063_rows(path: Path, record_ids: list[int]) -> dict[int, dict[str, Any]]:
    rows = {int(row["record_id"]): row for row in r053.read_jsonl(path) if int(row["record_id"]) in record_ids}
    missing = sorted(set(record_ids) - set(rows))
    if missing:
        raise ValueError(f"R063 comparison rows missing records: {missing}")
    return rows


def build_mismatch_audits(
    args: argparse.Namespace,
    record_ids: list[int],
    records: list[dict[str, Any]],
    offsets: dict[int, int],
    run_records: dict[str, list[dict[str, Any]]],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    r063_rows: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    audits = []
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
        page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        r063 = r063_rows[record_id]
        parsed = r063.get("parsed_evidence_demand") or {}
        rule_profile = build_question_profile(question)
        llm_profile = merge_evidence_demand_profile(question, parsed) if parsed else dict(rule_profile)
        candidates = rebuild_candidates(args, question, doc_id, candidate_pages, artifact_pages, original_pages, artifacts_by_page, llm_profile)
        selection = select_guarded_artifacts(candidates, page_contexts, llm_profile, max_artifacts=args.max_artifacts)
        matrix = build_coverage_matrix(parsed, llm_profile, candidates, page_contexts)
        root = attribute_root_cause(record_id, r063, parsed, llm_profile, selection, matrix)
        recommendations = recommendations_for(root, matrix, parsed, r063)
        payload = {
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "parsed_evidence_demand": parsed,
            "coverage_matrix": matrix,
            "selector_decision": selection,
            "root_cause": root,
        }
        audits.append({
            "schema_version": "r064_parser_artifact_mismatch_audit_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "answer_type": parsed.get("answer_type"),
            "retrieval_pages": {
                "top4_artifact_only_combined": artifact_pages,
                "top4_original_only_combined": original_pages,
                "candidate_union": candidate_pages,
            },
            "r063_summary": {
                "rule_guard": r063.get("rule_only", {}).get("guard_decision"),
                "llm_guard": r063.get("llm_evidence_demand", {}).get("guard_decision"),
                "llm_selected_artifact_count": r063.get("llm_evidence_demand", {}).get("selected_artifact_count"),
                "llm_artifact_support_sufficient": r063.get("llm_evidence_demand", {}).get("support_audit", {}).get("artifact_support_sufficient"),
                "interpretation": r063.get("comparison", {}).get("interpretation"),
            },
            "parsed_evidence_demand": parsed,
            "coverage_matrix": matrix,
            "selector_replay": {
                "guard_decision": selection.get("guard_decision"),
                "guard_reasons": selection.get("guard_reasons"),
                "answer_policy": selection.get("answer_policy"),
                "selected_artifact_count": len(selection.get("selected_artifacts") or []),
                "selected_artifact_ids": [row.get("artifact_id") for row in selection.get("selected_artifacts") or []],
                "positive_candidate_count": selection.get("positive_candidate_count"),
                "rejected_artifact_count": selection.get("rejected_artifact_count"),
            },
            "root_cause_attribution": root,
            "recommended_fix": recommendations,
            "forbidden_gold_fields_present": forbidden_public_fields(payload),
            "no_provider_calls": True,
            "not_prediction_or_eval": True,
            "not_full_qa": True,
            "not_official_score": True,
        })
    return audits


def rebuild_candidates(
    args: argparse.Namespace,
    question: str,
    doc_id: str,
    candidate_pages: list[int],
    artifact_pages: list[int],
    original_pages: list[int],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for page in candidate_pages:
        for artifact in artifacts_by_page.get((doc_id, page), []):
            rows.append(score_guarded_artifact(
                artifact,
                question,
                profile,
                page,
                artifact_pages=artifact_pages,
                original_pages=original_pages,
                max_chars=args.max_artifact_chars,
            ))
    return rows


def build_coverage_matrix(
    parsed: Mapping[str, Any],
    profile: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    page_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    requirements = profile.get("evidence_requirements") if isinstance(profile.get("evidence_requirements"), Mapping) else {}
    dimensions = list(requirements.get("dimensions") or [])
    artifact_texts = [(row, normalize(artifact_evidence_text(row))) for row in candidates]
    page_texts = [(ctx, normalize(str(ctx.get("text_preview") or ""))) for ctx in page_contexts]
    artifact_joined = normalize(" ".join(text for _, text in artifact_texts))
    page_joined = normalize(" ".join(text for _, text in page_texts))
    artifact_dimension_checks = dimension_checks(dimensions, artifact_joined)
    page_dimension_checks = dimension_checks(dimensions, page_joined)
    dimension_rows = []
    for dim in dimensions:
        aliases = [str(alias) for alias in dim.get("aliases") or []]
        artifact_hits = artifact_hits_for_aliases(artifact_texts, aliases)
        page_hits = page_hits_for_aliases(page_texts, aliases)
        dimension_rows.append({
            "dimension": dim.get("dimension"),
            "label": dim.get("label"),
            "aliases": aliases,
            "artifact_hit_count": len(artifact_hits),
            "artifact_hits": artifact_hits[:8],
            "page_hit_count": len(page_hits),
            "page_hits": page_hits[:8],
            "covered_by_artifact": bool(artifact_hits),
            "covered_by_page": bool(page_hits),
        })
    required_values = [str(item) for item in parsed.get("required_values_or_codes") or []]
    value_rows = []
    for value in required_values:
        artifact_hits = artifact_hits_for_aliases(artifact_texts, [value])
        page_hits = page_hits_for_aliases(page_texts, [value])
        value_rows.append({
            "value_or_code": value,
            "artifact_hit_count": len(artifact_hits),
            "artifact_hits": artifact_hits[:8],
            "page_hit_count": len(page_hits),
            "page_hits": page_hits[:8],
            "covered_by_artifact": bool(artifact_hits),
            "covered_by_page": bool(page_hits),
        })
    required_operands = [str(item) for item in profile.get("required_operands") or []]
    operand_rows = []
    for operand in required_operands:
        artifact_hits = [
            {
                "artifact_id": row.get("artifact_id"),
                "page_index": row.get("page_index"),
                "content_preview": row.get("content_preview"),
            }
            for row in candidates
            if operand in (row.get("operand_hits") or [])
        ]
        operand_rows.append({
            "operand": operand,
            "artifact_hit_count": len(artifact_hits),
            "artifact_hits": artifact_hits[:8],
            "covered_by_artifact": bool(artifact_hits),
        })
    return {
        "schema_version": "r064_coverage_matrix_v1",
        "candidate_artifact_count": len(candidates),
        "page_context_count": len(page_contexts),
        "dimension_count": len(dimensions),
        "artifact_dimension_covered_count": sum(1 for row in dimension_rows if row["covered_by_artifact"]),
        "page_dimension_covered_count": sum(1 for row in dimension_rows if row["covered_by_page"]),
        "missing_artifact_dimensions": [row["dimension"] for row in dimension_rows if not row["covered_by_artifact"]],
        "missing_page_dimensions": [row["dimension"] for row in dimension_rows if not row["covered_by_page"]],
        "dimension_rows": dimension_rows,
        "value_rows": value_rows,
        "required_operands": required_operands,
        "operand_rows": operand_rows,
        "artifact_numeric_values": extract_numeric_values(artifact_joined)[:30],
        "page_numeric_values": extract_numeric_values(page_joined)[:30],
        "artifact_dimension_checks": artifact_dimension_checks,
        "page_dimension_checks": page_dimension_checks,
    }


def artifact_hits_for_aliases(artifact_texts: list[tuple[dict[str, Any], str]], aliases: list[str]) -> list[dict[str, Any]]:
    hits = []
    for row, text in artifact_texts:
        matched = matched_aliases(text, aliases)
        if matched:
            hits.append({
                "artifact_id": row.get("artifact_id"),
                "artifact_type": row.get("artifact_type"),
                "page_index": row.get("page_index"),
                "matched_aliases": matched,
                "content_preview": row.get("content_preview"),
            })
    return hits


def page_hits_for_aliases(page_texts: list[tuple[dict[str, Any], str]], aliases: list[str]) -> list[dict[str, Any]]:
    hits = []
    for ctx, text in page_texts:
        matched = matched_aliases(text, aliases)
        if matched:
            hits.append({
                "page_index": ctx.get("page_index"),
                "page_id": ctx.get("page_id"),
                "matched_aliases": matched,
                "text_preview": str(ctx.get("text_preview") or "")[:220],
            })
    return hits


def matched_aliases(text: str, aliases: list[str]) -> list[str]:
    text_norm = normalize(text).replace("-", " ")
    matched = []
    for alias in aliases:
        alias_norm = normalize(alias).replace("-", " ")
        if alias_norm and alias_norm in text_norm:
            matched.append(alias)
    return matched


def attribute_root_cause(
    record_id: int,
    r063: Mapping[str, Any],
    parsed: Mapping[str, Any],
    profile: Mapping[str, Any],
    selection: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    missing_art = set(matrix.get("missing_artifact_dimensions") or [])
    missing_page = set(matrix.get("missing_page_dimensions") or [])
    dim_count = int(matrix.get("dimension_count") or 0)
    artifact_dim_covered = int(matrix.get("artifact_dimension_covered_count") or 0)
    page_dim_covered = int(matrix.get("page_dimension_covered_count") or 0)
    value_rows = list(matrix.get("value_rows") or [])
    operand_rows = list(matrix.get("operand_rows") or [])
    rule_guard = r063.get("rule_only", {}).get("guard_decision")
    llm_guard = r063.get("llm_evidence_demand", {}).get("guard_decision")
    categories = []
    evidence = []

    if not parsed or dim_count == 0:
        categories.append("parser_schema_too_sparse")
        evidence.append("parser emitted no usable evidence_dimensions")
    if rule_guard == "exact_code_absence_guard" and llm_guard == "document_metadata_refusal_guard":
        categories.append("parser_answer_type_misclassified")
        evidence.append("rule profile treated the question as exact-code/table lookup but parser routed it as metadata lookup")
    if profile.get("requires_exact_code_selection") and any(not row["covered_by_artifact"] for row in value_rows):
        categories.append("artifact_key_value_missing_required_code")
        evidence.append("required code/value is absent from artifact snippets")
    if profile.get("is_computation_question") and any(not row["covered_by_artifact"] for row in operand_rows):
        categories.append("artifact_operand_missing")
        missing_operands = [row["operand"] for row in operand_rows if not row["covered_by_artifact"]]
        evidence.append("missing artifact operands: " + ",".join(missing_operands))
    if dim_count and artifact_dim_covered == 0 and page_dim_covered == dim_count:
        categories.append("page_only_support_artifact_store_gap")
        evidence.append("all parser dimensions are visible in page text but none are covered by artifact snippets")
    elif dim_count and missing_art and not missing_page:
        categories.append("artifact_store_missing_required_dimensions")
        evidence.append("page covers parser dimensions, artifact snippets miss at least one dimension")
    elif dim_count and missing_page:
        categories.append("retrieval_context_or_parser_constraint_gap")
        evidence.append("some parser dimensions are missing from both selected page context and artifacts")
    if artifact_dim_covered > 0 and int(selection.get("selected_artifacts") and len(selection.get("selected_artifacts")) or 0) == 0:
        categories.append("selector_support_threshold_or_alias_gap")
        evidence.append("some artifact snippets match parser aliases, but selector still rejects support")
    if not categories:
        categories.append("manual_review_required")
        evidence.append("coverage matrix does not match a deterministic attribution rule")

    primary = choose_primary_category(categories)
    return {
        "schema_version": "r064_root_cause_attribution_v1",
        "primary_root_cause": primary,
        "all_categories": sorted(set(categories)),
        "evidence": evidence,
        "record_id": record_id,
        "claim_ceiling": "diagnostic_attribution_only_no_artifact_lift_claim",
    }


def choose_primary_category(categories: list[str]) -> str:
    priority = [
        "parser_answer_type_misclassified",
        "artifact_key_value_missing_required_code",
        "artifact_operand_missing",
        "page_only_support_artifact_store_gap",
        "artifact_store_missing_required_dimensions",
        "selector_support_threshold_or_alias_gap",
        "retrieval_context_or_parser_constraint_gap",
        "parser_schema_too_sparse",
        "manual_review_required",
    ]
    for item in priority:
        if item in categories:
            return item
    return categories[0]


def recommendations_for(root: Mapping[str, Any], matrix: Mapping[str, Any], parsed: Mapping[str, Any], r063: Mapping[str, Any]) -> list[str]:
    primary = root.get("primary_root_cause")
    if primary == "parser_answer_type_misclassified":
        return [
            "Tighten parser schema/examples so EPS/code questions are table_lookup with requires_exact_code_selection, not metadata_lookup.",
            "Add a parser post-normalization rule: literal code patterns force table/code lookup unless the question asks document revision/producer metadata.",
        ]
    if primary == "artifact_key_value_missing_required_code":
        return [
            "Repair artifact extraction/indexing for exact code-key/value pairs.",
            "Prioritize artifacts whose normalized row/column/code fields contain the required literal code.",
        ]
    if primary == "artifact_operand_missing":
        return [
            "Repair artifact extraction for computation operands before any provider QA run.",
            "Keep operand-completeness guard active; do not compute from partial artifact snippets.",
        ]
    if primary == "page_only_support_artifact_store_gap":
        return [
            "Treat this as artifact-store coverage gap, not parser failure.",
            "Improve page-to-artifact extraction for dimensions already visible in page text, then rerun no-provider coverage audit.",
        ]
    if primary == "artifact_store_missing_required_dimensions":
        return [
            "Improve artifact content/normalization so required dimensions visible in pages are preserved in snippets.",
            "Do not relax selector support thresholds until artifact snippets carry citable evidence.",
        ]
    if primary == "selector_support_threshold_or_alias_gap":
        return [
            "Inspect matched artifact aliases manually; if genuinely supporting, add alias/normalization bridge or adjust support audit.",
            "If matches are only lexical noise, keep current guard unchanged.",
        ]
    if primary == "retrieval_context_or_parser_constraint_gap":
        return [
            "Manually inspect whether parser dimensions are overconstrained or whether top-k page context missed required evidence.",
            "If parser is overconstrained, simplify evidence_dimensions; if context is missing, fix retrieval/coverage before QA.",
        ]
    return ["Manual review required before changing selector, parser, or artifact extraction."]


def build_gate(args: argparse.Namespace, record_ids: list[int], r063_gate: Mapping[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_records_match_r063_small_set": sorted(record_ids) == sorted(row["record_id"] for row in audits),
        "r063_gate_was_passed": r063_gate.get("gate_passed") is True,
        "all_records_have_root_cause": all(row.get("root_cause_attribution", {}).get("primary_root_cause") for row in audits),
        "all_records_have_coverage_matrix": all(row.get("coverage_matrix", {}).get("schema_version") == "r064_coverage_matrix_v1" for row in audits),
        "no_gold_fields_in_audits": all(not row.get("forbidden_gold_fields_present") for row in audits),
        "does_not_claim_artifact_lift": True,
        "not_official_score": True,
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r064_parser_artifact_mismatch_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r064_parser_artifact_mismatch_gate_pass" if not hard_failures else "r064_parser_artifact_mismatch_needs_fix",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "target_record_ids": record_ids,
        "root_cause_counts": dict(Counter(row["root_cause_attribution"]["primary_root_cause"] for row in audits)),
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, audits: list[dict[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "r064_parser_artifact_mismatch_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r064_parser_artifact_mismatch_complete" if gate["gate_passed"] else "r064_parser_artifact_mismatch_needs_fix",
        "scope": {
            "target_records_only": gate["target_record_ids"],
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_mmlongbench_result": True,
            "does_not_prove_artifact_positive_lift": True,
            "mismatch_attribution_only": True,
        },
        "inputs": {
            "r063_comparisons": args.r063_comparisons,
            "r063_gate": args.r063_gate,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
        },
        "summary": {
            "num_records": len(audits),
            "root_cause_counts": gate["root_cause_counts"],
            "records_by_root_cause": records_by_root_cause(audits),
        },
        "per_record": [compact_record(row) for row in audits],
        "gate": dict(gate),
        "recommended_next": [
            "Do not run more models or full QA from R064.",
            "First fix parser/code-type normalization for parser_answer_type_misclassified cases.",
            "Then repair artifact extraction/normalization for page-visible dimensions and exact key/value or operand gaps.",
            "Rerun a no-provider coverage audit before any provider QA experiment.",
        ],
    }


def compact_record(row: Mapping[str, Any]) -> dict[str, Any]:
    matrix = row["coverage_matrix"]
    return {
        "record_id": row["record_id"],
        "answer_type": row.get("answer_type"),
        "r063_llm_guard": row["r063_summary"].get("llm_guard"),
        "primary_root_cause": row["root_cause_attribution"]["primary_root_cause"],
        "all_categories": row["root_cause_attribution"].get("all_categories"),
        "dimension_count": matrix.get("dimension_count"),
        "artifact_dimension_covered_count": matrix.get("artifact_dimension_covered_count"),
        "page_dimension_covered_count": matrix.get("page_dimension_covered_count"),
        "missing_artifact_dimensions": matrix.get("missing_artifact_dimensions"),
        "missing_page_dimensions": matrix.get("missing_page_dimensions"),
        "recommended_fix": row.get("recommended_fix"),
    }


def records_by_root_cause(audits: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for row in audits:
        key = row["root_cause_attribution"]["primary_root_cause"]
        grouped.setdefault(key, []).append(int(row["record_id"]))
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def build_compact_index(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_record(row) for row in audits]


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = [
        "# R064 Parser/Artifact Mismatch Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Reads R063 parser/selector outputs and public artifact/page context only.",
        "- Attributes mismatch causes; does not report a score or artifact-lift claim.",
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
        "# R064 Parser/Artifact Mismatch Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- No official score and no artifact-lift claim.",
        "- Audits where R063 parser requirements fail to connect to artifact snippets/page context.",
        "",
        "## Summary",
        f"- records: {report['summary']['num_records']}",
        f"- root cause counts: `{json.dumps(report['summary']['root_cause_counts'], sort_keys=True)}`",
        f"- records by root cause: `{json.dumps(report['summary']['records_by_root_cause'], sort_keys=True)}`",
        "",
        "## Per Record",
    ]
    for row in report["per_record"]:
        lines.append(
            f"- {row['record_id']}: root=`{row['primary_root_cause']}`, answer_type=`{row['answer_type']}`, "
            f"artifact_dims={row['artifact_dimension_covered_count']}/{row['dimension_count']}, "
            f"page_dims={row['page_dimension_covered_count']}/{row['dimension_count']}, "
            f"missing_artifact=`{row['missing_artifact_dimensions']}`, missing_page=`{row['missing_page_dimensions']}`"
        )
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


def parse_record_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
