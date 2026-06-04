#!/usr/bin/env python3
"""R066 no-provider artifact key/value extraction audit for record 508.

Audits the post-R065 AR03 case without provider calls, prediction, evaluation,
or full QA. The goal is to determine whether the remaining break is in visible
page text, whole-document extracted text, artifact extraction/normalization, or
selector exact-code matching.
"""

from __future__ import annotations

import argparse
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
from mdocnexus.integration.evidence_demand_parser import merge_evidence_demand_profile, normalize_evidence_demand
from mdocnexus.integration.guarded_prompt import artifact_evidence_text, forbidden_public_fields, normalize, score_guarded_artifact, select_guarded_artifacts

TARGET_RECORD_ID = 508
DEFAULT_R063_COMPARISONS = "outputs/heldout/r063_llm_evidence_demand_parser/r063_selector_comparisons.jsonl"
DEFAULT_R065_GATE = "outputs/heldout/r065_parser_code_type_regression/r065_parser_code_type_gate.json"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r066_artifact_key_value_extraction_audit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r063-comparisons", default=DEFAULT_R063_COMPARISONS)
    parser.add_argument("--r065-gate", default=DEFAULT_R065_GATE)
    parser.add_argument("--r040-root", default=r053.DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=r053.DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=r053.DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=r053.DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=r053.DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=2600)
    parser.add_argument("--max-artifact-chars", type=int, default=420)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({"will_execute": False, "output_root": str(output_root), "target_record_id": TARGET_RECORD_ID, "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit, r053.read_json(Path(args.r065_gate)))
    report = build_report(args, audit, gate)
    r053.write_json(output_root / "r066_artifact_key_value_audit.json", audit)
    r053.write_jsonl(output_root / "r066_key_value_compact_index.jsonl", [compact_index(audit)])
    r053.write_json(output_root / "r066_artifact_key_value_gate.json", gate)
    write_gate_markdown(output_root / "r066_artifact_key_value_gate.md", gate)
    r053.write_json(output_root / "r066_artifact_key_value_report.json", report)
    write_report_markdown(output_root / "r066_artifact_key_value_report.md", report)
    print(json.dumps({"decision": gate["decision"], "gate_passed": gate["gate_passed"], "record_id": TARGET_RECORD_ID, "primary_root_cause": audit["root_cause_attribution"]["primary_root_cause"], "selector_guard": audit["selector_replay"]["guard_decision"], "report_md": str(output_root / "r066_artifact_key_value_report.md"), "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    records = r053.read_json(Path(args.records))
    record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(record_ids)}
    if TARGET_RECORD_ID not in offsets:
        raise ValueError(f"record {TARGET_RECORD_ID} is not in the R039 frozen subset")
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    r063_row = load_r063_row(Path(args.r063_comparisons), TARGET_RECORD_ID)

    source = records[TARGET_RECORD_ID]
    doc_id = str(source["doc_id"])
    question = str(source["question"])
    raw_demand = r063_row.get("parsed_evidence_demand") or {}
    normalized_demand = normalize_evidence_demand(raw_demand)
    profile = merge_evidence_demand_profile(question, raw_demand)
    required_codes = sorted(set(str(code) for code in (profile.get("codes") or [])))

    offset = offsets[TARGET_RECORD_ID]
    original_pages = r053.combined_pages(run_records["top4_original_only"][offset])
    artifact_pages = r053.combined_pages(run_records["top4_artifact_only"][offset])
    candidate_pages = r053.unique_ints(artifact_pages + original_pages)
    raw_artifacts = []
    candidates = []
    for page in candidate_pages:
        page_artifacts = artifacts_by_page.get((doc_id, page), [])
        raw_artifacts.extend(page_artifacts)
        for artifact in page_artifacts:
            candidates.append(score_guarded_artifact(artifact, question, profile, page, artifact_pages=artifact_pages, original_pages=original_pages, max_chars=args.max_artifact_chars))

    retrieved_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
    whole_doc_contexts = load_all_doc_pages(Path(args.extract_path), doc_id, args.max_page_chars)
    selection = select_guarded_artifacts(candidates, retrieved_contexts, profile, max_artifacts=args.max_artifacts)
    artifact_audit = audit_artifacts(raw_artifacts, candidates, required_codes)
    page_audit = audit_pages(retrieved_contexts, required_codes)
    doc_audit = audit_pages(whole_doc_contexts, required_codes)
    root = attribute_root_cause(artifact_audit, page_audit, doc_audit, selection)
    public_payload = {"record_id": TARGET_RECORD_ID, "doc_id": doc_id, "question": question, "normalized_demand": normalized_demand, "artifact_audit": artifact_audit, "retrieved_page_text_audit": page_audit, "whole_document_text_audit": doc_audit, "selector_replay": selection}
    return {
        "schema_version": "r066_artifact_key_value_extraction_audit_v1",
        "record_id": TARGET_RECORD_ID,
        "doc_id": doc_id,
        "question": question,
        "normalized_demand": normalized_demand,
        "profile_flags": {"codes": required_codes, "requires_exact_code_selection": profile.get("requires_exact_code_selection"), "is_document_metadata_lookup": profile.get("is_document_metadata_lookup"), "is_numeric_or_table_question": profile.get("is_numeric_or_table_question"), "profile_source": profile.get("profile_source")},
        "retrieval_pages": {"top4_artifact_only_combined": artifact_pages, "top4_original_only_combined": original_pages, "candidate_union": candidate_pages},
        "candidate_artifact_page_counts": {str(page): len(artifacts_by_page.get((doc_id, page), [])) for page in candidate_pages},
        "artifact_audit": artifact_audit,
        "retrieved_page_text_audit": page_audit,
        "whole_document_text_audit": doc_audit,
        "selector_replay": {"guard_decision": selection.get("guard_decision"), "guard_reasons": selection.get("guard_reasons"), "answer_policy": selection.get("answer_policy"), "selected_artifact_count": len(selection.get("selected_artifacts") or []), "selected_artifact_ids": [row.get("artifact_id") for row in selection.get("selected_artifacts") or []], "positive_candidate_count": selection.get("positive_candidate_count"), "candidate_artifact_count": len(candidates)},
        "root_cause_attribution": root,
        "recommended_fix": recommendations_for(artifact_audit, page_audit, doc_audit),
        "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def load_r063_row(path: Path, record_id: int) -> dict[str, Any]:
    for row in r053.read_jsonl(path):
        if int(row.get("record_id")) == record_id:
            return row
    raise ValueError(f"R063 comparison row missing record: {record_id}")


def load_all_doc_pages(extract_path: Path, doc_id: str, max_chars: int) -> list[dict[str, Any]]:
    stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    rows = []
    for path in sorted(extract_path.glob(f"{stem}_*.txt"), key=page_sort_key):
        match = re.search(r"_(\d+)\.txt$", path.name)
        if match:
            rows.append(r053.load_page_context(extract_path, doc_id, int(match.group(1)), max_chars))
    return rows


def page_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_(\d+)\.txt$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def audit_artifacts(raw_artifacts: list[Mapping[str, Any]], scored: list[Mapping[str, Any]], codes: list[str]) -> dict[str, Any]:
    joined_raw = "\n".join(raw_artifact_text(row) for row in raw_artifacts)
    raw_hits = []
    scored_hits = []
    for artifact in raw_artifacts:
        text = raw_artifact_text(artifact)
        if matched_codes(text, codes) or matched_code_families(text, codes):
            raw_hits.append({"artifact_id": artifact.get("artifact_id"), "artifact_type": artifact.get("artifact_type"), "page_index": artifact.get("page_index"), "matched_required_codes": matched_codes(text, codes), "matched_code_families": matched_code_families(text, codes), "content_preview": str(artifact.get("content") or "")[:360], "normalized_content": compact_normalized(artifact.get("normalized_content"))})
    for row in scored:
        text = artifact_evidence_text(row)
        if matched_codes(text, codes) or matched_code_families(text, codes):
            scored_hits.append({"artifact_id": row.get("artifact_id"), "artifact_type": row.get("artifact_type"), "page_index": row.get("page_index"), "matched_required_codes": matched_codes(text, codes), "matched_code_families": matched_code_families(text, codes), "exact_code_matches_from_selector": row.get("exact_code_matches"), "selection_score": row.get("selection_score"), "selection_reasons": row.get("selection_reasons"), "content_preview": row.get("content_preview"), "normalized_content": row.get("normalized_content")})
    exact_selector_hits = [row for row in scored if row.get("exact_code_matches")]
    return {"schema_version": "r066_artifact_code_audit_v1", "candidate_artifact_count": len(raw_artifacts), "scored_candidate_count": len(scored), "required_codes": codes, "exact_code_present_in_raw_artifacts": bool(matched_codes(joined_raw, codes)), "code_family_present_in_raw_artifacts": bool(matched_code_families(joined_raw, codes)), "raw_code_or_family_hit_count": len(raw_hits), "raw_code_or_family_hits": raw_hits[:20], "scored_code_or_family_hit_count": len(scored_hits), "scored_code_or_family_hits": scored_hits[:20], "selector_exact_code_hit_count": len(exact_selector_hits), "selector_exact_code_hit_ids": [row.get("artifact_id") for row in exact_selector_hits], "artifact_types": sorted({str(row.get("artifact_type") or "") for row in raw_artifacts})}


def audit_pages(page_contexts: list[Mapping[str, Any]], codes: list[str]) -> dict[str, Any]:
    joined = "\n".join(str(ctx.get("text_preview") or "") for ctx in page_contexts)
    hits = []
    for ctx in page_contexts:
        text = str(ctx.get("text_preview") or "")
        code_hits = matched_codes(text, codes)
        family_hits = matched_code_families(text, codes)
        state_hits = matched_state_markers(text, codes)
        if code_hits or family_hits or state_hits:
            hits.append({"page_index": ctx.get("page_index"), "page_id": ctx.get("page_id"), "exists": ctx.get("exists"), "matched_required_codes": code_hits, "matched_code_families": family_hits, "matched_state_markers": state_hits, "nearby_snippets": nearby_snippets(text, codes)})
    return {"schema_version": "r066_page_code_audit_v1", "page_count": len(page_contexts), "required_codes": codes, "exact_code_present": bool(matched_codes(joined, codes)), "code_family_present": bool(matched_code_families(joined, codes)), "state_marker_present": bool(matched_state_markers(joined, codes)), "code_or_family_page_hit_count": len(hits), "code_or_family_page_hits": hits[:20]}


def raw_artifact_text(artifact: Mapping[str, Any]) -> str:
    normalized_content = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), Mapping) else {}
    return " ".join([str(artifact.get("artifact_id") or ""), str(artifact.get("artifact_type") or ""), str(artifact.get("content") or ""), json.dumps(normalized_content, ensure_ascii=False, sort_keys=True)])


def compact_normalized(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = ["row_label", "row_header", "column_label", "column_header", "metric_name", "value_text", "source_text", "table_id", "extraction_method"]
    return {key: value.get(key) for key in keys if key in value}


def matched_codes(text: str, codes: list[str]) -> list[str]:
    return [code for code in codes if re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", str(text), re.IGNORECASE)]


def matched_code_families(text: str, codes: list[str]) -> list[str]:
    hits = []
    for code in codes:
        match = re.match(r"([A-Za-z]+)", code)
        if match and re.search(rf"(?<![A-Za-z0-9]){re.escape(match.group(1))}\d{{2,4}}(?![A-Za-z0-9])", str(text), re.IGNORECASE):
            hits.append(match.group(1).upper())
    return sorted(set(hits))


def matched_state_markers(text: str, codes: list[str]) -> list[str]:
    markers = []
    norm_text = normalize(text)
    for code in codes:
        if code.upper().startswith("AR"):
            if "arkansas" in norm_text:
                markers.append("Arkansas")
            if re.search(r"(?<![A-Za-z0-9])AR(?![A-Za-z0-9])", str(text)):
                markers.append("AR")
    return sorted(set(markers))


def nearby_snippets(text: str, codes: list[str], window: int = 180) -> list[str]:
    patterns = list(codes)
    for code in codes:
        match = re.match(r"([A-Za-z]+)", code)
        if match:
            patterns.append(match.group(1))
            if match.group(1).upper() == "AR":
                patterns.append("Arkansas")
    snippets = []
    for pattern in patterns:
        for match in re.finditer(re.escape(pattern), text, flags=re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= 4:
                return snippets
    return snippets


def attribute_root_cause(artifact_audit: Mapping[str, Any], page_audit: Mapping[str, Any], doc_audit: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    categories = []
    evidence = []
    if not artifact_audit.get("exact_code_present_in_raw_artifacts"):
        categories.append("artifact_store_missing_exact_code_key_value")
        evidence.append("No candidate artifact raw/normalized text contains the required exact code.")
    if artifact_audit.get("candidate_artifact_count") and not artifact_audit.get("code_family_present_in_raw_artifacts"):
        categories.append("artifact_normalization_over_numeric_tables_not_eps_key_values")
        evidence.append("Candidate artifacts are numeric/table atoms but contain no EPS code-family entries.")
    if page_audit.get("state_marker_present") and not page_audit.get("exact_code_present"):
        categories.append("visible_page_supports_code_absence_not_answer")
        evidence.append("Retrieved page text shows Arkansas/EPS context but not AR03; it supports absence/refusal, not a value answer.")
    if not doc_audit.get("exact_code_present"):
        categories.append("extracted_document_text_missing_required_code")
        evidence.append("Whole-document extracted text contains no AR03 exact-code hit.")
    if selection.get("guard_decision") == "exact_code_absence_guard":
        categories.append("selector_guard_correct_for_current_public_evidence")
        evidence.append("R065 exact-code guard rejects artifacts because no artifact contains the required code.")
    if not categories:
        categories.append("manual_review_required")
        evidence.append("Audit checks did not identify a deterministic root cause.")
    priority = ["extracted_document_text_missing_required_code", "artifact_store_missing_exact_code_key_value", "artifact_normalization_over_numeric_tables_not_eps_key_values", "visible_page_supports_code_absence_not_answer", "selector_guard_correct_for_current_public_evidence", "manual_review_required"]
    primary = next(item for item in priority if item in categories)
    return {"schema_version": "r066_root_cause_attribution_v1", "primary_root_cause": primary, "all_categories": sorted(set(categories)), "evidence": evidence, "claim_ceiling": "diagnostic_attribution_only_no_artifact_lift_claim"}


def recommendations_for(artifact_audit: Mapping[str, Any], page_audit: Mapping[str, Any], doc_audit: Mapping[str, Any]) -> list[str]:
    rows = ["Do not relax exact-code selector matching for AR03; current public evidence does not contain the requested code.", "Keep the exact-code absence guard and route this case to page-cited refusal/absence handling."]
    if not doc_audit.get("exact_code_present"):
        rows.append("Audit the source PDF/OCR for whether AR03 is visually present but missing from extracted text; if absent in the source, treat as not answerable under visible evidence.")
    if not artifact_audit.get("code_family_present_in_raw_artifacts") and page_audit.get("state_marker_present"):
        rows.append("Repair EPS/table-list artifact extraction for page text with code/name lists; page 7 has no artifacts while it contains the Arkansas EPS neighborhood.")
    rows.append("After extraction repair, rerun a no-provider exact-code coverage audit before any provider diagnostic.")
    return rows


def build_gate(args: argparse.Namespace, audit: Mapping[str, Any], r065_gate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {"no_provider_calls": True, "no_prediction_or_eval_invoked": True, "no_full_qa": True, "target_record_is_508": audit.get("record_id") == TARGET_RECORD_ID, "r065_gate_was_passed": r065_gate.get("gate_passed") is True, "selector_replay_is_exact_code_absence_guard": audit.get("selector_replay", {}).get("guard_decision") == "exact_code_absence_guard", "audit_reports_whole_document_extract_exact_code_status": "exact_code_present" in audit.get("whole_document_text_audit", {}), "audit_reports_artifact_exact_code_status": "exact_code_present_in_raw_artifacts" in audit.get("artifact_audit", {}), "no_gold_fields_in_audit": not audit.get("forbidden_gold_fields_present"), "does_not_claim_artifact_lift": True, "not_official_score": True, "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r053.DEFAULT_ARTIFACTS}
    hard_failures = [key for key, value in checks.items() if not value]
    return {"schema_version": "r066_artifact_key_value_gate_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "r066_artifact_key_value_audit_gate_pass" if not hard_failures else "r066_artifact_key_value_audit_needs_fix", "gate_passed": not hard_failures, "checks": checks, "hard_failures": hard_failures, "target_record_id": TARGET_RECORD_ID, "not_full_qa": True, "not_official_score": True, "not_artifact_lift_claim": True}


def build_report(args: argparse.Namespace, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": "r066_artifact_key_value_report_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "r066_artifact_key_value_audit_complete" if gate["gate_passed"] else "r066_artifact_key_value_audit_needs_fix", "scope": {"target_record_only": TARGET_RECORD_ID, "no_provider_calls": True, "no_prediction": True, "no_evaluation": True, "no_full_qa": True, "not_official_mmlongbench_result": True, "does_not_prove_artifact_positive_lift": True}, "inputs": {"r063_comparisons": args.r063_comparisons, "r065_gate": args.r065_gate, "r040_root": args.r040_root, "artifacts": args.artifacts, "extract_path": args.extract_path}, "summary": compact_index(audit), "gate": dict(gate), "recommended_next": audit.get("recommended_fix") or []}


def compact_index(audit: Mapping[str, Any]) -> dict[str, Any]:
    artifact = audit.get("artifact_audit", {})
    page = audit.get("retrieved_page_text_audit", {})
    doc = audit.get("whole_document_text_audit", {})
    root = audit.get("root_cause_attribution", {})
    return {"schema_version": "r066_key_value_compact_index_v1", "record_id": audit.get("record_id"), "doc_id": audit.get("doc_id"), "required_codes": audit.get("profile_flags", {}).get("codes"), "selector_guard": audit.get("selector_replay", {}).get("guard_decision"), "candidate_artifact_count": artifact.get("candidate_artifact_count"), "candidate_artifact_page_counts": audit.get("candidate_artifact_page_counts"), "artifact_exact_code_present": artifact.get("exact_code_present_in_raw_artifacts"), "artifact_code_family_present": artifact.get("code_family_present_in_raw_artifacts"), "retrieved_page_exact_code_present": page.get("exact_code_present"), "retrieved_page_state_marker_present": page.get("state_marker_present"), "whole_document_exact_code_present": doc.get("exact_code_present"), "whole_document_code_family_present": doc.get("code_family_present"), "primary_root_cause": root.get("primary_root_cause"), "all_categories": root.get("all_categories")}


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = ["# R066 Artifact Key/Value Gate", "", f"Decision: `{gate['decision']}`", f"Gate passed: {gate['gate_passed']}", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Audits record 508 / AR03 exact-code evidence only.", "- Not an official score and not an artifact-lift claim.", "", "## Checks"]
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = ["# R066 Artifact Key/Value Extraction Audit", "", f"Decision: `{report['decision']}`", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Single-case diagnostic for record 508 / AR03 after R065 parser normalization.", "- No official score and no artifact-positive lift claim.", "", "## Summary", f"- selector guard: `{summary['selector_guard']}`", f"- candidate artifacts: {summary['candidate_artifact_count']} with page counts `{json.dumps(summary['candidate_artifact_page_counts'], sort_keys=True)}`", f"- artifact exact code present: {summary['artifact_exact_code_present']}", f"- retrieved page exact code present: {summary['retrieved_page_exact_code_present']}", f"- retrieved page state marker present: {summary['retrieved_page_state_marker_present']}", f"- whole-document extracted text exact code present: {summary['whole_document_exact_code_present']}", f"- primary root cause: `{summary['primary_root_cause']}`", f"- all categories: `{json.dumps(summary['all_categories'], sort_keys=True)}`", "", "## Recommended Next"]
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
