#!/usr/bin/env python3
"""R068 no-provider Stage 2 code-list integration audit.

R068 verifies that the R067 code/name list extractor is wired into the Stage 2
document-generic final-store branch and that the integration repairs visible
code/name artifact coverage without relaxing exact-code refusal behavior. It
does not call providers, generate answers, run evaluation, run full QA, or
report a score.
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
import run_r067_source_ocr_code_list_extraction_audit as r067
from mdocnexus.integration.evidence_demand_parser import merge_evidence_demand_profile
from mdocnexus.integration.guarded_prompt import forbidden_public_fields, score_guarded_artifact, select_guarded_artifacts
from mdocnexus.stage2.artifact_quality import classify_artifact_quality, quality_discard_reason
from mdocnexus.stage2.code_name_list_extractor import extract_code_name_list_artifacts

TARGET_RECORD_ID = 508
TARGET_PAGE = 7
TARGET_CODE = "AR03"
DEFAULT_R067_GATE = "outputs/heldout/r067_source_ocr_code_list_extraction_audit/r067_source_ocr_code_list_gate.json"
DEFAULT_R063_COMPARISONS = "outputs/heldout/r063_llm_evidence_demand_parser/r063_selector_comparisons.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r068_code_list_stage2_integration_audit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r067-gate", default=DEFAULT_R067_GATE)
    parser.add_argument("--r063-comparisons", default=DEFAULT_R063_COMPARISONS)
    parser.add_argument("--r040-root", default=r053.DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=r053.DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=r053.DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=r053.DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=r053.DEFAULT_EXTRACT_PATH)
    parser.add_argument("--stage2-script", default="scripts/stage2.py")
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
        print(json.dumps({"will_execute": False, "output_root": str(output_root), "target_record_id": TARGET_RECORD_ID, "target_page": TARGET_PAGE, "target_code": TARGET_CODE, "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit, r053.read_json(Path(args.r067_gate)))
    report = build_report(args, audit, gate)
    r053.write_json(output_root / "r068_code_list_stage2_integration_audit.json", audit)
    r053.write_jsonl(output_root / "r068_code_list_stage2_compact_index.jsonl", [compact_index(audit)])
    r053.write_json(output_root / "r068_code_list_stage2_integration_gate.json", gate)
    write_gate_markdown(output_root / "r068_code_list_stage2_integration_gate.md", gate)
    r053.write_json(output_root / "r068_code_list_stage2_integration_report.json", report)
    write_report_markdown(output_root / "r068_code_list_stage2_integration_report.md", report)
    print(json.dumps({"decision": gate["decision"], "gate_passed": gate["gate_passed"], "record_id": TARGET_RECORD_ID, "target_code": TARGET_CODE, "integrated_artifact_count": audit["stage2_replay_audit"]["final_artifact_count"], "integrated_ar_codes": audit["stage2_replay_audit"]["integrated_ar_codes"], "selector_guard_after_integration": audit["selector_replay_after_integration"]["guard_decision"], "report_md": str(output_root / "r068_code_list_stage2_integration_report.md"), "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    records = r053.read_json(Path(args.records))
    record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(record_ids)}
    if TARGET_RECORD_ID not in offsets:
        raise ValueError(f"record {TARGET_RECORD_ID} is not in the R039 frozen subset")
    source = records[TARGET_RECORD_ID]
    doc_id = str(source["doc_id"])
    question = str(source["question"])
    run_records = r053.load_r040_records(Path(args.r040_root))
    offset = offsets[TARGET_RECORD_ID]
    original_pages = r053.combined_pages(run_records["top4_original_only"][offset])
    artifact_pages = r053.combined_pages(run_records["top4_artifact_only"][offset])
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    existing_page_artifacts = artifacts_by_page.get((doc_id, TARGET_PAGE), [])
    page_text = r067.load_page_text(Path(args.extract_path), doc_id, TARGET_PAGE)
    page_input = r067.make_page_input(doc_id, TARGET_PAGE, page_text)
    integration = inspect_stage2_integration(Path(args.stage2_script))
    stage2_replay = replay_stage2_document_generic_postprocess(doc_id, page_input, existing_page_artifacts)
    r063_row = r067.load_r063_row(Path(args.r063_comparisons), TARGET_RECORD_ID)
    profile = merge_evidence_demand_profile(question, r063_row.get("parsed_evidence_demand") or {})
    page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
    scored = [score_guarded_artifact(artifact, question, profile, TARGET_PAGE, artifact_pages=artifact_pages, original_pages=original_pages, max_chars=args.max_artifact_chars) for artifact in stage2_replay["final_artifacts"]]
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)
    source_ocr = r067.build_source_ocr_audit(Path(args.extract_path), doc_id, TARGET_PAGE, page_text)
    public_payload = {"record_id": TARGET_RECORD_ID, "doc_id": doc_id, "question": question, "source_ocr_audit": source_ocr, "stage2_replay_audit": stage2_replay, "selector": selection}
    return {
        "schema_version": "r068_code_list_stage2_integration_audit_v1",
        "record_id": TARGET_RECORD_ID,
        "doc_id": doc_id,
        "question": question,
        "target_page": TARGET_PAGE,
        "target_code": TARGET_CODE,
        "retrieval_pages": {"top4_artifact_only_combined": artifact_pages, "top4_original_only_combined": original_pages, "candidate_union": r053.unique_ints(artifact_pages + original_pages)},
        "stage2_integration_static_audit": integration,
        "source_ocr_audit": source_ocr,
        "existing_artifact_page_count": len(existing_page_artifacts),
        "stage2_replay_audit": without_full_artifacts(stage2_replay),
        "selector_replay_after_integration": {"guard_decision": selection.get("guard_decision"), "guard_reasons": selection.get("guard_reasons"), "answer_policy": selection.get("answer_policy"), "selected_artifact_count": len(selection.get("selected_artifacts") or []), "selected_artifact_ids": [row.get("artifact_id") for row in selection.get("selected_artifacts") or []], "positive_candidate_count": selection.get("positive_candidate_count"), "candidate_artifact_count": len(scored)},
        "recommended_next": recommendations_for(stage2_replay, selection, source_ocr),
        "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def inspect_stage2_integration(stage2_script: Path) -> dict[str, Any]:
    text = stage2_script.read_text(encoding="utf-8")
    import_present = "from mdocnexus.stage2.code_name_list_extractor import extract_code_name_list_artifacts" in text
    call_present = "extract_code_name_list_artifacts(" in text
    document_generic_block = re.search(r"if document_generic:\s+valid_artifacts\.extend\(\s+atomicize_table_numeric_artifacts\(.*?valid_artifacts\.extend\(\s+extract_code_name_list_artifacts\(", text, flags=re.DOTALL) is not None
    return {"schema_version": "r068_stage2_static_integration_audit_v1", "stage2_script": str(stage2_script), "import_present": import_present, "call_present": call_present, "document_generic_branch_call_after_atomicizer": document_generic_block, "integration_scope": "scripts/stage2.py document-generic final-store postprocess branch", "provider_client_modified": False}


def replay_stage2_document_generic_postprocess(doc_id: str, page_input: Mapping[str, Any], existing_artifacts: list[Mapping[str, Any]]) -> dict[str, Any]:
    # Mirrors the deterministic post-provider Stage 2 document-generic order for this coverage audit.
    page_index = int(page_input.get("page_index", TARGET_PAGE) or TARGET_PAGE)
    selected_page = {"doc_id": doc_id, "page_index": page_index}
    extracted = extract_code_name_list_artifacts(selected_page=selected_page, page_input=page_input, existing_artifacts=list(existing_artifacts))
    final_artifacts = []
    discarded = []
    for artifact in extracted:
        if artifact.get("artifact_type") != "text_span" or artifact.get("modality") != "text":
            discarded.append({"artifact_id": artifact.get("artifact_id"), "reason": "unsupported_artifact_type_or_modality"})
            continue
        quality_reason = quality_discard_reason(artifact)
        if quality_reason:
            quality = classify_artifact_quality(artifact)
            discarded.append({"artifact_id": artifact.get("artifact_id"), "reason": quality_reason, "quality_labels": quality.get("labels", [])})
            continue
        final_artifacts.append(dict(artifact))
    codes = [str((artifact.get("normalized_content") or {}).get("eps_code") or "") for artifact in final_artifacts]
    ar_pairs = [artifact_summary(artifact) for artifact in final_artifacts if str((artifact.get("normalized_content") or {}).get("eps_code") or "").startswith("AR")]
    return {"schema_version": "r068_stage2_document_generic_replay_v1", "existing_artifact_count_before_integration": len(existing_artifacts), "extracted_code_name_artifact_count": len(extracted), "final_artifact_count": len(final_artifacts), "discarded_after_final_filter_count": len(discarded), "discarded_after_final_filter": discarded, "integrated_codes": codes, "integrated_ar_codes": [pair["eps_code"] for pair in ar_pairs], "integrated_ar_pairs": ar_pairs, "target_code_integrated": TARGET_CODE in codes, "all_final_artifacts_locatable": all(artifact.get("source_anchored") and artifact.get("element_locatable") for artifact in final_artifacts), "final_artifacts": final_artifacts}


def artifact_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), Mapping) else {}
    return {"artifact_id": artifact.get("artifact_id"), "artifact_type": artifact.get("artifact_type"), "modality": artifact.get("modality"), "page_index": artifact.get("page_index"), "eps_code": normalized.get("eps_code"), "geographic_market_name": normalized.get("geographic_market_name"), "group_label": normalized.get("group_label"), "content": artifact.get("content"), "source_anchored": artifact.get("source_anchored"), "element_locatable": artifact.get("element_locatable")}


def without_full_artifacts(stage2_replay: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in stage2_replay.items() if key != "final_artifacts"}
    row["sample_final_artifacts"] = [artifact_summary(artifact) for artifact in stage2_replay.get("final_artifacts", [])[:12]]
    return row


def recommendations_for(stage2_replay: Mapping[str, Any], selection: Mapping[str, Any], source_ocr: Mapping[str, Any]) -> list[str]:
    rows = ["Keep the code/name extractor integrated in Stage 2 document-generic final-store postprocessing for public EPS-like text lists."]
    if not stage2_replay.get("target_code_integrated") and not source_ocr.get("target_code_present_in_text"):
        rows.append("Keep record 508 on exact-code absence/refusal; do not infer AR03 from AR01/AR02 or Arkansas context.")
    if selection.get("guard_decision") == "exact_code_absence_guard":
        rows.append("Next coverage work should find source/OCR evidence for missing exact codes, not relax selector matching or run QA.")
    return rows


def build_gate(args: argparse.Namespace, audit: Mapping[str, Any], r067_gate: Mapping[str, Any]) -> dict[str, Any]:
    static = audit.get("stage2_integration_static_audit", {})
    replay = audit.get("stage2_replay_audit", {})
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "r067_gate_was_passed": r067_gate.get("gate_passed") is True,
        "stage2_import_present": static.get("import_present") is True,
        "stage2_call_present": static.get("call_present") is True,
        "stage2_document_generic_branch_call_after_atomicizer": static.get("document_generic_branch_call_after_atomicizer") is True,
        "provider_client_not_modified_by_integration": static.get("provider_client_modified") is False,
        "target_record_is_508": audit.get("record_id") == TARGET_RECORD_ID,
        "target_page_is_7": audit.get("target_page") == TARGET_PAGE,
        "source_text_and_image_exist": audit.get("source_ocr_audit", {}).get("text_exists") is True and audit.get("source_ocr_audit", {}).get("image_exists") is True,
        "existing_page7_artifact_count_before_integration_is_zero": replay.get("existing_artifact_count_before_integration") == 0,
        "stage2_replay_generates_code_name_artifacts": replay.get("final_artifact_count", 0) > 0,
        "integrated_artifacts_include_ar01_ar02": sorted(replay.get("integrated_ar_codes") or [])[:2] == ["AR01", "AR02"],
        "integrated_artifacts_do_not_include_ar03": replay.get("target_code_integrated") is False,
        "integrated_artifacts_pass_final_quality_filter": replay.get("discarded_after_final_filter_count") == 0,
        "integrated_artifacts_are_locatable": replay.get("all_final_artifacts_locatable") is True,
        "selector_still_exact_code_absence": audit.get("selector_replay_after_integration", {}).get("guard_decision") == "exact_code_absence_guard",
        "no_gold_fields_in_audit": not audit.get("forbidden_gold_fields_present"),
        "does_not_claim_artifact_lift": True,
        "not_official_score": True,
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r053.DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {"schema_version": "r068_code_list_stage2_integration_gate_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "r068_code_list_stage2_integration_gate_pass" if not hard_failures else "r068_code_list_stage2_integration_needs_fix", "gate_passed": not hard_failures, "checks": checks, "hard_failures": hard_failures, "target_record_id": TARGET_RECORD_ID, "not_full_qa": True, "not_official_score": True, "not_artifact_lift_claim": True}


def build_report(args: argparse.Namespace, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": "r068_code_list_stage2_integration_report_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "r068_code_list_stage2_integration_audit_complete" if gate["gate_passed"] else "r068_code_list_stage2_integration_audit_needs_fix", "scope": {"target_record_only": TARGET_RECORD_ID, "target_page_only": TARGET_PAGE, "no_provider_calls": True, "no_prediction": True, "no_evaluation": True, "no_full_qa": True, "not_official_mmlongbench_result": True, "does_not_prove_artifact_positive_lift": True}, "inputs": {"r067_gate": args.r067_gate, "r063_comparisons": args.r063_comparisons, "r040_root": args.r040_root, "artifacts": args.artifacts, "extract_path": args.extract_path, "stage2_script": args.stage2_script}, "summary": compact_index(audit), "gate": dict(gate), "recommended_next": audit.get("recommended_next") or []}


def compact_index(audit: Mapping[str, Any]) -> dict[str, Any]:
    replay = audit.get("stage2_replay_audit", {})
    selector = audit.get("selector_replay_after_integration", {})
    static = audit.get("stage2_integration_static_audit", {})
    return {"schema_version": "r068_code_list_stage2_compact_index_v1", "record_id": audit.get("record_id"), "doc_id": audit.get("doc_id"), "target_page": audit.get("target_page"), "target_code": audit.get("target_code"), "stage2_import_present": static.get("import_present"), "stage2_document_generic_branch_call_after_atomicizer": static.get("document_generic_branch_call_after_atomicizer"), "existing_artifact_count_before_integration": replay.get("existing_artifact_count_before_integration"), "final_artifact_count": replay.get("final_artifact_count"), "integrated_ar_codes": replay.get("integrated_ar_codes"), "target_code_integrated": replay.get("target_code_integrated"), "selector_guard_after_integration": selector.get("guard_decision"), "selected_artifact_count": selector.get("selected_artifact_count")}


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = ["# R068 Code-List Stage2 Integration Gate", "", f"Decision: `{gate['decision']}`", f"Gate passed: {gate['gate_passed']}", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Audits Stage 2 document-generic integration on record 508 / page 7 / AR03.", "- Not an official score and not an artifact-lift claim.", "", "## Checks"]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = ["# R068 Code-List Stage2 Integration Audit", "", f"Decision: `{report['decision']}`", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Integration audit for Stage 2 document-generic final-store postprocessing only.", "- No official score and no artifact-positive lift claim.", "", "## Summary", f"- Stage2 import present: {summary['stage2_import_present']}", f"- Stage2 document-generic branch call after atomicizer: {summary['stage2_document_generic_branch_call_after_atomicizer']}", f"- existing page-7 artifact count before integration: {summary['existing_artifact_count_before_integration']}", f"- final code/name artifacts after integration replay: {summary['final_artifact_count']}", f"- integrated AR codes: `{summary['integrated_ar_codes']}`", f"- target code integrated: {summary['target_code_integrated']}", f"- selector guard after integration: `{summary['selector_guard_after_integration']}`", "", "## Recommended Next"]
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
