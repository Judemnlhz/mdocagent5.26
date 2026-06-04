#!/usr/bin/env python3
"""R067 no-provider source/OCR and EPS code-list extraction audit.

R067 audits record 508 / AR03 after R066. It checks page-image/text presence,
confirms whether extracted public text contains the requested exact code, runs a
page-local code/name list extractor on page 7, and replays exact-code selection
against the repaired extracted artifacts. It does not call providers, generate
answers, run evaluation, run full QA, or report a score.
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
from mdocnexus.integration.evidence_demand_parser import merge_evidence_demand_profile
from mdocnexus.integration.guarded_prompt import forbidden_public_fields, score_guarded_artifact, select_guarded_artifacts
from mdocnexus.stage2.code_name_list_extractor import extract_code_name_list_artifacts

TARGET_RECORD_ID = 508
TARGET_PAGE = 7
TARGET_CODE = "AR03"
DEFAULT_R063_COMPARISONS = "outputs/heldout/r063_llm_evidence_demand_parser/r063_selector_comparisons.jsonl"
DEFAULT_R066_GATE = "outputs/heldout/r066_artifact_key_value_extraction_audit/r066_artifact_key_value_gate.json"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r067_source_ocr_code_list_extraction_audit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r063-comparisons", default=DEFAULT_R063_COMPARISONS)
    parser.add_argument("--r066-gate", default=DEFAULT_R066_GATE)
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
        print(json.dumps({"will_execute": False, "output_root": str(output_root), "target_record_id": TARGET_RECORD_ID, "target_page": TARGET_PAGE, "target_code": TARGET_CODE, "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit, r053.read_json(Path(args.r066_gate)))
    report = build_report(args, audit, gate)
    r053.write_json(output_root / "r067_source_ocr_code_list_audit.json", audit)
    r053.write_jsonl(output_root / "r067_code_list_compact_index.jsonl", [compact_index(audit)])
    r053.write_json(output_root / "r067_source_ocr_code_list_gate.json", gate)
    write_gate_markdown(output_root / "r067_source_ocr_code_list_gate.md", gate)
    r053.write_json(output_root / "r067_source_ocr_code_list_report.json", report)
    write_report_markdown(output_root / "r067_source_ocr_code_list_report.md", report)
    print(json.dumps({"decision": gate["decision"], "gate_passed": gate["gate_passed"], "record_id": TARGET_RECORD_ID, "target_code": TARGET_CODE, "extracted_pair_count": audit["code_list_extractor_audit"]["extracted_pair_count"], "target_code_extracted": audit["code_list_extractor_audit"]["target_code_extracted"], "selector_guard_with_extracted_artifacts": audit["selector_replay_with_extracted_artifacts"]["guard_decision"], "report_md": str(output_root / "r067_source_ocr_code_list_report.md"), "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))


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
    profile = merge_evidence_demand_profile(question, r063_row.get("parsed_evidence_demand") or {})
    offset = offsets[TARGET_RECORD_ID]
    original_pages = r053.combined_pages(run_records["top4_original_only"][offset])
    artifact_pages = r053.combined_pages(run_records["top4_artifact_only"][offset])
    candidate_pages = r053.unique_ints(artifact_pages + original_pages)

    extract_root = Path(args.extract_path)
    page_text = load_page_text(extract_root, doc_id, TARGET_PAGE)
    page_input = make_page_input(doc_id, TARGET_PAGE, page_text)
    existing_page_artifacts = artifacts_by_page.get((doc_id, TARGET_PAGE), [])
    extracted_artifacts = extract_code_name_list_artifacts(selected_page={"doc_id": doc_id, "page_index": TARGET_PAGE}, page_input=page_input, existing_artifacts=existing_page_artifacts)
    page_contexts = [r053.load_page_context(extract_root, doc_id, page, args.max_page_chars) for page in artifact_pages]
    scored = [score_guarded_artifact(artifact, question, profile, TARGET_PAGE, artifact_pages=artifact_pages, original_pages=original_pages, max_chars=args.max_artifact_chars) for artifact in extracted_artifacts]
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)

    source_ocr = build_source_ocr_audit(extract_root, doc_id, TARGET_PAGE, page_text)
    code_list = build_code_list_audit(extracted_artifacts)
    public_payload = {"record_id": TARGET_RECORD_ID, "doc_id": doc_id, "question": question, "source_ocr_audit": source_ocr, "code_list_extractor_audit": code_list, "selector": selection}
    return {
        "schema_version": "r067_source_ocr_code_list_extraction_audit_v1",
        "record_id": TARGET_RECORD_ID,
        "doc_id": doc_id,
        "question": question,
        "target_page": TARGET_PAGE,
        "target_code": TARGET_CODE,
        "retrieval_pages": {"top4_artifact_only_combined": artifact_pages, "top4_original_only_combined": original_pages, "candidate_union": candidate_pages},
        "source_ocr_audit": source_ocr,
        "existing_artifact_page_count": len(existing_page_artifacts),
        "code_list_extractor_audit": code_list,
        "selector_replay_with_extracted_artifacts": {"guard_decision": selection.get("guard_decision"), "guard_reasons": selection.get("guard_reasons"), "answer_policy": selection.get("answer_policy"), "selected_artifact_count": len(selection.get("selected_artifacts") or []), "selected_artifact_ids": [row.get("artifact_id") for row in selection.get("selected_artifacts") or []], "positive_candidate_count": selection.get("positive_candidate_count"), "candidate_artifact_count": len(scored)},
        "recommended_fix": recommendations_for(source_ocr, code_list, selection),
        "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_source_ocr_audit(extract_root: Path, doc_id: str, page: int, page_text: str) -> dict[str, Any]:
    stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    text_path = extract_root / f"{stem}_{page}.txt"
    image_path = extract_root / f"{stem}_{page}.png"
    codes = sorted(set(re.findall(r"\b[A-Z]{2,4}\d{2,4}\b", page_text)))
    return {"schema_version": "r067_source_ocr_audit_v1", "text_path": str(text_path), "image_path": str(image_path), "text_exists": text_path.is_file(), "image_exists": image_path.is_file(), "text_char_count": len(page_text), "target_code_present_in_text": exact_code_present(page_text, TARGET_CODE), "code_family_present_in_text": bool(re.search(r"\bAR\d{2,4}\b", page_text)), "state_marker_present_in_text": "Arkansas" in page_text or re.search(r"\bAR\b", page_text) is not None, "codes_in_text": codes, "ar_codes_in_text": [code for code in codes if code.startswith("AR")], "target_code_nearby_snippets": nearby_snippets(page_text, TARGET_CODE)}


def build_code_list_audit(artifacts: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for artifact in artifacts:
        normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), Mapping) else {}
        rows.append({"artifact_id": artifact.get("artifact_id"), "artifact_type": artifact.get("artifact_type"), "page_index": artifact.get("page_index"), "eps_code": normalized.get("eps_code"), "geographic_market_name": normalized.get("geographic_market_name"), "group_label": normalized.get("group_label"), "content": artifact.get("content"), "source_anchored": artifact.get("source_anchored"), "element_locatable": artifact.get("element_locatable")})
    codes = [str(row.get("eps_code")) for row in rows if row.get("eps_code")]
    return {"schema_version": "r067_code_list_extractor_audit_v1", "extracted_pair_count": len(rows), "extracted_codes": codes, "ar_pairs": [row for row in rows if str(row.get("eps_code") or "").startswith("AR")], "target_code_extracted": TARGET_CODE in codes, "all_artifacts_locatable": all(row.get("source_anchored") and row.get("element_locatable") for row in rows), "sample_pairs": rows[:12]}


def recommendations_for(source_ocr: Mapping[str, Any], code_list: Mapping[str, Any], selection: Mapping[str, Any]) -> list[str]:
    rows = ["Keep exact-code matching strict; the repaired code-list extractor must not infer AR03 from AR01/AR02 or Arkansas context."]
    if source_ocr.get("image_exists") and not source_ocr.get("target_code_present_in_text"):
        rows.append("Manually inspect the page image for AR03 only if the benchmark expects an answer; current OCR/text evidence does not support it.")
    if code_list.get("extracted_pair_count"):
        rows.append("Integrate the code-name list extractor into Stage 2 for EPS-like text lists, then rerun no-provider coverage before any provider diagnostic.")
    if selection.get("guard_decision") == "exact_code_absence_guard":
        rows.append("For record 508, route to page-cited absence/refusal unless source/OCR repair reveals exact AR03.")
    return rows


def build_gate(args: argparse.Namespace, audit: Mapping[str, Any], r066_gate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {"no_provider_calls": True, "no_prediction_or_eval_invoked": True, "no_full_qa": True, "target_record_is_508": audit.get("record_id") == TARGET_RECORD_ID, "target_page_is_7": audit.get("target_page") == TARGET_PAGE, "r066_gate_was_passed": r066_gate.get("gate_passed") is True, "source_text_and_image_exist": audit.get("source_ocr_audit", {}).get("text_exists") is True and audit.get("source_ocr_audit", {}).get("image_exists") is True, "target_code_absent_from_ocr_text": audit.get("source_ocr_audit", {}).get("target_code_present_in_text") is False, "extractor_recovers_ar01_ar02": sorted(pair.get("eps_code") for pair in audit.get("code_list_extractor_audit", {}).get("ar_pairs", [])) == ["AR01", "AR02"], "extractor_does_not_invent_ar03": audit.get("code_list_extractor_audit", {}).get("target_code_extracted") is False, "selector_still_exact_code_absence": audit.get("selector_replay_with_extracted_artifacts", {}).get("guard_decision") == "exact_code_absence_guard", "no_gold_fields_in_audit": not audit.get("forbidden_gold_fields_present"), "does_not_claim_artifact_lift": True, "not_official_score": True, "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r053.DEFAULT_ARTIFACTS}
    hard_failures = [key for key, value in checks.items() if not value]
    return {"schema_version": "r067_source_ocr_code_list_gate_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "r067_source_ocr_code_list_gate_pass" if not hard_failures else "r067_source_ocr_code_list_needs_fix", "gate_passed": not hard_failures, "checks": checks, "hard_failures": hard_failures, "target_record_id": TARGET_RECORD_ID, "not_full_qa": True, "not_official_score": True, "not_artifact_lift_claim": True}


def build_report(args: argparse.Namespace, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": "r067_source_ocr_code_list_report_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "r067_source_ocr_code_list_audit_complete" if gate["gate_passed"] else "r067_source_ocr_code_list_audit_needs_fix", "scope": {"target_record_only": TARGET_RECORD_ID, "target_page_only": TARGET_PAGE, "no_provider_calls": True, "no_prediction": True, "no_evaluation": True, "no_full_qa": True, "not_official_mmlongbench_result": True, "does_not_prove_artifact_positive_lift": True}, "inputs": {"r063_comparisons": args.r063_comparisons, "r066_gate": args.r066_gate, "r040_root": args.r040_root, "artifacts": args.artifacts, "extract_path": args.extract_path}, "summary": compact_index(audit), "gate": dict(gate), "recommended_next": audit.get("recommended_fix") or []}


def compact_index(audit: Mapping[str, Any]) -> dict[str, Any]:
    source = audit.get("source_ocr_audit", {})
    code_list = audit.get("code_list_extractor_audit", {})
    selector = audit.get("selector_replay_with_extracted_artifacts", {})
    return {"schema_version": "r067_code_list_compact_index_v1", "record_id": audit.get("record_id"), "doc_id": audit.get("doc_id"), "target_page": audit.get("target_page"), "target_code": audit.get("target_code"), "text_exists": source.get("text_exists"), "image_exists": source.get("image_exists"), "target_code_present_in_text": source.get("target_code_present_in_text"), "ar_codes_in_text": source.get("ar_codes_in_text"), "existing_artifact_page_count": audit.get("existing_artifact_page_count"), "extracted_pair_count": code_list.get("extracted_pair_count"), "ar_pairs": code_list.get("ar_pairs"), "target_code_extracted": code_list.get("target_code_extracted"), "selector_guard_with_extracted_artifacts": selector.get("guard_decision"), "selected_artifact_count": selector.get("selected_artifact_count")}


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = ["# R067 Source/OCR Code-List Gate", "", f"Decision: `{gate['decision']}`", f"Gate passed: {gate['gate_passed']}", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Audits record 508 / page 7 / AR03 only.", "- Not an official score and not an artifact-lift claim.", "", "## Checks"]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = ["# R067 Source/OCR Code-List Extraction Audit", "", f"Decision: `{report['decision']}`", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Single-case diagnostic for record 508 / page 7 / AR03.", "- No official score and no artifact-positive lift claim.", "", "## Summary", f"- target code present in OCR text: {summary['target_code_present_in_text']}", f"- AR codes in OCR text: `{summary['ar_codes_in_text']}`", f"- existing page-7 artifact count: {summary['existing_artifact_page_count']}", f"- extracted code/name pairs: {summary['extracted_pair_count']}", f"- extracted AR pairs: `{json.dumps(summary['ar_pairs'], sort_keys=True)}`", f"- target code extracted: {summary['target_code_extracted']}", f"- selector guard with extracted artifacts: `{summary['selector_guard_with_extracted_artifacts']}`", "", "## Recommended Next"]
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


def load_r063_row(path: Path, record_id: int) -> dict[str, Any]:
    for row in r053.read_jsonl(path):
        if int(row.get("record_id")) == record_id:
            return row
    raise ValueError(f"R063 comparison row missing record: {record_id}")


def load_page_text(extract_root: Path, doc_id: str, page: int) -> str:
    stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    path = extract_root / f"{stem}_{page}.txt"
    return path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""


def make_page_input(doc_id: str, page: int, text: str) -> dict[str, Any]:
    return {"doc_id": doc_id, "page_index": page, "page_text": text, "layout_blocks": [{"block_id": f"p{page:03d}_text_0000", "block_type": "text_block", "page_index": page, "bbox": None, "text": text, "char_start": 0, "char_end": len(text)}]}


def exact_code_present(text: str, code: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def nearby_snippets(text: str, code: str, window: int = 160) -> list[str]:
    snippets = []
    for pattern in [code, re.match(r"([A-Za-z]+)", code).group(1) if re.match(r"([A-Za-z]+)", code) else "", "Arkansas"]:
        if not pattern:
            continue
        for match in re.finditer(re.escape(pattern), text, flags=re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= 4:
                return snippets
    return snippets


if __name__ == "__main__":
    main()
