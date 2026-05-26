"""Preflight one Stage 2 single-page trial without calling a real API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.mdocagent_compat import (
    build_api_run_config_from_mdocagent_yaml,
    find_record_by_id_or_doc_question,
    read_json_or_jsonl_records,
    summarize_mdocagent_model_config,
)
from mdocnexus.stage2.normalize_record import normalize_record
from mdocnexus.stage2.page_preparer import prepare_pages_for_compilation
from mdocnexus.stage2.page_range_validation import (
    OUT_OF_RANGE_ERROR,
    PAGE_COUNT_UNKNOWN_ERROR,
    apply_explicit_page_range_validation_to_canonical_record,
    infer_document_page_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one Stage 2 page trial without API calls.")
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--question-substring", default=None)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--pdf-root", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-page-index", required=True, type=int)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env-var", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def build_preflight_report(
    sample_path: str | Path,
    extract_root: str | Path,
    config_path: str | Path,
    target_page_index: int,
    record_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    question_substring: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    pdf_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Validate Stage 2 inputs for a single page without model calls."""

    target_page_index = int(target_page_index)
    records = read_json_or_jsonl_records(sample_path)
    record = find_record_by_id_or_doc_question(records, record_id, doc_id, question_substring)
    normalized = record if "canonical_record" in record else normalize_record(record)
    canonical_record = normalized["canonical_record"]
    page_count_info = infer_document_page_count(
        doc_id=canonical_record["document"]["doc_id"],
        pdf_root=pdf_root,
        extract_root=extract_root,
    )
    explicit_validation = apply_explicit_page_range_validation_to_canonical_record(
        canonical_record,
        page_count_info,
    )

    prepared_result = prepare_pages_for_compilation(canonical_record, extract_root)
    target_pages = [
        page for page in prepared_result["pages"] if int(page["page_index"]) == target_page_index
    ]
    target_page = target_pages[0] if target_pages else None
    api_config = build_api_run_config_from_mdocagent_yaml(config_path, overrides=overrides or {})

    invalid_target_refs = [
        ref
        for ref in explicit_validation["invalid_explicit_page_references"]
        if int(ref.get("page_index_zero_based", -1)) == target_page_index
    ]
    blocking_reasons = []
    if any(ref.get("error_type") == OUT_OF_RANGE_ERROR for ref in invalid_target_refs):
        blocking_reasons.append(OUT_OF_RANGE_ERROR)
    if any(ref.get("error_type") == PAGE_COUNT_UNKNOWN_ERROR for ref in invalid_target_refs):
        blocking_reasons.append(PAGE_COUNT_UNKNOWN_ERROR)

    if not invalid_target_refs:
        if target_page is None:
            blocking_reasons.append("target_page_not_prepared")
        elif not target_page.get("layout_blocks"):
            blocking_reasons.append("missing_source_anchors")
        if target_page and not _has_expected_full_page_block(target_page, target_page_index):
            blocking_reasons.append("missing_full_page_image_anchor")

    page_text_path = target_page.get("page_text_path") if target_page else None
    page_image_path = target_page.get("page_image_path") if target_page else None
    layout_block_ids = [block["block_id"] for block in target_page.get("layout_blocks", [])] if target_page else []
    should_call_api = not blocking_reasons
    return {
        "preflight_passed": not blocking_reasons,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "record_id": normalized.get("record_id"),
        "doc_id": canonical_record["document"]["doc_id"],
        "target_page_index": target_page_index,
        "target_page_number_one_based": target_page_index + 1,
        "extract_root": str(extract_root),
        "page_text_path": page_text_path,
        "page_image_path": page_image_path,
        "has_page_text": bool(target_page and target_page.get("has_page_text")),
        "has_page_image": bool(target_page and target_page.get("has_page_image")),
        "layout_block_ids": layout_block_ids,
        "page_count": page_count_info.get("page_count"),
        "page_count_info": page_count_info,
        "invalid_explicit_page_references": explicit_validation["invalid_explicit_page_references"],
        "valid_explicit_page_indices": explicit_validation["valid_explicit_page_indices"],
        "candidate_pool": canonical_record.get("candidate_pool", {}),
        "pages_to_compile": canonical_record.get("compilation_plan", {}).get("pages_to_compile", []),
        "model_config": summarize_mdocagent_model_config(config_path, api_config),
        "will_call_api": False,
        "should_call_api": should_call_api,
        "should_generate_artifact": should_call_api,
    }


def write_preflight_report(report: Dict[str, Any], output_report: str | Path) -> None:
    path = Path(output_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"enable_real_api": False, "timeout_seconds": args.timeout_seconds}
    for arg_name, field_name in (
        ("provider", "provider"),
        ("model_name", "model_name"),
        ("api_base_url", "api_base_url"),
        ("api_key_env_var", "api_key_env_var"),
        ("temperature", "temperature"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    return overrides


def _has_expected_full_page_block(target_page: Dict[str, Any], target_page_index: int) -> bool:
    expected_block_id = f"p{target_page_index:03d}_full_page_image"
    return any(block.get("block_id") == expected_block_id for block in target_page.get("layout_blocks", []))


def main() -> None:
    args = parse_args()
    report = build_preflight_report(
        sample_path=args.sample_path,
        extract_root=args.extract_root,
        config_path=args.config,
        target_page_index=args.target_page_index,
        record_id=args.record_id,
        doc_id=args.doc_id,
        question_substring=args.question_substring,
        overrides=build_overrides_from_args(args),
        pdf_root=args.pdf_root,
    )
    write_preflight_report(report, args.output_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
