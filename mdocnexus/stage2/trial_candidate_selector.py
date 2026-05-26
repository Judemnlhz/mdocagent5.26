"""Objective single-page trial candidate selection for Stage 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .mdocagent_compat import read_json_or_jsonl_records
from .normalize_record import normalize_record
from .page_preparer import prepare_pages_for_compilation
from .page_range_validation import (
    OUT_OF_RANGE_ERROR,
    apply_explicit_page_range_validation_to_canonical_record,
    infer_document_page_count,
)


SELECTION_REASONS = [
    "valid_explicit_page_with_image",
    "image_top1_with_image",
    "retrieval_union_first_image",
]


def select_single_page_trial_candidate(
    sample_path: str | Path,
    dataset_name: str,
    extract_root: str | Path,
    config_path: str | Path | None = None,
    max_records: int | None = None,
) -> Dict[str, Any]:
    """Select one legal single-page real-API smoke-test candidate.

    The selector is deterministic and uses only question constraints, retrieval
    rankings, inferred page counts, and extracted page sources. It does not use
    gold answers, evidence pages, baseline outputs, or any real provider call.
    """

    _ = dataset_name
    _ = config_path
    records = read_json_or_jsonl_records(sample_path)
    limited_records = records[: int(max_records)] if max_records is not None else records

    stats = {
        "scanned_records": 0,
        "num_out_of_range_explicit_pages": 0,
        "num_missing_page_sources": 0,
    }
    candidates: List[Dict[str, Any]] = []

    for record_index, source_record in enumerate(limited_records):
        stats["scanned_records"] += 1
        candidates.extend(
            _collect_record_candidates(
                source_record=source_record,
                record_index=record_index,
                extract_root=extract_root,
                stats=stats,
            )
        )

    selected = _select_best_candidate(candidates)
    return {
        "selection_passed": selected is not None,
        "blocking_reasons": [] if selected is not None else ["no_valid_single_page_trial_candidate"],
        "selected": selected,
        "selection_policy": _selection_policy(),
        "extract_root": str(extract_root),
        **stats,
    }


def _collect_record_candidates(
    source_record: Mapping[str, Any],
    record_index: int,
    extract_root: str | Path,
    stats: Dict[str, int],
) -> List[Dict[str, Any]]:
    normalized = normalize_record(strip_eval_only_fields(source_record))
    canonical_record = normalized["canonical_record"]
    doc_id = canonical_record["document"]["doc_id"]
    page_count_info = infer_document_page_count(
        doc_id=doc_id,
        pdf_root=None,
        extract_root=extract_root,
    )
    validation = apply_explicit_page_range_validation_to_canonical_record(
        canonical_record,
        page_count_info,
    )
    stats["num_out_of_range_explicit_pages"] += sum(
        1
        for ref in validation.get("invalid_explicit_page_references", [])
        if ref.get("error_type") == OUT_OF_RANGE_ERROR
    )

    prepared_result = prepare_pages_for_compilation(canonical_record, extract_root)
    pages_by_index = {
        int(page["page_index"]): page
        for page in prepared_result.get("pages", [])
    }
    pages_to_compile = [
        int(page_index)
        for page_index in canonical_record.get("compilation_plan", {}).get("pages_to_compile", [])
    ]
    compile_order = {page_index: order for order, page_index in enumerate(pages_to_compile)}
    valid_explicit_pages = [
        int(page_index)
        for page_index in canonical_record.get("candidate_pool", {}).get("explicit_constraint_pages_valid", [])
    ]
    invalid_explicit_pages = {
        int(ref["page_index_zero_based"])
        for ref in validation.get("invalid_explicit_page_references", [])
        if ref.get("page_index_zero_based") is not None
    }

    candidates: List[Dict[str, Any]] = []
    for page_index in pages_to_compile:
        page = pages_by_index.get(page_index)
        if page is None or not _has_any_page_source(page):
            stats["num_missing_page_sources"] += 1
            continue
        if page_index in invalid_explicit_pages:
            continue
        if not _page_index_in_range(page_index, page_count_info):
            continue
        if not _page_passes_preflight(page, page_index):
            continue

        reason = _selection_reason(
            canonical_record=canonical_record,
            page_index=page_index,
            valid_explicit_pages=valid_explicit_pages,
            has_explicit_page_references=_has_explicit_page_references(canonical_record),
        )
        if reason is None:
            continue
        candidates.append(
            _build_candidate(
                normalized=normalized,
                canonical_record=canonical_record,
                page=page,
                record_index=record_index,
                page_index=page_index,
                selection_reason=reason,
                compile_order=compile_order.get(page_index, len(compile_order)),
            )
        )

    return candidates


def strip_eval_only_fields(source_record: Mapping[str, Any]) -> Dict[str, Any]:
    blocked_keys = {
        "answer",
        "evidence_pages",
        "evidence_sources",
        "ans_mmlb-MDocAgent",
        "binary_correctness",
        "gold_annotation",
        "baseline_outputs",
    }
    sanitized = {
        key: value
        for key, value in dict(source_record).items()
        if key not in blocked_keys
    }
    canonical = sanitized.get("canonical_record")
    if isinstance(canonical, dict):
        clean_canonical = dict(canonical)
        clean_canonical.pop("gold_annotation", None)
        clean_canonical.pop("baseline_outputs", None)
        clean_canonical.pop("source_record", None)
        sanitized["canonical_record"] = clean_canonical
    return sanitized


def _selection_reason(
    canonical_record: Dict[str, Any],
    page_index: int,
    valid_explicit_pages: List[int],
    has_explicit_page_references: bool,
) -> Optional[str]:
    if page_index in valid_explicit_pages:
        return "valid_explicit_page_with_image"

    image_pages = _ranked_page_indices(canonical_record, "image")
    if not has_explicit_page_references and image_pages and page_index == image_pages[0]:
        return "image_top1_with_image"

    return "retrieval_union_first_image"


def _has_explicit_page_references(canonical_record: Dict[str, Any]) -> bool:
    return bool(
        canonical_record.get("question_constraints", {}).get("explicit_page_references", [])
    )


def _build_candidate(
    normalized: Dict[str, Any],
    canonical_record: Dict[str, Any],
    page: Dict[str, Any],
    record_index: int,
    page_index: int,
    selection_reason: str,
    compile_order: int,
) -> Dict[str, Any]:
    layout_block_ids = [
        str(block["block_id"])
        for block in page.get("layout_blocks", [])
        if block.get("block_id")
    ]
    return {
        "record_index": record_index,
        "record_id": normalized.get("record_id"),
        "doc_id": canonical_record["document"]["doc_id"],
        "question": canonical_record["question"]["text"],
        "page_index": page_index,
        "page_number_one_based": page_index + 1,
        "selection_reason": selection_reason,
        "page_image_path": page.get("page_image_path"),
        "page_text_path": page.get("page_text_path"),
        "layout_block_ids": layout_block_ids,
        "_priority_rank": SELECTION_REASONS.index(selection_reason),
        "_compile_order": compile_order,
    }


def _select_best_candidate(candidates: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["_priority_rank"]),
            int(item["record_index"]),
            int(item["_compile_order"]),
        ),
    )
    if not ordered:
        return None
    selected = dict(ordered[0])
    selected.pop("_priority_rank", None)
    selected.pop("_compile_order", None)
    return selected


def _ranked_page_indices(canonical_record: Dict[str, Any], retrieval_key: str) -> List[int]:
    return [
        int(item["page_index"])
        for item in canonical_record.get("retrieval", {}).get(retrieval_key, {}).get("ranked_pages_unique", [])
    ]


def _has_any_page_source(page: Dict[str, Any]) -> bool:
    return bool(page.get("has_page_image") or page.get("has_page_text"))


def _page_passes_preflight(page: Dict[str, Any], page_index: int) -> bool:
    if not page.get("has_page_image"):
        return False
    layout_block_ids = [block.get("block_id") for block in page.get("layout_blocks", [])]
    expected_full_page_id = f"p{page_index:03d}_full_page_image"
    return bool(layout_block_ids) and expected_full_page_id in layout_block_ids


def _page_index_in_range(page_index: int, page_count_info: Dict[str, Any]) -> bool:
    page_count = page_count_info.get("page_count")
    if page_count is not None:
        return 0 <= page_index < int(page_count)
    available_indices = {int(index) for index in page_count_info.get("available_page_indices", [])}
    return page_index in available_indices


def _selection_policy() -> Dict[str, Any]:
    return {
        "priority_order": list(SELECTION_REASONS),
        "uses_gold_answer": False,
        "uses_gold_evidence_pages": False,
        "uses_baseline_correctness": False,
    }
