"""Deterministic small-batch page selection from stage2-augmented records."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .stage2_sidecar_store import resolve_stage2_preflight


SELECTION_REASONS = (
    "valid_explicit_page_with_image",
    "image_top_10_first_available",
    "pages_to_compile_first_available",
)


def select_pages_for_small_batch(stage2_records: list[dict], max_pages: int) -> list[dict]:
    """Select up to max_pages legal page candidates without using eval fields."""

    max_pages = int(max_pages)
    if max_pages < 1:
        return []

    selected: List[Dict[str, Any]] = []
    for record_index, record in enumerate(stage2_records):
        if len(selected) >= max_pages:
            break
        candidate = select_one_page_from_record(record, record_index)
        if candidate is not None:
            selected.append(candidate)
    return selected


def select_one_page_from_record(record: Mapping[str, Any], record_index: int) -> Dict[str, Any] | None:
    stage2 = resolve_stage2_preflight(record)
    if not isinstance(stage2, dict):
        return None
    if not stage2.get("preflight", {}).get("passed", False):
        return None

    invalid_explicit_refs = stage2.get("explicit_page_validation", {}).get("invalid_explicit_page_references", [])
    if invalid_explicit_refs:
        return None

    page_sources_by_index = {
        int(source["page_index"]): source
        for source in stage2.get("page_sources", [])
        if isinstance(source, dict) and source.get("page_index") is not None
    }
    valid_explicit_pages = [
        int(page_index)
        for page_index in stage2.get("explicit_page_validation", {}).get("valid_explicit_page_indices", [])
    ]
    image_top_pages = [
        int(item["page_index"])
        for item in stage2.get("retrieval_pages", {}).get("image_top_10_question_unique", [])
        if isinstance(item, dict) and item.get("page_index") is not None
    ]
    pages_to_compile = [int(page_index) for page_index in stage2.get("pages_to_compile", [])]

    for page_index in valid_explicit_pages:
        source = page_sources_by_index.get(page_index)
        if page_source_is_eligible(source):
            return build_selected_page(record, record_index, page_index, source, "valid_explicit_page_with_image")

    for page_index in image_top_pages:
        source = page_sources_by_index.get(page_index)
        if page_source_is_eligible(source):
            return build_selected_page(record, record_index, page_index, source, "image_top_10_first_available")

    for page_index in pages_to_compile:
        source = page_sources_by_index.get(page_index)
        if page_source_is_eligible(source):
            return build_selected_page(record, record_index, page_index, source, "pages_to_compile_first_available")

    return None


def page_source_is_eligible(page_source: Mapping[str, Any] | None) -> bool:
    return bool(
        page_source
        and page_source.get("has_page_image")
        and page_source.get("page_image_path")
        and page_source.get("layout_block_ids")
    )


def build_selected_page(
    record: Mapping[str, Any],
    record_index: int,
    page_index: int,
    page_source: Mapping[str, Any],
    selection_reason: str,
) -> Dict[str, Any]:
    return {
        "record_index": int(record_index),
        "doc_id": record.get("doc_id"),
        "question": record.get("question"),
        "answer_format": record.get("answer_format"),
        "page_index": int(page_index),
        "page_number_one_based": int(page_index) + 1,
        "selection_reason": selection_reason,
        "page_image_path": page_source.get("page_image_path"),
        "page_text_path": page_source.get("page_text_path"),
        "layout_block_ids": list(page_source.get("layout_block_ids", [])),
        "stage2": record.get("stage2", {}),
    }
