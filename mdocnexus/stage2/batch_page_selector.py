"""Deterministic small-batch page selection from stage2-augmented records."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .mdocagent_aligned_stage2 import build_page_source, get_valid_explicit_page_indices


SELECTION_REASONS = (
    "valid_explicit_page_with_image",
    "image_top_10_first_available",
    "retrieval_union_first_available",
)


def select_pages_for_small_batch(
    stage2_records: list[dict],
    max_pages: int,
    extract_root: str = "tmp/MMLongBench",
) -> list[dict]:
    """Select up to max_pages legal page candidates without using eval fields."""

    max_pages = int(max_pages)
    if max_pages < 1:
        return []

    selected: List[Dict[str, Any]] = []
    for record_index, record in enumerate(stage2_records):
        if len(selected) >= max_pages:
            break
        candidate = select_one_page_from_record(record, record_index, extract_root)
        if candidate is not None:
            selected.append(candidate)
    return selected


def select_one_page_from_record(
    record: Mapping[str, Any],
    record_index: int,
    extract_root: str = "tmp/MMLongBench",
) -> Dict[str, Any] | None:
    stage2 = record.get("stage2", {})
    if not isinstance(stage2, dict):
        return None
    if not stage2.get("preflight", {}).get("passed", False):
        return None

    route_pages = candidate_route_pages(stage2)
    image_top_pages = candidate_route_pages(stage2, required_route="image")
    valid_explicit_pages = [
        page_index for page_index in get_valid_explicit_page_indices(record, extract_root) if page_index in route_pages
    ]

    for page_index in valid_explicit_pages:
        source = build_page_source(str(record.get("doc_id")), extract_root, page_index)
        if page_source_is_eligible(source):
            return build_selected_page(record, record_index, page_index, source, "valid_explicit_page_with_image")

    for page_index in image_top_pages:
        source = build_page_source(str(record.get("doc_id")), extract_root, page_index)
        if page_source_is_eligible(source):
            return build_selected_page(record, record_index, page_index, source, "image_top_10_first_available")

    for page_index in route_pages:
        source = build_page_source(str(record.get("doc_id")), extract_root, page_index)
        if page_source_is_eligible(source):
            return build_selected_page(record, record_index, page_index, source, "retrieval_union_first_available")

    return None


def page_source_is_eligible(page_source: Mapping[str, Any] | None) -> bool:
    return bool(
        page_source
        and page_source.get("has_page_image")
        and page_source.get("page_image_path")
        and page_source.get("layout_block_ids")
    )


def candidate_route_pages(stage2: Mapping[str, Any], required_route: str | None = None) -> List[int]:
    pages: List[int] = []
    for route in stage2.get("candidate_page_routes", []) or []:
        if not isinstance(route, dict) or route.get("page_index") is None:
            continue
        routes = route.get("routes", [])
        if required_route is not None and required_route not in routes:
            continue
        pages.append(int(route["page_index"]))
    return pages


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
