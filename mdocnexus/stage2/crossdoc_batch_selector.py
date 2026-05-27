"""Cross-document controlled page selection for Stage 2 batch compilation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .mdocagent_aligned_stage2 import build_page_source, get_valid_explicit_page_indices


SELECTION_REASONS = (
    "valid_explicit_page_with_image",
    "image_top_10_first_available",
    "retrieval_union_first_available",
)


def select_crossdoc_pages_for_batch(
    stage2_records: list[dict],
    max_docs: int = 5,
    max_pages_per_doc: int = 2,
    max_pages: int = 10,
    extract_root: str = "tmp/MMLongBench",
) -> list[dict]:
    """Select legal cross-document page candidates without using eval fields."""

    max_docs = max(0, int(max_docs))
    max_pages_per_doc = max(0, int(max_pages_per_doc))
    max_pages = max(0, int(max_pages))
    if not max_docs or not max_pages_per_doc or not max_pages:
        return []

    selected: List[Dict[str, Any]] = []
    doc_counts: Dict[str, int] = {}
    selected_pages_by_doc: Dict[str, set[int]] = {}

    for record_index, record in enumerate(stage2_records):
        if len(selected) >= max_pages:
            break
        doc_id = record.get("doc_id")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        if doc_id not in doc_counts and len(doc_counts) >= max_docs:
            continue
        if doc_counts.get(doc_id, 0) >= max_pages_per_doc:
            continue

        for page_index, reason in _prioritized_page_candidates(record, extract_root):
            if len(selected) >= max_pages:
                break
            if doc_id not in doc_counts and len(doc_counts) >= max_docs:
                break
            if doc_counts.get(doc_id, 0) >= max_pages_per_doc:
                break
            if page_index in selected_pages_by_doc.get(doc_id, set()):
                continue

            source = build_page_source(doc_id, extract_root, page_index)
            if not _page_source_is_eligible(source):
                continue

            selected.append(_build_selected_page(record, record_index, page_index, source, reason))
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            selected_pages_by_doc.setdefault(doc_id, set()).add(page_index)

    return selected


def _prioritized_page_candidates(record: Mapping[str, Any], extract_root: str) -> Iterable[tuple[int, str]]:
    stage2 = record.get("stage2", {})
    if not isinstance(stage2, dict):
        return []
    if not stage2.get("preflight", {}).get("passed", False):
        return []
    route_pages = _candidate_route_pages(stage2)
    valid_explicit_pages = [
        page_index for page_index in get_valid_explicit_page_indices(record, extract_root) if page_index in route_pages
    ]

    seen: set[int] = set()
    prioritized_groups = [
        (valid_explicit_pages, "valid_explicit_page_with_image"),
        (_candidate_route_pages(stage2, required_route="image"), "image_top_10_first_available"),
        (route_pages, "retrieval_union_first_available"),
    ]
    candidates: List[tuple[int, str]] = []
    for page_indices, reason in prioritized_groups:
        for page_index in page_indices:
            if page_index in seen:
                continue
            seen.add(page_index)
            candidates.append((page_index, reason))
    return candidates


def _candidate_route_pages(stage2: Mapping[str, Any], required_route: str | None = None) -> List[int]:
    pages: List[int] = []
    for route in stage2.get("candidate_page_routes", []) or []:
        if not isinstance(route, dict) or route.get("page_index") is None:
            continue
        routes = route.get("routes", [])
        if required_route is not None and required_route not in routes:
            continue
        pages.append(int(route["page_index"]))
    return pages


def _page_source_is_eligible(page_source: Mapping[str, Any] | None) -> bool:
    return bool(
        page_source
        and page_source.get("has_page_image") is True
        and page_source.get("page_image_path")
        and page_source.get("layout_block_ids")
    )


def _build_selected_page(
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
        "page_index": int(page_index),
        "page_number_one_based": int(page_index) + 1,
        "selection_reason": selection_reason,
        "page_image_path": page_source.get("page_image_path"),
        "page_text_path": page_source.get("page_text_path"),
        "layout_block_ids": list(page_source.get("layout_block_ids", [])),
        "stage2": record.get("stage2", {}),
    }
