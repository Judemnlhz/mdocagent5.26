"""Cross-document controlled page selection for Stage 2 batch compilation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .stage2_sidecar_store import resolve_stage2_preflight


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

        for page_index, reason in _prioritized_page_candidates(record):
            if len(selected) >= max_pages:
                break
            if doc_id not in doc_counts and len(doc_counts) >= max_docs:
                break
            if doc_counts.get(doc_id, 0) >= max_pages_per_doc:
                break
            if page_index in selected_pages_by_doc.get(doc_id, set()):
                continue

            source = _page_sources_by_index(record).get(page_index)
            if not _page_source_is_eligible(source):
                continue
            if not _page_in_range(record, page_index):
                continue
            if _is_invalid_explicit_reference(record, page_index):
                continue

            selected.append(_build_selected_page(record, record_index, page_index, source, reason))
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            selected_pages_by_doc.setdefault(doc_id, set()).add(page_index)

    return selected


def _prioritized_page_candidates(record: Mapping[str, Any]) -> Iterable[tuple[int, str]]:
    stage2 = resolve_stage2_preflight(record)
    if not isinstance(stage2, dict):
        return []
    if not stage2.get("preflight", {}).get("passed", False):
        return []

    seen: set[int] = set()
    prioritized_groups = [
        (_valid_explicit_pages(stage2), "valid_explicit_page_with_image"),
        (_image_top_pages(stage2), "image_top_10_first_available"),
        (_retrieval_union_pages(stage2), "retrieval_union_first_available"),
    ]
    candidates: List[tuple[int, str]] = []
    for page_indices, reason in prioritized_groups:
        for page_index in page_indices:
            if page_index in seen:
                continue
            seen.add(page_index)
            candidates.append((page_index, reason))
    return candidates


def _valid_explicit_pages(stage2: Mapping[str, Any]) -> List[int]:
    return _coerce_page_indices(
        stage2.get("explicit_page_validation", {}).get("valid_explicit_page_indices", [])
    )


def _image_top_pages(stage2: Mapping[str, Any]) -> List[int]:
    pages = []
    for item in stage2.get("retrieval_pages", {}).get("image_top_10_question_unique", []):
        if isinstance(item, dict) and item.get("page_index") is not None:
            pages.append(item.get("page_index"))
    return _coerce_page_indices(pages)


def _retrieval_union_pages(stage2: Mapping[str, Any]) -> List[int]:
    retrieval_pages = stage2.get("retrieval_pages", {})
    candidates = []
    candidates.extend(retrieval_pages.get("retrieval_candidate_pages", []) or [])
    candidates.extend(stage2.get("pages_to_compile", []) or [])
    return _coerce_page_indices(candidates)


def _coerce_page_indices(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    for value in values:
        try:
            page_index = int(value)
        except (TypeError, ValueError):
            continue
        if page_index >= 0:
            result.append(page_index)
    return result


def _page_sources_by_index(record: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    stage2 = resolve_stage2_preflight(record)
    if not isinstance(stage2, dict):
        return {}
    sources: Dict[int, Mapping[str, Any]] = {}
    for source in stage2.get("page_sources", []) or []:
        if not isinstance(source, dict) or source.get("page_index") is None:
            continue
        try:
            sources[int(source["page_index"])] = source
        except (TypeError, ValueError):
            continue
    return sources


def _page_source_is_eligible(page_source: Mapping[str, Any] | None) -> bool:
    return bool(
        page_source
        and page_source.get("has_page_image") is True
        and page_source.get("page_image_path")
        and page_source.get("layout_block_ids")
    )


def _page_in_range(record: Mapping[str, Any], page_index: int) -> bool:
    stage2 = resolve_stage2_preflight(record)
    page_count = stage2.get("page_count") if isinstance(stage2, dict) else None
    if isinstance(page_count, dict):
        available_indices = page_count.get("available_page_indices")
        if isinstance(available_indices, list):
            try:
                return int(page_index) in {int(index) for index in available_indices}
            except (TypeError, ValueError):
                return False
        page_count = page_count.get("value")
    try:
        if page_count is not None:
            return 0 <= int(page_index) < int(page_count)
    except (TypeError, ValueError):
        return False
    return int(page_index) >= 0


def _is_invalid_explicit_reference(record: Mapping[str, Any], page_index: int) -> bool:
    stage2 = resolve_stage2_preflight(record)
    if not isinstance(stage2, dict):
        return False
    invalid_refs = stage2.get("explicit_page_validation", {}).get("invalid_explicit_page_references", []) or []
    for ref in invalid_refs:
        if not isinstance(ref, dict):
            continue
        ref_index = ref.get("page_index_zero_based", ref.get("page_index"))
        try:
            if int(ref_index) == int(page_index):
                return True
        except (TypeError, ValueError):
            continue
    return False


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
        "stage2": resolve_stage2_preflight(record),
    }
