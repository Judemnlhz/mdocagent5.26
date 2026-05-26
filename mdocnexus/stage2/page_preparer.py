"""Prepare Stage 2 page inputs from canonical records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .layout_parser import build_basic_layout_blocks
from .page_loader import load_page_content


def prepare_pages_for_compilation(
    canonical_record: Dict[str, Any],
    extract_path: str | Path,
) -> Dict[str, Any]:
    """Create page compilation inputs for all pages in pages_to_compile."""

    doc_id = canonical_record["document"]["doc_id"]
    pages_to_compile = canonical_record["compilation_plan"]["pages_to_compile"]
    candidate_pool = canonical_record.get("candidate_pool", {})
    explicit_pages = set(candidate_pool.get("explicit_constraint_pages", []))
    retrieval_pages = set(candidate_pool.get("retrieval_candidate_pages", []))

    pages: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for raw_page_index in pages_to_compile:
        page_index = int(raw_page_index)
        page_content = load_page_content(
            canonical_record=canonical_record,
            extract_path=extract_path,
            page_index=page_index,
        )
        layout_blocks = build_basic_layout_blocks(
            doc_id=doc_id,
            page_index=page_index,
            page_text=page_content["page_text"],
            has_page_image=page_content["has_page_image"],
        )

        pages.append(
            {
                "doc_id": doc_id,
                "page_index": page_index,
                "page_text": page_content["page_text"],
                "page_text_path": page_content["page_text_path"],
                "page_image_path": page_content["page_image_path"],
                "has_page_text": page_content["has_page_text"],
                "has_page_image": page_content["has_page_image"],
                "layout_blocks": layout_blocks,
            }
        )

        if not layout_blocks:
            errors.append(
                {
                    "error_type": "missing_source_anchors",
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "required_by": _build_required_by(page_index, explicit_pages, retrieval_pages),
                    "text_paths_checked": page_content["text_paths_checked"],
                    "image_paths_checked": page_content["image_paths_checked"],
                }
            )

    return {
        "doc_id": doc_id,
        "pages": pages,
        "errors": errors,
    }


def _build_required_by(
    page_index: int,
    explicit_pages: set[int],
    retrieval_pages: set[int],
) -> List[str]:
    required_by: List[str] = []
    if page_index in explicit_pages:
        required_by.append("explicit_page_reference")
    if page_index in retrieval_pages:
        required_by.append("retrieval_candidate_page")
    return required_by
