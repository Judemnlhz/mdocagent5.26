"""Candidate-page pool construction for Stage 2 compilation planning."""

from __future__ import annotations

from typing import Any, Dict, List


PAGE_RANGE_POLICY = "retrieval_union_plus_valid_explicit_page_constraints"


def build_candidate_pool(
    text_ranked_pages_unique: List[Dict[str, Any]],
    image_ranked_pages_unique: List[Dict[str, Any]],
    explicit_page_references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge retrieval pages with explicit page constraints for compilation."""

    retrieval_pages = sorted(
        {item["page_index"] for item in text_ranked_pages_unique}
        | {item["page_index"] for item in image_ranked_pages_unique}
    )
    explicit_pages_raw = sorted(
        {
            item["page_index_zero_based"]
            for item in explicit_page_references
            if item["page_index_zero_based"] >= 0
        }
    )
    explicit_pages_valid = list(explicit_pages_raw)
    required_pages = sorted(set(retrieval_pages) | set(explicit_pages_valid))
    missed_explicit_pages = [page for page in explicit_pages_valid if page not in retrieval_pages]

    return {
        "retrieval_candidate_pages": retrieval_pages,
        "explicit_constraint_pages_raw": explicit_pages_raw,
        "explicit_constraint_pages_valid": explicit_pages_valid,
        "explicit_constraint_pages_invalid": [],
        "explicit_constraint_pages": explicit_pages_valid,
        "required_pages_for_compilation": required_pages,
        "retrieval_missed_explicit_pages": missed_explicit_pages,
        "candidate_pool_policy": PAGE_RANGE_POLICY,
    }
