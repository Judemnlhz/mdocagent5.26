"""Retrieval result normalization helpers for Stage 2."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Sequence


def parse_sequence_field(value: Any, field_name: str) -> List[Any]:
    """Parse a source field that may be a list or a stringified Python list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    raise ValueError(f"Cannot parse list field {field_name!r}: {value!r}")


def deduplicate_ranked_pages(
    pages: Sequence[Any],
    scores: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Deduplicate ranked page indices while retaining first rank and duplicates."""

    seen: Dict[int, Dict[str, Any]] = {}

    for index, raw_page in enumerate(pages):
        page_index = _coerce_page_index(raw_page)
        rank = index + 1
        score = _coerce_score(scores[index]) if index < len(scores) else None

        if page_index not in seen:
            seen[page_index] = {
                "page_index": page_index,
                "rank": rank,
                "score": score,
                "duplicate_ranks": [rank],
            }
        else:
            seen[page_index]["duplicate_ranks"].append(rank)

    ranked_pages_unique: List[Dict[str, Any]] = []
    for item in seen.values():
        if len(item["duplicate_ranks"]) == 1:
            item.pop("duplicate_ranks")
        ranked_pages_unique.append(item)

    return sorted(ranked_pages_unique, key=lambda item: item["rank"])


def _coerce_page_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean page index: {value!r}")
    page_index = int(value)
    if page_index < 0:
        raise ValueError(f"Invalid negative page index: {value!r}")
    return page_index


def _coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
