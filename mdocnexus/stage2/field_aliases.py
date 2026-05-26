"""Input field aliases for Stage 2 record normalization.

This module is the only Stage 2 core module that should know about legacy
MDocAgent field names containing hyphens. Downstream canonical records must use
snake_case field names only.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence


SOURCE_FIELD_ALIASES: Mapping[str, Sequence[str]] = {
    "doc_id": ("doc_id",),
    "doc_type": ("doc_type",),
    "question_text": ("question",),
    "answer_format": ("answer_format",),
    "answer": ("answer",),
    "evidence_pages": ("evidence_pages",),
    "evidence_sources": ("evidence_sources",),
    "mdocagent_retrieval_query": ("retrieval-query",),
    "mdocagent_retrieval_key": ("retrieval-key",),
    "qwen_retrieval_query": ("qwen_retrieval-query",),
    "qwen_retrieval_key": ("qwen_retrieval-key",),
    "text_index_path": ("text-index-path-question",),
    "text_ranked_pages": ("text-top-10-question",),
    "text_ranked_scores": ("text-top-10-question_score",),
    "image_ranked_pages": ("image-top-10-question",),
    "image_ranked_scores": ("image-top-10-question_score",),
    "mdocagent_answer": ("ans_mmlb-MDocAgent",),
    "binary_correctness": ("binary_correctness",),
}


PATTERN_FIELD_ALIASES: Mapping[str, Sequence[re.Pattern[str]]] = {
    "text_ranked_pages": (re.compile(r"^text-top-\d+-question$"),),
    "text_ranked_scores": (re.compile(r"^text-top-\d+-question_score$"),),
    "image_ranked_pages": (re.compile(r"^image-top-\d+-question$"),),
    "image_ranked_scores": (re.compile(r"^image-top-\d+-question_score$"),),
    "text_index_path": (re.compile(r"^text-index-path-question$"),),
}


COMPILER_EXCLUDED_FIELDS: Sequence[str] = (
    "source_record",
    "gold_annotation",
    "baseline_outputs",
)


def get_source_value(
    source_record: Mapping[str, Any],
    field_key: str,
    default: Any = None,
) -> Any:
    """Return a source value using approved aliases for a normalized field key."""

    for source_field_name in SOURCE_FIELD_ALIASES.get(field_key, ()):
        if source_field_name in source_record:
            return source_record[source_field_name]

    pattern_match = find_pattern_field_name(source_record, field_key)
    if pattern_match is not None:
        return source_record[pattern_match]

    return default


def require_source_value(source_record: Mapping[str, Any], field_key: str) -> Any:
    """Return a required source value or raise a field-specific ValueError."""

    value = get_source_value(source_record, field_key, default=None)
    if value is None:
        raise ValueError(f"Missing required source field for {field_key!r}")
    return value


def find_pattern_field_name(
    source_record: Mapping[str, Any],
    field_key: str,
) -> Optional[str]:
    """Find a dynamically named retrieval field such as top-4 or top-20."""

    patterns = PATTERN_FIELD_ALIASES.get(field_key, ())
    if not patterns:
        return None
    matches = [
        source_field_name
        for source_field_name in source_record.keys()
        if any(pattern.match(source_field_name) for pattern in patterns)
    ]
    if not matches:
        return None

    def _top_k_sort_key(source_field_name: str) -> tuple[int, str]:
        match = re.search(r"-top-(\d+)-", source_field_name)
        top_k = int(match.group(1)) if match else -1
        return (-top_k, source_field_name)

    return sorted(matches, key=_top_k_sort_key)[0]
