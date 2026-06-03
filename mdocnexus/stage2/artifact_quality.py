"""Artifact quality taxonomy for Stage 2 structured outputs.

The checks in this module intentionally use only Stage 2 artifact content,
normalized content, locators, and anchors. They do not inspect questions,
answers, evidence pages, or gold fields.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


NUMERIC_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|percentage|bps|bp|million|billion|thousand|m|bn)?", re.I)
TABLE_TITLE_ONLY_RE = re.compile(r"\b(table|summary|performance|financial|segment|brochure|caption|title)\b", re.I)
FINANCIAL_TABLE_RE = re.compile(
    r"\b(financial|performance|revenue|sales|income|margin|segment|operating|fiscal|year|quarter|percent|percentage|metric|amount)\b",
    re.I,
)
WEAK_LOCATOR_KINDS = {"full_page_anchor"}
STRONG_LOCATOR_KINDS = {"bbox", "table_cell", "figure_region", "caption_block", "text_offset", "source_block"}


def classify_artifact_quality(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic Stage 2-only quality labels for one artifact."""

    artifact_type = str(artifact.get("artifact_type") or "").lower()
    content = _compact_text(artifact.get("content"))
    normalized = artifact.get("normalized_content")
    if not isinstance(normalized, Mapping):
        normalized = {}

    has_numeric_value = _has_numeric_value(content, normalized)
    has_locator = _has_strong_locator_signal(artifact)
    weak_locator = _has_full_page_only_locator(artifact) or not has_locator
    caption_or_title_only = _is_caption_or_table_title_only(artifact_type, content, normalized)
    broad_table = _is_broad_table_artifact(artifact_type, content, normalized, has_numeric_value, caption_or_title_only)
    numeric_fact_incomplete = artifact_type == "numeric_fact" and not _numeric_fact_has_required_fields(content, normalized)
    table_cell_incomplete = artifact_type == "table_cell" and not _table_cell_has_required_fields(content, normalized)
    table_cell_atomic = artifact_type == "table_cell" and not table_cell_incomplete
    numeric_fact_atomic = artifact_type == "numeric_fact" and not numeric_fact_incomplete

    labels: list[str] = []
    if (table_cell_atomic or numeric_fact_atomic) and has_locator:
        labels.append("atomic_numeric_ok")
    if broad_table:
        labels.append("broad_table_only")
    if artifact_type == "numeric_fact" and numeric_fact_incomplete:
        labels.append("missing_numeric_fact")
    if table_cell_incomplete and _has_numeric_value(content, normalized):
        labels.append("missing_numeric_fact")
    if weak_locator:
        labels.append("weak_locator")
    if caption_or_title_only:
        labels.append("caption_or_table_title_only")
    if broad_table or numeric_fact_incomplete or table_cell_incomplete or caption_or_title_only or weak_locator:
        labels.append("schema_valid_but_semantically_weak")

    return {
        "labels": labels,
        "atomic_numeric_ok": "atomic_numeric_ok" in labels,
        "broad_table_only": broad_table,
        "missing_numeric_fact": "missing_numeric_fact" in labels,
        "weak_locator": weak_locator,
        "caption_or_table_title_only": caption_or_title_only,
        "schema_valid_but_semantically_weak": "schema_valid_but_semantically_weak" in labels,
        "has_numeric_value": has_numeric_value,
        "has_strong_locator": has_locator,
        "numeric_fact_incomplete": numeric_fact_incomplete,
        "table_cell_incomplete": table_cell_incomplete,
        "discard_from_atomic_strong": bool(broad_table or numeric_fact_incomplete or table_cell_incomplete or caption_or_title_only or weak_locator),
        "discard_from_stage2_store": bool(broad_table or numeric_fact_incomplete),
    }


def is_atomic_strong_eligible(artifact: Mapping[str, Any], eligibility_reason: str = "eligible") -> bool:
    """Return whether an artifact is both locator-eligible and atomic evidence."""

    quality = classify_artifact_quality(artifact)
    return eligibility_reason == "eligible" and bool(quality["atomic_numeric_ok"]) and not bool(quality["schema_valid_but_semantically_weak"])


def quality_discard_reason(artifact: Mapping[str, Any]) -> str | None:
    """Return a deterministic discard reason for semantically too-weak outputs."""

    quality = classify_artifact_quality(artifact)
    if quality["broad_table_only"]:
        return "broad_table_only_semantically_weak"
    if quality["numeric_fact_incomplete"]:
        return "numeric_fact_missing_value_metric_context"
    return None


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _has_numeric_value(content: str, normalized: Mapping[str, Any]) -> bool:
    for key in ("value", "value_text", "metric_value", "amount", "percentage", "percent"):
        if _compact_text(normalized.get(key)):
            return True
    return bool(NUMERIC_RE.search(content))


def _numeric_fact_has_required_fields(content: str, normalized: Mapping[str, Any]) -> bool:
    value = _compact_text(normalized.get("value_text") or normalized.get("value") or normalized.get("metric_value"))
    metric = _compact_text(normalized.get("metric_name") or normalized.get("metric") or normalized.get("row_label") or normalized.get("row_header"))
    context = _compact_text(
        normalized.get("context")
        or normalized.get("group")
        or normalized.get("column_label")
        or normalized.get("column_header")
        or normalized.get("date_or_period")
        or normalized.get("source_text")
    )
    return bool(value and metric and context and (_has_numeric_value(content, normalized) or NUMERIC_RE.search(value)))


def _table_cell_has_required_fields(content: str, normalized: Mapping[str, Any]) -> bool:
    value = _compact_text(normalized.get("value_text") or normalized.get("value"))
    row = _compact_text(normalized.get("row_label") or normalized.get("row_header"))
    column = _compact_text(normalized.get("column_label") or normalized.get("column_header"))
    has_position = normalized.get("row_index") is not None and normalized.get("column_index") is not None
    return bool(value and row and column and has_position and (_has_numeric_value(content, normalized) or NUMERIC_RE.search(value)))


def _is_caption_or_table_title_only(artifact_type: str, content: str, normalized: Mapping[str, Any]) -> bool:
    if artifact_type not in {"table", "caption"}:
        return False
    if _has_numeric_value(content, normalized):
        return False
    token_count = len(content.split())
    if token_count <= 8:
        return True
    normalized_values = " ".join(_compact_text(value) for value in normalized.values())
    return token_count <= 14 and bool(TABLE_TITLE_ONLY_RE.search(content + " " + normalized_values))


def _is_broad_table_artifact(artifact_type: str, content: str, normalized: Mapping[str, Any], has_numeric_value: bool, title_only: bool) -> bool:
    if artifact_type != "table":
        return False
    normalized_values = " ".join(_compact_text(value) for value in normalized.values())
    text = content + " " + normalized_values
    if title_only:
        return True
    if not has_numeric_value and FINANCIAL_TABLE_RE.search(text):
        return True
    return False


def _artifact_locator_kinds(artifact: Mapping[str, Any]) -> set[str]:
    kinds: set[str] = set()
    locators = artifact.get("locators")
    if isinstance(locators, list):
        for locator in locators:
            if not isinstance(locator, Mapping):
                continue
            kind = str(locator.get("locator_kind") or locator.get("kind") or "").lower()
            if kind:
                kinds.add(kind)
    return kinds


def _artifact_anchor_types(artifact: Mapping[str, Any]) -> set[str]:
    anchor_types: set[str] = set()
    anchors = artifact.get("source_anchors")
    if isinstance(anchors, list):
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                continue
            anchor_type = str(anchor.get("anchor_type") or "").lower()
            if anchor_type:
                anchor_types.add(anchor_type)
    return anchor_types


def _has_full_page_only_locator(artifact: Mapping[str, Any]) -> bool:
    locator_kinds = _artifact_locator_kinds(artifact)
    anchor_types = _artifact_anchor_types(artifact)
    has_full_page = "full_page_anchor" in locator_kinds or "full_page_image" in anchor_types
    has_strong_non_full_page = bool((locator_kinds - WEAK_LOCATOR_KINDS) & STRONG_LOCATOR_KINDS)
    return bool(has_full_page and not has_strong_non_full_page)


def _has_strong_locator_signal(artifact: Mapping[str, Any]) -> bool:
    locator_kinds = _artifact_locator_kinds(artifact)
    if (locator_kinds - WEAK_LOCATOR_KINDS) & STRONG_LOCATOR_KINDS:
        return True
    anchors = artifact.get("source_anchors")
    if isinstance(anchors, list):
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                continue
            if anchor.get("bbox") not in (None, "", [], {}):
                return True
            if str(anchor.get("anchor_type") or "") in {"table_cell", "figure_region"}:
                return True
    return False
