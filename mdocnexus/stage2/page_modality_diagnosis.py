"""Question and preflight based modality guidance for Stage 2 compilation."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping


CHART_TERMS = ("chart", "graph", "plot")
TABLE_TERMS = ("table", "tabular", "spreadsheet")
FIGURE_TERMS = ("figure", "image", "picture", "photo", "diagram", "map")
VISUAL_TERMS = CHART_TERMS + TABLE_TERMS + FIGURE_TERMS + (
    "handwritten",
    "handwriting",
    "color",
    "colour",
    "visual",
)
NUMERIC_PATTERNS = (
    r"\bhow\s+much\b",
    r"\bhow\s+many\b",
    r"\bpercent\b",
    r"\bpercentage\b",
    r"\bnumber\b",
    r"\bamount\b",
    r"\bvalue\b",
    r"\byear\b",
)


def diagnose_page_modality_from_question_and_preflight(
    record: dict,
    page_context: dict,
    page_index: int,
) -> dict:
    """Derive non-gold prompt guidance from question text and runtime page metadata."""

    question_text = _question_text(record, page_context).lower()
    page_source = _page_source(page_context, page_index)
    has_page_image = bool(page_source.get("has_page_image") and page_source.get("page_image_path"))
    mentions_chart = _contains_any(question_text, CHART_TERMS)
    mentions_table = _contains_any(question_text, TABLE_TERMS)
    mentions_figure = _contains_any(question_text, FIGURE_TERMS)
    mentions_numeric_value = any(re.search(pattern, question_text) for pattern in NUMERIC_PATTERNS)
    requires_visual_reasoning = (
        mentions_chart
        or mentions_table
        or mentions_figure
        or _contains_any(question_text, VISUAL_TERMS)
    )

    recommended_artifact_types: list[str] = []
    if requires_visual_reasoning and has_page_image:
        recommended_artifact_types.append("visual_observation")
    if mentions_chart:
        recommended_artifact_types.extend(["figure", "numeric_fact"])
    if mentions_table:
        recommended_artifact_types.extend(["table", "numeric_fact"])
    if mentions_figure:
        recommended_artifact_types.extend(["figure", "caption"])
    if mentions_numeric_value:
        recommended_artifact_types.append("numeric_fact")

    return {
        "requires_visual_reasoning": bool(requires_visual_reasoning),
        "mentions_chart": bool(mentions_chart),
        "mentions_table": bool(mentions_table),
        "mentions_figure": bool(mentions_figure),
        "mentions_numeric_value": bool(mentions_numeric_value),
        "has_page_image": bool(has_page_image),
        "recommended_artifact_types": _dedupe_preserve_order(recommended_artifact_types),
    }


def _question_text(record: Mapping[str, Any], page_context: Mapping[str, Any]) -> str:
    question = record.get("question")
    if isinstance(question, dict):
        return str(question.get("text") or "")
    if question is not None:
        return str(question)
    return str(page_context.get("question") or "")


def _page_source(page_context: Mapping[str, Any], page_index: int) -> dict:
    for source in page_context.get("page_sources", []) or []:
        if not isinstance(source, dict):
            continue
        try:
            if int(source.get("page_index")) == int(page_index):
                return source
        except (TypeError, ValueError):
            continue
    return {}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
