"""Question constraint parsing for Stage 2 canonical records.

The parser is deterministic and does not call an LLM. It extracts only explicit
question constraints that can safely guide candidate-page compilation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


PAGE_PATTERN = re.compile(r"\bpages?\s+(\d+)\b", re.IGNORECASE)
COLOR_TERMS = ("blue", "red", "black", "green", "yellow", "white")


def extract_explicit_page_references(question_text: str) -> List[Dict[str, Any]]:
    """Extract explicit one-based page references from the question text."""

    references: List[Dict[str, Any]] = []
    for match in PAGE_PATTERN.finditer(question_text or ""):
        page_number = int(match.group(1))
        if page_number < 1:
            continue

        references.append(
            {
                "surface_text": match.group(0),
                "page_number_one_based": page_number,
                "page_index_zero_based": page_number - 1,
                "source": "question_text",
            }
        )
    return references


def extract_visual_attribute_constraints(question_text: str) -> List[Dict[str, str]]:
    """Extract simple visual attributes that should bias compilation."""

    q = (question_text or "").lower()
    constraints: List[Dict[str, str]] = []

    for color in COLOR_TERMS:
        if re.search(rf"\b{re.escape(color)}\b", q):
            constraints.append(
                {
                    "attribute_type": "color",
                    "attribute_value": color,
                    "source": "question_text",
                }
            )

    if re.search(r"\b(handwritten|handwriting)\b", q):
        constraints.append(
            {
                "attribute_type": "script_type",
                "attribute_value": "handwritten",
                "source": "question_text",
            }
        )

    return constraints


def infer_target_object(question_text: str) -> Dict[str, str]:
    """Infer a coarse target object and modality requirement from the question."""

    q = (question_text or "").lower()

    if re.search(r"\bwords?\b", q):
        object_type = "words"
    elif re.search(r"\bsignature\b", q):
        object_type = "signature"
    elif re.search(r"\bstamp\b", q):
        object_type = "stamp"
    else:
        object_type = "unspecified"

    visual_terms = (
        "color",
        "blue",
        "red",
        "black",
        "green",
        "yellow",
        "white",
        "handwritten",
        "handwriting",
        "image",
        "figure",
        "page",
    )
    modality_requirement = "visual" if any(term in q for term in visual_terms) else "text"

    return {
        "object_type": object_type,
        "modality_requirement": modality_requirement,
    }


def parse_question_constraints(question_text: str) -> Dict[str, Any]:
    """Build the approved question_constraints block for canonical records."""

    return {
        "explicit_page_references": extract_explicit_page_references(question_text),
        "visual_attribute_constraints": extract_visual_attribute_constraints(question_text),
        "target_object": infer_target_object(question_text),
    }
