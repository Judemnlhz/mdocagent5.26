"""Deterministic locator policy for Stage 4 graph quality gates."""

from __future__ import annotations

from typing import Any

from mdocnexus.stage2.locator_enrichment import (
    classify_artifact_locator,
    is_element_locatable as _is_stage2_element_locatable,
    is_proof_trace_eligible as _is_stage2_proof_trace_eligible,
)


def classify_locator(artifact: dict[str, Any]) -> dict[str, Any]:
    """Classify source, element, and proof locators using Stage 2 fields."""

    return classify_artifact_locator(artifact)


def is_element_locatable(artifact: dict[str, Any]) -> bool:
    return bool(_is_stage2_element_locatable(artifact))


def is_proof_trace_eligible(artifact: dict[str, Any]) -> bool:
    return bool(_is_stage2_proof_trace_eligible(artifact))
