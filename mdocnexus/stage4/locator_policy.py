"""Deterministic locator policy for Stage 4 graph quality gates."""

from __future__ import annotations

from typing import Any


def classify_locator(artifact: dict[str, Any]) -> dict[str, Any]:
    """Classify source, element, and proof locators without semantic inference."""

    anchors = _source_anchors(artifact)
    anchor_types = {_anchor_type(anchor) for anchor in anchors}
    has_source_anchor = bool(anchors)
    has_full_page_anchor = "full_page_anchor" in anchor_types or any(
        str(anchor.get("source_id") or "").endswith("_full_page_image") for anchor in anchors
    )
    has_page_sha256 = _first_present(artifact, ("page_sha256",)) is not None
    artifact_type = str(artifact.get("artifact_type") or "").lower()
    locator_kind = "source_anchor" if has_source_anchor else "none"

    element_locatable = _has_element_locator_for_type(artifact, artifact_type)
    proof_eligible = bool(has_source_anchor and element_locatable)
    if artifact_type == "visual_observation" and has_full_page_anchor and not _has_bbox_or_block_id(artifact):
        proof_eligible = False
    if has_full_page_anchor and not element_locatable:
        proof_eligible = False
    if has_page_sha256 and not element_locatable:
        proof_eligible = False
    if element_locatable:
        locator_kind = _locator_kind_for_type(artifact, artifact_type)
    elif has_full_page_anchor:
        locator_kind = "full_page_anchor"
    elif has_page_sha256:
        locator_kind = "page_sha256"

    return {
        "source_anchored": has_source_anchor,
        "element_locatable": element_locatable,
        "proof_trace_eligible": proof_eligible,
        "locator_kind": locator_kind,
        "has_full_page_anchor": has_full_page_anchor,
        "has_page_sha256": has_page_sha256,
    }


def is_element_locatable(artifact: dict[str, Any]) -> bool:
    return bool(classify_locator(artifact)["element_locatable"])


def is_proof_trace_eligible(artifact: dict[str, Any]) -> bool:
    return bool(classify_locator(artifact)["proof_trace_eligible"])


def _has_element_locator_for_type(artifact: dict[str, Any], artifact_type: str) -> bool:
    if artifact_type == "text_span":
        return _has_block_id(artifact) and (_has_text_offset(artifact) or _has_bbox(artifact))
    if artifact_type == "numeric_fact":
        return (_has_block_id(artifact) and (_has_text_offset(artifact) or _has_bbox(artifact))) or _has_table_cell_locator(artifact)
    if artifact_type == "table_cell":
        return _has_table_cell_locator(artifact)
    if artifact_type in {"figure", "caption", "figure_caption", "visual_observation"}:
        return _has_bbox(artifact) or _has_block_id(artifact)
    return _has_bbox(artifact) or _has_block_id(artifact) or _has_table_cell_locator(artifact)


def _locator_kind_for_type(artifact: dict[str, Any], artifact_type: str) -> str:
    if _has_table_cell_locator(artifact):
        return "table_cell"
    if _has_bbox(artifact):
        return "bbox"
    if _has_text_offset(artifact):
        return "text_offset"
    if _has_block_id(artifact):
        return "block_id"
    return artifact_type or "element"


def _source_anchors(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = artifact.get("source_anchors")
    if not isinstance(anchors, list):
        return []
    return [anchor for anchor in anchors if isinstance(anchor, dict) and anchor.get("source_id") not in (None, "")]


def _anchor_type(anchor: dict[str, Any]) -> str:
    anchor_type = str(anchor.get("anchor_type") or "").lower()
    if anchor_type in {"full_page", "full_page_image", "page_image"}:
        return "full_page_anchor"
    return anchor_type


def _has_table_cell_locator(artifact: dict[str, Any]) -> bool:
    return (
        _first_present(artifact, ("table_id", "table")) is not None
        and _first_present(artifact, ("row_index", "row")) is not None
        and _first_present(artifact, ("column_index", "col_index", "col", "column")) is not None
    )


def _has_bbox_or_block_id(artifact: dict[str, Any]) -> bool:
    return _has_bbox(artifact) or _has_block_id(artifact)


def _has_bbox(artifact: dict[str, Any]) -> bool:
    value = _first_present(artifact, ("bbox",))
    if value not in (None, "", []):
        return True
    return any(anchor.get("bbox") not in (None, "", []) for anchor in _source_anchors(artifact))


def _has_block_id(artifact: dict[str, Any]) -> bool:
    if _first_present(artifact, ("block_id", "source_block_id", "source_id")) is not None:
        return True
    for anchor in _source_anchors(artifact):
        if _anchor_type(anchor) == "full_page_anchor":
            continue
        if anchor.get("source_id") not in (None, ""):
            return True
    return False


def _has_text_offset(artifact: dict[str, Any]) -> bool:
    return _first_present(
        artifact,
        ("text_span_offset", "text_span_start", "text_span_end", "start_offset", "end_offset"),
    ) is not None


def _first_present(artifact: dict[str, Any], keys: tuple[str, ...]) -> Any:
    containers: list[dict[str, Any]] = [artifact]
    normalized = artifact.get("normalized_content")
    if isinstance(normalized, dict):
        containers.append(normalized)
    provenance = artifact.get("provenance")
    if isinstance(provenance, dict):
        containers.append(provenance)
    for key in ("locator", "element_locator", "layout", "metadata"):
        value = artifact.get(key)
        if isinstance(value, dict):
            containers.append(value)
        if isinstance(normalized, dict) and isinstance(normalized.get(key), dict):
            containers.append(normalized[key])
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value not in (None, "", []):
                return value
    return None
