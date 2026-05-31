"""Document-generic Stage 2 locator enrichment.

The functions here derive locator metadata only from page layout blocks,
source anchors, and artifact-local structured fields. They do not inspect
questions, answers, gold evidence, document names, sample ids, or task types.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Iterable


LOCATOR_KINDS = {
    "full_page_anchor",
    "source_block",
    "text_offset",
    "bbox",
    "table_cell",
    "figure_region",
    "caption_block",
    "section_block",
}

TEXT_BLOCK_RE = re.compile(r"_text_(\d+)$")
TEXT_BLOCK_SIZE = 1000


def page_id_for(doc_id: Any, page_index: Any) -> str:
    return f"{doc_id}#p{int(page_index):03d}"


def enrich_artifact_locators(raw_output: Any, page_input: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Attach deterministic locator fields to a page artifact output."""

    if not isinstance(raw_output, dict) or not isinstance(raw_output.get("artifacts"), list):
        return raw_output, []

    doc_id = str(page_input.get("doc_id") or raw_output.get("doc_id") or "")
    page_index = int(page_input.get("page_index", raw_output.get("page_index", 0)))
    page_id = page_id_for(doc_id, page_index)
    block_by_id = _layout_block_by_id(page_input.get("layout_blocks", []))

    normalized_output = dict(raw_output)
    normalized_output["doc_id"] = doc_id
    normalized_output["page_index"] = page_index
    normalized_output["page_id"] = page_id

    notes: list[dict[str, Any]] = []
    normalized_artifacts: list[Any] = []
    for local_index, artifact in enumerate(raw_output.get("artifacts", [])):
        if not isinstance(artifact, dict):
            normalized_artifacts.append(artifact)
            continue

        normalized_artifact = _normalize_artifact_basics(artifact, doc_id, page_index, page_id)
        normalized_artifact["source_anchors"] = _normalize_source_anchors(
            normalized_artifact.get("source_anchors"),
            page_index=page_index,
            block_by_id=block_by_id,
        )
        normalized_artifact["locators"] = _dedupe_locators(
            [
                *_existing_locators(normalized_artifact),
                *_locators_from_anchors(normalized_artifact.get("source_anchors"), block_by_id),
                *_locators_from_artifact_fields(normalized_artifact),
            ]
        )

        classification = classify_artifact_locator(normalized_artifact)
        normalized_artifact["source_anchored"] = bool(classification["source_anchored"])
        normalized_artifact["element_locatable"] = bool(classification["element_locatable"])
        normalized_artifact["proof_trace_eligible"] = bool(classification["proof_trace_eligible"])

        note = _locator_note(normalized_artifact, classification, local_index)
        if note is not None:
            normalized_output.setdefault("uncertain_or_unreadable", [])
            if isinstance(normalized_output["uncertain_or_unreadable"], list):
                marker = f"{note['artifact_id']}:{note['reason']}"
                if marker not in normalized_output["uncertain_or_unreadable"]:
                    normalized_output["uncertain_or_unreadable"].append(marker)
            notes.append(note)
        normalized_artifacts.append(normalized_artifact)

    normalized_output["artifacts"] = normalized_artifacts
    return normalized_output, notes


def classify_artifact_locator(artifact: dict[str, Any]) -> dict[str, Any]:
    """Classify locator strength using Stage 2 locator fields."""

    artifact_type = str(artifact.get("artifact_type") or "").lower()
    source_anchored = _has_source_anchor(artifact)
    has_full_page_anchor = _has_full_page_anchor(artifact)
    has_page_sha256 = _first_present(artifact, ("page_sha256",)) is not None

    has_text_locator = _has_text_locator(artifact)
    has_table_cell_locator = _has_table_cell_locator(artifact)
    has_bbox = _has_bbox(artifact)
    has_source_block = _has_non_full_page_block_id(artifact)
    has_figure_region = _has_figure_region_locator(artifact)
    has_caption_block = _has_locator_kind(artifact, "caption_block")
    has_section_block = _has_locator_kind(artifact, "section_block")

    if artifact_type == "text_span":
        element_locatable = has_text_locator
    elif artifact_type == "numeric_fact":
        element_locatable = has_text_locator or has_table_cell_locator
    elif artifact_type == "table_cell":
        element_locatable = has_table_cell_locator
    elif artifact_type == "table":
        element_locatable = _first_present(artifact, ("table_id", "table")) is not None and (has_bbox or has_source_block)
    elif artifact_type == "figure":
        element_locatable = has_figure_region or (_first_present(artifact, ("figure_id", "figure")) is not None and (has_bbox or has_source_block))
    elif artifact_type == "caption":
        element_locatable = has_caption_block or has_bbox or has_source_block
    elif artifact_type == "visual_region":
        element_locatable = has_figure_region or has_bbox
    elif artifact_type == "visual_observation":
        element_locatable = has_bbox or has_source_block or has_figure_region
    elif artifact_type == "section_header":
        element_locatable = has_section_block or has_text_locator or has_source_block
    else:
        element_locatable = has_bbox or has_source_block or has_table_cell_locator

    only_full_page = has_full_page_anchor and not (has_bbox or has_source_block or has_table_cell_locator or has_figure_region)
    proof_trace_eligible = bool(source_anchored and element_locatable and not only_full_page)
    if has_page_sha256 and not element_locatable:
        proof_trace_eligible = False

    locator_kind = _primary_locator_kind(artifact, element_locatable, has_full_page_anchor, has_page_sha256)
    return {
        "source_anchored": bool(source_anchored),
        "element_locatable": bool(element_locatable),
        "proof_trace_eligible": bool(proof_trace_eligible),
        "locator_kind": locator_kind,
        "locator_kinds": sorted(locator_kind_counts(artifact)),
        "has_full_page_anchor": bool(has_full_page_anchor),
        "has_page_sha256": bool(has_page_sha256),
    }


def locator_kind_counts(artifact: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for locator in _all_locator_dicts(artifact):
        kind = str(locator.get("locator_kind") or locator.get("kind") or "")
        if kind in LOCATOR_KINDS:
            counts[kind] += 1
    if not counts:
        if _has_full_page_anchor(artifact):
            counts["full_page_anchor"] += 1
        elif _has_source_anchor(artifact):
            counts["source_block"] += 1
    return dict(sorted(counts.items()))


def is_element_locatable(artifact: dict[str, Any]) -> bool:
    return bool(classify_artifact_locator(artifact)["element_locatable"])


def is_proof_trace_eligible(artifact: dict[str, Any]) -> bool:
    return bool(classify_artifact_locator(artifact)["proof_trace_eligible"])


def _normalize_artifact_basics(artifact: dict[str, Any], doc_id: str, page_index: int, page_id: str) -> dict[str, Any]:
    normalized = dict(artifact)
    normalized["doc_id"] = doc_id
    normalized["page_index"] = page_index
    normalized["page_id"] = str(normalized.get("page_id") or page_id)
    status = normalized.get("status") or normalized.get("validation_status") or "candidate"
    validation_status = normalized.get("validation_status") or status
    normalized["status"] = str(status)
    normalized["validation_status"] = str(validation_status)
    if not isinstance(normalized.get("normalized_content"), dict):
        normalized["normalized_content"] = {}
    if "locators" not in normalized or not isinstance(normalized.get("locators"), list):
        normalized["locators"] = []
    return normalized


def _normalize_source_anchors(anchors: Any, page_index: int, block_by_id: dict[str, dict[str, Any]]) -> list[Any]:
    if not isinstance(anchors, list):
        return anchors
    normalized_anchors: list[Any] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            normalized_anchors.append(anchor)
            continue
        normalized = dict(anchor)
        source_id = str(normalized.get("source_id") or "")
        block = block_by_id.get(source_id)
        normalized["page_index"] = int(block.get("page_index", page_index)) if block else page_index
        if normalized.get("bbox") in (None, [], "") and block and block.get("bbox") not in (None, [], ""):
            normalized["bbox"] = block.get("bbox")
        normalized_anchors.append(normalized)
    return normalized_anchors


def _layout_block_by_id(layout_blocks: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(layout_blocks, list):
        return {}
    return {
        str(block.get("block_id")): block
        for block in layout_blocks
        if isinstance(block, dict) and block.get("block_id") not in (None, "")
    }


def _existing_locators(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    locators = artifact.get("locators")
    if not isinstance(locators, list):
        return []
    return [locator for locator in locators if isinstance(locator, dict)]


def _locators_from_anchors(anchors: Any, block_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    locators: list[dict[str, Any]] = []
    if not isinstance(anchors, list):
        return locators
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        source_id = str(anchor.get("source_id") or "")
        if not source_id:
            continue
        anchor_type = _anchor_type(anchor)
        if anchor_type == "full_page_anchor":
            locators.append({"locator_kind": "full_page_anchor", "source_id": source_id})
        else:
            locators.append({"locator_kind": "source_block", "block_id": source_id, "source_id": source_id})
        if anchor.get("bbox") not in (None, [], ""):
            locators.append({"locator_kind": "bbox", "source_id": source_id, "bbox": anchor.get("bbox")})
        block = block_by_id.get(source_id)
        if block and str(block.get("block_type") or "") == "text_block":
            start, end = _block_text_offsets(block, source_id)
            if start is not None and end is not None:
                locators.append(
                    {
                        "locator_kind": "text_offset",
                        "block_id": source_id,
                        "char_start": int(start),
                        "char_end": int(end),
                    }
                )
    return locators


def _locators_from_artifact_fields(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    locators: list[dict[str, Any]] = []
    table_id = _first_present(artifact, ("table_id", "table"))
    row_index = _first_present(artifact, ("row_index", "row"))
    column_index = _first_present(artifact, ("column_index", "col_index", "col", "column"))
    if table_id is not None and row_index is not None and column_index is not None:
        locators.append(
            {
                "locator_kind": "table_cell",
                "table_id": table_id,
                "row_index": row_index,
                "column_index": column_index,
            }
        )

    figure_id = _first_present(artifact, ("figure_id", "figure"))
    if str(artifact.get("artifact_type") or "").lower() in {"figure", "visual_region"} and figure_id is not None:
        locator = {"locator_kind": "figure_region", "figure_id": figure_id}
        bbox = _first_bbox(artifact)
        if bbox not in (None, [], ""):
            locator["bbox"] = bbox
            locators.append(locator)

    caption_id = _first_present(artifact, ("caption_id", "caption"))
    if str(artifact.get("artifact_type") or "").lower() == "caption" and caption_id is not None:
        locator = {"locator_kind": "caption_block", "caption_id": caption_id}
        block_id = _first_present(artifact, ("block_id", "source_block_id"))
        if block_id is not None:
            locator["block_id"] = block_id
        locators.append(locator)

    section_id = _first_present(artifact, ("section_id", "section"))
    if str(artifact.get("artifact_type") or "").lower() == "section_header" and section_id is not None:
        locator = {"locator_kind": "section_block", "section_id": section_id}
        block_id = _first_present(artifact, ("block_id", "source_block_id"))
        if block_id is not None:
            locator["block_id"] = block_id
        locators.append(locator)
    return locators


def _block_text_offsets(block: dict[str, Any], block_id: str) -> tuple[int | None, int | None]:
    if block.get("char_start") not in (None, "") and block.get("char_end") not in (None, ""):
        return int(block["char_start"]), int(block["char_end"])
    text = block.get("text")
    match = TEXT_BLOCK_RE.search(block_id)
    if not isinstance(text, str) or not match:
        return None, None
    start = int(match.group(1)) * TEXT_BLOCK_SIZE
    return start, start + len(text)


def _locator_note(artifact: dict[str, Any], classification: dict[str, Any], local_index: int) -> dict[str, Any] | None:
    if classification["element_locatable"]:
        return None
    reason = "missing_element_locator"
    if classification["source_anchored"] and not _has_bbox(artifact):
        reason = "missing_bbox_locator"
    elif not classification["source_anchored"]:
        reason = "missing_source_locator"
    return {
        "artifact_id": str(artifact.get("artifact_id") or f"artifact_{local_index:04d}"),
        "reason": reason,
        "has_page_locator": artifact.get("page_index") is not None or artifact.get("page_id") is not None,
        "has_source_locator": bool(classification["source_anchored"]),
        "has_element_locator": bool(classification["element_locatable"]),
        "has_bbox_locator": _has_bbox(artifact),
        "locator_kind": classification["locator_kind"],
    }


def _dedupe_locators(locators: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for locator in locators:
        if not isinstance(locator, dict):
            continue
        kind = str(locator.get("locator_kind") or locator.get("kind") or "")
        if kind not in LOCATOR_KINDS:
            continue
        normalized = {key: value for key, value in locator.items() if value not in (None, "", [])}
        normalized["locator_kind"] = kind
        key = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _primary_locator_kind(
    artifact: dict[str, Any],
    element_locatable: bool,
    has_full_page_anchor: bool,
    has_page_sha256: bool,
) -> str:
    counts = locator_kind_counts(artifact)
    if element_locatable:
        for kind in ("table_cell", "figure_region", "caption_block", "section_block", "text_offset", "bbox", "source_block"):
            if counts.get(kind):
                return kind
    if has_full_page_anchor:
        return "full_page_anchor"
    if counts.get("source_block"):
        return "source_block"
    if has_page_sha256:
        return "page_sha256"
    return "none"


def _has_source_anchor(artifact: dict[str, Any]) -> bool:
    anchors = artifact.get("source_anchors")
    return isinstance(anchors, list) and any(isinstance(anchor, dict) and anchor.get("source_id") not in (None, "") for anchor in anchors)


def _has_full_page_anchor(artifact: dict[str, Any]) -> bool:
    return any(_anchor_type(anchor) == "full_page_anchor" for anchor in _source_anchors(artifact)) or _has_locator_kind(artifact, "full_page_anchor")


def _has_text_locator(artifact: dict[str, Any]) -> bool:
    has_block = _has_non_full_page_block_id(artifact)
    has_offset = _first_present(artifact, ("char_start", "char_end", "text_span_offset", "text_span_start", "text_span_end", "start_offset", "end_offset")) is not None
    return bool(has_block and (has_offset or _has_bbox(artifact)))


def _has_table_cell_locator(artifact: dict[str, Any]) -> bool:
    return (
        _first_present(artifact, ("table_id", "table")) is not None
        and _first_present(artifact, ("row_index", "row")) is not None
        and _first_present(artifact, ("column_index", "col_index", "col", "column")) is not None
    )


def _has_non_full_page_block_id(artifact: dict[str, Any]) -> bool:
    for locator in _all_locator_dicts(artifact):
        if str(locator.get("locator_kind") or "") in {"source_block", "caption_block", "section_block"} and locator.get("block_id") not in (None, ""):
            return True
        if locator.get("source_id") not in (None, "") and str(locator.get("locator_kind") or "") == "source_block":
            return True
    if _first_present(artifact, ("block_id", "source_block_id")) is not None:
        return True
    for anchor in _source_anchors(artifact):
        if _anchor_type(anchor) == "full_page_anchor":
            continue
        if anchor.get("source_id") not in (None, ""):
            return True
    return False


def _has_bbox(artifact: dict[str, Any]) -> bool:
    return _first_bbox(artifact) not in (None, [], "")


def _first_bbox(artifact: dict[str, Any]) -> Any:
    for locator in _all_locator_dicts(artifact):
        if locator.get("bbox") not in (None, "", []):
            return locator.get("bbox")
    if artifact.get("bbox") not in (None, "", []):
        return artifact.get("bbox")
    for anchor in _source_anchors(artifact):
        if anchor.get("bbox") not in (None, "", []):
            return anchor.get("bbox")
    return None


def _has_locator_kind(artifact: dict[str, Any], kind: str) -> bool:
    return any(str(locator.get("locator_kind") or locator.get("kind") or "") == kind for locator in _all_locator_dicts(artifact))


def _has_figure_region_locator(artifact: dict[str, Any]) -> bool:
    for locator in _all_locator_dicts(artifact):
        if str(locator.get("locator_kind") or locator.get("kind") or "") != "figure_region":
            continue
        if locator.get("bbox") not in (None, "", []):
            return True
        if locator.get("block_id") not in (None, "", []):
            return True
    return False


def _all_locator_dicts(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    locators: list[dict[str, Any]] = []
    value = artifact.get("locators")
    if isinstance(value, list):
        locators.extend(locator for locator in value if isinstance(locator, dict))
    for container in _containers(artifact):
        for key in ("locator", "element_locator", "layout", "metadata"):
            child = container.get(key)
            if isinstance(child, dict):
                locators.append(child)
    return locators


def _source_anchors(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = artifact.get("source_anchors")
    if not isinstance(anchors, list):
        return []
    return [anchor for anchor in anchors if isinstance(anchor, dict)]


def _anchor_type(anchor: dict[str, Any]) -> str:
    anchor_type = str(anchor.get("anchor_type") or "").lower()
    if anchor_type in {"full_page", "full_page_image", "page_image", "full_page_anchor"}:
        return "full_page_anchor"
    return anchor_type


def _first_present(artifact: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for container in _containers(artifact):
        for key in keys:
            value = container.get(key)
            if value not in (None, "", []):
                return value
    for locator in _all_locator_dicts(artifact):
        for key in keys:
            value = locator.get(key)
            if value not in (None, "", []):
                return value
    return None


def _containers(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = [artifact]
    for key in ("normalized_content", "provenance"):
        value = artifact.get(key)
        if isinstance(value, dict):
            containers.append(value)
    normalized = artifact.get("normalized_content")
    if isinstance(normalized, dict):
        for key in ("locator", "element_locator", "layout", "metadata"):
            value = normalized.get(key)
            if isinstance(value, dict):
                containers.append(value)
    return containers
