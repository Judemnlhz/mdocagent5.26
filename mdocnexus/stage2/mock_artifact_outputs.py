"""Mock artifact outputs for Stage 2 integration dry-run tests only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


MOCK_COMPILER_METADATA = {
    "compiler_name": "mock_artifact_output_builder",
    "compiler_version": "mock_stage2_dry_run",
}


def build_mock_page_artifact_output(
    doc_id: str,
    page_index: int,
    layout_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build deterministic mock artifacts anchored to existing layout blocks."""

    full_page_block = _first_block_of_type(layout_blocks, "full_page_image")
    text_block = _first_block_of_type(layout_blocks, "text_block")
    artifacts: List[Dict[str, Any]] = []

    if full_page_block is not None:
        artifacts.append(_build_visual_observation(doc_id, page_index, full_page_block))
    if text_block is not None:
        artifacts.append(_build_text_span(doc_id, page_index, text_block))

    return {
        "doc_id": doc_id,
        "page_index": page_index,
        "artifacts": artifacts,
        "uncertain_or_unreadable": [],
    }


def _build_visual_observation(
    doc_id: str,
    page_index: int,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    source_id = block["block_id"]
    return {
        "artifact_id": f"{_doc_stem(doc_id)}_p{page_index:03d}_visual_observation_0001",
        "doc_id": doc_id,
        "page_index": page_index,
        "artifact_type": "visual_observation",
        "modality": "visual",
        "content": "Mock visual observation anchored to the full page image.",
        "normalized_content": {
            "observation_scope": "full_page",
            "presence": "undetermined",
        },
        "source_anchors": [
            {
                "source_id": source_id,
                "anchor_type": "full_page_image",
                "page_index": page_index,
                "bbox": None,
            }
        ],
        "provenance": {
            "op": "ATOM",
            "sources": [source_id],
        },
        "validation_status": "candidate",
        "compiler_metadata": dict(MOCK_COMPILER_METADATA),
    }


def _build_text_span(
    doc_id: str,
    page_index: int,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    source_id = block["block_id"]
    return {
        "artifact_id": f"{_doc_stem(doc_id)}_p{page_index:03d}_text_span_0001",
        "doc_id": doc_id,
        "page_index": page_index,
        "artifact_type": "text_span",
        "modality": "text",
        "content": "Mock text span anchored to a text block.",
        "normalized_content": {},
        "source_anchors": [
            {
                "source_id": source_id,
                "anchor_type": "text_block",
                "page_index": page_index,
                "bbox": None,
            }
        ],
        "provenance": {
            "op": "ATOM",
            "sources": [source_id],
        },
        "validation_status": "candidate",
        "compiler_metadata": dict(MOCK_COMPILER_METADATA),
    }


def _first_block_of_type(
    layout_blocks: List[Dict[str, Any]],
    block_type: str,
) -> Optional[Dict[str, Any]]:
    for block in layout_blocks:
        if block.get("block_type") == block_type:
            return block
    return None


def _doc_stem(doc_id: str) -> str:
    return Path(doc_id).name.removesuffix(".pdf")
