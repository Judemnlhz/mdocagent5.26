"""Deterministic artifact deduplication for replay validation."""

from __future__ import annotations

import json
from typing import Any


def normalize_for_hash(value: Any) -> str:
    """Return a stable string representation for deduplication keys."""

    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_artifact_dedup_key(artifact: dict) -> str:
    """Build the fixed Stage 2 artifact deduplication key."""

    source_ids = sorted(
        str(anchor.get("source_id"))
        for anchor in artifact.get("source_anchors", []) or []
        if isinstance(anchor, dict) and anchor.get("source_id") is not None
    )
    normalized_content = artifact.get("normalized_content")
    content_key = normalized_content if normalized_content else artifact.get("content")
    key = {
        "doc_id": artifact.get("doc_id"),
        "page_index": artifact.get("page_index"),
        "artifact_type": artifact.get("artifact_type"),
        "modality": artifact.get("modality"),
        "source_ids": source_ids,
        "content": normalize_for_hash(content_key),
    }
    return json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deduplicate_page_artifacts(page_output: dict) -> tuple[dict, list[dict]]:
    """Drop duplicate candidate artifacts while preserving the first instance."""

    deduped_output = dict(page_output)
    artifacts = page_output.get("artifacts", [])
    if not isinstance(artifacts, list):
        deduped_output["artifacts"] = artifacts
        return deduped_output, []

    kept_artifacts = []
    removed = []
    first_by_key: dict[str, dict] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            kept_artifacts.append(artifact)
            continue
        dedup_key = build_artifact_dedup_key(artifact)
        if dedup_key not in first_by_key:
            first_by_key[dedup_key] = artifact
            kept_artifacts.append(artifact)
            continue
        kept = first_by_key[dedup_key]
        removed.append(
            {
                "error_type": "duplicate_artifact_deduplicated",
                "artifact_id": artifact.get("artifact_id"),
                "duplicate_of": kept.get("artifact_id"),
                "dedup_key": dedup_key,
                "doc_id": artifact.get("doc_id"),
                "page_index": artifact.get("page_index"),
            }
        )

    deduped_output["artifacts"] = kept_artifacts
    return deduped_output, removed
