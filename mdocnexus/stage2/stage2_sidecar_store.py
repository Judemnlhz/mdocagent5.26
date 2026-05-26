"""Compact Stage 2 index and preflight sidecar helpers."""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping


COMPACT_STAGE2_ALLOWED_FIELDS = {
    "version",
    "status",
    "doc_name",
    "page_count",
    "page_count_source",
    "pages_to_compile",
    "valid_explicit_page_indices",
    "invalid_explicit_page_reference_count",
    "preflight_ref",
    "artifact_store_refs",
    "quality_summary_ref",
}

FORBIDDEN_SIDECAR_FIELDS = {
    "answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "api_key",
}


def build_stage2_record_index(
    record: Mapping[str, Any],
    stage2_preflight: Mapping[str, Any],
    preflight_ref: str | Path,
    record_index: int | None = None,
) -> Dict[str, Any]:
    """Build the compact stage2 index stored on the original record."""

    _ = record
    explicit_validation = stage2_preflight.get("explicit_page_validation", {})
    preflight = stage2_preflight.get("preflight", {})
    page_count_value, page_count_source = _compact_page_count(stage2_preflight.get("page_count"))
    compact = {
        "version": stage2_preflight.get("version", "stage2_preflight_v1"),
        "status": "preflight_passed" if preflight.get("passed", False) else "preflight_failed",
        "doc_name": stage2_preflight.get("doc_name"),
        "page_count": page_count_value,
        "page_count_source": page_count_source,
        "pages_to_compile": [int(page) for page in stage2_preflight.get("pages_to_compile", [])],
        "valid_explicit_page_indices": [
            int(page)
            for page in explicit_validation.get("valid_explicit_page_indices", [])
        ],
        "invalid_explicit_page_reference_count": len(
            explicit_validation.get("invalid_explicit_page_references", [])
        ),
        "preflight_ref": str(preflight_ref),
        "artifact_store_refs": [],
        "quality_summary_ref": None,
    }
    unexpected = sorted(set(compact) - COMPACT_STAGE2_ALLOWED_FIELDS)
    if unexpected:
        raise ValueError(f"Compact stage2 index has unexpected fields: {unexpected}")
    forbidden = sorted(field for field in FORBIDDEN_SIDECAR_FIELDS if contains_key(compact, field))
    if forbidden:
        raise ValueError(f"Compact stage2 index contains forbidden fields: {forbidden}")
    return compact


def build_stage2_preflight_sidecar(
    record: Mapping[str, Any],
    stage2_preflight: Mapping[str, Any],
    record_key: str | None = None,
) -> Dict[str, Any]:
    """Build a sidecar with detailed preflight data and no gold/eval fields."""

    sidecar = {
        "record_key": record_key or build_record_key(record),
        "doc_id": record.get("doc_id"),
        "question": record.get("question"),
        "question_constraints": stage2_preflight.get("question_constraints", {}),
        "retrieval_pages": stage2_preflight.get("retrieval_pages", {}),
        "explicit_page_validation": stage2_preflight.get("explicit_page_validation", {}),
        "page_sources": stage2_preflight.get("page_sources", []),
        "layout_blocks_by_page": build_layout_blocks_by_page(stage2_preflight.get("page_sources", [])),
        "preflight": stage2_preflight.get("preflight", {}),
    }
    forbidden = sorted(field for field in FORBIDDEN_SIDECAR_FIELDS if contains_key(sidecar, field))
    if forbidden:
        raise ValueError(f"Stage 2 sidecar contains forbidden fields: {forbidden}")
    return sidecar


def write_stage2_preflight_sidecar(sidecar: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a Stage 2 preflight sidecar as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sidecar), ensure_ascii=False, indent=2), encoding="utf-8")


def load_stage2_preflight_sidecar(path: str | Path) -> Dict[str, Any]:
    """Load a Stage 2 preflight sidecar and reject forbidden fields."""

    sidecar_path = Path(path)
    loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Stage 2 sidecar root must be an object: {path}")
    forbidden = sorted(field for field in FORBIDDEN_SIDECAR_FIELDS if contains_key(loaded, field))
    if forbidden:
        raise ValueError(f"Stage 2 sidecar contains forbidden fields: {forbidden}")
    return loaded


def resolve_stage2_preflight(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return legacy embedded preflight details or load compact sidecar details."""

    stage2 = record.get("stage2", {})
    if not isinstance(stage2, dict):
        return {}
    preflight_ref = stage2.get("preflight_ref")
    if preflight_ref:
        sidecar = load_stage2_preflight_sidecar(preflight_ref)
        return {
            "version": stage2.get("version", "stage2_preflight_v1"),
            "doc_name": stage2.get("doc_name"),
            "page_count": {
                "value": stage2.get("page_count"),
                "source": stage2.get("page_count_source"),
            },
            "question_constraints": sidecar.get("question_constraints", {}),
            "retrieval_pages": sidecar.get("retrieval_pages", {}),
            "explicit_page_validation": sidecar.get("explicit_page_validation", {}),
            "pages_to_compile": stage2.get("pages_to_compile", []),
            "page_sources": sidecar.get("page_sources", []),
            "layout_blocks_by_page": sidecar.get("layout_blocks_by_page", {}),
            "preflight": sidecar.get("preflight", {}),
        }
    return dict(stage2)


def build_record_key(record: Mapping[str, Any], record_index: int | None = None) -> str:
    """Build a deterministic filesystem-safe key for sidecar files."""

    doc_id = str(record.get("doc_id") or record.get("record_id") or "record")
    prefix = f"{int(record_index):06d}_" if record_index is not None else ""
    safe_doc = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc_id).strip("._") or "record"
    question = str(record.get("question") or "")
    digest = hashlib.sha1(f"{doc_id}\n{question}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{safe_doc}_{digest}"


def build_layout_blocks_by_page(page_sources: Any) -> Dict[str, Any]:
    """Build compact layout block descriptors from page source block ids."""

    result: Dict[str, Any] = {}
    if not isinstance(page_sources, list):
        return result
    for source in page_sources:
        if not isinstance(source, dict) or source.get("page_index") is None:
            continue
        page_index = int(source["page_index"])
        blocks = []
        for block_id in source.get("layout_block_ids", []) or []:
            block_id = str(block_id)
            blocks.append(
                {
                    "block_id": block_id,
                    "block_type": "full_page_image" if block_id.endswith("_full_page_image") else "text_block",
                    "page_index": page_index,
                }
            )
        result[str(page_index)] = blocks
    return result


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(contains_key(child, key) for child in value)
    return False


def _compact_page_count(page_count: Any) -> tuple[int | None, str]:
    if isinstance(page_count, dict):
        value = page_count.get("value")
        source = page_count.get("source") or "unknown"
    else:
        value = page_count
        source = "unknown"
    try:
        compact_value = int(value) if value is not None else None
    except (TypeError, ValueError):
        compact_value = None
    return compact_value, str(source)
