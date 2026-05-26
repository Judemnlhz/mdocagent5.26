"""Serializable schema dictionaries for Stage 2 artifact outputs."""

from __future__ import annotations

from typing import Any, Dict, List

from .schemas import (
    ALLOWED_ANCHOR_TYPES,
    ALLOWED_VALIDATION_STATUSES,
    ArtifactType,
    Modality,
    ProvenanceOp,
)


def get_allowed_artifact_types() -> List[str]:
    return [item.value for item in ArtifactType]


def get_allowed_modalities() -> List[str]:
    return [item.value for item in Modality]


def get_allowed_provenance_ops() -> List[str]:
    return [item.value for item in ProvenanceOp]


def get_allowed_anchor_types() -> List[str]:
    return sorted(ALLOWED_ANCHOR_TYPES)


def get_allowed_validation_statuses() -> List[str]:
    return sorted(ALLOWED_VALIDATION_STATUSES)


def build_evidence_artifact_schema_dict() -> Dict[str, Any]:
    """Return a JSON-schema-like dict for one EvidenceArtifact."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artifact_id",
            "doc_id",
            "page_index",
            "artifact_type",
            "modality",
            "content",
            "normalized_content",
            "source_anchors",
            "provenance",
            "validation_status",
            "compiler_metadata",
        ],
        "properties": {
            "artifact_id": {"type": "string", "minLength": 1},
            "doc_id": {"type": "string", "minLength": 1},
            "page_index": {"type": "integer", "minimum": 0},
            "artifact_type": {
                "type": "string",
                "enum": get_allowed_artifact_types(),
            },
            "modality": {
                "type": "string",
                "enum": get_allowed_modalities(),
            },
            "content": {"type": "string", "minLength": 1, "maxLength": 1200},
            "normalized_content": {"type": "object"},
            "source_anchors": {
                "type": "array",
                "minItems": 1,
                "items": build_source_anchor_schema_dict(),
            },
            "provenance": build_provenance_schema_dict(),
            "validation_status": {
                "type": "string",
                "enum": get_allowed_validation_statuses(),
            },
            "compiler_metadata": {"type": "object"},
        },
    }


def build_page_artifact_output_schema_dict() -> Dict[str, Any]:
    """Return a JSON-schema-like dict for one PageArtifactOutput."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["doc_id", "page_index", "artifacts"],
        "properties": {
            "doc_id": {"type": "string", "minLength": 1},
            "page_index": {"type": "integer", "minimum": 0},
            "artifacts": {
                "type": "array",
                "items": build_evidence_artifact_schema_dict(),
            },
            "uncertain_or_unreadable": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def build_source_anchor_schema_dict() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_id", "anchor_type", "page_index", "bbox"],
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "anchor_type": {
                "type": "string",
                "enum": get_allowed_anchor_types(),
            },
            "page_index": {"type": "integer", "minimum": 0},
            "bbox": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                ]
            },
        },
    }


def build_provenance_schema_dict() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "sources"],
        "properties": {
            "op": {
                "type": "string",
                "enum": get_allowed_provenance_ops(),
            },
            "sources": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
