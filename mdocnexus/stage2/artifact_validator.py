"""Deterministic validation skeleton for Stage 2 artifact outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from .schema_serialization import (
    get_allowed_anchor_types,
    get_allowed_artifact_types,
    get_allowed_modalities,
    get_allowed_provenance_ops,
    get_allowed_validation_statuses,
)
from .validation_errors import ValidationErrorType, ValidationIssue


PAGE_OUTPUT_ALLOWED_FIELDS = {"doc_id", "page_index", "artifacts", "uncertain_or_unreadable"}


ARTIFACT_REQUIRED_FIELDS = [
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
]


def validate_page_artifact_output_schema(
    raw_output: Dict[str, Any],
) -> Tuple[bool, List[ValidationIssue]]:
    """Validate only the page-level output container shape."""

    issues: List[ValidationIssue] = []
    if not isinstance(raw_output, dict):
        return False, [
            ValidationIssue(
                error_type=ValidationErrorType.invalid_output_container,
                message="Page artifact output must be a dict.",
                field_path="$",
                details={"observed_type": type(raw_output).__name__},
            )
        ]

    for field_name in sorted(set(raw_output) - PAGE_OUTPUT_ALLOWED_FIELDS):
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_output_container,
                message=f"Unexpected page output field: {field_name}",
                doc_id=_safe_doc_id(raw_output),
                page_index=_safe_page_index(raw_output),
                field_path=field_name,
                details={"unexpected_field": field_name},
            )
        )

    for field_name in ("doc_id", "page_index", "artifacts"):
        if field_name not in raw_output:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.missing_required_field,
                    message=f"Missing required page output field: {field_name}",
                    doc_id=_safe_doc_id(raw_output),
                    page_index=_safe_page_index(raw_output),
                    field_path=field_name,
                    details={"missing_field": field_name},
                )
            )

    if "artifacts" in raw_output and not isinstance(raw_output["artifacts"], list):
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_output_container,
                message="Page output artifacts must be a list.",
                doc_id=_safe_doc_id(raw_output),
                page_index=_safe_page_index(raw_output),
                field_path="artifacts",
                details={"observed_type": type(raw_output["artifacts"]).__name__},
            )
        )

    if (
        "uncertain_or_unreadable" in raw_output
        and not isinstance(raw_output["uncertain_or_unreadable"], list)
    ):
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_output_container,
                message="uncertain_or_unreadable must be a list when present.",
                doc_id=_safe_doc_id(raw_output),
                page_index=_safe_page_index(raw_output),
                field_path="uncertain_or_unreadable",
                details={
                    "observed_type": type(raw_output["uncertain_or_unreadable"]).__name__,
                },
            )
        )

    if "page_index" in raw_output and not _is_non_negative_int(raw_output["page_index"]):
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_page_index,
                message="Page output page_index must be a non-negative integer.",
                doc_id=_safe_doc_id(raw_output),
                page_index=_safe_page_index(raw_output),
                field_path="page_index",
                details={"observed_value": raw_output.get("page_index")},
            )
        )

    return not issues, issues


def validate_evidence_artifact_schema(
    raw_artifact: Dict[str, Any],
    doc_id: str,
    page_index: int,
) -> List[ValidationIssue]:
    """Validate one artifact dict against Stage 2 deterministic rules."""

    if not isinstance(raw_artifact, dict):
        return [
            ValidationIssue(
                error_type=ValidationErrorType.schema_invalid,
                message="Evidence artifact must be a dict.",
                doc_id=doc_id,
                page_index=page_index,
                field_path="artifacts[]",
                details={"observed_type": type(raw_artifact).__name__},
            )
        ]

    issues: List[ValidationIssue] = []
    artifact_id = raw_artifact.get("artifact_id")

    for field_name in ARTIFACT_REQUIRED_FIELDS:
        if field_name not in raw_artifact:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.missing_required_field,
                    message=f"Missing required artifact field: {field_name}",
                    doc_id=doc_id,
                    page_index=page_index,
                    artifact_id=artifact_id,
                    field_path=field_name,
                    details={"missing_field": field_name},
                )
            )

    for field_name in sorted(set(raw_artifact) - set(ARTIFACT_REQUIRED_FIELDS)):
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.schema_invalid,
                message=f"Unexpected artifact field: {field_name}",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path=field_name,
                details={"unexpected_field": field_name},
            )
        )

    if issues:
        return issues

    if not isinstance(raw_artifact["artifact_id"], str) or not raw_artifact["artifact_id"].strip():
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_artifact_id,
                message="artifact_id must be a non-empty string.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="artifact_id",
                details={"observed_value": raw_artifact.get("artifact_id")},
            )
        )

    if raw_artifact["doc_id"] != doc_id:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.schema_invalid,
                message="artifact.doc_id must match page output doc_id.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="doc_id",
                details={"observed_value": raw_artifact.get("doc_id"), "expected_value": doc_id},
            )
        )

    if raw_artifact["page_index"] != page_index:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_page_index,
                message="artifact.page_index must match page output page_index.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="page_index",
                details={
                    "observed_value": raw_artifact.get("page_index"),
                    "expected_value": page_index,
                },
            )
        )

    if raw_artifact["artifact_type"] not in get_allowed_artifact_types():
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_enum_value,
                message="artifact_type is not allowed.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="artifact_type",
                details={"observed_value": raw_artifact.get("artifact_type")},
            )
        )

    if raw_artifact["modality"] not in get_allowed_modalities():
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_enum_value,
                message="modality is not allowed.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="modality",
                details={"observed_value": raw_artifact.get("modality")},
            )
        )

    if raw_artifact["validation_status"] not in get_allowed_validation_statuses():
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.invalid_validation_status,
                message="validation_status is not allowed.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="validation_status",
                details={"observed_value": raw_artifact.get("validation_status")},
            )
        )

    if not isinstance(raw_artifact["content"], str) or not raw_artifact["content"].strip():
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.empty_content,
                message="content must be a non-empty string.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="content",
                details={"observed_type": type(raw_artifact.get("content")).__name__},
            )
        )

    if not isinstance(raw_artifact["normalized_content"], dict):
        issues.append(_schema_type_issue(raw_artifact, doc_id, page_index, "normalized_content", "dict"))

    if not isinstance(raw_artifact["compiler_metadata"], dict):
        issues.append(_schema_type_issue(raw_artifact, doc_id, page_index, "compiler_metadata", "dict"))

    if not isinstance(raw_artifact["source_anchors"], list) or not raw_artifact["source_anchors"]:
        issues.append(
            ValidationIssue(
                error_type=ValidationErrorType.missing_source_anchor,
                message="source_anchors must be a non-empty list.",
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=artifact_id,
                field_path="source_anchors",
                details={"observed_type": type(raw_artifact.get("source_anchors")).__name__},
            )
        )

    provenance = raw_artifact["provenance"]
    if not isinstance(provenance, dict):
        issues.append(_schema_type_issue(raw_artifact, doc_id, page_index, "provenance", "dict"))
    else:
        if "op" not in provenance:
            issues.append(_missing_nested_issue(raw_artifact, doc_id, page_index, "provenance.op"))
        elif provenance["op"] not in get_allowed_provenance_ops():
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.invalid_enum_value,
                    message="provenance.op is not allowed.",
                    doc_id=doc_id,
                    page_index=page_index,
                    artifact_id=artifact_id,
                    field_path="provenance.op",
                    details={"observed_value": provenance.get("op")},
                )
            )
        if "sources" not in provenance:
            issues.append(_missing_nested_issue(raw_artifact, doc_id, page_index, "provenance.sources"))
        elif not isinstance(provenance["sources"], list) or not provenance["sources"]:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.schema_invalid,
                    message="provenance.sources must be a non-empty list.",
                    doc_id=doc_id,
                    page_index=page_index,
                    artifact_id=artifact_id,
                    field_path="provenance.sources",
                    details={"observed_type": type(provenance.get("sources")).__name__},
                )
            )

    return issues


def validate_source_anchors(
    raw_artifact: Dict[str, Any],
    layout_blocks: List[Dict[str, Any]],
) -> List[ValidationIssue]:
    """Validate artifact source anchors against available layout blocks."""

    issues: List[ValidationIssue] = []
    artifact_id = raw_artifact.get("artifact_id")
    doc_id = raw_artifact.get("doc_id")
    page_index = raw_artifact.get("page_index")
    valid_source_ids = _layout_block_ids(layout_blocks)
    source_anchors = raw_artifact.get("source_anchors")
    if not isinstance(source_anchors, list):
        return issues

    for anchor_index, anchor in enumerate(source_anchors):
        field_prefix = f"source_anchors[{anchor_index}]"
        if not isinstance(anchor, dict):
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.schema_invalid,
                    message="source_anchor must be a dict.",
                    doc_id=doc_id,
                    page_index=page_index,
                    artifact_id=artifact_id,
                    field_path=field_prefix,
                    details={"observed_type": type(anchor).__name__},
                )
            )
            continue

        source_id = anchor.get("source_id")
        if source_id not in valid_source_ids:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.source_anchor_not_found,
                    message="source_anchor.source_id was not found in layout_blocks.",
                    doc_id=doc_id,
                    page_index=page_index,
                    artifact_id=artifact_id,
                    field_path=f"{field_prefix}.source_id",
                    details={"source_id": source_id},
                )
            )
        if anchor.get("page_index") != page_index:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.invalid_page_index,
                    message="source_anchor.page_index must match artifact.page_index.",
                    doc_id=doc_id,
                    page_index=page_index,
                    artifact_id=artifact_id,
                    field_path=f"{field_prefix}.page_index",
                    details={
                        "observed_value": anchor.get("page_index"),
                        "expected_value": page_index,
                    },
                )
            )
        if anchor.get("anchor_type") not in get_allowed_anchor_types():
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.invalid_enum_value,
                    message="source_anchor.anchor_type is not allowed.",
                    doc_id=doc_id,
                    page_index=page_index,
                    artifact_id=artifact_id,
                    field_path=f"{field_prefix}.anchor_type",
                    details={"observed_value": anchor.get("anchor_type")},
                )
            )

    return issues


def validate_provenance(
    raw_artifact: Dict[str, Any],
    layout_blocks: List[Dict[str, Any]],
) -> List[ValidationIssue]:
    """Validate provenance source ids against available layout blocks."""

    provenance = raw_artifact.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("sources"), list):
        return []

    issues: List[ValidationIssue] = []
    valid_source_ids = _layout_block_ids(layout_blocks)
    for source_index, source_id in enumerate(provenance["sources"]):
        if source_id not in valid_source_ids:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.provenance_source_not_found,
                    message="provenance source id was not found in layout_blocks.",
                    doc_id=raw_artifact.get("doc_id"),
                    page_index=raw_artifact.get("page_index"),
                    artifact_id=raw_artifact.get("artifact_id"),
                    field_path=f"provenance.sources[{source_index}]",
                    details={"source_id": source_id},
                )
            )
    return issues


def detect_duplicate_artifacts(raw_artifacts: List[Dict[str, Any]]) -> List[ValidationIssue]:
    """Detect deterministic duplicate artifacts by page, type, content, anchors."""

    grouped_artifacts: Dict[Tuple[Any, Any, str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            continue
        key = (
            raw_artifact.get("page_index"),
            raw_artifact.get("artifact_type"),
            _normalize_content(raw_artifact.get("content")),
            _source_anchor_id_tuple(raw_artifact.get("source_anchors")),
        )
        grouped_artifacts[key].append(raw_artifact)

    issues: List[ValidationIssue] = []
    for duplicate_group in grouped_artifacts.values():
        if len(duplicate_group) <= 1:
            continue
        artifact_ids = [item.get("artifact_id") for item in duplicate_group]
        for raw_artifact in duplicate_group[1:]:
            issues.append(
                ValidationIssue(
                    error_type=ValidationErrorType.duplicate_artifact,
                    message="Duplicate artifact detected.",
                    doc_id=raw_artifact.get("doc_id"),
                    page_index=raw_artifact.get("page_index"),
                    artifact_id=raw_artifact.get("artifact_id"),
                    field_path="artifacts",
                    details={"duplicate_artifact_ids": artifact_ids},
                )
            )
    return issues


def validate_page_artifact_output(
    raw_output: Dict[str, Any],
    layout_blocks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[ValidationIssue]]:
    """Validate a page output and return anchored artifacts plus issues."""

    container_valid, issues = validate_page_artifact_output_schema(raw_output)
    if not container_valid:
        return [], issues

    doc_id = raw_output["doc_id"]
    page_index = raw_output["page_index"]
    artifact_issues_by_id: Dict[int, List[ValidationIssue]] = {}
    artifacts = raw_output.get("artifacts", [])

    for index, raw_artifact in enumerate(artifacts):
        artifact_issues = validate_evidence_artifact_schema(raw_artifact, doc_id, page_index)
        if not artifact_issues:
            artifact_issues.extend(validate_source_anchors(raw_artifact, layout_blocks))
            artifact_issues.extend(validate_provenance(raw_artifact, layout_blocks))
        artifact_issues_by_id[index] = artifact_issues
        issues.extend(artifact_issues)

    duplicate_issues = detect_duplicate_artifacts(artifacts)
    duplicate_artifact_ids = {issue.artifact_id for issue in duplicate_issues}
    issues.extend(duplicate_issues)

    valid_artifacts: List[Dict[str, Any]] = []
    for index, raw_artifact in enumerate(artifacts):
        if artifact_issues_by_id.get(index):
            continue
        if raw_artifact.get("artifact_id") in duplicate_artifact_ids:
            continue
        anchored_artifact = dict(raw_artifact)
        anchored_artifact["validation_status"] = "anchored"
        valid_artifacts.append(anchored_artifact)

    return valid_artifacts, issues


def _schema_type_issue(
    raw_artifact: Dict[str, Any],
    doc_id: str,
    page_index: int,
    field_path: str,
    expected_type: str,
) -> ValidationIssue:
    return ValidationIssue(
        error_type=ValidationErrorType.schema_invalid,
        message=f"{field_path} must be a {expected_type}.",
        doc_id=doc_id,
        page_index=page_index,
        artifact_id=raw_artifact.get("artifact_id"),
        field_path=field_path,
        details={"observed_type": type(raw_artifact.get(field_path)).__name__},
    )


def _missing_nested_issue(
    raw_artifact: Dict[str, Any],
    doc_id: str,
    page_index: int,
    field_path: str,
) -> ValidationIssue:
    return ValidationIssue(
        error_type=ValidationErrorType.missing_required_field,
        message=f"Missing required artifact field: {field_path}",
        doc_id=doc_id,
        page_index=page_index,
        artifact_id=raw_artifact.get("artifact_id"),
        field_path=field_path,
        details={"missing_field": field_path},
    )


def _layout_block_ids(layout_blocks: List[Dict[str, Any]]) -> Set[Any]:
    return {block.get("block_id") for block in layout_blocks if isinstance(block, dict)}


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    return " ".join(str(content).strip().lower().split())


def _source_anchor_id_tuple(source_anchors: Any) -> Tuple[str, ...]:
    if not isinstance(source_anchors, list):
        return tuple()
    source_ids = [anchor.get("source_id") for anchor in source_anchors if isinstance(anchor, dict)]
    return tuple(sorted(str(source_id) for source_id in source_ids))


def _safe_doc_id(raw_output: Dict[str, Any]) -> Any:
    if isinstance(raw_output, dict):
        return raw_output.get("doc_id")
    return None


def _safe_page_index(raw_output: Dict[str, Any]) -> Any:
    if isinstance(raw_output, dict) and _is_non_negative_int(raw_output.get("page_index")):
        return raw_output.get("page_index")
    return None


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
