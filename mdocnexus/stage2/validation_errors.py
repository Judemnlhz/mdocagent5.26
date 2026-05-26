"""Validation issue types for deterministic Stage 2 artifact checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ValidationErrorType(str, Enum):
    schema_invalid = "schema_invalid"
    missing_required_field = "missing_required_field"
    invalid_enum_value = "invalid_enum_value"
    invalid_page_index = "invalid_page_index"
    empty_content = "empty_content"
    missing_source_anchor = "missing_source_anchor"
    source_anchor_not_found = "source_anchor_not_found"
    provenance_source_not_found = "provenance_source_not_found"
    duplicate_artifact = "duplicate_artifact"
    invalid_validation_status = "invalid_validation_status"
    invalid_artifact_id = "invalid_artifact_id"
    invalid_output_container = "invalid_output_container"


@dataclass
class ValidationIssue:
    error_type: ValidationErrorType
    message: str
    doc_id: Optional[str] = None
    page_index: Optional[int] = None
    artifact_id: Optional[str] = None
    field_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.error_type, ValidationErrorType):
            self.error_type = ValidationErrorType(self.error_type)
        if self.details is None:
            self.details = {}
        if not isinstance(self.details, dict):
            raise ValueError("ValidationIssue.details must be a dict")

    def to_dict(self) -> Dict[str, Any]:
        issue_dict = asdict(self)
        issue_dict["error_type"] = self.error_type.value
        return issue_dict
