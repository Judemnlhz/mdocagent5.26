"""Discard log structures and JSONL serialization helpers for Stage 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .validation_errors import ValidationIssue


@dataclass
class DiscardLogEntry:
    doc_id: Optional[str]
    page_index: Optional[int]
    artifact_id: Optional[str]
    error_type: str
    message: str
    field_path: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)
    stage: str = "stage2_validation"
    compiler_version: str = "unversioned"

    def to_dict(self) -> Dict[str, Any]:
        return serialize_discard_log_entry(self)


def issue_to_discard_log_entry(
    issue: ValidationIssue,
    stage: str,
    compiler_version: str,
) -> DiscardLogEntry:
    return DiscardLogEntry(
        doc_id=issue.doc_id,
        page_index=issue.page_index,
        artifact_id=issue.artifact_id,
        error_type=issue.error_type.value,
        message=issue.message,
        field_path=issue.field_path,
        details=dict(issue.details),
        stage=stage,
        compiler_version=compiler_version,
    )


def serialize_discard_log_entry(entry: DiscardLogEntry) -> Dict[str, Any]:
    return asdict(entry)


def write_discard_log_entry(path: str | Path, entry: DiscardLogEntry) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(serialize_discard_log_entry(entry), ensure_ascii=False))
        file_obj.write("\n")
