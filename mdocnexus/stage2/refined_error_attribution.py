"""Offline attribution for refined Stage 2 validation failures."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .artifact_deduplicator import build_artifact_dedup_key


DEDUP_BLOCKING_ISSUES = {
    "source_anchor_not_found",
    "provenance_source_not_found",
    "forbidden_field",
    "missing_required_field",
}


def summarize_refined_validation_failures(
    discard_log_path: str | Path,
    raw_output_log_path: str | Path,
) -> dict:
    """Summarize refined validation failures and whether deterministic dedup applies."""

    discard_entries = _read_jsonl(discard_log_path)
    raw_entries = _read_jsonl(raw_output_log_path)
    issue_types = Counter(str(entry.get("error_type", "unknown")) for entry in discard_entries)
    affected_pages = sorted({_page_ref(entry.get("doc_id"), entry.get("page_index")) for entry in discard_entries})
    duplicate_pages = sorted(
        {
            _page_ref(entry.get("doc_id"), entry.get("page_index"))
            for entry in discard_entries
            if entry.get("error_type") == "duplicate_artifact"
        }
    )
    same_artifact_id_duplicates = _count_same_artifact_id_duplicates(raw_entries)
    same_type_anchor_content_duplicates = _count_same_type_anchor_content_duplicates(raw_entries)
    non_dedup_blocking = sorted(issue for issue in issue_types if issue in DEDUP_BLOCKING_ISSUES)
    duplicate_count = int(issue_types.get("duplicate_artifact", 0))
    deduplication_applicable = (
        duplicate_count > 0
        and not non_dedup_blocking
        and duplicate_count == len(discard_entries)
        and (same_artifact_id_duplicates > 0 or same_type_anchor_content_duplicates > 0)
    )

    return {
        "num_validation_issues": len(discard_entries),
        "issue_types": dict(sorted(issue_types.items())),
        "affected_pages": affected_pages,
        "duplicate_artifact": {
            "count": duplicate_count,
            "affected_pages": duplicate_pages,
            "same_artifact_id_duplicates": same_artifact_id_duplicates,
            "same_type_anchor_content_duplicates": same_type_anchor_content_duplicates,
            "deduplication_applicable": bool(deduplication_applicable),
        },
        "non_dedup_blocking_issues": non_dedup_blocking,
    }


def _read_jsonl(path: str | Path) -> list[dict]:
    entries = []
    log_path = Path(path)
    if not log_path.is_file():
        return entries
    with log_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                entries.append(loaded)
    return entries


def _count_same_artifact_id_duplicates(raw_entries: list[dict]) -> int:
    count = 0
    for artifacts in _iter_page_artifacts(raw_entries):
        artifact_ids = [artifact.get("artifact_id") for artifact in artifacts if isinstance(artifact, dict)]
        counts = Counter(artifact_ids)
        count += sum(value - 1 for value in counts.values() if value > 1)
    return count


def _count_same_type_anchor_content_duplicates(raw_entries: list[dict]) -> int:
    count = 0
    for artifacts in _iter_page_artifacts(raw_entries):
        keys = [build_artifact_dedup_key(artifact) for artifact in artifacts if isinstance(artifact, dict)]
        counts = Counter(keys)
        count += sum(value - 1 for value in counts.values() if value > 1)
    return count


def _iter_page_artifacts(raw_entries: list[dict]) -> Any:
    for entry in raw_entries:
        raw_output = entry.get("raw_output")
        if not isinstance(raw_output, dict):
            continue
        artifacts = raw_output.get("artifacts", [])
        if isinstance(artifacts, list):
            yield artifacts


def _page_ref(doc_id: Any, page_index: Any) -> str:
    return f"{doc_id}#p{page_index}"
