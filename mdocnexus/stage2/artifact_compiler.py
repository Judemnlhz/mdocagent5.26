"""Controlled Stage 2 artifact compiler orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .artifact_deduplicator import deduplicate_page_artifacts
from .artifact_validator import validate_page_artifact_output
from .compiler_client import ArtifactCompilerClient
from .compiler_prompt import (
    build_artifact_compiler_system_prompt,
    build_artifact_compiler_user_prompt,
)
from .discard_log import DiscardLogEntry, issue_to_discard_log_entry, write_discard_log_entry
from .raw_output_log import build_raw_output_log_entry, write_raw_output_log


def compile_page_with_client(
    canonical_record: Dict[str, Any],
    page_input: Dict[str, Any],
    client: ArtifactCompilerClient,
    schema_dict: Dict[str, Any],
    compiler_metadata: Dict[str, Any],
    raw_output_log_path: str | Path | None = None,
    discard_log_path: str | Path | None = None,
    compiler_version: str = "stage2_compiler_v1",
    prompt_version: str = "artifact_compiler_prompt_v1",
    deterministic_dedup_enabled: bool = True,
) -> Dict[str, Any]:
    """Compile one page with a client, then deterministically validate output."""

    system_prompt = build_artifact_compiler_system_prompt()
    user_prompt = build_artifact_compiler_user_prompt(
        canonical_record=canonical_record,
        page_input=page_input,
        schema_dict=schema_dict,
    )
    raw_output = client.generate_page_artifacts(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_dict=schema_dict,
    )
    if raw_output_log_path is not None:
        write_raw_output_log(
            raw_output_log_path,
            build_raw_output_log_entry(
                doc_id=page_input["doc_id"],
                page_index=int(page_input["page_index"]),
                provider=str(compiler_metadata.get("provider", compiler_metadata.get("compiler_name", "unknown"))),
                model_name=compiler_metadata.get("model_name"),
                compiler_version=compiler_version,
                prompt_version=prompt_version,
                raw_output=raw_output if isinstance(raw_output, dict) else {"raw_output": raw_output},
            ),
        )

    num_raw_artifacts_before_dedup = _count_raw_artifacts(raw_output)
    dedup_removed: List[Dict[str, Any]] = []
    validation_input = raw_output
    if deterministic_dedup_enabled and isinstance(raw_output, dict):
        validation_input, dedup_removed = deduplicate_page_artifacts(raw_output)
        if discard_log_path is not None:
            for removed in dedup_removed:
                write_discard_log_entry(
                    discard_log_path,
                    DiscardLogEntry(
                        doc_id=page_input["doc_id"],
                        page_index=int(page_input["page_index"]),
                        artifact_id=removed.get("artifact_id"),
                        error_type="duplicate_artifact_deduplicated",
                        message="Duplicate artifact removed before deterministic validation.",
                        field_path="artifacts",
                        details={
                            "duplicate_of": removed.get("duplicate_of"),
                            "dedup_key": removed.get("dedup_key"),
                            "dedup_rule": "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash",
                        },
                        stage="stage2_compiler_deterministic_dedup",
                        compiler_version=compiler_version,
                    ),
                )

    valid_artifacts, validation_issues = validate_page_artifact_output(
        raw_output=validation_input,
        layout_blocks=page_input.get("layout_blocks", []),
    )
    if discard_log_path is not None:
        for issue in validation_issues:
            write_discard_log_entry(
                discard_log_path,
                issue_to_discard_log_entry(
                    issue,
                    stage="stage2_compiler_validation",
                    compiler_version=compiler_version,
                ),
            )

    return {
        "page_index": page_input["page_index"],
        "raw_output": validation_input,
        "raw_output_before_dedup": raw_output,
        "valid_artifacts": valid_artifacts,
        "validation_issues": [issue.to_dict() for issue in validation_issues],
        "compilation_statistics": {
            "deterministic_dedup_enabled": bool(deterministic_dedup_enabled),
            "dedup_stage": "after_raw_output_log_before_validation" if deterministic_dedup_enabled else None,
            "dedup_rule": (
                "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash"
                if deterministic_dedup_enabled
                else None
            ),
            "num_raw_artifacts_before_dedup": num_raw_artifacts_before_dedup,
            "num_deduplicated_artifacts": len(dedup_removed),
            "deduplicated_artifact_issue_type_count": len(dedup_removed),
            "num_raw_artifacts": _count_raw_artifacts(validation_input),
            "num_valid_artifacts": len(valid_artifacts),
            "num_validation_issues": len(validation_issues),
            "schema_valid_rate_before_dedup": _rate(len(valid_artifacts), num_raw_artifacts_before_dedup),
            "schema_valid_rate_after_dedup": _rate(len(valid_artifacts), _count_raw_artifacts(validation_input)),
            "discard_rate_before_dedup": _discard_rate(num_raw_artifacts_before_dedup, len(valid_artifacts)),
            "discard_rate_after_dedup": _discard_rate(_count_raw_artifacts(validation_input), len(valid_artifacts)),
        },
    }


def compile_pages_with_client(
    canonical_record: Dict[str, Any],
    prepared_pages: List[Dict[str, Any]],
    client: ArtifactCompilerClient,
    schema_dict: Dict[str, Any],
    compiler_metadata: Dict[str, Any],
    raw_output_log_path: str | Path | None = None,
    discard_log_path: str | Path | None = None,
    compiler_version: str = "stage2_compiler_v1",
    prompt_version: str = "artifact_compiler_prompt_v1",
    deterministic_dedup_enabled: bool = True,
) -> Dict[str, Any]:
    """Compile prepared pages with a client without retry or repair."""

    page_results: Dict[int, Dict[str, Any]] = {}
    for page_input in prepared_pages:
        page_index = int(page_input["page_index"])
        page_results[page_index] = compile_page_with_client(
            canonical_record=canonical_record,
            page_input=page_input,
            client=client,
            schema_dict=schema_dict,
            compiler_metadata=compiler_metadata,
            raw_output_log_path=raw_output_log_path,
            discard_log_path=discard_log_path,
            compiler_version=compiler_version,
            prompt_version=prompt_version,
            deterministic_dedup_enabled=deterministic_dedup_enabled,
        )

    stats = [result["compilation_statistics"] for result in page_results.values()]
    num_raw_artifacts = sum(int(stat.get("num_raw_artifacts", 0)) for stat in stats)
    num_raw_artifacts_before_dedup = sum(
        int(stat.get("num_raw_artifacts_before_dedup", stat.get("num_raw_artifacts", 0)))
        for stat in stats
    )
    num_deduplicated_artifacts = sum(int(stat.get("num_deduplicated_artifacts", 0)) for stat in stats)
    num_valid_artifacts = sum(len(result["valid_artifacts"]) for result in page_results.values())
    num_validation_issues = sum(len(result["validation_issues"]) for result in page_results.values())

    return {
        "page_results": page_results,
        "summary": {
            "num_pages": len(prepared_pages),
            "deterministic_dedup_enabled": bool(deterministic_dedup_enabled),
            "dedup_stage": "after_raw_output_log_before_validation" if deterministic_dedup_enabled else None,
            "dedup_rule": (
                "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash"
                if deterministic_dedup_enabled
                else None
            ),
            "num_raw_artifacts_before_dedup": num_raw_artifacts_before_dedup,
            "num_deduplicated_artifacts": num_deduplicated_artifacts,
            "deduplicated_artifact_issue_type_count": num_deduplicated_artifacts,
            "num_raw_artifacts": num_raw_artifacts,
            "num_valid_artifacts": num_valid_artifacts,
            "num_validation_issues": num_validation_issues,
            "schema_valid_rate_before_dedup": _rate(num_valid_artifacts, num_raw_artifacts_before_dedup),
            "schema_valid_rate_after_dedup": _rate(num_valid_artifacts, num_raw_artifacts),
            "discard_rate_before_dedup": _discard_rate(num_raw_artifacts_before_dedup, num_valid_artifacts),
            "discard_rate_after_dedup": _discard_rate(num_raw_artifacts, num_valid_artifacts),
        },
    }


def _count_raw_artifacts(raw_output: Any) -> int:
    if isinstance(raw_output, dict) and isinstance(raw_output.get("artifacts"), list):
        return len(raw_output["artifacts"])
    return 0


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(max(1, denominator))


def _discard_rate(num_raw_artifacts: int, num_valid_artifacts: int) -> float:
    return max(0, num_raw_artifacts - num_valid_artifacts) / float(max(1, num_raw_artifacts))
