"""Controlled Stage 2 artifact compiler orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .artifact_validator import validate_page_artifact_output
from .compiler_client import ArtifactCompilerClient
from .compiler_prompt import (
    build_artifact_compiler_system_prompt,
    build_artifact_compiler_user_prompt,
)
from .discard_log import issue_to_discard_log_entry, write_discard_log_entry
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

    valid_artifacts, validation_issues = validate_page_artifact_output(
        raw_output=raw_output,
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
        "raw_output": raw_output,
        "valid_artifacts": valid_artifacts,
        "validation_issues": [issue.to_dict() for issue in validation_issues],
        "compilation_statistics": {
            "num_raw_artifacts": _count_raw_artifacts(raw_output),
            "num_valid_artifacts": len(valid_artifacts),
            "num_validation_issues": len(validation_issues),
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
        )

    return {
        "page_results": page_results,
        "summary": {
            "num_pages": len(prepared_pages),
            "num_valid_artifacts": sum(
                len(result["valid_artifacts"]) for result in page_results.values()
            ),
            "num_validation_issues": sum(
                len(result["validation_issues"]) for result in page_results.values()
            ),
        },
    }


def _count_raw_artifacts(raw_output: Any) -> int:
    if isinstance(raw_output, dict) and isinstance(raw_output.get("artifacts"), list):
        return len(raw_output["artifacts"])
    return 0
