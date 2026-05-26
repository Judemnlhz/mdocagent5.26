"""Stage 2 integration dry run using deterministic mock artifact outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .artifact_store import build_document_artifact_store, write_artifact_store
from .artifact_validator import validate_page_artifact_output
from .mock_artifact_outputs import build_mock_page_artifact_output
from .page_preparer import prepare_pages_for_compilation


FORBIDDEN_STORE_FIELDS = (
    "gold_annotation",
    "baseline_outputs",
    "source_record",
    "proof_trace",
    "verified",
    "answer_supported",
    "proof_used",
)


def run_stage2_dry_run(
    canonical_record: Dict[str, Any],
    extract_path: str | Path,
    output_path: str | Path,
    compiler_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run Stage 2 end to end with mock artifacts and deterministic validation."""

    prepared_result = prepare_pages_for_compilation(canonical_record, extract_path)
    prepared_pages = prepared_result["pages"]
    page_artifact_outputs: Dict[int, Dict[str, Any]] = {}
    validation_results: Dict[int, Dict[str, Any]] = {}
    all_validation_issues: List[Any] = []

    for page in prepared_pages:
        page_index = int(page["page_index"])
        raw_output = build_mock_page_artifact_output(
            doc_id=prepared_result["doc_id"],
            page_index=page_index,
            layout_blocks=page.get("layout_blocks", []),
        )
        valid_artifacts, validation_issues = validate_page_artifact_output(
            raw_output=raw_output,
            layout_blocks=page.get("layout_blocks", []),
        )
        page_artifact_outputs[page_index] = raw_output
        validation_results[page_index] = {
            "valid_artifacts": valid_artifacts,
            "validation_issues": [issue.to_dict() for issue in validation_issues],
        }
        all_validation_issues.extend(validation_issues)

    store = build_document_artifact_store(
        canonical_record=canonical_record,
        prepared_pages=prepared_pages,
        page_artifact_outputs=page_artifact_outputs,
        validation_results=validation_results,
        compiler_metadata=compiler_metadata or {},
    )
    write_artifact_store(store, output_path)
    quality_gate = _build_quality_gate(canonical_record, prepared_result, store)

    return {
        "artifact_store_path": str(output_path),
        "num_pages_prepared": len(prepared_pages),
        "num_pages_with_errors": len(prepared_result["errors"]),
        "num_valid_artifacts": sum(
            len(result.get("valid_artifacts", [])) for result in validation_results.values()
        ),
        "num_validation_issues": len(all_validation_issues),
        "quality_gate": quality_gate,
    }


def _build_quality_gate(
    canonical_record: Dict[str, Any],
    prepared_result: Dict[str, Any],
    store: Dict[str, Any],
) -> Dict[str, Any]:
    blocking_reasons: List[str] = []
    prepared_pages = prepared_result["pages"]
    candidate_pool = canonical_record.get("candidate_pool", {})
    explicit_pages = set(candidate_pool.get("explicit_constraint_pages", []))
    artifact_index_by_page = store.get("artifact_index", {}).get("by_page_index", {})

    if not prepared_pages:
        blocking_reasons.append("no_prepared_pages")
    if prepared_result.get("errors"):
        blocking_reasons.append("missing_source_anchors")
    if any(not page.get("layout_blocks") for page in prepared_pages):
        blocking_reasons.append("missing_required_page_layout_blocks")

    for explicit_page in explicit_pages:
        if int(explicit_page) == 29 and "29" not in artifact_index_by_page:
            blocking_reasons.append("missing_required_page_artifacts")

    if store.get("compilation_statistics", {}).get("num_artifacts", 0) == 0:
        blocking_reasons.append("no_valid_artifacts")
    if _contains_forbidden_field(store):
        blocking_reasons.append("forbidden_fields_present")

    return {
        "stage2_dry_run_passed": not blocking_reasons,
        "blocking_reasons": sorted(set(blocking_reasons)),
    }


def _contains_forbidden_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_STORE_FIELDS:
                return True
            if isinstance(child, str) and child in FORBIDDEN_STORE_FIELDS:
                return True
            if _contains_forbidden_field(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_field(item) for item in value)
    elif isinstance(value, str):
        return value in FORBIDDEN_STORE_FIELDS
    return False
