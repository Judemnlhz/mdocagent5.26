"""Document-level artifact store construction for Stage 2 dry runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_COMPILER_METADATA = {
    "compiler_name": "mock_stage2_dry_run",
    "compiler_version": "mock_stage2_dry_run",
    "schema_version": "stage2_artifact_schema_v1",
    "model_name": "mock",
    "temperature": None,
    "max_repair_attempts": 0,
}


def build_artifact_index(pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """Build deterministic artifact indexes by type and page index."""

    by_artifact_type: Dict[str, List[str]] = {}
    by_page_index: Dict[str, List[str]] = {}
    for page in pages:
        page_index_key = str(page["page_index"])
        for artifact in page.get("artifacts", []):
            artifact_id = artifact["artifact_id"]
            artifact_type = artifact["artifact_type"]
            by_artifact_type.setdefault(artifact_type, []).append(artifact_id)
            by_page_index.setdefault(page_index_key, []).append(artifact_id)

    return {
        "by_artifact_type": _sort_index_values(by_artifact_type),
        "by_page_index": _sort_index_values(by_page_index),
    }


def compute_compilation_statistics(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute document-level Stage 2 compilation statistics."""

    num_pages_compiled = len(pages)
    num_pages_with_layout_blocks = sum(1 for page in pages if page.get("layout_blocks"))
    num_pages_with_artifacts = sum(1 for page in pages if page.get("artifacts"))
    num_artifacts = sum(len(page.get("artifacts", [])) for page in pages)
    num_raw_artifacts = sum(
        page.get("page_compilation_statistics", {}).get("num_raw_artifacts", len(page.get("artifacts", [])))
        for page in pages
    )
    num_schema_valid_artifacts = sum(
        page.get("page_compilation_statistics", {}).get("num_schema_valid_artifacts", 0)
        for page in pages
    )
    num_anchored_artifacts = sum(
        page.get("page_compilation_statistics", {}).get("num_anchored_artifacts", 0)
        for page in pages
    )
    num_discarded_artifacts = sum(
        page.get("page_compilation_statistics", {}).get("num_discarded_artifacts", 0)
        for page in pages
    )
    num_explicit_constraint_pages_compiled = sum(
        1 for page in pages if page.get("_is_explicit_constraint_page", False)
    )
    num_retrieval_missed_explicit_pages_compiled = sum(
        1 for page in pages if page.get("_is_retrieval_missed_explicit_page", False)
    )

    denominator = max(1, num_raw_artifacts)
    return {
        "num_pages_compiled": num_pages_compiled,
        "num_pages_with_layout_blocks": num_pages_with_layout_blocks,
        "num_pages_with_artifacts": num_pages_with_artifacts,
        "num_artifacts": num_artifacts,
        "num_schema_valid_artifacts": num_schema_valid_artifacts,
        "num_anchored_artifacts": num_anchored_artifacts,
        "num_discarded_artifacts": num_discarded_artifacts,
        "schema_valid_rate": num_schema_valid_artifacts / denominator,
        "anchoring_rate": num_anchored_artifacts / denominator,
        "discard_rate": num_discarded_artifacts / denominator,
        "num_explicit_constraint_pages_compiled": num_explicit_constraint_pages_compiled,
        "num_retrieval_missed_explicit_pages_compiled": num_retrieval_missed_explicit_pages_compiled,
    }


def build_document_artifact_store(
    canonical_record: Dict[str, Any],
    prepared_pages: List[Dict[str, Any]],
    page_artifact_outputs: Dict[int, Dict[str, Any]],
    validation_results: Dict[int, Dict[str, Any]],
    compiler_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a provenance-preserving document artifact store from valid artifacts."""

    doc_id = canonical_record["document"]["doc_id"]
    candidate_pool = canonical_record.get("candidate_pool", {})
    explicit_pages = set(candidate_pool.get("explicit_constraint_pages", []))
    missed_explicit_pages = set(candidate_pool.get("retrieval_missed_explicit_pages", []))
    compiler = _build_compiler_metadata(compiler_metadata)

    store_pages: List[Dict[str, Any]] = []
    for prepared_page in prepared_pages:
        page_index = int(prepared_page["page_index"])
        raw_output = page_artifact_outputs.get(page_index, {})
        validation_result = validation_results.get(page_index, {})
        valid_artifacts = list(validation_result.get("valid_artifacts", []))
        num_raw_artifacts = len(raw_output.get("artifacts", [])) if isinstance(raw_output, dict) else 0
        num_schema_valid_artifacts = len(valid_artifacts)
        num_anchored_artifacts = sum(
            1 for artifact in valid_artifacts if artifact.get("validation_status") == "anchored"
        )
        num_discarded_artifacts = max(0, num_raw_artifacts - len(valid_artifacts))

        store_pages.append(
            {
                "page_index": page_index,
                "page_number_one_based": page_index + 1,
                "page_source": {
                    "page_text_path": prepared_page.get("page_text_path"),
                    "page_image_path": prepared_page.get("page_image_path"),
                },
                "layout_blocks": prepared_page.get("layout_blocks", []),
                "artifacts": valid_artifacts,
                "page_compilation_statistics": {
                    "num_raw_artifacts": num_raw_artifacts,
                    "num_schema_valid_artifacts": num_schema_valid_artifacts,
                    "num_anchored_artifacts": num_anchored_artifacts,
                    "num_discarded_artifacts": num_discarded_artifacts,
                },
                "_is_explicit_constraint_page": page_index in explicit_pages,
                "_is_retrieval_missed_explicit_page": page_index in missed_explicit_pages,
            }
        )

    compilation_statistics = compute_compilation_statistics(store_pages)
    for page in store_pages:
        page.pop("_is_explicit_constraint_page", None)
        page.pop("_is_retrieval_missed_explicit_page", None)

    return {
        "document": {
            "doc_id": doc_id,
            "dataset": canonical_record["document"].get("dataset"),
            "page_index_base": 0,
        },
        "compiler": compiler,
        "pages": store_pages,
        "artifact_index": build_artifact_index(store_pages),
        "compilation_statistics": compilation_statistics,
    }


def write_artifact_store(store: Dict[str, Any], output_path: str | Path) -> None:
    """Write an artifact store as pretty JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_compiler_metadata(compiler_metadata: Dict[str, Any]) -> Dict[str, Any]:
    compiler = dict(DEFAULT_COMPILER_METADATA)
    compiler.update(compiler_metadata or {})
    return {
        "compiler_name": compiler.get("compiler_name"),
        "compiler_version": compiler.get("compiler_version"),
        "schema_version": compiler.get("schema_version"),
        "model_name": compiler.get("model_name", "mock"),
        "temperature": compiler.get("temperature"),
        "max_repair_attempts": compiler.get("max_repair_attempts", 0),
    }


def _sort_index_values(index: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {key: sorted(value) for key, value in sorted(index.items(), key=lambda item: item[0])}
