"""Controlled Stage 2 compiler integration using fake client by default."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .api_config import ApiRunConfig, assert_real_api_allowed
from .artifact_compiler import compile_pages_with_client
from .artifact_store import build_document_artifact_store, write_artifact_store
from .compiler_client import ArtifactCompilerClient, FakeArtifactCompilerClient, RealArtifactCompilerClient
from .page_preparer import prepare_pages_for_compilation
from .provider_errors import ProviderError
from .raw_output_log import build_raw_output_log_entry, write_raw_output_log
from .real_api_client import RealApiArtifactCompilerClient
from .schema_serialization import build_page_artifact_output_schema_dict


FORBIDDEN_OUTPUT_TERMS = (
    "gold_annotation",
    "baseline_outputs",
    "source_record",
    "proof_trace",
    "verified",
    "answer_supported",
    "proof_used",
)


def run_stage2_compiler_dry_run(
    canonical_record: Dict[str, Any],
    extract_path: str | Path,
    output_path: str | Path,
    client: ArtifactCompilerClient | None = None,
    enable_real_api: bool = False,
) -> Dict[str, Any]:
    """Run controlled Stage 2 compiler flow with fake client by default."""

    active_client = client or FakeArtifactCompilerClient()
    if isinstance(active_client, RealArtifactCompilerClient) and not enable_real_api:
        active_client.enable_real_api = False

    compiler_metadata = _build_compiler_metadata(active_client, enable_real_api)
    prepared_result = prepare_pages_for_compilation(canonical_record, extract_path)
    schema_dict = build_page_artifact_output_schema_dict()
    compile_result = compile_pages_with_client(
        canonical_record=canonical_record,
        prepared_pages=prepared_result["pages"],
        client=active_client,
        schema_dict=schema_dict,
        compiler_metadata=compiler_metadata,
    )

    page_artifact_outputs = {
        page_index: result["raw_output"]
        for page_index, result in compile_result["page_results"].items()
    }
    validation_results = {
        page_index: {
            "valid_artifacts": result["valid_artifacts"],
            "validation_issues": result["validation_issues"],
        }
        for page_index, result in compile_result["page_results"].items()
    }
    store = build_document_artifact_store(
        canonical_record=canonical_record,
        prepared_pages=prepared_result["pages"],
        page_artifact_outputs=page_artifact_outputs,
        validation_results=validation_results,
        compiler_metadata=compiler_metadata,
    )
    write_artifact_store(store, output_path)

    return {
        "artifact_store_path": str(output_path),
        "num_pages_prepared": len(prepared_result["pages"]),
        "num_pages_with_errors": len(prepared_result["errors"]),
        "num_valid_artifacts": compile_result["summary"]["num_valid_artifacts"],
        "num_validation_issues": compile_result["summary"]["num_validation_issues"],
        "quality_gate": _build_quality_gate(prepared_result, store),
    }


def run_stage2_single_page_real_api_smoke_test(
    canonical_record: Dict[str, Any],
    extract_path: str | Path,
    output_path: str | Path,
    api_config: ApiRunConfig,
    target_page_index: int | None = None,
    run_real_trial: bool = False,
) -> Dict[str, Any]:
    """Run one guarded real-provider smoke-test page through validation and store writing."""

    if not run_real_trial:
        raise RuntimeError("Real provider trial requires run_real_trial=True.")
    assert_real_api_allowed(api_config)
    prepared_result = prepare_pages_for_compilation(canonical_record, extract_path)
    selected_page_index = _select_target_page_index(canonical_record, target_page_index)
    selected_pages = [
        page for page in prepared_result["pages"] if int(page["page_index"]) == selected_page_index
    ][:1]

    raw_output_log_path = _default_log_path(
        api_config.raw_output_dir,
        output_path,
        f"raw_outputs_p{selected_page_index:03d}.jsonl",
    )
    discard_log_path = _default_log_path(
        api_config.discard_log_dir,
        output_path,
        f"discard_p{selected_page_index:03d}.jsonl",
    )
    compiler_metadata = _real_api_compiler_metadata(api_config)

    schema_dict = build_page_artifact_output_schema_dict()
    client = RealApiArtifactCompilerClient(api_config)
    try:
        compile_result = compile_pages_with_client(
            canonical_record=canonical_record,
            prepared_pages=selected_pages,
            client=client,
            schema_dict=schema_dict,
            compiler_metadata=compiler_metadata,
            raw_output_log_path=raw_output_log_path,
            discard_log_path=discard_log_path,
            compiler_version="stage2_compiler_v1",
            prompt_version="artifact_compiler_prompt_v1",
        )
        provider_error = None
    except Exception as exc:
        if not isinstance(exc, (ProviderError, NotImplementedError, RuntimeError)):
            raise
        compile_result = {"page_results": {}, "summary": {"num_valid_artifacts": 0, "num_validation_issues": 0}}
        provider_error = exc
        if selected_pages:
            _write_provider_error_raw_log(
                raw_output_log_path=raw_output_log_path,
                page_input=selected_pages[0],
                api_config=api_config,
                provider_error=exc,
            )

    page_artifact_outputs = {
        page_index: result["raw_output"]
        for page_index, result in compile_result["page_results"].items()
    }
    validation_results = {
        page_index: {
            "valid_artifacts": result["valid_artifacts"],
            "validation_issues": result["validation_issues"],
        }
        for page_index, result in compile_result["page_results"].items()
    }
    store = build_document_artifact_store(
        canonical_record=canonical_record,
        prepared_pages=selected_pages,
        page_artifact_outputs=page_artifact_outputs,
        validation_results=validation_results,
        compiler_metadata=compiler_metadata,
    )
    write_artifact_store(store, output_path)
    quality_gate = _build_single_page_quality_gate(selected_pages, compile_result, store, provider_error)

    return {
        "artifact_store_path": str(output_path),
        "target_page_index": selected_page_index,
        "num_pages_compiled": len(selected_pages),
        "num_raw_artifacts": sum(
            result["compilation_statistics"]["num_raw_artifacts"]
            for result in compile_result["page_results"].values()
        ),
        "num_valid_artifacts": compile_result["summary"]["num_valid_artifacts"],
        "num_validation_issues": compile_result["summary"]["num_validation_issues"],
        "raw_output_log_path": str(raw_output_log_path),
        "discard_log_path": str(discard_log_path),
        "quality_gate": quality_gate,
    }


def _build_compiler_metadata(
    client: ArtifactCompilerClient,
    enable_real_api: bool,
) -> Dict[str, Any]:
    if isinstance(client, FakeArtifactCompilerClient):
        return {
            "compiler_name": "fake_artifact_compiler_client",
            "compiler_version": "stage2_step5_fake",
            "schema_version": "stage2_artifact_schema_v1",
            "model_name": "mock",
            "temperature": None,
            "max_repair_attempts": 0,
        }
    return {
        "compiler_name": "real_artifact_compiler_client",
        "compiler_version": "stage2_step5_interface_only",
        "schema_version": "stage2_artifact_schema_v1",
        "model_name": "real_api_enabled" if enable_real_api else "real_api_disabled",
        "temperature": None,
        "max_repair_attempts": 0,
    }


def _real_api_compiler_metadata(api_config: ApiRunConfig) -> Dict[str, Any]:
    return {
        "compiler_name": "real_api_artifact_compiler_client",
        "compiler_version": "stage2_step7_real_provider_adapter",
        "schema_version": "stage2_artifact_schema_v1",
        "provider": api_config.provider,
        "model_name": api_config.model_name,
        "temperature": api_config.temperature,
        "max_repair_attempts": 0,
    }


def _build_quality_gate(
    prepared_result: Dict[str, Any],
    store: Dict[str, Any],
) -> Dict[str, Any]:
    blocking_reasons = []
    if not prepared_result["pages"]:
        blocking_reasons.append("no_prepared_pages")
    if prepared_result["errors"]:
        blocking_reasons.append("missing_source_anchors")
    if store.get("compilation_statistics", {}).get("num_artifacts", 0) == 0:
        blocking_reasons.append("no_valid_artifacts")
    if _contains_forbidden_output(store):
        blocking_reasons.append("forbidden_fields_present")
    return {
        "stage2_compiler_dry_run_passed": not blocking_reasons,
        "blocking_reasons": sorted(set(blocking_reasons)),
    }


def _build_single_page_quality_gate(
    selected_pages: list[Dict[str, Any]],
    compile_result: Dict[str, Any],
    store: Dict[str, Any],
    provider_error: Exception | None = None,
) -> Dict[str, Any]:
    blocking_reasons = []
    if provider_error is not None:
        blocking_reasons.append("provider_error")
    if len(selected_pages) != 1:
        blocking_reasons.append("single_page_selection_failed")
    if selected_pages and not selected_pages[0].get("layout_blocks"):
        blocking_reasons.append("missing_source_anchors")
    raw_outputs = [result.get("raw_output") for result in compile_result["page_results"].values()]
    if any(not isinstance(raw_output, dict) for raw_output in raw_outputs):
        blocking_reasons.append("invalid_raw_output_container")
    if compile_result["summary"]["num_valid_artifacts"] == 0:
        blocking_reasons.append("no_valid_artifacts")
    if _contains_forbidden_output(store):
        blocking_reasons.append("forbidden_fields_present")
    return {
        "single_page_smoke_test_passed": not blocking_reasons,
        "blocking_reasons": sorted(set(blocking_reasons)),
    }


def _write_provider_error_raw_log(
    raw_output_log_path: Path,
    page_input: Dict[str, Any],
    api_config: ApiRunConfig,
    provider_error: Exception,
) -> None:
    raw_text = getattr(provider_error, "raw_text", None)
    raw_output = {
        "provider_error": {
            "type": type(provider_error).__name__,
            "message": str(provider_error),
        }
    }
    write_raw_output_log(
        raw_output_log_path,
        build_raw_output_log_entry(
            doc_id=page_input["doc_id"],
            page_index=int(page_input["page_index"]),
            provider=api_config.provider,
            model_name=api_config.model_name,
            compiler_version="stage2_compiler_v1",
            prompt_version="artifact_compiler_prompt_v1",
            raw_output=raw_output,
            raw_text=raw_text,
            provider_error_type=type(provider_error).__name__,
            provider_error_message=str(provider_error),
        ),
    )


def _contains_forbidden_output(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_OUTPUT_TERMS:
                return True
            if _contains_forbidden_output(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_output(item) for item in value)
    elif isinstance(value, str):
        return value in FORBIDDEN_OUTPUT_TERMS
    return False


def _select_target_page_index(
    canonical_record: Dict[str, Any],
    target_page_index: int | None,
) -> int:
    if target_page_index is not None:
        return int(target_page_index)
    explicit_pages = canonical_record.get("candidate_pool", {}).get("explicit_constraint_pages", [])
    if explicit_pages:
        return int(explicit_pages[0])
    return int(canonical_record["compilation_plan"]["pages_to_compile"][0])


def _default_log_path(
    configured_dir: str | Path | None,
    output_path: str | Path,
    file_name: str,
) -> Path:
    if configured_dir is not None:
        return Path(configured_dir) / file_name
    return Path(output_path).parent / file_name
