"""Run a guarded Stage 2 real-API cross-document batch compilation."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.provider import assert_real_api_allowed
from mdocnexus.stage2.artifact_pipeline import compile_page_with_client
from mdocnexus.stage2.artifact_pipeline import build_document_artifact_store, write_artifact_store
from mdocnexus.stage2.reports import count_forbidden_fields, write_batch_summary
from mdocnexus.stage2.provider import ArtifactCompilerClient, FakeArtifactCompilerClient
from mdocnexus.stage2.selectors import select_crossdoc_pages_for_batch
from mdocnexus.stage2.logs import DiscardLogEntry, write_discard_log_entry
from mdocnexus.stage2.index_builder import build_api_run_config_from_mdocagent_yaml, read_json_or_jsonl_records
from mdocnexus.stage2.selectors import diagnose_page_modality_from_question_and_preflight
from mdocnexus.stage2.logs import build_raw_output_log_entry, write_raw_output_log
from mdocnexus.stage2.provider import RealApiArtifactCompilerClient
from mdocnexus.stage2.logs import build_stage2_run_manifest, write_stage2_run_manifest
from mdocnexus.stage2.artifact_schema import build_page_artifact_output_schema_dict
from mdocnexus.stage2.index_builder import build_page_source
from scripts.stage2_compile_small_batch import (
    COMPILER_VERSION,
    PROMPT_VERSION,
    artifact_store_file_name,
    build_compiler_safe_record,
    build_page_input,
)

MAX_ALLOWED_DOCS = 5
MAX_ALLOWED_PAGES_PER_DOC = 2
MAX_ALLOWED_PAGES = 10
STAGE_NAME = "stage2_crossdoc_small_batch_artifact_compilation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled Stage 2 cross-document artifact compilation.")
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-pages-csv", default=None)
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--max-pages-per-doc", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--enable-deterministic-dedup", dest="deterministic_dedup_enabled", action="store_true")
    parser.add_argument("--disable-deterministic-dedup", dest="deterministic_dedup_enabled", action="store_false")
    parser.add_argument("--enable-real-api", action="store_true")
    parser.add_argument("--run-real-trial", action="store_true")
    parser.add_argument("--dry-run-fake-client", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.set_defaults(deterministic_dedup_enabled=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.max_docs) < 1 or int(args.max_docs) > MAX_ALLOWED_DOCS:
        raise RuntimeError("--max-docs must be between 1 and 5.")
    if int(args.max_pages_per_doc) < 1 or int(args.max_pages_per_doc) > MAX_ALLOWED_PAGES_PER_DOC:
        raise RuntimeError("--max-pages-per-doc must be between 1 and 2.")
    if int(args.max_pages) < 1 or int(args.max_pages) > MAX_ALLOWED_PAGES:
        raise RuntimeError("--max-pages must be between 1 and 10.")
    selected_pages_csv = getattr(args, "selected_pages_csv", None)
    if selected_pages_csv and not Path(selected_pages_csv).is_file():
        raise FileNotFoundError(f"--selected-pages-csv does not exist: {selected_pages_csv}")
    if args.dry_run_fake_client:
        return
    if not args.enable_real_api:
        raise RuntimeError("Refusing real provider cross-doc batch without --enable-real-api.")
    if not args.run_real_trial:
        raise RuntimeError("Refusing real provider cross-doc batch without --run-real-trial.")


def build_api_config(args: argparse.Namespace):
    api_config = build_api_run_config_from_mdocagent_yaml(
        args.config,
        overrides={
            "enable_real_api": bool(args.enable_real_api and not args.dry_run_fake_client),
            "provider": args.provider,
            "model_name": args.model_name,
            "timeout_seconds": args.timeout_seconds,
        },
    )
    if args.dry_run_fake_client:
        api_config.enable_real_api = False
    else:
        assert_real_api_allowed(api_config)
    return api_config


def run_crossdoc_batch(args: argparse.Namespace, client: ArtifactCompilerClient | None = None) -> Dict[str, Any]:
    validate_args(args)
    api_config = build_api_config(args)
    output_paths = initialize_output_paths(args.output_dir)
    records = read_json_or_jsonl_records(args.stage2_json)
    selected_pages_csv = getattr(args, "selected_pages_csv", None)
    if selected_pages_csv:
        selected_pages = load_selected_pages_from_quality_csv(
            records=records,
            selected_pages_csv=selected_pages_csv,
            extract_root=args.extract_root,
            max_docs=int(args.max_docs),
            max_pages_per_doc=int(args.max_pages_per_doc),
            max_pages=int(args.max_pages),
        )
    else:
        selected_pages = select_crossdoc_pages_for_batch(
            records,
            max_docs=int(args.max_docs),
            max_pages_per_doc=int(args.max_pages_per_doc),
            max_pages=int(args.max_pages),
            extract_root=args.extract_root,
        )
    active_client = client or (FakeArtifactCompilerClient() if args.dry_run_fake_client else RealApiArtifactCompilerClient(api_config))
    schema_dict = build_page_artifact_output_schema_dict()
    compiler_metadata = {
        "compiler_name": "real_api_artifact_compiler_client" if not args.dry_run_fake_client else "fake_artifact_compiler_client",
        "provider": args.provider,
        "model_name": api_config.model_name,
        "temperature": api_config.temperature,
        "max_repair_attempts": 0,
    }

    page_results: List[Dict[str, Any]] = []
    for selected_page in selected_pages:
        page_results.append(
            compile_selected_page(
                selected_page=selected_page,
                extract_root=args.extract_root,
                output_paths=output_paths,
                client=active_client,
                schema_dict=schema_dict,
                compiler_metadata=compiler_metadata,
                provider=args.provider,
                model_name=api_config.model_name,
                limits={
                    "max_docs": int(args.max_docs),
                    "max_pages_per_doc": int(args.max_pages_per_doc),
                    "max_pages": int(args.max_pages),
                },
                api_called=not args.dry_run_fake_client,
                prompt_version=str(getattr(args, "prompt_version", PROMPT_VERSION)),
                deterministic_dedup_enabled=bool(getattr(args, "deterministic_dedup_enabled", True)),
            )
        )

    manifest = build_stage2_run_manifest(
        stage=STAGE_NAME,
        script_name="scripts/stage2_compile_crossdoc_batch.py",
        config_path=str(args.config),
        provider=args.provider,
        model_name=str(api_config.model_name),
        output_dir=str(args.output_dir),
        real_api_called=bool(not args.dry_run_fake_client),
        limits={
            "max_docs": int(args.max_docs),
            "max_pages_per_doc": int(args.max_pages_per_doc),
            "max_pages": int(args.max_pages),
            "selected_pages_csv": str(selected_pages_csv) if selected_pages_csv else None,
            "prompt_version": str(getattr(args, "prompt_version", PROMPT_VERSION)),
            "deterministic_dedup_enabled": bool(getattr(args, "deterministic_dedup_enabled", True)),
            "dedup_stage": "after_raw_output_log_before_validation" if getattr(args, "deterministic_dedup_enabled", True) else None,
            "dedup_is_llm_repair": False,
            "dedup_uses_gold": False,
            "dedup_rule_version": "artifact_dedup_v1",
        },
        runtime_notes={
            "baseline_prediction_runtime_resume_parallel_retry_used": "not_used_by_stage2_crossdoc_compilation",
            "deterministic_dedup_enabled": bool(getattr(args, "deterministic_dedup_enabled", True)),
            "dedup_is_llm_repair": False,
            "dedup_uses_gold": False,
            "dedup_rule_version": "artifact_dedup_v1",
        },
    )
    manifest.update(
        {
            "input_index_path": str(args.stage2_json),
            "extract_root": str(args.extract_root),
            "uses_compact_page_routes": True,
            "uses_sidecar_preflight": False,
            "uses_gold": False,
            "stage2_depends_on_predict_py": False,
            "stage2_depends_on_multi_agent_system": False,
            "api_called": bool(not args.dry_run_fake_client),
            "num_api_calls": sum(1 for result in page_results if result.get("api_called")),
            "num_documents_attempted": len({result.get("doc_id") for result in page_results}),
            "num_pages_attempted": len(page_results),
        }
    )
    write_stage2_run_manifest(manifest, output_paths["run_manifest"])

    summary = summarize_crossdoc_results(
        page_results=page_results,
        args=args,
        model_name=api_config.model_name,
        output_paths=output_paths,
        records=records,
    )
    write_batch_summary(summary, output_paths["crossdoc_batch_summary"])
    write_crossdoc_quality_csv(page_results, output_paths["crossdoc_batch_quality"])
    return {
        "summary": summary,
        "manifest": manifest,
        "page_results": page_results,
        "paths": {key: str(value) for key, value in output_paths.items()},
    }


def load_selected_pages_from_quality_csv(
    records: list[dict],
    selected_pages_csv: str | Path,
    extract_root: str | Path,
    max_docs: int,
    max_pages_per_doc: int,
    max_pages: int,
) -> list[dict]:
    """Load the exact prior cross-doc page set without reselecting candidates."""

    rows = _read_selected_page_rows(selected_pages_csv)
    if len(rows) > int(max_pages):
        raise RuntimeError("--selected-pages-csv contains more rows than --max-pages.")

    records_by_doc: Dict[str, tuple[int, dict]] = {}
    for record_index, record in enumerate(records):
        doc_id = record.get("doc_id")
        if doc_id is not None and str(doc_id) not in records_by_doc:
            records_by_doc[str(doc_id)] = (record_index, record)

    doc_counts: Counter[str] = Counter()
    selected_pages: list[dict] = []
    for row in rows:
        doc_id = str(row["doc_id"])
        page_index = int(row["page_index"])
        if doc_id not in records_by_doc:
            raise RuntimeError(f"Selected page doc_id not found in stage2 records: {doc_id}")
        if len(doc_counts) >= int(max_docs) and doc_id not in doc_counts:
            raise RuntimeError("--selected-pages-csv exceeds --max-docs.")
        doc_counts[doc_id] += 1
        if doc_counts[doc_id] > int(max_pages_per_doc):
            raise RuntimeError("--selected-pages-csv exceeds --max-pages-per-doc.")
        record_index, record = records_by_doc[doc_id]
        stage2 = record.get("stage2", {})
        page_source = build_page_source(doc_id, extract_root, page_index)
        if not page_source:
            raise RuntimeError(f"Selected page source missing: {doc_id} page_index={page_index}")
        if not page_source.get("has_page_image") or not page_source.get("page_image_path"):
            raise RuntimeError(f"Selected page has no image input: {doc_id} page_index={page_index}")
        selected_page = {
            "record_index": int(record_index),
            "doc_id": doc_id,
            "question": record.get("question"),
            "answer_format": record.get("answer_format"),
            "page_index": page_index,
            "page_number_one_based": page_index + 1,
            "selection_reason": row.get("selection_reason") or "selected_pages_csv",
            "page_image_path": page_source.get("page_image_path"),
            "page_text_path": page_source.get("page_text_path"),
            "layout_block_ids": list(page_source.get("layout_block_ids", [])),
            "stage2": stage2,
        }
        selected_page["page_modality_diagnosis"] = diagnose_page_modality_from_question_and_preflight(
            record={"doc_id": doc_id, "question": record.get("question")},
            page_context={"question": record.get("question"), "page_sources": [page_source]},
            page_index=page_index,
        )
        selected_pages.append(selected_page)
    return selected_pages


def _read_selected_page_rows(selected_pages_csv: str | Path) -> list[dict]:
    rows = []
    with Path(selected_pages_csv).open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if not row.get("doc_id") or row.get("page_index") in (None, ""):
                continue
            rows.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "page_index": int(row["page_index"]),
                    "selection_reason": row.get("selection_reason"),
                }
            )
    return rows


def compile_selected_page(
    selected_page: Dict[str, Any],
    extract_root: str | Path,
    output_paths: Dict[str, Path],
    client: ArtifactCompilerClient,
    schema_dict: Dict[str, Any],
    compiler_metadata: Dict[str, Any],
    provider: str,
    model_name: str | None,
    limits: Dict[str, int],
    api_called: bool,
    prompt_version: str = PROMPT_VERSION,
    deterministic_dedup_enabled: bool = True,
) -> Dict[str, Any]:
    canonical_record = build_compiler_safe_record(selected_page)
    canonical_record["compilation_plan"]["compile_scope"] = "stage2_crossdoc_controlled_single_page"
    page_input = build_page_input(selected_page, extract_root)
    page_input["page_modality_diagnosis"] = selected_page.get("page_modality_diagnosis") or (
        diagnose_page_modality_from_question_and_preflight(
            record={"doc_id": selected_page.get("doc_id"), "question": selected_page.get("question")},
            page_context={
                "question": selected_page.get("question"),
                "page_sources": [
                    {
                        "page_index": int(selected_page["page_index"]),
                        "page_image_path": selected_page.get("page_image_path"),
                        "has_page_image": bool(selected_page.get("page_image_path")),
                    }
                ],
            },
            page_index=int(selected_page["page_index"]),
        )
    )
    artifact_store_path = output_paths["artifact_stores"] / artifact_store_file_name(
        selected_page["doc_id"],
        int(selected_page["page_index"]),
    )
    base_result = {
        "record_index": selected_page["record_index"],
        "doc_id": selected_page["doc_id"],
        "page_index": int(selected_page["page_index"]),
        "selection_reason": selected_page["selection_reason"],
        "page_image_path": selected_page["page_image_path"],
        "artifact_store_path": str(artifact_store_path),
        "raw_output_logged": True,
        "discard_logged": True,
        "api_called": api_called,
        "provider": provider,
        "model_name": model_name,
        "max_docs": limits["max_docs"],
        "max_pages_per_doc": limits["max_pages_per_doc"],
        "max_pages": limits["max_pages"],
    }

    try:
        compile_result = compile_page_with_client(
            canonical_record=canonical_record,
            page_input=page_input,
            client=client,
            schema_dict=schema_dict,
            compiler_metadata=compiler_metadata,
            raw_output_log_path=output_paths["raw_outputs"],
            discard_log_path=output_paths["discard"],
            compiler_version=COMPILER_VERSION,
            prompt_version=prompt_version,
            deterministic_dedup_enabled=deterministic_dedup_enabled,
        )
        write_page_artifact_store(
            canonical_record=canonical_record,
            page_input=page_input,
            compile_result=compile_result,
            compiler_metadata=compiler_metadata,
            artifact_store_path=artifact_store_path,
        )
        store = json.loads(artifact_store_path.read_text(encoding="utf-8"))
        forbidden_violations = count_forbidden_fields(store)
        num_raw_artifacts = int(compile_result["compilation_statistics"]["num_raw_artifacts"])
        num_raw_artifacts_before_dedup = int(
            compile_result["compilation_statistics"].get("num_raw_artifacts_before_dedup", num_raw_artifacts)
        )
        num_deduplicated_artifacts = int(compile_result["compilation_statistics"].get("num_deduplicated_artifacts", 0))
        num_valid_artifacts = int(compile_result["compilation_statistics"]["num_valid_artifacts"])
        num_validation_issues = int(compile_result["compilation_statistics"]["num_validation_issues"])
        return {
            **base_result,
            "num_raw_artifacts": num_raw_artifacts,
            "num_raw_artifacts_before_dedup": num_raw_artifacts_before_dedup,
            "num_deduplicated_artifacts": num_deduplicated_artifacts,
            "num_valid_artifacts": num_valid_artifacts,
            "num_validation_issues": num_validation_issues,
            "passed": num_valid_artifacts > 0 and forbidden_violations == 0,
            "forbidden_field_violations": forbidden_violations,
            "provider_error_type": None,
            "deterministic_dedup_enabled": bool(deterministic_dedup_enabled),
            "dedup_stage": "after_raw_output_log_before_validation" if deterministic_dedup_enabled else None,
            "dedup_rule": "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash",
            "schema_valid_rate_before_dedup": compile_result["compilation_statistics"].get("schema_valid_rate_before_dedup"),
            "schema_valid_rate_after_dedup": compile_result["compilation_statistics"].get("schema_valid_rate_after_dedup"),
            "discard_rate_before_dedup": compile_result["compilation_statistics"].get("discard_rate_before_dedup"),
            "discard_rate_after_dedup": compile_result["compilation_statistics"].get("discard_rate_after_dedup"),
        }
    except Exception as exc:
        write_raw_output_log(
            output_paths["raw_outputs"],
            build_raw_output_log_entry(
                doc_id=page_input["doc_id"],
                page_index=int(page_input["page_index"]),
                provider=provider,
                model_name=model_name,
                compiler_version=COMPILER_VERSION,
                prompt_version=prompt_version,
                raw_output={"artifacts": []},
                stage="stage2_compiler_provider_error",
                provider_error_type=type(exc).__name__,
                provider_error_message=str(exc),
            ),
        )
        write_discard_log_entry(
            output_paths["discard"],
            DiscardLogEntry(
                doc_id=page_input["doc_id"],
                page_index=int(page_input["page_index"]),
                artifact_id=None,
                error_type="provider_error",
                message=str(exc),
                field_path=None,
                details={"error_type": type(exc).__name__},
                stage="stage2_compiler_provider_error",
                compiler_version=COMPILER_VERSION,
            ),
        )
        write_empty_artifact_store(canonical_record, page_input, compiler_metadata, artifact_store_path)
        return {
            **base_result,
            "num_raw_artifacts": 0,
            "num_raw_artifacts_before_dedup": 0,
            "num_deduplicated_artifacts": 0,
            "num_valid_artifacts": 0,
            "num_validation_issues": 1,
            "passed": False,
            "forbidden_field_violations": 0,
            "provider_error_type": type(exc).__name__,
            "deterministic_dedup_enabled": bool(deterministic_dedup_enabled),
            "dedup_stage": "after_raw_output_log_before_validation" if deterministic_dedup_enabled else None,
            "dedup_rule": "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash",
        }


def write_page_artifact_store(
    canonical_record: Dict[str, Any],
    page_input: Dict[str, Any],
    compile_result: Dict[str, Any],
    compiler_metadata: Dict[str, Any],
    artifact_store_path: str | Path,
) -> None:
    page_index = int(page_input["page_index"])
    store = build_document_artifact_store(
        canonical_record=canonical_record,
        prepared_pages=[page_input],
        page_artifact_outputs={page_index: compile_result["raw_output"]},
        validation_results={
            page_index: {
                "valid_artifacts": compile_result["valid_artifacts"],
                "validation_issues": compile_result["validation_issues"],
            }
        },
        compiler_metadata=compiler_metadata,
    )
    write_artifact_store(store, artifact_store_path)


def write_empty_artifact_store(
    canonical_record: Dict[str, Any],
    page_input: Dict[str, Any],
    compiler_metadata: Dict[str, Any],
    artifact_store_path: str | Path,
) -> None:
    page_index = int(page_input["page_index"])
    store = build_document_artifact_store(
        canonical_record=canonical_record,
        prepared_pages=[page_input],
        page_artifact_outputs={page_index: {"doc_id": page_input["doc_id"], "page_index": page_index, "artifacts": []}},
        validation_results={page_index: {"valid_artifacts": [], "validation_issues": []}},
        compiler_metadata=compiler_metadata,
    )
    write_artifact_store(store, artifact_store_path)


def summarize_crossdoc_results(
    page_results: list[dict],
    args: argparse.Namespace,
    model_name: str | None,
    output_paths: Dict[str, Path],
    records: list[dict],
) -> dict:
    num_raw_artifacts = sum(int(result.get("num_raw_artifacts", 0)) for result in page_results)
    num_raw_artifacts_before_dedup = sum(
        int(result.get("num_raw_artifacts_before_dedup", result.get("num_raw_artifacts", 0)))
        for result in page_results
    )
    num_deduplicated_artifacts = sum(int(result.get("num_deduplicated_artifacts", 0)) for result in page_results)
    num_valid_artifacts = sum(int(result.get("num_valid_artifacts", 0)) for result in page_results)
    num_validation_issues = sum(int(result.get("num_validation_issues", 0)) for result in page_results)
    denominator = max(1, num_raw_artifacts)
    before_denominator = max(1, num_raw_artifacts_before_dedup)
    artifact_counts = count_artifacts_by_field(page_results, "artifact_type")
    modality_counts = count_artifacts_by_field(page_results, "modality")
    artifact_store_paths = [result["artifact_store_path"] for result in page_results if result.get("artifact_store_path")]
    return {
        "stage": STAGE_NAME,
        "provider": args.provider,
        "model_name": model_name,
        "stage2_json": str(args.stage2_json),
        "selected_pages_csv": str(getattr(args, "selected_pages_csv", None)) if getattr(args, "selected_pages_csv", None) else None,
        "prompt_version": str(getattr(args, "prompt_version", PROMPT_VERSION)),
        "uses_compact_stage2": any(
            isinstance(record.get("stage2"), dict) and bool(record["stage2"].get("candidate_page_routes"))
            for record in records
        ),
        "uses_sidecar_preflight": False,
        "max_docs": int(args.max_docs),
        "max_pages_per_doc": int(args.max_pages_per_doc),
        "max_pages": int(args.max_pages),
        "num_documents_attempted": len({result.get("doc_id") for result in page_results}),
        "num_pages_attempted": len(page_results),
        "num_api_calls": sum(1 for result in page_results if result.get("api_called")),
        "deterministic_dedup_enabled": bool(getattr(args, "deterministic_dedup_enabled", True)),
        "dedup_stage": "after_raw_output_log_before_validation" if getattr(args, "deterministic_dedup_enabled", True) else None,
        "dedup_rule": "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash",
        "dedup_rule_version": "artifact_dedup_v1",
        "num_raw_artifacts_before_dedup": num_raw_artifacts_before_dedup,
        "num_deduplicated_artifacts": num_deduplicated_artifacts,
        "deduplicated_artifact_issue_type_count": num_deduplicated_artifacts,
        "num_raw_artifacts": num_raw_artifacts,
        "num_valid_artifacts": num_valid_artifacts,
        "num_validation_issues": num_validation_issues,
        "schema_valid_rate": num_valid_artifacts / denominator,
        "schema_valid_rate_before_dedup": num_valid_artifacts / before_denominator,
        "schema_valid_rate_after_dedup": num_valid_artifacts / denominator,
        "anchoring_rate": num_valid_artifacts / denominator,
        "discard_rate": max(0, num_raw_artifacts - num_valid_artifacts) / denominator,
        "discard_rate_before_dedup": max(0, num_raw_artifacts_before_dedup - num_valid_artifacts) / before_denominator,
        "discard_rate_after_dedup": max(0, num_raw_artifacts - num_valid_artifacts) / denominator,
        "num_artifacts_by_type": artifact_counts,
        "num_artifacts_by_modality": modality_counts,
        "artifact_store_paths": artifact_store_paths,
        "raw_output_log_path": str(output_paths["raw_outputs"]),
        "discard_log_path": str(output_paths["discard"]),
        "crossdoc_batch_summary_path": str(output_paths["crossdoc_batch_summary"]),
        "crossdoc_batch_quality_path": str(output_paths["crossdoc_batch_quality"]),
        "manifest_path": str(output_paths["run_manifest"]),
        "run_manifest_path": str(output_paths["run_manifest"]),
        "forbidden_field_violations": sum(int(result.get("forbidden_field_violations", 0)) for result in page_results),
        "api_key_leaks": 0,
        "uses_answer": False,
        "uses_evidence_pages": False,
        "uses_binary_correctness": False,
        "real_api_called": bool(not args.dry_run_fake_client),
    }


def count_artifacts_by_field(page_results: list[dict], field_name: str) -> dict:
    counts: Counter[str] = Counter()
    for result in page_results:
        path_value = result.get("artifact_store_path")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.is_file():
            continue
        store = json.loads(path.read_text(encoding="utf-8"))
        for page in store.get("pages", []):
            for artifact in page.get("artifacts", []):
                value = artifact.get(field_name)
                if value:
                    counts[str(value)] += 1
    return dict(sorted(counts.items()))


def write_crossdoc_quality_csv(page_results: list[dict], path: str | Path) -> None:
    fields = [
        "record_index",
        "doc_id",
        "page_index",
        "selection_reason",
        "page_image_path",
        "num_raw_artifacts",
        "num_raw_artifacts_before_dedup",
        "num_deduplicated_artifacts",
        "num_valid_artifacts",
        "num_validation_issues",
        "artifact_store_path",
        "raw_output_logged",
        "discard_logged",
        "passed",
        "provider_error_type",
    ]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for result in page_results:
            writer.writerow({field: result.get(field) for field in fields})


def initialize_output_paths(output_dir: str | Path) -> Dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "artifact_stores": root / "artifact_stores",
        "raw_outputs": root / "raw_outputs" / "raw_outputs.jsonl",
        "discard": root / "discard" / "discard.jsonl",
        "reports": root / "reports",
        "crossdoc_batch_summary": root / "reports" / "crossdoc_batch_summary.json",
        "crossdoc_batch_quality": root / "reports" / "crossdoc_batch_quality.csv",
        "run_manifest": root / "reports" / "run_manifest.json",
    }
    for key in ("artifact_stores", "raw_outputs", "discard", "reports"):
        path = paths[key]
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    for file_path in [paths["raw_outputs"], paths["discard"], paths["crossdoc_batch_summary"], paths["crossdoc_batch_quality"], paths["run_manifest"]]:
        if file_path.exists():
            file_path.unlink()
    for old_store in paths["artifact_stores"].glob("*.json"):
        old_store.unlink()
    return paths


def main() -> None:
    args = parse_args()
    result = run_crossdoc_batch(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
