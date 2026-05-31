"""Unified Stage 2 compact page-route command entrypoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.artifact_pipeline import (
    build_document_artifact_store,
    compile_page_with_client,
    run_stage2_single_page_real_api_smoke_test,
    write_artifact_store,
)
from mdocnexus.common.model_config import QWEN3VL_CONFIG, stage2_model_fields
from mdocnexus.stage2.artifact_schema import build_page_artifact_output_schema_dict
from mdocnexus.stage2.index_builder import (
    OUT_OF_RANGE_ERROR,
    PAGE_COUNT_UNKNOWN_ERROR,
    apply_explicit_page_range_validation_to_canonical_record,
    augment_retrieval_results_file,
    build_api_run_config_from_mdocagent_yaml,
    build_mdocagent_extract_paths,
    build_page_source,
    find_record_by_id_or_doc_question,
    infer_document_page_count,
    normalize_record,
    read_json_or_jsonl_records,
    select_trial_candidate_from_stage2_file,
    summarize_mdocagent_model_config,
)
from mdocnexus.stage2.logs import (
    DiscardLogEntry,
    build_raw_output_log_entry,
    build_stage2_run_manifest,
    write_discard_log_entry,
    write_raw_output_log,
    write_stage2_run_manifest,
)
from mdocnexus.stage2.locator_enrichment import (
    classify_artifact_locator,
    locator_kind_counts as stage2_locator_kind_counts,
    page_id_for,
)
from mdocnexus.stage2.page_input import build_basic_layout_blocks, candidate_input_routes_for_page, load_page_content, prepare_pages_for_compilation
from mdocnexus.stage2.provider import (
    ArtifactCompilerClient,
    FakeArtifactCompilerClient,
    RealApiArtifactCompilerClient,
    assert_real_api_allowed,
    build_artifact_compiler_system_prompt,
    build_artifact_compiler_user_prompt,
    build_document_generic_artifact_compiler_user_prompt,
    describe_image_payload,
)
from mdocnexus.stage2.reports import (
    audit_batch_artifact_outputs,
    audit_crossdoc_batch_with_options,
    compare_crossdoc_audits,
    count_forbidden_fields,
    summarize_batch_results,
    write_audit_csv,
    write_audit_json,
    write_batch_quality_csv,
    write_batch_summary,
    write_page_quality_csv,
    write_refinement_comparison,
)
from mdocnexus.stage2.selectors import (
    diagnose_page_modality_from_question_and_preflight,
    select_crossdoc_pages_for_batch,
    select_pages_for_small_batch,
    select_single_page_trial_candidate,
    strip_eval_only_fields,
)

MAX_ALLOWED_PAGES = 10
COMPILER_VERSION = "stage2_compiler_v1"
PROMPT_VERSION = "artifact_compiler_prompt_v1"


def parse_small_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled Stage 2 small-batch artifact compilation.")
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-pages-total", type=int, default=None)
    parser.add_argument("--max-pages-per-call", type=int, default=1)
    parser.add_argument("--max-pages-real-cap", type=int, default=10)
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--enable-deterministic-dedup", dest="deterministic_dedup_enabled", action="store_true")
    parser.add_argument("--disable-deterministic-dedup", dest="deterministic_dedup_enabled", action="store_false")
    parser.add_argument("--enable-real-api", action="store_true")
    parser.add_argument("--run-real-trial", action="store_true")
    parser.add_argument("--dry-run-fake-client", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.set_defaults(deterministic_dedup_enabled=True)
    return parser.parse_args()


def validate_small_batch_args(args: argparse.Namespace) -> None:
    if int(args.max_pages) < 1:
        raise RuntimeError("--max-pages must be at least 1.")
    if int(args.max_pages) > MAX_ALLOWED_PAGES:
        raise RuntimeError("--max-pages must not exceed 10.")
    if args.dry_run_fake_client:
        return
    if not args.enable_real_api:
        raise RuntimeError("Refusing real provider batch without --enable-real-api.")
    if not args.run_real_trial:
        raise RuntimeError("Refusing real provider batch without --run-real-trial.")
    validate_real_page_limits(args)


def validate_real_page_limits(args: argparse.Namespace) -> None:
    max_pages_total = getattr(args, "max_pages_total", None)
    if max_pages_total is None:
        max_pages_total = getattr(args, "max_pages", None)
    if max_pages_total is None:
        raise RuntimeError("Real provider mode requires finite --max-pages-total.")
    max_pages_total = int(max_pages_total)
    max_pages_per_call = int(getattr(args, "max_pages_per_call", 1))
    max_pages_real_cap = int(getattr(args, "max_pages_real_cap", 10))
    if max_pages_per_call != 1:
        raise RuntimeError("Real provider mode requires --max-pages-per-call=1.")
    if max_pages_total < 1 or max_pages_total > max_pages_real_cap:
        raise RuntimeError("--max-pages-total must be between 1 and --max-pages-real-cap in real provider mode.")


def build_small_batch_api_config(args: argparse.Namespace):
    api_config = build_api_run_config_from_mdocagent_yaml(
        args.config,
        overrides={
            "enable_real_api": bool(args.enable_real_api and not args.dry_run_fake_client),
            "provider": args.provider,
            "model_name": args.model_name,
            "timeout_seconds": args.timeout_seconds,
            "max_pages_total": getattr(args, "max_pages_total", None) or args.max_pages,
            "max_pages_per_call": getattr(args, "max_pages_per_call", 1),
        },
    )
    if args.dry_run_fake_client:
        api_config.enable_real_api = False
    else:
        assert_real_api_allowed(api_config)
    return api_config


def run_small_batch(args: argparse.Namespace, client: ArtifactCompilerClient | None = None) -> Dict[str, Any]:
    validate_small_batch_args(args)
    api_config = build_small_batch_api_config(args)
    output_paths = build_output_paths(args.output_dir)
    records = read_json_or_jsonl_records(args.stage2_json)
    selected_pages = select_pages_for_small_batch(
        records,
        max_pages=args.max_pages,
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
        page_result = compile_selected_page(
            selected_page=selected_page,
            extract_root=args.extract_root,
            output_paths=output_paths,
            client=active_client,
            schema_dict=schema_dict,
            compiler_metadata=compiler_metadata,
            provider=args.provider,
            model_name=api_config.model_name,
            max_pages=int(args.max_pages),
            api_called=not args.dry_run_fake_client,
            deterministic_dedup_enabled=bool(getattr(args, "deterministic_dedup_enabled", True)),
        )
        page_results.append(page_result)

    summary = summarize_batch_results(page_results)
    summary.update(
        {
            "provider": args.provider,
            "model_name": api_config.model_name,
            "max_pages": int(args.max_pages),
            "deterministic_dedup_enabled": bool(getattr(args, "deterministic_dedup_enabled", True)),
            "dedup_stage": "after_raw_output_log_before_validation" if getattr(args, "deterministic_dedup_enabled", True) else None,
            "dedup_rule": "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash",
        }
    )
    write_batch_summary(summary, output_paths["batch_summary"])
    write_batch_quality_csv(page_results, output_paths["batch_quality"])
    return {
        "summary": summary,
        "page_results": page_results,
        "paths": {key: str(value) for key, value in output_paths.items()},
    }


def compile_selected_page(
    selected_page: Dict[str, Any],
    extract_root: str | Path,
    output_paths: Dict[str, Path],
    client: ArtifactCompilerClient,
    schema_dict: Dict[str, Any],
    compiler_metadata: Dict[str, Any],
    provider: str,
    model_name: str | None,
    max_pages: int,
    api_called: bool,
    deterministic_dedup_enabled: bool = True,
) -> Dict[str, Any]:
    canonical_record = build_compiler_safe_record(selected_page)
    page_input = build_page_input(selected_page, extract_root)
    compile_result = compile_page_with_client(
        canonical_record=canonical_record,
        page_input=page_input,
        client=client,
        schema_dict=schema_dict,
        compiler_metadata=compiler_metadata,
        raw_output_log_path=output_paths["raw_outputs"],
        discard_log_path=output_paths["discard"],
        compiler_version=COMPILER_VERSION,
        prompt_version=PROMPT_VERSION,
        deterministic_dedup_enabled=deterministic_dedup_enabled,
    )
    artifact_store_path = output_paths["artifact_stores"] / artifact_store_file_name(
        selected_page["doc_id"],
        int(selected_page["page_index"]),
    )
    write_small_page_artifact_store(
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
        "record_index": selected_page["record_index"],
        "doc_id": selected_page["doc_id"],
        "page_index": int(selected_page["page_index"]),
        "selection_reason": selected_page["selection_reason"],
        "page_image_path": selected_page["page_image_path"],
        "num_raw_artifacts": num_raw_artifacts,
        "num_raw_artifacts_before_dedup": num_raw_artifacts_before_dedup,
        "num_deduplicated_artifacts": num_deduplicated_artifacts,
        "num_valid_artifacts": num_valid_artifacts,
        "num_validation_issues": num_validation_issues,
        "artifact_store_path": str(artifact_store_path),
        "raw_output_logged": output_paths["raw_outputs"].is_file(),
        "discard_logged": output_paths["discard"].is_file(),
        "passed": num_valid_artifacts > 0 and forbidden_violations == 0,
        "api_called": api_called,
        "provider": provider,
        "model_name": model_name,
        "max_pages": int(max_pages),
        "forbidden_field_violations": forbidden_violations,
        "deterministic_dedup_enabled": bool(deterministic_dedup_enabled),
        "dedup_stage": "after_raw_output_log_before_validation" if deterministic_dedup_enabled else None,
        "dedup_rule": "doc_id+page_index+artifact_type+modality+source_anchor_ids+content_hash",
        "schema_valid_rate_before_dedup": compile_result["compilation_statistics"].get("schema_valid_rate_before_dedup"),
        "schema_valid_rate_after_dedup": compile_result["compilation_statistics"].get("schema_valid_rate_after_dedup"),
        "discard_rate_before_dedup": compile_result["compilation_statistics"].get("discard_rate_before_dedup"),
        "discard_rate_after_dedup": compile_result["compilation_statistics"].get("discard_rate_after_dedup"),
    }


def build_compiler_safe_record(selected_page: Dict[str, Any]) -> Dict[str, Any]:
    page_index = int(selected_page["page_index"])
    return {
        "document": {
            "doc_id": selected_page["doc_id"],
            "doc_type": None,
            "dataset": None,
        },
        "question": {
            "text": selected_page.get("question"),
            "answer_format": selected_page.get("answer_format"),
        },
        "question_constraints": {},
        "candidate_pool": {
            "explicit_constraint_pages": [page_index] if selected_page.get("selection_reason") == "valid_explicit_page_with_image" else [],
            "retrieval_candidate_pages": [page_index],
            "retrieval_missed_explicit_pages": [],
        },
        "compilation_plan": {
            "compile_scope": "stage2_small_batch_single_page",
            "pages_to_compile": [page_index],
            "priority_pages": [page_index] if selected_page.get("selection_reason") == "valid_explicit_page_with_image" else [],
            "compilation_reasons": [
                {
                    "page_index": page_index,
                    "reason_type": selected_page.get("selection_reason"),
                    "reason_text": selected_page.get("selection_reason"),
                }
            ],
        },
    }


def build_document_generic_compiler_record(selected_page: Dict[str, Any]) -> Dict[str, Any]:
    page_index = int(selected_page["page_index"])
    return {
        "document": {
            "doc_id": selected_page["doc_id"],
            "doc_type": None,
            "dataset": None,
        },
        "question": {
            "text": None,
            "answer_format": None,
        },
        "question_constraints": {},
        "candidate_pool": {
            "explicit_constraint_pages": [],
            "retrieval_candidate_pages": [page_index],
            "retrieval_missed_explicit_pages": [],
        },
        "compilation_plan": {
            "compile_scope": "stage2_document_generic_single_page",
            "pages_to_compile": [page_index],
            "priority_pages": [],
            "compilation_reasons": [
                {
                    "page_index": page_index,
                    "reason_type": "document_generic_page_compilation",
                    "reason_text": "compile available page evidence without question conditioning",
                }
            ],
        },
    }


def _strip_stage2_routes_for_document_generic(selected_page: Dict[str, Any]) -> Dict[str, Any]:
    clean_page = dict(selected_page)
    clean_page["question"] = None
    clean_page["answer_format"] = None
    clean_page["stage2"] = {}
    return clean_page


def build_page_input(selected_page: Dict[str, Any], extract_root: str | Path) -> Dict[str, Any]:
    page_index = int(selected_page["page_index"])
    page_content = load_page_content(
        canonical_record={"document": {"doc_id": selected_page["doc_id"]}},
        extract_path=extract_root,
        page_index=page_index,
    )
    input_routes = candidate_input_routes_for_page(selected_page.get("stage2", {}), page_index)
    route_metadata_present = input_routes is not None
    allow_text_input = not route_metadata_present or "text" in input_routes
    allow_image_input = not route_metadata_present or "image" in input_routes
    page_text = page_content["page_text"] if allow_text_input else None
    page_text_path = page_content["page_text_path"] if allow_text_input else None
    page_image_path = page_content["page_image_path"] if allow_image_input else None
    layout_blocks = build_basic_layout_blocks(
        doc_id=selected_page["doc_id"],
        page_index=page_index,
        page_text=page_text,
        has_page_image=page_image_path is not None,
    )
    page_input = {
        "doc_id": selected_page["doc_id"],
        "page_index": page_index,
        "page_id": page_id_for(selected_page["doc_id"], page_index),
        "page_text": page_text,
        "page_text_path": page_text_path,
        "page_image_path": page_image_path,
        "has_page_text": page_text is not None,
        "has_page_image": page_image_path is not None,
        "layout_blocks": layout_blocks,
    }
    if route_metadata_present:
        page_input["input_routes"] = input_routes
    return page_input


def write_small_page_artifact_store(
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


def build_output_paths(output_dir: str | Path) -> Dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "artifact_stores": root / "artifact_stores",
        "raw_outputs": root / "raw_outputs" / "raw_outputs.jsonl",
        "discard": root / "discard" / "discard.jsonl",
        "reports": root / "reports",
        "batch_summary": root / "reports" / "batch_summary.json",
        "batch_quality": root / "reports" / "batch_quality.csv",
    }


def artifact_store_file_name(doc_id: str, page_index: int) -> str:
    doc_name = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    return f"{doc_name}_p{int(page_index):03d}.json"


def small_batch_main() -> None:
    args = parse_args()
    result = run_small_batch(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


MAX_ALLOWED_DOCS = 50
MAX_ALLOWED_PAGES_PER_DOC = 5
MAX_ALLOWED_CROSSDOC_PAGES = 50
STAGE_NAME = "stage2_crossdoc_small_batch_artifact_compilation"


def parse_crossdoc_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled Stage 2 cross-document artifact compilation.")
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-pages-csv", default=None)
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--max-pages-per-doc", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-pages-total", type=int, default=None)
    parser.add_argument("--max-pages-per-call", type=int, default=1)
    parser.add_argument("--max-pages-real-cap", type=int, default=10)
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


def validate_crossdoc_args(args: argparse.Namespace) -> None:
    if int(args.max_docs) < 1 or int(args.max_docs) > MAX_ALLOWED_DOCS:
        raise RuntimeError("--max-docs must be between 1 and 50.")
    if int(args.max_pages_per_doc) < 1 or int(args.max_pages_per_doc) > MAX_ALLOWED_PAGES_PER_DOC:
        raise RuntimeError("--max-pages-per-doc must be between 1 and 5.")
    if int(args.max_pages) < 1 or int(args.max_pages) > MAX_ALLOWED_CROSSDOC_PAGES:
        raise RuntimeError("--max-pages must be between 1 and 50.")
    selected_pages_csv = getattr(args, "selected_pages_csv", None)
    if selected_pages_csv and not Path(selected_pages_csv).is_file():
        raise FileNotFoundError(f"--selected-pages-csv does not exist: {selected_pages_csv}")
    if args.dry_run_fake_client:
        return
    if not args.enable_real_api:
        raise RuntimeError("Refusing real provider cross-doc batch without --enable-real-api.")
    if not args.run_real_trial:
        raise RuntimeError("Refusing real provider cross-doc batch without --run-real-trial.")
    validate_real_page_limits(args)


def build_crossdoc_api_config(args: argparse.Namespace):
    provider_name = _provider_config_name(args)
    api_config = build_api_run_config_from_mdocagent_yaml(
        args.config,
        overrides={
            "enable_real_api": bool(args.enable_real_api and not args.dry_run_fake_client),
            "provider": provider_name,
            "model_name": args.model_name,
            "timeout_seconds": args.timeout_seconds,
            "max_pages_total": getattr(args, "max_pages_total", None) or args.max_pages,
            "max_pages_per_call": getattr(args, "max_pages_per_call", 1),
        },
    )
    api_config.image_payload_mode = _image_payload_mode(args)
    if args.dry_run_fake_client:
        api_config.enable_real_api = False
    else:
        assert_real_api_allowed(api_config)
    return api_config


def run_crossdoc_batch(args: argparse.Namespace, client: ArtifactCompilerClient | None = None) -> Dict[str, Any]:
    validate_crossdoc_args(args)
    api_config = build_crossdoc_api_config(args)
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
            compile_crossdoc_selected_page(
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
                max_retries=int(getattr(args, "max_retries", 0)),
                document_generic=bool(getattr(args, "document_generic", False)),
            )
        )

    manifest = build_stage2_run_manifest(
        stage=STAGE_NAME,
        script_name="scripts/stage2.py compile",
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


def compile_crossdoc_selected_page(
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


def crossdoc_batch_main() -> None:
    args = parse_args()
    result = run_crossdoc_batch(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def build_preflight_report(
    sample_path: str | Path,
    extract_root: str | Path,
    config_path: str | Path,
    target_page_index: int,
    record_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    question_substring: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    pdf_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Validate Stage 2 inputs for a single page without model calls."""

    target_page_index = int(target_page_index)
    records = read_json_or_jsonl_records(sample_path)
    record = find_record_by_id_or_doc_question(records, record_id, doc_id, question_substring)
    normalized = record if "canonical_record" in record else normalize_record(record)
    canonical_record = normalized["canonical_record"]
    page_count_info = infer_document_page_count(
        doc_id=canonical_record["document"]["doc_id"],
        pdf_root=pdf_root,
        extract_root=extract_root,
    )
    explicit_validation = apply_explicit_page_range_validation_to_canonical_record(
        canonical_record,
        page_count_info,
    )

    prepared_result = prepare_pages_for_compilation(canonical_record, extract_root)
    target_pages = [
        page for page in prepared_result["pages"] if int(page["page_index"]) == target_page_index
    ]
    target_page = target_pages[0] if target_pages else None
    api_config = build_api_run_config_from_mdocagent_yaml(config_path, overrides=overrides or {})

    invalid_target_refs = [
        ref
        for ref in explicit_validation["invalid_explicit_page_references"]
        if int(ref.get("page_index_zero_based", -1)) == target_page_index
    ]
    blocking_reasons = []
    if any(ref.get("error_type") == OUT_OF_RANGE_ERROR for ref in invalid_target_refs):
        blocking_reasons.append(OUT_OF_RANGE_ERROR)
    if any(ref.get("error_type") == PAGE_COUNT_UNKNOWN_ERROR for ref in invalid_target_refs):
        blocking_reasons.append(PAGE_COUNT_UNKNOWN_ERROR)

    if not invalid_target_refs:
        if target_page is None:
            blocking_reasons.append("target_page_not_prepared")
        elif not target_page.get("layout_blocks"):
            blocking_reasons.append("missing_source_anchors")
        if target_page and not _has_expected_full_page_block(target_page, target_page_index):
            blocking_reasons.append("missing_full_page_image_anchor")

    page_text_path = target_page.get("page_text_path") if target_page else None
    page_image_path = target_page.get("page_image_path") if target_page else None
    layout_block_ids = [block["block_id"] for block in target_page.get("layout_blocks", [])] if target_page else []
    should_call_api = not blocking_reasons
    return {
        "preflight_passed": not blocking_reasons,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "record_id": normalized.get("record_id"),
        "doc_id": canonical_record["document"]["doc_id"],
        "target_page_index": target_page_index,
        "target_page_number_one_based": target_page_index + 1,
        "extract_root": str(extract_root),
        "page_text_path": page_text_path,
        "page_image_path": page_image_path,
        "has_page_text": bool(target_page and target_page.get("has_page_text")),
        "has_page_image": bool(target_page and target_page.get("has_page_image")),
        "layout_block_ids": layout_block_ids,
        "page_count": page_count_info.get("page_count"),
        "page_count_info": page_count_info,
        "invalid_explicit_page_references": explicit_validation["invalid_explicit_page_references"],
        "valid_explicit_page_indices": explicit_validation["valid_explicit_page_indices"],
        "candidate_pool": canonical_record.get("candidate_pool", {}),
        "pages_to_compile": canonical_record.get("compilation_plan", {}).get("pages_to_compile", []),
        "model_config": summarize_mdocagent_model_config(config_path, api_config),
        "will_call_api": False,
        "should_call_api": should_call_api,
        "should_generate_artifact": should_call_api,
    }


def write_preflight_report(report: Dict[str, Any], output_report: str | Path) -> None:
    path = Path(output_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"enable_real_api": False, "timeout_seconds": args.timeout_seconds}
    for arg_name, field_name in (
        ("provider", "provider"),
        ("model_name", "model_name"),
        ("api_base_url", "api_base_url"),
        ("api_key_env_var", "api_key_env_var"),
        ("temperature", "temperature"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    return overrides


def _has_expected_full_page_block(target_page: Dict[str, Any], target_page_index: int) -> bool:
    expected_block_id = f"p{target_page_index:03d}_full_page_image"
    return any(block.get("block_id") == expected_block_id for block in target_page.get("layout_blocks", []))


def write_candidate_report(report: Dict[str, Any], output_report: str | Path) -> None:
    path = Path(output_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_step7b_command(args: argparse.Namespace, report: Dict[str, Any]) -> str | None:
    selected = report.get("selected")
    if not selected:
        return None
    return " ".join(
        [
            "python3",
            "scripts/stage2.py single-page-real-trial",
            f"--config {args.config}",
            f"--sample-path {args.sample_path}",
            f"--extract-root {args.extract_root}",
            f"--candidate-report {args.output_report}",
            "--enable-real-api",
            "--run-real-trial",
            "--output-path outputs/stage2/real_single_page_trial/artifact_store.json",
        ]
    )


def apply_candidate_report_to_args(args: argparse.Namespace) -> argparse.Namespace:
    """Populate single-page trial args from an objective candidate report."""

    if not args.candidate_report:
        return args
    report = _load_candidate_report(args.candidate_report)
    selected = report["selected"]
    args.record_id = selected["record_id"]
    args.doc_id = selected["doc_id"]
    args.question_substring = selected.get("question")
    args.target_page_index = int(selected["page_index"])
    if not args.extract_root and report.get("extract_root"):
        args.extract_root = report["extract_root"]
    return args


def _load_candidate_report(candidate_report: str) -> Dict[str, Any]:
    report = json.loads(Path(candidate_report).read_text(encoding="utf-8"))
    if not report.get("selection_passed"):
        raise RuntimeError("Candidate report did not pass selection.")
    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("Candidate report does not contain a selected candidate.")
    required_fields = {"record_id", "doc_id", "page_index"}
    missing_fields = sorted(field for field in required_fields if selected.get(field) in (None, ""))
    if missing_fields:
        raise RuntimeError(f"Candidate report selected block is missing fields: {missing_fields}.")
    return report


def validate_real_trial_args(args: argparse.Namespace) -> None:
    """Reject any invocation that is not an explicit single-page real trial."""

    if getattr(args, "preflight_only", False):
        return
    if not getattr(args, "enable_real_api", False):
        raise RuntimeError("Refusing real provider trial without --enable-real-api.")
    if not getattr(args, "run_real_trial", False):
        raise RuntimeError("Refusing real provider trial without --run-real-trial.")
    if getattr(args, "target_page_index", None) is None:
        raise RuntimeError("A single --target-page-index is required.")


def load_canonical_record_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    record_path = args.normalized_record_path or args.sample_path
    if not record_path:
        raise RuntimeError("Either --normalized-record-path or --sample-path is required.")
    records = read_json_or_jsonl_records(record_path)
    if args.candidate_report:
        records = [strip_eval_only_fields(record) for record in records]
    record = find_record_by_id_or_doc_question(
        records=records,
        record_id=args.record_id,
        doc_id=args.doc_id,
        question_substring=args.question_substring,
    )
    normalized = record if "canonical_record" in record else normalize_record(record)
    canonical_record = normalized["canonical_record"]
    apply_page_range_validation_from_args(canonical_record, args)
    return canonical_record


def build_single_page_api_run_config(args: argparse.Namespace):
    return build_api_run_config_from_mdocagent_yaml(
        args.config,
        overrides=build_config_overrides(args, enable_real_api=bool(args.enable_real_api)),
    )


def build_config_overrides(args: argparse.Namespace, enable_real_api: bool) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "enable_real_api": enable_real_api,
        "timeout_seconds": args.timeout_seconds,
        "raw_output_dir": args.raw_output_dir,
        "discard_log_dir": args.discard_log_dir,
    }
    for arg_name, field_name in (
        ("provider", "provider"),
        ("model_name", "model_name"),
        ("api_base_url", "api_base_url"),
        ("api_key_env_var", "api_key_env_var"),
        ("temperature", "temperature"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    return overrides


def resolve_extract_root(args: argparse.Namespace) -> str:
    extract_root = args.extract_root or args.extract_path
    if not extract_root:
        raise RuntimeError("Either --extract-root or --extract-path is required.")
    return extract_root


def run_preflight_only(args: argparse.Namespace) -> Dict[str, Any]:
    record_path = args.sample_path or args.normalized_record_path
    if not record_path:
        raise RuntimeError("Either --sample-path or --normalized-record-path is required for preflight.")
    return build_preflight_report(
        sample_path=record_path,
        extract_root=resolve_extract_root(args),
        config_path=args.config,
        target_page_index=args.target_page_index,
        record_id=args.record_id,
        doc_id=args.doc_id,
        question_substring=args.question_substring,
        overrides=build_config_overrides(args, enable_real_api=False),
        pdf_root=args.pdf_root,
    )


def apply_page_range_validation_from_args(
    canonical_record: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    page_count_info = infer_document_page_count(
        doc_id=canonical_record["document"]["doc_id"],
        pdf_root=args.pdf_root,
        extract_root=resolve_extract_root(args),
    )
    validation = apply_explicit_page_range_validation_to_canonical_record(
        canonical_record,
        page_count_info,
    )
    invalid_target_refs = [
        ref
        for ref in validation["invalid_explicit_page_references"]
        if int(ref.get("page_index_zero_based", -1)) == int(args.target_page_index)
    ]
    if invalid_target_refs:
        error_types = sorted({ref.get("error_type") for ref in invalid_target_refs})
        if OUT_OF_RANGE_ERROR in error_types:
            raise RuntimeError(f"Target page is outside the document page range: {args.target_page_index}.")
        if PAGE_COUNT_UNKNOWN_ERROR in error_types:
            raise RuntimeError(f"Target page range could not be validated: {args.target_page_index}.")
        raise RuntimeError(f"Target page is not valid for compilation: {args.target_page_index}.")
    return validation


def single_page_real_trial_main() -> None:
    args = apply_candidate_report_to_args(parse_args())
    if args.preflight_only:
        print(json.dumps(run_preflight_only(args), ensure_ascii=False, indent=2))
        return

    validate_real_trial_args(args)
    canonical_record = load_canonical_record_from_args(args)
    summary = run_stage2_single_page_real_api_smoke_test(
        canonical_record=canonical_record,
        extract_path=resolve_extract_root(args),
        output_path=args.output_path,
        api_config=build_single_page_api_run_config(args),
        target_page_index=args.target_page_index,
        run_real_trial=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_compare_audits_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 2 cross-doc audit reports.")
    parser.add_argument("--baseline-audit", required=True)
    parser.add_argument("--refined-audit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def compare_audits_main() -> None:
    args = parse_args()
    report = compare_crossdoc_audits(args.baseline_audit, args.refined_audit)
    write_refinement_comparison(report, args.output_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_small_audit_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit existing Stage 2 small-batch artifacts without API calls.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def small_audit_main() -> None:
    args = parse_args()
    # Read the aligned Stage 2 JSON only to confirm it exists and is parseable;
    # do not use gold/eval fields for quality decisions.
    _ = len(read_json_or_jsonl_records(args.stage2_json))
    audit = audit_batch_artifact_outputs(args.batch_dir)
    write_audit_json(audit, args.output_json)
    write_audit_csv(audit, args.output_csv)
    print(json.dumps({key: value for key, value in audit.items() if key != "artifact_store_audits"}, ensure_ascii=False, indent=2))


def parse_crossdoc_audit_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an existing Stage 2 artifact batch offline.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--stage2-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def crossdoc_audit_main() -> None:
    args = parse_args()
    validate_inputs(args)
    report = audit_crossdoc_batch_with_options(
        batch_dir=args.batch_dir,
        stage2_json=args.stage2_json,
    )
    write_audit_json(report, args.output_json)
    write_page_quality_csv(report, args.output_csv)
    public_report = {key: value for key, value in report.items() if not key.startswith("_")}
    print(json.dumps(public_report, ensure_ascii=False, indent=2))


def validate_inputs(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
    artifact_store_dir = batch_dir / "artifact_stores"
    if not artifact_store_dir.is_dir():
        raise FileNotFoundError(f"Artifact store directory does not exist: {artifact_store_dir}")
    if args.stage2_json is not None and not Path(args.stage2_json).is_file():
        raise FileNotFoundError(f"Stage 2 JSON does not exist: {args.stage2_json}")


DEFAULT_STAGE2_INPUT = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_STAGE2_INDEX = "outputs/stage2/clean/sample-with-stage2-index.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_CONFIG = "config/model/qwen3vl.yaml"
DEFAULT_CLEAN_BATCH_DIR = "outputs/stage2/clean"
DEFAULT_DOC_GENERIC_BATCH_DIR = "outputs/stage2_doc"
DOC_COMPILE_DEFAULT_MAX_REAL_PAGES = 10

FINAL_ARTIFACT_FIELDS = [
    "record_index",
    "doc_id",
    "page_id",
    "page_index",
    "artifact_id",
    "artifact_type",
    "modality",
    "content",
    "normalized_content",
    "source_anchors",
    "provenance",
    "status",
    "validation_status",
    "locators",
    "source_anchored",
    "element_locatable",
    "proof_trace_eligible",
]
FINAL_ARTIFACT_TYPES = {
    "text_span",
    "numeric_fact",
    "table",
    "table_cell",
    "figure",
    "caption",
    "visual_region",
    "visual_observation",
    "section_header",
}
FINAL_MODALITIES = {"text", "image", "table", "layout", "numeric"}
CLEAN_OUTPUT_FILENAMES = (
    "artifacts.jsonl",
    "discard.jsonl",
    "quality_report.json",
    "manifest.json",
    "raw_outputs.jsonl",
    "crossdoc_batch_summary.json",
    "crossdoc_quality_audit.json",
    "crossdoc_quality_by_page.csv",
    "crossdoc_batch_quality.csv",
    "sample-with-stage2-preflight.json",
)


def _stage2_index_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "sample-with-stage2-index.json"


def _final_output_paths(output_dir: str | Path, debug_raw_output: bool = False) -> Dict[str, Path | None]:
    root = Path(output_dir)
    paths: Dict[str, Path | None] = {
        "root": root,
        "artifacts": root / "artifacts.jsonl",
        "discard": root / "discard.jsonl",
        "quality_report": root / "quality_report.json",
        "call_log": root / "call_log.jsonl",
        "raw_outputs": root / "raw_outputs.jsonl" if debug_raw_output else None,
    }
    return paths


def _prepare_final_output_dir(output_dir: str | Path, debug_raw_output: bool = False) -> Dict[str, Path | None]:
    paths = _final_output_paths(output_dir, debug_raw_output=debug_raw_output)
    root = Path(paths["root"])
    root.mkdir(parents=True, exist_ok=True)
    for dirname in ("artifact_stores", "raw_outputs", "reports", "preflight"):
        target = root / dirname
        if target.exists():
            shutil.rmtree(target)
    for filename in CLEAN_OUTPUT_FILENAMES:
        target = root / filename
        if target.exists():
            target.unlink()
    Path(paths["artifacts"]).write_text("", encoding="utf-8")
    Path(paths["discard"]).write_text("", encoding="utf-8")
    Path(paths["call_log"]).write_text("", encoding="utf-8")
    raw_path = paths.get("raw_outputs")
    if raw_path is not None:
        Path(raw_path).write_text("", encoding="utf-8")
    return paths


def _append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        file_obj.write("\n")


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path: Any) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return _file_sha256(file_path)


def _normalize_doc_compile_provider_mode(args: argparse.Namespace) -> str:
    provider_mode = str(getattr(args, "provider", "dry_run") or "dry_run").strip().lower()
    legacy_fake_modes = {"siliconflow", "compatible_chat", "custom", "mock"}
    if provider_mode in legacy_fake_modes and not bool(getattr(args, "enable_real_api", False)):
        provider_mode = "dry_run"
    if provider_mode not in {"dry_run", "fake", "real"}:
        raise RuntimeError("--provider for doc-compile must be one of: dry_run, fake, real.")
    return provider_mode


def _provider_config_name(args: argparse.Namespace) -> str:
    provider_name = str(getattr(args, "model_provider", "") or "").strip()
    if provider_name:
        return provider_name
    config_provider = str(getattr(args, "config_provider", "") or "").strip()
    if config_provider:
        return config_provider
    provider_mode = str(getattr(args, "provider", "") or "").strip()
    if provider_mode in {"dry_run", "fake", "real", ""}:
        return "siliconflow"
    return provider_mode


def _image_payload_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "image_payload_mode", "image_url") or "image_url").strip().lower()
    if mode not in {"image_url", "base64", "none"}:
        raise RuntimeError("--image-payload-mode must be one of: image_url, base64, none.")
    return mode


def _build_call_prompt_hash(
    canonical_record: Dict[str, Any],
    page_input: Dict[str, Any],
    schema_dict: Dict[str, Any],
    document_generic: bool,
) -> str:
    system_prompt = build_artifact_compiler_system_prompt()
    if document_generic:
        user_prompt = build_document_generic_artifact_compiler_user_prompt(
            canonical_record=canonical_record,
            page_input=page_input,
            schema_dict=schema_dict,
        )
    else:
        user_prompt = build_artifact_compiler_user_prompt(
            canonical_record=canonical_record,
            page_input=page_input,
            schema_dict=schema_dict,
        )
    return _stable_json_hash(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema_version": "stage2_artifact_schema_v1",
        }
    )


def _modality_route_for_page_input(page_input: Dict[str, Any]) -> str:
    has_text = bool(page_input.get("page_text"))
    has_image = bool(page_input.get("page_image_path"))
    if has_text and has_image:
        return "mixed"
    if has_image:
        return "image"
    return "text"


def _image_payload_sent(page_input: Dict[str, Any], image_payload_mode: str) -> bool:
    if image_payload_mode == "none":
        return False
    image_path = page_input.get("page_image_path")
    if not image_path:
        return False
    return Path(image_path).is_file()


def _image_sha256_unavailable_reason(
    page_input: Dict[str, Any],
    image_payload_mode: str,
    payload_sent: bool,
    image_sha256: str | None,
) -> str | None:
    if image_sha256:
        return None
    if image_payload_mode == "none":
        return "image_payload_mode_none"
    image_path = page_input.get("page_image_path")
    if not image_path:
        return "no_page_image"
    if not Path(image_path).is_file():
        return "image_file_unavailable"
    if not payload_sent:
        return "image_payload_not_sent"
    return "image_sha256_unavailable"


def _build_call_id(doc_id: str, page_index: int, provider_mode: str, prompt_hash: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "doc_id": doc_id,
                "page_index": int(page_index),
                "provider_mode": provider_mode,
                "prompt_hash": prompt_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]


def _build_provider_call_log_row(
    selected_page: Dict[str, Any],
    canonical_record: Dict[str, Any],
    page_input: Dict[str, Any],
    schema_dict: Dict[str, Any],
    page_result: Dict[str, Any],
    image_payload_mode: str,
    provider_mode: str,
    model_version: str | None,
    document_generic: bool,
) -> Dict[str, Any]:
    image_path = page_input.get("page_image_path")
    payload_sent = _image_payload_sent(page_input, image_payload_mode)
    payload_audit = describe_image_payload(image_path if payload_sent else None, image_payload_mode)
    failure_type = None
    provider_error_type = str(page_result.get("provider_error_type") or "")
    if provider_error_type:
        if "ProviderResponseFormatError" in provider_error_type or "JSONDecodeError" in provider_error_type:
            failure_type = "parse_failure"
        else:
            failure_type = "provider_error"
    elif int(page_result.get("num_raw_artifacts", 0)) == 0 and int(page_result.get("num_discarded_artifacts", 0)) > 0:
        failure_type = "parse_failure"
    prompt_hash = _build_call_prompt_hash(canonical_record, page_input, schema_dict, document_generic)
    image_sha256 = _optional_file_sha256(image_path) if payload_sent else None
    doc_id = str(selected_page.get("doc_id"))
    page_index = int(selected_page.get("page_index", 0))
    return {
        "call_id": _build_call_id(doc_id, page_index, str(provider_mode), prompt_hash),
        "doc_id": doc_id,
        "page_id": f"{doc_id}#p{page_index:03d}",
        "page_index": page_index,
        "modality_route": _modality_route_for_page_input(page_input),
        "image_payload_sent": bool(payload_sent),
        "image_payload_mode": str(image_payload_mode),
        "actual_image_payload_kind": payload_audit["actual_image_payload_kind"],
        "image_sha256": image_sha256,
        "image_sha256_unavailable_reason": _image_sha256_unavailable_reason(
            page_input,
            image_payload_mode,
            bool(payload_sent),
            image_sha256,
        ),
        "prompt_hash": prompt_hash,
        "response_schema_version": "stage2_artifact_schema_v1",
        "provider_mode": str(provider_mode),
        "model_version": str(model_version or "unknown"),
        "call_succeeded": page_result.get("provider_error_type") is None,
        "parsed_artifact_count": int(page_result.get("num_raw_artifacts", 0)),
        "discarded_artifact_count": int(page_result.get("num_discarded_artifacts", 0)),
        "failure_type": failure_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _prepare_private_debug_dir(path: str | Path) -> Path:
    debug_dir = Path(path)
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = debug_dir / "raw_outputs.jsonl"
    if raw_output_path.exists():
        raw_output_path.unlink()
    raw_output_path.write_text("", encoding="utf-8")
    return debug_dir


def _current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _minimal_discard_row(
    selected_page: Dict[str, Any],
    reason: str,
    artifact_id: Any = None,
    message: str | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "record_index": int(selected_page["record_index"]),
        "doc_id": str(selected_page["doc_id"]),
        "page_index": int(selected_page["page_index"]),
        "reason": str(reason),
    }
    if artifact_id not in (None, ""):
        row["artifact_id"] = str(artifact_id)
    if message:
        row["message"] = str(message)
    return row




def _project_final_artifact(selected_page: Dict[str, Any], artifact: Dict[str, Any]) -> Dict[str, Any]:
    projected = {field: artifact.get(field) for field in FINAL_ARTIFACT_FIELDS if field not in {"record_index"}}
    projected["record_index"] = int(selected_page["record_index"])
    projected["doc_id"] = str(selected_page["doc_id"])
    projected["page_index"] = int(selected_page["page_index"])
    projected["page_id"] = str(artifact.get("page_id") or page_id_for(projected["doc_id"], projected["page_index"]))
    projected["status"] = str(artifact.get("status") or artifact.get("validation_status") or "candidate")
    projected["validation_status"] = str(artifact.get("validation_status") or projected["status"])
    projected["locators"] = artifact.get("locators") if isinstance(artifact.get("locators"), list) else []
    classification = classify_artifact_locator(projected)
    projected["source_anchored"] = bool(artifact.get("source_anchored", classification["source_anchored"]))
    projected["element_locatable"] = bool(artifact.get("element_locatable", classification["element_locatable"]))
    projected["proof_trace_eligible"] = bool(artifact.get("proof_trace_eligible", classification["proof_trace_eligible"]))
    return {field: projected.get(field) for field in FINAL_ARTIFACT_FIELDS}


def _has_artifact_locator(artifact: Dict[str, Any]) -> bool:
    return bool(classify_artifact_locator(artifact)["source_anchored"])


def _has_artifact_bbox_locator(artifact: Dict[str, Any]) -> bool:
    return bool(stage2_locator_kind_counts(artifact).get("bbox"))


def _count_blocking_reasons(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        stage2 = record.get("stage2", {})
        if not isinstance(stage2, dict):
            continue
        preflight = stage2.get("preflight", {})
        if not isinstance(preflight, dict):
            continue
        for reason in preflight.get("blocking_reasons", []) or []:
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _build_quality_report_from_files(
    records: List[Dict[str, Any]],
    selected_pages: List[Dict[str, Any]],
    artifacts_path: str | Path,
    discard_path: str | Path,
    call_log_path: str | Path | None = None,
) -> Dict[str, Any]:
    artifacts = _read_jsonl(artifacts_path)
    discarded = _read_jsonl(discard_path)
    call_rows = _read_jsonl(call_log_path) if call_log_path is not None else []
    artifact_type_counts = Counter(str(row.get("artifact_type")) for row in artifacts if row.get("artifact_type"))
    modality_counts = Counter(str(row.get("modality")) for row in artifacts if row.get("modality"))
    status_counts = Counter(str(row.get("status") or row.get("validation_status") or "unknown") for row in artifacts)
    locator_kind_counter: Counter[str] = Counter()
    proof_trace_eligible_by_type: Counter[str] = Counter()
    num_source_anchored = 0
    num_element_locatable = 0
    num_proof_trace_eligible = 0
    for row in artifacts:
        classification = classify_artifact_locator(row)
        if classification["source_anchored"]:
            num_source_anchored += 1
        if classification["element_locatable"]:
            num_element_locatable += 1
        if classification["proof_trace_eligible"]:
            num_proof_trace_eligible += 1
            proof_trace_eligible_by_type[str(row.get("artifact_type") or "unknown")] += 1
        kind_counts = stage2_locator_kind_counts(row)
        if kind_counts:
            locator_kind_counter.update(kind_counts)
        else:
            locator_kind_counter[str(classification["locator_kind"])] += 1
    locator_counts = Counter(
        "with_locator" if _has_artifact_locator(row) else "uncertain_or_unreadable"
        for row in artifacts
    )
    bbox_locator_counts = Counter(
        "with_bbox" if _has_artifact_bbox_locator(row) else "without_bbox"
        for row in artifacts
    )
    denominator = len(artifacts) + len(discarded)
    artifact_denominator = max(1, len(artifacts))
    schema_valid_rate = (len(artifacts) / denominator) if denominator else 1.0
    anchoring_rate = (len(artifacts) / denominator) if denominator else 1.0
    discard_rate = (len(discarded) / denominator) if denominator else 0.0
    return {
        "num_records": len(records),
        "num_documents_attempted": len({page.get("doc_id") for page in selected_pages}),
        "num_pages_attempted": len(selected_pages),
        "num_artifacts": len(artifacts),
        "num_valid_artifacts": len(artifacts),
        "num_discarded_artifacts": len(discarded),
        "schema_valid_rate": schema_valid_rate,
        "anchoring_rate": anchoring_rate,
        "discard_rate": discard_rate,
        "artifact_type_counts": dict(sorted(artifact_type_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "num_source_anchored": num_source_anchored,
        "source_anchored_rate": num_source_anchored / artifact_denominator,
        "num_element_locatable": num_element_locatable,
        "element_locator_rate": num_element_locatable / artifact_denominator,
        "num_proof_trace_eligible": num_proof_trace_eligible,
        "proof_trace_eligible_rate": num_proof_trace_eligible / artifact_denominator,
        "proof_trace_eligible_by_type": dict(sorted(proof_trace_eligible_by_type.items())),
        "locator_kind_counts": dict(sorted(locator_kind_counter.items())),
        "num_artifacts_without_element_locator": len(artifacts) - num_element_locatable,
        "discard_reason_counts": dict(sorted(Counter(str(row.get("reason") or "unknown") for row in discarded).items())),
        "locator_status_counts": dict(sorted(locator_counts.items())),
        "bbox_locator_counts": dict(sorted(bbox_locator_counts.items())),
        "blocking_reason_counts": _count_blocking_reasons(records),
        "storage_format": "artifacts_jsonl",
        "pages_attempted": len(selected_pages),
        "pages_with_image_payload": sum(1 for row in call_rows if row.get("image_payload_sent") is True),
        "pages_without_image_payload": sum(1 for row in call_rows if row.get("image_payload_sent") is False),
        "image_payload_rate": (
            sum(1 for row in call_rows if row.get("image_payload_sent") is True) / len(call_rows)
            if call_rows
            else 0.0
        ),
        "visual_artifact_count": sum(1 for row in artifacts if row.get("artifact_type") == "visual_observation"),
        "figure_artifact_count": sum(1 for row in artifacts if row.get("artifact_type") == "figure"),
        "caption_artifact_count": sum(1 for row in artifacts if row.get("artifact_type") == "caption"),
        "table_artifact_count": sum(1 for row in artifacts if row.get("artifact_type") == "table"),
        "table_cell_artifact_count": sum(1 for row in artifacts if row.get("artifact_type") == "table_cell"),
        "numeric_fact_count": sum(1 for row in artifacts if row.get("artifact_type") == "numeric_fact"),
        "visual_region_count": sum(1 for row in artifacts if row.get("artifact_type") == "visual_region"),
        "empty_response_count": sum(1 for row in call_rows if int(row.get("parsed_artifact_count", 0)) == 0 and row.get("call_succeeded") is True),
        "parse_failure_count": sum(1 for row in call_rows if row.get("failure_type") == "parse_failure"),
        "schema_failure_count": sum(1 for row in discarded if str(row.get("reason", "")).startswith("schema") or row.get("reason") == "missing_required_field"),
        "anchor_failure_count": sum(1 for row in discarded if row.get("reason") in {"source_anchor_not_found", "missing_source_anchor", "provenance_source_not_found"}),
        "provider_call_success_count": sum(1 for row in call_rows if row.get("call_succeeded") is True),
        "provider_call_failed_count": sum(1 for row in call_rows if row.get("call_succeeded") is False),
        "json_parse_success_count": sum(1 for row in call_rows if row.get("call_succeeded") is True),
        "schema_valid_artifact_count": len(artifacts),
        "anchored_artifact_count": num_source_anchored,
        "discarded_artifact_count": len(discarded),
    }


def _write_quality_report(report: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _public_stage2_command_args(args: argparse.Namespace | None) -> Dict[str, Any]:
    if args is None:
        return {}
    public_args: Dict[str, Any] = {
        "provider": str(getattr(args, "provider", "")),
        "provider_mode": str(getattr(args, "provider_mode", getattr(args, "provider", ""))),
        "model_name_recorded": bool(getattr(args, "model_name", None)),
        "enable_real_api": bool(getattr(args, "enable_real_api", False)),
        "run_real_trial": bool(getattr(args, "run_real_trial", False)),
        "max_docs": int(getattr(args, "max_docs", 0)) if getattr(args, "max_docs", None) is not None else None,
        "max_pages_per_doc": int(getattr(args, "max_pages_per_doc", 0)) if getattr(args, "max_pages_per_doc", None) is not None else None,
        "max_pages": int(getattr(args, "max_pages", 0)) if getattr(args, "max_pages", None) is not None else None,
        "max_pages_total": int(getattr(args, "max_pages_total", 0)) if getattr(args, "max_pages_total", None) is not None else None,
        "max_pages_per_call": int(getattr(args, "max_pages_per_call", 1)),
        "max_pages_real_cap": int(getattr(args, "max_pages_real_cap", 10)),
        "subset_file_used": bool(getattr(args, "subset_file", None)),
        "scope_mode": str(getattr(args, "scope_mode", "doc_first")),
        "retrieval_topk_file_used": bool(getattr(args, "retrieval_topk_file", None)),
        "retrieval_topk": int(getattr(args, "retrieval_topk", 0)) if getattr(args, "retrieval_topk", None) is not None else None,
        "image_payload_mode": str(getattr(args, "image_payload_mode", "image_url")),
        "save_private_debug": bool(getattr(args, "save_private_debug", False)),
        "deterministic_dedup_enabled": bool(getattr(args, "deterministic_dedup_enabled", True)),
        "max_retries": int(getattr(args, "max_retries", 0)) if getattr(args, "max_retries", None) is not None else None,
    }
    if getattr(args, "max_pages_real", None) is not None:
        public_args["max_pages_real"] = int(getattr(args, "max_pages_real"))
    return public_args


def _write_stage2_jsonl_manifest(
    *,
    output_dir: str | Path,
    stage2_json: str | Path,
    prompt_version: str,
    model_version: str | None,
    report: Dict[str, Any],
    document_generic: bool,
    call_log_path: str | Path | None = None,
    args: argparse.Namespace | None = None,
) -> Dict[str, Any]:
    root = Path(output_dir)
    artifacts_path = root / "artifacts.jsonl"
    discard_path = root / "discard.jsonl"
    quality_report_path = root / "quality_report.json"
    call_rows = _read_jsonl(call_log_path) if call_log_path is not None else []
    provider_modes = sorted({str(row.get("provider_mode")) for row in call_rows if row.get("provider_mode")})
    image_payload_modes = sorted(
        {str(row.get("image_payload_mode")) for row in call_rows if row.get("image_payload_mode")}
    )
    actual_image_payload_kinds = sorted(
        {str(row.get("actual_image_payload_kind")) for row in call_rows if row.get("actual_image_payload_kind")}
    )
    image_sha256_values = sorted(
        {str(row.get("image_sha256")) for row in call_rows if row.get("image_sha256")}
    )
    input_path = Path(stage2_json)
    manifest = {
        "schema_version": "stage2_artifact_schema_v1",
        "compiler_mode": "document_generic" if document_generic else "crossdoc_clean",
        "prompt_version": str(prompt_version),
        "model_version": str(model_version or "unknown"),
        **stage2_model_fields(
            str(getattr(args, "provider_mode", getattr(args, "provider", "dry_run")) if args is not None else "dry_run"),
            getattr(args, "model_config", None) if args is not None else None,
        ),
        "git_commit": _current_git_commit(),
        "num_artifacts": int(report.get("num_artifacts", 0)),
        "num_nodes": 0,
        "num_edges": 0,
        "edge_provenance_summary": {},
        "document_generic": bool(document_generic),
        "artifacts_hash": _file_sha256(artifacts_path) if artifacts_path.is_file() else "missing",
        "discard_hash": _file_sha256(discard_path) if discard_path.is_file() else "missing",
        "call_log_hash": _file_sha256(call_log_path) if call_log_path is not None and Path(call_log_path).is_file() else "missing",
        "quality_report_hash": _file_sha256(quality_report_path) if quality_report_path.is_file() else "missing",
        "provider_call_count": len(call_rows),
        "pages_with_image_payload": sum(1 for row in call_rows if row.get("image_payload_sent") is True),
        "pages_without_image_payload": sum(1 for row in call_rows if row.get("image_payload_sent") is False),
        "provider_modes": provider_modes,
        "image_payload_modes": image_payload_modes,
        "actual_image_payload_kinds": actual_image_payload_kinds,
        "image_sha256_values": image_sha256_values,
        "image_sha256_summary": {
            "count": len(image_sha256_values),
            "values": image_sha256_values,
        },
        "created_by_script": "scripts/stage2.py doc-compile" if document_generic else "scripts/stage2.py compile",
        "command_args": _public_stage2_command_args(args),
        "forbidden_fields_checked": True,
        "no_public_leakage_checked": True,
        "public_provider_outputs_written": (root / "raw_outputs.jsonl").is_file(),
        "private_debug_enabled": bool(getattr(args, "save_private_debug", False)) if args is not None else False,
        "private_debug_dir_recorded_as_public": False,
        "provider_body_redacted": True,
        "encoded_payload_redacted": True,
        "filesystem_locations_redacted": True,
        "credentials_redacted": True,
    }
    if input_path.is_file():
        manifest["input_hash"] = _file_sha256(input_path)
    else:
        manifest["input_hash_unavailable_reason"] = "input_file_missing"
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _build_crossdoc_compiler_metadata(args: argparse.Namespace, api_config: Any) -> Dict[str, Any]:
    return {
        "compiler_name": "real_api_artifact_compiler_client" if not args.dry_run_fake_client else "fake_artifact_compiler_client",
        "provider": args.provider,
        "model_name": api_config.model_name,
        "temperature": api_config.temperature,
        "max_repair_attempts": int(getattr(args, "max_retries", 0)),
    }


def _compile_selected_page_to_jsonl(
    selected_page: Dict[str, Any],
    extract_root: str | Path,
    output_paths: Dict[str, Path | None],
    client: ArtifactCompilerClient,
    schema_dict: Dict[str, Any],
    compiler_metadata: Dict[str, Any],
    prompt_version: str,
    deterministic_dedup_enabled: bool,
    max_retries: int = 0,
    document_generic: bool = False,
    raw_output_log_path: str | Path | None = None,
    call_log_path: str | Path | None = None,
    image_payload_mode: str = "image_url",
    provider_mode: str = "real",
    model_version: str | None = None,
) -> Dict[str, Any]:
    canonical_record = (
        build_document_generic_compiler_record(selected_page)
        if document_generic
        else build_compiler_safe_record(selected_page)
    )
    canonical_record["compilation_plan"]["compile_scope"] = (
        "stage2_document_generic_single_page" if document_generic else "stage2_clean_single_page"
    )
    if document_generic:
        page_input = build_page_input(_strip_stage2_routes_for_document_generic(selected_page), extract_root)
    else:
        page_input = build_page_input(selected_page, extract_root)
        page_input["page_modality_diagnosis"] = selected_page.get("page_modality_diagnosis") or diagnose_page_modality_from_question_and_preflight(
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
    try:
        compile_result = compile_page_with_client(
            canonical_record=canonical_record,
            page_input=page_input,
            client=client,
            schema_dict=schema_dict,
            compiler_metadata=compiler_metadata,
            raw_output_log_path=raw_output_log_path if raw_output_log_path is not None else output_paths.get("raw_outputs"),
            discard_log_path=None,
            compiler_version=COMPILER_VERSION,
            prompt_version=prompt_version,
            deterministic_dedup_enabled=deterministic_dedup_enabled,
            max_retries=max_retries,
            document_generic=document_generic,
        )
    except Exception as exc:
        _append_jsonl(
            Path(output_paths["discard"]),
            _minimal_discard_row(selected_page, reason="provider_error", message=type(exc).__name__),
        )
        page_result = {
            "record_index": selected_page["record_index"],
            "doc_id": selected_page["doc_id"],
            "page_index": int(selected_page["page_index"]),
            "num_raw_artifacts": 0,
            "num_valid_artifacts": 0,
            "num_discarded_artifacts": 1,
            "provider_error_type": type(exc).__name__,
            "retry_count": 0,
            "document_generic": bool(document_generic),
        }
        if call_log_path is not None:
            _append_jsonl(
                call_log_path,
                _build_provider_call_log_row(
                    selected_page=selected_page,
                    canonical_record=canonical_record,
                    page_input=page_input,
                    schema_dict=schema_dict,
                    page_result=page_result,
                    image_payload_mode=image_payload_mode,
                    provider_mode=provider_mode,
                    model_version=model_version,
                    document_generic=document_generic,
                ),
            )
        return page_result

    discarded = 0
    seen_discard_keys: set[tuple[Any, str, str]] = set()
    for issue in compile_result.get("validation_issues", []):
        reason = str(issue.get("error_type", "schema_invalid"))
        artifact_id = issue.get("artifact_id")
        message = str(issue.get("message", ""))
        key = (artifact_id, reason, message)
        if key in seen_discard_keys:
            continue
        seen_discard_keys.add(key)
        _append_jsonl(Path(output_paths["discard"]), _minimal_discard_row(selected_page, reason, artifact_id, message))
        discarded += 1

    written = 0
    for artifact in compile_result.get("valid_artifacts", []):
        if artifact.get("artifact_type") not in FINAL_ARTIFACT_TYPES or artifact.get("modality") not in FINAL_MODALITIES:
            _append_jsonl(
                Path(output_paths["discard"]),
                _minimal_discard_row(
                    selected_page,
                    reason="unsupported_artifact_type_or_modality",
                    artifact_id=artifact.get("artifact_id"),
                ),
            )
            discarded += 1
            continue
        _append_jsonl(Path(output_paths["artifacts"]), _project_final_artifact(selected_page, artifact))
        written += 1

    stats = compile_result.get("compilation_statistics", {})
    page_result = {
        "record_index": selected_page["record_index"],
        "doc_id": selected_page["doc_id"],
        "page_index": int(selected_page["page_index"]),
        "num_raw_artifacts": int(stats.get("num_raw_artifacts", 0)),
        "num_valid_artifacts": written,
        "num_discarded_artifacts": discarded,
        "provider_error_type": None,
        "retry_count": int(compile_result.get("compilation_statistics", {}).get("retry_count", 0)),
        "document_generic": bool(document_generic),
    }
    if call_log_path is not None:
        _append_jsonl(
            call_log_path,
            _build_provider_call_log_row(
                selected_page=selected_page,
                canonical_record=canonical_record,
                page_input=page_input,
                schema_dict=schema_dict,
                page_result=page_result,
                image_payload_mode=image_payload_mode,
                provider_mode=provider_mode,
                model_version=model_version,
                document_generic=document_generic,
            ),
        )
    return page_result


def run_crossdoc_batch(args: argparse.Namespace, client: ArtifactCompilerClient | None = None) -> Dict[str, Any]:
    validate_crossdoc_args(args)
    debug_raw_output = bool(getattr(args, "debug_raw_output", False))
    output_paths = _prepare_final_output_dir(args.output_dir, debug_raw_output=debug_raw_output)
    api_config = build_crossdoc_api_config(args)
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
    compiler_metadata = _build_crossdoc_compiler_metadata(args, api_config)

    page_results: List[Dict[str, Any]] = []
    for selected_page in selected_pages:
        page_results.append(
            _compile_selected_page_to_jsonl(
                selected_page=selected_page,
                extract_root=args.extract_root,
                output_paths=output_paths,
                client=active_client,
                schema_dict=schema_dict,
                compiler_metadata=compiler_metadata,
                prompt_version=str(getattr(args, "prompt_version", PROMPT_VERSION)),
                deterministic_dedup_enabled=bool(getattr(args, "deterministic_dedup_enabled", True)),
            )
        )

    report = _build_quality_report_from_files(
        records=records,
        selected_pages=selected_pages,
        artifacts_path=Path(output_paths["artifacts"]),
        discard_path=Path(output_paths["discard"]),
        call_log_path=output_paths.get("call_log"),
    )
    _write_quality_report(report, Path(output_paths["quality_report"]))
    manifest = _write_stage2_jsonl_manifest(
        output_dir=args.output_dir,
        stage2_json=args.stage2_json,
        prompt_version=str(getattr(args, "prompt_version", PROMPT_VERSION)),
        model_version=api_config.model_name,
        report=report,
        document_generic=bool(getattr(args, "document_generic", False)),
        call_log_path=output_paths.get("call_log"),
        args=args,
    )
    return {
        "summary": report,
        "manifest": manifest,
        "page_results": page_results,
        "paths": {key: str(value) for key, value in output_paths.items() if value is not None},
    }


def _build_compile_scope_quality_fields(
    selected_pages: List[Dict[str, Any]],
    report: Dict[str, Any],
    scope_mode: str,
) -> Dict[str, Any]:
    num_pages = len(selected_pages)
    num_docs = len({str(page.get("doc_id")) for page in selected_pages if page.get("doc_id") not in (None, "")})
    num_artifacts = int(report.get("num_artifacts", 0))
    source_counts = Counter(str(page.get("selection_source") or page.get("selection_reason") or "unknown") for page in selected_pages)
    return {
        "compile_scope_mode": str(scope_mode),
        "num_selected_docs": int(num_docs),
        "num_selected_pages": int(num_pages),
        "num_artifacts_per_doc_avg": num_artifacts / max(1, num_docs),
        "num_artifacts_per_page_avg": num_artifacts / max(1, num_pages),
        "page_selection_source_counts": dict(sorted(source_counts.items())),
    }


def run_index_command(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output) if getattr(args, "output", None) else _stage2_index_path(args.output_dir)
    records = augment_retrieval_results_file(
        input_path=args.input,
        output_path=output,
        extract_root=args.extract_root,
        config_path=args.config,
        max_records=args.max_records,
    )
    return {
        "command": "index",
        "output": str(output),
        "num_records": len(records),
        "schema": "compact_page_routes",
        "will_call_api": False,
        "will_generate_artifact": False,
    }


def run_compile_command(args: argparse.Namespace) -> Dict[str, Any]:
    args.stage2_json = args.stage2_json or str(_stage2_index_path(args.output_dir))
    args.config = args.config or DEFAULT_CONFIG
    args.extract_root = args.extract_root or DEFAULT_EXTRACT_ROOT
    args.output_dir = args.output_dir or DEFAULT_CLEAN_BATCH_DIR
    args.document_generic = bool(getattr(args, "document_generic", False))
    return run_crossdoc_batch(args)


def _resolve_doc_compile_max_pages(args: argparse.Namespace) -> int:
    raw_max_pages_total = getattr(args, "max_pages_total", None)
    if raw_max_pages_total not in (None, ""):
        raw_value = raw_max_pages_total
    else:
        raw_value = None
    raw_max_pages_real = getattr(args, "max_pages_real", None)
    if raw_value is not None:
        pass
    elif raw_max_pages_real not in (None, ""):
        raw_value = raw_max_pages_real
    elif hasattr(args, "max_pages"):
        raw_value = getattr(args, "max_pages")
    else:
        raw_value = DOC_COMPILE_DEFAULT_MAX_REAL_PAGES
    if raw_value in (None, ""):
        raise RuntimeError("Real provider doc-compile requires finite --max-pages or --max-pages-real.")
    try:
        max_pages = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Real provider doc-compile requires finite --max-pages or --max-pages-real.") from exc
    if max_pages < 1:
        raise RuntimeError("--max-pages must be at least 1.")
    return max_pages


def _validate_doc_compile_real_args(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "enable_real_api", False)):
        raise RuntimeError("Refusing real provider doc-compile without --enable-real-api.")
    if not bool(getattr(args, "run_real_trial", False)):
        raise RuntimeError("Refusing real provider doc-compile without --run-real-trial.")
    validate_real_page_limits(args)
    _ = _resolve_doc_compile_max_pages(args)


def _validate_private_debug_dir(private_debug_dir: str | Path, output_dir: str | Path) -> None:
    debug_path = Path(private_debug_dir).resolve()
    public_root = Path(output_dir).resolve()
    if debug_path == public_root or public_root in debug_path.parents:
        raise RuntimeError("--private-debug-dir must not be inside the public Stage 2 output directory.")


def run_doc_compile_command(args: argparse.Namespace) -> Dict[str, Any]:
    args.stage2_json = args.stage2_json or str(_stage2_index_path(args.output_dir))
    args.config = args.config or DEFAULT_CONFIG
    args.extract_root = args.extract_root or DEFAULT_EXTRACT_ROOT
    args.output_dir = args.output_dir or DEFAULT_DOC_GENERIC_BATCH_DIR
    args.max_docs = int(getattr(args, "max_docs", 5))
    args.max_pages_per_doc = int(getattr(args, "max_pages_per_doc", 2))
    args.max_pages = _resolve_doc_compile_max_pages(args)
    if getattr(args, "max_pages_real", None) not in (None, ""):
        args.max_pages_real = int(getattr(args, "max_pages_real"))
    args.selected_pages_csv = getattr(args, "selected_pages_csv", None)
    args.prompt_version = getattr(args, "prompt_version", PROMPT_VERSION)
    args.debug_raw_output = bool(getattr(args, "debug_raw_output", False))
    args.max_retries = int(getattr(args, "max_retries", 2))
    args.image_payload_mode = _image_payload_mode(args)
    args.save_private_debug = bool(getattr(args, "save_private_debug", False))
    args.private_debug_dir = str(getattr(args, "private_debug_dir", "outputs_private/stage2_debug/"))
    args.subset_file = getattr(args, "subset_file", None)
    args.scope_mode = str(getattr(args, "scope_mode", "doc_first") or "doc_first")
    args.retrieval_topk_file = getattr(args, "retrieval_topk_file", None)
    args.retrieval_topk = int(getattr(args, "retrieval_topk", 5))
    args.model_config = getattr(args, "model_config", QWEN3VL_CONFIG)
    args.provider_mode = _normalize_doc_compile_provider_mode(args)
    if (args.subset_file or args.retrieval_topk_file) and args.provider_mode == "real":
        raise RuntimeError("Refusing real provider doc-compile for fixed coverage subset runs.")
    if args.provider_mode in {"dry_run", "fake"}:
        args.dry_run_fake_client = True
        args.enable_real_api = False
        args.run_real_trial = False
    else:
        args.dry_run_fake_client = False
        args.enable_real_api = bool(getattr(args, "enable_real_api", False))
        args.run_real_trial = bool(getattr(args, "run_real_trial", False))
        _validate_doc_compile_real_args(args)
    args.document_generic = True
    return run_document_generic_batch(args)


def _read_document_generic_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    stage2_json = getattr(args, "stage2_json", None)
    if stage2_json not in (None, "") and Path(stage2_json).is_file():
        return read_json_or_jsonl_records(stage2_json)
    if getattr(args, "subset_file", None) not in (None, "") or getattr(args, "retrieval_topk_file", None) not in (None, ""):
        return []
    return read_json_or_jsonl_records(stage2_json)

def run_document_generic_batch(args: argparse.Namespace, client: ArtifactCompilerClient | None = None) -> Dict[str, Any]:
    validate_crossdoc_args(args)
    output_paths = _prepare_final_output_dir(args.output_dir, debug_raw_output=False)
    if bool(getattr(args, "save_private_debug", False)):
        _validate_private_debug_dir(getattr(args, "private_debug_dir", "outputs_private/stage2_debug/"), args.output_dir)
    private_debug_dir = (
        _prepare_private_debug_dir(getattr(args, "private_debug_dir", "outputs_private/stage2_debug/"))
        if bool(getattr(args, "save_private_debug", False))
        else None
    )
    raw_output_log_path = private_debug_dir / "raw_outputs.jsonl" if private_debug_dir is not None else None
    api_config = build_crossdoc_api_config(args)
    api_config.image_payload_mode = _image_payload_mode(args)
    records = _read_document_generic_records(args)
    selected_pages = select_document_generic_pages(
        records=records,
        extract_root=args.extract_root,
        max_docs=int(args.max_docs),
        max_pages_per_doc=int(args.max_pages_per_doc),
        max_pages=int(args.max_pages),
        subset_file=getattr(args, "subset_file", None),
        scope_mode=str(getattr(args, "scope_mode", "doc_first")),
        retrieval_topk_file=getattr(args, "retrieval_topk_file", None),
        retrieval_topk=int(getattr(args, "retrieval_topk", 5)),
    )
    active_client = client or (FakeArtifactCompilerClient() if args.dry_run_fake_client else RealApiArtifactCompilerClient(api_config))
    schema_dict = build_page_artifact_output_schema_dict()
    compiler_metadata = _build_crossdoc_compiler_metadata(args, api_config)
    page_results: List[Dict[str, Any]] = []
    for selected_page in selected_pages:
        page_results.append(
            _compile_selected_page_to_jsonl(
                selected_page=selected_page,
                extract_root=args.extract_root,
                output_paths=output_paths,
                client=active_client,
                schema_dict=schema_dict,
                compiler_metadata=compiler_metadata,
                prompt_version=str(getattr(args, "prompt_version", PROMPT_VERSION)),
                deterministic_dedup_enabled=bool(getattr(args, "deterministic_dedup_enabled", True)),
                max_retries=int(getattr(args, "max_retries", 0)),
                document_generic=True,
                raw_output_log_path=raw_output_log_path,
                call_log_path=output_paths.get("call_log"),
                image_payload_mode=_image_payload_mode(args),
                provider_mode=str(getattr(args, "provider_mode", "real")),
                model_version=api_config.model_name,
            )
        )
    report = _build_quality_report_from_files(
        records=records,
        selected_pages=selected_pages,
        artifacts_path=Path(output_paths["artifacts"]),
        discard_path=Path(output_paths["discard"]),
        call_log_path=output_paths.get("call_log"),
    )
    report["document_generic"] = True
    report.update(_build_compile_scope_quality_fields(selected_pages, report, str(getattr(args, "scope_mode", "doc_first"))))
    _write_quality_report(report, Path(output_paths["quality_report"]))
    manifest = _write_stage2_jsonl_manifest(
        output_dir=args.output_dir,
        stage2_json=args.stage2_json,
        prompt_version=str(getattr(args, "prompt_version", PROMPT_VERSION)),
        model_version=api_config.model_name,
        report=report,
        document_generic=True,
        call_log_path=output_paths.get("call_log"),
        args=args,
    )
    return {
        "summary": report,
        "manifest": manifest,
        "page_results": page_results,
        "paths": {key: str(value) for key, value in output_paths.items() if value is not None},
    }


def select_document_generic_pages(
    records: List[Dict[str, Any]],
    extract_root: str | Path,
    max_docs: int,
    max_pages_per_doc: int,
    max_pages: int,
    subset_file: str | Path | None = None,
    scope_mode: str = "doc_first",
    retrieval_topk_file: str | Path | None = None,
    retrieval_topk: int = 5,
) -> List[Dict[str, Any]]:
    if subset_file not in (None, ""):
        return select_document_generic_pages_from_subset(
            subset_file=subset_file,
            records=records,
            extract_root=extract_root,
            max_docs=max_docs,
            max_pages_per_doc=max_pages_per_doc,
            max_pages=max_pages,
        )
    if str(scope_mode) == "retrieval_topk_scope" and retrieval_topk_file not in (None, ""):
        return select_document_generic_pages_from_retrieval_topk(
            retrieval_topk_file=retrieval_topk_file,
            extract_root=extract_root,
            max_docs=max_docs,
            max_pages_per_doc=max_pages_per_doc,
            max_pages=max_pages,
            top_k=retrieval_topk,
        )

    selected: List[Dict[str, Any]] = []
    doc_counts: Counter[str] = Counter()
    for record_index, record in enumerate(records):
        doc_id = record.get("doc_id")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        if len(doc_counts) >= int(max_docs) and doc_id not in doc_counts:
            continue
        page_indices = discover_document_page_indices(doc_id, extract_root)
        if not page_indices:
            page_indices = document_generic_candidate_page_indices(record)
        for page_index in page_indices:
            if len(selected) >= int(max_pages):
                return selected
            if doc_counts[doc_id] >= int(max_pages_per_doc):
                break
            page_source = build_page_source(doc_id, extract_root, int(page_index))
            if not page_source.get("has_page_text") and not page_source.get("has_page_image"):
                continue
            selected.append(
                {
                    "record_index": int(record_index),
                    "doc_id": doc_id,
                    "question": None,
                    "answer_format": None,
                    "page_index": int(page_index),
                    "page_number_one_based": int(page_index) + 1,
                    "selection_reason": "document_generic_page_available",
                    "selection_source": str(scope_mode),
                    "page_image_path": page_source.get("page_image_path"),
                    "page_text_path": page_source.get("page_text_path"),
                    "layout_block_ids": list(page_source.get("layout_block_ids", [])),
                    "stage2": record.get("stage2", {}) if isinstance(record.get("stage2"), dict) else {},
                }
            )
            doc_counts[doc_id] += 1
    return selected


def select_document_generic_pages_from_subset(
    subset_file: str | Path,
    records: List[Dict[str, Any]],
    extract_root: str | Path,
    max_docs: int,
    max_pages_per_doc: int,
    max_pages: int,
) -> List[Dict[str, Any]]:
    record_index_by_doc = _record_index_by_doc_id(records)
    selected: List[Dict[str, Any]] = []
    for subset_row in _load_subset_doc_page_indices(subset_file, extract_root):
        doc_id = str(subset_row["doc_id"])
        subset_page_indices = list(subset_row.get("page_indices", []))
        selection_source = str(subset_row.get("selection_source") or "coverage_subset")
        if len({page["doc_id"] for page in selected}) >= int(max_docs):
            break
        page_indices = subset_page_indices or discover_document_page_indices(doc_id, extract_root) or [0]
        pages_added_for_doc = 0
        for page_index in sorted({int(index) for index in page_indices if int(index) >= 0}):
            if len(selected) >= int(max_pages):
                return selected
            if pages_added_for_doc >= int(max_pages_per_doc):
                break
            page_source = build_page_source(doc_id, extract_root, int(page_index))
            if not page_source.get("has_page_text") and not page_source.get("has_page_image"):
                continue
            selected.append(
                {
                    "record_index": int(record_index_by_doc.get(doc_id, len(selected))),
                    "doc_id": doc_id,
                    "question": None,
                    "answer_format": None,
                    "page_index": int(page_index),
                    "page_number_one_based": int(page_index) + 1,
                    "selection_reason": "document_generic_coverage_subset",
                    "selection_source": selection_source,
                    "page_image_path": page_source.get("page_image_path"),
                    "page_text_path": page_source.get("page_text_path"),
                    "layout_block_ids": list(page_source.get("layout_block_ids", [])),
                    "stage2": {},
                }
            )
            pages_added_for_doc += 1
    return selected


def select_document_generic_pages_from_retrieval_topk(
    retrieval_topk_file: str | Path,
    extract_root: str | Path,
    max_docs: int,
    max_pages_per_doc: int,
    max_pages: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    records = read_json_or_jsonl_records(retrieval_topk_file)
    page_scores_by_doc: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for record in records:
        if not isinstance(record, dict) or record.get("doc_id") in (None, ""):
            continue
        doc_id = str(record["doc_id"])
        for page_index, score in _retrieval_topk_pages(record, int(top_k)):
            if page_index >= 0:
                page_scores_by_doc[doc_id][page_index] += score
    selected: List[Dict[str, Any]] = []
    for doc_id in sorted(page_scores_by_doc)[: int(max_docs)]:
        ranked_pages = sorted(page_scores_by_doc[doc_id], key=lambda page: (-page_scores_by_doc[doc_id][page], page))
        pages_added_for_doc = 0
        for page_index in ranked_pages:
            if len(selected) >= int(max_pages):
                return selected
            if pages_added_for_doc >= int(max_pages_per_doc):
                break
            page_source = build_page_source(doc_id, extract_root, int(page_index))
            if not page_source.get("has_page_text") and not page_source.get("has_page_image"):
                continue
            selected.append(
                {
                    "record_index": len(selected),
                    "doc_id": doc_id,
                    "question": None,
                    "answer_format": None,
                    "page_index": int(page_index),
                    "page_number_one_based": int(page_index) + 1,
                    "selection_reason": "document_generic_retrieval_topk_scope",
                    "selection_source": "retrieval_topk_non_gold",
                    "page_image_path": page_source.get("page_image_path"),
                    "page_text_path": page_source.get("page_text_path"),
                    "layout_block_ids": list(page_source.get("layout_block_ids", [])),
                    "stage2": {},
                }
            )
            pages_added_for_doc += 1
    return selected


def _retrieval_topk_pages(record: Dict[str, Any], top_k: int) -> List[tuple[int, float]]:
    pages: List[tuple[int, float]] = []
    for key in sorted(record):
        key_text = str(key)
        lowered = key_text.lower()
        if _is_scope_forbidden_key(key_text):
            continue
        if "top" not in lowered or "score" in lowered:
            continue
        values = record.get(key)
        if not isinstance(values, list):
            continue
        for rank, value in enumerate(values[: int(top_k)]):
            try:
                page_index = int(value)
            except (TypeError, ValueError):
                continue
            pages.append((page_index, 1.0 / float(rank + 1)))
    return pages


def _is_scope_forbidden_key(key: str) -> bool:
    return key in {"answer", "gold_answer", "evidence_pages", "evidence_sources", "binary_correctness"} or str(key).startswith("gold_")


def _record_index_by_doc_id(records: List[Dict[str, Any]]) -> Dict[str, int]:
    index_by_doc: Dict[str, int] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("doc_id") in (None, ""):
            continue
        index_by_doc.setdefault(str(record["doc_id"]), int(record.get("record_index", record_index)))
    return index_by_doc


def _load_subset_doc_page_indices(subset_file: str | Path, extract_root: str | Path) -> List[Dict[str, Any]]:
    subset_path = Path(subset_file)
    rows = _read_jsonl(subset_path) if subset_path.suffix.lower() == ".jsonl" else read_json_or_jsonl_records(subset_path)
    page_indices_by_doc: Dict[str, List[int]] = {}
    selection_source_by_doc: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("doc_id") in (None, ""):
            continue
        doc_id = str(row["doc_id"])
        if doc_id not in page_indices_by_doc:
            page_indices_by_doc[doc_id] = []
        if row.get("selection_source") not in (None, ""):
            selection_source_by_doc.setdefault(doc_id, str(row.get("selection_source")))
        raw_page_indices = row.get("page_indices")
        if isinstance(raw_page_indices, list):
            for value in raw_page_indices:
                try:
                    page_indices_by_doc[doc_id].append(int(value))
                except (TypeError, ValueError):
                    continue
    result: List[Dict[str, Any]] = []
    for doc_id in sorted(page_indices_by_doc):
        page_indices = sorted({index for index in page_indices_by_doc[doc_id] if index >= 0})
        if not page_indices:
            page_indices = discover_document_page_indices(doc_id, extract_root)
        result.append({"doc_id": doc_id, "page_indices": page_indices, "selection_source": selection_source_by_doc.get(doc_id, "coverage_subset")})
    return result


def document_generic_candidate_page_indices(record: Dict[str, Any]) -> List[int]:
    candidates: set[int] = set()
    for field_name in ("page_indices", "pages_to_compile", "document_page_indices"):
        values = record.get(field_name)
        if isinstance(values, list):
            for value in values:
                try:
                    candidates.add(int(value))
                except (TypeError, ValueError):
                    continue
    if not candidates:
        candidates.add(0)
    return sorted(index for index in candidates if index >= 0)


def discover_document_page_indices(doc_id: str, extract_root: str | Path) -> List[int]:
    paths = build_mdocagent_extract_paths(extract_root, doc_id, 0)
    doc_name = str(paths["doc_name"])
    root = Path(extract_root)
    indices: set[int] = set()
    for suffix in ("txt", "png"):
        for path in root.glob(f"{doc_name}_*.{suffix}"):
            parsed = _parse_doc_page_index(path, doc_name)
            if parsed is not None:
                indices.add(parsed)
        for directory_name in ("texts", "images"):
            directory = root / directory_name
            if not directory.is_dir():
                continue
            for path in directory.glob(f"{doc_name}_*.{suffix}"):
                parsed = _parse_doc_page_index(path, doc_name)
                if parsed is not None:
                    indices.add(parsed)
    return sorted(index for index in indices if index >= 0)


def _parse_doc_page_index(path: Path, doc_name: str) -> int | None:
    prefix = f"{doc_name}_"
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    raw_index = stem[len(prefix):]
    if not raw_index.isdigit():
        return None
    return int(raw_index)


def run_audit_command(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    stage2_json = Path(args.stage2_json) if args.stage2_json else output_dir / "sample-with-stage2-index.json"
    records = read_json_or_jsonl_records(stage2_json) if stage2_json.is_file() else []
    artifacts_path = Path(args.artifacts_jsonl) if args.artifacts_jsonl else output_dir / "artifacts.jsonl"
    discard_path = Path(args.discard_jsonl) if args.discard_jsonl else output_dir / "discard.jsonl"
    report_path = output_dir / "quality_report.json"
    existing_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    selected_refs = {
        (row.get("record_index"), row.get("doc_id"), row.get("page_index"))
        for row in _read_jsonl(artifacts_path) + _read_jsonl(discard_path)
    }
    selected_pages = [
        {"record_index": ref[0], "doc_id": ref[1], "page_index": ref[2]}
        for ref in sorted(selected_refs, key=lambda item: (str(item[1]), int(item[2] or 0), int(item[0] or 0)))
    ]
    report = _build_quality_report_from_files(records, selected_pages, artifacts_path, discard_path)
    for field_name in ("num_documents_attempted", "num_pages_attempted"):
        if field_name in existing_report:
            report[field_name] = existing_report[field_name]
    _write_quality_report(report, report_path)
    return report


def run_clean_command(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(getattr(args, "output_dir", DEFAULT_CLEAN_BATCH_DIR))
    targets = [
        Path("outputs/stage2/MMLongBench/preflight"),
        Path("outputs/stage2/MMLongBench/sample-with-stage2-preflight.json"),
        Path("outputs/stage2/preflight"),
        Path("outputs/stage2/artifacts_real_trial"),
        Path("outputs/stage2/artifacts_real_batch"),
        Path("outputs/stage2/artifacts_real_crossdoc_batch"),
        Path("outputs/stage2/artifacts_real_crossdoc_batch_refined"),
        Path("outputs/stage2/artifacts_real_crossdoc_batch_refined_replay_dedup"),
        Path("outputs/stage2/artifacts_compact_routes_clean"),
        output_dir,
    ]
    deleted = []
    missing = []
    seen: set[Path] = set()
    for target in targets:
        target = Path(target)
        if target in seen:
            continue
        seen.add(target)
        if not target.exists():
            missing.append(str(target))
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deleted.append(str(target))
    return {"command": "clean", "deleted": deleted, "missing": missing}


def run_all_command(args: argparse.Namespace) -> Dict[str, Any]:
    clean_result = run_clean_command(argparse.Namespace(output_dir=args.output_dir))
    index_path = _stage2_index_path(args.output_dir)
    index_result = run_index_command(
        argparse.Namespace(
            input=args.input,
            output=str(index_path),
            output_dir=args.output_dir,
            extract_root=args.extract_root,
            config=args.config,
            max_records=getattr(args, "max_records", None),
        )
    )
    compile_result = run_compile_command(
        argparse.Namespace(
            stage2_json=str(index_path),
            config=args.config,
            extract_root=args.extract_root,
            output_dir=args.output_dir,
            selected_pages_csv=getattr(args, "selected_pages_csv", None),
            max_docs=int(args.max_docs),
            max_pages_per_doc=int(args.max_pages_per_doc),
            max_pages=int(args.max_pages),
            max_pages_total=getattr(args, "max_pages_total", None),
            max_pages_per_call=getattr(args, "max_pages_per_call", 1),
            max_pages_real_cap=getattr(args, "max_pages_real_cap", 10),
            max_pages_real=getattr(args, "max_pages_real", None),
            provider=args.provider,
            model_name=args.model_name,
            prompt_version=args.prompt_version,
            enable_real_api=bool(args.enable_real_api),
            run_real_trial=bool(args.run_real_trial),
            dry_run_fake_client=bool(args.dry_run_fake_client),
            deterministic_dedup_enabled=bool(args.deterministic_dedup_enabled),
            timeout_seconds=int(args.timeout_seconds),
            debug_raw_output=bool(args.debug_raw_output),
            max_retries=int(getattr(args, "max_retries", 2)),
            image_payload_mode=getattr(args, "image_payload_mode", "image_url"),
            save_private_debug=bool(getattr(args, "save_private_debug", False)),
            private_debug_dir=getattr(args, "private_debug_dir", "outputs_private/stage2_debug/"),
            document_generic=False,
        )
    )
    return {
        "command": "all",
        "clean": clean_result,
        "index": index_result,
        "compile": compile_result["summary"],
        "paths": compile_result["paths"],
    }


def _add_compile_options(
    parser: argparse.ArgumentParser,
    provider_default: str = "siliconflow",
    provider_choices: tuple[str, ...] | None = None,
) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_CLEAN_BATCH_DIR)
    parser.add_argument("--selected-pages-csv", default=None)
    parser.add_argument("--max-documents", "--max-docs", dest="max_docs", type=int, default=5)
    parser.add_argument("--max-pages-per-doc", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-pages-real", type=int, default=None)
    parser.add_argument("--max-pages-total", type=int, default=None)
    parser.add_argument("--max-pages-per-call", type=int, default=1)
    parser.add_argument("--max-pages-real-cap", type=int, default=10)
    parser.add_argument("--provider", default=provider_default, choices=provider_choices)
    parser.add_argument("--model-config", default=QWEN3VL_CONFIG)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--image-payload-mode", choices=("image_url", "base64", "none"), default="image_url")
    parser.add_argument("--save-private-debug", action="store_true")
    parser.add_argument("--private-debug-dir", default="outputs_private/stage2_debug/")
    parser.add_argument("--enable-deterministic-dedup", dest="deterministic_dedup_enabled", action="store_true")
    parser.add_argument("--disable-deterministic-dedup", dest="deterministic_dedup_enabled", action="store_false")
    parser.add_argument("--enable-real-api", action="store_true")
    parser.add_argument("--run-real-trial", action="store_true")
    parser.add_argument("--dry-run-fake-client", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--debug-raw-output", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.set_defaults(deterministic_dedup_enabled=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Stage 2 clean storage entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser("clean", help="Delete old Stage 2 temporary outputs.")
    clean_parser.add_argument("--output-dir", default=DEFAULT_CLEAN_BATCH_DIR)
    clean_parser.set_defaults(func=run_clean_command)

    index_parser = subparsers.add_parser("index", help="Build compact Stage 2 page-route index.")
    index_parser.add_argument("--input", default=DEFAULT_STAGE2_INPUT)
    index_parser.add_argument("--output", default=None)
    index_parser.add_argument("--output-dir", default=DEFAULT_CLEAN_BATCH_DIR)
    index_parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    index_parser.add_argument("--config", default=None)
    index_parser.add_argument("--max-records", type=int, default=None)
    index_parser.set_defaults(func=run_index_command)

    compile_parser = subparsers.add_parser("compile", help="Compile clean artifacts.jsonl storage.")
    compile_parser.add_argument("--input", "--stage2-json", dest="stage2_json", default=None)
    _add_compile_options(compile_parser)
    compile_parser.set_defaults(func=run_compile_command)

    doc_compile_parser = subparsers.add_parser("doc-compile", help="Compile question-independent artifacts into outputs/stage2_doc.")
    doc_compile_parser.add_argument("--input", "--stage2-json", dest="stage2_json", default=None)
    doc_compile_parser.add_argument("--subset-file", default=None)
    doc_compile_parser.add_argument("--scope-mode", choices=("doc_first", "query_doc_all", "retrieval_topk_scope"), default="doc_first")
    doc_compile_parser.add_argument("--retrieval-topk-file", default=None)
    doc_compile_parser.add_argument("--retrieval-topk", type=int, default=5)
    _add_compile_options(
        doc_compile_parser,
        provider_default="dry_run",
        provider_choices=("dry_run", "fake", "real"),
    )
    doc_compile_parser.set_defaults(output_dir=DEFAULT_DOC_GENERIC_BATCH_DIR)
    doc_compile_parser.set_defaults(func=run_doc_compile_command)

    audit_parser = subparsers.add_parser("audit", help="Audit clean artifacts.jsonl storage.")
    audit_parser.add_argument("--output-dir", default=DEFAULT_CLEAN_BATCH_DIR)
    audit_parser.add_argument("--input", "--stage2-json", dest="stage2_json", default=None)
    audit_parser.add_argument("--artifacts-jsonl", default=None)
    audit_parser.add_argument("--discard-jsonl", default=None)
    audit_parser.set_defaults(func=run_audit_command)

    all_parser = subparsers.add_parser("all", help="Clean old outputs, build index, compile, and write quality_report.json.")
    all_parser.add_argument("--input", default=DEFAULT_STAGE2_INPUT)
    all_parser.add_argument("--max-records", type=int, default=None)
    _add_compile_options(all_parser)
    all_parser.set_defaults(func=run_all_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
