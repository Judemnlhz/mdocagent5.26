"""Run a guarded Stage 2 real-API small-batch artifact compilation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.api_config import assert_real_api_allowed
from mdocnexus.stage2.artifact_compiler import compile_page_with_client
from mdocnexus.stage2.artifact_store import build_document_artifact_store, write_artifact_store
from mdocnexus.stage2.batch_page_selector import select_pages_for_small_batch
from mdocnexus.stage2.batch_quality_report import (
    count_forbidden_fields,
    summarize_batch_results,
    write_batch_quality_csv,
    write_batch_summary,
)
from mdocnexus.stage2.compiler_client import ArtifactCompilerClient, FakeArtifactCompilerClient
from mdocnexus.stage2.layout_parser import build_basic_layout_blocks
from mdocnexus.stage2.mdocagent_compat import (
    build_api_run_config_from_mdocagent_yaml,
    read_json_or_jsonl_records,
)
from mdocnexus.stage2.page_loader import load_page_content
from mdocnexus.stage2.real_api_client import RealApiArtifactCompilerClient
from mdocnexus.stage2.schema_serialization import build_page_artifact_output_schema_dict


MAX_ALLOWED_PAGES = 10
COMPILER_VERSION = "stage2_compiler_v1"
PROMPT_VERSION = "artifact_compiler_prompt_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled Stage 2 small-batch artifact compilation.")
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pages", type=int, default=5)
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


def validate_args(args: argparse.Namespace) -> None:
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


def run_small_batch(args: argparse.Namespace, client: ArtifactCompilerClient | None = None) -> Dict[str, Any]:
    validate_args(args)
    api_config = build_api_config(args)
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


def build_page_input(selected_page: Dict[str, Any], extract_root: str | Path) -> Dict[str, Any]:
    page_index = int(selected_page["page_index"])
    page_content = load_page_content(
        canonical_record={"document": {"doc_id": selected_page["doc_id"]}},
        extract_path=extract_root,
        page_index=page_index,
    )
    layout_blocks = build_basic_layout_blocks(
        doc_id=selected_page["doc_id"],
        page_index=page_index,
        page_text=page_content["page_text"],
        has_page_image=page_content["has_page_image"],
    )
    return {
        "doc_id": selected_page["doc_id"],
        "page_index": page_index,
        "page_text": page_content["page_text"],
        "page_text_path": page_content["page_text_path"],
        "page_image_path": page_content["page_image_path"],
        "has_page_text": page_content["has_page_text"],
        "has_page_image": page_content["has_page_image"],
        "layout_blocks": layout_blocks,
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


def main() -> None:
    args = parse_args()
    result = run_small_batch(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
