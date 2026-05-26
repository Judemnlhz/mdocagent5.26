"""Replay refined Stage 2 raw outputs with deterministic artifact deduplication."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.artifact_deduplicator import deduplicate_page_artifacts
from mdocnexus.stage2.artifact_store import build_document_artifact_store, write_artifact_store
from mdocnexus.stage2.artifact_validator import validate_page_artifact_output
from mdocnexus.stage2.crossdoc_quality_audit import audit_crossdoc_batch_with_options, write_audit_json, write_page_quality_csv
from mdocnexus.stage2.discard_log import DiscardLogEntry, issue_to_discard_log_entry, write_discard_log_entry
from mdocnexus.stage2.mdocagent_compat import read_json_or_jsonl_records
from mdocnexus.stage2.refined_error_attribution import summarize_refined_validation_failures
from mdocnexus.stage2.refinement_comparison import compare_crossdoc_audits, write_refinement_comparison
from mdocnexus.stage2.stage2_sidecar_store import load_stage2_preflight_sidecar


BASELINE_AUDIT = "outputs/stage2/artifacts_real_crossdoc_batch/reports/crossdoc_quality_audit.json"
REFINED_AUDIT = "outputs/stage2/artifacts_real_crossdoc_batch_refined/reports/crossdoc_quality_audit.json"
REFINED_DISCARD = "outputs/stage2/artifacts_real_crossdoc_batch_refined/discard/discard.jsonl"
STAGE_NAME = "stage2_refined_replay_dedup_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay refined raw outputs with deterministic deduplication.")
    parser.add_argument("--raw-output-log", required=True)
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = replay_refined_with_dedup(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def replay_refined_with_dedup(args: argparse.Namespace) -> dict:
    raw_entries = _read_jsonl(args.raw_output_log)
    records = read_json_or_jsonl_records(args.stage2_json)
    stage2_by_doc = _load_stage2_by_doc(records, args.stage2_json, args.sidecar_dir)
    refined_page_context = _load_refined_page_context(Path(args.raw_output_log).parents[1])
    output_paths = _initialize_output_paths(args.output_dir)

    page_results = []
    for raw_entry in raw_entries:
        page_results.append(_replay_one_page(raw_entry, stage2_by_doc, refined_page_context, output_paths))

    summary = _build_replay_summary(page_results, args)
    _write_json(summary, output_paths["crossdoc_batch_summary"])
    _write_quality_csv(page_results, output_paths["crossdoc_batch_quality"])

    attribution = summarize_refined_validation_failures(REFINED_DISCARD, args.raw_output_log)
    _write_json({**summary, "error_attribution": attribution}, output_paths["replay_dedup_summary"])
    replay_audit = audit_crossdoc_batch_with_options(
        batch_dir=args.output_dir,
        stage2_json=args.stage2_json,
        sidecar_dir=args.sidecar_dir,
    )
    write_audit_json(replay_audit, output_paths["replay_quality_audit"])
    write_page_quality_csv(replay_audit, output_paths["crossdoc_quality_by_page"])

    baseline_vs_replay = compare_crossdoc_audits(BASELINE_AUDIT, output_paths["replay_quality_audit"])
    refined_vs_replay = compare_crossdoc_audits(REFINED_AUDIT, output_paths["replay_quality_audit"])
    comparison = {
        "baseline_vs_replay": baseline_vs_replay,
        "refined_vs_replay": refined_vs_replay,
        "acceptance": baseline_vs_replay["acceptance"],
        "error_attribution": attribution,
    }
    write_refinement_comparison(comparison, output_paths["replay_vs_refined_comparison"])
    return {
        "summary": {
            **summary,
            "replay_quality_audit_path": str(output_paths["replay_quality_audit"]),
            "replay_vs_refined_comparison_path": str(output_paths["replay_vs_refined_comparison"]),
            "acceptance": comparison["acceptance"],
            "error_attribution": attribution,
        },
        "page_results": page_results,
        "audit": replay_audit,
        "comparison": comparison,
    }


def _replay_one_page(
    raw_entry: dict,
    stage2_by_doc: dict[str, dict],
    refined_page_context: dict[tuple[str, int], dict],
    output_paths: dict[str, Path],
) -> dict:
    doc_id = str(raw_entry["doc_id"])
    page_index = int(raw_entry["page_index"])
    raw_output = raw_entry.get("raw_output") if isinstance(raw_entry.get("raw_output"), dict) else {}
    deduped_output, dedup_removed = deduplicate_page_artifacts(raw_output)
    stage2 = stage2_by_doc[doc_id]
    page_context = refined_page_context.get((doc_id, page_index), {})
    layout_blocks = page_context.get("layout_blocks") or _layout_blocks_for_page(stage2, page_index)
    valid_artifacts, validation_issues = validate_page_artifact_output(deduped_output, layout_blocks)
    artifact_store_path = output_paths["artifact_stores"] / _artifact_store_file_name(doc_id, page_index)

    for removed in dedup_removed:
        write_discard_log_entry(
            output_paths["discard"],
            DiscardLogEntry(
                doc_id=doc_id,
                page_index=page_index,
                artifact_id=removed.get("artifact_id"),
                error_type="duplicate_artifact_deduplicated",
                message="Duplicate artifact removed before deterministic replay validation.",
                field_path="artifacts",
                details={
                    "duplicate_of": removed.get("duplicate_of"),
                    "dedup_key": removed.get("dedup_key"),
                },
                stage=STAGE_NAME,
                compiler_version=str(raw_entry.get("compiler_version", "stage2_compiler_v1")),
            ),
        )
    for issue in validation_issues:
        write_discard_log_entry(
            output_paths["discard"],
            issue_to_discard_log_entry(issue, stage=STAGE_NAME, compiler_version=str(raw_entry.get("compiler_version", "stage2_compiler_v1"))),
        )

    page_source = page_context.get("page_source") or _page_source_for_page(stage2, page_index)
    page_input = {
        "doc_id": doc_id,
        "page_index": page_index,
        "page_text_path": page_source.get("page_text_path"),
        "page_image_path": page_source.get("page_image_path"),
        "layout_blocks": layout_blocks,
    }
    canonical_record = {
        "document": {"doc_id": doc_id, "dataset": None},
        "candidate_pool": {
            "explicit_constraint_pages": stage2.get("explicit_page_validation", {}).get("valid_explicit_page_indices", []),
            "retrieval_missed_explicit_pages": [],
        },
    }
    write_artifact_store(
        build_document_artifact_store(
            canonical_record=canonical_record,
            prepared_pages=[page_input],
            page_artifact_outputs={page_index: deduped_output},
            validation_results={page_index: {"valid_artifacts": valid_artifacts, "validation_issues": [issue.to_dict() for issue in validation_issues]}},
            compiler_metadata={
                "compiler_name": "stage2_replay_dedup_validator",
                "compiler_version": "stage2_replay_dedup_v1",
                "schema_version": "stage2_artifact_schema_v1",
                "model_name": raw_entry.get("model_name"),
                "temperature": None,
                "max_repair_attempts": 0,
            },
        ),
        artifact_store_path,
    )

    return {
        "record_index": None,
        "doc_id": doc_id,
        "page_index": page_index,
        "selection_reason": "replay_refined_raw_output",
        "page_image_path": page_source.get("page_image_path"),
        "num_raw_artifacts_before_dedup": _count_raw_artifacts(raw_output),
        "num_deduplicated_artifacts": len(dedup_removed),
        "num_raw_artifacts": _count_raw_artifacts(deduped_output),
        "num_valid_artifacts": len(valid_artifacts),
        "num_validation_issues": len(validation_issues),
        "artifact_store_path": str(artifact_store_path),
        "raw_output_logged": True,
        "discard_logged": True,
        "passed": bool(valid_artifacts) and not validation_issues,
        "provider_error_type": None,
        "api_called": False,
    }


def _load_stage2_by_doc(records: list[dict], stage2_json: str | Path, sidecar_dir: str | Path) -> dict[str, dict]:
    result = {}
    for record in records:
        doc_id = record.get("doc_id")
        stage2 = record.get("stage2")
        if doc_id is None or not isinstance(stage2, dict):
            continue
        preflight_ref = stage2.get("preflight_ref")
        if not preflight_ref:
            result[str(doc_id)] = dict(stage2)
            continue
        sidecar_path = _resolve_sidecar_path(preflight_ref, stage2_json, sidecar_dir)
        sidecar = load_stage2_preflight_sidecar(sidecar_path)
        result[str(doc_id)] = {
            "version": stage2.get("version", "stage2_preflight_v1"),
            "doc_name": stage2.get("doc_name"),
            "page_count": {"value": stage2.get("page_count"), "source": stage2.get("page_count_source")},
            "question_constraints": sidecar.get("question_constraints", {}),
            "retrieval_pages": sidecar.get("retrieval_pages", {}),
            "explicit_page_validation": sidecar.get("explicit_page_validation", {}),
            "pages_to_compile": stage2.get("pages_to_compile", []),
            "page_sources": sidecar.get("page_sources", []),
            "layout_blocks_by_page": sidecar.get("layout_blocks_by_page", {}),
            "preflight": sidecar.get("preflight", {}),
        }
    return result


def _load_refined_page_context(refined_batch_dir: Path) -> dict[tuple[str, int], dict]:
    context = {}
    artifact_store_dir = refined_batch_dir / "artifact_stores"
    if not artifact_store_dir.is_dir():
        return context
    for store_path in artifact_store_dir.glob("*.json"):
        store = json.loads(store_path.read_text(encoding="utf-8"))
        doc_id = store.get("document", {}).get("doc_id")
        if not doc_id:
            continue
        for page in store.get("pages", []) or []:
            if not isinstance(page, dict) or page.get("page_index") is None:
                continue
            page_index = int(page["page_index"])
            context[(str(doc_id), page_index)] = {
                "layout_blocks": page.get("layout_blocks", []),
                "page_source": page.get("page_source", {}),
            }
    return context


def _resolve_sidecar_path(preflight_ref: str, stage2_json: str | Path, sidecar_dir: str | Path) -> Path:
    ref = Path(str(preflight_ref))
    candidates = [ref] if ref.is_absolute() else [ref, Path(stage2_json).parent / ref]
    candidates.extend([Path(sidecar_dir) / ref.name, Path(sidecar_dir) / ref])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to resolve sidecar: {preflight_ref}")


def _layout_blocks_for_page(stage2: dict, page_index: int) -> list[dict]:
    layout_by_page = stage2.get("layout_blocks_by_page", {})
    blocks = layout_by_page.get(str(page_index)) if isinstance(layout_by_page, dict) else None
    if isinstance(blocks, list) and blocks:
        return blocks
    source = _page_source_for_page(stage2, page_index)
    result = []
    for block_id in source.get("layout_block_ids", []) or []:
        block_id = str(block_id)
        result.append(
            {
                "block_id": block_id,
                "block_type": "full_page_image" if block_id.endswith("_full_page_image") else "text_block",
                "page_index": int(page_index),
                "bbox": None,
                "text": None,
            }
        )
    return result


def _page_source_for_page(stage2: dict, page_index: int) -> dict:
    for source in stage2.get("page_sources", []) or []:
        if not isinstance(source, dict):
            continue
        try:
            if int(source.get("page_index")) == int(page_index):
                return source
        except (TypeError, ValueError):
            continue
    return {}


def _build_replay_summary(page_results: list[dict], args: argparse.Namespace) -> dict:
    num_raw_artifacts = sum(int(result["num_raw_artifacts"]) for result in page_results)
    num_valid_artifacts = sum(int(result["num_valid_artifacts"]) for result in page_results)
    num_validation_issues = sum(int(result["num_validation_issues"]) for result in page_results)
    denominator = max(1, num_raw_artifacts)
    return {
        "stage": STAGE_NAME,
        "raw_output_log": str(args.raw_output_log),
        "stage2_json": str(args.stage2_json),
        "uses_compact_stage2": True,
        "uses_sidecar_preflight": True,
        "num_documents_attempted": len({result["doc_id"] for result in page_results}),
        "num_pages_attempted": len(page_results),
        "num_api_calls": 0,
        "num_raw_artifacts_before_dedup": sum(int(result["num_raw_artifacts_before_dedup"]) for result in page_results),
        "num_deduplicated_artifacts": sum(int(result["num_deduplicated_artifacts"]) for result in page_results),
        "num_raw_artifacts": num_raw_artifacts,
        "num_valid_artifacts": num_valid_artifacts,
        "num_validation_issues": num_validation_issues,
        "schema_valid_rate": num_valid_artifacts / denominator,
        "anchoring_rate": num_valid_artifacts / denominator,
        "discard_rate": max(0, num_raw_artifacts - num_valid_artifacts) / denominator,
        "num_artifacts_by_type": _count_artifacts_by_field(page_results, "artifact_type"),
        "num_artifacts_by_modality": _count_artifacts_by_field(page_results, "modality"),
        "artifact_store_paths": [result["artifact_store_path"] for result in page_results],
        "forbidden_field_violations": 0,
        "api_key_leaks": 0,
        "uses_answer": False,
        "uses_evidence_pages": False,
        "uses_binary_correctness": False,
        "real_api_called": False,
    }


def _count_artifacts_by_field(page_results: list[dict], field_name: str) -> dict:
    counts: Counter[str] = Counter()
    for result in page_results:
        path = Path(result["artifact_store_path"])
        if not path.is_file():
            continue
        store = json.loads(path.read_text(encoding="utf-8"))
        for page in store.get("pages", []):
            for artifact in page.get("artifacts", []):
                value = artifact.get(field_name)
                if value:
                    counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _write_quality_csv(page_results: list[dict], output_path: Path) -> None:
    fields = [
        "record_index",
        "doc_id",
        "page_index",
        "selection_reason",
        "page_image_path",
        "num_raw_artifacts",
        "num_valid_artifacts",
        "num_validation_issues",
        "artifact_store_path",
        "raw_output_logged",
        "discard_logged",
        "passed",
        "provider_error_type",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for result in page_results:
            writer.writerow({field: result.get(field) for field in fields})


def _initialize_output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "artifact_stores": root / "artifact_stores",
        "discard": root / "discard" / "discard.jsonl",
        "reports": root / "reports",
        "crossdoc_batch_summary": root / "reports" / "crossdoc_batch_summary.json",
        "crossdoc_batch_quality": root / "reports" / "crossdoc_batch_quality.csv",
        "crossdoc_quality_by_page": root / "reports" / "crossdoc_quality_by_page.csv",
        "replay_quality_audit": root / "reports" / "replay_quality_audit.json",
        "replay_dedup_summary": root / "reports" / "replay_dedup_summary.json",
        "replay_vs_refined_comparison": root / "reports" / "replay_vs_refined_comparison.json",
    }
    paths["artifact_stores"].mkdir(parents=True, exist_ok=True)
    paths["discard"].parent.mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    for file_path in [
        paths["discard"],
        paths["crossdoc_batch_summary"],
        paths["crossdoc_batch_quality"],
        paths["crossdoc_quality_by_page"],
        paths["replay_quality_audit"],
        paths["replay_dedup_summary"],
        paths["replay_vs_refined_comparison"],
    ]:
        if file_path.exists():
            file_path.unlink()
    for old_store in paths["artifact_stores"].glob("*.json"):
        old_store.unlink()
    return paths


def _read_jsonl(path: str | Path) -> list[dict]:
    entries = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                entries.append(loaded)
    return entries


def _write_json(value: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_store_file_name(doc_id: str, page_index: int) -> str:
    doc_name = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    safe_doc = "".join(char if char.isalnum() or char in "._-" else "_" for char in doc_name)
    return f"{safe_doc}_p{int(page_index):03d}.json"


def _count_raw_artifacts(raw_output: dict) -> int:
    if isinstance(raw_output, dict) and isinstance(raw_output.get("artifacts"), list):
        return len(raw_output["artifacts"])
    return 0


if __name__ == "__main__":
    main()
