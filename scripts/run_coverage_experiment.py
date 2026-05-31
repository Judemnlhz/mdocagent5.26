#!/usr/bin/env python3
"""Run a no-API Stage 2/3/4 coverage experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_PUBLIC_QUERIES = "outputs/stage3_query/public_queries.jsonl"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    row = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def current_git_commit(cwd: Path) -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic no-API coverage experiment.")
    parser.add_argument("--scope-mode", choices=("doc_first", "query_doc_all", "retrieval_topk_scope"), required=True)
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--max-pages-per-doc", type=int, default=3)
    parser.add_argument("--retrieval-topk-file", default=DEFAULT_RECORDS)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--retrieval-method", choices=("deterministic_lexical", "deterministic_hybrid"), default="deterministic_hybrid")
    parser.add_argument("--expansion-mode", choices=("direct_structural", "page_neighborhood", "source_anchor_neighborhood"), default="page_neighborhood")
    parser.add_argument("--run-name", default="coverage_experiment")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--public-queries", default=DEFAULT_PUBLIC_QUERIES)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root or f"outputs/experiments/{args.run_name}")
    subset_dir = output_root / "subset"
    stage2_dir = output_root / "stage2_doc_coverage"
    stage3_dir = output_root / "stage3_doc_artifact_retrieval"
    eval3_dir = output_root / "eval" / "stage3_retrieval_eval"
    stage4_dir = output_root / "stage4" / "evidence_graph"
    eval4_dir = output_root / "eval" / "stage4_graph_expansion_eval"
    subset_path = subset_dir / f"stage2_coverage_subset_{args.scope_mode}.jsonl"
    subset_manifest = subset_dir / "manifest.json"
    max_pages = min(50, max(1, int(args.max_docs) * int(args.max_pages_per_doc)))

    commands: list[dict[str, Any]] = []
    build_subset_cmd = [
         "python3",
        "scripts/build_stage2_coverage_subset.py",
        "--scope-mode",
        args.scope_mode,
        "--public-query-input",
        args.public_queries,
        "--records",
        args.records,
        "--extract-root",
        args.extract_root,
        "--max-docs",
        str(args.max_docs),
        "--max-pages-per-doc",
        str(args.max_pages_per_doc),
        "--retrieval-topk-file",
        args.retrieval_topk_file,
        "--retrieval-topk",
        str(args.retrieval_topk),
        "--output",
        str(subset_path),
        "--manifest-output",
        str(subset_manifest),
    ]
    commands.append(run_command(build_subset_cmd, repo))

    stage2_cmd = [
         "python3",
        "scripts/stage2.py",
        "doc-compile",
        "--subset-file",
        str(subset_path),
        "--scope-mode",
        args.scope_mode,
        "--retrieval-topk-file",
        args.retrieval_topk_file,
        "--retrieval-topk",
        str(args.retrieval_topk),
        "--max-docs",
        str(args.max_docs),
        "--max-pages-per-doc",
        str(args.max_pages_per_doc),
        "--max-pages",
        str(max_pages),
        "--extract-root",
        args.extract_root,
        "--output-dir",
        str(stage2_dir),
        "--provider",
        "fake",
        "--image-payload-mode",
        "none",
    ]
    commands.append(run_command(stage2_cmd, repo))

    stage3_cmd = [
         "python3",
        "scripts/stage3_doc_artifact_retrieval.py",
        "--artifacts",
        str(stage2_dir / "artifacts.jsonl"),
        "--queries",
        args.public_queries,
        "--retrieval-method",
        args.retrieval_method,
        "--output-dir",
        str(stage3_dir),
    ]
    commands.append(run_command(stage3_cmd, repo))

    eval3_cmd = [
         "python3",
        "scripts/eval_stage3_retrieval.py",
        "--retrieval",
        str(stage3_dir / "retrieval.jsonl"),
        "--artifacts",
        str(stage2_dir / "artifacts.jsonl"),
        "--records",
        args.records,
        "--output-dir",
        str(eval3_dir),
    ]
    commands.append(run_command(eval3_cmd, repo))

    stage4_cmd = [
         "python3",
        "scripts/stage4_build_evidence_graph.py",
        "--artifacts",
        str(stage2_dir / "artifacts.jsonl"),
        "--retrieval",
        str(stage3_dir / "retrieval.jsonl"),
        "--output-dir",
        str(stage4_dir),
    ]
    commands.append(run_command(stage4_cmd, repo))

    eval4_cmd = [
         "python3",
        "scripts/eval_stage4_graph_expansion.py",
        "--retrieval",
        str(stage3_dir / "retrieval.jsonl"),
        "--graph",
        str(stage4_dir),
        "--artifacts",
        str(stage2_dir / "artifacts.jsonl"),
        "--records",
        args.records,
        "--expansion-mode",
        args.expansion_mode,
        "--output-dir",
        str(eval4_dir),
    ]
    commands.append(run_command(eval4_cmd, repo))

    audit_no_gold_cmd = [
         "python3",
        "scripts/audit_no_gold_leakage.py",
        "--scan-dir",
        str(stage2_dir),
        "--scan-dir",
        str(stage3_dir),
        "--scan-dir",
        str(stage4_dir),
    ]
    commands.append(run_command(audit_no_gold_cmd, repo))

    audit_repro_cmd = [
         "python3",
        "scripts/audit_reproducibility.py",
        "--stage2-dir",
        str(stage2_dir),
        "--stage3-dir",
        str(stage3_dir),
        "--stage4-dir",
        str(stage4_dir),
    ]
    commands.append(run_command(audit_repro_cmd, repo))

    stage2_report = read_json(stage2_dir / "quality_report.json")
    stage3_report = read_json(stage3_dir / "quality_report.json")
    eval3_report = read_json(eval3_dir / "report.json")
    stage4_report = read_json(stage4_dir / "quality_report.json")
    eval4_report = read_json(eval4_dir / "report.json")
    summary = {
        "run_name": args.run_name,
        "scope_mode": args.scope_mode,
        "retrieval_method": args.retrieval_method,
        "expansion_mode": args.expansion_mode,
        "stage2_artifact_coverage": {
            "num_artifacts": stage2_report.get("num_artifacts"),
            "num_selected_docs": stage2_report.get("num_selected_docs"),
            "num_selected_pages": stage2_report.get("num_selected_pages"),
            "artifact_type_counts": stage2_report.get("artifact_type_counts"),
            "proof_trace_eligible_rate": stage2_report.get("proof_trace_eligible_rate"),
            "image_payload_rate": stage2_report.get("image_payload_rate"),
        },
        "stage3_retrieval": {
            "artifact_coverage_rate": stage3_report.get("artifact_coverage_rate"),
            "num_queries_with_doc_artifacts": stage3_report.get("num_queries_with_doc_artifacts"),
            "num_queries_with_nonzero_scores": stage3_report.get("num_queries_with_nonzero_scores"),
            "zero_hit_query_count": eval3_report.get("zero_hit_query_count"),
            "recall_at_k_by_page": eval3_report.get("recall_at_k_by_page"),
            "coverage_at_k_by_page": eval3_report.get("coverage_at_k_by_page"),
        },
        "stage4_graph_expansion": {
            "num_edges": stage4_report.get("num_edges"),
            "flat_recall_at_k": eval4_report.get("flat_recall_at_k"),
            "expanded_recall_at_k": eval4_report.get("expanded_recall_at_k"),
            "delta_recall_at_k": eval4_report.get("delta_recall_at_k"),
            "flat_coverage_at_k": eval4_report.get("flat_coverage_at_k"),
            "expanded_coverage_at_k": eval4_report.get("expanded_coverage_at_k"),
            "delta_coverage_at_k": eval4_report.get("delta_coverage_at_k"),
            "expansion_factor": eval4_report.get("expansion_factor"),
            "avg_added_artifacts": eval4_report.get("avg_added_artifacts"),
            "used_debug_edges": eval4_report.get("used_debug_edges"),
            "used_semantic_edges": eval4_report.get("used_semantic_edges"),
        },
        "no_gold_fields_used": True,
        "used_debug_edges": False,
        "used_semantic_edges": False,
    }
    manifest = {
        "run_name": args.run_name,
        "git_commit": current_git_commit(repo),
        "command_args": vars(args),
        "input_hashes": {
            "public_queries": file_sha256(repo / args.public_queries),
            "records": file_sha256(repo / args.records),
            "retrieval_topk_file": file_sha256(repo / args.retrieval_topk_file),
        },
        "commands": commands,
        "no_real_api": True,
        "no_answer_generation": True,
        "no_gold_fields_used": True,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
