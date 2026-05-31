#!/usr/bin/env python3
"""Run deterministic Stage 2/3/4 coverage experiment matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_PUBLIC_QUERIES = "outputs/stage3_query/public_queries.jsonl"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_OUTPUT_ROOT = "outputs/experiments/matrix"


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


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic coverage experiment matrix.")
    parser.add_argument("--scope-mode", default=None, help="Comma-separated: retrieval_topk_scope,query_doc_all")
    parser.add_argument("--retrieval-topk", default=None, help="Comma-separated integers, for example 4,8")
    parser.add_argument("--retrieval-method", default=None, help="Comma-separated: deterministic_lexical,deterministic_hybrid")
    parser.add_argument("--hybrid-preset", default=None, help="Comma-separated: lexical_only,full_hybrid,hybrid_no_graph")
    parser.add_argument("--expansion-mode", default=None, help="Comma-separated: none,page_neighborhood,source_anchor_neighborhood")
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--max-pages-per-doc", type=int, default=3)
    parser.add_argument("--retrieval-topk-file", default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--public-queries", default=DEFAULT_PUBLIC_QUERIES)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    return parser


def split_values(value: str | None, default: list[str]) -> list[str]:
    if value in (None, ""):
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def split_ints(value: str | None, default: list[int]) -> list[int]:
    return [int(item) for item in split_values(value, [str(number) for number in default])]


def default_specs() -> list[dict[str, Any]]:
    return [
        {
            "scope_mode": "retrieval_topk_scope",
            "retrieval_topk": 4,
            "retrieval_method": "deterministic_hybrid",
            "hybrid_preset": "full_hybrid",
            "expansion_mode": "page_neighborhood",
        },
        {
            "scope_mode": "retrieval_topk_scope",
            "retrieval_topk": 8,
            "retrieval_method": "deterministic_hybrid",
            "hybrid_preset": "full_hybrid",
            "expansion_mode": "page_neighborhood",
        },
        {
            "scope_mode": "retrieval_topk_scope",
            "retrieval_topk": 4,
            "retrieval_method": "deterministic_lexical",
            "hybrid_preset": "lexical_only",
            "expansion_mode": "none",
        },
    ]


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if all(
        getattr(args, name) in (None, "")
        for name in ("scope_mode", "retrieval_topk", "retrieval_method", "hybrid_preset", "expansion_mode")
    ):
        return default_specs()
    scope_modes = split_values(args.scope_mode, ["retrieval_topk_scope", "query_doc_all"])
    retrieval_topks = split_ints(args.retrieval_topk, [4, 8])
    retrieval_methods = split_values(args.retrieval_method, ["deterministic_lexical", "deterministic_hybrid"])
    hybrid_presets = split_values(args.hybrid_preset, ["lexical_only", "full_hybrid", "hybrid_no_graph"])
    expansion_modes = split_values(args.expansion_mode, ["none", "page_neighborhood", "source_anchor_neighborhood"])
    specs: list[dict[str, Any]] = []
    for scope_mode in scope_modes:
        for topk in retrieval_topks:
            for method in retrieval_methods:
                presets = ["lexical_only"] if method == "deterministic_lexical" else hybrid_presets
                for preset in presets:
                    for expansion_mode in expansion_modes:
                        specs.append(
                            {
                                "scope_mode": scope_mode,
                                "retrieval_topk": int(topk),
                                "retrieval_method": method,
                                "hybrid_preset": preset,
                                "expansion_mode": expansion_mode,
                            }
                        )
    return dedupe_specs(specs)


def dedupe_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for spec in specs:
        key = json.dumps(spec, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def run_name_for(spec: dict[str, Any]) -> str:
    method = "hybrid" if spec["retrieval_method"] == "deterministic_hybrid" else "lexical"
    expansion = str(spec["expansion_mode"]).replace("source_anchor_neighborhood", "source_anchor")
    return f"{spec['scope_mode']}_topk{spec['retrieval_topk']}_{method}_{spec['hybrid_preset']}_{expansion}"


def run_spec(spec: dict[str, Any], args: argparse.Namespace, repo: Path, output_root: Path) -> dict[str, Any]:
    run_name = run_name_for(spec)
    run_output = output_root / run_name
    command = [
        "python3",
        "scripts/run_coverage_experiment.py",
        "--scope-mode",
        spec["scope_mode"],
        "--max-docs",
        str(args.max_docs),
        "--max-pages-per-doc",
        str(args.max_pages_per_doc),
        "--retrieval-topk-file",
        args.retrieval_topk_file,
        "--retrieval-topk",
        str(spec["retrieval_topk"]),
        "--retrieval-method",
        spec["retrieval_method"],
        "--hybrid-preset",
        spec["hybrid_preset"],
        "--expansion-mode",
        spec["expansion_mode"],
        "--run-name",
        run_name,
        "--output-root",
        str(run_output),
        "--public-queries",
        args.public_queries,
        "--records",
        args.records,
        "--extract-root",
        args.extract_root,
    ]
    command_result = run_command(command, repo)
    summary = read_json(run_output / "summary.json")
    row = flatten_summary(run_name, spec, summary)
    row["command_returncode"] = command_result["returncode"]
    return row


def flatten_summary(run_name: str, spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    stage2 = summary.get("stage2_artifact_coverage") or {}
    stage3 = summary.get("stage3_retrieval") or {}
    stage4 = summary.get("stage4_graph_expansion") or {}
    recall = stage3.get("recall_at_k_by_page") or {}
    coverage = stage3.get("coverage_at_k_by_page") or {}
    expanded_recall = stage4.get("expanded_recall_at_k") or {}
    expanded_coverage = stage4.get("expanded_coverage_at_k") or {}
    delta_recall = stage4.get("delta_recall_at_k") or {}
    delta_coverage = stage4.get("delta_coverage_at_k") or {}
    return {
        "run_name": run_name,
        "scope_mode": spec["scope_mode"],
        "retrieval_topk": spec["retrieval_topk"],
        "retrieval_method": spec["retrieval_method"],
        "hybrid_preset": spec["hybrid_preset"],
        "expansion_mode": spec["expansion_mode"],
        "num_selected_docs": stage2.get("num_selected_docs"),
        "num_selected_pages": stage2.get("num_selected_pages"),
        "num_artifacts": stage2.get("num_artifacts"),
        "artifact_coverage_rate": stage3.get("artifact_coverage_rate"),
        "zero_hit_query_count": stage3.get("zero_hit_query_count"),
        "recall@5": recall.get("5"),
        "coverage@5": coverage.get("5"),
        "expanded_recall@5": expanded_recall.get("5"),
        "expanded_coverage@5": expanded_coverage.get("5"),
        "delta_recall@5": delta_recall.get("5"),
        "delta_coverage@5": delta_coverage.get("5"),
        "avg_added_artifacts": stage4.get("avg_added_artifacts"),
        "expansion_ratio": stage4.get("expansion_ratio") or stage4.get("expansion_factor"),
        "used_debug_edges": bool(summary.get("used_debug_edges", False)),
        "used_semantic_edges": bool(summary.get("used_semantic_edges", False)),
        "no_gold_fields_used": bool(summary.get("no_gold_fields_used", True)),
    }


def write_markdown(path: str | Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "scope_mode",
        "retrieval_topk",
        "retrieval_method",
        "hybrid_preset",
        "expansion_mode",
        "num_selected_docs",
        "num_selected_pages",
        "num_artifacts",
        "artifact_coverage_rate",
        "zero_hit_query_count",
        "recall@5",
        "coverage@5",
        "expanded_recall@5",
        "expanded_coverage@5",
        "delta_recall@5",
        "delta_coverage@5",
        "avg_added_artifacts",
        "expansion_ratio",
        "used_debug_edges",
        "used_semantic_edges",
        "no_gold_fields_used",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    rows = [run_spec(spec, args, repo, output_root) for spec in specs]
    write_json(output_root / "summary_matrix.json", rows)
    write_markdown(output_root / "summary_matrix.md", rows)
    print(json.dumps({"num_runs": len(rows), "summary_matrix": str(output_root / "summary_matrix.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
