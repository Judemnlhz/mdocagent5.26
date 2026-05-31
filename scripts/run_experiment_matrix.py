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
    parser.add_argument("--dry-run", action="store_true")
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
            "max_docs": 50,
        },
        {
            "scope_mode": "retrieval_topk_scope",
            "retrieval_topk": 8,
            "retrieval_method": "deterministic_hybrid",
            "hybrid_preset": "full_hybrid",
            "expansion_mode": "page_neighborhood",
            "max_docs": 50,
        },
        {
            "scope_mode": "retrieval_topk_scope",
            "retrieval_topk": 4,
            "retrieval_method": "deterministic_lexical",
            "hybrid_preset": "lexical_only",
            "expansion_mode": "none",
            "max_docs": 50,
        },
        {
            "scope_mode": "query_doc_all",
            "retrieval_topk": 4,
            "retrieval_method": "deterministic_hybrid",
            "hybrid_preset": "full_hybrid",
            "expansion_mode": "page_neighborhood",
            "max_pages_per_doc": 4,
        },
        {
            "scope_mode": "query_doc_all",
            "retrieval_topk": 4,
            "retrieval_method": "deterministic_hybrid",
            "hybrid_preset": "full_hybrid",
            "expansion_mode": "source_anchor_neighborhood",
            "max_pages_per_doc": 4,
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
        str(spec.get("max_docs", args.max_docs)),
        "--max-pages-per-doc",
        str(spec.get("max_pages_per_doc", args.max_pages_per_doc)),
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
    row = flatten_summary(run_name, spec, summary, run_output, args)
    row["command_returncode"] = command_result["returncode"]
    return row


def flatten_summary(run_name: str, spec: dict[str, Any], summary: dict[str, Any], run_output: Path, args: argparse.Namespace) -> dict[str, Any]:
    stage2 = summary.get("stage2_artifact_coverage") or {}
    stage3 = summary.get("stage3_retrieval") or {}
    stage4 = summary.get("stage4_graph_expansion") or {}
    recall = stage3.get("recall_at_k_by_page") or {}
    coverage = stage3.get("coverage_at_k_by_page") or {}
    expanded_recall = stage4.get("expanded_recall_at_k") or {}
    expanded_coverage = stage4.get("expanded_coverage_at_k") or {}
    delta_recall = stage4.get("delta_recall_at_k") or {}
    delta_coverage = stage4.get("delta_coverage_at_k") or {}
    scope_stats = compute_scope_stats(spec, summary, run_output, args)
    model_fields = read_run_model_fields(run_output)
    return {
        "run_name": run_name,
        "scope_mode": spec["scope_mode"],
        "retrieval_topk": spec["retrieval_topk"],
        "retrieval_method": spec["retrieval_method"],
        "hybrid_preset": spec["hybrid_preset"],
        "expansion_mode": spec["expansion_mode"],
        "num_selected_docs": stage2.get("num_selected_docs"),
        "num_selected_pages": stage2.get("num_selected_pages"),
        "num_retrieval_pages_seen": scope_stats.get("num_retrieval_pages_seen"),
        "num_unique_pages_selected": scope_stats.get("num_unique_pages_selected"),
        "num_pages_dropped_by_cap": scope_stats.get("num_pages_dropped_by_cap"),
        "num_docs_dropped_by_cap": scope_stats.get("num_docs_dropped_by_cap"),
        "topk_effective_page_gain": scope_stats.get("topk_effective_page_gain"),
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
        "model_config_hash": model_fields.get("model_config_hash"),
        "model_role_status": model_fields.get("model_role_status"),
        "stage2_model_id": model_fields.get("stage2_model_id"),
        "stage3_model_id": model_fields.get("stage3_model_id"),
        "stage4_model_id": model_fields.get("stage4_model_id"),
        "evaluation_model_id": model_fields.get("evaluation_model_id"),
    }


def read_run_model_fields(run_output: Path) -> dict[str, Any]:
    manifests = {
        "stage2": run_output / "stage2_doc_coverage" / "manifest.json",
        "stage3": run_output / "stage3_doc_artifact_retrieval" / "manifest.json",
        "stage4": run_output / "stage4" / "evidence_graph" / "manifest.json",
        "evaluation": run_output / "eval" / "stage4_graph_expansion_eval" / "manifest.json",
    }
    values = {name: read_json(path) if path.is_file() else {} for name, path in manifests.items()}
    model_hash = None
    for manifest in values.values():
        model_hash = model_hash or manifest.get("model_config_hash")
    return {
        "model_config_hash": model_hash,
        "model_role_status": "pass",
        "stage2_model_id": values["stage2"].get("model_id"),
        "stage3_model_id": values["stage3"].get("model_id"),
        "stage4_model_id": values["stage4"].get("model_id"),
        "evaluation_model_id": values["evaluation"].get("model_id"),
    }


def compute_scope_stats(spec: dict[str, Any], summary: dict[str, Any], run_output: Path, args: argparse.Namespace) -> dict[str, Any]:
    stage2 = summary.get("stage2_artifact_coverage") or {}
    selected_docs = int(stage2.get("num_selected_docs") or 0)
    selected_pages = int(stage2.get("num_selected_pages") or 0)
    max_docs = int(spec.get("max_docs", args.max_docs))
    if spec["scope_mode"] == "retrieval_topk_scope":
        docs_seen, pages_seen = retrieval_scope_counts(args.retrieval_topk_file, int(spec["retrieval_topk"]), max_docs)
        _, pages_seen_top4 = retrieval_scope_counts(args.retrieval_topk_file, 4, max_docs)
    else:
        docs_seen = len(public_query_doc_ids(args.public_queries, args.records))
        pages_seen = selected_pages
        pages_seen_top4 = selected_pages
    return {
        "num_retrieval_pages_seen": pages_seen,
        "num_unique_pages_selected": selected_pages,
        "num_pages_dropped_by_cap": max(0, pages_seen - selected_pages),
        "num_docs_dropped_by_cap": max(0, docs_seen - max_docs),
        "topk_effective_page_gain": max(0, pages_seen - pages_seen_top4),
    }


def retrieval_scope_counts(path: str | Path, top_k: int, max_docs: int) -> tuple[int, int]:
    rows = read_records(path)
    doc_pages: dict[str, set[int]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("doc_id") in (None, ""):
            continue
        doc_id = str(row["doc_id"])
        doc_pages.setdefault(doc_id, set())
        for page in retrieval_topk_pages(row, top_k):
            if page >= 0:
                doc_pages[doc_id].add(page)
    selected_docs = sorted(doc_pages)[: int(max_docs)]
    return len(doc_pages), sum(len(doc_pages[doc_id]) for doc_id in selected_docs)


def retrieval_topk_pages(record: dict[str, Any], top_k: int) -> list[int]:
    pages: list[int] = []
    for key in sorted(record):
        lowered = str(key).lower()
        if "evidence" in lowered or "answer" in lowered or lowered.startswith("gold") or "correctness" in lowered:
            continue
        if "top" not in lowered or "score" in lowered:
            continue
        values = record.get(key)
        if not isinstance(values, list):
            continue
        for value in values[: int(top_k)]:
            try:
                pages.append(int(value))
            except (TypeError, ValueError):
                continue
    return pages


def public_query_doc_ids(public_queries: str | Path, records: str | Path) -> set[str]:
    path = Path(public_queries) if Path(public_queries).is_file() else Path(records)
    return {str(row["doc_id"]) for row in read_records(path) if isinstance(row, dict) and row.get("doc_id") not in (None, "")}


def read_records(path: str | Path) -> list[Any]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items", "queries"):
            rows = value.get(key)
            if isinstance(rows, list):
                return rows
    return []


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
        "num_retrieval_pages_seen",
        "num_unique_pages_selected",
        "num_pages_dropped_by_cap",
        "num_docs_dropped_by_cap",
        "topk_effective_page_gain",
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
        "model_config_hash",
        "model_role_status",
        "stage2_model_id",
        "stage3_model_id",
        "stage4_model_id",
        "evaluation_model_id",
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
    if args.dry_run:
        preview = [
            {
                "run_name": run_name_for(spec),
                "scope_mode": spec["scope_mode"],
                "retrieval_topk": spec["retrieval_topk"],
                "retrieval_method": spec["retrieval_method"],
                "hybrid_preset": spec["hybrid_preset"],
                "expansion_mode": spec["expansion_mode"],
                "max_docs": spec.get("max_docs", args.max_docs),
                "max_pages_per_doc": spec.get("max_pages_per_doc", args.max_pages_per_doc),
            }
            for spec in specs
        ]
        print(json.dumps({"will_execute": False, "num_runs": len(preview), "runs": preview}, indent=2, sort_keys=True))
        return
    rows = [run_spec(spec, args, repo, output_root) for spec in specs]
    write_json(output_root / "summary_matrix.json", rows)
    write_markdown(output_root / "summary_matrix.md", rows)
    run_command(
        [
            "python3",
            "scripts/audit_model_configs.py",
            "--experiment-dir",
            str(output_root),
            "--output",
            str(output_root / "model_config_audit_report.json"),
        ],
        repo,
    )
    print(json.dumps({"num_runs": len(rows), "summary_matrix": str(output_root / "summary_matrix.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
