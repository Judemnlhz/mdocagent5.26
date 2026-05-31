#!/usr/bin/env python3
"""Evaluate formal-edge Stage 4 graph expansion without debug or semantic edges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.evaluation.retrieval_metrics import (
    evaluate_stage4_graph_expansion,
    read_jsonl,
    read_records,
    write_json,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate graph-expanded retrieval using formal Stage 4 edges only.")
    parser.add_argument("--retrieval", "--retrieval-jsonl", dest="retrieval_jsonl", default="outputs/stage3_doc_artifact_retrieval/retrieval.jsonl")
    parser.add_argument("--artifacts", "--artifacts-jsonl", dest="artifacts_jsonl", default="outputs/stage2_doc/artifacts.jsonl")
    parser.add_argument("--records", default="data/MMLongBench/sample-with-retrieval-results.json")
    parser.add_argument("--graph", default=None, help="Stage 4 graph directory or edges.jsonl path.")
    parser.add_argument("--edges-jsonl", default="outputs/stage4/evidence_graph/edges.jsonl")
    parser.add_argument("--output-dir", default="outputs/eval/stage4_graph_expansion_eval")
    parser.add_argument("--expansion-mode", choices=("flat", "direct_structural", "page_neighborhood", "source_anchor_neighborhood"), default="direct_structural")
    parser.add_argument("--edge-ablation", action="store_true")
    parser.add_argument("--allowed-edge-types", default=None, help="Comma-separated formal edge type allowlist for expansion.")
    parser.add_argument("--blocked-edge-types", default=None, help="Comma-separated formal edge types to remove from expansion.")
    return parser


def resolve_edges_jsonl(graph: str | None, edges_jsonl: str) -> Path:
    if graph in (None, ""):
        return Path(edges_jsonl)
    graph_path = Path(graph)
    if graph_path.is_dir():
        return graph_path / "edges.jsonl"
    return graph_path


def parse_edge_types(value: str | None) -> set[str] | None:
    if value in (None, ""):
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def default_ablation_settings() -> list[dict[str, Any]]:
    return [
        {"setting": "flat", "expansion_mode": "flat"},
        {"setting": "page_neighborhood", "expansion_mode": "page_neighborhood"},
        {"setting": "source_anchor_neighborhood", "expansion_mode": "source_anchor_neighborhood"},
        {
            "setting": "page_neighborhood_without_adjacent_page",
            "expansion_mode": "page_neighborhood",
            "blocked_edge_types": {"adjacent_page"},
        },
        {
            "setting": "page_neighborhood_without_table_edges",
            "expansion_mode": "page_neighborhood",
            "blocked_edge_types": {"table_contains_cell", "row_contains_cell", "column_contains_cell"},
        },
        {
            "setting": "page_neighborhood_without_caption_edges",
            "expansion_mode": "page_neighborhood",
            "blocked_edge_types": {"caption_of", "figure_has_caption"},
        },
        {
            "setting": "source_anchor_only",
            "expansion_mode": "source_anchor_neighborhood",
            "allowed_edge_types": {"supported_by_anchor"},
        },
    ]


def run_single_eval(args: argparse.Namespace, edges_jsonl: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return evaluate_stage4_graph_expansion(
        retrieval_rows=read_jsonl(args.retrieval_jsonl),
        artifacts=read_jsonl(args.artifacts_jsonl),
        records=read_records(args.records),
        formal_edges=read_jsonl(edges_jsonl),
        expansion_mode=args.expansion_mode,
        allowed_edge_types=parse_edge_types(args.allowed_edge_types),
        blocked_edge_types=parse_edge_types(args.blocked_edge_types),
    )


def run_edge_ablation(args: argparse.Namespace, edges_jsonl: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retrieval_rows = read_jsonl(args.retrieval_jsonl)
    artifacts = read_jsonl(args.artifacts_jsonl)
    records = read_records(args.records)
    formal_edges = read_jsonl(edges_jsonl)
    cli_allowed = parse_edge_types(args.allowed_edge_types)
    cli_blocked = parse_edge_types(args.blocked_edge_types) or set()
    per_setting: list[dict[str, Any]] = []
    for setting in default_ablation_settings():
        allowed = set(setting["allowed_edge_types"]) if setting.get("allowed_edge_types") else (set(cli_allowed) if cli_allowed else None)
        blocked = set(setting.get("blocked_edge_types") or set()) | set(cli_blocked)
        report, _ = evaluate_stage4_graph_expansion(
            retrieval_rows=retrieval_rows,
            artifacts=artifacts,
            records=records,
            formal_edges=formal_edges,
            expansion_mode=str(setting["expansion_mode"]),
            allowed_edge_types=allowed,
            blocked_edge_types=blocked,
        )
        row = {
            "setting": setting["setting"],
            "expansion_mode": setting["expansion_mode"],
            "allowed_edge_types": sorted(allowed) if allowed is not None else None,
            "blocked_edge_types": sorted(blocked),
            "flat_recall_at_k": report.get("flat_recall_at_k"),
            "expanded_recall_at_k": report.get("expanded_recall_at_k"),
            "delta_recall_at_k": report.get("delta_recall_at_k"),
            "flat_coverage_at_k": report.get("flat_coverage_at_k"),
            "expanded_coverage_at_k": report.get("expanded_coverage_at_k"),
            "delta_coverage_at_k": report.get("delta_coverage_at_k"),
            "avg_added_artifacts": report.get("avg_added_artifacts"),
            "expansion_ratio": report.get("expansion_ratio"),
            "added_ratio": report.get("added_ratio"),
            "added_gold_page_hit_rate": report.get("added_gold_page_hit_rate"),
            "edge_types_used": report.get("edge_types_used"),
            "used_debug_edges": False,
            "used_semantic_edges": False,
            "evaluation_only": True,
        }
        per_setting.append(row)
    best_by_delta_recall_5 = max(per_setting, key=lambda row: float((row.get("delta_recall_at_k") or {}).get("5", 0.0)), default={})
    report = {
        "edge_ablation": True,
        "num_settings": len(per_setting),
        "settings": [row["setting"] for row in per_setting],
        "best_by_delta_recall_at_5": best_by_delta_recall_5.get("setting"),
        "used_debug_edges": False,
        "used_semantic_edges": False,
        "evaluation_only": True,
    }
    return report, per_setting


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.edge_ablation and args.output_dir == "outputs/eval/stage4_graph_expansion_eval":
        args.output_dir = "outputs/eval/stage4_graph_ablation_eval"
    output_dir = Path(args.output_dir)
    edges_jsonl = resolve_edges_jsonl(args.graph, args.edges_jsonl)
    if args.edge_ablation:
        report, per_rows = run_edge_ablation(args, edges_jsonl)
        per_rows_filename = "per_setting.jsonl"
    else:
        report, per_rows = run_single_eval(args, edges_jsonl)
        per_rows_filename = "per_query.jsonl"
    manifest = {
        "evaluation_only": True,
        "not_consumed_by_stage2_stage3_stage4": True,
        "stage": "stage4_graph_expansion_eval",
        "retrieval_jsonl": args.retrieval_jsonl,
        "artifacts_jsonl": args.artifacts_jsonl,
        "records": args.records,
        "edges_jsonl": str(edges_jsonl),
        "debug_edges_read": False,
        "used_debug_edges": False,
        "used_semantic_edges": False,
        "expansion_mode": args.expansion_mode,
        "edge_ablation": bool(args.edge_ablation),
        "allowed_edge_types": sorted(parse_edge_types(args.allowed_edge_types) or []),
        "blocked_edge_types": sorted(parse_edge_types(args.blocked_edge_types) or []),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "report.json", report)
    write_jsonl(output_dir / per_rows_filename, per_rows)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
