#!/usr/bin/env python3
"""Evaluate formal-edge Stage 4 graph expansion without debug or semantic edges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

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
    parser.add_argument("--retrieval-jsonl", default="outputs/stage3_doc_artifact_retrieval/retrieval.jsonl")
    parser.add_argument("--artifacts-jsonl", default="outputs/stage2_doc/artifacts.jsonl")
    parser.add_argument("--records", default="data/MMLongBench/sample-with-retrieval-results.json")
    parser.add_argument("--edges-jsonl", default="outputs/stage4/evidence_graph/edges.jsonl")
    parser.add_argument("--output-dir", default="outputs/eval/stage4_graph_expansion_eval")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    report, per_query = evaluate_stage4_graph_expansion(
        retrieval_rows=read_jsonl(args.retrieval_jsonl),
        artifacts=read_jsonl(args.artifacts_jsonl),
        records=read_records(args.records),
        formal_edges=read_jsonl(args.edges_jsonl),
    )
    manifest = {
        "evaluation_only": True,
        "stage": "stage4_graph_expansion_eval",
        "retrieval_jsonl": args.retrieval_jsonl,
        "artifacts_jsonl": args.artifacts_jsonl,
        "records": args.records,
        "edges_jsonl": args.edges_jsonl,
        "debug_edges_read": False,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "report.json", report)
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
