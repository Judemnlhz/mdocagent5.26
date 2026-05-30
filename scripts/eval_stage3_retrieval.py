#!/usr/bin/env python3
"""Evaluate Stage 3 retrieval only; gold is used only under outputs/eval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.evaluation.retrieval_metrics import (
    evaluate_stage3_retrieval,
    read_jsonl,
    read_records,
    write_json,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 artifact retrieval against gold pages.")
    parser.add_argument("--retrieval-jsonl", default="outputs/stage3_doc_artifact_retrieval/retrieval.jsonl")
    parser.add_argument("--artifacts-jsonl", default="outputs/stage2_doc/artifacts.jsonl")
    parser.add_argument("--records", default="data/MMLongBench/sample-with-retrieval-results.json")
    parser.add_argument("--output-dir", default="outputs/eval/stage3_retrieval_eval")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    report, per_query = evaluate_stage3_retrieval(
        retrieval_rows=read_jsonl(args.retrieval_jsonl),
        artifacts=read_jsonl(args.artifacts_jsonl),
        records=read_records(args.records),
    )
    manifest = {
        "evaluation_only": True,
        "stage": "stage3_retrieval_eval",
        "retrieval_jsonl": args.retrieval_jsonl,
        "artifacts_jsonl": args.artifacts_jsonl,
        "records": args.records,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "report.json", report)
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
