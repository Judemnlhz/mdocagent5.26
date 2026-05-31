#!/usr/bin/env python3
"""Build a fixed document-only subset for Stage 2 coverage runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.stage2 import discover_document_page_indices


DEFAULT_PUBLIC_QUERY_INPUT = "outputs/stage3_query/public_queries.jsonl"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_OUTPUT = "outputs/subsets/stage2_coverage_subset.jsonl"


def build_coverage_subset(
    public_query_input: str | Path = DEFAULT_PUBLIC_QUERY_INPUT,
    records_path: str | Path = DEFAULT_RECORDS,
    extract_root: str | Path = DEFAULT_EXTRACT_ROOT,
    max_docs: int = 5,
    max_pages_per_doc: int = 2,
) -> list[dict[str, Any]]:
    source_path = Path(public_query_input) if Path(public_query_input).is_file() else Path(records_path)
    rows = read_records(source_path)
    doc_ids = sorted({str(row["doc_id"]) for row in rows if isinstance(row, dict) and row.get("doc_id") not in (None, "")})
    selected_doc_ids = doc_ids[: int(max_docs)]
    subset_rows: list[dict[str, Any]] = []
    for doc_id in selected_doc_ids:
        page_indices = discover_document_page_indices(doc_id, extract_root)[: int(max_pages_per_doc)]
        if not page_indices:
            page_indices = [0]
        subset_rows.append(
            {
                "doc_id": doc_id,
                "page_indices": [int(index) for index in page_indices],
                "selection_policy": "doc_id_deduplicated_stable_sort_first_n",
                "source_input": public_path(source_path),
            }
        )
    return subset_rows


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
    raise ValueError(f"Expected records in {input_path}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            assert_no_forbidden_selection_fields(row)
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assert_no_forbidden_selection_fields(value: Any) -> None:
    forbidden = {"answer", "gold_answer", "evidence_pages", "evidence_sources", "binary_correctness"}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in forbidden or str(key).startswith("gold_"):
                raise ValueError(f"Forbidden subset field: {key}")
            assert_no_forbidden_selection_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_selection_fields(child)


def public_path(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path_obj.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Stage 2 coverage subset by doc_id only.")
    parser.add_argument("--public-query-input", default=DEFAULT_PUBLIC_QUERY_INPUT)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--max-pages-per-doc", type=int, default=2)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = build_coverage_subset(
        public_query_input=args.public_query_input,
        records_path=args.records,
        extract_root=args.extract_root,
        max_docs=args.max_docs,
        max_pages_per_doc=args.max_pages_per_doc,
    )
    write_jsonl(args.output, rows)
    print(json.dumps({"output": args.output, "num_documents": len(rows), "selection_policy": "doc_id_only"}, sort_keys=True))


if __name__ == "__main__":
    main()
