#!/usr/bin/env python3
"""Build public Stage 3 query inputs from raw records without gold fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


GOLD_FIELDS = {
    "answer",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}


def build_public_query_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        row: dict[str, Any] = {
            "record_index": int(record.get("record_index", record_index)),
            "doc_id": record.get("doc_id"),
            "question": record.get("question") or record.get("query") or "",
        }
        if record.get("query_id") not in (None, ""):
            row["query_id"] = str(record["query_id"])
        elif record.get("record_id") not in (None, ""):
            row["query_id"] = str(record["record_id"])
        if record.get("dataset") not in (None, ""):
            row["dataset"] = record.get("dataset")
        assert_no_gold_fields(row)
        rows.append(row)
    return rows


def read_records(path: str | Path) -> list[Any]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items", "queries"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"Expected records in {input_path}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assert_no_gold_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in GOLD_FIELDS or key_text.startswith("gold_"):
                raise ValueError(f"Forbidden gold field in public query output: {key_text}")
            assert_no_gold_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_gold_fields(child)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create public Stage 3 query input JSONL without gold fields.")
    parser.add_argument("--input", default="data/MMLongBench/sample-with-retrieval-results.json")
    parser.add_argument("--output", default="outputs/stage3_query/public_queries.jsonl")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = build_public_query_rows(read_records(args.input))
    write_jsonl(args.output, rows)
    print(json.dumps({"output": args.output, "num_queries": len(rows), "no_gold_fields_used": True}, indent=2))


if __name__ == "__main__":
    main()
