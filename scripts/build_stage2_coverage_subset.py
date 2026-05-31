#!/usr/bin/env python3
"""Build fixed document/page scopes for Stage 2 coverage runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.stage2 import discover_document_page_indices


DEFAULT_PUBLIC_QUERY_INPUT = "outputs/stage3_query/public_queries.jsonl"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_OUTPUT_TEMPLATE = "outputs/subsets/stage2_coverage_subset_{mode}.jsonl"
SCOPE_MODES = {"doc_first", "query_doc_all", "retrieval_topk_scope"}
FORBIDDEN_SELECTION_FIELDS = {
    "answer",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
}


def build_coverage_subset(
    public_query_input: str | Path = DEFAULT_PUBLIC_QUERY_INPUT,
    records_path: str | Path = DEFAULT_RECORDS,
    extract_root: str | Path = DEFAULT_EXTRACT_ROOT,
    max_docs: int | None = 5,
    max_pages_per_doc: int = 2,
    scope_mode: str = "doc_first",
    retrieval_topk_file: str | Path | None = None,
    retrieval_topk: int = 5,
) -> list[dict[str, Any]]:
    """Return document/page scope rows without copying question or gold fields."""

    if scope_mode not in SCOPE_MODES:
        raise ValueError(f"Unsupported scope_mode: {scope_mode}")
    if int(max_pages_per_doc) < 1:
        raise ValueError("max_pages_per_doc must be at least 1")

    if scope_mode == "retrieval_topk_scope":
        source_path = Path(retrieval_topk_file or records_path)
        rows = read_records(source_path)
        subset_rows = build_retrieval_topk_scope(
            rows=rows,
            extract_root=extract_root,
            max_docs=max_docs,
            max_pages_per_doc=max_pages_per_doc,
            top_k=retrieval_topk,
            source_path=source_path,
        )
    else:
        source_path = Path(public_query_input) if Path(public_query_input).is_file() else Path(records_path)
        rows = read_records(source_path)
        subset_rows = build_public_query_doc_scope(
            rows=rows,
            extract_root=extract_root,
            max_docs=max_docs,
            max_pages_per_doc=max_pages_per_doc,
            scope_mode=scope_mode,
            source_path=source_path,
        )
    for row in subset_rows:
        assert_no_forbidden_selection_fields(row)
    return subset_rows


def build_public_query_doc_scope(
    rows: list[Any],
    extract_root: str | Path,
    max_docs: int | None,
    max_pages_per_doc: int,
    scope_mode: str,
    source_path: Path,
) -> list[dict[str, Any]]:
    doc_ids = sorted({str(row["doc_id"]) for row in rows if isinstance(row, dict) and row.get("doc_id") not in (None, "")})
    if scope_mode == "doc_first":
        limit = 5 if max_docs is None else int(max_docs)
        selected_doc_ids = doc_ids[:limit]
        selection_policy = "doc_id_deduplicated_stable_sort_first_n"
        selection_source = "doc_first_public_query_doc_id"
    elif scope_mode == "query_doc_all":
        selected_doc_ids = doc_ids if max_docs is None else doc_ids[: int(max_docs)]
        selection_policy = "public_query_doc_id_deduplicated_stable_sort_all"
        selection_source = "query_doc_all_public_query_doc_id"
    else:
        raise ValueError(f"Unsupported public query scope mode: {scope_mode}")

    subset_rows: list[dict[str, Any]] = []
    for doc_id in selected_doc_ids:
        page_indices = discover_document_page_indices(doc_id, extract_root)[: int(max_pages_per_doc)]
        if not page_indices:
            page_indices = [0]
        subset_rows.append(
            {
                "doc_id": doc_id,
                "page_indices": [int(index) for index in page_indices],
                "scope_mode": scope_mode,
                "selection_policy": selection_policy,
                "selection_source": selection_source,
                "source_input": public_path(source_path),
            }
        )
    return subset_rows


def build_retrieval_topk_scope(
    rows: list[Any],
    extract_root: str | Path,
    max_docs: int | None,
    max_pages_per_doc: int,
    top_k: int,
    source_path: Path,
) -> list[dict[str, Any]]:
    page_scores_by_doc: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    page_seen_by_doc: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if not isinstance(row, dict) or row.get("doc_id") in (None, ""):
            continue
        doc_id = str(row["doc_id"])
        for page_index, score in retrieval_topk_pages(row, top_k=top_k):
            if page_index < 0:
                continue
            page_scores_by_doc[doc_id][page_index] += score
            page_seen_by_doc[doc_id].add(page_index)

    doc_ids = sorted(page_scores_by_doc)
    if max_docs is not None:
        doc_ids = doc_ids[: int(max_docs)]

    subset_rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        ranked_pages = sorted(page_scores_by_doc[doc_id], key=lambda page: (-page_scores_by_doc[doc_id][page], page))
        selected_pages = ranked_pages[: int(max_pages_per_doc)]
        if not selected_pages:
            continue
        subset_rows.append(
            {
                "doc_id": doc_id,
                "page_indices": [int(index) for index in selected_pages],
                "scope_mode": "retrieval_topk_scope",
                "selection_policy": "retrieval_topk_page_ids_rank_aggregate",
                "selection_source": "retrieval_topk_non_gold",
                "compile_scope_source": "mdocagent_retrieval_topk",
                "retrieval_topk": int(top_k),
                "source_input": public_path(source_path),
            }
        )
    return subset_rows


def retrieval_topk_pages(record: dict[str, Any], top_k: int) -> list[tuple[int, float]]:
    pages: list[tuple[int, float]] = []
    for key in sorted(record):
        key_text = str(key)
        lowered = key_text.lower()
        if is_forbidden_selection_key(key_text):
            continue
        if "top" not in lowered or "score" in lowered:
            continue
        values = record.get(key)
        if not isinstance(values, list):
            continue
        for rank, value in enumerate(values[: int(top_k)]):
            try:
                page_index = int(value)
            except (TypeError, ValueError):
                continue
            pages.append((page_index, 1.0 / float(rank + 1)))
    return pages


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


def build_manifest(
    *,
    subset_rows: list[dict[str, Any]],
    output: str | Path,
    input_path: str | Path,
    scope_mode: str,
    max_docs: int | None,
    max_pages_per_doc: int,
    top_k: int,
) -> dict[str, Any]:
    manifest = {
        "scope_mode": scope_mode,
        "max_docs": None if max_docs is None else int(max_docs),
        "max_pages_per_doc": int(max_pages_per_doc),
        "top_k": int(top_k),
        "input_hash": file_sha256(input_path),
        "output": public_path(output),
        "num_documents": len(subset_rows),
        "num_pages": sum(len(row.get("page_indices", [])) for row in subset_rows),
        "no_gold_fields_used": True,
        "selection_stable": True,
        "created_by_script": "scripts/build_stage2_coverage_subset.py",
        "git_commit": current_git_commit(),
    }
    if scope_mode == "retrieval_topk_scope":
        manifest["compile_scope_source"] = "mdocagent_retrieval_topk"
    assert_no_forbidden_selection_fields(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def assert_no_forbidden_selection_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if is_forbidden_selection_key(str(key)):
                raise ValueError(f"Forbidden subset field: {key}")
            assert_no_forbidden_selection_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_selection_fields(child)


def is_forbidden_selection_key(key: str) -> bool:
    key_text = str(key)
    return key_text in FORBIDDEN_SELECTION_FIELDS or key_text.startswith("gold_")


def default_output_for_mode(scope_mode: str) -> str:
    return DEFAULT_OUTPUT_TEMPLATE.format(mode=scope_mode)


def default_manifest_for_output(output: str | Path) -> str:
    output_path = Path(output)
    return str(output_path.with_name(f"{output_path.stem}_manifest.json"))


def public_path(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path_obj.name


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Stage 2 coverage subset by deterministic document/page scope.")
    parser.add_argument("--scope-mode", choices=sorted(SCOPE_MODES), default="doc_first")
    parser.add_argument("--public-query-input", default=DEFAULT_PUBLIC_QUERY_INPUT)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--retrieval-topk-file", default=None)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-pages-per-doc", type=int, default=2)
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = args.output or default_output_for_mode(args.scope_mode)
    input_path = Path(args.retrieval_topk_file or args.records) if args.scope_mode == "retrieval_topk_scope" else (Path(args.public_query_input) if Path(args.public_query_input).is_file() else Path(args.records))
    rows = build_coverage_subset(
        public_query_input=args.public_query_input,
        records_path=args.records,
        extract_root=args.extract_root,
        max_docs=args.max_docs,
        max_pages_per_doc=args.max_pages_per_doc,
        scope_mode=args.scope_mode,
        retrieval_topk_file=args.retrieval_topk_file,
        retrieval_topk=args.retrieval_topk,
    )
    write_jsonl(output, rows)
    manifest = build_manifest(
        subset_rows=rows,
        output=output,
        input_path=input_path,
        scope_mode=args.scope_mode,
        max_docs=args.max_docs,
        max_pages_per_doc=args.max_pages_per_doc,
        top_k=args.retrieval_topk,
    )
    manifest_output = args.manifest_output or default_manifest_for_output(output)
    write_manifest(manifest_output, manifest)
    print(
        json.dumps(
            {
                "output": output,
                "manifest": manifest_output,
                "num_documents": len(rows),
                "num_pages": manifest["num_pages"],
                "scope_mode": args.scope_mode,
                "no_gold_fields_used": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
