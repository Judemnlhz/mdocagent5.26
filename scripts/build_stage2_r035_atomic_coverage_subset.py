#!/usr/bin/env python3
"""Build an R035 bounded subset from generic atomicizer page-text signals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.index_builder import build_mdocagent_extract_paths  # noqa: E402
from mdocnexus.stage2.page_input import build_basic_layout_blocks  # noqa: E402
from mdocnexus.stage2.table_numeric_atomicizer import atomicize_table_numeric_artifacts  # noqa: E402


FORBIDDEN_SELECTION_KEYS = {
    "answer",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
}


def main() -> None:
    args = parse_args()
    excluded_pages = load_excluded_pages(args.exclude_subset)
    records = read_records(args.records)
    candidates = score_candidates(
        records=records,
        extract_root=Path(args.extract_root),
        retrieval_topk=int(args.retrieval_topk),
        excluded_pages=excluded_pages,
    )
    selected = select_rows(
        candidates,
        max_pages=int(args.max_pages),
        max_pages_per_doc=int(args.max_pages_per_doc),
        selection_source=str(args.selection_source),
    )
    write_jsonl(Path(args.output), selected)
    report = {
        "schema_version": "r035_atomic_coverage_subset_v1",
        "records": str(args.records),
        "extract_root": str(args.extract_root),
        "output": str(args.output),
        "selection_source": str(args.selection_source),
        "max_pages": int(args.max_pages),
        "max_pages_per_doc": int(args.max_pages_per_doc),
        "retrieval_topk": int(args.retrieval_topk),
        "exclude_subset": str(args.exclude_subset) if args.exclude_subset else None,
        "num_excluded_pages": len(excluded_pages),
        "num_candidates": len(candidates),
        "num_selected_docs": len(selected),
        "num_selected_pages": sum(len(row["page_indices"]) for row in selected),
        "no_gold_fields_used": True,
        "selection_policy": "retrieval_topk_pages_ranked_by_generic_atomicizer_offline_count",
        "selected": selected,
        "top_candidates": candidates[: min(50, len(candidates))],
    }
    if args.report_json:
        write_json(Path(args.report_json), report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_selected_docs": report["num_selected_docs"],
                "num_selected_pages": report["num_selected_pages"],
                "num_candidates": len(candidates),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="data/MMLongBench/sample-with-retrieval-results.json")
    parser.add_argument("--extract-root", default="tmp/MMLongBench")
    parser.add_argument("--output", default="outputs/stage2_structured_incremental/r035_atomic_coverage_probe/subset_r035_atomic_coverage.jsonl")
    parser.add_argument("--report-json", default="outputs/stage2_structured_incremental/r035_atomic_coverage_probe/subset_r035_atomic_coverage_report.json")
    parser.add_argument("--exclude-subset", action="append", default=[])
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-pages-per-doc", type=int, default=2)
    parser.add_argument("--retrieval-topk", type=int, default=10)
    parser.add_argument("--selection-source", default="r035_generic_atomic_coverage_probe")
    return parser.parse_args()


def score_candidates(
    *,
    records: list[dict[str, Any]],
    extract_root: Path,
    retrieval_topk: int,
    excluded_pages: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        doc_id = str(record.get("doc_id") or "")
        if not doc_id:
            continue
        for page_index, retrieval_score in retrieval_pages(record, retrieval_topk):
            key = (doc_id, int(page_index))
            if key in excluded_pages:
                continue
            text = read_page_text(extract_root, doc_id, int(page_index))
            if not text:
                continue
            page_input = {
                "doc_id": doc_id,
                "page_index": int(page_index),
                "page_text": text,
                "layout_blocks": build_basic_layout_blocks(doc_id, int(page_index), text, has_page_image=True),
            }
            artifacts = atomicize_table_numeric_artifacts(
                selected_page={"doc_id": doc_id, "page_index": int(page_index)},
                page_input=page_input,
                existing_artifacts=[],
                max_cells=12,
            )
            type_counts = Counter(str(artifact.get("artifact_type") or "") for artifact in artifacts)
            atomic_pair_count = min(type_counts.get("table_cell", 0), type_counts.get("numeric_fact", 0))
            if atomic_pair_count < 2:
                continue
            score = float(atomic_pair_count) * 10.0 + float(retrieval_score)
            row = {
                "doc_id": doc_id,
                "page_index": int(page_index),
                "record_index": int(record.get("record_index", record_index) or record_index),
                "score": round(score, 6),
                "retrieval_score": round(float(retrieval_score), 6),
                "offline_table_cell_count": int(type_counts.get("table_cell", 0)),
                "offline_numeric_fact_count": int(type_counts.get("numeric_fact", 0)),
                "offline_atomic_pair_count": int(atomic_pair_count),
                "selection_reasons": ["generic_atomicizer_offline_table_numeric_signal"],
            }
            if key not in best or row["score"] > best[key]["score"]:
                best[key] = row
    return sorted(best.values(), key=lambda row: (-float(row["score"]), row["doc_id"], int(row["page_index"])))


def retrieval_pages(record: dict[str, Any], topk: int) -> list[tuple[int, float]]:
    pages: dict[int, float] = {}
    for key, value in record.items():
        key_text = str(key)
        lowered = key_text.lower()
        if is_forbidden_key(key_text):
            continue
        if "top" not in lowered or "score" in lowered or not isinstance(value, list):
            continue
        for rank, page in enumerate(value[: int(topk)]):
            try:
                page_index = int(page)
            except (TypeError, ValueError):
                continue
            pages[page_index] = max(pages.get(page_index, 0.0), 1.0 / float(rank + 1))
    return sorted(pages.items(), key=lambda item: (-item[1], item[0]))


def select_rows(candidates: list[dict[str, Any]], max_pages: int, max_pages_per_doc: int, selection_source: str) -> list[dict[str, Any]]:
    pages_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_count = 0
    for candidate in candidates:
        if selected_count >= int(max_pages):
            break
        doc_id = str(candidate["doc_id"])
        if len(pages_by_doc[doc_id]) >= int(max_pages_per_doc):
            continue
        pages_by_doc[doc_id].append(candidate)
        selected_count += 1

    rows: list[dict[str, Any]] = []
    for doc_id in sorted(pages_by_doc):
        pages = sorted(pages_by_doc[doc_id], key=lambda row: int(row["page_index"]))
        rows.append(
            {
                "doc_id": doc_id,
                "page_indices": [int(row["page_index"]) for row in pages],
                "selection_source": selection_source,
                "selection_reasons": sorted({reason for row in pages for reason in row["selection_reasons"]}),
                "offline_atomic_pair_count": sum(int(row["offline_atomic_pair_count"]) for row in pages),
                "offline_table_cell_count": sum(int(row["offline_table_cell_count"]) for row in pages),
                "offline_numeric_fact_count": sum(int(row["offline_numeric_fact_count"]) for row in pages),
            }
        )
    return rows


def load_excluded_pages(paths: list[str] | None) -> set[tuple[str, int]]:
    excluded: set[tuple[str, int]] = set()
    for path in paths or []:
        if path in (None, ""):
            continue
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"--exclude-subset does not exist: {file_path}")
        rows = read_records(file_path)
        for row in rows:
            if not isinstance(row, dict) or row.get("doc_id") in (None, ""):
                continue
            doc_id = str(row["doc_id"])
            page_indices = row.get("page_indices")
            if isinstance(page_indices, list):
                for value in page_indices:
                    try:
                        excluded.add((doc_id, int(value)))
                    except (TypeError, ValueError):
                        continue
                continue
            try:
                excluded.add((doc_id, int(row.get("page_index"))))
            except (TypeError, ValueError):
                continue
    return excluded


def read_page_text(extract_root: Path, doc_id: str, page_index: int) -> str:
    paths = build_mdocagent_extract_paths(extract_root, doc_id, page_index)
    for path in paths["text_candidate_paths"]:
        if Path(path).is_file():
            return Path(path).read_text(encoding="utf-8", errors="replace")
    return ""


def read_records(path: str | Path) -> list[Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected JSON list or JSONL records: {file_path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def is_forbidden_key(key: str) -> bool:
    return key in FORBIDDEN_SELECTION_KEYS or key.startswith("gold_")


if __name__ == "__main__":
    main()
