#!/usr/bin/env python3
"""Build a tiny public Stage 2 subset biased toward structured artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

FORBIDDEN_KEYS = {"answer", "gold_answer", "evidence_pages", "evidence_sources", "binary_correctness"}
DOC_HINTS = ("pew", "survey", "report", "paper", "brochure", "catalog")
TABLE_HINTS = ("table", "survey", "respondents", "percentage", "percent", "%", "figure", "chart", "graph")
NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?\s*(?:%|percent|percentage|million|billion|thousand|\$)?", re.I)


def main() -> None:
    args = parse_args()
    records = read_records(args.records)
    excluded_pages = load_excluded_pages(args.exclude_subset)
    candidates = score_candidate_pages(records, Path(args.extract_root), int(args.retrieval_topk), excluded_pages, str(args.selection_source))
    selected = select_rows(candidates, int(args.max_pages), int(args.max_pages_per_doc), str(args.selection_source))
    write_jsonl(selected, Path(args.output))
    report = {
        "schema_version": "stage2_structured_subset_v1",
        "records": str(args.records),
        "extract_root": str(args.extract_root),
        "output": str(args.output),
        "max_pages": int(args.max_pages),
        "max_pages_per_doc": int(args.max_pages_per_doc),
        "exclude_subset": str(args.exclude_subset) if args.exclude_subset else None,
        "num_excluded_pages": len(excluded_pages),
        "selection_source": str(args.selection_source),
        "num_candidates": len(candidates),
        "num_selected_pages": sum(len(row["page_indices"]) for row in selected),
        "num_selected_docs": len(selected),
        "no_gold_fields_used": True,
        "selected": selected,
    }
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_selected_docs": len(selected), "num_selected_pages": report["num_selected_pages"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="data/MMLongBench/sample-with-retrieval-results.json")
    parser.add_argument("--extract-root", default="tmp/MMLongBench")
    parser.add_argument("--output", default="outputs/subsets/stage2_structured_real_subset.jsonl")
    parser.add_argument("--report-json", default="outputs/subsets/stage2_structured_real_subset_report.json")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-pages-per-doc", type=int, default=3)
    parser.add_argument("--retrieval-topk", type=int, default=10)
    parser.add_argument("--exclude-subset", default=None, help="Optional JSONL/JSON subset whose doc/page pairs should be skipped.")
    parser.add_argument("--selection-source", default="structured_real_stage2_small_sample")
    return parser.parse_args()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("records must be a JSON list or JSONL")
    return [row for row in data if isinstance(row, dict)]


def score_candidate_pages(
    records: list[dict[str, Any]],
    extract_root: Path,
    retrieval_topk: int,
    excluded_pages: set[tuple[str, int]] | None = None,
    selection_source: str = "structured_real_stage2_small_sample",
) -> list[dict[str, Any]]:
    excluded_pages = excluded_pages or set()
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        if any(key in record for key in FORBIDDEN_KEYS):
            public_record = {key: value for key, value in record.items() if key not in FORBIDDEN_KEYS and not str(key).startswith("gold_")}
        else:
            public_record = record
        doc_id = str(public_record.get("doc_id") or "")
        if not doc_id:
            continue
        for page_index, retrieval_score in retrieval_pages(public_record, retrieval_topk):
            if (doc_id, int(page_index)) in excluded_pages:
                continue
            text_path = page_text_path(extract_root, doc_id, page_index)
            image_path = page_image_path(extract_root, doc_id, page_index)
            if not text_path and not image_path:
                continue
            page_text = text_path.read_text(encoding="utf-8", errors="replace") if text_path else ""
            score, reasons = structured_score(doc_id, page_index, page_text, bool(image_path), retrieval_score)
            key = (doc_id, int(page_index))
            row = {
                "doc_id": doc_id,
                "page_index": int(page_index),
                "score": round(score, 6),
                "selection_source": selection_source,
                "selection_reasons": reasons,
                "has_page_text": bool(text_path),
                "has_page_image": bool(image_path),
                "record_index": int(record_index),
            }
            if key not in best or row["score"] > best[key]["score"]:
                best[key] = row
    return sorted(best.values(), key=lambda row: (-float(row["score"]), row["doc_id"], int(row["page_index"])))


def retrieval_pages(record: dict[str, Any], topk: int) -> list[tuple[int, float]]:
    pages: dict[int, float] = {}
    for key, value in record.items():
        lowered = str(key).lower()
        if key in FORBIDDEN_KEYS or str(key).startswith("gold_"):
            continue
        if "top" not in lowered or "score" in lowered or not isinstance(value, list):
            continue
        for rank, page in enumerate(value[:topk]):
            try:
                page_index = int(page)
            except (TypeError, ValueError):
                continue
            pages[page_index] = max(pages.get(page_index, 0.0), 1.0 / float(rank + 1))
    return sorted(pages.items(), key=lambda item: (-item[1], item[0]))


def structured_score(doc_id: str, page_index: int, text: str, has_image: bool, retrieval_score: float) -> tuple[float, list[str]]:
    lower = text.lower()
    reasons: list[str] = []
    number_count = len(NUMERIC_RE.findall(text))
    percent_count = lower.count("%") + lower.count("percent")
    table_hint_count = sum(lower.count(hint) for hint in TABLE_HINTS)
    doc_hint_count = sum(doc_id.lower().count(hint) for hint in DOC_HINTS)
    if number_count >= 8:
        reasons.append("numeric_dense_text")
    if percent_count:
        reasons.append("percentage_text")
    if table_hint_count:
        reasons.append("table_or_chart_terms")
    if doc_hint_count:
        reasons.append("document_type_hint")
    if has_image:
        reasons.append("image_available")
    score = float(retrieval_score)
    score += min(number_count, 40) * 0.08
    score += min(percent_count, 20) * 0.25
    score += min(table_hint_count, 20) * 0.18
    score += min(doc_hint_count, 5) * 0.3
    if has_image:
        score += 0.5
    if page_index <= 1:
        score -= 0.25
    return score, reasons or ["retrieval_candidate"]


def page_text_path(root: Path, doc_id: str, page_index: int) -> Path | None:
    stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    for candidate in [root / f"{stem}_{page_index}.txt", root / f"{stem}_{page_index:03d}.txt"]:
        if candidate.is_file():
            return candidate
    return None


def page_image_path(root: Path, doc_id: str, page_index: int) -> Path | None:
    stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    for candidate in [root / f"{stem}_{page_index}.png", root / f"{stem}_{page_index:03d}.png"]:
        if candidate.is_file():
            return candidate
    return None


def select_rows(candidates: list[dict[str, Any]], max_pages: int, max_pages_per_doc: int, selection_source: str = "structured_real_stage2_small_sample") -> list[dict[str, Any]]:
    pages_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_count = 0
    for candidate in candidates:
        if selected_count >= max_pages:
            break
        doc_id = str(candidate["doc_id"])
        if len(pages_by_doc[doc_id]) >= max_pages_per_doc:
            continue
        pages_by_doc[doc_id].append(candidate)
        selected_count += 1
    rows = []
    for doc_id in sorted(pages_by_doc):
        pages = sorted(pages_by_doc[doc_id], key=lambda row: int(row["page_index"]))
        rows.append(
            {
                "doc_id": doc_id,
                "page_indices": [int(row["page_index"]) for row in pages],
                "selection_source": selection_source,
                "selection_reasons": sorted({reason for row in pages for reason in row["selection_reasons"]}),
            }
        )
    return rows



def load_excluded_pages(path: str | Path | None) -> set[tuple[str, int]]:
    if path in (None, ""):
        return set()
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"--exclude-subset does not exist: {file_path}")
    rows = read_records(file_path)
    excluded: set[tuple[str, int]] = set()
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
        elif row.get("page_index") not in (None, ""):
            try:
                excluded.add((doc_id, int(row["page_index"])))
            except (TypeError, ValueError):
                continue
    return excluded

def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
