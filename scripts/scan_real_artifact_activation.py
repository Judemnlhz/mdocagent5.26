#!/usr/bin/env python3
"""Scan whether real strong eligible artifacts activate retrieval records.

This is a deterministic pre-QA gate. It does not run MDocAgent prediction,
evaluation, rerank tuning, graph expansion, or full ablation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdocnexus.integration.mdocagent_adapter import (  # noqa: E402
    DEFAULT_LAMBDA_WEIGHT,
    artifact_rerank_eligibility_reason,
    load_artifacts_by_page,
    load_mdocagent_retrieval_records,
    rerank_pages_with_artifacts,
    retrieval_page_keys,
)

DEFAULT_INPUT_RETRIEVAL = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_ARTIFACTS = "outputs/stage2_structured_real_gate/artifacts.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/experiments/mdocagent_module_ablation/run_tags/real_structured_activation_scan"
PRIOR_POLICY_TOP30 = "outputs/experiments/mdocagent_module_ablation/record_ids/activation_rich_matrix_top30.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-retrieval", default=DEFAULT_INPUT_RETRIEVAL)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--lambda-weight", type=float, default=DEFAULT_LAMBDA_WEIGHT)
    parser.add_argument("--min-heldout", type=int, default=30)
    parser.add_argument("--max-heldout", type=int, default=50)
    parser.add_argument("--exclude-record-ids-file", default=PRIOR_POLICY_TOP30)
    parser.add_argument("--max-records-per-doc", type=int, default=5)
    parser.add_argument("--max-records-per-page", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_mdocagent_retrieval_records(args.input_retrieval)
    artifacts_by_page = load_artifacts_by_page(args.artifacts)
    strong_pages = strong_artifact_pages(artifacts_by_page)
    prior_excluded = read_record_ids(args.exclude_record_ids_file)

    original = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=args.top_k, mode="original_only", lambda_weight=args.lambda_weight)
    artifact_only = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=args.top_k, mode="artifact_only", lambda_weight=args.lambda_weight)
    original_plus = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=args.top_k, mode="original_plus_artifact", lambda_weight=args.lambda_weight)

    rows: list[dict[str, Any]] = []
    activated_ids: list[int] = []
    changed_ids: list[int] = []
    changed_plus_ids: list[int] = []
    strong_doc_counter: Counter[str] = Counter()
    strong_page_counter: Counter[str] = Counter()

    for index, record in enumerate(records):
        record_id = int(record.get("record_index", index))
        doc_id = str(record.get("doc_id") or "")
        candidate_pages = candidate_pages_for_record(record, int(args.top_k))
        matching_pages = sorted(page for page in candidate_pages if page in strong_pages.get(doc_id, set()))
        branch_original = branch_top_pages(original[index])
        branch_artifact = branch_top_pages(artifact_only[index])
        branch_plus = branch_top_pages(original_plus[index])
        artifact_changed = branch_changed(branch_original, branch_artifact)
        plus_changed = branch_changed(branch_original, branch_plus)
        activated = bool(matching_pages)
        if activated:
            activated_ids.append(record_id)
            strong_doc_counter[doc_id] += 1
            for page in matching_pages:
                strong_page_counter[f"{doc_id}#p{page:03d}"] += 1
        if artifact_changed:
            changed_ids.append(record_id)
        if plus_changed:
            changed_plus_ids.append(record_id)
        if activated or artifact_changed or plus_changed:
            rows.append(
                {
                    "record_id": record_id,
                    "record_index": index,
                    "doc_id": doc_id,
                    "question_preview": str(record.get("question") or "")[:240],
                    "strong_pages_in_candidates": matching_pages,
                    "activated": activated,
                    "artifact_only_changed": artifact_changed,
                    "original_plus_changed": plus_changed,
                    "original_pages": branch_original,
                    "artifact_only_pages": branch_artifact,
                    "original_plus_pages": branch_plus,
                    "excluded_by_prior_policy_top30": record_id in prior_excluded,
                }
            )

    eligible_for_heldout = [row for row in rows if row["activated"] and not row["excluded_by_prior_policy_top30"]]
    heldout_rows = capped_heldout_rows(
        eligible_for_heldout,
        max_records=int(args.max_heldout),
        max_records_per_doc=int(args.max_records_per_doc),
        max_records_per_page=int(args.max_records_per_page),
    )
    heldout_ids = [int(row["record_id"]) for row in heldout_rows]
    heldout_available = len(heldout_ids) >= int(args.min_heldout)

    report = {
        "schema_version": "real_structured_artifact_activation_scan_v1",
        "input_retrieval": args.input_retrieval,
        "artifacts": args.artifacts,
        "top_k": int(args.top_k),
        "lambda_weight": float(args.lambda_weight),
        "total_records": len(records),
        "strong_eligible_artifact_pages": {doc: sorted(pages) for doc, pages in sorted(strong_pages.items())},
        "strong_eligible_page_count": sum(len(pages) for pages in strong_pages.values()),
        "activated_count": len(activated_ids),
        "changed_count": len(changed_ids),
        "original_plus_changed_count": len(changed_plus_ids),
        "activated_ids": activated_ids,
        "changed_ids": changed_ids,
        "original_plus_changed_ids": changed_plus_ids,
        "excluded_prior_policy_top30_count": sum(1 for row in rows if row.get("excluded_by_prior_policy_top30")),
        "eligible_for_heldout_count": len(eligible_for_heldout),
        "heldout_activation_rich_subset": {
            "available": heldout_available,
            "min_required": int(args.min_heldout),
            "max_requested": int(args.max_heldout),
            "num_records": len(heldout_ids),
            "record_ids": heldout_ids,
            "record_ids_path": str(output_dir / "heldout_activation_rich_record_ids.txt") if heldout_available else None,
            "reason_unavailable": None if heldout_available else "doc/page capped activated records are below min-heldout",
            "sampling_policy": {
                "method": "doc_page_capped_sampling",
                "max_records_per_doc": int(args.max_records_per_doc),
                "max_records_per_page": int(args.max_records_per_page),
                "concentration_metrics_used_for_sampling": False,
            },
        },
        "concentration": {
            "activated_by_doc": dict(strong_doc_counter.most_common()),
            "activated_by_page": dict(strong_page_counter.most_common()),
            "max_doc_share": max(strong_doc_counter.values(), default=0) / max(1, len(activated_ids)),
            "max_page_share": max(strong_page_counter.values(), default=0) / max(1, len(activated_ids)),
            "effective_num_docs": effective_number(strong_doc_counter),
            "effective_num_pages": effective_number(strong_page_counter),
            "used_for_reranking": False,
            "uses_gold_fields": False,
            "purpose": "external_validity_diagnostic_only",
        },
        "decision": "build_heldout_subset" if heldout_available else "do_not_run_effectiveness_gate_expand_stage2_coverage",
        "no_qa_run": True,
        "no_rerank_tuning": True,
        "no_full_ablation": True,
        "uses_gold_fields": False,
        "concentration_metrics_used_for_reranking": False,
    }

    write_json(output_dir / "real_structured_activation_scan_report.json", report)
    write_jsonl(output_dir / "real_structured_activation_scan_rows.jsonl", rows)
    write_markdown(output_dir / "real_structured_activation_scan_report.md", report)
    if heldout_available:
        write_ids(output_dir / "heldout_activation_rich_record_ids.txt", heldout_ids)
    print(json.dumps({"decision": report["decision"], "activated_count": len(activated_ids), "eligible_for_heldout_count": len(eligible_for_heldout)}, indent=2))



def capped_heldout_rows(
    rows: list[dict[str, Any]],
    max_records: int,
    max_records_per_doc: int,
    max_records_per_page: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    doc_counts: Counter[str] = Counter()
    page_counts: Counter[str] = Counter()
    for row in rows:
        doc_id = str(row.get("doc_id") or "")
        pages = [int(page) for page in row.get("strong_pages_in_candidates", [])]
        primary_page = pages[0] if pages else -1
        page_key = f"{doc_id}#p{primary_page:03d}"
        if doc_counts[doc_id] >= max_records_per_doc:
            continue
        if page_counts[page_key] >= max_records_per_page:
            continue
        selected.append(row)
        doc_counts[doc_id] += 1
        page_counts[page_key] += 1
        if len(selected) >= max_records:
            break
    return selected


def effective_number(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    concentration = sum((count / total) ** 2 for count in counter.values())
    if concentration <= 0:
        return 0.0
    return round(1.0 / concentration, 6)

def strong_artifact_pages(artifacts_by_page: Mapping[str, Mapping[int, list[dict[str, Any]]]]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for doc_id, pages in artifacts_by_page.items():
        for page_index, artifacts in pages.items():
            if any(artifact_rerank_eligibility_reason(artifact) == "eligible" for artifact in artifacts):
                result[str(doc_id)].add(int(page_index))
    return dict(result)


def candidate_pages_for_record(record: Mapping[str, Any], top_k: int) -> set[int]:
    pages: set[int] = set()
    for key in retrieval_page_keys(record):
        values = record.get(key)
        if not isinstance(values, list):
            continue
        for value in values[:top_k]:
            try:
                pages.add(int(value))
            except (TypeError, ValueError):
                continue
    return pages


def branch_top_pages(record: Mapping[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for key in retrieval_page_keys(record):
        values = record.get(key)
        if not isinstance(values, list):
            continue
        result[key] = [int(value) for value in values]
    return result


def branch_changed(left: Mapping[str, list[int]], right: Mapping[str, list[int]]) -> bool:
    keys = sorted(set(left) | set(right))
    for key in keys:
        if list(left.get(key, [])) != list(right.get(key, [])):
            return True
    return False


def read_record_ids(path: str | Path | None) -> set[int]:
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.is_file():
        return set()
    ids: set[int] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            ids.add(int(stripped))
        except ValueError:
            continue
    return ids


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_ids(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{record_id}\n" for record_id in ids), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    heldout = report["heldout_activation_rich_subset"]
    concentration = report["concentration"]
    lines = [
        "# Real Structured Artifact Activation Scan",
        "",
        f"Decision: `{report['decision']}`",
        "",
        f"Total records: {report['total_records']}",
        f"Strong eligible pages: {report['strong_eligible_page_count']}",
        f"Activated records: {report['activated_count']}",
        f"Changed records (artifact_only): {report['changed_count']}",
        f"Changed records (original_plus_artifact): {report['original_plus_changed_count']}",
        f"Eligible for held-out after excluding prior policy top30: {report['eligible_for_heldout_count']}",
        "",
        "## Held-out Subset",
        f"Available: `{heldout['available']}`",
        f"Records: {heldout['num_records']}",
        f"Reason unavailable: {heldout['reason_unavailable']}",
        "",
        "## Concentration",
        f"Max doc share: {concentration['max_doc_share']:.4f}",
        f"Max page share: {concentration['max_page_share']:.4f}",
        f"Effective docs: {concentration['effective_num_docs']}",
        f"Effective pages: {concentration['effective_num_pages']}",
        "Concentration metrics are external-validity diagnostics only; they are not used for reranking or scoring.",
        f"Activated by doc: `{concentration['activated_by_doc']}`",
        f"Activated by page: `{concentration['activated_by_page']}`",
        "",
        "## Scope",
        "- No QA run.",
        "- No rerank tuning.",
        "- No full ablation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
