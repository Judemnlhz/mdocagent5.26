#!/usr/bin/env python3
"""R038a audit-only noise check for the repaired R028 20 -> 30 pages.

This script is deliberately offline: it reads the existing R028 delta subset
and page OCR/layout text, runs the generic table/numeric atomicizer, and writes
summary reports only. It does not call a provider, write artifacts.jsonl, merge
stores, run activation, run QA, touch graph code, or tune reranking.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdocnexus.integration.mdocagent_adapter import artifact_rerank_eligibility_reason  # noqa: E402
from mdocnexus.stage2.artifact_quality import classify_artifact_quality, is_atomic_strong_eligible  # noqa: E402
from mdocnexus.stage2.index_builder import build_mdocagent_extract_paths  # noqa: E402
from mdocnexus.stage2.page_input import build_basic_layout_blocks  # noqa: E402
from mdocnexus.stage2.table_numeric_atomicizer import MAX_ATOMIC_CELLS_PER_PAGE, atomicize_table_numeric_artifacts  # noqa: E402


DEFAULT_SUBSET = "outputs/stage2_structured_incremental/r028_20_to_30/subset_delta_20_to_30.jsonl"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_OUTPUT_ROOT = "outputs/stage2_structured_incremental/r038a_repaired_20_to_30_noise_audit"
DEFAULT_R037_REPORT = "outputs/stage2_structured_incremental/r037_budgeted_targeted_coverage/r037_budgeted_targeted_coverage_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--r037-report", default=DEFAULT_R037_REPORT)
    parser.add_argument("--atomicizer-max-cells", type=int, default=MAX_ATOMIC_CELLS_PER_PAGE)
    parser.add_argument("--max-artifacts-per-page", type=int, default=16)
    parser.add_argument("--max-artifact-growth-vs-r037", type=float, default=1.25)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    report_json = output_root / "r038a_noise_audit_report.json"
    report_md = output_root / "r038a_noise_audit_report.md"
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "subset": args.subset,
                    "report_json": str(report_json),
                    "report_md": str(report_md),
                    "scope": "offline_noise_audit_only_no_provider_no_merge_no_qa",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    rows = flatten_subset(read_jsonl(Path(args.subset)))
    page_reports = [
        audit_page(row, Path(args.extract_root), int(args.atomicizer_max_cells))
        for row in rows
    ]
    r037 = read_json(Path(args.r037_report))
    report = build_report(args, page_reports, r037)
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "pages_audited": report["summary"]["pages_audited"],
                "total_artifacts": report["summary"]["total_artifacts"],
                "artifacts_per_page": report["summary"]["artifacts_per_page"],
                "noise_failures": report["summary"]["noise_failure_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def flatten_subset(rows: list[Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        page_indices = row.get("page_indices")
        if isinstance(page_indices, list):
            values = page_indices
        else:
            values = [row.get("page_index")]
        for value in values:
            try:
                page_index = int(value)
            except (TypeError, ValueError):
                continue
            key = (doc_id, page_index)
            if key in seen:
                continue
            seen.add(key)
            pages.append(
                {
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "selection_source": row.get("selection_source"),
                    "selection_reasons": row.get("selection_reasons", []),
                }
            )
    return pages


def audit_page(row: dict[str, Any], extract_root: Path, max_cells: int) -> dict[str, Any]:
    doc_id = str(row["doc_id"])
    page_index = int(row["page_index"])
    text = read_page_text(extract_root, doc_id, page_index)
    page_key = f"{doc_id}#p{page_index:03d}"
    if not text:
        return {
            **row,
            "page_key": page_key,
            "has_page_text": False,
            "artifacts": [],
            "type_counts": {},
            "quality_counts": {},
            "eligible_count": 0,
            "duplicate_atomic_key_count": 0,
            "noise_flags": ["missing_page_text"],
        }

    page_input = {
        "doc_id": doc_id,
        "page_index": page_index,
        "page_text": text,
        "layout_blocks": build_basic_layout_blocks(doc_id, page_index, text, has_page_image=True),
    }
    artifacts = atomicize_table_numeric_artifacts(
        selected_page={"doc_id": doc_id, "page_index": page_index},
        page_input=page_input,
        existing_artifacts=[],
        max_cells=max_cells,
    )
    type_counts = Counter(str(artifact.get("artifact_type") or "") for artifact in artifacts)
    quality_counts: Counter[str] = Counter()
    eligible_count = 0
    key_counts: Counter[tuple[str, str, str, str]] = Counter()
    samples: list[dict[str, Any]] = []
    for artifact in artifacts:
        quality = classify_artifact_quality(artifact)
        quality_counts.update(str(label) for label in quality.get("labels", []))
        reason = artifact_rerank_eligibility_reason(artifact)
        if is_atomic_strong_eligible(artifact, reason):
            eligible_count += 1
        key_counts[atomic_key(artifact)] += 1
        if len(samples) < 6:
            normalized = artifact.get("normalized_content")
            if not isinstance(normalized, dict):
                normalized = {}
            samples.append(
                {
                    "artifact_type": artifact.get("artifact_type"),
                    "content": artifact.get("content"),
                    "row_label": normalized.get("row_label") or normalized.get("row_header") or normalized.get("metric_name"),
                    "column_label": normalized.get("column_label") or normalized.get("column_header"),
                    "value_text": normalized.get("value_text"),
                    "quality_labels": quality.get("labels", []),
                }
            )
    duplicate_count = sum(count - 1 for count in key_counts.values() if count > 1)
    noise_flags = page_noise_flags(
        artifact_count=len(artifacts),
        eligible_count=eligible_count,
        duplicate_count=duplicate_count,
        quality_counts=quality_counts,
        max_artifacts=2 * max_cells,
    )
    return {
        **row,
        "page_key": page_key,
        "has_page_text": True,
        "page_text_chars": len(text),
        "artifact_count": len(artifacts),
        "type_counts": dict(sorted(type_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "eligible_count": eligible_count,
        "duplicate_atomic_key_count": duplicate_count,
        "noise_flags": noise_flags,
        "samples": samples,
    }


def page_noise_flags(
    *,
    artifact_count: int,
    eligible_count: int,
    duplicate_count: int,
    quality_counts: Counter[str],
    max_artifacts: int,
) -> list[str]:
    flags: list[str] = []
    if artifact_count > max_artifacts:
        flags.append("artifact_count_exceeds_budget")
    if artifact_count and eligible_count < artifact_count:
        flags.append("ineligible_atomic_artifacts_present")
    if duplicate_count:
        flags.append("duplicate_atomic_keys_present")
    for label in ("broad_table_only", "weak_locator", "schema_valid_but_semantically_weak", "caption_or_table_title_only"):
        if int(quality_counts.get(label, 0) or 0) > 0:
            flags.append(f"quality_{label}")
    return flags


def build_report(args: argparse.Namespace, page_reports: list[dict[str, Any]], r037: dict[str, Any]) -> dict[str, Any]:
    total_artifacts = sum(int(row.get("artifact_count", 0) or 0) for row in page_reports)
    pages_audited = len(page_reports)
    type_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    for row in page_reports:
        type_counts.update(row.get("type_counts", {}))
        quality_counts.update(row.get("quality_counts", {}))
        doc_counts[str(row.get("doc_id") or "")] += 1
    artifacts_per_page = round(total_artifacts / pages_audited, 6) if pages_audited else 0.0
    r037_delta = r037.get("delta_eligibility") if isinstance(r037.get("delta_eligibility"), dict) else {}
    r037_pages = int(r037_delta.get("eligible_pages_with_atomic_artifact", 10) or 10)
    r037_artifacts = int(r037_delta.get("total_artifacts", 0) or 0)
    r037_artifacts_per_page = round(r037_artifacts / r037_pages, 6) if r037_pages else 0.0
    growth = round(artifacts_per_page / r037_artifacts_per_page, 6) if r037_artifacts_per_page else None
    noise_failures = [row["page_key"] for row in page_reports if row.get("noise_flags")]
    checks = {
        "no_provider_calls": True,
        "not_merged_into_cumulative_artifacts": True,
        "no_artifacts_jsonl_written": True,
        "artifact_budget_per_page_respected": all(int(row.get("artifact_count", 0) or 0) <= int(args.max_artifacts_per_page) for row in page_reports),
        "no_quality_noise_flags": not noise_failures,
        "artifact_growth_vs_r037_within_limit": growth is not None and growth <= float(args.max_artifact_growth_vs_r037),
        "all_pages_have_text": all(bool(row.get("has_page_text")) for row in page_reports),
    }
    decision = "proceed_to_small_repaired_provider_gate" if all(checks.values()) else "stop_before_provider_review_noise"
    return {
        "schema_version": "r038a_repaired_20_to_30_noise_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": {
            "audit_only": True,
            "offline_atomicizer_only": True,
            "no_provider_calls": True,
            "no_stage2_compile": True,
            "no_activation_scan": True,
            "no_qa": True,
            "no_effectiveness_claim": True,
            "no_graph": True,
            "no_rerank_tuning": True,
            "not_merged_into_cumulative_artifacts": True,
            "uses_gold_fields": False,
            "model_config_used": None,
        },
        "inputs": {
            "subset": str(args.subset),
            "extract_root": str(args.extract_root),
            "r037_report": str(args.r037_report),
            "atomicizer_max_cells": int(args.atomicizer_max_cells),
        },
        "thresholds": {
            "max_artifacts_per_page": int(args.max_artifacts_per_page),
            "max_artifact_growth_vs_r037": float(args.max_artifact_growth_vs_r037),
        },
        "summary": {
            "pages_audited": pages_audited,
            "docs_audited": len([doc for doc in doc_counts if doc]),
            "total_artifacts": total_artifacts,
            "artifacts_per_page": artifacts_per_page,
            "r037_delta_artifacts_per_page": r037_artifacts_per_page,
            "artifact_growth_vs_r037": growth,
            "eligible_count": sum(int(row.get("eligible_count", 0) or 0) for row in page_reports),
            "duplicate_atomic_key_count": sum(int(row.get("duplicate_atomic_key_count", 0) or 0) for row in page_reports),
            "noise_failure_count": len(noise_failures),
            "noise_failure_pages": noise_failures,
            "type_counts": dict(sorted(type_counts.items())),
            "quality_counts": dict(sorted(quality_counts.items())),
            "pages_by_doc": dict(sorted(doc_counts.items())),
        },
        "checks": checks,
        "page_reports": page_reports,
        "next_step": next_step(decision),
    }


def next_step(decision: str) -> str:
    if decision == "proceed_to_small_repaired_provider_gate":
        return "Run a very small repaired provider gate on 2-3 R028 pages before any full 20 -> 30 replay."
    return "Do not call the provider yet; inspect page-level noise flags and tighten selection or atomicizer filters."


def atomic_key(artifact: dict[str, Any]) -> tuple[str, str, str, str]:
    normalized = artifact.get("normalized_content")
    if not isinstance(normalized, dict):
        normalized = {}
    return (
        str(artifact.get("artifact_type") or ""),
        compact(normalized.get("row_label") or normalized.get("row_header") or normalized.get("metric_name")),
        compact(normalized.get("column_label") or normalized.get("column_header")),
        compact(normalized.get("value_text") or normalized.get("value") or normalized.get("metric_value")),
    )


def read_page_text(extract_root: Path, doc_id: str, page_index: int) -> str:
    paths = build_mdocagent_extract_paths(extract_root, doc_id, page_index)
    for path in paths["text_candidate_paths"]:
        file_path = Path(path)
        if file_path.is_file():
            return file_path.read_text(encoding="utf-8", errors="replace")
    return ""


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# R038a Repaired 20 -> 30 Noise Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Scope",
        "- Offline atomicizer noise audit only.",
        "- No provider calls, no Stage 2 compile, no artifact-store merge.",
        "- No activation scan, QA, graph, effectiveness claim, or rerank tuning.",
        "- No model config or API key is used by this audit.",
        "",
        "## Summary",
        f"- Pages/docs audited: {summary['pages_audited']} / {summary['docs_audited']}",
        f"- Total artifacts: {summary['total_artifacts']}",
        f"- Artifacts/page: {summary['artifacts_per_page']}",
        f"- R037 delta artifacts/page: {summary['r037_delta_artifacts_per_page']}",
        f"- Artifact growth vs R037: {summary['artifact_growth_vs_r037']}",
        f"- Eligible atomic artifacts: {summary['eligible_count']}",
        f"- Duplicate atomic keys: {summary['duplicate_atomic_key_count']}",
        f"- Noise failure pages: {summary['noise_failure_count']}",
        f"- Type counts: `{json.dumps(summary['type_counts'], sort_keys=True)}`",
        f"- Quality counts: `{json.dumps(summary['quality_counts'], sort_keys=True)}`",
        "",
        "## Checks",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Page Results"])
    for row in report["page_reports"]:
        lines.append(
            f"- {row['page_key']}: artifacts={row.get('artifact_count', 0)}, "
            f"eligible={row.get('eligible_count', 0)}, flags={row.get('noise_flags', [])}"
        )
    lines.extend(["", "## Next Step", report["next_step"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


if __name__ == "__main__":
    main()
