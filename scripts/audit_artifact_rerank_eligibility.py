#!/usr/bin/env python3
"""Audit which Stage 2 artifacts are eligible to affect artifact reranking."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdocnexus.integration.mdocagent_adapter import (  # noqa: E402
    artifact_anchor_types,
    artifact_locator_kinds,
    artifact_rerank_eligibility_reason,
    read_records,
)


def main() -> None:
    args = parse_args()
    rows = [row for row in read_records(args.artifacts) if isinstance(row, dict)]
    report = build_report(rows, str(args.artifacts))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        write_markdown(report, Path(args.output_md))
    print(json.dumps({"output_json": str(output_json), "eligible_artifacts": report["eligible_artifacts"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="Stage 2 artifacts JSON/JSONL path")
    parser.add_argument("--output-json", required=True, help="Output quality report JSON path")
    parser.add_argument("--output-md", help="Optional output quality report Markdown path")
    return parser.parse_args()


def build_report(rows: list[dict[str, Any]], artifact_path: str) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    eligible_type_counts: Counter[str] = Counter()
    locator_counts: Counter[str] = Counter()
    eligible_locator_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    eligible_docs: set[str] = set()
    eligible_pages: set[tuple[str, int]] = set()
    examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for artifact in rows:
        reason = artifact_rerank_eligibility_reason(artifact)
        reason_counts[reason] += 1
        artifact_type = str(artifact.get("artifact_type") or "")
        type_counts[artifact_type] += 1
        for kind in artifact_locator_kinds(artifact):
            locator_counts[kind] += 1
        for anchor_type in artifact_anchor_types(artifact):
            anchor_counts[anchor_type] += 1
        if len(examples_by_reason[reason]) < 5:
            examples_by_reason[reason].append(public_example(artifact))
        if reason == "eligible":
            eligible_type_counts[artifact_type] += 1
            doc_id = str(artifact.get("doc_id") or "")
            page_index = int(artifact.get("page_index", -1))
            if doc_id:
                eligible_docs.add(doc_id)
                eligible_pages.add((doc_id, page_index))
            for kind in artifact_locator_kinds(artifact):
                eligible_locator_counts[kind] += 1

    total = len(rows)
    eligible = reason_counts["eligible"]
    return {
        "schema_version": "artifact_rerank_eligibility_audit_v1",
        "artifact_path": artifact_path,
        "total_artifacts": total,
        "eligible_artifacts": eligible,
        "eligible_rate": eligible / max(total, 1),
        "eligible_docs": len(eligible_docs),
        "eligible_pages": len(eligible_pages),
        "reason_counts": dict(sorted(reason_counts.items())),
        "artifact_type_counts": dict(sorted(type_counts.items())),
        "eligible_artifact_type_counts": dict(sorted(eligible_type_counts.items())),
        "locator_kind_counts": dict(sorted(locator_counts.items())),
        "eligible_locator_kind_counts": dict(sorted(eligible_locator_counts.items())),
        "anchor_type_counts": dict(sorted(anchor_counts.items())),
        "examples_by_reason": {key: value for key, value in sorted(examples_by_reason.items())},
        "no_gold_fields_used": True,
    }


def public_example(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": artifact.get("artifact_id"),
        "doc_id": artifact.get("doc_id"),
        "page_index": artifact.get("page_index"),
        "artifact_type": artifact.get("artifact_type"),
        "locator_kinds": sorted(artifact_locator_kinds(artifact)),
        "anchor_types": sorted(artifact_anchor_types(artifact)),
        "content_preview": str(artifact.get("content") or "")[:160],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Artifact Rerank Eligibility Audit",
        "",
        f"Artifact path: `{report['artifact_path']}`",
        f"Total artifacts: {report['total_artifacts']}",
        f"Eligible artifacts: {report['eligible_artifacts']}",
        f"Eligible rate: {report['eligible_rate']:.6f}",
        f"Eligible docs: {report['eligible_docs']}",
        f"Eligible pages: {report['eligible_pages']}",
        "",
        "## Rejection Reasons",
    ]
    for reason, count in report["reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.append("")
    lines.append("## Eligible Types")
    for artifact_type, count in report["eligible_artifact_type_counts"].items():
        lines.append(f"- `{artifact_type}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
