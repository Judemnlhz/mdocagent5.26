#!/usr/bin/env python3
"""R045 support/citation-aware post-hoc rubric for R044 diagnostics.

No provider calls, no new prediction, no evaluation, and no full QA. This
script inspects R044 transition cases plus a fixed all-miss sample and writes a
manual support attribution report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_R044_REPORT = "outputs/heldout/r044_small_contrastive_provider/r044_diagnostic_attribution_report.json"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r045_support_rubric"
CONDITIONS = [
    "original_pages_only",
    "page_rerank_only",
    "original_pages_plus_artifact_snippets",
    "artifact_snippets_only",
]


MANUAL_OVERRIDES: dict[int, dict[str, Any]] = {
    384: {
        "rubric_label": "artifact_injection_introduces_false_positive_risk",
        "support_summary": "Page text supports a refusal because the visible revision is May 2016, not May 2018. Artifact snippets mention Strategic Planning Services Team on page 10 but do not support the requested producer/date relation.",
        "condition_support": {
            "original_pages_only": "supported_refusal",
            "page_rerank_only": "supported_refusal",
            "original_pages_plus_artifact_snippets": "mixed_refusal_with_unneeded_answer",
            "artifact_snippets_only": "unsupported_false_positive",
        },
        "artifact_evidence_status": "misleading_or_irrelevant",
        "page_text_evidence_status": "sufficient_for_refusal",
        "action": "Add an unsupported-answer guard for artifact snippets on not-answerable questions.",
    },
    508: {
        "rubric_label": "rerank_and_artifact_context_help_unanswerable_refusal",
        "support_summary": "The visible Arkansas codes include AR01 and AR02 but not AR03. Original pages led to a false positive, while reranked pages, page+artifact snippets, and snippet-only conditions all support refusal.",
        "condition_support": {
            "original_pages_only": "unsupported_false_positive",
            "page_rerank_only": "supported_refusal",
            "original_pages_plus_artifact_snippets": "supported_refusal",
            "artifact_snippets_only": "supported_refusal",
        },
        "artifact_evidence_status": "sufficient_for_absence_check",
        "page_text_evidence_status": "sufficient_when_ordered_correctly",
        "action": "Preserve this as a positive diagnostic for unanswerable guard behavior, not as an answer-generation win.",
    },
    569: {
        "rubric_label": "diagnostic_gold_match_undercounts_supported_refusal",
        "support_summary": "The task asks about children with STEM degrees, but visible contexts do not provide that data. Page-rerank and snippet-only predictions explicitly refuse; the simple R044 matcher counted only page-rerank as a match.",
        "condition_support": {
            "original_pages_only": "unsupported_partial_calculation",
            "page_rerank_only": "supported_refusal",
            "original_pages_plus_artifact_snippets": "unsupported_partial_calculation",
            "artifact_snippets_only": "supported_refusal",
        },
        "artifact_evidence_status": "supports_insufficient_data_not_answer_value",
        "page_text_evidence_status": "supports_insufficient_data_when_prompt_order_emphasizes_relevant_pages",
        "action": "Improve diagnostic matching for insufficient-data refusals and separate refusal quality from exact string matching.",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r044-report", default=DEFAULT_R044_REPORT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--all-miss-sample-size", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    report_json = output_root / "r045_support_rubric_report.json"
    report_md = output_root / "r045_support_rubric_report.md"
    cases_jsonl = output_root / "r045_support_rubric_cases.jsonl"
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "r044_report": args.r044_report,
                    "output_root": str(output_root),
                    "no_provider_calls": True,
                    "no_full_qa": True,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    r044 = read_json(Path(args.r044_report))
    cases = build_cases(r044, args.all_miss_sample_size)
    report = build_report(args, r044, cases)
    write_jsonl(cases_jsonl, cases)
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "num_cases": report["num_cases"],
                "transition_cases": report["case_counts"]["transition_case"],
                "all_miss_samples": report["case_counts"]["all_conditions_miss_sample"],
                "report_md": str(report_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_cases(r044: dict[str, Any], all_miss_sample_size: int) -> list[dict[str, Any]]:
    transition_rows = [
        row
        for row in r044["per_record"]
        if row.get("transition_labels") and row.get("transition_labels") != ["all_conditions_miss"]
    ]
    all_miss_rows = [
        row
        for row in r044["per_record"]
        if row.get("transition_labels") == ["all_conditions_miss"]
    ][:all_miss_sample_size]
    cases = []
    for row in transition_rows + all_miss_rows:
        record_id = int(row["record_id"])
        override = MANUAL_OVERRIDES.get(record_id)
        case_type = "transition_case" if row in transition_rows else "all_conditions_miss_sample"
        if override is None:
            override = default_all_miss_rubric(row)
        cases.append(
            {
                "schema_version": "r045_support_rubric_case_v1",
                "record_id": record_id,
                "case_type": case_type,
                "r042_label": row.get("r042_label"),
                "r044_transition_labels": row.get("transition_labels"),
                "question": row.get("question"),
                "gold_answer_for_posthoc_diagnostic_only": row.get("gold_answer_for_posthoc_diagnostic_only"),
                "condition_predictions": row.get("condition_predictions"),
                "r044_diagnostic_gold_match": row.get("diagnostic_gold_match"),
                "r044_match_level": row.get("match_level"),
                **override,
            }
        )
    return cases


def default_all_miss_rubric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rubric_label": "all_conditions_miss_requires_error_analysis",
        "support_summary": "All four R044 conditions missed under the diagnostic matcher. This fixed sample is retained for manual error analysis, not for aggregate scoring.",
        "condition_support": {condition: "not_manually_adjudicated" for condition in CONDITIONS},
        "artifact_evidence_status": "not_adjudicated",
        "page_text_evidence_status": "not_adjudicated",
        "action": "Inspect only if the next prompt/artifact-selection iteration needs more negative examples.",
    }


def build_report(args: argparse.Namespace, r044: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(case["rubric_label"] for case in cases)
    case_counts = Counter(case["case_type"] for case in cases)
    support_counts = Counter()
    for case in cases:
        for condition, label in case["condition_support"].items():
            support_counts[f"{condition}:{label}"] += 1
    return {
        "schema_version": "r045_support_rubric_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r045_support_rubric_complete",
        "scope": {
            "posthoc_support_rubric_only": True,
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
        },
        "inputs": {
            "r044_report": args.r044_report,
        },
        "num_cases": len(cases),
        "case_counts": dict(case_counts),
        "rubric_label_counts": dict(sorted(labels.items())),
        "condition_support_counts": dict(sorted(support_counts.items())),
        "key_findings": [
            "R044's simple diagnostic matcher undercounted at least one supported refusal: record 569 snippet-only is a supported insufficient-data answer.",
            "Record 384 shows artifact snippets can introduce unsupported false-positive risk on not-answerable questions.",
            "Record 508 is the clearest positive diagnostic: reranked pages and artifact snippets support refusing AR03 rather than hallucinating a market.",
            "The next iteration should improve refusal/support rubric and artifact selection before any full-data run.",
        ],
        "recommended_next": [
            "Implement question-aware artifact selection rather than fixed first-N artifacts per page.",
            "Add explicit unsupported-answer/refusal instructions for Not answerable cases.",
            "Require cited page ids or artifact ids in future diagnostic prompts before reporting any broader score.",
            "Treat R044 counts as preliminary; R045 support labels supersede simple gold-match labels for transition cases.",
        ],
        "cases": cases,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R045 Support/Citation-Aware Rubric",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- Post-hoc support rubric only.",
        "- No provider calls, no new prediction, no new evaluation, no full QA.",
        "- Not an official score.",
        "",
        "## Key Findings",
    ]
    for item in report["key_findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Cases",
            "| record_id | case_type | rubric_label | support summary |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for case in report["cases"]:
        summary = str(case["support_summary"]).replace("|", "/")
        lines.append(f"| {case['record_id']} | {case['case_type']} | {case['rubric_label']} | {summary} |")
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
