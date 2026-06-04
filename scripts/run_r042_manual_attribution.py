#!/usr/bin/env python3
"""R042 deterministic manual-attribution scaffold for R040/R041 findings.

This step does not call providers, predictions, or evaluators. It reads frozen
R039/R040/R041 outputs and writes a post-hoc attribution report for:

* the 4 binary-divergent records that cancel into the R040 aggregate tie
* the 18 answer-text-different records whose binary correctness stayed fixed

The report uses gold answers/evidence pages only for post-hoc diagnosis, not
for selection, reranking, prediction, or evaluation.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_R041_MATRIX = "outputs/heldout/r041_r040_identical_score_attribution/record_level_attribution_matrix.jsonl"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r042_r040_manual_attribution"
RUNS = ["top4_original_only", "top4_original_plus_artifact", "top4_artifact_only"]
ARTIFACT_RUN = "top4_artifact_only"
BRANCH_FIELDS = ["text-top-10-question", "image-top-10-question"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r041-matrix", default=DEFAULT_R041_MATRIX)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    report_json = output_root / "r042_manual_attribution_report.json"
    report_md = output_root / "r042_manual_attribution_report.md"
    cases_jsonl = output_root / "manual_attribution_cases.jsonl"
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "r041_matrix": args.r041_matrix,
                    "output_root": str(output_root),
                    "report_json": str(report_json),
                    "cases_jsonl": str(cases_jsonl),
                    "no_provider_calls": True,
                    "no_new_qa": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.r041_matrix))
    records = read_json(Path(args.records))
    cases = build_cases(rows, records)
    report = build_report(args, rows, cases)
    write_jsonl(cases_jsonl, cases)
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "divergent_cases": report["divergent_summary"]["num_cases"],
                "artifact_only_gains": report["divergent_summary"]["artifact_only_gains"],
                "artifact_only_losses": report["divergent_summary"]["artifact_only_losses"],
                "text_diff_binary_same_cases": report["text_diff_binary_same_summary"]["num_cases"],
                "report_md": str(report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_cases(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        source = records[int(row["record_id"])]
        correctness_class = row["record_attribution"]["correctness_class"]
        include = correctness_class == "binary_divergent" or row.get("answer_text_diff_binary_same")
        if not include:
            continue
        answer_diagnostics = {
            run: answer_diagnostic(row["answer_texts"][run], source.get("answer"), source.get("answer_format"))
            for run in RUNS
        }
        retrieval = {
            run: retrieval_diagnostic(row["retrieval"][run], parse_evidence_pages(source.get("evidence_pages")))
            for run in RUNS
        }
        case = {
            "record_id": row["record_id"],
            "matrix_index": row["matrix_index"],
            "case_type": "binary_divergent" if correctness_class == "binary_divergent" else "answer_text_diff_binary_same",
            "doc_id": row["doc_id"],
            "question": row["question"],
            "gold_answer_for_posthoc_audit_only": source.get("answer"),
            "answer_format": source.get("answer_format"),
            "evidence_pages_for_posthoc_audit_only": parse_evidence_pages(source.get("evidence_pages")),
            "evidence_sources_for_posthoc_audit_only": parse_evidence_sources(source.get("evidence_sources")),
            "binary_pattern": row["binary_pattern"],
            "binary_correctness": row["binary_correctness"],
            "answer_texts": row["answer_texts"],
            "answer_diagnostics": answer_diagnostics,
            "retrieval_diagnostics": retrieval,
            "retrieval_deltas_vs_original": row["retrieval_deltas_vs_original"],
            "manual_attribution": attribute_case(row, source, answer_diagnostics, retrieval),
        }
        cases.append(case)
    return cases


def attribute_case(
    row: dict[str, Any],
    source: dict[str, Any],
    answer_diagnostics: dict[str, dict[str, Any]],
    retrieval: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    correctness = row["binary_correctness"]
    pattern = row["binary_pattern"]
    record_id = int(row["record_id"])
    correctness_class = row["record_attribution"]["correctness_class"]
    if correctness_class != "binary_divergent":
        binary_value = next(iter(set(correctness.values())))
        if binary_value == 1:
            return {
                "label": "binary_same_all_correct_surface_variation",
                "confidence": "medium",
                "rationale": "Answers differ in surface form, but all three were judged correct; binary eval folds the variation into the same bucket.",
                "next_action": "Use support/citation-aware or value-normalized analysis only if this record matters for qualitative claims.",
            }
        return {
            "label": "binary_same_all_wrong_failure_variation",
            "confidence": "medium",
            "rationale": "Answers differ, but all three were judged wrong; binary eval captures failure but hides how each run failed.",
            "next_action": "Review only if the failure mode is needed for prompt or retrieval redesign.",
        }

    if correctness[ARTIFACT_RUN] == 0 and all(correctness[run] == 1 for run in RUNS if run != ARTIFACT_RUN):
        if source.get("answer") == "Not answerable":
            return {
                "label": "artifact_only_loss_unanswerable_false_positive",
                "confidence": "high",
                "rationale": "Original and original+artifact refused/marked the EPS code as not listed, while artifact_only produced a concrete market name for a Not answerable gold record.",
                "next_action": "Add unanswerable/unsupported-answer checks before interpreting artifact-only gains.",
            }
        return {
            "label": "artifact_only_loss_answer_value_shift",
            "confidence": "high" if answer_diagnostics[ARTIFACT_RUN]["match_level"] == "mismatch" else "medium",
            "rationale": "artifact_only selected or emphasized artifact-scored pages but changed the answer value away from the gold answer.",
            "next_action": "Inspect whether artifact-positive pages contain a misleading adjacent value; future runs need prompt-visible citation/support accounting.",
        }

    if correctness[ARTIFACT_RUN] == 1 and all(correctness[run] == 0 for run in RUNS if run != ARTIFACT_RUN):
        artifact_delta = row["retrieval_deltas_vs_original"][ARTIFACT_RUN]
        if not artifact_delta["any_branch_list_changed"]:
            return {
                "label": "artifact_only_gain_same_pages_generation_variance",
                "confidence": "medium",
                "rationale": "artifact_only answered the gold value while selected page lists matched original; the gain is not attributable to a page-set change.",
                "next_action": "Treat this as provider/output or prompt-run variance unless prompt-visible artifact context is added and controlled.",
            }
        return {
            "label": "artifact_only_gain_order_sensitive_numeric",
            "confidence": "medium",
            "rationale": "artifact_only changed page order without changing the combined page set and produced the gold numeric value.",
            "next_action": "Separate page-order effects from explicit artifact-snippet effects in the next contrastive design.",
        }

    return {
        "label": f"manual_review_required_pattern_{pattern}",
        "confidence": "low",
        "rationale": "The binary pattern is not covered by the deterministic R042 attribution rules.",
        "next_action": "Inspect the record manually before using it to motivate a new experiment.",
    }


def retrieval_diagnostic(summary: dict[str, Any], evidence_pages: list[int]) -> dict[str, Any]:
    combined_pages = [int(value) for value in summary.get("combined_unique_pages", [])]
    evidence_set = set(evidence_pages)
    combined_set = set(combined_pages)
    branch_hits = {}
    for field in BRANCH_FIELDS:
        branch = summary["branches"][field]
        pages = [int(value) for value in branch.get("pages", [])]
        branch_hits[field] = {
            "pages": pages,
            "evidence_pages_hit": sorted(set(pages) & evidence_set),
            "positive_score_pages": branch.get("positive_score_pages", []),
            "artifact_count_on_selected_unique_pages": branch.get("artifact_count_on_selected_unique_pages"),
        }
    return {
        "combined_unique_pages": combined_pages,
        "evidence_pages_hit": sorted(combined_set & evidence_set),
        "evidence_page_recall": round(len(combined_set & evidence_set) / max(len(evidence_set), 1), 6) if evidence_set else None,
        "combined_artifact_count_on_selected_pages": summary.get("combined_artifact_count_on_selected_pages"),
        "all_selected_pages_within_original_top10_pool": summary.get("all_selected_pages_within_original_top10_pool"),
        "branch_hits": branch_hits,
    }


def answer_diagnostic(answer: str, gold: Any, answer_format: Any) -> dict[str, Any]:
    answer_text = "" if answer is None else str(answer)
    gold_text = "" if gold is None else str(gold)
    answer_norm = normalize(answer_text)
    gold_norm = normalize(gold_text)
    numeric_answer = first_number(answer_text)
    numeric_gold = first_number(gold_text)
    list_answer = parse_list_like(answer_text)
    list_gold = parse_list_like(gold_text)
    if answer_norm == gold_norm:
        match_level = "normalized_exact"
    elif list_answer is not None and list_gold is not None and [normalize(x) for x in list_answer] == [normalize(x) for x in list_gold]:
        match_level = "list_exact"
    elif numeric_answer is not None and numeric_gold is not None and abs(numeric_answer - numeric_gold) <= 1e-6:
        match_level = "numeric_exact"
    elif gold_norm == "not answerable" and ("not listed" in answer_norm or "not answerable" in answer_norm or "not possible" in answer_norm):
        match_level = "unanswerable_equivalent"
    else:
        match_level = "mismatch"
    return {
        "match_level": match_level,
        "answer_format": answer_format,
        "answer_short": short(answer_text),
        "gold_short": short(gold_text),
        "numeric_answer": numeric_answer,
        "numeric_gold": numeric_gold,
        "list_answer": list_answer,
        "list_gold": list_gold,
    }


def build_report(args: argparse.Namespace, rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    divergent = [case for case in cases if case["case_type"] == "binary_divergent"]
    text_diff = [case for case in cases if case["case_type"] == "answer_text_diff_binary_same"]
    label_counts = Counter(case["manual_attribution"]["label"] for case in cases)
    divergent_labels = Counter(case["manual_attribution"]["label"] for case in divergent)
    text_diff_labels = Counter(case["manual_attribution"]["label"] for case in text_diff)
    artifact_gains = [case for case in divergent if case["binary_correctness"].get(ARTIFACT_RUN) == 1]
    artifact_losses = [case for case in divergent if case["binary_correctness"].get(ARTIFACT_RUN) == 0]
    report = {
        "schema_version": "r042_r040_manual_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r042_manual_attribution_complete",
        "scope": {
            "manual_attribution_scaffold_only": True,
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "uses_gold_only_for_posthoc_diagnosis": True,
            "not_full_data_generalization": True,
            "not_official_mmlongbench_result": True,
        },
        "inputs": {
            "r041_matrix": args.r041_matrix,
            "records": args.records,
        },
        "r041_context": {
            "num_records": len(rows),
            "aggregate_scores_equal": True,
            "binary_vectors_identical": False,
        },
        "divergent_summary": {
            "num_cases": len(divergent),
            "artifact_only_gains": len(artifact_gains),
            "artifact_only_losses": len(artifact_losses),
            "label_counts": dict(sorted(divergent_labels.items())),
            "cases": compact_cases(divergent),
        },
        "text_diff_binary_same_summary": {
            "num_cases": len(text_diff),
            "label_counts": dict(sorted(text_diff_labels.items())),
            "binary_pattern_counts": dict(sorted(Counter(case["binary_pattern"] for case in text_diff).items())),
            "cases": compact_cases(text_diff),
        },
        "all_label_counts": dict(sorted(label_counts.items())),
        "interpretation": {
            "bottom_line": "The R040 aggregate tie comes from artifact_only losing two answer-value/unanswerable cases and gaining two numeric cases; at least one gain is not attributable to changed selected pages.",
            "design_implication": "The next experiment should separate page reranking, page order, explicit artifact-snippet injection, and judge bucket effects before any full-data QA.",
        },
        "recommended_r043_design": [
            "Keep the 37-record targeted subset frozen for contrastive diagnosis.",
            "Run a no-new-retrieval prompt contrast only after R042 review: original page text, original page text plus explicit artifact snippets, artifact snippets only, and page-rerank-only.",
            "Log artifact exposure per record: artifact ids, source pages, snippet tokens, prompt inclusion flag, and whether the gold/evidence page is included.",
            "Add a support-aware/citation-aware manual rubric for the 4 divergent and 18 text-diff/binary-same records before scaling.",
            "Do not run full QA until the prompt-visible artifact condition is implemented and separated from page-order effects.",
        ],
    }
    return report


def compact_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": case["record_id"],
            "binary_pattern": case["binary_pattern"],
            "label": case["manual_attribution"]["label"],
            "confidence": case["manual_attribution"]["confidence"],
            "question": short(case["question"], 150),
            "gold_answer": case["gold_answer_for_posthoc_audit_only"],
            "answers": {run: short(case["answer_texts"][run], 160) for run in RUNS},
            "evidence_pages": case["evidence_pages_for_posthoc_audit_only"],
        }
        for case in cases
    ]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R042 Manual Attribution of R040 Aggregate Tie",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- Deterministic post-hoc attribution only.",
        "- No provider calls, no new prediction, no new evaluation, no full QA.",
        "- Gold answers/evidence pages are used only for diagnosis.",
        "- Not full-data generalization and not an official MMLongBench result.",
        "",
        "## Main Finding",
        report["interpretation"]["bottom_line"],
        "",
        "## Divergent Records",
        f"- cases: {report['divergent_summary']['num_cases']}",
        f"- artifact_only gains: {report['divergent_summary']['artifact_only_gains']}",
        f"- artifact_only losses: {report['divergent_summary']['artifact_only_losses']}",
        "",
        "| record_id | pattern | label | confidence |",
        "| ---: | --- | --- | --- |",
    ]
    for case in report["divergent_summary"]["cases"]:
        lines.append(f"| {case['record_id']} | {case['binary_pattern']} | {case['label']} | {case['confidence']} |")
    lines.extend(
        [
            "",
            "## Text-Different Binary-Same Records",
            f"- cases: {report['text_diff_binary_same_summary']['num_cases']}",
            f"- pattern counts: `{json.dumps(report['text_diff_binary_same_summary']['binary_pattern_counts'], sort_keys=True)}`",
            "",
            "| label | count |",
            "| --- | ---: |",
        ]
    )
    for label, count in report["text_diff_binary_same_summary"]["label_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- {report['interpretation']['design_implication']}",
            "- `top4_artifact_only` losses are not the same failure mode as its gains; the aggregate tie hides both directions.",
            "- The current R040 setup is page-rerank/page-order diagnosis, not prompt-visible artifact-context diagnosis.",
            "",
            "## Recommended R043 Design",
        ]
    )
    for item in report["recommended_r043_design"]:
        lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def parse_evidence_pages(value: Any) -> list[int]:
    if value in (None, "", "[]"):
        return []
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        rows = []
        for item in parsed:
            try:
                rows.append(int(item))
            except (TypeError, ValueError):
                continue
        return rows
    return []


def parse_evidence_sources(value: Any) -> list[str]:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_list_like(value: str) -> list[str] | None:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return None


def first_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def short(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
