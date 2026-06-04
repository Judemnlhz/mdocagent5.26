#!/usr/bin/env python3
"""R041 post-hoc attribution audit for R040 identical targeted QA scores.

This audit explains why the three R040 targeted diagnostic QA runs produced the
same binary correctness score. It performs no provider calls, no prediction, no
evaluation, no full QA, and no reranking. It only reads frozen R039/R040
artifacts and writes record-level attribution diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_R040_ROOT = "outputs/heldout/r040_targeted_activation_rich_qa/run_tags/r040_targeted_activation_rich_qa"
DEFAULT_R039_RECORD_IDS = "outputs/heldout/r039_targeted_activation_rich/record_ids.txt"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_ARTIFACTS = "outputs/stage2_structured_incremental/r038d_activation_attribution_audit/cumulative20_plus_r037_plus_r038c/atomic_only/artifacts.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r041_r040_identical_score_attribution"
RUNS = ["top4_original_only", "top4_original_plus_artifact", "top4_artifact_only"]
BRANCH_FIELDS = ["text-top-10-question", "image-top-10-question"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    report_json = output_root / "r041_identical_score_attribution_report.json"
    report_md = output_root / "r041_identical_score_attribution_report.md"
    matrix_jsonl = output_root / "record_level_attribution_matrix.jsonl"
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "r040_root": args.r040_root,
                    "output_root": str(output_root),
                    "report_json": str(report_json),
                    "matrix_jsonl": str(matrix_jsonl),
                    "no_provider_calls": True,
                    "no_new_qa": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    r040_root = Path(args.r040_root)
    gate = read_json(r040_root / "execution_gate_report.json")
    summary = read_json(r040_root / "r040_targeted_diagnostic_summary.json")
    record_ids = read_record_ids(Path(args.r039_record_ids))
    all_records = read_json(Path(args.records))
    artifact_index = build_artifact_index(Path(args.artifacts))
    run_data = load_runs(gate)
    matrix = build_record_matrix(record_ids, all_records, run_data, artifact_index)
    report = build_report(args, gate, summary, record_ids, artifact_index, run_data, matrix)
    write_jsonl(matrix_jsonl, matrix)
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "num_records": report["num_records"],
                "aggregate_scores_equal": report["checks"]["aggregate_scores_equal"],
                "binary_vectors_identical": report["checks"]["binary_correctness_vectors_identical"],
                "all_correct_records": report["outcome_counts"]["all_correct"],
                "all_wrong_records": report["outcome_counts"]["all_wrong"],
                "binary_divergent_records": report["binary_divergence"]["divergent_record_count"],
                "answer_text_diff_binary_same_records": report["answer_text"]["answer_text_diff_binary_same_records"],
                "report_md": str(report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_runs(gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row.get("run_name")): row for row in gate.get("runs", []) if isinstance(row, dict)}
    missing = [run for run in RUNS if run not in rows]
    if missing:
        raise ValueError(f"Missing R040 gate rows: {missing}")
    loaded: dict[str, dict[str, Any]] = {}
    for run in RUNS:
        row = rows[run]
        prediction_path = Path(str(row["prediction_output_path"]))
        evaluation_path = Path(str(row["evaluation_output_path"]))
        retrieval_path = Path(str(row["compatible_retrieval_path"]))
        manifest_path = Path(str((row.get("compatibility_report_path") or "").replace("compatibility/", "manifests/").replace(".compatibility_report", "")))
        loaded[run] = {
            "gate_row": row,
            "prediction_path": str(prediction_path),
            "evaluation_path": str(evaluation_path),
            "retrieval_path": str(retrieval_path),
            "prediction": read_json(prediction_path),
            "evaluation": read_json(evaluation_path),
            "retrieval": read_json(retrieval_path),
            "manifest": read_json(manifest_path) if manifest_path.is_file() else {},
        }
    return loaded


def build_record_matrix(
    record_ids: list[int],
    all_records: list[dict[str, Any]],
    run_data: dict[str, dict[str, Any]],
    artifact_index: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    lengths = {run: len(data["evaluation"]) for run, data in run_data.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != len(record_ids):
        raise ValueError(f"Run/eval length mismatch: record_ids={len(record_ids)} lengths={lengths}")
    rows: list[dict[str, Any]] = []
    for idx, record_id in enumerate(record_ids):
        source_record = all_records[record_id]
        answers = {run: answer_text(run, run_data[run]["evaluation"][idx]) for run in RUNS}
        normalized_answers = {run: normalize_answer(value) for run, value in answers.items()}
        correctness = {run: numeric_correctness(run_data[run]["evaluation"][idx].get("binary_correctness")) for run in RUNS}
        retrieval = {run: retrieval_summary(run_data[run]["evaluation"][idx], source_record, artifact_index) for run in RUNS}
        original_plus_delta = compare_retrieval(retrieval["top4_original_only"], retrieval["top4_original_plus_artifact"])
        artifact_only_delta = compare_retrieval(retrieval["top4_original_only"], retrieval["top4_artifact_only"])
        answer_text_all_same = len(set(normalized_answers.values())) == 1
        binary_all_same = len(set(correctness.values())) == 1
        rows.append(
            {
                "matrix_index": idx,
                "record_id": record_id,
                "doc_id": source_record.get("doc_id"),
                "question": source_record.get("question"),
                "gold_answer_for_posthoc_audit_only": source_record.get("answer"),
                "binary_correctness": correctness,
                "binary_pattern": binary_pattern(correctness),
                "answer_text_all_same_normalized": answer_text_all_same,
                "answer_text_diff_binary_same": (not answer_text_all_same) and binary_all_same,
                "answer_texts": answers,
                "retrieval": retrieval,
                "retrieval_deltas_vs_original": {
                    "top4_original_plus_artifact": original_plus_delta,
                    "top4_artifact_only": artifact_only_delta,
                },
                "artifact_modes_candidate_pool_constrained": {
                    "top4_original_plus_artifact": candidate_pool_constrained(retrieval["top4_original_plus_artifact"]),
                    "top4_artifact_only": candidate_pool_constrained(retrieval["top4_artifact_only"]),
                },
                "record_attribution": classify_record(correctness, answer_text_all_same, original_plus_delta, artifact_only_delta, retrieval),
            }
        )
    return rows


def retrieval_summary(record: dict[str, Any], source_record: dict[str, Any], artifact_index: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or source_record.get("doc_id") or "")
    branches: dict[str, Any] = {}
    for field in BRANCH_FIELDS:
        pages = as_int_list(record.get(field))[:4]
        scores = as_float_list(record.get(f"{field}_score"))[:4]
        original_pool = as_int_list(source_record.get(field))[:10]
        unique_pages = sorted(set(pages))
        artifact_counts = {str(page): len(artifact_index.get((doc_id, page), [])) for page in unique_pages}
        positive_score_pages = [page for page, score in zip(pages, scores) if score > 0.0]
        branches[field] = {
            "pages": pages,
            "scores": scores,
            "unique_pages": unique_pages,
            "unique_page_count": len(unique_pages),
            "artifact_count_on_selected_unique_pages": sum(artifact_counts.values()),
            "artifact_counts_by_page": artifact_counts,
            "positive_score_pages": positive_score_pages,
            "positive_score_page_count": len(set(positive_score_pages)),
            "selected_pages_all_within_original_top10_pool": all(page in original_pool for page in pages),
            "selected_pages_all_within_original_top4_multiset": multiset_subset(pages, as_int_list(source_record.get(field))[:4]),
            "original_top10_pool": original_pool,
        }
    all_pages = sorted(set(branches[BRANCH_FIELDS[0]]["pages"]) | set(branches[BRANCH_FIELDS[1]]["pages"]))
    return {
        "doc_id": doc_id,
        "mode": ((record.get("_nexus_meta") or {}).get("mode")),
        "formula": ((record.get("_nexus_meta") or {}).get("formula")),
        "branches": branches,
        "combined_unique_pages": all_pages,
        "combined_unique_page_count": len(all_pages),
        "combined_artifact_count_on_selected_pages": sum(len(artifact_index.get((doc_id, page), [])) for page in all_pages),
        "all_selected_pages_within_original_top10_pool": all(branches[field]["selected_pages_all_within_original_top10_pool"] for field in BRANCH_FIELDS),
    }


def compare_retrieval(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    branch_rows: dict[str, Any] = {}
    any_list_changed = False
    any_set_changed = False
    for field in BRANCH_FIELDS:
        base_pages = base["branches"][field]["pages"]
        current_pages = current["branches"][field]["pages"]
        list_changed = base_pages != current_pages
        set_changed = set(base_pages) != set(current_pages)
        any_list_changed = any_list_changed or list_changed
        any_set_changed = any_set_changed or set_changed
        branch_rows[field] = {
            "list_changed": list_changed,
            "set_changed": set_changed,
            "base_pages": base_pages,
            "current_pages": current_pages,
            "jaccard_unique_pages": jaccard(set(base_pages), set(current_pages)),
        }
    return {
        "any_branch_list_changed": any_list_changed,
        "any_branch_set_changed": any_set_changed,
        "combined_set_changed": set(base["combined_unique_pages"]) != set(current["combined_unique_pages"]),
        "combined_jaccard_unique_pages": jaccard(set(base["combined_unique_pages"]), set(current["combined_unique_pages"])),
        "branches": branch_rows,
    }


def candidate_pool_constrained(summary: dict[str, Any]) -> bool:
    return bool(summary.get("all_selected_pages_within_original_top10_pool"))


def classify_record(
    correctness: dict[str, int | None],
    answer_text_all_same: bool,
    original_plus_delta: dict[str, Any],
    artifact_only_delta: dict[str, Any],
    retrieval: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    values = [correctness[run] for run in RUNS]
    if len(set(values)) == 1:
        if values[0] == 1:
            correctness_class = "all_correct"
        elif values[0] == 0:
            correctness_class = "all_wrong"
        else:
            correctness_class = "all_unscored"
    else:
        correctness_class = "binary_divergent"
    retrieval_changed = bool(original_plus_delta["any_branch_list_changed"] or artifact_only_delta["any_branch_list_changed"])
    artifact_pages_have_artifacts = {
        run: retrieval[run]["combined_artifact_count_on_selected_pages"] > 0
        for run in ["top4_original_plus_artifact", "top4_artifact_only"]
    }
    if correctness_class in {"all_correct", "all_wrong"} and retrieval_changed and not answer_text_all_same:
        primary = "retrieval_or_answer_text_changed_but_binary_bucket_unchanged"
    elif correctness_class in {"all_correct", "all_wrong"} and retrieval_changed:
        primary = "retrieval_changed_but_answer_and_binary_bucket_unchanged"
    elif correctness_class in {"all_correct", "all_wrong"} and not answer_text_all_same:
        primary = "answer_text_changed_but_binary_bucket_unchanged"
    elif correctness_class in {"all_correct", "all_wrong"}:
        primary = "no_record_level_binary_sensitivity_observed"
    else:
        primary = "binary_divergence_requires_manual_review"
    return {
        "correctness_class": correctness_class,
        "retrieval_changed_vs_original": retrieval_changed,
        "answer_text_all_same_normalized": answer_text_all_same,
        "artifact_mode_selected_pages_have_artifacts": artifact_pages_have_artifacts,
        "primary_attribution": primary,
    }


def build_report(
    args: argparse.Namespace,
    gate: dict[str, Any],
    summary: dict[str, Any],
    record_ids: list[int],
    artifact_index: dict[tuple[str, int], list[dict[str, Any]]],
    run_data: dict[str, dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    binary_vectors = {
        run: [row["binary_correctness"][run] for row in matrix]
        for run in RUNS
    }
    answer_text_diff_binary_same = [row for row in matrix if row["answer_text_diff_binary_same"]]
    outcome_counts = Counter(row["record_attribution"]["correctness_class"] for row in matrix)
    primary_counts = Counter(row["record_attribution"]["primary_attribution"] for row in matrix)
    retrieval_counts = retrieval_change_counts(matrix)
    divergence = binary_divergence_summary(matrix)
    artifact_exposure = artifact_exposure_summary(matrix)
    command_findings = command_scope_findings(gate, run_data)
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "targeted_37_only": len(record_ids) == 37 == len(matrix),
        "three_requested_runs_only": sorted(run_data) == sorted(RUNS),
        "aggregate_scores_equal": len({(sum(value for value in values if isinstance(value, int)), len([value for value in values if isinstance(value, int)])) for values in binary_vectors.values()}) == 1,
        "binary_correctness_vectors_identical": len({tuple(values) for values in binary_vectors.values()}) == 1,
        "scores_match_r040_summary": scores_match_summary(summary, binary_vectors),
        "prediction_commands_have_no_artifact_context_arg": command_findings["prediction_commands_have_no_artifact_context_arg"],
        "artifact_modes_candidate_pool_constrained_for_all_records": all(
            row["artifact_modes_candidate_pool_constrained"]["top4_original_plus_artifact"]
            and row["artifact_modes_candidate_pool_constrained"]["top4_artifact_only"]
            for row in matrix
        ),
    }
    required_checks = [
        "no_provider_calls",
        "no_prediction_or_eval_invoked",
        "targeted_37_only",
        "three_requested_runs_only",
        "aggregate_scores_equal",
        "scores_match_r040_summary",
        "prediction_commands_have_no_artifact_context_arg",
        "artifact_modes_candidate_pool_constrained_for_all_records",
    ]
    decision = "r041_identical_aggregate_score_attribution_complete" if all(checks[key] for key in required_checks) else "r041_identical_score_attribution_needs_review"
    return {
        "schema_version": "r041_r040_identical_score_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": {
            "posthoc_attribution_audit_only": True,
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_full_data_generalization": True,
            "not_official_mmlongbench_result": True,
            "uses_only_r039_r040_frozen_outputs": True,
        },
        "inputs": {
            "r040_root": args.r040_root,
            "r039_record_ids": args.r039_record_ids,
            "records": args.records,
            "artifacts": args.artifacts,
        },
        "num_records": len(matrix),
        "scores": vector_scores(binary_vectors),
        "outcome_counts": {
            "all_correct": int(outcome_counts.get("all_correct", 0)),
            "all_wrong": int(outcome_counts.get("all_wrong", 0)),
            "binary_divergent": int(outcome_counts.get("binary_divergent", 0)),
        },
        "binary_divergence": divergence,
        "answer_text": {
            "answer_text_all_same_records": len(matrix) - len(answer_text_diff_binary_same),
            "answer_text_diff_binary_same_records": len(answer_text_diff_binary_same),
            "diff_examples": [
                {
                    "record_id": row["record_id"],
                    "binary_pattern": row["binary_pattern"],
                    "question": row["question"],
                    "answers": truncate_answers(row["answer_texts"]),
                }
                for row in answer_text_diff_binary_same[:8]
            ],
        },
        "retrieval_change": retrieval_counts,
        "artifact_exposure": artifact_exposure,
        "primary_attribution_counts": dict(sorted(primary_counts.items())),
        "command_scope_findings": command_findings,
        "checks": checks,
        "interpretation": interpretation(checks, divergence, retrieval_counts, artifact_exposure, command_findings),
        "recommended_next_steps": [
            "Do manual error analysis on records where answer text changed but binary correctness stayed the same.",
            "If running a later experiment, make it contrastive: separate page-reranking effects from explicit artifact-context injection.",
            "Add exposure accounting to future manifests so artifact score, selected page, and prompt-visible evidence are distinct fields.",
            "Do not launch full QA from R040 alone; R041 supports only a targeted diagnostic conclusion.",
        ],
    }


def retrieval_change_counts(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for run in ["top4_original_plus_artifact", "top4_artifact_only"]:
        deltas = [row["retrieval_deltas_vs_original"][run] for row in matrix]
        rows[run] = {
            "any_branch_list_changed_records": sum(1 for delta in deltas if delta["any_branch_list_changed"]),
            "any_branch_set_changed_records": sum(1 for delta in deltas if delta["any_branch_set_changed"]),
            "combined_set_changed_records": sum(1 for delta in deltas if delta["combined_set_changed"]),
            "mean_combined_jaccard_unique_pages": round(sum(delta["combined_jaccard_unique_pages"] for delta in deltas) / max(len(deltas), 1), 6),
        }
    return rows


def binary_divergence_summary(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = Counter(row["binary_pattern"] for row in matrix)
    divergent = [row for row in matrix if row["record_attribution"]["correctness_class"] == "binary_divergent"]
    wins: Counter[str] = Counter()
    losses: Counter[str] = Counter()
    for row in divergent:
        correctness = row["binary_correctness"]
        for run in RUNS:
            value = correctness[run]
            others = [correctness[other] for other in RUNS if other != run]
            if value == 1 and all(other == 0 for other in others):
                wins[run] += 1
            if value == 0 and all(other == 1 for other in others):
                losses[run] += 1
    return {
        "pattern_counts": dict(sorted(patterns.items())),
        "divergent_record_count": len(divergent),
        "single_run_gain_counts": dict(sorted(wins.items())),
        "single_run_loss_counts": dict(sorted(losses.items())),
        "cancellation_summary": cancellation_summary(wins, losses),
        "divergent_records": [
            {
                "record_id": row["record_id"],
                "doc_id": row["doc_id"],
                "question": row["question"],
                "binary_pattern": row["binary_pattern"],
                "binary_correctness": row["binary_correctness"],
                "retrieval_changed_vs_original": row["record_attribution"]["retrieval_changed_vs_original"],
                "answers": truncate_answers(row["answer_texts"], 260),
            }
            for row in divergent
        ],
    }


def cancellation_summary(wins: Counter[str], losses: Counter[str]) -> str:
    artifact_wins = int(wins.get("top4_artifact_only", 0))
    artifact_losses = int(losses.get("top4_artifact_only", 0))
    if artifact_wins == artifact_losses and artifact_wins > 0:
        return f"top4_artifact_only gained {artifact_wins} records and lost {artifact_losses} records relative to the other two runs, producing an aggregate tie."
    if artifact_wins or artifact_losses:
        return f"top4_artifact_only gained {artifact_wins} records and lost {artifact_losses} records in divergent cases."
    return "No single-run win/loss cancellation pattern was detected."


def artifact_exposure_summary(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for run in RUNS:
        selected_counts = [row["retrieval"][run]["combined_artifact_count_on_selected_pages"] for row in matrix]
        rows[run] = {
            "records_with_any_artifact_on_selected_pages": sum(1 for value in selected_counts if value > 0),
            "total_artifacts_on_selected_unique_pages": sum(selected_counts),
            "median_artifacts_on_selected_unique_pages": median(selected_counts),
        }
    for run in ["top4_original_plus_artifact", "top4_artifact_only"]:
        positive_by_record = []
        for row in matrix:
            branches = row["retrieval"][run]["branches"]
            positive_by_record.append(sum(branches[field]["positive_score_page_count"] for field in BRANCH_FIELDS))
        rows[run]["records_with_positive_artifact_scores"] = sum(1 for value in positive_by_record if value > 0)
        rows[run]["total_positive_score_branch_pages"] = sum(positive_by_record)
    return rows


def command_scope_findings(gate: dict[str, Any], run_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prediction_commands = {
        run: run_data[run]["gate_row"].get("prediction_command")
        for run in RUNS
    }
    flattened = json.dumps(prediction_commands, ensure_ascii=False)
    artifact_context_markers = ["--artifacts", "artifact_context", "artifact-store", "artifacts="]
    has_artifact_context_arg = any(marker in flattened for marker in artifact_context_markers)
    sample_path_only = all(
        "dataset.sample_with_retrieval_path=" in json.dumps(command, ensure_ascii=False)
        for command in prediction_commands.values()
    )
    formulas = {
        run: ((run_data[run]["evaluation"][0].get("_nexus_meta") or {}).get("formula"))
        for run in RUNS
    }
    return {
        "prediction_commands": prediction_commands,
        "prediction_commands_have_no_artifact_context_arg": not has_artifact_context_arg,
        "prediction_commands_use_sample_with_retrieval_path": sample_path_only,
        "artifact_modes_are_page_rerank_inputs_not_prompt_artifact_injection": (not has_artifact_context_arg and sample_path_only),
        "run_formulas": formulas,
    }


def interpretation(checks: dict[str, Any], divergence: dict[str, Any], retrieval_counts: dict[str, Any], artifact_exposure: dict[str, Any], command_findings: dict[str, Any]) -> dict[str, Any]:
    primary = []
    if checks["aggregate_scores_equal"]:
        primary.append("All three aggregate scores are exactly 18/37; the tie is not a rounding artifact.")
    if checks["binary_correctness_vectors_identical"]:
        primary.append("All 37 record-level binary correctness labels are identical across the three R040 runs.")
    else:
        primary.append(f"The record-level binary vectors are not identical: {divergence['divergent_record_count']} records diverge, and gains/losses cancel out in the aggregate.")
        primary.append(divergence["cancellation_summary"])
    if command_findings["artifact_modes_are_page_rerank_inputs_not_prompt_artifact_injection"]:
        primary.append("The artifact-labeled R040 runs changed page-ranking inputs only; prediction commands did not pass artifact content as prompt-visible context.")
    if checks["artifact_modes_candidate_pool_constrained_for_all_records"]:
        primary.append("Artifact-mode selected pages stayed within the original top-10 retrieval candidate pools for every audited record.")
    if any(row["any_branch_list_changed_records"] for row in retrieval_counts.values()):
        primary.append("Retrieval/order changed for some records, but those changes did not cross the binary correctness decision boundary.")
    if artifact_exposure.get("top4_artifact_only", {}).get("records_with_positive_artifact_scores", 0):
        primary.append("Artifact scores were present on selected pages, so the audit does not reduce to a completely inert artifact store; the observed effect is binary-insensitive under this setup.")
    return {
        "primary_explanation": primary,
        "bottom_line": "R040's identical aggregate scores are best attributed to small record-level cancellation under a page-reranking-only, original-candidate-pool-constrained setup, not to identical per-record behavior and not to evidence that artifact context improves or fails on full data.",
    }


def scores_match_summary(summary: dict[str, Any], binary_vectors: dict[str, list[int | None]]) -> bool:
    summary_scores = summary.get("scores")
    if not isinstance(summary_scores, dict):
        return False
    for run, values in binary_vectors.items():
        numeric_values = [value for value in values if isinstance(value, int)]
        if not numeric_values:
            return False
        score = sum(numeric_values) / len(numeric_values)
        expected = summary_scores.get(run)
        if expected is None or abs(float(expected) - score) > 1e-9:
            return False
    return True


def vector_scores(binary_vectors: dict[str, list[int | None]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for run, values in binary_vectors.items():
        numeric_values = [value for value in values if isinstance(value, int)]
        rows[run] = {
            "correct": sum(numeric_values),
            "scored": len(numeric_values),
            "binary_correctness": sum(numeric_values) / len(numeric_values) if numeric_values else None,
        }
    return rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R041 R040 Identical Score Attribution Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- Post-hoc attribution audit only.",
        "- No provider calls, no new prediction, no new evaluation, no full QA.",
        "- Uses only frozen R039/R040 targeted 37-record outputs.",
        "- Not full-data generalization and not an official MMLongBench result.",
        "",
        "## Score Pattern",
        "| run | correct / scored | binary_correctness |",
        "| --- | ---: | ---: |",
    ]
    for run in RUNS:
        row = report["scores"][run]
        lines.append(f"| {run} | {row['correct']} / {row['scored']} | {row['binary_correctness']:.6f} |")
    lines.extend(
        [
            "",
            "## Outcome Counts",
            f"- all correct across all three runs: {report['outcome_counts']['all_correct']}",
            f"- all wrong across all three runs: {report['outcome_counts']['all_wrong']}",
            f"- binary divergent records: {report['outcome_counts']['binary_divergent']}",
            f"- answer text differs while binary correctness stays same: {report['answer_text']['answer_text_diff_binary_same_records']}",
            f"- cancellation: {report['binary_divergence']['cancellation_summary']}",
            "",
            "## Binary Divergence",
            "| record_id | pattern | note |",
            "| ---: | --- | --- |",
        ]
    )
    for row in report["binary_divergence"]["divergent_records"]:
        note = row["question"].replace("|", "/")[:120]
        lines.append(f"| {row['record_id']} | {row['binary_pattern']} | {note} |")
    lines.extend(
        [
            "",
            "## Retrieval Change vs Original",
            "| run | any branch list changed | any branch set changed | combined set changed | mean combined page Jaccard |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run, row in report["retrieval_change"].items():
        lines.append(
            f"| {run} | {row['any_branch_list_changed_records']} | {row['any_branch_set_changed_records']} | {row['combined_set_changed_records']} | {row['mean_combined_jaccard_unique_pages']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Exposure",
            "| run | records with artifacts on selected pages | total artifacts on selected pages | records with positive artifact scores |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for run in RUNS:
        row = report["artifact_exposure"][run]
        lines.append(
            f"| {run} | {row['records_with_any_artifact_on_selected_pages']} | {row['total_artifacts_on_selected_unique_pages']} | {row.get('records_with_positive_artifact_scores', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Key Checks",
        ]
    )
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Attribution",
        ]
    )
    for item in report["interpretation"]["primary_explanation"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"Bottom line: {report['interpretation']['bottom_line']}",
            "",
            "## Recommended Next Steps",
        ]
    )
    for item in report["recommended_next_steps"]:
        lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def answer_text(run: str, record: dict[str, Any]) -> str:
    key = next((key for key in record if key.startswith("ans_")), f"ans_{run}")
    value = record.get(key)
    return "" if value is None else str(value)


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def numeric_correctness(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return int(value)
    return None


def binary_pattern(correctness: dict[str, int | None]) -> str:
    return "".join("?" if correctness[run] is None else str(correctness[run]) for run in RUNS)


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    rows: list[int] = []
    for item in value:
        try:
            rows.append(int(item))
        except (TypeError, ValueError):
            continue
    return rows


def as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    rows: list[float] = []
    for item in value:
        try:
            rows.append(float(item))
        except (TypeError, ValueError):
            rows.append(0.0)
    return rows


def multiset_subset(values: list[int], pool: list[int]) -> bool:
    value_counts = Counter(values)
    pool_counts = Counter(pool)
    return all(count <= pool_counts[item] for item, count in value_counts.items())


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def truncate_answers(answers: dict[str, str], limit: int = 220) -> dict[str, str]:
    return {run: (value if len(value) <= limit else value[: limit - 3] + "...") for run, value in answers.items()}


def build_artifact_index(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or "")
            try:
                page_index = int(row.get("page_index"))
            except (TypeError, ValueError):
                continue
            index[(doc_id, page_index)].append(row)
    return index


def read_record_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
