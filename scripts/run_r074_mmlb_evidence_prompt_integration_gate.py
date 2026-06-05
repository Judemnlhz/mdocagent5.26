#!/usr/bin/env python3
"""R074 no-provider baseline-aligned evidence prompt integration gate.

R074 turns the R071-R073 evidence layer into a default-off, MDocAgent-compatible
prompt variant for MMLB top-4 comparison. It preserves the original question for
evaluation, writes the evidence-layer prompt to ``_nexus_prompt_question``, and
prepares runnable predict/eval commands that must be launched explicitly later.
It does not call providers, generate predictions, run QA, evaluate, or report an
official score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053
import run_r071_evidence_skill_graph_registry_gate as r071
from mdocnexus.integration.evidence_skill_registry import estimate_tokens, render_evidence_capsule
from mdocnexus.integration.guarded_prompt import build_question_profile, forbidden_public_fields, select_guarded_artifacts
from mdocnexus.integration.mdocagent_adapter import (
    assert_no_forbidden_public_fields,
    canonical_json_hash,
    load_mdocagent_retrieval_records,
)

DEFAULT_BASELINE_RESULTS = "results/MMLongBench/mmlb-MDocAgent-top4/2026-05-19-14-19_results.json"
DEFAULT_RECORDS = r053.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r053.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r074_mmlb_evidence_prompt_integration_gate"
DEFAULT_RUN_NAME = "mmlb-MDocAgent-r074-evidence-layer-top4"
STRICT_GUARDS = {"exact_code_absence_guard", "operand_completeness_guard"}
PROMPT_QUESTION_KEY = "_nexus_prompt_question"
MAX_EXAMPLES = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0, help="Optional debug cap; 0 means all records.")
    parser.add_argument("--max-page-chars", type=int, default=1600)
    parser.add_argument("--max-artifact-chars", type=int, default=280)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--capsule-units", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "baseline_results": args.baseline_results,
            "run_name": args.run_name,
            "prompt_question_key": PROMPT_QUESTION_KEY,
            "no_provider_calls": True,
            "no_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit)
    report = build_report(args, audit, gate)
    retrieval_path = output_root / "r074_mmlb_evidence_layer_top4_retrieval.json"
    r053.write_json(retrieval_path, audit["retrieval_records"])
    r053.write_jsonl(output_root / "r074_mmlb_evidence_prompt_records.jsonl", audit["audit_records"])
    r053.write_json(output_root / "r074_mmlb_evidence_prompt_summary.json", audit["summary"])
    r053.write_json(output_root / "r074_mmlb_evidence_prompt_gate.json", gate)
    write_gate_markdown(output_root / "r074_mmlb_evidence_prompt_gate.md", gate)
    r053.write_json(output_root / "r074_mmlb_evidence_prompt_report.json", report)
    write_report_markdown(output_root / "r074_mmlb_evidence_prompt_report.md", report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "records_scanned": audit["summary"]["records_scanned"],
        "baseline_top4_score_reference": audit["summary"]["baseline_top4_score_reference"],
        "mean_prompt_question_token_ratio": audit["summary"]["token_stats"]["prompt_question_vs_original_question_ratio"]["mean"],
        "retrieval_path": str(retrieval_path),
        "predict_command": report["recommended_commands"]["predict"],
        "report_md": str(output_root / "r074_mmlb_evidence_prompt_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
        "not_official_score": True,
    }, ensure_ascii=False, indent=2))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    raw_baseline_rows = r053.read_json(Path(args.baseline_results))
    raw_source_rows = r053.read_json(Path(args.records))
    public_records = load_mdocagent_retrieval_records(args.records)
    baseline_public = load_mdocagent_retrieval_records(args.baseline_results)
    if args.max_records and args.max_records > 0:
        raw_baseline_rows = raw_baseline_rows[: args.max_records]
        raw_source_rows = raw_source_rows[: args.max_records]
        public_records = public_records[: args.max_records]
        baseline_public = baseline_public[: args.max_records]
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    rows = []
    retrieval_records = []
    for record_id, (source, baseline_raw, baseline_clean) in enumerate(zip(raw_source_rows, raw_baseline_rows, baseline_public)):
        audit_record, retrieval_record = build_record(record_id, source, baseline_raw, baseline_clean, artifacts_by_page, args)
        rows.append(audit_record)
        retrieval_records.append(retrieval_record)
    summary = summarize(args, rows, retrieval_records, raw_baseline_rows)
    public_payload = {"summary": summary, "retrieval_records": retrieval_records, "audit_records": rows}
    summary["forbidden_gold_fields_present"] = forbidden_public_fields(public_payload)
    assert_no_forbidden_public_fields(retrieval_records)
    return {
        "summary": summary,
        "audit_records": rows,
        "retrieval_records": retrieval_records,
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_record(record_id: int, source: Mapping[str, Any], baseline_raw: Mapping[str, Any], baseline_clean: Mapping[str, Any], artifacts_by_page: Mapping[tuple[str, int], list[dict[str, Any]]], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    doc_id = str(baseline_clean.get("doc_id") or "")
    question = str(baseline_clean.get("question") or "")
    profile = build_question_profile(question)
    pages = r071.retrieval_pages(baseline_clean, args.top_k)
    page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in pages]
    current_artifacts = [artifact for page in pages for artifact in artifacts_by_page.get((doc_id, page), [])]
    scored = r071.score_artifacts(current_artifacts, question, profile, pages, args)
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)
    capsule = render_evidence_capsule(question, profile, selection, scored, max_units=args.capsule_units, include_guard_trace=True, max_chars=args.max_artifact_chars)
    prompt_mode = prompt_mode_for_selection(selection)
    prompt_question = render_prompt_question(question, capsule, prompt_mode)
    prompt_tokens = estimate_tokens(prompt_question)
    original_tokens = estimate_tokens(question)
    baseline_outcome = "correct" if int(baseline_raw.get("binary_correctness") or 0) == 1 else "wrong"
    bucket = comparison_bucket(baseline_outcome, selection, capsule)
    retrieval_record = build_public_retrieval_record(record_id, baseline_clean, prompt_question, selection, capsule, prompt_tokens, original_tokens, args)
    audit_record = {
        "schema_version": "r074_mmlb_evidence_prompt_record_v1",
        "record_id": record_id,
        "doc_id": doc_id,
        "question": question,
        "baseline_top4_outcome": baseline_outcome,
        "comparison_bucket": bucket,
        "retrieval_pages": pages,
        "page_text_exists_count": sum(1 for ctx in page_contexts if ctx.get("exists")),
        "candidate_artifact_count": len(scored),
        "selected_artifact_count": len(selection.get("selected_artifacts") or []),
        "guard_decision": selection.get("guard_decision"),
        "answer_policy": selection.get("answer_policy"),
        "activated_skill_names": capsule.get("activated_skill_names"),
        "missing_requirements": capsule.get("missing_requirements"),
        "prompt_mode": prompt_mode,
        "prompt_question_tokens": prompt_tokens,
        "original_question_tokens": original_tokens,
        "prompt_question_token_ratio": ratio(prompt_tokens, original_tokens),
        "prompt_question_sha256": canonical_json_hash({"prompt": prompt_question}),
        "retrieval_record_sha256": canonical_json_hash(retrieval_record),
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
    }
    return audit_record, retrieval_record


def build_public_retrieval_record(record_id: int, baseline_clean: Mapping[str, Any], prompt_question: str, selection: Mapping[str, Any], capsule: Mapping[str, Any], prompt_tokens: int, original_tokens: int, args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {
        "record_index": record_id,
        "doc_id": str(baseline_clean.get("doc_id") or ""),
        "question": str(baseline_clean.get("question") or ""),
        PROMPT_QUESTION_KEY: prompt_question,
    }
    for key, value in baseline_clean.items():
        if key in {"record_index", "doc_id", "question"}:
            continue
        if key.startswith(("text-top-", "image-top-", "mix-top-")):
            output[key] = value
    output["_nexus_meta"] = {
        "schema_version": "r074_evidence_prompt_retrieval_meta_v1",
        "mode": prompt_mode_for_selection(selection),
        "top_k": args.top_k,
        "prompt_question_key": PROMPT_QUESTION_KEY,
        "original_question_preserved_for_eval": True,
        "same_retrieval_budget_as_mdocagent_top4": True,
        "guard_decision": selection.get("guard_decision"),
        "answer_policy": selection.get("answer_policy"),
        "selected_artifact_count": len(selection.get("selected_artifacts") or []),
        "activated_skill_names": capsule.get("activated_skill_names"),
        "missing_requirement_count": len(capsule.get("missing_requirements") or []),
        "prompt_question_tokens": prompt_tokens,
        "original_question_tokens": original_tokens,
        "no_gold_fields_used": True,
        "no_provider_calls_in_r074": True,
        "not_prediction_or_eval": True,
    }
    return output


def prompt_mode_for_selection(selection: Mapping[str, Any]) -> str:
    selected_count = len(selection.get("selected_artifacts") or [])
    guard = str(selection.get("guard_decision") or "")
    if selected_count == 0 and guard not in STRICT_GUARDS:
        return "original_question_passthrough_no_artifact"
    return "page_plus_capsule_plus_guard_prompt_question"


def render_prompt_question(question: str, capsule: Mapping[str, Any], prompt_mode: str) -> str:
    if prompt_mode == "original_question_passthrough_no_artifact":
        return question.strip() + "\n"
    lines = [
        "[MDocAgent Evidence Layer - page plus capsule plus guard]",
        "Use the normal MDocAgent retrieved page text and images as the primary evidence.",
        "Use the evidence capsule below as a compact checklist of selected evidence, missing requirements, and guard policy.",
        "If capsule evidence conflicts with visible page evidence, rely on visible page evidence and mention the missing support.",
        "Do not infer an exact code, numeric value, or citation from partial evidence when the guard says it is missing.",
        "Return the final answer in the dataset's expected concise format.",
        "",
        "[Original Question]",
        question,
        "",
        str(capsule.get("text") or "").strip(),
    ]
    return "\n".join(lines).strip() + "\n"


def comparison_bucket(baseline_outcome: str, selection: Mapping[str, Any], capsule: Mapping[str, Any]) -> str:
    selected_count = len(selection.get("selected_artifacts") or [])
    missing_count = len(capsule.get("missing_requirements") or [])
    guard = str(selection.get("guard_decision") or "")
    if baseline_outcome == "wrong" and selected_count > 0 and missing_count == 0:
        return "baseline_wrong_capsule_supported_candidate"
    if baseline_outcome == "wrong" and selected_count > 0:
        return "baseline_wrong_capsule_partial_candidate"
    if baseline_outcome == "wrong" and guard in {"exact_code_absence_guard", "operand_completeness_guard", "artifact_dimension_support_guard"}:
        return "baseline_wrong_guarded_or_page_routed_candidate"
    if baseline_outcome == "correct" and selected_count == 0:
        return "baseline_correct_no_selected_artifact_risk"
    if baseline_outcome == "correct" and missing_count > 0:
        return "baseline_correct_missing_requirement_risk"
    return f"baseline_{baseline_outcome}_stable_candidate"


def summarize(args: argparse.Namespace, rows: list[Mapping[str, Any]], retrieval_records: list[Mapping[str, Any]], baseline_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_values = [int(row.get("binary_correctness") or 0) for row in baseline_rows]
    guard_counts = Counter(str(row.get("guard_decision") or "") for row in rows)
    bucket_counts = Counter(str(row.get("comparison_bucket") or "") for row in rows)
    skill_counts = Counter(skill for row in rows for skill in row.get("activated_skill_names") or [])
    selected_positive = sum(1 for row in rows if int(row.get("selected_artifact_count") or 0) > 0)
    prompt_ratios = [float(row.get("prompt_question_token_ratio") or 0.0) for row in rows]
    prompt_tokens = [int(row.get("prompt_question_tokens") or 0) for row in rows]
    original_tokens = [int(row.get("original_question_tokens") or 0) for row in rows]
    prompt_mode_counts = Counter(str(row.get("prompt_mode") or "") for row in rows)
    return {
        "schema_version": "r074_mmlb_evidence_prompt_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records_scanned": len(rows),
        "baseline_results": args.baseline_results,
        "baseline_top4_score_reference": round(statistics.fmean(baseline_values), 6) if baseline_values else 0.0,
        "baseline_correct_count": sum(baseline_values),
        "baseline_wrong_count": len(baseline_values) - sum(baseline_values),
        "prompt_question_key": PROMPT_QUESTION_KEY,
        "run_name": args.run_name,
        "top_k": args.top_k,
        "retrieval_records_sha256": canonical_json_hash(retrieval_records),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "comparison_bucket_counts": dict(sorted(bucket_counts.items())),
        "activated_skill_counts": dict(sorted(skill_counts.items())),
        "selected_artifact_record_count": selected_positive,
        "selected_artifact_record_rate": ratio(selected_positive, len(rows)),
        "prompt_mode_counts": dict(sorted(prompt_mode_counts.items())),
        "token_stats": {
            "original_question": number_stats(original_tokens),
            "prompt_question": number_stats(prompt_tokens),
            "prompt_question_vs_original_question_ratio": number_stats(prompt_ratios),
        },
        "examples": compact_examples(rows),
        "boundary": {
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "original_question_preserved_for_eval": True,
            "prompt_augmentation_default_off": True,
            "no_artifact_passthrough_enabled": True,
            "same_top4_page_budget_as_baseline": True,
            "does_not_use_answer_or_evidence_pages_in_prompt_input": True,
        },
    }


def build_gate(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    retrieval_records = audit["retrieval_records"]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "not_official_score": True,
        "records_scanned_positive": summary.get("records_scanned", 0) > 0,
        "baseline_top4_reference_matches_known_result": abs(float(summary.get("baseline_top4_score_reference") or 0.0) - 0.493) <= 0.001,
        "original_question_preserved_for_eval": all(row.get("question") and row.get(PROMPT_QUESTION_KEY) for row in retrieval_records),
        "prompt_question_key_present_all_records": all(PROMPT_QUESTION_KEY in row for row in retrieval_records),
        "text_retrieval_branch_preserved": all(isinstance(row.get("text-top-10-question"), list) and len(row.get("text-top-10-question") or []) > 0 for row in retrieval_records),
        "image_retrieval_branch_preserved": all(isinstance(row.get("image-top-10-question"), list) and len(row.get("image-top-10-question") or []) > 0 for row in retrieval_records),
        "no_gold_fields_in_retrieval_output": not forbidden_public_fields(retrieval_records),
        "no_gold_fields_in_public_outputs": not summary.get("forbidden_gold_fields_present"),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "candidate_buckets_available": bool(summary.get("comparison_bucket_counts")),
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r074_mmlb_evidence_prompt_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r074_mmlb_evidence_prompt_integration_ready_for_provider_diagnostic" if not hard_failures else "r074_mmlb_evidence_prompt_integration_invalid",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(args: argparse.Namespace, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    retrieval_path = Path(args.output_root) / "r074_mmlb_evidence_layer_top4_retrieval.json"
    predict = [
        "python3",
        "scripts/predict.py",
        "--config-name",
        "mmlb",
        f"run-name={args.run_name}",
        f"dataset.top_k={args.top_k}",
        f"dataset.sample_with_retrieval_path={retrieval_path}",
        f"+dataset.prompt_question_key={PROMPT_QUESTION_KEY}",
    ]
    evaluate = ["python3", "scripts/eval.py", "--config-name", "mmlb", f"run-name={args.run_name}"]
    return {
        "schema_version": "r074_mmlb_evidence_prompt_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": gate["decision"],
        "scope": summary["boundary"],
        "summary": summary,
        "gate": gate,
        "retrieval_output_path": str(retrieval_path),
        "recommended_commands": {
            "predict": predict,
            "evaluate_after_prediction": evaluate,
        },
        "recommended_next": recommendations(summary),
    }


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    buckets = summary.get("comparison_bucket_counts") or {}
    help_candidates = buckets.get("baseline_wrong_capsule_supported_candidate", 0) + buckets.get("baseline_wrong_capsule_partial_candidate", 0) + buckets.get("baseline_wrong_guarded_or_page_routed_candidate", 0)
    risk_candidates = buckets.get("baseline_correct_no_selected_artifact_risk", 0) + buckets.get("baseline_correct_missing_requirement_risk", 0)
    return [
        f"Run a small provider diagnostic before full QA, sampling baseline-wrong help candidates ({help_candidates}) and baseline-correct risk candidates ({risk_candidates}).",
        "Use the generated retrieval JSON with dataset.prompt_question_key set to _nexus_prompt_question; do not overwrite the original question field.",
        "If the diagnostic shows help <= hurt, revise prompt format or use a page+capsule hybrid with weaker guard wording before full MMLB.",
    ]


def number_stats(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    nums = [float(value) for value in values]
    return {
        "mean": round(statistics.fmean(nums), 6),
        "median": round(statistics.median(nums), 6),
        "min": round(min(nums), 6),
        "max": round(max(nums), 6),
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0:
        return 1.0 if float(numerator) > 0 else 0.0
    return round(float(numerator) / float(denominator), 6)


def compact_examples(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if len(selected) >= MAX_EXAMPLES:
            break
        if str(row.get("comparison_bucket") or "").startswith("baseline_wrong"):
            selected.append(row)
    for row in rows:
        if len(selected) >= MAX_EXAMPLES:
            break
        if row not in selected:
            selected.append(row)
    return [{
        "record_id": row.get("record_id"),
        "doc_id": row.get("doc_id"),
        "question": str(row.get("question") or "")[:180],
        "baseline_top4_outcome": row.get("baseline_top4_outcome"),
        "comparison_bucket": row.get("comparison_bucket"),
        "guard_decision": row.get("guard_decision"),
        "selected_artifact_count": row.get("selected_artifact_count"),
        "prompt_mode": row.get("prompt_mode"),
        "activated_skill_names": row.get("activated_skill_names"),
    } for row in selected]


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = [
        "# R074 MMLB Evidence Prompt Integration Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Original question is preserved; evidence-layer prompt uses `_nexus_prompt_question` only when explicitly configured.",
        "- Not an official score.",
        "",
        "## Checks",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# R074 MMLB Baseline-Aligned Evidence Prompt Integration",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Builds a default-off MDocAgent prompt variant for later explicit provider runs.",
        "- Keeps original `question` for evaluation and stores the evidence prompt in `_nexus_prompt_question`.",
        "",
        "## Summary",
        f"- records scanned: {summary['records_scanned']}",
        f"- baseline top-4 score reference: {summary['baseline_top4_score_reference']}",
        f"- selected artifact record rate: {summary['selected_artifact_record_rate']}",
        f"- mean prompt/original question token ratio: {summary['token_stats']['prompt_question_vs_original_question_ratio']['mean']}",
        f"- prompt modes: {summary.get('prompt_mode_counts')}",
        "",
        "## Comparison Buckets",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("comparison_bucket_counts", {}).items())
    lines.extend(["", "## Recommended Commands", "```bash", " ".join(report["recommended_commands"]["predict"]), " ".join(report["recommended_commands"]["evaluate_after_prediction"]), "```", "", "## Recommended Next"])
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()