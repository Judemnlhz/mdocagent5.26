#!/usr/bin/env python3
"""R075 small provider diagnostic for R074 evidence prompt integration.

This runner samples R074 help/risk buckets, calls a text provider on the R074
page+capsule+guard prompt plus retrieved page text, evaluates only those sampled
predictions, and compares the sampled binary outcomes against the existing
MDocAgent top-4 baseline results. It is a small diagnostic only: not full
MDocAgent multi-agent QA, not full MMLB, and not an official score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053
from mdocnexus.integration.evidence_skill_registry import estimate_tokens
from mdocnexus.integration.guarded_prompt import forbidden_public_fields, sha256

DEFAULT_R074_ROOT = "outputs/heldout/r074_mmlb_evidence_prompt_integration_gate"
DEFAULT_BASELINE_RESULTS = "results/MMLongBench/mmlb-MDocAgent-top4/2026-05-19-14-19_results.json"
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r075_mmlb_evidence_prompt_small_provider_diagnostic"
HELP_BUCKETS = ["baseline_wrong_capsule_supported_candidate", "baseline_wrong_guarded_or_page_routed_candidate"]
RISK_BUCKETS = ["baseline_correct_no_selected_artifact_risk"]
STABLE_BUCKETS = ["baseline_correct_stable_candidate"]
REFUSAL_TERMS = ["not answerable", "cannot be determined", "insufficient", "not provided", "not visible", "missing"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r074-root", default=DEFAULT_R074_ROOT)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--eval-model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--eval-max-tokens", type=int, default=64)
    parser.add_argument("--max-help", type=int, default=29)
    parser.add_argument("--max-risk", type=int, default=29)
    parser.add_argument("--max-stable", type=int, default=8)
    parser.add_argument("--include-record-ids", default="", help="Comma-separated record ids to force-include in the sampled diagnostic set.")
    parser.add_argument("--max-page-contexts", type=int, default=4)
    parser.add_argument("--max-page-chars", type=int, default=900)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="Write selection/report without provider calls unless --execute-provider is also set.")
    parser.add_argument("--execute-provider", action="store_true", help="Call provider and evaluator on the sampled diagnostic set.")
    parser.add_argument("--paired-original-baseline", action="store_true", help="Also run the same provider/evaluator on original-question prompts for paired diagnostic comparison.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.parallel_workers = min(max(1, int(args.parallel_workers)), 3)
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "will_call_provider": False,
            "output_root": str(output_root),
            "stage": "r075_mmlb_evidence_prompt_small_provider_diagnostic",
            "not_full_mdocagent_qa": True,
            "not_full_mmlb": True,
            "not_official_score": True,
            "parallel_workers_capped_at": 3,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    selected = select_cases(args)
    r053.write_jsonl(output_root / "r075_selected_cases.jsonl", selected)
    if args.execute_provider:
        predictions = run_provider_predictions(args, selected, output_root / "predictions" / "r075_predictions.jsonl")
        evaluations = run_evaluations(args, selected, predictions, output_root / "predictions" / "r075_evaluations.jsonl")
        if args.paired_original_baseline:
            original_predictions = run_original_provider_predictions(args, selected, output_root / "predictions" / "r075_original_predictions.jsonl")
            original_evaluations = run_evaluations(args, selected, original_predictions, output_root / "predictions" / "r075_original_evaluations.jsonl")
        else:
            original_predictions = []
            original_evaluations = []
    else:
        predictions = load_jsonl(output_root / "predictions" / "r075_predictions.jsonl")
        evaluations = load_jsonl(output_root / "predictions" / "r075_evaluations.jsonl")
        original_predictions = load_jsonl(output_root / "predictions" / "r075_original_predictions.jsonl")
        original_evaluations = load_jsonl(output_root / "predictions" / "r075_original_evaluations.jsonl")
    summary = build_summary(args, selected, predictions, evaluations, original_predictions, original_evaluations)
    gate = build_gate(args, summary)
    report = build_report(args, summary, gate)
    r053.write_json(output_root / "r075_small_provider_summary.json", summary)
    r053.write_json(output_root / "r075_small_provider_gate.json", gate)
    write_gate_markdown(output_root / "r075_small_provider_gate.md", gate)
    r053.write_json(output_root / "r075_small_provider_report.json", report)
    write_report_markdown(output_root / "r075_small_provider_report.md", report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "selected_cases": len(selected),
        "provider_predictions": len(predictions),
        "evaluations": len(evaluations),
        "changed_to_right": summary["outcome_counts"].get("changed_to_right", 0),
        "changed_to_wrong": summary["outcome_counts"].get("changed_to_wrong", 0),
        "sample_accuracy_not_official": summary.get("sample_accuracy_not_official"),
        "paired_original_sample_accuracy_not_official": summary.get("paired_original_sample_accuracy_not_official"),
        "paired_changed_to_right": summary.get("paired_outcome_counts", {}).get("changed_to_right", 0),
        "paired_changed_to_wrong": summary.get("paired_outcome_counts", {}).get("changed_to_wrong", 0),
        "report_md": str(output_root / "r075_small_provider_report.md"),
        "not_full_mdocagent_qa": True,
        "not_official_score": True,
    }, ensure_ascii=False, indent=2))


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    r074_root = Path(args.r074_root)
    audits = [json.loads(line) for line in (r074_root / "r074_mmlb_evidence_prompt_records.jsonl").read_text(encoding="utf-8").splitlines()]
    retrieval_rows = json.loads((r074_root / "r074_mmlb_evidence_layer_top4_retrieval.json").read_text(encoding="utf-8"))
    baseline_rows = r053.read_json(Path(args.baseline_results))
    retrieval_by_id = {int(row["record_index"]): row for row in retrieval_rows}
    baseline_by_id = {idx: row for idx, row in enumerate(baseline_rows)}
    selected: list[Mapping[str, Any]] = []
    selected.extend(take_bucket(audits, HELP_BUCKETS, args.max_help))
    selected.extend(take_bucket(audits, RISK_BUCKETS, args.max_risk))
    selected.extend(take_bucket(audits, STABLE_BUCKETS, args.max_stable))
    selected = force_include_records(selected, audits, parse_record_ids(args.include_record_ids))
    output = []
    for row in selected:
        record_id = int(row["record_id"])
        retrieval = retrieval_by_id[record_id]
        baseline = baseline_by_id[record_id]
        prompt = build_provider_prompt(args, row, retrieval)
        output.append({
            "schema_version": "r075_selected_case_v1",
            "record_id": record_id,
            "doc_id": row["doc_id"],
            "question": row["question"],
            "comparison_bucket": row["comparison_bucket"],
            "baseline_top4_correct": int(baseline.get("binary_correctness") or 0),
            "baseline_answer_key": "ans_mmlb-MDocAgent-top4",
            "baseline_prediction_preview": str(baseline.get("ans_mmlb-MDocAgent-top4") or "")[:260],
            "guard_decision": row.get("guard_decision"),
            "selected_artifact_count": row.get("selected_artifact_count"),
            "activated_skill_names": row.get("activated_skill_names"),
            "prompt_sha256": sha256(prompt),
            "prompt_tokens": estimate_tokens(prompt),
            "prompt_preview": prompt[:1200],
            "boundary": {
                "small_provider_diagnostic": True,
                "not_full_mdocagent_qa": True,
                "not_full_mmlb": True,
                "not_official_score": True,
            },
        })
    return output


def parse_record_ids(raw: str) -> list[int]:
    ids = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return sorted(set(ids))


def force_include_records(selected: list[Mapping[str, Any]], rows: list[Mapping[str, Any]], record_ids: list[int]) -> list[Mapping[str, Any]]:
    if not record_ids:
        return selected
    by_id = {int(row.get("record_id") or -1): row for row in rows}
    output = list(selected)
    seen = {int(row.get("record_id") or -1) for row in output}
    for record_id in record_ids:
        if record_id in seen:
            continue
        if record_id not in by_id:
            raise ValueError(f"include-record-id not found in audit rows: {record_id}")
        output.append(by_id[record_id])
        seen.add(record_id)
    output.sort(key=lambda row: int(row.get("record_id") or 0))
    return output


def take_bucket(rows: list[Mapping[str, Any]], buckets: list[str], limit: int) -> list[Mapping[str, Any]]:
    if limit <= 0:
        return []
    filtered = [row for row in rows if str(row.get("comparison_bucket")) in buckets]
    filtered.sort(key=lambda row: (buckets.index(str(row.get("comparison_bucket"))), int(row.get("record_id") or 0)))
    return filtered[:limit]


def build_provider_prompt(args: argparse.Namespace, audit_row: Mapping[str, Any], retrieval_row: Mapping[str, Any]) -> str:
    doc_id = str(retrieval_row.get("doc_id") or "")
    pages = unique_ints(list(retrieval_row.get("text-top-10-question") or []) + list(retrieval_row.get("image-top-10-question") or []))[: args.max_page_contexts]
    page_lines = []
    for page in pages:
        ctx = r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars)
        text = " ".join(str(ctx.get("text_preview") or "").split())[: args.max_page_chars]
        page_lines.append(f"Page {page} ({'present' if ctx.get('exists') else 'missing'}): {text}")
    lines = [
        "[R075 small provider diagnostic: R074 evidence prompt plus retrieved page text]",
        "Use only the retrieved page text and the evidence capsule/guard prompt below.",
        "This is a diagnostic prompt, not full MDocAgent multi-agent inference.",
        "Return a concise answer. If evidence is insufficient, say Not answerable and state the missing support.",
        "",
        "[Retrieved page text]",
        *page_lines,
        "",
        "[Evidence-layer prompt]",
        str(retrieval_row.get("_nexus_prompt_question") or "").strip(),
    ]
    return "\n".join(lines).strip() + "\n"


def run_provider_predictions(args: argparse.Namespace, selected: list[Mapping[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = [normalize_prediction_row(row) for row in load_jsonl(output_path)]
    existing = {(int(row["record_id"]), row["prompt_sha256"]): row for row in existing_rows}
    rows = list(existing.values())
    missing = [case for case in selected if (int(case["record_id"]), str(case["prompt_sha256"])) not in existing]
    if not missing:
        rows.sort(key=lambda row: int(row["record_id"]))
        write_jsonl(output_path, rows)
        return rows

    if int(args.parallel_workers) <= 1:
        for case in missing:
            rows.append(build_prediction_row(vars(args), dict(case)))
            rows.sort(key=lambda row: int(row["record_id"]))
            write_jsonl(output_path, rows)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.parallel_workers))) as executor:
            futures = [executor.submit(build_prediction_row, vars(args), dict(case)) for case in missing]
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: int(row["record_id"]))
                write_jsonl(output_path, rows)
    rows.sort(key=lambda row: int(row["record_id"]))
    write_jsonl(output_path, rows)
    return rows


def normalize_prediction_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "provider_call_succeeded" not in normalized:
        normalized["provider_call_succeeded"] = bool(str(normalized.get("prediction_text") or "").strip())
    normalized.setdefault("provider_error", None if normalized["provider_call_succeeded"] else "legacy_missing_provider_error")
    normalized.setdefault("refusal_like", refusal_like(str(normalized.get("prediction_text") or "")))
    normalized.setdefault("not_full_mdocagent_qa", True)
    normalized.setdefault("not_official_score", True)
    return normalized


def build_prediction_row(args_dict: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    args = argparse.Namespace(**dict(args_dict))
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)
    full_prompt = rebuild_prompt_from_case(args, case)
    provider_error = None
    try:
        prediction = call_chat(client, args.model, args.temperature, args.max_tokens, full_prompt, args.max_retries, args.request_timeout)
    except Exception as exc:  # noqa: BLE001
        prediction = ""
        provider_error = str(exc)[:500]
    return {
        "schema_version": "r075_prediction_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": int(case["record_id"]),
        "doc_id": case["doc_id"],
        "comparison_bucket": case["comparison_bucket"],
        "baseline_top4_correct": int(case["baseline_top4_correct"]),
        "model": args.model,
        "temperature": args.temperature,
        "prompt_sha256": sha256(full_prompt),
        "prompt_tokens": estimate_tokens(full_prompt),
        "prediction_text": prediction,
        "provider_call_succeeded": provider_error is None,
        "provider_error": provider_error,
        "refusal_like": refusal_like(prediction),
        "not_full_mdocagent_qa": True,
        "not_official_score": True,
    }

def run_original_provider_predictions(args: argparse.Namespace, selected: list[Mapping[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = [normalize_prediction_row(row) for row in load_jsonl(output_path)]
    existing = {(int(row["record_id"]), row["prompt_sha256"]): row for row in existing_rows}
    rows = list(existing.values())
    missing = []
    for case in selected:
        full_prompt = build_original_provider_prompt(args, case)
        if (int(case["record_id"]), sha256(full_prompt)) not in existing:
            missing.append(case)
    if not missing:
        rows.sort(key=lambda row: int(row["record_id"]))
        write_jsonl(output_path, rows)
        return rows

    if int(args.parallel_workers) <= 1:
        for case in missing:
            rows.append(build_original_prediction_row(vars(args), dict(case)))
            rows.sort(key=lambda row: int(row["record_id"]))
            write_jsonl(output_path, rows)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.parallel_workers))) as executor:
            futures = [executor.submit(build_original_prediction_row, vars(args), dict(case)) for case in missing]
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: int(row["record_id"]))
                write_jsonl(output_path, rows)
    rows.sort(key=lambda row: int(row["record_id"]))
    write_jsonl(output_path, rows)
    return rows


def build_original_prediction_row(args_dict: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    args = argparse.Namespace(**dict(args_dict))
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)
    full_prompt = build_original_provider_prompt(args, case)
    provider_error = None
    try:
        prediction = call_chat(client, args.model, args.temperature, args.max_tokens, full_prompt, args.max_retries, args.request_timeout)
    except Exception as exc:  # noqa: BLE001
        prediction = ""
        provider_error = str(exc)[:500]
    return {
        "schema_version": "r075_original_prediction_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": int(case["record_id"]),
        "doc_id": case["doc_id"],
        "comparison_bucket": case["comparison_bucket"],
        "baseline_top4_correct": int(case["baseline_top4_correct"]),
        "model": args.model,
        "temperature": args.temperature,
        "prompt_sha256": sha256(full_prompt),
        "prompt_tokens": estimate_tokens(full_prompt),
        "prediction_text": prediction,
        "provider_call_succeeded": provider_error is None,
        "provider_error": provider_error,
        "refusal_like": refusal_like(prediction),
        "paired_original_baseline": True,
        "not_full_mdocagent_qa": True,
        "not_official_score": True,
    }


def build_original_provider_prompt(args: argparse.Namespace, case: Mapping[str, Any]) -> str:
    r074_root = Path(args.r074_root)
    retrieval_rows = json.loads((r074_root / "r074_mmlb_evidence_layer_top4_retrieval.json").read_text(encoding="utf-8"))
    retrieval = next(row for row in retrieval_rows if int(row["record_index"]) == int(case["record_id"]))
    doc_id = str(retrieval.get("doc_id") or "")
    pages = unique_ints(list(retrieval.get("text-top-10-question") or []) + list(retrieval.get("image-top-10-question") or []))[: args.max_page_contexts]
    page_lines = []
    for page in pages:
        ctx = r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars)
        text = " ".join(str(ctx.get("text_preview") or "").split())[: args.max_page_chars]
        page_lines.append(f"Page {page} ({'present' if ctx.get('exists') else 'missing'}): {text}")
    lines = [
        "[R075 paired original-question diagnostic: retrieved page text plus original question]",
        "Use only the retrieved page text below to answer the question.",
        "This is a diagnostic prompt, not full MDocAgent multi-agent inference.",
        "Return a concise answer. If evidence is insufficient, say Not answerable and state the missing support.",
        "",
        "[Retrieved page text]",
        *page_lines,
        "",
        "[Original question]",
        str(case.get("question") or "").strip(),
    ]
    return "\n".join(lines).strip() + "\n"


def rebuild_prompt_from_case(args: argparse.Namespace, case: Mapping[str, Any]) -> str:
    r074_root = Path(args.r074_root)
    retrieval_rows = json.loads((r074_root / "r074_mmlb_evidence_layer_top4_retrieval.json").read_text(encoding="utf-8"))
    retrieval = next(row for row in retrieval_rows if int(row["record_index"]) == int(case["record_id"]))
    return build_provider_prompt(args, case, retrieval)


def run_evaluations(args: argparse.Namespace, selected: list[Mapping[str, Any]], predictions: list[Mapping[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    baseline_rows = r053.read_json(Path(args.baseline_results))
    by_prediction = {int(row["record_id"]): row for row in predictions}
    by_case = {int(row["record_id"]): row for row in selected}
    existing_rows = [normalize_evaluation_row(row) for row in load_jsonl(output_path)]
    existing = {int(row["record_id"]): row for row in existing_rows}
    rows = list(existing.values())
    missing_ids = [record_id for record_id in sorted(by_prediction) if record_id not in existing]
    if not missing_ids:
        rows.sort(key=lambda row: int(row["record_id"]))
        write_jsonl(output_path, rows)
        return rows

    if int(args.parallel_workers) <= 1:
        for record_id in missing_ids:
            rows.append(build_evaluation_row(vars(args), dict(by_prediction[record_id]), dict(by_case[record_id]), dict(baseline_rows[record_id])))
            rows.sort(key=lambda row: int(row["record_id"]))
            write_jsonl(output_path, rows)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.parallel_workers))) as executor:
            futures = [
                executor.submit(
                    build_evaluation_row,
                    vars(args),
                    dict(by_prediction[record_id]),
                    dict(by_case[record_id]),
                    dict(baseline_rows[record_id]),
                )
                for record_id in missing_ids
            ]
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: int(row["record_id"]))
                write_jsonl(output_path, rows)
    rows.sort(key=lambda row: int(row["record_id"]))
    write_jsonl(output_path, rows)
    return rows


def normalize_evaluation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "evaluation_call_succeeded" not in normalized:
        normalized["evaluation_call_succeeded"] = bool(str(normalized.get("raw_eval_text") or "").strip())
    normalized.setdefault("evaluation_error", None if normalized["evaluation_call_succeeded"] else "legacy_missing_evaluation_error")
    normalized.setdefault("not_full_mdocagent_qa", True)
    normalized.setdefault("not_official_score", True)
    return normalized


def build_evaluation_row(
    args_dict: Mapping[str, Any],
    pred: Mapping[str, Any],
    case: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
) -> dict[str, Any]:
    from openai import OpenAI

    args = argparse.Namespace(**dict(args_dict))
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)
    record_id = int(pred["record_id"])
    prompt = eval_prompt(str(baseline_row.get("question") or ""), str(pred.get("prediction_text") or ""), str(baseline_row.get("answer") or ""))
    eval_error = None
    if pred.get("provider_call_succeeded") is False:
        raw_eval = ""
        binary = 0
        eval_error = "provider_prediction_failed"
    else:
        try:
            raw_eval = call_chat(client, args.eval_model, 0.0, args.eval_max_tokens, prompt, args.max_retries, args.request_timeout)
            binary = parse_binary_correctness(raw_eval)
        except Exception as exc:  # noqa: BLE001
            raw_eval = ""
            binary = 0
            eval_error = str(exc)[:500]
    baseline_correct = int(case["baseline_top4_correct"])
    outcome = outcome_label(baseline_correct, binary)
    return {
        "schema_version": "r075_evaluation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": record_id,
        "doc_id": case["doc_id"],
        "comparison_bucket": case["comparison_bucket"],
        "baseline_top4_correct": baseline_correct,
        "r075_binary_correctness": binary,
        "outcome_vs_baseline": outcome,
        "eval_model": args.eval_model,
        "eval_prompt_sha256": sha256(prompt),
        "raw_eval_text": raw_eval[:500],
        "evaluation_call_succeeded": eval_error is None,
        "evaluation_error": eval_error,
        "not_full_mdocagent_qa": True,
        "not_official_score": True,
    }

def eval_prompt(question: str, answer: str, gt: str) -> str:
    return (
        f"Question: {question}\n"
        f"Predicted Answer: {answer}\n"
        f"Ground Truth Answer: {gt}\n\n"
        "Please evaluate if the predicted answer is correct compared to the ground truth.\n"
        "Return only JSON parsable text like {\"binary_correctness\": 1} or {\"binary_correctness\": 0}.\n"
    )


def call_chat(
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
    max_retries: int,
    request_timeout: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            response = call_chat_once_with_alarm(client, model, temperature, max_tokens, prompt, request_timeout)
            return str(response.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"Provider call failed after {max_retries} retries: {last_error}")


def call_chat_once_with_alarm(
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
    request_timeout: float,
) -> Any:
    def _raise_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise TimeoutError(f"provider request exceeded {request_timeout:.1f}s")

    timeout = max(1.0, float(request_timeout))
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def parse_binary_correctness(text: str) -> int:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return 1 if int(data.get("binary_correctness", 0)) == 1 else 0
    except Exception:
        return 0


def outcome_label(baseline_correct: int, ours_correct: int) -> str:
    if baseline_correct == 0 and ours_correct == 1:
        return "changed_to_right"
    if baseline_correct == 1 and ours_correct == 0:
        return "changed_to_wrong"
    if baseline_correct == 1 and ours_correct == 1:
        return "kept_right"
    return "kept_wrong"


def build_summary(
    args: argparse.Namespace,
    selected: list[Mapping[str, Any]],
    predictions: list[Mapping[str, Any]],
    evaluations: list[Mapping[str, Any]],
    original_predictions: list[Mapping[str, Any]] | None = None,
    original_evaluations: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    eval_by_id = {int(row["record_id"]): row for row in evaluations}
    selected_ids = {int(row["record_id"]) for row in selected}
    evaluated_ids = selected_ids & set(eval_by_id)
    outcome_counts = Counter(str(eval_by_id[record_id].get("outcome_vs_baseline")) for record_id in evaluated_ids)
    bucket_counts = Counter(str(row.get("comparison_bucket")) for row in selected)
    eval_bucket_counts = Counter(str(eval_by_id[record_id].get("comparison_bucket")) for record_id in evaluated_ids)
    correct_values = [int(eval_by_id[record_id].get("r075_binary_correctness") or 0) for record_id in evaluated_ids]
    baseline_values = [int(next(row for row in selected if int(row["record_id"]) == record_id)["baseline_top4_correct"]) for record_id in evaluated_ids]
    provider_failures = sum(1 for row in predictions if row.get("provider_call_succeeded") is False)
    evaluation_failures = sum(1 for row in evaluations if row.get("evaluation_call_succeeded") is False)
    provider_failure_rate = round(provider_failures / len(predictions), 6) if predictions else None
    evaluation_failure_rate = round(evaluation_failures / len(evaluations), 6) if evaluations else None
    original_predictions = original_predictions or []
    original_evaluations = original_evaluations or []
    original_eval_by_id = {int(row["record_id"]): row for row in original_evaluations}
    paired_ids = evaluated_ids & set(original_eval_by_id)
    paired_outcome_counts = Counter(
        outcome_label(
            int(original_eval_by_id[record_id].get("r075_binary_correctness") or 0),
            int(eval_by_id[record_id].get("r075_binary_correctness") or 0),
        )
        for record_id in paired_ids
    )
    original_correct_values = [int(original_eval_by_id[record_id].get("r075_binary_correctness") or 0) for record_id in paired_ids]
    original_provider_failures = sum(1 for row in original_predictions if row.get("provider_call_succeeded") is False)
    original_evaluation_failures = sum(1 for row in original_evaluations if row.get("evaluation_call_succeeded") is False)
    return {
        "schema_version": "r075_small_provider_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_cases": len(selected),
        "provider_predictions": len(predictions),
        "evaluations": len(evaluations),
        "all_selected_evaluated": len(evaluated_ids) == len(selected),
        "provider_failures": provider_failures,
        "provider_failure_rate": provider_failure_rate,
        "evaluation_failures": evaluation_failures,
        "evaluation_failure_rate": evaluation_failure_rate,
        "selection_bucket_counts": dict(sorted(bucket_counts.items())),
        "evaluated_bucket_counts": dict(sorted(eval_bucket_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "sample_accuracy_not_official": round(sum(correct_values) / len(correct_values), 6) if correct_values else None,
        "baseline_sample_accuracy_reference_not_official": round(sum(baseline_values) / len(baseline_values), 6) if baseline_values else None,
        "changed_to_right_minus_wrong": outcome_counts.get("changed_to_right", 0) - outcome_counts.get("changed_to_wrong", 0),
        "paired_original_predictions": len(original_predictions),
        "paired_original_evaluations": len(original_evaluations),
        "paired_original_provider_failures": original_provider_failures,
        "paired_original_evaluation_failures": original_evaluation_failures,
        "paired_original_sample_accuracy_not_official": round(sum(original_correct_values) / len(original_correct_values), 6) if original_correct_values else None,
        "paired_outcome_counts": dict(sorted(paired_outcome_counts.items())),
        "paired_changed_to_right_minus_wrong": paired_outcome_counts.get("changed_to_right", 0) - paired_outcome_counts.get("changed_to_wrong", 0),
        "model": args.model,
        "eval_model": args.eval_model,
        "parallel_workers": int(args.parallel_workers),
        "request_timeout": float(args.request_timeout),
        "boundary": {
            "small_provider_diagnostic": True,
            "not_full_mdocagent_qa": True,
            "not_full_mmlb": True,
            "not_official_score": True,
            "uses_existing_baseline_results_for_comparison": True,
        },
        "forbidden_gold_fields_in_selected_cases": forbidden_public_fields(selected),
        "evaluation_outputs_contain_binary_correctness_not_gold_answers": True,
    }


def build_gate(args: argparse.Namespace, summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "provider_execution_requested": bool(args.execute_provider),
        "selected_cases_positive": summary.get("selected_cases", 0) > 0,
        "all_selected_have_predictions": summary.get("provider_predictions", 0) == summary.get("selected_cases", 0),
        "all_selected_evaluated": summary.get("all_selected_evaluated") is True,
        "not_full_mdocagent_qa": True,
        "not_full_mmlb": True,
        "not_official_score": True,
        "selected_cases_do_not_expose_gold_fields": not summary.get("forbidden_gold_fields_in_selected_cases"),
    }
    hard_failures = [key for key, value in checks.items() if not value]
    decision = "r075_small_provider_diagnostic_complete" if not hard_failures else "r075_small_provider_diagnostic_incomplete"
    return {
        "schema_version": "r075_small_provider_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "not_full_mdocagent_qa": True,
        "not_official_score": True,
    }


def build_report(args: argparse.Namespace, summary: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "r075_small_provider_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": gate["decision"],
        "scope": summary["boundary"],
        "summary": summary,
        "gate": gate,
        "recommended_next": recommendations(summary),
    }


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    delta = int(summary.get("changed_to_right_minus_wrong") or 0)
    provider_failures = int(summary.get("provider_failures") or 0)
    if summary.get("paired_outcome_counts"):
        paired_delta = int(summary.get("paired_changed_to_right_minus_wrong") or 0)
        paired_hurts = int(summary.get("paired_outcome_counts", {}).get("changed_to_wrong", 0) or 0)
        if paired_delta >= 0 and paired_hurts == 0:
            if paired_delta > 0:
                return [
                    f"Paired original-vs-evidence diagnostic is positive ({paired_delta}) with no paired hurts; treat this as a bounded positive signal.",
                    "Stop general guard repair and proceed to bounded MDocAgent QA plus claim-scope audit, not another open-ended prompt repair.",
                ]
            return [
                "Paired original-vs-evidence diagnostic is flat (0) with no paired hurts; this satisfies the bounded stop rule.",
                "Stop general guard repair and proceed to bounded MDocAgent QA; frame any QA result as bounded/partial and emphasize token efficiency, evidence auditability, and guarded citation faithfulness.",
            ]
        return [
            f"Paired original-vs-evidence diagnostic still has paired hurts ({paired_hurts}) or negative delta ({paired_delta}); do not launch full MMLB QA from this prompt.",
            "Repair only the identified systematic hurt before another bounded paired diagnostic.",
        ]
    if provider_failures:
        return [
            f"Provider stability is a blocker: {provider_failures} sampled rows timed out or failed and were conservatively scored as incorrect.",
            f"Observed help-hurt delta is negative ({delta}); do not launch full MMLB QA from the current R074 prompt.",
            "First rerun a smaller balanced diagnostic with a stable provider or longer timeout, then reduce baseline-correct no-selected-artifact prompt intervention strength.",
        ]
    if summary.get("all_selected_evaluated") and delta > 0:
        return [
            f"Small diagnostic help-hurt delta is positive ({delta}); next run can expand to a larger bounded MDocAgent pipeline diagnostic, not full MMLB yet.",
            "Inspect changed-to-wrong examples before full QA, especially baseline-correct no-selected-artifact risk cases.",
        ]
    if summary.get("all_selected_evaluated"):
        return [
            f"Small diagnostic help-hurt delta is not positive ({delta}); revise R074 prompt wording before any full MMLB QA.",
            "Reduce guard strength for baseline-correct no-selected-artifact cases and rerun this diagnostic.",
        ]
    return ["Complete provider predictions/evaluations before deciding whether to expand or revise prompts."]


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = ["# R075 Small Provider Gate", "", f"Decision: `{gate['decision']}`", f"Gate passed: {gate['gate_passed']}", "", "## Checks"]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# R075 MMLB Evidence Prompt Small Provider Diagnostic",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- Small text-provider diagnostic only; not full MDocAgent multi-agent QA.",
        "- Not full MMLB and not an official score.",
        "- Uses existing top-4 baseline correctness only for sampled help/hurt comparison.",
        "",
        "## Summary",
        f"- selected cases: {summary['selected_cases']}",
        f"- provider predictions: {summary['provider_predictions']}",
        f"- evaluations: {summary['evaluations']}",
        f"- provider failures: {summary.get('provider_failures')} ({summary.get('provider_failure_rate')})",
        f"- evaluation failures: {summary.get('evaluation_failures')} ({summary.get('evaluation_failure_rate')})",
        f"- sample accuracy not official: {summary.get('sample_accuracy_not_official')}",
        f"- baseline sample accuracy reference not official: {summary.get('baseline_sample_accuracy_reference_not_official')}",
        f"- changed_to_right_minus_wrong: {summary.get('changed_to_right_minus_wrong')}",
        f"- paired original predictions/evaluations: {summary.get('paired_original_predictions')}/{summary.get('paired_original_evaluations')}",
        f"- paired original sample accuracy not official: {summary.get('paired_original_sample_accuracy_not_official')}",
        f"- paired changed_to_right_minus_wrong: {summary.get('paired_changed_to_right_minus_wrong')}",
        "",
        "## Outcomes",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("outcome_counts", {}).items())
    if summary.get("paired_outcome_counts"):
        lines.extend(["", "## Paired Original vs Evidence Outcomes"])
        lines.extend(f"- `{key}`: {value}" for key, value in summary.get("paired_outcome_counts", {}).items())
    lines.extend(["", "## Selection Buckets"])
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("selection_bucket_counts", {}).items())
    lines.extend(["", "## Recommended Next"])
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


def unique_ints(values: list[Any]) -> list[int]:
    output = []
    seen = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def refusal_like(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in REFUSAL_TERMS)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()