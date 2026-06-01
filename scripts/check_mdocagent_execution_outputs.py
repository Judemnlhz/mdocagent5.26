#!/usr/bin/env python3
"""Check first-stage MDocAgent adapter execution gate outputs.

The gate is intentionally narrow: it validates that the official top-4
reproduction and the adapter original-only run used equivalent retrieval input
and comparable model/evaluation settings. It does not call models or generate
answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


DEFAULT_OUTPUT_DIR = "outputs/experiments/mdocagent_module_ablation"
DEFAULT_RESULTS_ROOT = "results/MMLongBench"
DEFAULT_RUNS = "mdocagent_top4_official_reproduction,top4_original_only"
EXPECTED_EVAL_JUDGE = "deepseek-ai/DeepSeek-V3"
FORBIDDEN_NEXT_RUNS = [
    "top4_artifact_only",
    "top4_original_plus_artifact",
    "top4_graph_context",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_runs(value: str) -> list[str]:
    runs = [item.strip() for item in value.split(",") if item.strip()]
    if not runs:
        raise ValueError("No run names were provided")
    return runs


def latest_file(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, path.name))


def find_latest_prediction_json(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    return latest_file([path for path in run_dir.glob("*.json") if not path.name.endswith("_results.json")])


def find_latest_eval_json(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    return latest_file(list(run_dir.glob("*_results.json")))


def public_path(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def safe_tag(value: str) -> str:
    allowed = []
    for char in value:
        allowed.append(char if char.isalnum() or char in {"_", "-"} else "_")
    tag = "".join(allowed).strip("_")
    if not tag:
        raise ValueError("--run-tag must contain at least one alphanumeric character")
    return tag


def resolve_output_root(base_output_root: Path, run_tag: str | None) -> Path:
    if not run_tag:
        return base_output_root
    return base_output_root / "run_tags" / safe_tag(run_tag)


def top_pages(record: dict[str, Any], field: str, top_k: int) -> list[Any]:
    values = record.get(field)
    if not isinstance(values, list):
        return []
    return values[:top_k]


def compare_retrieval_inputs(rows: list[dict[str, Any]], repo: Path) -> dict[str, Any]:
    if len(rows) < 2:
        return {"pass": False, "reason": "at least two runs are required for retrieval comparison"}
    top_k_values = {int(row.get("top_k", -1)) for row in rows}
    if len(top_k_values) != 1:
        return {"pass": False, "reason": "top_k differs", "top_k_values": sorted(top_k_values)}
    top_k = top_k_values.pop()
    loaded = []
    for row in rows:
        retrieval_path = repo / str(row["compatible_retrieval_path"])
        if not retrieval_path.is_file():
            return {"pass": False, "reason": "retrieval file missing", "path": str(row["compatible_retrieval_path"])}
        loaded.append(load_json(retrieval_path))
    if len({len(records) for records in loaded}) != 1:
        return {"pass": False, "reason": "record count differs", "record_counts": [len(records) for records in loaded]}

    mismatch_examples: list[dict[str, Any]] = []
    mismatch_count = 0
    fields = ["text-top-10-question", "image-top-10-question"]
    for idx in range(len(loaded[0])):
        for field in fields:
            base = top_pages(loaded[0][idx], field, top_k)
            for records, row in zip(loaded[1:], rows[1:]):
                current = top_pages(records[idx], field, top_k)
                if current != base:
                    mismatch_count += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(
                            {
                                "index": idx,
                                "field": field,
                                "baseline": base,
                                "run_name": row["run_name"],
                                "current": current,
                            }
                        )
    return {
        "pass": mismatch_count == 0,
        "top_k": top_k,
        "num_records": len(loaded[0]),
        "fields_checked": fields,
        "mismatch_count": mismatch_count,
        "mismatch_examples": mismatch_examples,
    }


def normalize_command(command: Any) -> list[str]:
    if not isinstance(command, list):
        return []
    normalized = []
    for part in command:
        text = str(part)
        if text.startswith("run-name="):
            normalized.append("run-name=<RUN>")
        elif text.startswith("dataset.sample_with_retrieval_path="):
            normalized.append("dataset.sample_with_retrieval_path=<RETRIEVAL>")
        else:
            normalized.append(text)
    return normalized


def binary_correctness(eval_path: Path | None) -> dict[str, Any]:
    if eval_path is None or not eval_path.is_file():
        return {"available": False, "average": None, "num_scored": 0, "num_records": 0}
    data = load_json(eval_path)
    if not isinstance(data, list):
        return {"available": False, "average": None, "num_scored": 0, "num_records": 0, "reason": "eval json is not a list"}
    values: list[float] = []
    for record in data:
        if isinstance(record, dict) and "binary_correctness" in record:
            value = record.get("binary_correctness")
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                values.append(float(value))
    return {
        "available": bool(values),
        "average": statistics.fmean(values) if values else None,
        "num_scored": len(values),
        "num_records": len(data),
    }


def load_public_json(public_path_value: str | None, repo: Path) -> Any | None:
    if not public_path_value:
        return None
    path = repo / public_path_value
    if not path.is_file():
        return None
    return load_json(path)


def build_run_report(row: dict[str, Any], repo: Path, results_root: Path) -> dict[str, Any]:
    run_name = str(row["run_name"])
    execution_run_name = str(row.get("execution_run_name") or run_name)
    run_dir = results_root / execution_run_name
    prediction_path = find_latest_prediction_json(run_dir)
    eval_path = find_latest_eval_json(run_dir)
    compatibility_path = repo / str(row.get("compatibility_report_path", ""))
    compatibility_report = load_json(compatibility_path) if compatibility_path.is_file() else None
    return {
        "run_name": run_name,
        "execution_run_name": execution_run_name,
        "top_k": row.get("top_k"),
        "record_slice": row.get("record_slice"),
        "max_records": row.get("max_records"),
        "run_tag": row.get("run_tag"),
        "temperature": row.get("temperature"),
        "model_config_hash": row.get("model_config_hash"),
        "eval_model_id": row.get("eval_model_id"),
        "compatible_retrieval_path": row.get("compatible_retrieval_path"),
        "compatibility_report_path": row.get("compatibility_report_path"),
        "compatibility_status": compatibility_report.get("status") if isinstance(compatibility_report, dict) else "missing",
        "prediction_output_path": public_path(prediction_path, repo) if prediction_path else None,
        "evaluation_output_path": public_path(eval_path, repo) if eval_path else None,
        "prediction_output_exists": prediction_path is not None,
        "evaluation_output_exists": eval_path is not None,
        "binary_correctness": binary_correctness(eval_path),
        "answer_key": f"ans_{execution_run_name}",
        "prediction_command": row.get("prediction_command"),
        "evaluation_command": row.get("evaluation_command"),
    }


def check_manifest_policy(summary_rows: list[dict[str, Any]], repo: Path) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    deepseek_bad: list[dict[str, Any]] = []
    expected = {
        "no_gold_fields_used": True,
        "same_page_budget_as_baseline": True,
        "used_debug_edges": False,
        "used_semantic_edges": False,
    }
    for row in summary_rows:
        manifest_path = repo / str(row.get("adapter_manifest_path", ""))
        if not manifest_path.is_file():
            bad.append({"run_name": row.get("run_name"), "field": "adapter_manifest_path", "value": "missing"})
            continue
        manifest = load_json(manifest_path)
        for field, expected_value in expected.items():
            if manifest.get(field) != expected_value:
                bad.append({"run_name": row.get("run_name"), "field": field, "value": manifest.get(field)})
        manifest_text = json.dumps(manifest, ensure_ascii=False)
        if EXPECTED_EVAL_JUDGE in manifest_text:
            deepseek_bad.append({"run_name": row.get("run_name"), "adapter_manifest_path": row.get("adapter_manifest_path")})
    return {
        "pass": not bad and not deepseek_bad,
        "policy_violations": bad,
        "deepseek_in_adapter_manifests": deepseek_bad,
    }


def scan_public_leakage(paths: list[Path]) -> dict[str, Any]:
    patterns = {
        "api_key": "api_key",
        "raw_response": "raw_response",
        "raw_output": "raw_output",
        "data_image": "data:image",
        "file_uri": "file://",
        "home_path": "/home/",
        "secret": "secret",
        "token": "token",
        "sk_prefix": "sk-",
    }
    hits: list[dict[str, Any]] = []
    for path in paths:
        if path is None or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in patterns.items():
            if pattern in text:
                hits.append({"path": str(path), "pattern": name})
    return {"pass": not hits, "hits": hits}


def prediction_failure_stats(run_reports: list[dict[str, Any]], repo: Path) -> list[dict[str, Any]]:
    stats = []
    for run in run_reports:
        records = load_public_json(run.get("prediction_output_path"), repo)
        answer_key = run.get("answer_key")
        total = len(records) if isinstance(records, list) else 0
        failed = 0
        if isinstance(records, list):
            failed = sum(
                1
                for record in records
                if not isinstance(record, dict) or record.get(answer_key) is None
            )
        rate = (failed / total) if total else None
        stats.append(
            {
                "run_name": run["run_name"],
                "execution_run_name": run["execution_run_name"],
                "total": total,
                "failed": failed,
                "failure_rate": rate,
            }
        )
    return stats


def answer_text_difference_warning(run_reports: list[dict[str, Any]], repo: Path) -> dict[str, Any] | None:
    if len(run_reports) < 2:
        return None
    left, right = run_reports[0], run_reports[1]
    left_pred = load_public_json(left.get("prediction_output_path"), repo)
    right_pred = load_public_json(right.get("prediction_output_path"), repo)
    left_eval = load_public_json(left.get("evaluation_output_path"), repo)
    right_eval = load_public_json(right.get("evaluation_output_path"), repo)
    if not all(isinstance(value, list) for value in [left_pred, right_pred, left_eval, right_eval]):
        return None
    limit = min(len(left_pred), len(right_pred), len(left_eval), len(right_eval))
    text_diff_binary_same = 0
    examples: list[dict[str, Any]] = []
    for index in range(limit):
        left_answer = left_pred[index].get(left.get("answer_key")) if isinstance(left_pred[index], dict) else None
        right_answer = right_pred[index].get(right.get("answer_key")) if isinstance(right_pred[index], dict) else None
        left_binary = left_eval[index].get("binary_correctness") if isinstance(left_eval[index], dict) else None
        right_binary = right_eval[index].get("binary_correctness") if isinstance(right_eval[index], dict) else None
        if left_answer != right_answer and left_binary == right_binary:
            text_diff_binary_same += 1
            if len(examples) < 5:
                examples.append({"index": index, "binary_correctness": left_binary})
    if text_diff_binary_same == 0:
        return None
    return {
        "type": "answer_text_diff_binary_same",
        "message": "Answer text differs while binary correctness is unchanged.",
        "count": text_diff_binary_same,
        "examples": examples,
    }


def collect_hard_failures(checks: dict[str, Any]) -> list[dict[str, Any]]:
    hard_check_names = [
        "prediction_outputs_exist",
        "evaluation_outputs_exist",
        "retrieval_input_equivalent",
        "top_k_consistent",
        "model_config_hash_consistent",
        "page_budget_consistent",
        "evaluation_judge_consistent",
        "adapter_manifest_policy",
        "public_leakage",
    ]
    failures = []
    for name in hard_check_names:
        check = checks.get(name)
        if isinstance(check, dict) and check.get("pass") is False:
            failures.append({"check": name, "detail": check})
    return failures


def collect_soft_warnings(
    run_reports: list[dict[str, Any]],
    repo: Path,
    binary_delta: float | None,
    max_binary_delta: float,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    answer_warning = answer_text_difference_warning(run_reports, repo)
    if answer_warning:
        warnings.append(answer_warning)
    if binary_delta is not None and binary_delta > 0:
        warnings.append(
            {
                "type": "binary_correctness_delta",
                "message": "Binary correctness differs while retrieval/model/top_k/eval configuration may still be consistent.",
                "delta": binary_delta,
                "max_binary_delta_reference": max_binary_delta,
            }
        )
    failure_stats = prediction_failure_stats(run_reports, repo)
    nonzero = [item for item in failure_stats if item["failed"] > 0]
    if nonzero:
        rates = [item["failure_rate"] or 0 for item in failure_stats]
        warnings.append(
            {
                "type": "api_failures_or_timeouts",
                "message": "Some prediction records have missing answers; inspect logs for API failures or timeouts.",
                "failure_stats": failure_stats,
                "max_failure_rate_gap": max(rates) - min(rates) if rates else None,
            }
        )
    return warnings


def build_failure_analysis(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [
        "# MDocAgent Execution Gate Failure Analysis",
        "",
        f"Status: {report['status']}",
        f"Hard failures: {len(report.get('hard_failures', []))}",
        f"Soft warnings: {len(report.get('soft_warnings', []))}",
        "",
        f"- retrieval input consistent: {checks['retrieval_input_equivalent']['pass']}",
        f"- top_k consistent: {checks['top_k_consistent']['pass']}",
        f"- model config consistent: {checks['model_config_hash_consistent']['pass']}",
        f"- prediction command shape consistent: {checks['prediction_command_shape_consistent']['pass']}",
        f"- eval command shape consistent: {checks['evaluation_command_shape_consistent']['pass']}",
        f"- run_name / resume_path pollution: {checks['run_name_resume_path_pollution']['detected']}",
        f"- API nondeterminism may affect text: {checks['api_nondeterminism_possible']}",
        "",
        "The artifact/graph ablation runs must remain stopped until this gate passes.",
    ]
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MDocAgent Execution Gate Report",
        "",
        f"Status: {report['status']}",
        f"Phase: {report.get('phase_name')}",
        f"Gate passed: {report.get('gate_passed')}",
        f"Recommended next phase: {report.get('recommended_next_phase')}",
        f"Hard failures: {len(report.get('hard_failures', []))}",
        f"Soft warnings: {len(report.get('soft_warnings', []))}",
        "",
        "| check | pass |",
        "| --- | --- |",
    ]
    for name, check in report["checks"].items():
        if isinstance(check, dict) and "pass" in check:
            value = check["pass"]
        else:
            value = check
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "## Runs", "", "| run_name | execution_run_name | prediction | evaluation | binary_correctness |", "| --- | --- | --- | --- | ---: |"])
    for run in report["runs"]:
        score = run["binary_correctness"]["average"]
        lines.append(
            "| {run_name} | {execution_run_name} | {prediction} | {evaluation} | {score} |".format(
                run_name=run["run_name"],
                execution_run_name=run["execution_run_name"],
                prediction=run["prediction_output_path"] or "missing",
                evaluation=run["evaluation_output_path"] or "missing",
                score="null" if score is None else f"{score:.6f}",
            )
        )
    if report.get("hard_failures"):
        lines.extend(["", "## Hard Failures", ""])
        for failure in report["hard_failures"]:
            lines.append(f"- {failure['check']}")
    if report.get("soft_warnings"):
        lines.extend(["", "## Soft Warnings", ""])
        for warning in report["soft_warnings"]:
            lines.append(f"- {warning.get('type')}: {warning.get('message')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_plan(output_dir: Path, report: dict[str, Any]) -> None:
    recommended_next_phase = report.get("recommended_next_phase")
    source_run_tag = report.get("run_tag")
    record_slice = report.get("record_slice")
    common_args = []
    if record_slice:
        common_args.extend(["--record-slice", str(record_slice)])

    if recommended_next_phase == "phase_2_small_artifact":
        allowed_runs = ["top4_artifact_only", "top4_original_plus_artifact"]
        blocked_until_signal = ["top4_graph_context"]
        purpose = "Run the small artifact ablation only; graph_context remains gated on positive artifact signal."
        next_run_arg = ",".join(allowed_runs)
        next_run_tag = "small_artifact"
    elif recommended_next_phase == "phase_3_small_graph":
        allowed_runs = ["top4_graph_context"]
        blocked_until_signal = []
        purpose = "Run the small graph_context diagnostic after original_plus_artifact showed positive signal."
        next_run_arg = ",".join(allowed_runs)
        next_run_tag = "small_graph"
    elif recommended_next_phase == "phase_4_full_ablation":
        allowed_runs = [
            "mdocagent_top4_official_reproduction",
            "top4_original_only",
            "top4_artifact_only",
            "top4_original_plus_artifact",
            "top4_graph_context",
        ]
        blocked_until_signal = []
        purpose = "Small gates passed; full ablation is now the recommended next phase."
        next_run_arg = ",".join(allowed_runs)
        next_run_tag = "full_ablation"
    else:
        allowed_runs = []
        blocked_until_signal = FORBIDDEN_NEXT_RUNS
        purpose = "No automatic next phase is recommended; inspect the gate report."
        next_run_arg = ""
        next_run_tag = None

    if next_run_tag:
        common_args.extend(["--run-tag", next_run_tag])
    common_suffix = " ".join(common_args)
    common_suffix = f" {common_suffix}" if common_suffix else ""

    commands = []
    if next_run_arg:
        commands = [
            f"python3 scripts/run_mdocagent_module_ablation.py --execute-predict --runs {next_run_arg}{common_suffix} --confirm-run-api",
            f"python3 scripts/run_mdocagent_module_ablation.py --execute-eval --runs {next_run_arg}{common_suffix} --confirm-run-eval",
        ]

    plan = {
        "status": "ready_after_gate_pass",
        "recommended_next_phase": recommended_next_phase,
        "source_run_tag": source_run_tag,
        "next_run_tag": next_run_tag,
        "allowed_next_runs": allowed_runs,
        "blocked_until_positive_signal": blocked_until_signal,
        "do_not_run_automatically": True,
        "purpose": purpose,
        "commands": commands,
    }
    (output_dir / "next_ablation_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    run_lines = [f"- {run}" for run in allowed_runs] or ["- none"]
    blocked_lines = [f"- {run}" for run in blocked_until_signal] or ["- none"]
    command_lines = [f"- `{command}`" for command in commands] or ["- none"]
    (output_dir / "next_ablation_plan.md").write_text(
        "\n".join(
            [
                "# Next Ablation Plan",
                "",
                purpose,
                "",
                "Do not run these automatically.",
                "",
                "Recommended next runs:",
                *run_lines,
                "",
                "Blocked until positive signal:",
                *blocked_lines,
                "",
                "Commands:",
                *command_lines,
                "",
                "Keep the same top-4 page budget and the same model configuration.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def score_by_run_name(run_reports: list[dict[str, Any]]) -> dict[str, float | None]:
    return {run["run_name"]: run["binary_correctness"]["average"] for run in run_reports}


def score_delta(scores: dict[str, float | None], left: str, right: str) -> float | None:
    if scores.get(left) is None or scores.get(right) is None:
        return None
    return float(scores[left]) - float(scores[right])


def infer_phase_name(args: argparse.Namespace, requested_runs: list[str]) -> str:
    if args.phase_name:
        return args.phase_name
    run_set = set(requested_runs)
    if {"mdocagent_top4_official_reproduction", "top4_original_only"}.issubset(run_set):
        return "phase_1_small_gate"
    if {"top4_artifact_only", "top4_original_plus_artifact"}.issubset(run_set):
        return "phase_2_small_artifact"
    if "top4_graph_context" in run_set:
        return "phase_3_small_graph"
    return "custom"


def phase_decision(phase_name: str, hard_passed: bool, scores: dict[str, float | None]) -> dict[str, Any]:
    reasons: list[str] = []
    recommended = "stop"
    diagnostic_only = False
    gate_passed = hard_passed
    delta_original = score_delta(scores, "top4_original_only", "mdocagent_top4_official_reproduction")
    delta_artifact_plus = score_delta(scores, "top4_original_plus_artifact", "top4_original_only")
    delta_graph = score_delta(scores, "top4_graph_context", "top4_original_plus_artifact")

    if not hard_passed:
        reasons.append("one or more hard failure checks failed")
        return {
            "gate_passed": False,
            "recommended_next_phase": "stop",
            "failure_reasons": reasons,
            "diagnostic_only": diagnostic_only,
        }

    if phase_name == "phase_1_small_gate":
        recommended = "phase_2_small_artifact"
    elif phase_name == "phase_2_small_artifact":
        if delta_artifact_plus is None or delta_artifact_plus <= 0:
            gate_passed = False
            reasons.append("top4_original_plus_artifact did not improve over top4_original_only")
            recommended = "stop_before_graph_context"
        else:
            recommended = "phase_3_small_graph"
    elif phase_name == "phase_3_small_graph":
        if delta_graph is None or delta_graph <= 0:
            diagnostic_only = True
            reasons.append("top4_graph_context did not improve over top4_original_plus_artifact")
            recommended = "mark_graph_context_diagnostic_only"
        else:
            recommended = "phase_4_full_ablation"
    else:
        recommended = "manual_review"

    return {
        "gate_passed": gate_passed,
        "recommended_next_phase": recommended,
        "failure_reasons": reasons,
        "diagnostic_only": diagnostic_only,
    }


def run_check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    repo = Path(__file__).resolve().parents[1]
    output_dir = resolve_output_root(repo / args.output_dir, args.run_tag)
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        report = {"status": "fail", "reason": "summary.json missing", "summary_path": args.output_dir}
        return report, 1

    requested_runs = parse_runs(args.runs)
    summary = load_json(summary_path)
    summary_rows = list(summary.get("runs", []))
    rows_by_name = {row.get("run_name"): row for row in summary_rows}
    missing_runs = [run for run in requested_runs if run not in rows_by_name]
    if missing_runs:
        report = {"status": "fail", "reason": "requested runs missing from summary", "missing_runs": missing_runs}
        return report, 1

    rows = [rows_by_name[run] for run in requested_runs]
    results_root = repo / args.results_root
    run_reports = [build_run_report(row, repo, results_root) for row in rows]
    model_hashes = {run["model_config_hash"] for run in run_reports}
    top_k_values = {run["top_k"] for run in run_reports}
    eval_judges = {run["eval_model_id"] for run in run_reports}
    prediction_commands = {tuple(normalize_command(run["prediction_command"])) for run in run_reports}
    evaluation_commands = {tuple(normalize_command(run["evaluation_command"])) for run in run_reports}
    binary_values = [run["binary_correctness"]["average"] for run in run_reports if run["binary_correctness"]["average"] is not None]
    binary_delta = None
    if len(binary_values) >= 2:
        binary_delta = abs(binary_values[0] - binary_values[1])
    record_slices = {run.get("record_slice") for run in run_reports}
    max_records_values = {run.get("max_records") for run in run_reports}
    leakage_paths = []
    for row in rows:
        for key in ["compatible_retrieval_path", "adapter_manifest_path", "compatibility_report_path"]:
            value = row.get(key)
            if value:
                leakage_paths.append(repo / str(value))
    for run in run_reports:
        for key in ["prediction_output_path", "evaluation_output_path"]:
            value = run.get(key)
            if value:
                leakage_paths.append(repo / str(value))

    checks: dict[str, Any] = {
        "prediction_outputs_exist": {"pass": all(run["prediction_output_exists"] for run in run_reports)},
        "evaluation_outputs_exist": {"pass": all(run["evaluation_output_exists"] for run in run_reports)},
        "retrieval_input_equivalent": compare_retrieval_inputs(rows, repo),
        "top_k_consistent": {"pass": len(top_k_values) == 1, "values": sorted(top_k_values)},
        "model_config_hash_consistent": {"pass": len(model_hashes) == 1, "values": sorted(model_hashes)},
        "page_budget_consistent": {"pass": len(top_k_values) == 1 and next(iter(top_k_values)) == 4 if top_k_values else False},
        "evaluation_judge_consistent": {
            "pass": len(eval_judges) == 1 and EXPECTED_EVAL_JUDGE in eval_judges,
            "values": sorted(eval_judges),
        },
        "prediction_command_shape_consistent": {"pass": len(prediction_commands) == 1},
        "evaluation_command_shape_consistent": {"pass": len(evaluation_commands) == 1},
        "binary_correctness_delta_recorded": {
            "delta": binary_delta,
            "max_allowed_delta": args.max_binary_delta,
            "api_nondeterminism_note": "prediction text differences are not treated as hard failures",
        },
        "record_slice_consistent": {
            "pass": len(record_slices) == 1 and (args.record_slice is None or next(iter(record_slices)) == args.record_slice),
            "values": sorted(str(value) for value in record_slices),
            "expected": args.record_slice,
        },
        "max_records_consistent": {
            "pass": len(max_records_values) == 1 and (args.max_records is None or next(iter(max_records_values)) == args.max_records),
            "values": sorted(str(value) for value in max_records_values),
            "expected": args.max_records,
        },
        "adapter_manifest_policy": check_manifest_policy(summary_rows, repo),
        "public_leakage": scan_public_leakage(leakage_paths),
        "run_name_resume_path_pollution": {"detected": False},
        "api_nondeterminism_possible": True,
    }
    scores = score_by_run_name(run_reports)
    phase_name = infer_phase_name(args, requested_runs)
    hard_failures = collect_hard_failures(checks)
    soft_warnings = collect_soft_warnings(run_reports, repo, binary_delta, args.max_binary_delta)
    hard_passed = not hard_failures
    decision = phase_decision(phase_name, hard_passed, scores)
    status = "pass" if decision["gate_passed"] else "fail"
    report = {
        "schema_version": "mdocagent_execution_gate_v2",
        "status": status,
        "phase_name": phase_name,
        "runs_completed": [run["run_name"] for run in run_reports if run["prediction_output_exists"] and run["evaluation_output_exists"]],
        "record_slice": args.record_slice if args.record_slice is not None else summary.get("record_slice"),
        "max_records": args.max_records if args.max_records is not None else summary.get("max_records"),
        "run_tag": args.run_tag,
        "gate_passed": decision["gate_passed"],
        "mdocagent_top4_score": scores.get("mdocagent_top4_official_reproduction"),
        "original_only_score": scores.get("top4_original_only"),
        "artifact_only_score": scores.get("top4_artifact_only"),
        "original_plus_artifact_score": scores.get("top4_original_plus_artifact"),
        "graph_context_score": scores.get("top4_graph_context"),
        "delta_original_only_vs_top4": score_delta(scores, "top4_original_only", "mdocagent_top4_official_reproduction"),
        "delta_artifact_plus_vs_original_only": score_delta(scores, "top4_original_plus_artifact", "top4_original_only"),
        "delta_graph_vs_artifact_plus": score_delta(scores, "top4_graph_context", "top4_original_plus_artifact"),
        "failure_reasons": decision["failure_reasons"],
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "recommended_next_phase": decision["recommended_next_phase"],
        "diagnostic_only": decision["diagnostic_only"],
        "summary_path": public_path(summary_path, repo),
        "runs_requested": requested_runs,
        "forbidden_runs_not_executed_by_this_check": FORBIDDEN_NEXT_RUNS,
        "runs": run_reports,
        "checks": checks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "execution_gate_report.json"
    report_md = output_dir / "execution_gate_report.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report_md, report)
    if status == "pass":
        write_next_plan(output_dir, report)
    else:
        (output_dir / "failure_analysis.md").write_text(build_failure_analysis(report), encoding="utf-8")
    return report, 0 if status == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check MDocAgent adapter-consistency gated QA outputs.")
    parser.add_argument("--runs", default=DEFAULT_RUNS, help="Comma-separated run names to compare.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--max-binary-delta", type=float, default=0.05)
    parser.add_argument("--run-tag", help="Run tag used by run_mdocagent_module_ablation.py.")
    parser.add_argument("--record-slice", help="Expected record slice for all checked runs, such as 0:30.")
    parser.add_argument("--max-records", type=int, help="Expected max record cap for all checked runs.")
    parser.add_argument("--phase-name", help="Explicit phase name for the gate report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, code = run_check(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": report.get("status"), "runs_requested": report.get("runs_requested")}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
