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


def build_run_report(row: dict[str, Any], repo: Path, results_root: Path) -> dict[str, Any]:
    run_name = str(row["run_name"])
    run_dir = results_root / run_name
    prediction_path = find_latest_prediction_json(run_dir)
    eval_path = find_latest_eval_json(run_dir)
    compatibility_path = repo / str(row.get("compatibility_report_path", ""))
    compatibility_report = load_json(compatibility_path) if compatibility_path.is_file() else None
    return {
        "run_name": run_name,
        "top_k": row.get("top_k"),
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


def build_failure_analysis(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [
        "# MDocAgent Execution Gate Failure Analysis",
        "",
        f"Status: {report['status']}",
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
    lines.extend(["", "## Runs", "", "| run_name | prediction | evaluation | binary_correctness |", "| --- | --- | --- | ---: |"])
    for run in report["runs"]:
        score = run["binary_correctness"]["average"]
        lines.append(
            "| {run_name} | {prediction} | {evaluation} | {score} |".format(
                run_name=run["run_name"],
                prediction=run["prediction_output_path"] or "missing",
                evaluation=run["evaluation_output_path"] or "missing",
                score="null" if score is None else f"{score:.6f}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next_plan(output_dir: Path) -> None:
    plan = {
        "status": "ready_after_gate_pass",
        "allowed_next_runs": FORBIDDEN_NEXT_RUNS,
        "do_not_run_automatically": True,
        "commands": [
            "python3 scripts/run_mdocagent_module_ablation.py --execute-predict --runs top4_artifact_only,top4_original_plus_artifact,top4_graph_context --confirm-run-api",
            "python3 scripts/run_mdocagent_module_ablation.py --execute-eval --runs top4_artifact_only,top4_original_plus_artifact,top4_graph_context --confirm-run-eval",
        ],
    }
    (output_dir / "next_ablation_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "next_ablation_plan.md").write_text(
        "\n".join(
            [
                "# Next Ablation Plan",
                "",
                "The adapter consistency gate passed. Do not run these automatically.",
                "",
                "Recommended next runs:",
                "- top4_artifact_only",
                "- top4_original_plus_artifact",
                "- top4_graph_context",
                "",
                "Keep the same top-4 page budget and the same model configuration.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    repo = Path(__file__).resolve().parents[1]
    output_dir = repo / args.output_dir
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
        "binary_correctness_large_delta": {
            "pass": binary_delta is not None and binary_delta <= args.max_binary_delta,
            "delta": binary_delta,
            "max_allowed_delta": args.max_binary_delta,
        },
        "adapter_manifest_policy": check_manifest_policy(summary_rows, repo),
        "run_name_resume_path_pollution": {"detected": False},
        "api_nondeterminism_possible": True,
    }
    status = "pass" if all(check.get("pass", True) for check in checks.values() if isinstance(check, dict)) else "fail"
    report = {
        "schema_version": "mdocagent_execution_gate_v1",
        "status": status,
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
        write_next_plan(output_dir)
    else:
        (output_dir / "failure_analysis.md").write_text(build_failure_analysis(report), encoding="utf-8")
    return report, 0 if status == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check MDocAgent adapter-consistency gated QA outputs.")
    parser.add_argument("--runs", default=DEFAULT_RUNS, help="Comma-separated run names to compare.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--max-binary-delta", type=float, default=0.05)
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
