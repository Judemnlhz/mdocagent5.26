#!/usr/bin/env python3
"""Prepare MDocAgent-compatible module ablation runs.

Default behavior is prepare-only: write adapted retrieval records, adapter
manifests, compatibility reports, and runnable predict/eval commands. It does
not call real APIs unless --execute-predict or --execute-eval is passed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdocnexus.integration.mdocagent_adapter import (
    build_mdocagent_adapter_manifest,
    canonical_json_hash,
    combined_model_config_hash,
    load_artifacts_by_page,
    load_mdocagent_retrieval_records,
    rerank_pages_with_artifacts,
    select_pages_with_graph,
    write_manifest,
    write_mdocagent_compatible_records,
)


DEFAULT_INPUT_RETRIEVAL = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_ARTIFACTS = "outputs/stage2_doc/artifacts.jsonl"
DEFAULT_GRAPH_DIR = "outputs/stage4/evidence_graph"
DEFAULT_OUTPUT_DIR = "outputs/experiments/mdocagent_module_ablation"
EVAL_MODEL_ID = "deepseek-ai/DeepSeek-V3"
PREDICT_CALLS_PER_RECORD = 5
EVAL_CALLS_PER_RECORD = 1


RUN_SPECS = [
    {
        "run_name": "mdocagent_top1_official_reproduction",
        "experiment_group": "official_reproduction",
        "baseline_relation": "mdocagent_top1_scope",
        "official_mdocagent_setting": True,
        "top_k": 1,
        "mode": "original_only",
        "module_enabled": "none_original_mdocagent",
    },
    {
        "run_name": "mdocagent_top4_official_reproduction",
        "experiment_group": "official_reproduction",
        "baseline_relation": "mdocagent_top4_scope",
        "official_mdocagent_setting": True,
        "top_k": 4,
        "mode": "original_only",
        "module_enabled": "none_original_mdocagent",
    },
    {
        "run_name": "top4_original_only",
        "experiment_group": "adapter_sanity_check",
        "baseline_relation": "mdocagent_top4_scope",
        "official_mdocagent_setting": False,
        "top_k": 4,
        "mode": "original_only",
        "module_enabled": "adapter_original_only",
    },
    {
        "run_name": "top4_artifact_only",
        "experiment_group": "module_ablation",
        "baseline_relation": "mdocagent_top4_scope",
        "official_mdocagent_setting": False,
        "top_k": 4,
        "mode": "artifact_only",
        "module_enabled": "artifact_reranking",
    },
    {
        "run_name": "top4_original_plus_artifact",
        "experiment_group": "module_ablation",
        "baseline_relation": "mdocagent_top4_scope",
        "official_mdocagent_setting": False,
        "top_k": 4,
        "mode": "original_plus_artifact",
        "module_enabled": "original_plus_artifact_reranking",
    },
    {
        "run_name": "top4_graph_context",
        "experiment_group": "module_ablation",
        "baseline_relation": "mdocagent_top4_scope",
        "official_mdocagent_setting": False,
        "top_k": 4,
        "mode": "graph_context",
        "module_enabled": "graph_guided_page_selection",
    },
]


def run_module_ablation(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    base_output_root = Path(args.output_dir)
    output_root = resolve_output_root(base_output_root, args.run_tag)
    records_dir = output_root / "reranked_records"
    manifests_dir = output_root / "manifests"
    compatibility_dir = output_root / "compatibility"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    compatibility_dir.mkdir(parents=True, exist_ok=True)

    records = select_records(
        load_mdocagent_retrieval_records(args.input_retrieval),
        record_slice=args.record_slice,
        max_records=args.max_records,
        record_ids_file=args.record_ids_file,
    )
    artifacts_by_page = load_artifacts_by_page(args.artifacts)
    model_config_hash = combined_model_config_hash(repo)
    selected_run_names = parse_run_filter(args.runs)
    write_phased_experiment_plan(base_output_root)

    summary_rows: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        run_name = str(spec["run_name"])
        execute_selected = selected_run_names is None or run_name in selected_run_names
        row = prepare_run(
            spec,
            args,
            repo,
            records,
            artifacts_by_page,
            records_dir,
            manifests_dir,
            compatibility_dir,
            model_config_hash,
            execute_selected=execute_selected,
        )
        summary_rows.append(row)
    cost_summary = write_cost_log_if_requested(args, repo, output_root, summary_rows, len(records))

    summary = {
        "schema_version": "mdocagent_module_ablation_v3",
        "prepare_only": not args.execute_predict and not args.execute_eval,
        "official_reproduction_top_k": [1, 4],
        "additional_budget_policy": "top-8/top-10/top-20 are additional_budget_diagnostic only, not official reproduction",
        "selected_runs": sorted(selected_run_names) if selected_run_names is not None else "all",
        "run_tag": args.run_tag,
        "record_slice": args.record_slice,
        "max_records": args.max_records,
        "record_ids_file": args.record_ids_file,
        "num_records": len(records),
        "temperature": args.temperature,
        "resume_safe": bool(args.resume_safe),
        "overwrite_output": bool(args.overwrite_output),
        "cost_log_path": cost_summary.get("path") if cost_summary else None,
        "runs": summary_rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_md(output_root / "summary.md", summary_rows)
    failed_rows = [
        row
        for row in summary_rows
        if row.get("status") not in {"prepared_not_run", "prediction_complete", "complete"}
    ]
    return {
        "summary_path": str(summary_path),
        "summary_md_path": str(output_root / "summary.md"),
        "num_runs": len(summary_rows),
        "selected_runs": sorted(selected_run_names) if selected_run_names is not None else "all",
        "run_tag": args.run_tag,
        "record_slice": args.record_slice,
        "max_records": args.max_records,
        "num_records": len(records),
        "cost_log_path": cost_summary.get("path") if cost_summary else None,
        "num_failed_runs": len(failed_rows),
        "prepare_only": not args.execute_predict and not args.execute_eval,
        "status": "fail" if failed_rows else ("prepared" if not args.execute_predict and not args.execute_eval else "executed_or_attempted"),
    }


def prepare_run(
    spec: dict[str, Any],
    args: argparse.Namespace,
    repo: Path,
    records: list[dict[str, Any]],
    artifacts_by_page: dict[str, dict[int, list[dict[str, Any]]]],
    records_dir: Path,
    manifests_dir: Path,
    compatibility_dir: Path,
    model_config_hash: str,
    execute_selected: bool = True,
) -> dict[str, Any]:
    run_name = str(spec["run_name"])
    execution_run_name = tagged_run_name(run_name, args.run_tag)
    top_k = int(spec["top_k"])
    mode = str(spec["mode"])
    output_retrieval = records_dir / f"{run_name}.json"
    adapter_manifest_path = manifests_dir / f"{run_name}.adapter.json"
    run_manifest_path = manifests_dir / f"{run_name}.json"
    compatibility_report_path = compatibility_dir / f"{run_name}.compatibility_report.json"

    if mode == "graph_context":
        adapted = select_pages_with_graph(records, artifacts_by_page, args.graph_dir, top_k=top_k, expansion_mode=args.expansion_mode)
    else:
        adapted = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=top_k, mode=mode, lambda_weight=args.lambda_weight)
    write_mdocagent_compatible_records(adapted, output_retrieval)

    adapter_manifest = build_mdocagent_adapter_manifest(
        mode=mode,
        top_k=top_k,
        lambda_weight=args.lambda_weight,
        input_retrieval=args.input_retrieval,
        artifacts=args.artifacts,
        graph_dir=args.graph_dir if mode == "graph_context" else None,
        output_retrieval=output_retrieval,
        expansion_mode=args.expansion_mode if mode == "graph_context" else None,
        command_args={
            "input_retrieval": args.input_retrieval,
            "artifacts": args.artifacts,
            "graph_dir": args.graph_dir if mode == "graph_context" else None,
            "output_retrieval": output_retrieval,
            "mode": mode,
            "top_k": top_k,
            "lambda_weight": args.lambda_weight,
            "expansion_mode": args.expansion_mode if mode == "graph_context" else None,
            "record_slice": args.record_slice,
            "max_records": args.max_records,
            "record_ids_file": args.record_ids_file,
            "run_tag": args.run_tag,
            "temperature": args.temperature,
        },
        repo_root=repo,
    )
    adapter_manifest.update(
        {
            "record_slice": args.record_slice,
            "max_records": args.max_records,
            "record_ids_file": public_path(args.record_ids_file, repo) if args.record_ids_file else None,
            "num_records": len(adapted),
            "run_tag": args.run_tag,
            "temperature": args.temperature,
        }
    )
    write_manifest(adapter_manifest, adapter_manifest_path)

    compatibility_returncode = run_command(
        [
            sys.executable,
            "scripts/check_mdocagent_adapter_compatibility.py",
            "--input-retrieval",
            public_path(output_retrieval, repo),
            "--output-report",
            public_path(compatibility_report_path, repo),
            "--top-k",
            "1,4",
        ],
        repo,
    )
    compatibility_report = load_json_if_exists(compatibility_report_path)

    prediction_command = build_prediction_command(args, execution_run_name, top_k, output_retrieval)
    evaluation_command = build_evaluation_command(args, execution_run_name)
    prediction_returncode: int | None = None
    evaluation_returncode: int | None = None
    status = "prepared_not_run"
    output_blocker = find_existing_execution_output(repo, execution_run_name)
    if compatibility_returncode != 0:
        status = "compatibility_failed"
    elif execute_selected and (args.execute_predict or args.execute_eval) and output_blocker and not args.overwrite_output and not args.resume_safe:
        status = "blocked_existing_output"
    if compatibility_returncode == 0 and execute_selected and args.execute_predict:
        if status != "blocked_existing_output":
            prediction_returncode = run_command(prediction_command, repo)
            status = "prediction_failed" if prediction_returncode != 0 else "prediction_complete"
    if compatibility_returncode == 0 and execute_selected and args.execute_eval:
        if status == "blocked_existing_output":
            evaluation_returncode = None
        elif args.execute_predict and prediction_returncode not in (0, None):
            status = "prediction_failed"
        else:
            evaluation_returncode = run_command(evaluation_command, repo)
            status = "complete" if evaluation_returncode == 0 else "evaluation_failed"

    row = {
        "run_name": run_name,
        "execution_run_name": execution_run_name,
        "experiment_group": spec["experiment_group"],
        "baseline_relation": spec["baseline_relation"],
        "official_mdocagent_setting": bool(spec["official_mdocagent_setting"]),
        "top_k": top_k,
        "module_enabled": spec["module_enabled"],
        "same_model_as_baseline": True,
        "same_page_budget_as_baseline": True,
        "model_config_hash": model_config_hash,
        "compatible_retrieval_path": public_path(output_retrieval, repo),
        "adapter_manifest_path": public_path(adapter_manifest_path, repo),
        "compatibility_report_path": public_path(compatibility_report_path, repo),
        "compatibility_status": compatibility_report.get("status") if isinstance(compatibility_report, dict) else "missing",
        "prediction_command": prediction_command,
        "evaluation_command": evaluation_command,
        "prediction_returncode": prediction_returncode,
        "evaluation_returncode": evaluation_returncode,
        "qa_accuracy_or_binary_correctness": None,
        "delta_vs_mdocagent_top4": None,
        "eval_model_id": EVAL_MODEL_ID,
        "no_gold_fields_used": True,
        "used_debug_edges": False,
        "used_semantic_edges": False,
        "status": status,
        "record_slice": args.record_slice,
        "max_records": args.max_records,
        "record_ids_file": public_path(args.record_ids_file, repo) if args.record_ids_file else None,
        "num_records": len(adapted),
        "run_tag": args.run_tag,
        "temperature": args.temperature,
        "resume_safe": bool(args.resume_safe),
        "overwrite_output": bool(args.overwrite_output),
        "existing_output_path": public_path(output_blocker, repo) if output_blocker else None,
        "adapter_manifest_hash": canonical_json_hash(adapter_manifest),
        "retrieval_records_hash": adapter_manifest["output_hash"],
        "additional_budget_diagnostic": False,
    }
    run_manifest = {
        **row,
        "adapter_manifest": adapter_manifest,
        "compatibility_report": compatibility_report,
    }
    write_manifest(run_manifest, run_manifest_path)
    return row


def build_prediction_command(args: argparse.Namespace, run_name: str, top_k: int, retrieval_path: Path) -> list[str]:
    command = [
        "python3",
        "scripts/predict.py",
        "--config-name",
        args.config_name,
        f"run-name={run_name}",
        f"dataset.top_k={top_k}",
        f"dataset.sample_with_retrieval_path={public_path(retrieval_path, Path(__file__).resolve().parents[1])}",
    ]
    if args.temperature is not None:
        command.append(f"+runtime.temperature={args.temperature}")
    if args.resume_safe:
        resume_path = latest_prediction_path(Path(__file__).resolve().parents[1], run_name)
        if resume_path:
            command.append(f"+runtime.resume_path={public_path(resume_path, Path(__file__).resolve().parents[1])}")
    return command


def build_evaluation_command(args: argparse.Namespace, run_name: str) -> list[str]:
    return ["python3", "scripts/eval.py", "--config-name", args.config_name, f"run-name={run_name}"]


def run_command(command: list[str], cwd: Path) -> int:
    completed = subprocess.run(command, cwd=cwd, check=False)
    return int(completed.returncode)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def public_path(path: str | Path | None, repo_root: str | Path) -> str | None:
    if path in (None, ""):
        return None
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(Path(repo_root).resolve()))
    except ValueError:
        return path_obj.name


def resolve_output_root(base_output_root: Path, run_tag: str | None) -> Path:
    if not run_tag:
        return base_output_root
    return base_output_root / "run_tags" / safe_tag(run_tag)


def safe_tag(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("_")
    tag = "".join(allowed).strip("_")
    if not tag:
        raise ValueError("--run-tag must contain at least one alphanumeric character")
    return tag


def tagged_run_name(run_name: str, run_tag: str | None) -> str:
    if not run_tag:
        return run_name
    return f"{run_name}__{safe_tag(run_tag)}"


def parse_record_slice(value: str | None, total: int) -> slice:
    if value in (None, ""):
        return slice(None)
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError("--record-slice must use Python slice syntax such as 0:30")
    parsed: list[int | None] = []
    for part in parts:
        parsed.append(int(part) if part.strip() else None)
    while len(parsed) < 3:
        parsed.append(None)
    result = slice(parsed[0], parsed[1], parsed[2])
    range(total)[result]
    return result


def load_record_ids(path: str | None) -> list[int] | None:
    if not path:
        return None
    ids_path = Path(path)
    text = ids_path.read_text(encoding="utf-8")
    if ids_path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("--record-ids-file JSON must contain a list of 0-based integer indices")
        values = data
    else:
        values = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    record_ids = [int(value) for value in values]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("--record-ids-file contains duplicate indices")
    return record_ids


def select_records(
    records: list[dict[str, Any]],
    record_slice: str | None,
    max_records: int | None,
    record_ids_file: str | None,
) -> list[dict[str, Any]]:
    selected = records
    record_ids = load_record_ids(record_ids_file)
    if record_ids is not None:
        selected = [records[index] for index in record_ids]
    if record_slice:
        selected = selected[parse_record_slice(record_slice, len(selected))]
    if max_records is not None:
        if max_records < 0:
            raise ValueError("--max-records must be non-negative")
        selected = selected[:max_records]
    return list(selected)


def result_dir(repo: Path, execution_run_name: str) -> Path:
    return repo / "results" / "MMLongBench" / execution_run_name


def latest_prediction_path(repo: Path, execution_run_name: str) -> Path | None:
    directory = result_dir(repo, execution_run_name)
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.glob("*-*-*-*-*.json") if not path.name.endswith("_results.json")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def find_existing_execution_output(repo: Path, execution_run_name: str) -> Path | None:
    directory = result_dir(repo, execution_run_name)
    if not directory.is_dir():
        return None
    candidates = list(directory.glob("*.json")) + list(directory.glob("results.txt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def count_prediction_outputs(repo: Path, execution_run_name: str, ans_key: str) -> dict[str, int]:
    path = latest_prediction_path(repo, execution_run_name)
    if not path:
        return {"completed": 0, "failed": 0}
    records = json.loads(path.read_text(encoding="utf-8"))
    completed = sum(1 for record in records if isinstance(record, dict) and record.get(ans_key) is not None)
    return {"completed": completed, "failed": max(len(records) - completed, 0)}


def write_cost_log_if_requested(
    args: argparse.Namespace,
    repo: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
    num_records: int,
) -> dict[str, Any] | None:
    if args.cost_log is None:
        return None
    path = output_root / "cost_log.json" if args.cost_log == "" else Path(args.cost_log)
    if not path.is_absolute():
        path = repo / path
    selected = parse_run_filter(args.runs)
    selected_rows = [row for row in rows if selected is None or row["run_name"] in selected]
    run_items = []
    totals = {"estimated_api_calls": 0, "completed_api_calls": 0, "failed_api_calls": 0}
    for row in selected_rows:
        calls_per_record = 0
        if args.execute_predict:
            calls_per_record += PREDICT_CALLS_PER_RECORD
        if args.execute_eval:
            calls_per_record += EVAL_CALLS_PER_RECORD
        estimate = num_records * calls_per_record
        ans_key = f"ans_{row['execution_run_name']}"
        counts = count_prediction_outputs(repo, row["execution_run_name"], ans_key)
        completed = counts["completed"] * (PREDICT_CALLS_PER_RECORD if args.execute_predict else 0)
        failed = counts["failed"] * (PREDICT_CALLS_PER_RECORD if args.execute_predict else 0)
        item = {
            "run_name": row["run_name"],
            "execution_run_name": row["execution_run_name"],
            "estimated_api_calls": estimate,
            "completed_api_calls": completed,
            "failed_api_calls": failed,
        }
        for key in totals:
            totals[key] += item[key]
        run_items.append(item)
    cost_log = {
        "schema_version": "mdocagent_cost_log_v1",
        "run_tag": args.run_tag,
        "record_slice": args.record_slice,
        "max_records": args.max_records,
        "num_records": num_records,
        "runs": run_items,
        **totals,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cost_log, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": public_path(path, repo), **totals}


def parse_run_filter(value: str | None) -> set[str] | None:
    if value in (None, ""):
        return None
    selected = {item.strip() for item in value.split(",") if item.strip()}
    valid = {str(spec["run_name"]) for spec in RUN_SPECS}
    unknown = sorted(selected - valid)
    if unknown:
        raise ValueError(f"Unknown run name(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("--runs was provided but no run names were parsed")
    return selected


def has_siliconflow_credentials(repo: Path) -> bool:
    if os.environ.get("SILICONFLOW_API_KEY"):
        return True
    model_dir = repo / "config" / "model"
    for path in model_dir.glob("*.yaml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("api_key:"):
                _, value = stripped.split(":", 1)
                value = value.strip().strip("'\"")
                if value and not value.startswith("${"):
                    return True
    return False


def write_phased_experiment_plan(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    phases = [
        {
            "phase_name": "phase_1_small_gate",
            "runs": ["mdocagent_top4_official_reproduction", "top4_original_only"],
            "record_slice": "0:30",
            "purpose": "adapter consistency under real QA",
            "allowed_if": "adapter gate has passed",
        },
        {
            "phase_name": "phase_2_small_artifact",
            "runs": ["top4_artifact_only", "top4_original_plus_artifact"],
            "record_slice": "0:30",
            "purpose": "test artifact-aware reranking signal under same page budget",
            "allowed_if": "phase_1_small_gate passes",
        },
        {
            "phase_name": "phase_3_small_graph",
            "runs": ["top4_graph_context"],
            "record_slice": "0:30",
            "purpose": "test graph_context only after artifact signal is positive",
            "allowed_if": "top4_original_plus_artifact has positive signal",
        },
        {
            "phase_name": "phase_4_full_ablation",
            "runs": [
                "mdocagent_top4_official_reproduction",
                "top4_original_only",
                "top4_artifact_only",
                "top4_original_plus_artifact",
                "top4_graph_context",
            ],
            "record_slice": None,
            "purpose": "paper-facing full top-4 QA ablation",
            "allowed_if": "small runs pass sanity",
        },
    ]
    plan = {
        "schema_version": "mdocagent_phased_experiment_plan_v1",
        "default_record_slice": "0:30",
        "do_not_run_all_phases_automatically": True,
        "phases": phases,
    }
    (output_root / "phased_experiment_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# MDocAgent Phased Experiment Plan",
        "",
        "Run phases sequentially. Do not run artifact or graph ablations until the preceding gate passes.",
    ]
    for phase in phases:
        lines.extend(
            [
                "",
                f"## {phase['phase_name']}",
                f"- runs: {', '.join(phase['runs'])}",
                f"- record_slice: {phase['record_slice']}",
                f"- purpose: {phase['purpose']}",
                f"- allowed_if: {phase['allowed_if']}",
            ]
        )
    (output_root / "phased_experiment_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# MDocAgent Module Ablation",
        "",
        "Status: prepared_not_run unless explicitly executed",
        "",
        "| run_name | group | top_k | module | compatibility | status |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {experiment_group} | {top_k} | {module_enabled} | {compatibility_status} | {status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Official reproduction is limited to top-1 and top-4. Larger budgets are additional diagnostics only.",
            "Prepared runs do not generate answers or call evaluation APIs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MDocAgent module ablation retrieval records and manifests.")
    parser.add_argument("--input-retrieval", default=DEFAULT_INPUT_RETRIEVAL)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--graph-dir", default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument(
        "--expansion-mode",
        choices=["page_neighborhood", "source_anchor_neighborhood", "direct_structural"],
        default="page_neighborhood",
    )
    parser.add_argument("--config-name", default="mmlb")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare files only. This is the default.")
    parser.add_argument("--execute-predict", action="store_true", help="Run predict.py for each prepared run.")
    parser.add_argument("--execute-eval", action="store_true", help="Run eval.py for each prepared run.")
    parser.add_argument("--execute", action="store_true", help="Deprecated alias for --execute-predict --execute-eval.")
    parser.add_argument("--runs", help="Comma-separated run names to execute. All runs are still prepared for manifest consistency.")
    parser.add_argument("--confirm-run-api", action="store_true", help="Required with --execute-predict to allow real prediction API calls.")
    parser.add_argument("--confirm-run-eval", action="store_true", help="Required with --execute-eval to allow real evaluation API calls.")
    parser.add_argument("--max-records", type=int, help="Optional cap applied after record ids and record slice.")
    parser.add_argument("--record-slice", help="Optional Python-style slice such as 0:30. Applied to all runs.")
    parser.add_argument("--record-ids-file", help="Optional file containing 0-based record indices, one per line or JSON list.")
    parser.add_argument("--run-tag", help="Tag used to isolate small or phased run outputs.")
    parser.add_argument("--temperature", type=float, help="Optional prediction model temperature override recorded in manifests.")
    parser.add_argument("--resume-safe", action="store_true", help="Resume from the latest prediction JSON instead of blocking on existing outputs.")
    parser.add_argument("--overwrite-output", action="store_true", help="Allow execution when prior outputs exist for the tagged run.")
    parser.add_argument(
        "--cost-log",
        nargs="?",
        const="",
        help="Write API call cost log. With no value, writes cost_log.json under the run output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        args.execute_predict = True
        args.execute_eval = True
    try:
        parse_run_filter(args.runs)
        if args.run_tag:
            safe_tag(args.run_tag)
        if args.record_slice:
            parse_record_slice(args.record_slice, 10**12)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.execute_predict and not args.confirm_run_api:
        print("blocked_missing_confirm_run_api", file=sys.stderr)
        print(json.dumps({"status": "blocked_missing_confirm_run_api"}, ensure_ascii=False, sort_keys=True))
        return 1
    if args.execute_eval and not args.confirm_run_eval:
        print("blocked_missing_confirm_run_eval", file=sys.stderr)
        print(json.dumps({"status": "blocked_missing_confirm_run_eval"}, ensure_ascii=False, sort_keys=True))
        return 1
    repo = Path(__file__).resolve().parents[1]
    if (args.execute_predict or args.execute_eval) and not has_siliconflow_credentials(repo):
        print(
            "SILICONFLOW_API_KEY or a non-empty config/model/*.yaml api_key is required for --execute-predict/--execute-eval",
            file=sys.stderr,
        )
        return 2
    result = run_module_ablation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
