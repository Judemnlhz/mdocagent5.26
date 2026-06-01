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
    output_root = Path(args.output_dir)
    records_dir = output_root / "reranked_records"
    manifests_dir = output_root / "manifests"
    compatibility_dir = output_root / "compatibility"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    compatibility_dir.mkdir(parents=True, exist_ok=True)

    records = load_mdocagent_retrieval_records(args.input_retrieval)
    artifacts_by_page = load_artifacts_by_page(args.artifacts)
    model_config_hash = combined_model_config_hash(repo)
    selected_run_names = parse_run_filter(args.runs)

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

    summary = {
        "schema_version": "mdocagent_module_ablation_v2",
        "prepare_only": not args.execute_predict and not args.execute_eval,
        "official_reproduction_top_k": [1, 4],
        "additional_budget_policy": "top-8/top-10/top-20 are additional_budget_diagnostic only, not official reproduction",
        "selected_runs": sorted(selected_run_names) if selected_run_names is not None else "all",
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
        },
        repo_root=repo,
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

    prediction_command = build_prediction_command(args, run_name, top_k, output_retrieval)
    evaluation_command = build_evaluation_command(args, run_name)
    prediction_returncode: int | None = None
    evaluation_returncode: int | None = None
    status = "prepared_not_run"
    if compatibility_returncode != 0:
        status = "compatibility_failed"
    if compatibility_returncode == 0 and execute_selected and args.execute_predict:
        prediction_returncode = run_command(prediction_command, repo)
        status = "prediction_failed" if prediction_returncode != 0 else "prediction_complete"
    if compatibility_returncode == 0 and execute_selected and args.execute_eval:
        if args.execute_predict and prediction_returncode not in (0, None):
            status = "prediction_failed"
        else:
            evaluation_returncode = run_command(evaluation_command, repo)
            status = "complete" if evaluation_returncode == 0 else "evaluation_failed"

    row = {
        "run_name": run_name,
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
    return [
        "python3",
        "scripts/predict.py",
        "--config-name",
        args.config_name,
        f"run-name={run_name}",
        f"dataset.top_k={top_k}",
        f"dataset.sample_with_retrieval_path={public_path(retrieval_path, Path(__file__).resolve().parents[1])}",
    ]


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        args.execute_predict = True
        args.execute_eval = True
    try:
        parse_run_filter(args.runs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.execute_predict and not args.confirm_run_api:
        print("--confirm-run-api is required with --execute-predict", file=sys.stderr)
        return 2
    if args.execute_eval and not args.confirm_run_eval:
        print("--confirm-run-eval is required with --execute-eval", file=sys.stderr)
        return 2
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
