#!/usr/bin/env python3
"""Unified CLI for Stage 2/3/4, evaluation, experiments, and audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_MAP = {
    "stage2-build-subset": ["scripts/build_stage2_coverage_subset.py"],
    "stage2-compile": ["scripts/stage2.py", "doc-compile"],
    "stage3-retrieve": ["scripts/stage3_doc_artifact_retrieval.py"],
    "stage4-build-graph": ["scripts/stage4_build_evidence_graph.py"],
    "eval-stage3": ["scripts/eval_stage3_retrieval.py"],
    "eval-stage4": ["scripts/eval_stage4_graph_expansion.py"],
    "run-coverage": ["scripts/run_coverage_experiment.py"],
    "run-matrix": ["scripts/run_experiment_matrix.py"],
    "run-real-smoke-small": ["scripts/run_real_smoke_small.py"],
    "verify": ["scripts/verify_stage2_stage3_stage4_refactor.py"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified MDocNexus CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in SCRIPT_MAP:
        subparser = subparsers.add_parser(name, help=f"Forward to {' '.join(SCRIPT_MAP[name])}.")

    audit_parser = subparsers.add_parser("audit", help="Run leakage, reproducibility, and model config audits.")
    audit_parser.add_argument("--model-configs", action="store_true")
    audit_parser.add_argument("--no-gold", action="store_true")
    audit_parser.add_argument("--reproducibility", action="store_true")
    audit_parser.add_argument("--all", action="store_true")
    return parser


def run_command(command: list[str], cwd: Path) -> int:
    completed = subprocess.run(command, cwd=cwd)
    return int(completed.returncode)


def run_forwarded(command_name: str, forwarded_args: list[str], repo: Path) -> int:
    command = ["python3", *SCRIPT_MAP[command_name], *forwarded_args]
    return run_command(command, repo)


def run_audit(args: argparse.Namespace, repo: Path, forwarded_args: list[str]) -> int:
    selected = {
        "model_configs": bool(args.model_configs),
        "no_gold": bool(args.no_gold),
        "reproducibility": bool(args.reproducibility),
    }
    if args.all or not any(selected.values()):
        selected = {key: True for key in selected}
    commands: list[list[str]] = []
    if selected["no_gold"]:
        commands.append(["python3", "scripts/audit_no_gold_leakage.py"])
    if selected["reproducibility"]:
        commands.append(["python3", "scripts/audit_reproducibility.py"])
    if selected["model_configs"]:
        commands.append(["python3", "scripts/audit_model_configs.py", *forwarded_args])
    results: list[dict[str, Any]] = []
    status = 0
    for command in commands:
        completed = subprocess.run(command, cwd=repo)
        results.append({"command": command, "returncode": completed.returncode})
        if completed.returncode != 0:
            status = completed.returncode
    print(json.dumps({"audit_results": results, "status": "pass" if status == 0 else "fail"}, indent=2, sort_keys=True))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.command == "audit":
        return run_audit(args, repo, forwarded_args)
    return run_forwarded(args.command, forwarded_args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
