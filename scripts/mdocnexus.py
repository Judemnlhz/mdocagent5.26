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
    "stage2-real-structured-gate": ["scripts/run_stage2_real_structured_gate.py"],
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

    adapt_parser = subparsers.add_parser("mdocagent-adapt", help="Write MDocAgent-compatible MDocNexus retrieval records.")
    adapt_parser.add_argument("--input-retrieval", required=True)
    adapt_parser.add_argument("--artifacts", required=True)
    adapt_parser.add_argument("--graph-dir", default=None)
    adapt_parser.add_argument("--output-retrieval", required=True)
    adapt_parser.add_argument(
        "--mode",
        choices=["original_only", "artifact_only", "original_plus_artifact", "graph_context"],
        required=True,
    )
    adapt_parser.add_argument("--top-k", type=int, default=4)
    adapt_parser.add_argument("--lambda-weight", type=float, default=0.5)
    adapt_parser.add_argument(
        "--expansion-mode",
        choices=["page_neighborhood", "source_anchor_neighborhood", "direct_structural"],
        default="page_neighborhood",
    )
    adapt_parser.add_argument("--manifest-path", required=True)
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


def run_mdocagent_adapt(args: argparse.Namespace, repo: Path) -> int:
    sys.path.insert(0, str(repo))
    from mdocnexus.integration.mdocagent_adapter import (
        build_mdocagent_adapter_manifest,
        load_artifacts_by_page,
        load_mdocagent_retrieval_records,
        rerank_pages_with_artifacts,
        select_pages_with_graph,
        write_manifest,
        write_mdocagent_compatible_records,
    )

    records = load_mdocagent_retrieval_records(args.input_retrieval)
    artifacts_by_page = load_artifacts_by_page(args.artifacts)
    if args.mode == "graph_context":
        if not args.graph_dir:
            raise SystemExit("--graph-dir is required when --mode graph_context")
        adapted = select_pages_with_graph(
            records,
            artifacts_by_page,
            args.graph_dir,
            top_k=args.top_k,
            expansion_mode=args.expansion_mode,
        )
    else:
        adapted = rerank_pages_with_artifacts(
            records,
            artifacts_by_page,
            top_k=args.top_k,
            mode=args.mode,
            lambda_weight=args.lambda_weight,
        )
    write_mdocagent_compatible_records(adapted, args.output_retrieval)
    manifest = build_mdocagent_adapter_manifest(
        mode=args.mode,
        top_k=args.top_k,
        lambda_weight=args.lambda_weight,
        input_retrieval=args.input_retrieval,
        artifacts=args.artifacts,
        graph_dir=args.graph_dir,
        output_retrieval=args.output_retrieval,
        expansion_mode=args.expansion_mode if args.mode == "graph_context" else None,
        command_args=args,
        repo_root=repo,
    )
    write_manifest(manifest, args.manifest_path)
    print(
        json.dumps(
            {
                "output_retrieval": str(Path(args.output_retrieval)),
                "manifest_path": str(Path(args.manifest_path)),
                "output_hash": manifest["output_hash"],
                "status": "prepared",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.command == "audit":
        return run_audit(args, repo, forwarded_args)
    if args.command == "mdocagent-adapt":
        return run_mdocagent_adapt(args, repo)
    return run_forwarded(args.command, forwarded_args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
