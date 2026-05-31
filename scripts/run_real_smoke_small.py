#!/usr/bin/env python3
"""Prepare a tightly bounded Stage 2 real-provider smoke run.

The default mode is dry-run and only prints the commands. Real execution
requires all three switches: --execute, --enable-real-api, and --run-real-trial.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.common.model_config import QWEN3VL_CONFIG, QWEN3VL_MODEL_ID, load_model_config, model_id_from_config

DEFAULT_OUTPUT_DIR = "outputs/stage2_doc_real_smoke_qwen3vl"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
MAX_PAGES_TOTAL_CAP = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or print a small bounded real-provider Stage 2 smoke command.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing. This is the default.")
    parser.add_argument("--enable-real-api", action="store_true")
    parser.add_argument("--run-real-trial", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--input", "--stage2-json", dest="stage2_json", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-config", default=QWEN3VL_CONFIG)
    parser.add_argument("--provider", default="real", choices=("real",))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-pages-total", type=int, default=3)
    parser.add_argument("--max-pages-per-call", type=int, default=1)
    parser.add_argument("--image-payload-mode", choices=("image_url", "base64", "none"), default="image_url")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if int(args.max_pages_total) < 1:
        raise RuntimeError("--max-pages-total must be at least 1.")
    if int(args.max_pages_total) > MAX_PAGES_TOTAL_CAP:
        raise RuntimeError("--max-pages-total must not exceed 5 for real smoke small.")
    if int(args.max_pages_per_call) != 1:
        raise RuntimeError("--max-pages-per-call must be 1 for real smoke small.")
    if args.execute and (not args.enable_real_api or not args.run_real_trial):
        raise RuntimeError("Real smoke execution requires both --enable-real-api and --run-real-trial.")
    config = load_model_config(args.model_config)
    model_id = model_id_from_config(config)
    if model_id != QWEN3VL_MODEL_ID:
        raise RuntimeError("Real smoke small requires Qwen/Qwen3-VL-8B-Instruct model config.")


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    max_pages_total = int(args.max_pages_total)
    model_config = load_model_config(args.model_config)
    model_name = args.model_name or model_id_from_config(model_config) or QWEN3VL_MODEL_ID
    stage2_cmd = [
        "python3",
        "scripts/stage2.py",
        "doc-compile",
        "--provider",
        "real",
        "--enable-real-api",
        "--run-real-trial",
        "--max-pages-total",
        str(max_pages_total),
        "--max-pages",
        str(max_pages_total),
        "--max-pages-real-cap",
        str(MAX_PAGES_TOTAL_CAP),
        "--max-pages-per-call",
        "1",
        "--max-docs",
        str(max_pages_total),
        "--max-pages-per-doc",
        "1",
        "--extract-root",
        args.extract_root,
        "--output-dir",
        args.output_dir,
        "--model-config",
        args.model_config,
        "--model-name",
        model_name,
        "--image-payload-mode",
        args.image_payload_mode,
    ]
    if args.stage2_json:
        stage2_cmd.extend(["--input", args.stage2_json])
    if args.config:
        stage2_cmd.extend(["--config", args.config])
    audit_cmd = [
        "python3",
        "scripts/audit_real_provider_smoke.py",
        "--output-dir",
        args.output_dir,
    ]
    audit_model_cmd = [
        "python3",
        "scripts/audit_model_configs.py",
        "--stage2-dir",
        args.output_dir,
        "--output",
        "outputs/audits/model_config_audit_report.json",
    ]
    audit_no_gold_cmd = ["python3", "scripts/audit_no_gold_leakage.py", "--scan-dir", args.output_dir]
    audit_repro_cmd = ["python3", "scripts/audit_reproducibility.py", "--stage2-dir", args.output_dir]
    return [stage2_cmd, audit_cmd, audit_model_cmd, audit_no_gold_cmd, audit_repro_cmd]


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    row = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    repo = Path(__file__).resolve().parents[1]
    commands = build_commands(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "requires_execute": True,
                    "requires_enable_real_api": True,
                    "requires_run_real_trial": True,
                    "max_pages_total_cap": MAX_PAGES_TOTAL_CAP,
                    "commands": commands,
                    "output_dir": args.output_dir,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    results = [run_command(command, repo) for command in commands]
    print(json.dumps({"will_execute": True, "results": results, "output_dir": args.output_dir}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
