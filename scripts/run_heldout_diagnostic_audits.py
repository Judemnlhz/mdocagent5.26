#!/usr/bin/env python3
"""Unified entrypoint for heldout diagnostic audit runners.

This wrapper keeps the R043-R045 and R053-R056 diagnostic workflow discoverable without
duplicating the implementation in each focused runner. It deliberately excludes
R041/R042 one-off attribution scripts; their final reports are retained under
``outputs/heldout``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


RUNNERS = {
    "r043": "run_r043_contrastive_prompt_exposure.py",
    "r044": "run_r044_small_contrastive_provider.py",
    "r045": "run_r045_support_rubric.py",
    "r053": "run_r053_question_aware_scaffold.py",
    "r054": "run_r054_guarded_selector_repair.py",
    "r055": "run_r055_guarded_prompt_provider.py",
    "r056": "run_r056_guarded_scaffold_audit.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(RUNNERS), help="Diagnostic stage to run.")
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected stage runner. Use -- before forwarded options.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script_path = Path(__file__).resolve().parent / RUNNERS[args.stage]
    forwarded = args.runner_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = [sys.executable, str(script_path), *forwarded]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())