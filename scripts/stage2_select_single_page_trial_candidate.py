"""Select an objective Stage 2 single-page real-trial candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.trial_candidate_selector import select_single_page_trial_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one legal single-page Stage 2 trial candidate without API calls."
    )
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def write_candidate_report(report: Dict[str, Any], output_report: str | Path) -> None:
    path = Path(output_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_step7b_command(args: argparse.Namespace, report: Dict[str, Any]) -> str | None:
    selected = report.get("selected")
    if not selected:
        return None
    return " ".join(
        [
            "python3",
            "scripts/run_stage2_real_single_page_trial.py",
            f"--config {args.config}",
            f"--sample-path {args.sample_path}",
            f"--extract-root {args.extract_root}",
            f"--candidate-report {args.output_report}",
            "--enable-real-api",
            "--run-real-trial",
            "--output-path outputs/stage2/real_single_page_trial/artifact_store.json",
        ]
    )


def main() -> None:
    args = parse_args()
    report = select_single_page_trial_candidate(
        sample_path=args.sample_path,
        dataset_name=args.dataset_name,
        extract_root=args.extract_root,
        config_path=args.config,
        max_records=args.max_records,
    )
    write_candidate_report(report, args.output_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    command = build_step7b_command(args, report)
    if command:
        print()
        print("Step 7B command, not executed:")
        print(command)


if __name__ == "__main__":
    main()
