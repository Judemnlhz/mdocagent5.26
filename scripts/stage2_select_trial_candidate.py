"""Select one Stage 2 trial candidate from MDocAgent-aligned stage2 JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.mdocagent_aligned_stage2 import select_trial_candidate_from_stage2_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a deterministic single-page trial candidate from stage2-augmented JSON."
    )
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = select_trial_candidate_from_stage2_file(
        stage2_json=args.stage2_json,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
