"""Append MDocAgent-aligned Stage 2 preflight fields to retrieval results."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.mdocagent_aligned_stage2 import augment_retrieval_results_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a stage2 preflight block to original MDocAgent retrieval records."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = augment_retrieval_results_file(
        input_path=args.input,
        output_path=args.output,
        extract_root=args.extract_root,
        config_path=args.config,
        max_records=args.max_records,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "num_records": len(records),
                "schema": "compact_page_routes",
                "will_call_api": False,
                "will_generate_artifact": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
