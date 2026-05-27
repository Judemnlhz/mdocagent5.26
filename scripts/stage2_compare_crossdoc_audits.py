"""Compare baseline and refined Stage 2 cross-document audit reports."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.reports import compare_crossdoc_audits, write_refinement_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 2 cross-doc audit reports.")
    parser.add_argument("--baseline-audit", required=True)
    parser.add_argument("--refined-audit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_crossdoc_audits(args.baseline_audit, args.refined_audit)
    write_refinement_comparison(report, args.output_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
