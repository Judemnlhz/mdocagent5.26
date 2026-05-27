"""Generate offline Stage 2 cross-document quality and modality audit reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.crossdoc_quality_audit import (
    audit_crossdoc_batch_with_options,
    write_audit_json,
    write_page_quality_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an existing Stage 2 artifact batch offline.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--stage2-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    report = audit_crossdoc_batch_with_options(
        batch_dir=args.batch_dir,
        stage2_json=args.stage2_json,
    )
    write_audit_json(report, args.output_json)
    write_page_quality_csv(report, args.output_csv)
    public_report = {key: value for key, value in report.items() if not key.startswith("_")}
    print(json.dumps(public_report, ensure_ascii=False, indent=2))


def validate_inputs(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
    artifact_store_dir = batch_dir / "artifact_stores"
    if not artifact_store_dir.is_dir():
        raise FileNotFoundError(f"Artifact store directory does not exist: {artifact_store_dir}")
    if args.stage2_json is not None and not Path(args.stage2_json).is_file():
        raise FileNotFoundError(f"Stage 2 JSON does not exist: {args.stage2_json}")

if __name__ == "__main__":
    main()
