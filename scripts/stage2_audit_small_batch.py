"""Offline audit for Stage 2 small-batch artifact outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.artifact_quality_audit import (
    audit_batch_artifact_outputs,
    write_audit_csv,
    write_audit_json,
)
from mdocnexus.stage2.mdocagent_compat import read_json_or_jsonl_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit existing Stage 2 small-batch artifacts without API calls.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Read the aligned Stage 2 JSON only to confirm it exists and is parseable;
    # do not use gold/eval fields for quality decisions.
    _ = len(read_json_or_jsonl_records(args.stage2_json))
    audit = audit_batch_artifact_outputs(args.batch_dir)
    write_audit_json(audit, args.output_json)
    write_audit_csv(audit, args.output_csv)
    print(json.dumps({key: value for key, value in audit.items() if key != "artifact_store_audits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
