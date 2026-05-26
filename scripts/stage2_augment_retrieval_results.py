"""Append MDocAgent-aligned Stage 2 preflight fields to retrieval results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.mdocagent_aligned_stage2 import augment_retrieval_results_file
from mdocnexus.stage2.stage2_sidecar_store import (
    build_record_key,
    build_stage2_preflight_sidecar,
    build_stage2_record_index,
    write_stage2_preflight_sidecar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a stage2 preflight block to original MDocAgent retrieval records."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--sidecar-dir", default=None)
    parser.add_argument("--compact-stage2", action="store_true")
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
    sidecar_dir = None
    if args.compact_stage2:
        if not args.sidecar_dir:
            raise RuntimeError("--compact-stage2 requires --sidecar-dir.")
        sidecar_dir = Path(args.sidecar_dir)
        records = compact_stage2_records(records, sidecar_dir)
        Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output,
                "num_records": len(records),
                "compact_stage2": bool(args.compact_stage2),
                "sidecar_dir": str(sidecar_dir) if sidecar_dir is not None else None,
                "will_call_api": False,
                "will_generate_artifact": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def compact_stage2_records(records: list[dict], sidecar_dir: Path) -> list[dict]:
    """Move detailed stage2 preflight fields to sidecars and keep compact indexes."""

    compact_records = []
    for record_index, record in enumerate(records):
        stage2_preflight = record.get("stage2", {})
        if not isinstance(stage2_preflight, dict):
            compact_records.append(dict(record))
            continue
        record_key = build_record_key(record, record_index)
        preflight_ref = sidecar_dir / f"{record_key}.json"
        sidecar = build_stage2_preflight_sidecar(record, stage2_preflight, record_key=record_key)
        write_stage2_preflight_sidecar(sidecar, preflight_ref)
        compact_record = dict(record)
        compact_record["stage2"] = build_stage2_record_index(
            record=record,
            stage2_preflight=stage2_preflight,
            preflight_ref=preflight_ref,
            record_index=record_index,
        )
        compact_records.append(compact_record)
    return compact_records


if __name__ == "__main__":
    main()
