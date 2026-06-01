#!/usr/bin/env python3
"""Check that adapted retrieval records remain MDocAgent-loader compatible.

This script is intentionally model-free: it does not call predict.py, does not
generate answers, and does not read gold labels. It validates the public
retrieval-record schema expected by BaseDataset.load_sample_retrieval_data and,
when extracted page files are available, exercises the original loader method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydatasets.base_dataset import BaseDataset
from mdocnexus.integration.mdocagent_adapter import load_mdocagent_retrieval_records


FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}

PRIVATE_TEXT_FRAGMENTS = ("file://", "/home/", "data:image", "api_key", "secret")
REQUIRED_FIELDS = ("doc_id", "question")
DEFAULT_TOP_K_CHECKED = (1, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check MDocAgent-compatible adapted retrieval records.")
    parser.add_argument("--input-retrieval", required=True, help="Adapter output JSON file.")
    parser.add_argument("--output-report", default=None, help="Path for compatibility_report.json.")
    parser.add_argument("--dataset-name", default="MMLongBench")
    parser.add_argument("--extract-path", default="tmp/MMLongBench")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for loader execution; 0 means all records.")
    parser.add_argument("--top-k", default="1,4", help="Comma-separated top-k values to check.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    top_k_checked = parse_top_k(args.top_k)
    records = load_mdocagent_retrieval_records(args.input_retrieval)
    checked_records = records[: args.max_records] if args.max_records and args.max_records > 0 else records

    missing_required_fields: list[str] = []
    forbidden_fields_found: list[str] = []
    loader_errors: list[str] = []

    for index, record in enumerate(checked_records):
        for field in REQUIRED_FIELDS:
            if record.get(field) in (None, ""):
                missing_required_fields.append(f"records[{index}].{field}")
        if not has_retrieval_fields(record):
            missing_required_fields.append(f"records[{index}].retrieval_page_fields")
        find_forbidden(record, f"records[{index}]", forbidden_fields_found)

    for top_k in top_k_checked:
        dataset = BaseDataset(build_dataset_config(args, top_k))
        for index, record in enumerate(checked_records):
            try:
                dataset.load_sample_retrieval_data(record)
            except Exception as exc:  # noqa: BLE001 - report loader compatibility, do not mask.
                loader_errors.append(f"top_k={top_k}:records[{index}]:{type(exc).__name__}:{exc}")
                if len(loader_errors) >= 20:
                    break
        if len(loader_errors) >= 20:
            break

    loader_compatible = not missing_required_fields and not forbidden_fields_found and not loader_errors
    report = {
        "input_retrieval_path": str(Path(args.input_retrieval)),
        "num_records": len(records),
        "top_k_checked": top_k_checked,
        "loader_compatible": loader_compatible,
        "missing_required_fields": missing_required_fields[:100],
        "forbidden_fields_found": forbidden_fields_found[:100],
        "loader_errors": loader_errors[:100],
        "status": "pass" if loader_compatible else "fail",
    }

    output_report = Path(args.output_report) if args.output_report else Path(args.input_retrieval).with_name("compatibility_report.json")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if loader_compatible else 1


def build_dataset_config(args: argparse.Namespace, top_k: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=args.dataset_name,
        top_k=top_k,
        question_key="question",
        gt_key="answer",
        page_id_key="page_ids",
        truncate_len=None,
        max_page=1000,
        max_character_per_page=100000,
        use_mix=False,
        r_text_key="text-top-10-question",
        r_image_key="image-top-10-question",
        r_mix_key="mix-top-10-question",
        data_dir=f"./data/{args.dataset_name}",
        result_dir=f"./results/{args.dataset_name}/adapter_compatibility_check",
        extract_path=args.extract_path,
        document_path=f"./data/{args.dataset_name}/documents",
        sample_path=f"./data/{args.dataset_name}/samples.json",
        sample_with_retrieval_path=args.input_retrieval,
    )


def parse_top_k(value: str) -> list[int]:
    if not value:
        return list(DEFAULT_TOP_K_CHECKED)
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    return parsed or list(DEFAULT_TOP_K_CHECKED)


def has_retrieval_fields(record: dict[str, Any]) -> bool:
    return any(
        key.startswith(("text-top-", "image-top-", "mix-top-")) and not key.endswith("_score")
        for key in record
    )


def find_forbidden(value: Any, path: str, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if lowered in FORBIDDEN_KEYS or lowered.startswith("gold_"):
                found.append(f"{path}.{key_text}")
            find_forbidden(child, f"{path}.{key_text}", found)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            find_forbidden(child, f"{path}[{index}]", found)
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in PRIVATE_TEXT_FRAGMENTS:
            if fragment in lowered:
                found.append(f"{path}:{fragment}")


if __name__ == "__main__":
    raise SystemExit(main())
