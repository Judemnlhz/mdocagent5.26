#!/usr/bin/env python3
"""Audit reproducibility hashes for Stage 2, Stage 3, and Stage 4 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STAGE2_DIR = "outputs/stage2_doc"
DEFAULT_STAGE3_DIR = "outputs/stage3_doc_artifact_retrieval"
DEFAULT_STAGE4_DIR = "outputs/stage4/evidence_graph"


HASH_SPECS = {
    "stage2": {
        "artifacts_hash": "artifacts.jsonl",
        "call_log_hash": "call_log.jsonl",
        "discard_hash": "discard.jsonl",
        "quality_report_hash": "quality_report.json",
    },
    "stage3": {
        "retrieval_hash": "retrieval.jsonl",
        "quality_report_hash": "quality_report.json",
    },
    "stage4": {
        "nodes_hash": "nodes.jsonl",
        "edges_hash": "edges.jsonl",
        "debug_edges_hash": "debug_edges.jsonl",
        "quality_report_hash": "quality_report.json",
    },
}


def run_audit(stage2_dir: str | Path = DEFAULT_STAGE2_DIR, stage3_dir: str | Path = DEFAULT_STAGE3_DIR, stage4_dir: str | Path = DEFAULT_STAGE4_DIR, strict_missing: bool = False) -> dict[str, Any]:
    stages = {"stage2": Path(stage2_dir), "stage3": Path(stage3_dir), "stage4": Path(stage4_dir)}
    checked: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    warnings: list[str] = []
    for stage_name, root in stages.items():
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            warnings.append(f"missing_manifest:{stage_name}:{manifest_path}")
            continue
        manifest = read_json(manifest_path)
        specs = HASH_SPECS[stage_name]
        for field, relative_path in specs.items():
            target_path = root / relative_path
            if field not in manifest:
                message = f"missing_hash_field:{stage_name}:{field}"
                if strict_missing:
                    mismatches.append({"stage": stage_name, "field": field, "reason": message})
                else:
                    warnings.append(message)
                continue
            if not target_path.is_file():
                message = f"missing_hashed_file:{stage_name}:{relative_path}"
                if strict_missing:
                    mismatches.append({"stage": stage_name, "field": field, "reason": message})
                else:
                    warnings.append(message)
                continue
            expected = str(manifest[field])
            actual = canonical_file_hash(target_path)
            legacy = file_sha256(target_path)
            checked.append({"stage": stage_name, "field": field, "path": str(target_path), "hash": actual})
            if expected == actual:
                continue
            if stage_name == "stage2" and expected == legacy:
                warnings.append(f"legacy_file_sha256_accepted:{stage_name}:{field}")
                continue
            mismatches.append({"stage": stage_name, "field": field, "expected": expected, "actual": actual})
    return {
        "checked_hashes": checked,
        "num_checked_hashes": len(checked),
        "mismatches": mismatches,
        "warnings": warnings,
        "status": "fail" if mismatches else "pass",
    }


def canonical_file_hash(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return canonical_json_hash(read_jsonl(path))
    if path.suffix.lower() == ".json":
        return canonical_json_hash(read_json(path))
    return file_sha256(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def canonical_json_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    serialized = serialized.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit canonical reproducibility hashes.")
    parser.add_argument("--stage2-dir", default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--stage3-dir", default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--stage4-dir", default=DEFAULT_STAGE4_DIR)
    parser.add_argument("--strict-missing", action="store_true", help="Fail if a known hash field or file is missing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(args.stage2_dir, args.stage3_dir, args.stage4_dir, strict_missing=args.strict_missing)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
