#!/usr/bin/env python3
"""Audit public outputs for gold/evaluation and private payload leakage."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_SCAN_DIRS = [
    "outputs/stage2_doc",
    "outputs/stage3_doc_artifact_retrieval",
    "outputs/stage4/evidence_graph",
]
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".log"}
SAFE_FIELD_NAMES = {
    "no_answer_generation",
    "no_gold_fields_used",
    "no_public_raw_response",
    "no_public_base64_payload",
    "no_public_local_paths",
    "no_public_api_keys",
    "public_raw_outputs_written",
}
FORBIDDEN_FIELD_NAMES = {
    "answer",
    "answers",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
    "raw_output",
    "raw_response",
    "api_key",
    "local_path",
    "absolute_path",
    "image_path",
}
FORBIDDEN_CONTENT_FRAGMENTS = {
    "file://",
    "/home/",
    "data:image",
}
FORBIDDEN_BASE64_KEYS = {"base64", "image_base64", "base64_payload", "image_payload_base64"}
BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{80,}={0,2})")


def run_audit(scan_dirs: Iterable[str | Path] = DEFAULT_SCAN_DIRS) -> dict[str, Any]:
    files_checked: list[str] = []
    violations: list[dict[str, str]] = []
    warnings: list[str] = []
    for scan_dir in scan_dirs:
        root = Path(scan_dir)
        if not root.exists():
            warnings.append(f"missing_dir:{root}")
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name.startswith(".") or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            files_checked.append(str(path))
            audit_file(path, violations)
    return {
        "files_checked": len(files_checked),
        "public_leakage_violations": violations,
        "warnings": warnings,
        "status": "fail" if violations else "pass",
    }


def audit_file(path: Path, violations: list[dict[str, str]]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        violations.append({"path": str(path), "reason": "non_utf8_public_text"})
        return
    for fragment in FORBIDDEN_CONTENT_FRAGMENTS:
        if fragment in text.lower():
            violations.append({"path": str(path), "reason": f"forbidden_content:{fragment}"})
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                audit_plain_text(path, text, violations)
                return
            audit_value(value, str(path), violations, location=f"line:{line_number}")
    elif path.suffix.lower() == ".json":
        try:
            audit_value(json.loads(text), str(path), violations, location="json")
        except json.JSONDecodeError:
            audit_plain_text(path, text, violations)
    else:
        audit_plain_text(path, text, violations)


def audit_value(value: Any, path: str, violations: list[dict[str, str]], location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text not in SAFE_FIELD_NAMES and key_text in FORBIDDEN_FIELD_NAMES:
                violations.append({"path": path, "location": location, "reason": f"forbidden_field:{key_text}"})
            if key_text in FORBIDDEN_BASE64_KEYS:
                violations.append({"path": path, "location": location, "reason": f"forbidden_field:{key_text}"})
            if key_text in {"api_token", "access_token", "secret_token"}:
                violations.append({"path": path, "location": location, "reason": f"forbidden_field:{key_text}"})
            audit_value(child, path, violations, location=f"{location}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            audit_value(child, path, violations, location=f"{location}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in FORBIDDEN_CONTENT_FRAGMENTS:
            if fragment in lowered:
                violations.append({"path": path, "location": location, "reason": f"forbidden_content:{fragment}"})
        if looks_like_base64_payload(value):
            violations.append({"path": path, "location": location, "reason": "forbidden_content:base64_payload"})


def audit_plain_text(path: Path, text: str, violations: list[dict[str, str]]) -> None:
    lowered = text.lower()
    for field in sorted(FORBIDDEN_FIELD_NAMES):
        if field in SAFE_FIELD_NAMES:
            continue
        if re.search(rf'(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])', lowered):
            violations.append({"path": str(path), "reason": f"forbidden_text:{field}"})
    if looks_like_base64_payload(text):
        violations.append({"path": str(path), "reason": "forbidden_text:base64_payload"})


def looks_like_base64_payload(value: str) -> bool:
    if "sha256" in value.lower():
        return False
    for match in BASE64_RE.findall(value):
        try:
            decoded = base64.b64decode(match, validate=True)
        except Exception:
            continue
        if len(decoded) >= 48:
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit public outputs for gold and private payload leakage.")
    parser.add_argument("--scan-dir", action="append", dest="scan_dirs", help="Directory to scan. Repeatable.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(args.scan_dirs or DEFAULT_SCAN_DIRS)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
