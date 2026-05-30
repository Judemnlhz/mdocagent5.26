#!/usr/bin/env python3
"""Audit Stage 2 real-provider multimodal smoke outputs without API calls."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_OUTPUT_DIR = "outputs/stage2_doc/"

REQUIRED_MANIFEST_FIELDS = [
    "schema_version",
    "compiler_mode",
    "provider_modes",
    "image_payload_modes",
    "call_log_hash",
    "quality_report_hash",
    "created_by_script",
    "command_args",
]
REQUIRED_QUALITY_FIELDS = [
    "pages_attempted",
    "pages_with_image_payload",
    "pages_without_image_payload",
    "image_payload_rate",
    "visual_artifact_count",
    "figure_artifact_count",
    "caption_artifact_count",
    "table_artifact_count",
    "empty_response_count",
    "parse_failure_count",
    "schema_failure_count",
    "anchor_failure_count",
    "provider_call_success_count",
    "provider_call_failed_count",
    "json_parse_success_count",
    "schema_valid_artifact_count",
    "anchored_artifact_count",
    "discarded_artifact_count",
]
REQUIRED_CALL_LOG_FIELDS = [
    "call_id",
    "doc_id",
    "page_id",
    "provider_mode",
    "modality_route",
    "image_payload_sent",
    "image_payload_mode",
    "prompt_hash",
    "response_schema_version",
    "call_succeeded",
    "parsed_artifact_count",
    "discarded_artifact_count",
    "timestamp_utc",
]
PUBLIC_FILENAMES = [
    "manifest.json",
    "quality_report.json",
    "call_log.jsonl",
    "artifacts.jsonl",
    "discard.jsonl",
]
SAFE_FORBIDDEN_SUBSTRING_KEYS = {
    "no_public_api_keys",
    "no_public_raw_response",
    "no_public_base64_payload",
    "public_raw_outputs_written",
    "no_public_local_paths",
}
TOKEN_ALLOWED_KEYS = {"token_count", "input_tokens", "output_tokens", "max_tokens", "tokenizer"}
TOKEN_ALLOWED_STRING_FRAGMENTS = ("token_count", "input_tokens", "output_tokens", "max_tokens", "tokenizer")
TOKEN_FORBIDDEN_KEY_PATTERNS = ("api_token", "access_token", "secret_token")
STRING_FORBIDDEN_PATTERNS = (
    "data:image",
    "file://",
    "/home/",
    "api_key",
    "raw_response",
    "raw_output",
    "raw_outputs",
    "provider_response",
    "local_path",
    "absolute_path",
    "image_path",
    "secret",
)
KEY_FORBIDDEN_PATTERNS = (
    "api_key",
    "raw_response",
    "raw_output",
    "raw_outputs",
    "provider_response",
    "local_path",
    "absolute_path",
    "image_path",
    "secret",
)
BASE64_RE = re.compile(r"^[A-Za-z0-9+/\s]{120,}={0,2}$")
ABSOLUTE_PATH_RE = re.compile(r"(^|[\s\"'=:(\[])(/(?:home|tmp|var|root|users|mnt|opt)/|[a-z]:[\\/])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 2 real-provider smoke public outputs.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-large-real-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, passed = audit_output_dir(Path(args.output_dir), allow_large_real_smoke=bool(args.allow_large_real_smoke))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


def audit_output_dir(output_dir: Path, allow_large_real_smoke: bool = False) -> Tuple[Dict[str, Any], bool]:
    failures: List[str] = []
    warnings: List[str] = []
    missing_required_fields: List[str] = []
    public_leakage_violations: List[str] = []
    files_checked: List[str] = []

    manifest_path = output_dir / "manifest.json"
    quality_path = output_dir / "quality_report.json"
    call_log_path = output_dir / "call_log.jsonl"
    artifacts_path = output_dir / "artifacts.jsonl"
    discard_path = output_dir / "discard.jsonl"

    manifest: Dict[str, Any] = {}
    quality: Dict[str, Any] = {}
    call_rows: List[Dict[str, Any]] = []

    if not manifest_path.is_file():
        failures.append("missing_required_file:manifest.json")
    else:
        manifest = _read_json_file(manifest_path, failures)
        files_checked.append("manifest.json")

    if not quality_path.is_file():
        failures.append("missing_required_file:quality_report.json")
    else:
        quality = _read_json_file(quality_path, failures)
        files_checked.append("quality_report.json")

    calls_attempted = _provider_calls_attempted(manifest, quality)
    if calls_attempted and not call_log_path.is_file():
        failures.append("missing_required_file:call_log.jsonl")
    if call_log_path.is_file():
        call_rows = _read_jsonl_file(call_log_path, failures)
        files_checked.append("call_log.jsonl")

    if artifacts_path.is_file():
        files_checked.append("artifacts.jsonl")
    if discard_path.is_file():
        files_checked.append("discard.jsonl")

    missing_required_fields.extend(_validate_manifest_fields(manifest, artifacts_path, discard_path))
    missing_required_fields.extend(_validate_quality_fields(quality))
    missing_required_fields.extend(_validate_call_log_fields(call_rows))

    failures.extend(_validate_hashes(manifest, manifest_path, quality_path, call_log_path, artifacts_path, discard_path))
    failures.extend(_validate_image_payload_consistency(call_rows, output_dir))
    real_failures, real_warnings = _validate_real_mode_safety(
        manifest,
        quality,
        output_dir,
        allow_large_real_smoke=allow_large_real_smoke,
    )
    failures.extend(real_failures)
    warnings.extend(real_warnings)
    failures.extend(_validate_private_debug_safety(manifest, output_dir))

    for filename in PUBLIC_FILENAMES:
        path = output_dir / filename
        if not path.is_file():
            continue
        for violation in _scan_public_file(path):
            public_leakage_violations.append(f"{filename}:{violation}")

    if public_leakage_violations:
        failures.append("public_leakage_violations_present")
    if missing_required_fields:
        failures.append("missing_required_fields_present")

    provider_modes = list(manifest.get("provider_modes", [])) if isinstance(manifest.get("provider_modes"), list) else []
    image_payload_modes = list(manifest.get("image_payload_modes", [])) if isinstance(manifest.get("image_payload_modes"), list) else []
    status = "pass" if not failures else "fail"
    report = {
        "files_checked": files_checked,
        "call_log_rows": len(call_rows),
        "provider_modes": provider_modes,
        "image_payload_modes": image_payload_modes,
        "pages_attempted": quality.get("pages_attempted"),
        "pages_with_image_payload": quality.get("pages_with_image_payload"),
        "image_payload_rate": quality.get("image_payload_rate"),
        "public_leakage_violations": public_leakage_violations,
        "missing_required_fields": missing_required_fields,
        "warnings": warnings,
        "failures": failures,
        "status": status,
    }
    return report, status == "pass"


def _read_json_file(path: Path, failures: List[str]) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid_json:{path.name}:{type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"invalid_json_object:{path.name}")
        return {}
    return value


def _read_jsonl_file(path: Path, failures: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception as exc:
            failures.append(f"invalid_jsonl:{path.name}:{line_number}:{type(exc).__name__}")
            continue
        if not isinstance(value, dict):
            failures.append(f"invalid_jsonl_row_object:{path.name}:{line_number}")
            continue
        rows.append(value)
    return rows


def _provider_calls_attempted(manifest: Dict[str, Any], quality: Dict[str, Any]) -> bool:
    if int(manifest.get("provider_call_count") or 0) > 0:
        return True
    if int(quality.get("pages_attempted") or quality.get("num_pages_attempted") or 0) > 0:
        return True
    provider_modes = manifest.get("provider_modes")
    return isinstance(provider_modes, list) and bool(provider_modes)


def _validate_manifest_fields(manifest: Dict[str, Any], artifacts_path: Path, discard_path: Path) -> List[str]:
    missing = [f"manifest.{field}" for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if "input_hash" not in manifest and "input_hash_unavailable_reason" not in manifest:
        missing.append("manifest.input_hash_or_input_hash_unavailable_reason")
    if artifacts_path.is_file() and "artifacts_hash" not in manifest:
        missing.append("manifest.artifacts_hash")
    if discard_path.is_file() and "discard_hash" not in manifest:
        missing.append("manifest.discard_hash")
    if "forbidden_fields_checked" not in manifest and "no_public_leakage_checked" not in manifest:
        missing.append("manifest.forbidden_fields_checked_or_no_public_leakage_checked")
    return missing


def _validate_quality_fields(quality: Dict[str, Any]) -> List[str]:
    return [f"quality_report.{field}" for field in REQUIRED_QUALITY_FIELDS if field not in quality]


def _validate_call_log_fields(call_rows: List[Dict[str, Any]]) -> List[str]:
    missing: List[str] = []
    for index, row in enumerate(call_rows, start=1):
        for field in REQUIRED_CALL_LOG_FIELDS:
            if field not in row:
                missing.append(f"call_log[{index}].{field}")
        if "image_sha256" not in row and "image_sha256_unavailable_reason" not in row:
            missing.append(f"call_log[{index}].image_sha256_or_image_sha256_unavailable_reason")
    return missing


def _validate_hashes(
    manifest: Dict[str, Any],
    manifest_path: Path,
    quality_path: Path,
    call_log_path: Path,
    artifacts_path: Path,
    discard_path: Path,
) -> List[str]:
    _ = manifest_path
    failures: List[str] = []
    expected_files = [
        ("quality_report_hash", quality_path),
        ("call_log_hash", call_log_path),
        ("artifacts_hash", artifacts_path),
        ("discard_hash", discard_path),
    ]
    for field_name, path in expected_files:
        if not path.is_file() or field_name not in manifest:
            continue
        recorded = str(manifest.get(field_name))
        if recorded in {"", "missing"}:
            failures.append(f"hash_unavailable:{field_name}")
            continue
        actual = _file_sha256(path)
        if recorded != actual:
            failures.append(f"hash_mismatch:{field_name}")
    return failures


def _validate_image_payload_consistency(call_rows: List[Dict[str, Any]], output_dir: Path) -> List[str]:
    failures: List[str] = []
    for index, row in enumerate(call_rows, start=1):
        mode = str(row.get("image_payload_mode"))
        sent = row.get("image_payload_sent") is True
        if sent and mode not in {"image_url", "base64"}:
            failures.append(f"call_log[{index}].image_payload_sent_requires_image_url_or_base64")
        if sent and not row.get("image_sha256"):
            failures.append(f"call_log[{index}].image_payload_sent_requires_image_sha256")
        if mode == "none" and sent:
            failures.append(f"call_log[{index}].image_payload_mode_none_requires_sent_false")
    for path in _public_paths(output_dir):
        for field_path, key, value in _iter_json_keys(path):
            if key == "image_url":
                failures.append(f"public_image_url_field:{path.name}:{field_path}")
            if isinstance(value, str) and _looks_like_base64_payload(value):
                failures.append(f"public_base64_payload:{path.name}:{field_path}")
    return failures


def _validate_real_mode_safety(
    manifest: Dict[str, Any],
    quality: Dict[str, Any],
    output_dir: Path,
    allow_large_real_smoke: bool,
) -> Tuple[List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []
    provider_modes = manifest.get("provider_modes") if isinstance(manifest.get("provider_modes"), list) else []
    if "real" not in provider_modes:
        return failures, warnings
    command_args = manifest.get("command_args") if isinstance(manifest.get("command_args"), dict) else {}
    if command_args.get("enable_real_api") is not True:
        failures.append("real_mode_requires_command_args.enable_real_api_true")
    if command_args.get("run_real_trial") is not True:
        failures.append("real_mode_requires_command_args.run_real_trial_true")
    max_pages = command_args.get("max_pages_real", command_args.get("max_pages"))
    try:
        if max_pages is None or int(max_pages) < 1:
            raise ValueError
    except (TypeError, ValueError):
        failures.append("real_mode_requires_finite_max_pages_or_max_pages_real")
    pages_attempted = int(quality.get("pages_attempted") or 0)
    if pages_attempted > 10 and not allow_large_real_smoke:
        warnings.append("real_mode_pages_attempted_gt_10")
    if (output_dir / "raw_outputs.jsonl").exists():
        failures.append("real_mode_public_raw_outputs_jsonl_present")
    return failures, warnings


def _validate_private_debug_safety(manifest: Dict[str, Any], output_dir: Path) -> List[str]:
    failures: List[str] = []
    public_raw_outputs = output_dir / "raw_outputs.jsonl"
    if public_raw_outputs.exists():
        failures.append("public_raw_outputs_jsonl_present")
    if manifest.get("private_debug_dir_recorded_as_public") is True:
        failures.append("private_debug_dir_recorded_as_public_true")
    command_args = manifest.get("command_args") if isinstance(manifest.get("command_args"), dict) else {}
    private_debug_dir = command_args.get("private_debug_dir")
    if private_debug_dir:
        debug_path = Path(str(private_debug_dir)).resolve()
        public_root = output_dir.resolve()
        if debug_path == public_root or public_root in debug_path.parents:
            failures.append("private_debug_dir_inside_public_output")
        if "outputs_private" not in debug_path.parts:
            failures.append("private_debug_dir_not_under_outputs_private")
    return failures


def _scan_public_file(path: Path) -> List[str]:
    violations: List[str] = []
    if path.suffix == ".jsonl":
        for row_index, row in enumerate(_safe_read_jsonl(path), start=1):
            violations.extend(_scan_value(row, f"row[{row_index}]"))
    else:
        violations.extend(_scan_value(_safe_read_json(path), "$"))
    return violations


def _scan_value(value: Any, field_path: str) -> List[str]:
    violations: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            violations.extend(_scan_key(key_text, f"{field_path}.{key_text}"))
            violations.extend(_scan_value(child, f"{field_path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_scan_value(child, f"{field_path}[{index}]"))
    elif isinstance(value, str):
        violations.extend(_scan_string(value, field_path))
    return violations


def _scan_key(key: str, field_path: str) -> List[str]:
    key_lower = key.lower()
    if key_lower in SAFE_FORBIDDEN_SUBSTRING_KEYS or key_lower in TOKEN_ALLOWED_KEYS:
        return []
    violations: List[str] = []
    for pattern in KEY_FORBIDDEN_PATTERNS:
        if pattern in key_lower:
            violations.append(f"forbidden_key:{field_path}:{pattern}")
    if any(pattern in key_lower for pattern in TOKEN_FORBIDDEN_KEY_PATTERNS):
        violations.append(f"forbidden_key:{field_path}:token")
    elif "token" in key_lower and key_lower not in TOKEN_ALLOWED_KEYS:
        violations.append(f"forbidden_key:{field_path}:token")
    return violations


def _scan_string(value: str, field_path: str) -> List[str]:
    lowered = value.lower()
    violations: List[str] = []
    for pattern in STRING_FORBIDDEN_PATTERNS:
        if pattern in lowered:
            violations.append(f"forbidden_string:{field_path}:{pattern}")
    if any(pattern in lowered for pattern in TOKEN_FORBIDDEN_KEY_PATTERNS):
        violations.append(f"forbidden_string:{field_path}:token")
    elif _contains_forbidden_token_text(lowered):
        violations.append(f"forbidden_string:{field_path}:token")
    if ABSOLUTE_PATH_RE.search(lowered):
        violations.append(f"local_absolute_path:{field_path}")
    if _looks_like_base64_payload(value):
        violations.append(f"base64_payload:{field_path}")
    return violations


def _contains_forbidden_token_text(lowered: str) -> bool:
    sanitized = lowered
    for allowed in TOKEN_ALLOWED_STRING_FRAGMENTS:
        sanitized = sanitized.replace(allowed, "")
    return "token" in sanitized


def _looks_like_base64_payload(value: str) -> bool:
    stripped = "".join(value.split())
    if "base64," in value.lower():
        return True
    if len(stripped) < 120 or not BASE64_RE.match(stripped):
        return False
    try:
        base64.b64decode(stripped, validate=True)
    except Exception:
        return False
    return True


def _iter_json_keys(path: Path) -> Iterable[Tuple[str, str, Any]]:
    if path.suffix == ".jsonl":
        for row_index, row in enumerate(_safe_read_jsonl(path), start=1):
            yield from _iter_keys(row, f"row[{row_index}]")
    else:
        yield from _iter_keys(_safe_read_json(path), "$")


def _iter_keys(value: Any, field_path: str) -> Iterable[Tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{field_path}.{key}"
            yield child_path, str(key), child
            yield from _iter_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_keys(child, f"{field_path}[{index}]")


def _public_paths(output_dir: Path) -> List[Path]:
    return [output_dir / filename for filename in PUBLIC_FILENAMES if (output_dir / filename).is_file()]


def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_jsonl(path: Path) -> List[Any]:
    rows: List[Any] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()