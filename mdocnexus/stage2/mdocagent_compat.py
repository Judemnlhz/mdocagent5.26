"""Compatibility helpers for using Stage 2 with the original MDocAgent layout."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api_config import ApiRunConfig


def normalize_doc_name_for_mdocagent(doc_id: str) -> str:
    """Return the MDocAgent extracted page stem by removing only a .pdf suffix."""

    return doc_id[:-4] if doc_id.endswith(".pdf") else doc_id


def build_mdocagent_extract_paths(
    extract_root: str | Path,
    doc_id: str,
    page_index: int,
) -> Dict[str, Any]:
    """Build original MDocAgent page paths first, followed by legacy fallbacks."""

    root = Path(extract_root)
    page_index = int(page_index)
    doc_name = normalize_doc_name_for_mdocagent(doc_id)
    text_candidates = [
        root / f"{doc_name}_{page_index}.txt",
        root / f"{doc_name}_{page_index:03d}.txt",
        root / "texts" / f"{doc_name}_{page_index}.txt",
        root / "texts" / f"{doc_name}_{page_index:03d}.txt",
        root / "texts" / f"page_{page_index}.txt",
        root / "texts" / f"page_{page_index:03d}.txt",
    ]
    image_candidates = [
        root / f"{doc_name}_{page_index}.png",
        root / f"{doc_name}_{page_index:03d}.png",
        root / "images" / f"{doc_name}_{page_index}.png",
        root / "images" / f"{doc_name}_{page_index:03d}.png",
        root / "images" / f"page_{page_index}.png",
        root / "images" / f"page_{page_index:03d}.png",
    ]
    return {
        "doc_name": doc_name,
        "text_path": text_candidates[0],
        "image_path": image_candidates[0],
        "text_candidate_paths": text_candidates,
        "image_candidate_paths": image_candidates,
    }


def load_mdocagent_model_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a MDocAgent model yaml without printing or logging secrets."""

    path = Path(config_path)
    data = _load_yaml_mapping(path)
    allowed_fields = {
        "model_id",
        "model",
        "model_name",
        "base_url",
        "api_base_url",
        "api_key",
        "api_key_env",
        "api_key_env_var",
        "module_name",
        "class_name",
        "temperature",
        "max_tokens",
        "max_new_tokens",
    }
    return {key: value for key, value in data.items() if key in allowed_fields}


def build_api_run_config_from_mdocagent_yaml(
    config_path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> ApiRunConfig:
    """Map MDocAgent model yaml fields to ApiRunConfig with max_pages fixed to 1."""

    overrides = dict(overrides or {})
    if "max_pages" in overrides and int(overrides["max_pages"]) != 1:
        raise RuntimeError("Stage 2 real API trials require max_pages=1.")

    yaml_config = load_mdocagent_model_config(config_path)
    model_name = yaml_config.get("model") or yaml_config.get("model_name") or yaml_config.get("model_id")
    max_tokens = yaml_config.get("max_tokens", yaml_config.get("max_new_tokens"))
    return ApiRunConfig(
        enable_real_api=bool(overrides.get("enable_real_api", False)),
        provider=str(overrides.get("provider", "siliconflow")),
        model_name=overrides.get("model_name", model_name),
        max_pages=1,
        temperature=float(overrides.get("temperature", yaml_config.get("temperature", 0.0) or 0.0)),
        timeout_seconds=int(overrides.get("timeout_seconds", 120)),
        raw_output_dir=overrides.get("raw_output_dir"),
        discard_log_dir=overrides.get("discard_log_dir"),
        api_base_url=overrides.get("api_base_url", yaml_config.get("api_base_url") or yaml_config.get("base_url")),
        api_key_env_var=str(
            overrides.get(
                "api_key_env_var",
                yaml_config.get("api_key_env_var") or yaml_config.get("api_key_env") or "SILICONFLOW_API_KEY",
            )
        ),
        api_key=overrides.get("api_key", yaml_config.get("api_key")),
        max_tokens=int(max_tokens) if max_tokens not in (None, "") else None,
    )


def read_json_or_jsonl_records(path: str | Path) -> List[Dict[str, Any]]:
    """Read either a JSON list file or a JSONL file into a list of records."""

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return [record for record in parsed if isinstance(record, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        raise ValueError(f"Unsupported JSON root type: {type(parsed).__name__}")

    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL line {line_number} is not an object.")
        records.append(record)
    return records


def find_record_by_id_or_doc_question(
    records: List[Dict[str, Any]],
    record_id: Optional[str],
    doc_id: Optional[str],
    question_substring: Optional[str],
) -> Dict[str, Any]:
    """Locate one normalized or raw MDocAgent record without exposing eval-only fields."""

    if record_id:
        for record in records:
            if record.get("record_id") == record_id:
                return record

    normalized_question = (question_substring or "").lower()
    for record in records:
        candidate = record.get("canonical_record", record)
        document = candidate.get("document", {}) if isinstance(candidate, dict) else {}
        raw_question = candidate.get("question", {}) if isinstance(candidate, dict) else {}
        candidate_doc_id = document.get("doc_id") or record.get("doc_id")
        if isinstance(raw_question, dict):
            candidate_question = raw_question.get("text") or record.get("question") or record.get("question_text") or ""
        else:
            candidate_question = raw_question or record.get("question") or record.get("question_text") or ""
        if doc_id and candidate_doc_id != doc_id:
            continue
        if normalized_question and normalized_question not in str(candidate_question).lower():
            continue
        return record

    raise ValueError("No matching record found for the requested identifiers.")


def summarize_mdocagent_model_config(config_path: str | Path, api_config: ApiRunConfig) -> Dict[str, Any]:
    """Build a secret-free config summary for preflight reports."""

    yaml_config = load_mdocagent_model_config(config_path)
    return {
        "provider": api_config.provider,
        "model_name": api_config.model_name,
        "api_base_url_present": bool(api_config.api_base_url),
        "api_key_env_var": api_config.api_key_env_var,
        "api_key_present_in_yaml": bool(yaml_config.get("api_key")),
    }


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_yaml_mapping(path)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(loaded)


def _load_simple_yaml_mapping(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("-") or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            continue
        result[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip().strip('"').strip("'")
