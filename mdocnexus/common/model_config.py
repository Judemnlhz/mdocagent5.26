"""Model config loading and role audit helpers.

The model role source of truth is the existing ``config/model/*.yaml`` files.
This module intentionally supports only the small YAML subset used by those
files so the audit stays dependency-free and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V3"
QWEN3_MODEL_ID = "Qwen/Qwen3-8B"
QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

DEEPSEEK_CONFIG = "config/model/deepseekv3.yaml"
QWEN3_CONFIG = "config/model/qwen3.yaml"
QWEN3VL_CONFIG = "config/model/qwen3vl.yaml"
MODEL_CONFIGS = {
    "evaluation_judge": DEEPSEEK_CONFIG,
    "text_only_processing": QWEN3_CONFIG,
    "multimodal_extraction": QWEN3VL_CONFIG,
}
SECRET_KEYS = {"api_key", "secret", "access_token", "api_token"}
TEXT_CLASS_MARKERS = ("Text", "Chat", "Language")
VISION_CLASS_MARKERS = ("Vision", "VL", "Image", "Multimodal")


def load_model_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.endswith(":") and ":" in stripped:
            current_list_key = stripped[:-1]
            if current_list_key == "defaults":
                result[current_list_key] = []
            else:
                result[current_list_key] = None
                current_list_key = None
            continue
        if stripped.startswith("- ") and current_list_key:
            value = stripped[2:].strip()
            if isinstance(result.get(current_list_key), list):
                result[current_list_key].append(parse_scalar(value))
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = parse_scalar(value.strip())
    return result


def parse_scalar(value: str) -> Any:
    if value == "" or value.lower() in {"null", "none"}:
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path_obj.name


def model_id_from_config(config: dict[str, Any]) -> str | None:
    value = config.get("model_id") or config.get("model")
    return str(value) if value not in (None, "") else None


def model_provider_from_config(config: dict[str, Any]) -> str | None:
    value = config.get("module_name") or config.get("provider") or config.get("base_url")
    return str(value) if value not in (None, "") else None


def manifest_model_fields(
    *,
    model_config_path: str | Path | None,
    model_id: str | None,
    model_role: str,
    model_used_for: str,
    model_provider: str | None = None,
    evaluator_model_used: bool = False,
) -> dict[str, Any]:
    return {
        "model_config_path": public_path(model_config_path),
        "model_config_hash": file_sha256(model_config_path) if model_config_path not in (None, "") and Path(model_config_path).is_file() else None,
        "model_id": model_id,
        "model_role": model_role,
        "model_used_for": model_used_for,
        "model_provider": model_provider,
        "evaluator_model_used": bool(evaluator_model_used),
    }


def deterministic_stage_model_fields(model_role: str, model_used_for: str) -> dict[str, Any]:
    return manifest_model_fields(
        model_config_path=None,
        model_id=None,
        model_role=model_role,
        model_used_for=model_used_for,
        model_provider=None,
        evaluator_model_used=False,
    )


def stage2_model_fields(provider_mode: str, model_config_path: str | Path | None = None) -> dict[str, Any]:
    if str(provider_mode) in {"dry_run", "fake", "mock", "none"}:
        return manifest_model_fields(
            model_config_path=None,
            model_id=None,
            model_role="none_or_fake",
            model_used_for="stage2_document_generic_fake_or_dry_run",
            model_provider=None,
            evaluator_model_used=False,
        )
    path = model_config_path or QWEN3VL_CONFIG
    config = load_model_config(path)
    return manifest_model_fields(
        model_config_path=path,
        model_id=model_id_from_config(config),
        model_role="multimodal_extraction",
        model_used_for="stage2_document_generic_artifact_compilation",
        model_provider=model_provider_from_config(config),
        evaluator_model_used=False,
    )


def evaluation_model_fields(model_config_path: str | Path | None = None, evaluator_model_used: bool = False) -> dict[str, Any]:
    if not evaluator_model_used:
        return manifest_model_fields(
            model_config_path=None,
            model_id=None,
            model_role="evaluation_only_no_model",
            model_used_for="deterministic_retrieval_metric_evaluation",
            model_provider=None,
            evaluator_model_used=False,
        )
    path = model_config_path or DEEPSEEK_CONFIG
    config = load_model_config(path)
    return manifest_model_fields(
        model_config_path=path,
        model_id=model_id_from_config(config),
        model_role="evaluation_judge",
        model_used_for="evaluation_judge_only",
        model_provider=model_provider_from_config(config),
        evaluator_model_used=True,
    )


def audit_model_configs(
    *,
    config_paths: Iterable[str | Path] = (DEEPSEEK_CONFIG, QWEN3_CONFIG, QWEN3VL_CONFIG),
    stage2_dirs: Iterable[str | Path] = ("outputs/stage2_doc",),
    stage3_dirs: Iterable[str | Path] = ("outputs/stage3_doc_artifact_retrieval",),
    stage4_dirs: Iterable[str | Path] = ("outputs/stage4/evidence_graph",),
    evaluation_dirs: Iterable[str | Path] = ("outputs/eval",),
    experiment_dirs: Iterable[str | Path] = ("outputs/experiments/matrix",),
) -> dict[str, Any]:
    config_files_checked: list[str] = []
    model_ids: dict[str, str | None] = {}
    public_credential_violations: list[dict[str, str]] = []
    config_by_path: dict[str, dict[str, Any]] = {}
    for path in config_paths:
        config_path = Path(path)
        config_files_checked.append(str(config_path))
        config = load_model_config(config_path)
        config_by_path[str(config_path)] = config
        model_ids[str(config_path)] = model_id_from_config(config)
        if config.get("api_key") not in (None, ""):
            public_credential_violations.append({"path": str(config_path), "reason": "non_empty_api_key_in_model_config"})
        for key, value in config.items():
            if str(key).lower() in SECRET_KEYS and value not in (None, ""):
                public_credential_violations.append({"path": str(config_path), "reason": f"non_empty_credential_field:{key}"})

    stage2_model_violations = validate_stage_outputs(stage2_dirs, stage_name="stage2", allow_deepseek=False)
    stage3_model_violations = validate_stage_outputs(stage3_dirs, stage_name="stage3", allow_deepseek=False)
    stage4_model_violations = validate_stage_outputs(stage4_dirs, stage_name="stage4", allow_deepseek=False)
    evaluation_model_violations = validate_evaluation_outputs(evaluation_dirs)
    experiment_violations = validate_experiment_outputs(experiment_dirs)
    config_violations = validate_required_model_configs(config_by_path)
    all_violations = (
        stage2_model_violations
        + stage3_model_violations
        + stage4_model_violations
        + evaluation_model_violations
        + experiment_violations
        + public_credential_violations
        + config_violations
    )
    return {
        "config_files_checked": config_files_checked,
        "model_ids": model_ids,
        "stage2_model_violations": stage2_model_violations,
        "stage3_model_violations": stage3_model_violations,
        "stage4_model_violations": stage4_model_violations,
        "evaluation_model_violations": evaluation_model_violations,
        "experiment_model_violations": experiment_violations,
        "public_credential_violations": public_credential_violations,
        "config_violations": config_violations,
        "status": "fail" if all_violations else "pass",
    }


def validate_required_model_configs(config_by_path: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    expected = {
        DEEPSEEK_CONFIG: DEEPSEEK_MODEL_ID,
        QWEN3_CONFIG: QWEN3_MODEL_ID,
        QWEN3VL_CONFIG: QWEN3VL_MODEL_ID,
    }
    for path, expected_id in expected.items():
        config = config_by_path.get(path) or load_model_config(path)
        actual = model_id_from_config(config)
        if actual != expected_id:
            violations.append({"path": path, "reason": f"unexpected_model_id:{actual}"})
    deepseek = config_by_path.get(DEEPSEEK_CONFIG) or load_model_config(DEEPSEEK_CONFIG)
    if not any(marker in str(deepseek.get("class_name") or "") for marker in TEXT_CLASS_MARKERS):
        violations.append({"path": DEEPSEEK_CONFIG, "reason": "deepseek_class_not_text_model"})
    qwen3vl = config_by_path.get(QWEN3VL_CONFIG) or load_model_config(QWEN3VL_CONFIG)
    if not any(marker in str(qwen3vl.get("class_name") or "") for marker in VISION_CLASS_MARKERS):
        violations.append({"path": QWEN3VL_CONFIG, "reason": "qwen3vl_class_not_vision_model"})
    return violations


def validate_stage_outputs(paths: Iterable[str | Path], stage_name: str, allow_deepseek: bool) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for path in iter_public_json_files(paths):
        text = safe_read_text(path)
        if not allow_deepseek and DEEPSEEK_MODEL_ID in text:
            violations.append({"path": str(path), "stage": stage_name, "reason": "deepseek_in_stage_main_flow"})
        for value in iter_json_values(path):
            if isinstance(value, dict):
                if value.get("model_id") == DEEPSEEK_MODEL_ID and not allow_deepseek:
                    violations.append({"path": str(path), "stage": stage_name, "reason": "deepseek_model_id_in_manifest"})
                if value.get("evaluator_model_used") is True:
                    violations.append({"path": str(path), "stage": stage_name, "reason": "evaluator_model_used_in_main_flow"})
    return violations


def validate_evaluation_outputs(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for path in iter_public_json_files(paths):
        for value in iter_json_values(path):
            if not isinstance(value, dict):
                continue
            if value.get("model_id") == DEEPSEEK_MODEL_ID or value.get("evaluator_model_used") is True:
                if value.get("evaluation_only") is not True or value.get("not_consumed_by_stage2_stage3_stage4") is not True:
                    violations.append({"path": str(path), "stage": "evaluation", "reason": "evaluation_model_missing_eval_only_flags"})
            if value.get("model_id") == QWEN3VL_MODEL_ID and value.get("model_role") == "evaluation_judge":
                violations.append({"path": str(path), "stage": "evaluation", "reason": "qwen3vl_used_as_evaluation_judge"})
    return violations


def validate_experiment_outputs(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for path in iter_public_json_files(paths):
        if path.name not in {"summary_matrix.json", "summary.json", "manifest.json"}:
            continue
        for value in iter_json_values(path):
            for row in value if isinstance(value, list) else [value]:
                if not isinstance(row, dict):
                    continue
                if row.get("stage2_model_id") == DEEPSEEK_MODEL_ID or row.get("stage3_model_id") == DEEPSEEK_MODEL_ID or row.get("stage4_model_id") == DEEPSEEK_MODEL_ID:
                    violations.append({"path": str(path), "stage": "experiment", "reason": "deepseek_in_stage_summary"})
    return violations


def iter_public_json_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in {".json", ".jsonl"}:
                yield root
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                yield path


def iter_json_values(path: Path) -> Iterable[Any]:
    try:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)
        else:
            yield json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
