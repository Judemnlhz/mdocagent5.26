"""Run manifest helpers for Stage 2 experiment auditability."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Dict


DEFAULT_RUNTIME_NOTES = {
    "predict_py_modified": True,
    "multi_agent_system_modified": True,
    "stage2_depends_on_predict_py": False,
    "stage2_depends_on_multi_agent_system": False,
}

FORBIDDEN_MANIFEST_FIELDS = {
    "api_key",
    "answer",
    "evidence_pages",
    "binary_correctness",
}


def build_stage2_run_manifest(
    stage: str,
    script_name: str,
    config_path: str,
    provider: str,
    model_name: str,
    output_dir: str,
    real_api_called: bool,
    limits: dict,
    runtime_notes: dict | None = None,
) -> dict:
    """Build a secret-free Stage 2 run manifest without calling providers."""

    merged_runtime_notes: Dict[str, Any] = dict(DEFAULT_RUNTIME_NOTES)
    if runtime_notes:
        merged_runtime_notes.update(_strip_forbidden_keys(runtime_notes))

    manifest = {
        "stage": stage,
        "script_name": script_name,
        "git_commit": _current_git_commit(),
        "config_path": str(config_path),
        "provider": provider,
        "model_name": model_name,
        "real_api_called": bool(real_api_called),
        "limits": _strip_forbidden_keys(dict(limits or {})),
        "runtime_notes": merged_runtime_notes,
        "output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    forbidden = sorted(field for field in FORBIDDEN_MANIFEST_FIELDS if _contains_key(manifest, field))
    if forbidden:
        raise ValueError(f"Run manifest contains forbidden fields: {forbidden}")
    return manifest


def write_stage2_run_manifest(manifest: dict, path: str | Path) -> None:
    """Write a Stage 2 run manifest as pretty JSON."""

    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _strip_forbidden_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden_keys(child)
            for key, child in value.items()
            if key not in FORBIDDEN_MANIFEST_FIELDS
        }
    if isinstance(value, list):
        return [_strip_forbidden_keys(child) for child in value]
    return value


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False
