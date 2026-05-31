"""Shared deterministic IO helpers for experiment runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def run_command(command: list[str], cwd: str | Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=Path(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    row = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def current_git_commit(cwd: str | Path) -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"
