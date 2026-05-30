#!/usr/bin/env python3
"""Run the Stage 2/3/4 refactor verification suite."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


COMMANDS = [
    ("stage2_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/stage2/tests", "-v"]),
    ("stage3_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/stage3/tests", "-v"]),
    ("stage4_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/stage4/tests", "-v"]),
    ("audit_no_gold_leakage", [sys.executable, "scripts/audit_no_gold_leakage.py"]),
    ("audit_reproducibility", [sys.executable, "scripts/audit_reproducibility.py"]),
    ("compileall", [sys.executable, "-m", "compileall", "-q", "mdocnexus"]),
    ("diff_check", ["git", "diff", "--check", "--", "mdocnexus", "scripts"]),
]


def main() -> int:
    results: list[dict[str, Any]] = []
    maybe_add_real_smoke_audit(results)
    for name, command in COMMANDS:
        results.append(run_command(name, command))
    failed = [result for result in results if result.get("returncode", 0) != 0]
    report = {
        "results": results,
        "status": "fail" if failed else "pass",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


def maybe_add_real_smoke_audit(results: list[dict[str, Any]]) -> None:
    manifest_path = Path("outputs/stage2_doc/manifest.json")
    quality_path = Path("outputs/stage2_doc/quality_report.json")
    call_log_path = Path("outputs/stage2_doc/call_log.jsonl")
    if not manifest_path.is_file() or not quality_path.is_file():
        results.append({"name": "audit_real_provider_smoke", "status": "skipped", "reason": "no_stage2_doc_smoke_manifest"})
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        results.append({"name": "audit_real_provider_smoke", "status": "skipped", "reason": "unreadable_stage2_manifest"})
        return
    provider_modes = manifest.get("provider_modes")
    if not isinstance(provider_modes, list) or "real" in provider_modes:
        results.append({"name": "audit_real_provider_smoke", "status": "skipped", "reason": "not_safe_fake_output"})
        return
    if not call_log_path.is_file():
        results.append({"name": "audit_real_provider_smoke", "status": "skipped", "reason": "no_call_log"})
        return
    results.append(run_command("audit_real_provider_smoke", [sys.executable, "scripts/audit_real_provider_smoke.py"]))


def run_command(name: str, command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def tail(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


if __name__ == "__main__":
    raise SystemExit(main())
