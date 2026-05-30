#!/usr/bin/env python3
"""Run the Stage 2/3/4 refactor verification suite."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


COMMANDS = [
    ("stage2_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/stage2/tests", "-v"]),
    ("stage3_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/stage3/tests", "-v"]),
    ("stage4_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/stage4/tests", "-v"]),
    ("evaluation_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/evaluation/tests", "-v"]),
    ("audit_no_gold_leakage", [sys.executable, "scripts/audit_no_gold_leakage.py"]),
    ("audit_real_provider_smoke", [sys.executable, "scripts/audit_real_provider_smoke.py"]),
    ("audit_reproducibility", [sys.executable, "scripts/audit_reproducibility.py"]),
    ("compileall", [sys.executable, "-m", "compileall", "-q", "mdocnexus", "scripts"]),
    ("diff_check", ["git", "diff", "--check", "--", "mdocnexus", "scripts"]),
]


def main() -> int:
    results: list[dict[str, Any]] = []
    for name, command in COMMANDS:
        results.append(run_command(name, command))
    failed = [result for result in results if result.get("returncode", 0) != 0]
    report = {
        "results": results,
        "status": "fail" if failed else "pass",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


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
