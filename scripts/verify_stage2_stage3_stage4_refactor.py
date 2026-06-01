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
    ("integration_tests", [sys.executable, "-m", "unittest", "discover", "mdocnexus/integration/tests", "-v"]),
    ("adapter_compatibility_help", [sys.executable, "scripts/check_mdocagent_adapter_compatibility.py", "--help"]),
    ("mdocagent_adapt_help", [sys.executable, "scripts/mdocnexus.py", "mdocagent-adapt", "--help"]),
    ("mdocagent_module_ablation_prepare", [sys.executable, "scripts/run_mdocagent_module_ablation.py", "--prepare-only"]),
    ("audit_no_gold_leakage", [sys.executable, "scripts/audit_no_gold_leakage.py"]),
    ("audit_real_provider_smoke", [sys.executable, "scripts/audit_real_provider_smoke.py"]),
    ("audit_reproducibility", [sys.executable, "scripts/audit_reproducibility.py"]),
    ("audit_model_configs", [sys.executable, "scripts/audit_model_configs.py"]),
    ("compileall", [sys.executable, "-m", "compileall", "-q", "mdocnexus", "scripts"]),
    ("diff_check", ["git", "diff", "--check", "--", "mdocnexus", "scripts"]),
]


def main() -> int:
    results: list[dict[str, Any]] = []
    for name, command in COMMANDS:
        results.append(run_command(name, command))
    results.append(run_integration_checks())
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


def run_integration_checks() -> dict[str, Any]:
    root = "outputs/experiments/mdocagent_module_ablation"
    summary_path = f"{root}/summary.json"
    manifest_root = f"{root}/manifests"
    errors: list[str] = []
    try:
        from pathlib import Path

        summary_file = Path(summary_path)
        if not summary_file.is_file():
            errors.append(f"missing_summary:{summary_path}")
        else:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if not isinstance(summary.get("runs"), list) or not summary["runs"]:
                errors.append("summary_runs_missing")
            for row in summary.get("runs", []):
                if row.get("same_page_budget_as_baseline") is not True:
                    errors.append(f"summary_page_budget_false:{row.get('run_name')}")
                if row.get("no_gold_fields_used") is not True:
                    errors.append(f"summary_no_gold_false:{row.get('run_name')}")
                if row.get("used_debug_edges") is not False:
                    errors.append(f"summary_debug_edges_true:{row.get('run_name')}")
                if row.get("used_semantic_edges") is not False:
                    errors.append(f"summary_semantic_edges_true:{row.get('run_name')}")
                compatibility_path = row.get("compatibility_report_path")
                if not compatibility_path:
                    errors.append(f"summary_missing_compatibility_path:{row.get('run_name')}")
                elif not Path(str(compatibility_path)).is_file():
                    errors.append(f"missing_compatibility_report:{compatibility_path}")
                else:
                    compatibility = json.loads(Path(str(compatibility_path)).read_text(encoding="utf-8"))
                    if compatibility.get("status") != "pass":
                        errors.append(f"compatibility_not_pass:{compatibility_path}")
        manifest_dir = Path(manifest_root)
        if not manifest_dir.is_dir():
            errors.append(f"missing_manifest_dir:{manifest_root}")
        for manifest_path in sorted(manifest_dir.glob("*.json")) if manifest_dir.is_dir() else []:
            if manifest_path.name.endswith(".adapter.json"):
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            adapter_manifest = manifest.get("adapter_manifest")
            if not isinstance(adapter_manifest, dict):
                errors.append(f"missing_adapter_manifest:{manifest_path}")
                continue
            if adapter_manifest.get("no_gold_fields_used") is not True:
                errors.append(f"adapter_no_gold_false:{manifest_path}")
            if adapter_manifest.get("same_page_budget_as_baseline") is not True:
                errors.append(f"adapter_page_budget_false:{manifest_path}")
            if adapter_manifest.get("used_debug_edges") is not False:
                errors.append(f"adapter_debug_edges_true:{manifest_path}")
            if adapter_manifest.get("used_semantic_edges") is not False:
                errors.append(f"adapter_semantic_edges_true:{manifest_path}")
            if adapter_manifest.get("model_role") != "none_deterministic":
                errors.append(f"adapter_model_role_not_deterministic:{manifest_path}")
            if adapter_manifest.get("evaluator_model_used") is not False:
                errors.append(f"adapter_evaluator_model_used:{manifest_path}")
            if "DeepSeek-V3" in json.dumps(adapter_manifest, ensure_ascii=False, sort_keys=True):
                errors.append(f"deepseek_in_adapter_manifest:{manifest_path}")
    except Exception as exc:
        errors.append(f"integration_check_exception:{exc}")
    return {
        "name": "mdocagent_integration_checks",
        "command": ["internal"],
        "returncode": 1 if errors else 0,
        "stdout_tail": json.dumps({"errors": errors}, ensure_ascii=False, sort_keys=True),
        "stderr_tail": "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
