#!/usr/bin/env python3
"""Run the Stage 2 real structured artifact gate on a tiny fixed subset.

This intentionally does not run QA, rerank tuning, graph expansion, or full
ablation. The gate is only: choose structured pages, compile real artifacts,
audit strong eligibility, and write a go/no-go report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_OUTPUT_DIR = "outputs/stage2_structured_real_gate"
DEFAULT_SUBSET = "outputs/subsets/stage2_structured_real_gate_subset.jsonl"
DEFAULT_SUBSET_REPORT = "outputs/subsets/stage2_structured_real_gate_subset_report.json"
DEFAULT_PROMPT_VERSION = "artifact_compiler_prompt_v2_structured_real_gate"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_CONFIG = "config/model/qwen3vl.yaml"
MAX_REAL_PAGES_CAP = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run commands. Default only prints the plan.")
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-output", default=DEFAULT_SUBSET)
    parser.add_argument("--subset-report", default=DEFAULT_SUBSET_REPORT)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-pages-per-doc", type=int, default=3)
    parser.add_argument("--retrieval-topk", type=int, default=10)
    parser.add_argument("--provider-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--provider", default="real", choices=("real",))
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--image-payload-mode", choices=("image_url", "base64", "none"), default="image_url")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if int(args.max_pages) < 1 or int(args.max_pages) > MAX_REAL_PAGES_CAP:
        raise RuntimeError(f"--max-pages must be between 1 and {MAX_REAL_PAGES_CAP}.")
    if int(args.max_pages_per_doc) < 1:
        raise RuntimeError("--max-pages-per-doc must be at least 1.")
    if int(args.retrieval_topk) < 1:
        raise RuntimeError("--retrieval-topk must be at least 1.")


def build_subset_command(args: argparse.Namespace) -> list[str]:
    return [
        "python3", "scripts/build_stage2_structured_subset.py",
        "--records", args.records,
        "--extract-root", args.extract_root,
        "--output", args.subset_output,
        "--report-json", args.subset_report,
        "--max-pages", str(args.max_pages),
        "--max-pages-per-doc", str(args.max_pages_per_doc),
        "--retrieval-topk", str(args.retrieval_topk),
    ]


def build_compile_command(args: argparse.Namespace) -> list[str]:
    return [
        "python3", "scripts/stage2.py", "doc-compile",
        "--provider", "real",
        "--enable-real-api",
        "--run-real-trial",
        "--allow-real-subset",
        "--input", args.records,
        "--subset-file", args.subset_output,
        "--extract-root", args.extract_root,
        "--output-dir", args.output_dir,
        "--model-config", args.model_config,
        "--model-name", args.model_name,
        "--prompt-version", args.prompt_version,
        "--image-payload-mode", args.image_payload_mode,
        "--max-pages", str(args.max_pages),
        "--max-pages-total", str(args.max_pages),
        "--max-pages-real-cap", str(MAX_REAL_PAGES_CAP),
        "--max-pages-per-call", "1",
        "--max-docs", str(args.max_pages),
        "--max-pages-per-doc", str(args.max_pages_per_doc),
        "--timeout-seconds", str(args.timeout_seconds),
        "--max-retries", str(args.max_retries),
    ]


def build_audit_command(args: argparse.Namespace) -> list[str]:
    return [
        "python3", "scripts/audit_artifact_rerank_eligibility.py",
        "--artifacts", str(Path(args.output_dir) / "artifacts.jsonl"),
        "--output-json", str(Path(args.output_dir) / "eligibility_audit.json"),
        "--output-md", str(Path(args.output_dir) / "eligibility_audit.md"),
    ]


def run_command(command: list[str], cwd: Path, allow_failure: bool = False) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    row = {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_gate_report(args: argparse.Namespace, status: str, commands: list[dict[str, Any]], provider_configured: bool) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    subset_report = read_json(args.subset_report)
    quality_report = read_json(output_dir / "quality_report.json")
    eligibility = read_json(output_dir / "eligibility_audit.json")
    manifest = read_json(output_dir / "manifest.json")

    eligible_artifacts = int(eligibility.get("eligible_artifacts", 0) or 0)
    eligible_pages = int(eligibility.get("eligible_pages", 0) or 0)
    mock_count = int(eligibility.get("mock_or_placeholder_content", 0) or 0)
    full_page_only = int(eligibility.get("full_page_only_locator", 0) or 0)
    provider_success = int(quality_report.get("provider_call_success_count", 0) or 0)
    provider_failed = int(quality_report.get("provider_call_failed_count", 0) or 0)

    if status == "executed" and eligible_artifacts > 0 and eligible_pages > 0 and mock_count == 0:
        decision = "stage2_structured_gate_pass"
    elif status == "executed":
        decision = "stage2_structured_gate_fail"
    elif status == "blocked_provider_not_configured":
        decision = "blocked_provider_not_configured"
    else:
        decision = status

    return {
        "schema_version": "stage2_real_structured_gate_v1",
        "status": status,
        "decision": decision,
        "purpose": "Stage 2 real structured artifact extraction quality gate; no rerank, no QA, no full ablation.",
        "constraints": {
            "no_rerank_tuning": True,
            "no_full_ablation": True,
            "no_qa_effectiveness_run": True,
            "max_pages": int(args.max_pages),
            "provider_key_env": args.provider_key_env,
            "provider_configured": bool(provider_configured),
        },
        "subset": {
            "subset_output": args.subset_output,
            "subset_report": args.subset_report,
            "num_selected_docs": subset_report.get("num_selected_docs"),
            "num_selected_pages": subset_report.get("num_selected_pages"),
            "selection_source": "structured_real_stage2_small_sample",
        },
        "stage2_quality": quality_report,
        "eligibility_gate": {
            "artifact_path": str(output_dir / "artifacts.jsonl"),
            "eligible_artifacts": eligible_artifacts,
            "strong_eligible_artifacts": int(eligibility.get("strong_eligible_artifacts", eligible_artifacts) or 0),
            "eligible_pages": eligible_pages,
            "mock_or_placeholder_content": mock_count,
            "full_page_only_locator": full_page_only,
            "provider_call_success_count": provider_success,
            "provider_call_failed_count": provider_failed,
            "reason_counts": eligibility.get("reason_counts", {}),
            "eligible_artifact_type_counts": eligibility.get("eligible_artifact_type_counts", {}),
            "eligible_locator_kind_counts": eligibility.get("eligible_locator_kind_counts", {}),
        },
        "manifest": {
            "prompt_version": manifest.get("prompt_version"),
            "model_id": manifest.get("model_id"),
            "model_role": manifest.get("model_role"),
            "api_called": manifest.get("api_called"),
            "provider_call_count": manifest.get("provider_call_count"),
        },
        "next_step": next_step(decision),
        "commands": commands,
    }


def next_step(decision: str) -> str:
    if decision == "stage2_structured_gate_pass":
        return "Build a new held-out activation-rich subset from records activated by strong eligible artifacts, excluding prior policy-tuning top30."
    if decision == "blocked_provider_not_configured":
        return "Set the Stage 2 provider key in the remote environment and rerun this gate; do not use fake artifacts as evidence."
    return "Inspect provider/parser failures and revise Stage 2 structured extraction before any QA or ablation run."


def provider_key_configured(env_var_name: str) -> bool:
    if os.environ.get(env_var_name, "").strip():
        return True
    for env_path in (Path.home() / ".config" / "mdocagent" / "stage2.env", Path("/etc/environment")):
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(f"export {env_var_name}=") or stripped.startswith(f"{env_var_name}="):
                return bool(stripped.split("=", 1)[1].strip().strip("'\""))
    return False


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> None:
    gate = report.get("eligibility_gate", {})
    subset = report.get("subset", {})
    lines = [
        "# Stage 2 Real Structured Artifact Gate",
        "",
        f"Decision: `{report.get('decision')}`",
        f"Status: `{report.get('status')}`",
        "",
        "## Scope",
        "- No rerank tuning.",
        "- No QA effectiveness run.",
        "- No full artifact ablation.",
        "",
        "## Subset",
        f"- Selected docs: {subset.get('num_selected_docs')}",
        f"- Selected pages: {subset.get('num_selected_pages')}",
        f"- Subset file: `{subset.get('subset_output')}`",
        "",
        "## Eligibility",
        f"- Strong eligible artifacts: {gate.get('strong_eligible_artifacts')}",
        f"- Eligible pages: {gate.get('eligible_pages')}",
        f"- Mock or placeholder content: {gate.get('mock_or_placeholder_content')}",
        f"- Full-page-only locator: {gate.get('full_page_only_locator')}",
        f"- Provider successes: {gate.get('provider_call_success_count')}",
        f"- Provider failures: {gate.get('provider_call_failed_count')}",
        "",
        "## Next Step",
        report.get("next_step", ""),
    ]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    repo = Path(__file__).resolve().parents[1]
    commands = [build_subset_command(args), build_compile_command(args), build_audit_command(args)]
    if not args.execute:
        print(json.dumps({"will_execute": False, "commands": commands, "output_dir": args.output_dir}, ensure_ascii=False, indent=2))
        return

    command_results: list[dict[str, Any]] = []
    provider_configured = provider_key_configured(args.provider_key_env)
    command_results.append(run_command(commands[0], repo))
    if not provider_configured:
        report = build_gate_report(args, "blocked_provider_not_configured", command_results, provider_configured)
    else:
        command_results.append(run_command(commands[1], repo, allow_failure=True))
        artifacts_path = Path(args.output_dir) / "artifacts.jsonl"
        if artifacts_path.is_file():
            command_results.append(run_command(commands[2], repo, allow_failure=True))
        report = build_gate_report(args, "executed", command_results, provider_configured)

    output_dir = Path(args.output_dir)
    write_json(output_dir / "structured_gate_report.json", report)
    write_markdown_report(output_dir / "structured_gate_report.md", report)
    print(json.dumps({"decision": report["decision"], "report": str(output_dir / "structured_gate_report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
