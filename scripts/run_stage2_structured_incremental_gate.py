#!/usr/bin/env python3
"""Run bounded incremental Stage 2 structured expansion gates.

The expansion policy is cumulative 10 -> 20 -> 30 -> 50 pages. Each jump
compiles only the delta pages, then audits cumulative artifacts and runs the
activation scan. Stop/go metrics are diagnostic only and never participate in
reranking, retrieval scoring, QA, or gold-field access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_BASE_SUBSET = "outputs/subsets/stage2_structured_real_gate_subset.jsonl"
DEFAULT_BASE_ARTIFACTS = "outputs/stage2_structured_real_gate/artifacts.jsonl"
DEFAULT_BASE_QUALITY = "outputs/stage2_structured_real_gate/quality_report.json"
DEFAULT_BASE_ELIGIBILITY = "outputs/stage2_structured_real_gate/eligibility_audit.json"
DEFAULT_BASE_ACTIVATION = "outputs/experiments/mdocagent_module_ablation/run_tags/real_structured_activation_scan/real_structured_activation_scan_report.json"
DEFAULT_PROMPT_VERSION = "artifact_compiler_prompt_v2_structured_real_gate"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_CONFIG = "config/model/qwen3vl.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-size", type=int, default=10)
    parser.add_argument("--to-size", type=int, required=True, choices=(20, 30, 50))
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--previous-subset", default=DEFAULT_BASE_SUBSET)
    parser.add_argument("--previous-artifacts", default=DEFAULT_BASE_ARTIFACTS)
    parser.add_argument("--previous-quality", default=DEFAULT_BASE_QUALITY)
    parser.add_argument("--previous-eligibility", default=DEFAULT_BASE_ELIGIBILITY)
    parser.add_argument("--previous-activation", default=DEFAULT_BASE_ACTIVATION)
    parser.add_argument("--output-root", default="outputs/stage2_structured_incremental")
    parser.add_argument("--max-pages-per-doc", type=int, default=2)
    parser.add_argument("--retrieval-topk", type=int, default=10)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.to_size <= args.from_size:
        raise SystemExit("--to-size must be greater than --from-size")
    delta_pages = args.to_size - args.from_size
    if delta_pages < 1:
        raise SystemExit("delta pages must be positive")

    repo = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root) / f"r028_{args.from_size}_to_{args.to_size}"
    subset_delta = output_root / f"subset_delta_{args.from_size}_to_{args.to_size}.jsonl"
    subset_delta_report = output_root / f"subset_delta_{args.from_size}_to_{args.to_size}_report.json"
    delta_dir = output_root / "stage2_delta"
    cumulative_dir = output_root / "cumulative"
    activation_dir = output_root / "activation_scan"
    report_path = output_root / "incremental_gate_report.json"

    commands = build_commands(args, delta_pages, subset_delta, subset_delta_report, delta_dir, cumulative_dir, activation_dir)
    if not args.execute:
        print(json.dumps({"will_execute": False, "commands": commands, "report": str(report_path)}, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    command_results = []
    for command in commands[:2]:
        command_results.append(run_command(command, repo))

    merge_artifacts(Path(args.previous_artifacts), delta_dir / "artifacts.jsonl", cumulative_dir / "artifacts.jsonl")
    merge_subsets(Path(args.previous_subset), subset_delta, cumulative_dir / f"subset_cumulative_{args.to_size}.jsonl")

    for command in commands[2:]:
        command_results.append(run_command(command, repo))

    previous_metrics = read_metrics(args.previous_quality, args.previous_eligibility, args.previous_activation)
    current_metrics = read_metrics(
        delta_dir / "quality_report.json",
        cumulative_dir / "eligibility_audit.json",
        activation_dir / "real_structured_activation_scan_report.json",
    )
    cumulative_quality = read_json(delta_dir / "quality_report.json")
    current_metrics["delta_parse_success_rate"] = parse_success_rate(cumulative_quality)
    gate = compare_metrics(previous_metrics, current_metrics)
    report = {
        "schema_version": "stage2_structured_incremental_gate_v1",
        "from_size": args.from_size,
        "to_size": args.to_size,
        "delta_pages": delta_pages,
        "policy": {
            "expansion_sequence": [10, 20, 30, 50],
            "stop_go_metrics_only": True,
            "concentration_metrics_external_validity_only": True,
            "uses_gold_fields": False,
            "no_rerank_tuning": True,
            "no_qa_run": True,
            "no_full_ablation": True,
        },
        "previous_metrics": previous_metrics,
        "current_metrics": current_metrics,
        "gate": gate,
        "paths": {
            "delta_subset": str(subset_delta),
            "delta_dir": str(delta_dir),
            "cumulative_dir": str(cumulative_dir),
            "activation_dir": str(activation_dir),
        },
        "commands": command_results,
    }
    write_json(report_path, report)
    write_markdown(output_root / "incremental_gate_report.md", report)
    print(json.dumps({"decision": gate["decision"], "to_size": args.to_size, "current_metrics": current_metrics}, ensure_ascii=False, indent=2))


def build_commands(args: argparse.Namespace, delta_pages: int, subset_delta: Path, subset_delta_report: Path, delta_dir: Path, cumulative_dir: Path, activation_dir: Path) -> list[list[str]]:
    return [
        [
            "python3", "scripts/build_stage2_structured_subset.py",
            "--records", args.records,
            "--extract-root", args.extract_root,
            "--output", str(subset_delta),
            "--report-json", str(subset_delta_report),
            "--max-pages", str(delta_pages),
            "--max-pages-per-doc", str(args.max_pages_per_doc),
            "--retrieval-topk", str(args.retrieval_topk),
            "--exclude-subset", args.previous_subset,
            "--selection-source", f"structured_real_stage2_r028_{args.from_size}_to_{args.to_size}_delta",
        ],
        [
            "python3", "scripts/stage2.py", "doc-compile",
            "--provider", "real",
            "--enable-real-api",
            "--run-real-trial",
            "--allow-real-subset",
            "--input", args.records,
            "--subset-file", str(subset_delta),
            "--extract-root", args.extract_root,
            "--output-dir", str(delta_dir),
            "--model-config", args.model_config,
            "--model-name", args.model_name,
            "--prompt-version", args.prompt_version,
            "--image-payload-mode", "image_url",
            "--max-pages", str(delta_pages),
            "--max-pages-total", str(delta_pages),
            "--max-pages-real-cap", str(delta_pages),
            "--max-pages-per-call", "1",
            "--max-docs", str(delta_pages),
            "--max-pages-per-doc", str(args.max_pages_per_doc),
            "--timeout-seconds", str(args.timeout_seconds),
            "--max-retries", str(args.max_retries),
        ],
        [
            "python3", "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts", str(cumulative_dir / "artifacts.jsonl"),
            "--output-json", str(cumulative_dir / "eligibility_audit.json"),
            "--output-md", str(cumulative_dir / "eligibility_audit.md"),
        ],
        [
            "python3", "scripts/scan_real_artifact_activation.py",
            "--artifacts", str(cumulative_dir / "artifacts.jsonl"),
            "--output-dir", str(activation_dir),
        ],
    ]


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    row = {"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]}
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def merge_artifacts(previous: Path, delta: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in (previous, delta):
        if path.is_file():
            lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def merge_subsets(previous: Path, delta: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in (previous, delta):
        if path.is_file():
            lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_metrics(quality_path: str | Path, eligibility_path: str | Path, activation_path: str | Path) -> dict[str, Any]:
    quality = read_json(quality_path)
    eligibility = read_json(eligibility_path)
    activation = read_json(activation_path)
    return {
        "parse_success_rate": parse_success_rate(quality),
        "provider_success": int(quality.get("provider_call_success_count", 0) or 0),
        "provider_failed": int(quality.get("provider_call_failed_count", 0) or 0),
        "num_artifacts": int(quality.get("num_artifacts", 0) or 0),
        "strong_eligible_artifacts": int(eligibility.get("strong_eligible_artifacts", eligibility.get("eligible_artifacts", 0)) or 0),
        "eligible_pages": int(eligibility.get("eligible_pages", 0) or 0),
        "mock_or_placeholder_content": int(eligibility.get("mock_or_placeholder_content", 0) or 0),
        "full_page_only_locator": int(eligibility.get("full_page_only_locator", 0) or 0),
        "activated_count": int(activation.get("activated_count", 0) or 0),
        "eligible_for_heldout_count": int(activation.get("eligible_for_heldout_count", 0) or 0),
        "concentration": activation.get("concentration", {}),
        "heldout_available": bool((activation.get("heldout_activation_rich_subset") or {}).get("available", False)),
    }


def parse_success_rate(quality: dict[str, Any]) -> float:
    success = int(quality.get("provider_call_success_count", 0) or 0)
    failed = int(quality.get("provider_call_failed_count", 0) or 0)
    total = success + failed
    if total <= 0:
        return 0.0
    return round(success / total, 6)


def compare_metrics(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "parse_success_rate_non_decreasing": current["parse_success_rate"] >= previous["parse_success_rate"],
        "strong_eligible_artifacts_increased": current["strong_eligible_artifacts"] > previous["strong_eligible_artifacts"],
        "eligible_pages_increased": current["eligible_pages"] > previous["eligible_pages"],
        "activated_count_increased": current["activated_count"] > previous["activated_count"],
        "mock_or_placeholder_content_still_zero": current["mock_or_placeholder_content"] == 0,
    }
    should_continue = all(checks.values())
    return {
        "checks": checks,
        "decision": "continue_to_next_increment" if should_continue else "stop_expansion_review_noise",
        "reason": "all stop/go metrics passed" if should_continue else "one or more stop/go metrics failed; do not expand further until inspected",
        "metrics_do_not_use_gold_fields": True,
        "metrics_do_not_participate_in_reranking": True,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    prev = report["previous_metrics"]
    cur = report["current_metrics"]
    gate = report["gate"]
    lines = [
        "# Stage 2 Structured Incremental Gate",
        "",
        f"Range: {report['from_size']} -> {report['to_size']} pages",
        f"Decision: `{gate['decision']}`",
        "",
        "## Stop/Go Metrics",
        f"- parse_success_rate: {prev['parse_success_rate']} -> {cur['parse_success_rate']}",
        f"- strong_eligible_artifacts: {prev['strong_eligible_artifacts']} -> {cur['strong_eligible_artifacts']}",
        f"- eligible_pages: {prev['eligible_pages']} -> {cur['eligible_pages']}",
        f"- activated_count: {prev['activated_count']} -> {cur['activated_count']}",
        f"- mock_or_placeholder_content: {prev['mock_or_placeholder_content']} -> {cur['mock_or_placeholder_content']}",
        "",
        "## Checks",
    ]
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Scope",
        "- Metrics are stop/go diagnostics only.",
        "- Concentration metrics are external-validity diagnostics only.",
        "- No gold fields, no QA, no rerank tuning, no full ablation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
