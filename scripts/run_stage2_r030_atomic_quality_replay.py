#!/usr/bin/env python3
"""R030 bounded Stage 2 atomic-quality replay on the same three failed pages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_SUBSET = "outputs/stage2_structured_incremental/r028_20_to_30/parse_repair_replay_3/subset_failed_pages.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/stage2_structured_incremental/r028_20_to_30/r030_atomic_quality_replay_3"
DEFAULT_BASELINE = "outputs/stage2_structured_incremental/r028_20_to_30/atomic_prompt_replay_3/atomic_prompt_quality_report.json"
DEFAULT_BASELINE_ELIGIBILITY = "outputs/stage2_structured_incremental/r028_20_to_30/atomic_prompt_replay_3/eligibility_audit_r030_taxonomy_check.json"
DEFAULT_PROMPT_VERSION = "artifact_compiler_prompt_v3_atomic_quality_r030"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_CONFIG = "config/model/qwen3vl.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--subset-file", default=DEFAULT_SUBSET)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-report", default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-eligibility", default=DEFAULT_BASELINE_ELIGIBILITY)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    stage2_dir = output_root / "stage2_delta"
    eligibility_json = output_root / "eligibility_audit.json"
    eligibility_md = output_root / "eligibility_audit.md"
    report_json = output_root / "atomic_quality_report.json"
    report_md = output_root / "atomic_quality_report.md"

    commands = [
        [
            "python3", "scripts/stage2.py", "doc-compile",
            "--provider", "real",
            "--enable-real-api",
            "--run-real-trial",
            "--allow-real-subset",
            "--input", args.records,
            "--subset-file", args.subset_file,
            "--extract-root", args.extract_root,
            "--output-dir", str(stage2_dir),
            "--model-config", args.model_config,
            "--model-name", args.model_name,
            "--prompt-version", args.prompt_version,
            "--image-payload-mode", "image_url",
            "--max-pages", "3",
            "--max-pages-total", "3",
            "--max-pages-real-cap", "3",
            "--max-pages-per-call", "1",
            "--max-docs", "3",
            "--max-pages-per-doc", "1",
            "--timeout-seconds", str(args.timeout_seconds),
            "--max-retries", str(args.max_retries),
        ],
        [
            "python3", "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts", str(stage2_dir / "artifacts.jsonl"),
            "--output-json", str(eligibility_json),
            "--output-md", str(eligibility_md),
        ],
    ]

    if not args.execute:
        print(json.dumps({"will_execute": False, "commands": commands, "report": str(report_json)}, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    command_results = [run_command(command, repo) for command in commands]
    report = build_report(args, stage2_dir, eligibility_json, command_results)
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(json.dumps({"decision": report["decision"], "current": report["r030_metrics"]}, ensure_ascii=False, indent=2))


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    row = {"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]}
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def build_report(args: argparse.Namespace, stage2_dir: Path, eligibility_json: Path, command_results: list[dict[str, Any]]) -> dict[str, Any]:
    quality = read_json(stage2_dir / "quality_report.json")
    eligibility = read_json(eligibility_json)
    baseline = read_json(args.baseline_report)
    baseline_eligibility = read_json(args.baseline_eligibility)
    baseline_metrics = extract_baseline_metrics(baseline, baseline_eligibility)
    current_metrics = extract_current_metrics(quality, eligibility)
    checks = {
        "parse_failure_still_zero": current_metrics["parse_failure_count"] == 0,
        "mock_still_zero": current_metrics["mock_or_placeholder_content"] == 0,
        "full_page_only_still_zero": current_metrics["full_page_only_locator"] == 0,
        "table_cell_kept_or_increased": current_metrics["table_cell_count"] >= baseline_metrics["table_cell_count"],
        "numeric_fact_appeared": current_metrics["numeric_fact_count"] > 0,
        "broad_table_only_declined": current_metrics["broad_table_only_count"] < baseline_metrics["broad_table_only_count"],
        "eligible_pages_not_decreased": current_metrics["eligible_pages"] >= baseline_metrics["eligible_pages"],
    }
    passed = all(checks.values())
    return {
        "schema_version": "stage2_r030_atomic_quality_replay_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "bounded_same_failed_pages_only": True,
            "num_pages": 3,
            "subset_file": args.subset_file,
            "no_stage2_expansion": True,
            "no_activation_scan": True,
            "no_qa": True,
            "no_graph": True,
            "no_rerank_tuning": True,
            "probe_outputs_not_merged": True,
            "public_raw_provider_responses": False,
            "uses_gold_fields": False,
        },
        "baseline_r029_metrics": baseline_metrics,
        "r030_metrics": current_metrics,
        "checks": checks,
        "decision": "ready_for_activation_scan_review" if passed else "continue_stage2_quality_repair_no_activation",
        "paths": {
            "stage2_dir": str(stage2_dir),
            "artifacts": str(stage2_dir / "artifacts.jsonl"),
            "discard": str(stage2_dir / "discard.jsonl"),
            "quality_report": str(stage2_dir / "quality_report.json"),
            "eligibility_audit": str(eligibility_json),
        },
        "commands": command_results,
    }


def extract_baseline_metrics(report: dict[str, Any], eligibility: dict[str, Any]) -> dict[str, Any]:
    replay = report.get("atomic_prompt_replay", {}) if isinstance(report.get("atomic_prompt_replay"), dict) else {}
    return {
        "parse_failure_count": int(replay.get("parse_failure_count", 0) or 0),
        "json_parse_success_count": int(replay.get("json_parse_success_count", 0) or 0),
        "valid_artifacts": int(replay.get("num_valid_artifacts", 0) or 0),
        "discarded_artifacts": int(replay.get("num_discarded_artifacts", 0) or 0),
        "strong_eligible_artifacts": int(eligibility.get("strong_eligible_artifacts", replay.get("strong_eligible_artifacts", 0)) or 0),
        "atomic_strong_eligible_artifacts": int(eligibility.get("atomic_strong_eligible_artifacts", replay.get("atomic_strong_eligible_artifacts", 0)) or 0),
        "eligible_pages": int(eligibility.get("eligible_pages", replay.get("eligible_pages", 0)) or 0),
        "mock_or_placeholder_content": int(replay.get("mock_or_placeholder_content", 0) or 0),
        "full_page_only_locator": int(replay.get("full_page_only_locator", 0) or 0),
        "table_cell_count": int(replay.get("table_cell_artifacts", 0) or 0),
        "numeric_fact_count": int(replay.get("numeric_fact_artifacts", 0) or 0),
        "broad_table_only_count": int(eligibility.get("broad_table_only_count", replay.get("broad_table_only_count", 2)) or 0),
        "eligible_pages_with_atomic_artifact": int(eligibility.get("eligible_pages_with_atomic_artifact", replay.get("eligible_pages_with_atomic_artifact", 1)) or 0),
    }


def extract_current_metrics(quality: dict[str, Any], eligibility: dict[str, Any]) -> dict[str, Any]:
    return {
        "parse_failure_count": int(quality.get("parse_failure_count", 0) or 0),
        "json_parse_success_count": int(quality.get("json_parse_success_count", 0) or 0),
        "valid_artifacts": int(quality.get("num_valid_artifacts", 0) or 0),
        "discarded_artifacts": int(quality.get("num_discarded_artifacts", 0) or 0),
        "strong_eligible_artifacts": int(eligibility.get("strong_eligible_artifacts", 0) or 0),
        "atomic_strong_eligible_artifacts": int(eligibility.get("atomic_strong_eligible_artifacts", 0) or 0),
        "eligible_pages": int(eligibility.get("eligible_pages", 0) or 0),
        "mock_or_placeholder_content": int(eligibility.get("mock_or_placeholder_content", 0) or 0),
        "full_page_only_locator": int(eligibility.get("full_page_only_locator", 0) or 0),
        "table_cell_count": int(eligibility.get("table_cell_count", quality.get("table_cell_artifact_count", 0)) or 0),
        "numeric_fact_count": int(eligibility.get("numeric_fact_count", quality.get("numeric_fact_count", 0)) or 0),
        "broad_table_only_count": int(eligibility.get("broad_table_only_count", 0) or 0),
        "broad_table_only_discarded_count": int((quality.get("discard_reason_counts") or {}).get("broad_table_only_semantically_weak", 0) or 0),
        "eligible_pages_with_atomic_artifact": int(eligibility.get("eligible_pages_with_atomic_artifact", 0) or 0),
    }


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    base = report["baseline_r029_metrics"]
    cur = report["r030_metrics"]
    lines = [
        "# R030 Atomic Quality Replay",
        "",
        "Scope: same 3 R028 failed pages only; no expansion, no activation scan, no QA, no graph, no rerank tuning. Replay outputs are probes and are not merged into cumulative artifacts.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Metrics",
        "",
        "| Metric | R029 atomic prompt | R030 atomic quality | Delta |",
        "|---|---:|---:|---:|",
    ]
    metric_order = [
        "parse_failure_count",
        "json_parse_success_count",
        "valid_artifacts",
        "discarded_artifacts",
        "strong_eligible_artifacts",
        "atomic_strong_eligible_artifacts",
        "eligible_pages",
        "eligible_pages_with_atomic_artifact",
        "mock_or_placeholder_content",
        "full_page_only_locator",
        "table_cell_count",
        "numeric_fact_count",
        "broad_table_only_count",
    ]
    for key in metric_order:
        delta = cur.get(key, 0) - base.get(key, 0)
        lines.append(f"| `{key}` | {base.get(key, 0)} | {cur.get(key, 0)} | {delta:+} |")
    lines.extend(["", "## Checks"])
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Scope Guard",
        "- Uses only Stage 2 output quality taxonomy; no gold fields.",
        "- Broad/table-title-only schema-valid artifacts are excluded from atomic strong eligibility.",
        "- Activation scan remains blocked unless atomic artifacts are stable on this bounded replay.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
