#!/usr/bin/env python3
"""R038b small repaired provider gate for selected R028 20 -> 30 pages.

Runs a tiny real-provider Stage 2 compile on three R028 pages that previously
failed parsing but passed R038a's offline noise audit. It does not merge
artifacts, run activation, run QA, use graph context, or tune reranking.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdocnexus.integration.mdocagent_adapter import artifact_rerank_eligibility_reason  # noqa: E402
from mdocnexus.stage2.artifact_quality import classify_artifact_quality, is_atomic_strong_eligible  # noqa: E402


DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_EXTRACT_ROOT = "tmp/MMLongBench"
DEFAULT_OUTPUT_ROOT = "outputs/stage2_structured_incremental/r038b_small_repaired_provider_gate"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_CONFIG = "config/model/qwen3vl.yaml"
DEFAULT_PROMPT_VERSION = "artifact_compiler_prompt_v2_structured_real_gate"
DEFAULT_R038A_REPORT = "outputs/stage2_structured_incremental/r038a_repaired_20_to_30_noise_audit/r038a_noise_audit_report.json"
SELECTED_PAGES = [
    {"doc_id": "2401.18059v1.pdf", "page_indices": [6]},
    {"doc_id": "936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf", "page_indices": [3]},
    {"doc_id": "BESTBUY_2023_10K.pdf", "page_indices": [26]},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--r038a-report", default=DEFAULT_R038A_REPORT)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    subset = output_root / "subset_r038b_small_provider_gate.jsonl"
    stage2_dir = output_root / "stage2_delta"
    eligibility_json = output_root / "eligibility_audit.json"
    eligibility_md = output_root / "eligibility_audit.md"
    report_json = output_root / "r038b_small_provider_gate_report.json"
    report_md = output_root / "r038b_small_provider_gate_report.md"
    commands = build_commands(args, subset, stage2_dir, eligibility_json, eligibility_md)

    if not args.execute:
        print(json.dumps({"will_execute": False, "selected_pages": selected_page_keys(), "commands": commands, "report_json": str(report_json)}, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset, subset_rows())
    command_results = [run_command(command, repo) for command in commands]
    quality = read_json(stage2_dir / "quality_report.json")
    eligibility = read_json(eligibility_json)
    artifacts = read_jsonl(stage2_dir / "artifacts.jsonl")
    r038a = read_json(args.r038a_report)
    report = build_report(args, quality, eligibility, artifacts, r038a, command_results)
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(json.dumps({"decision": report["decision"], "provider_success": report["quality"]["provider_call_success_count"], "parse_failure": report["quality"]["parse_failure_count"], "total_artifacts": report["artifact_quality"]["total_artifacts"], "atomic_strong_eligible": report["artifact_quality"]["atomic_strong_eligible_artifacts"]}, ensure_ascii=False, indent=2))


def build_commands(args: argparse.Namespace, subset: Path, stage2_dir: Path, eligibility_json: Path, eligibility_md: Path) -> list[list[str]]:
    page_count = len(selected_page_keys())
    return [
        [
            "python3", "scripts/stage2.py", "doc-compile",
            "--provider", "real", "--enable-real-api", "--run-real-trial", "--allow-real-subset",
            "--input", args.records, "--subset-file", str(subset), "--extract-root", args.extract_root,
            "--output-dir", str(stage2_dir), "--model-config", args.model_config, "--model-name", args.model_name,
            "--prompt-version", args.prompt_version, "--image-payload-mode", "image_url",
            "--max-pages", str(page_count), "--max-pages-total", str(page_count),
            "--max-pages-real-cap", str(page_count), "--max-pages-per-call", "1",
            "--max-docs", str(page_count), "--max-pages-per-doc", "1",
            "--timeout-seconds", str(args.timeout_seconds), "--max-retries", str(args.max_retries),
        ],
        [
            "python3", "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts", str(stage2_dir / "artifacts.jsonl"),
            "--output-json", str(eligibility_json), "--output-md", str(eligibility_md),
        ],
    ]


def build_report(args: argparse.Namespace, quality: dict[str, Any], eligibility: dict[str, Any], artifacts: list[dict[str, Any]], r038a: dict[str, Any], command_results: list[dict[str, Any]]) -> dict[str, Any]:
    artifact_quality = summarize_artifacts(artifacts)
    r038a_by_page = {str(row.get("page_key")): int(row.get("artifact_count", 0) or 0) for row in r038a.get("page_reports", []) if isinstance(row, dict)}
    expected = sum(r038a_by_page.get(page, 0) for page in selected_page_keys())
    observed = int(artifact_quality["total_artifacts"])
    growth = round(observed / expected, 6) if expected else None
    checks = {
        "provider_success_all_pages": int(quality.get("provider_call_success_count", 0) or 0) == len(selected_page_keys()),
        "parse_failure_zero": int(quality.get("parse_failure_count", quality.get("provider_call_failed_count", 0)) or 0) == 0,
        "mock_or_placeholder_zero": int(eligibility.get("mock_or_placeholder_content", 0) or 0) == 0,
        "full_page_only_locator_zero": int(eligibility.get("full_page_only_locator", 0) or 0) == 0,
        "broad_table_only_zero": int(artifact_quality["broad_table_only_count"]) == 0,
        "atomic_artifact_present": int(artifact_quality["atomic_strong_eligible_artifacts"]) > 0,
        "artifact_growth_vs_r038a_bounded": growth is not None and growth <= 1.5,
        "no_activation_scan": True,
        "not_merged_into_cumulative_artifacts": True,
    }
    decision = "proceed_to_repaired_20_to_30_full_replay_gate" if all(checks.values()) else "stop_repair_provider_noise_before_full_replay"
    return {
        "schema_version": "r038b_small_repaired_provider_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": {
            "small_real_provider_gate": True,
            "selected_from_r028_failed_pages": True,
            "no_artifact_store_merge": True,
            "no_activation_scan": True,
            "no_qa": True,
            "no_effectiveness_claim": True,
            "no_graph": True,
            "no_rerank_tuning": True,
            "uses_gold_fields": False,
        },
        "inputs": {
            "records": args.records,
            "extract_root": args.extract_root,
            "model_config": args.model_config,
            "model_name": args.model_name,
            "prompt_version": args.prompt_version,
            "r038a_report": args.r038a_report,
            "selected_pages": selected_page_keys(),
        },
        "quality": {
            "provider_call_success_count": int(quality.get("provider_call_success_count", 0) or 0),
            "provider_call_failed_count": int(quality.get("provider_call_failed_count", 0) or 0),
            "json_parse_success_count": int(quality.get("json_parse_success_count", 0) or 0),
            "parse_failure_count": int(quality.get("parse_failure_count", quality.get("provider_call_failed_count", 0)) or 0),
            "num_valid_artifacts": int(quality.get("num_valid_artifacts", quality.get("num_artifacts", 0)) or 0),
            "num_discarded_artifacts": int(quality.get("num_discarded_artifacts", 0) or 0),
        },
        "eligibility": eligibility,
        "artifact_quality": artifact_quality,
        "r038a_comparison": {
            "expected_offline_artifacts_for_selected_pages": expected,
            "observed_provider_artifacts": observed,
            "artifact_growth_vs_r038a": growth,
        },
        "checks": checks,
        "commands": command_results,
        "next_step": next_step(decision),
    }


def summarize_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    page_counts: Counter[str] = Counter()
    atomic_strong = 0
    for artifact in artifacts:
        type_counts[str(artifact.get("artifact_type") or "")] += 1
        page_counts[f"{artifact.get('doc_id')}#p{int(artifact.get('page_index', 0) or 0):03d}"] += 1
        quality = classify_artifact_quality(artifact)
        quality_counts.update(str(label) for label in quality.get("labels", []))
        if is_atomic_strong_eligible(artifact, artifact_rerank_eligibility_reason(artifact)):
            atomic_strong += 1
    return {
        "total_artifacts": len(artifacts),
        "type_counts": dict(sorted(type_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "page_counts": dict(sorted(page_counts.items())),
        "atomic_strong_eligible_artifacts": atomic_strong,
        "broad_table_only_count": int(quality_counts.get("broad_table_only", 0) or 0),
        "weak_locator_count": int(quality_counts.get("weak_locator", 0) or 0),
    }


def next_step(decision: str) -> str:
    if decision == "proceed_to_repaired_20_to_30_full_replay_gate":
        return "Run the repaired 20 -> 30 full replay gate only; still do not run QA/effectiveness until that gate passes."
    return "Inspect provider outputs, prompt/parser behavior, and atomicizer quality before replaying all 10 pages."


def subset_rows() -> list[dict[str, Any]]:
    return [{**row, "selection_source": "r038b_small_repaired_provider_gate", "selection_reasons": ["r028_parse_failure_page", "r038a_offline_noise_passed", "bounded_small_provider_gate"]} for row in SELECTED_PAGES]


def selected_page_keys() -> list[str]:
    return [f"{row['doc_id']}#p{int(page):03d}" for row in SELECTED_PAGES for page in row["page_indices"]]


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        row = {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:],
            "stderr_tail": completed.stderr[-3000:],
        }
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-500:],
        "stderr_tail": completed.stderr[-500:],
    }


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    quality = report["quality"]
    artifact_quality = report["artifact_quality"]
    comparison = report["r038a_comparison"]
    lines = [
        "# R038b Small Repaired Provider Gate",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Scope",
        "- Tiny real-provider Stage 2 compile on 3 selected R028 failed pages.",
        "- No artifact-store merge, activation scan, QA, graph, effectiveness claim, or rerank tuning.",
        "- Model key remains environment-only; no config or key file is written by this runner.",
        "",
        "## Selected Pages",
        *[f"- {page}" for page in report["inputs"]["selected_pages"]],
        "",
        "## Provider Quality",
        f"- Provider success/fail: {quality['provider_call_success_count']} / {quality['provider_call_failed_count']}",
        f"- JSON parse success: {quality['json_parse_success_count']}",
        f"- Parse failures: {quality['parse_failure_count']}",
        f"- Valid/discarded artifacts: {quality['num_valid_artifacts']} / {quality['num_discarded_artifacts']}",
        "",
        "## Artifact Quality",
        f"- Total artifacts: {artifact_quality['total_artifacts']}",
        f"- Atomic strong eligible: {artifact_quality['atomic_strong_eligible_artifacts']}",
        f"- Type counts: `{json.dumps(artifact_quality['type_counts'], sort_keys=True)}`",
        f"- Quality counts: `{json.dumps(artifact_quality['quality_counts'], sort_keys=True)}`",
        f"- Page counts: `{json.dumps(artifact_quality['page_counts'], sort_keys=True)}`",
        f"- R038a expected artifacts for selected pages: {comparison['expected_offline_artifacts_for_selected_pages']}",
        f"- Artifact growth vs R038a: {comparison['artifact_growth_vs_r038a']}",
        "",
        "## Checks",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Next Step", report["next_step"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
