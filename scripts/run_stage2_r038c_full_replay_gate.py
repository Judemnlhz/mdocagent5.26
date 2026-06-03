#!/usr/bin/env python3
"""R038c repaired 20 -> 30 full replay gate.

This runner replays the original R028 20 -> 30 ten-page delta with the repaired
Stage 2 path. It then audits the repaired delta and a temporary cumulative20
plus repaired-delta artifact store, including diagnostic activation scans. It
does not write to the final cumulative store, run QA/effectiveness, use graph
context, or tune reranking.
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
DEFAULT_SUBSET = "outputs/stage2_structured_incremental/r028_20_to_30/subset_delta_20_to_30.jsonl"
DEFAULT_BASE_ARTIFACTS = "outputs/stage2_structured_incremental/r028_10_to_20/cumulative/artifacts.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/stage2_structured_incremental/r038c_repaired_20_to_30_full_replay_gate"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_CONFIG = "config/model/qwen3vl.yaml"
DEFAULT_PROMPT_VERSION = "artifact_compiler_prompt_v2_structured_real_gate"
DEFAULT_R038A_REPORT = "outputs/stage2_structured_incremental/r038a_repaired_20_to_30_noise_audit/r038a_noise_audit_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--base-artifacts", default=DEFAULT_BASE_ARTIFACTS)
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
    stage2_dir = output_root / "stage2_delta"
    delta_eligibility_json = output_root / "delta_eligibility_audit.json"
    delta_eligibility_md = output_root / "delta_eligibility_audit.md"
    merged_dir = output_root / "temporary_cumulative20_plus_repaired_delta"
    merged_artifacts = merged_dir / "artifacts.jsonl"
    merged_eligibility_json = merged_dir / "eligibility_audit.json"
    merged_eligibility_md = merged_dir / "eligibility_audit.md"
    atomic_dir = output_root / "activation_scan_review" / "atomic_only"
    atomic_artifacts = atomic_dir / "artifacts.jsonl"
    atomic_eligibility_json = atomic_dir / "eligibility_audit.json"
    atomic_eligibility_md = atomic_dir / "eligibility_audit.md"
    activation_dir = output_root / "activation_scan_review" / "activation_scan_atomic"
    report_json = output_root / "r038c_full_replay_gate_report.json"
    report_md = output_root / "r038c_full_replay_gate_report.md"
    page_count = count_subset_pages(Path(args.subset))

    commands = build_commands(
        args=args,
        page_count=page_count,
        stage2_dir=stage2_dir,
        delta_eligibility_json=delta_eligibility_json,
        delta_eligibility_md=delta_eligibility_md,
        merged_artifacts=merged_artifacts,
        merged_eligibility_json=merged_eligibility_json,
        merged_eligibility_md=merged_eligibility_md,
        atomic_eligibility_json=atomic_eligibility_json,
        atomic_eligibility_md=atomic_eligibility_md,
        activation_dir=activation_dir,
    )
    if not args.execute:
        print(json.dumps({"will_execute": False, "page_count": page_count, "commands": commands, "report_json": str(report_json)}, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    command_results = []
    command_results.append(run_command(commands[0], repo))
    command_results.append(run_command(commands[1], repo))

    merge_report = build_temporary_stores(
        base_path=Path(args.base_artifacts),
        delta_path=stage2_dir / "artifacts.jsonl",
        merged_path=merged_artifacts,
        atomic_path=atomic_artifacts,
    )
    write_json(output_root / "activation_scan_review" / "merge_report.json", merge_report)

    for command in commands[2:]:
        command_results.append(run_command(command, repo))

    quality = read_json(stage2_dir / "quality_report.json")
    delta_eligibility = read_json(delta_eligibility_json)
    merged_eligibility = read_json(merged_eligibility_json)
    atomic_eligibility = read_json(atomic_eligibility_json)
    activation = read_json(activation_dir / "real_structured_activation_scan_report.json")
    artifacts = read_jsonl(stage2_dir / "artifacts.jsonl")
    r038a = read_json(args.r038a_report)
    report = build_report(
        args=args,
        page_count=page_count,
        quality=quality,
        delta_eligibility=delta_eligibility,
        merged_eligibility=merged_eligibility,
        atomic_eligibility=atomic_eligibility,
        activation=activation,
        artifacts=artifacts,
        r038a=r038a,
        merge_report=merge_report,
        command_results=command_results,
    )
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(json.dumps({"decision": report["decision"], "provider_success": report["quality"]["provider_call_success_count"], "parse_failure": report["quality"]["parse_failure_count"], "delta_artifacts": report["delta_artifact_quality"]["total_artifacts"], "eligible_for_heldout": report["activation"]["eligible_for_heldout_count"]}, ensure_ascii=False, indent=2))


def build_commands(
    *,
    args: argparse.Namespace,
    page_count: int,
    stage2_dir: Path,
    delta_eligibility_json: Path,
    delta_eligibility_md: Path,
    merged_artifacts: Path,
    merged_eligibility_json: Path,
    merged_eligibility_md: Path,
    atomic_eligibility_json: Path,
    atomic_eligibility_md: Path,
    activation_dir: Path,
) -> list[list[str]]:
    return [
        [
            "python3", "scripts/stage2.py", "doc-compile",
            "--provider", "real", "--enable-real-api", "--run-real-trial", "--allow-real-subset",
            "--input", args.records, "--subset-file", args.subset, "--extract-root", args.extract_root,
            "--output-dir", str(stage2_dir), "--model-config", args.model_config, "--model-name", args.model_name,
            "--prompt-version", args.prompt_version, "--image-payload-mode", "image_url",
            "--max-pages", str(page_count), "--max-pages-total", str(page_count),
            "--max-pages-real-cap", str(page_count), "--max-pages-per-call", "1",
            "--max-docs", str(page_count), "--max-pages-per-doc", "2",
            "--timeout-seconds", str(args.timeout_seconds), "--max-retries", str(args.max_retries),
        ],
        [
            "python3", "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts", str(stage2_dir / "artifacts.jsonl"),
            "--output-json", str(delta_eligibility_json), "--output-md", str(delta_eligibility_md),
        ],
        [
            "python3", "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts", str(merged_artifacts),
            "--output-json", str(merged_eligibility_json), "--output-md", str(merged_eligibility_md),
        ],
        [
            "python3", "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts", str(atomic_eligibility_json.parent / "artifacts.jsonl"),
            "--output-json", str(atomic_eligibility_json), "--output-md", str(atomic_eligibility_md),
        ],
        [
            "python3", "scripts/scan_real_artifact_activation.py",
            "--artifacts", str(atomic_eligibility_json.parent / "artifacts.jsonl"),
            "--output-dir", str(activation_dir),
        ],
    ]


def build_temporary_stores(base_path: Path, delta_path: Path, merged_path: Path, atomic_path: Path) -> dict[str, Any]:
    base_rows = read_jsonl(base_path)
    delta_rows = read_jsonl(delta_path)
    delta_pages = {(str(row.get("doc_id") or ""), int(row.get("page_index", -1))) for row in delta_rows}
    base_kept = [row for row in base_rows if (str(row.get("doc_id") or ""), int(row.get("page_index", -1))) not in delta_pages]
    merged_rows = base_kept + delta_rows
    atomic_rows = [row for row in merged_rows if is_atomic_strong_eligible(row, artifact_rerank_eligibility_reason(row))]
    write_jsonl(merged_path, merged_rows)
    write_jsonl(atomic_path, atomic_rows)
    return {
        "schema_version": "r038c_temporary_cumulative20_plus_repaired_delta_v1",
        "base_artifacts": str(base_path),
        "delta_artifacts": str(delta_path),
        "merged_artifacts": str(merged_path),
        "atomic_artifacts": str(atomic_path),
        "base_artifact_count": len(base_rows),
        "base_kept_count": len(base_kept),
        "delta_artifact_count": len(delta_rows),
        "delta_page_count": len(delta_pages),
        "merged_artifact_count": len(merged_rows),
        "atomic_artifact_count": len(atomic_rows),
        "not_merged_into_final_cumulative_artifacts": True,
        "no_qa": True,
        "no_graph": True,
        "no_rerank_tuning": True,
    }


def build_report(
    *,
    args: argparse.Namespace,
    page_count: int,
    quality: dict[str, Any],
    delta_eligibility: dict[str, Any],
    merged_eligibility: dict[str, Any],
    atomic_eligibility: dict[str, Any],
    activation: dict[str, Any],
    artifacts: list[dict[str, Any]],
    r038a: dict[str, Any],
    merge_report: dict[str, Any],
    command_results: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_quality = summarize_artifacts(artifacts)
    r038a_expected = int((r038a.get("summary") or {}).get("total_artifacts", 0) or 0)
    observed = int(artifact_quality["total_artifacts"])
    growth = round(observed / r038a_expected, 6) if r038a_expected else None
    concentration = activation.get("concentration") if isinstance(activation.get("concentration"), dict) else {}
    activation_metrics = activation_metrics_from_report(activation)
    checks = {
        "provider_success_all_pages": int(quality.get("provider_call_success_count", 0) or 0) == page_count,
        "parse_failure_zero": int(quality.get("parse_failure_count", quality.get("provider_call_failed_count", 0)) or 0) == 0,
        "mock_or_placeholder_zero": int(delta_eligibility.get("mock_or_placeholder_content", 0) or 0) == 0,
        "full_page_only_locator_zero": int(delta_eligibility.get("full_page_only_locator", 0) or 0) == 0,
        "broad_table_only_zero_or_discarded": int(artifact_quality["broad_table_only_count"]) == 0,
        "atomic_artifact_present": int(artifact_quality["atomic_strong_eligible_artifacts"]) > 0,
        "artifact_growth_vs_r038a_bounded": growth is not None and growth <= 1.5,
        "eligible_for_heldout_at_least_r037": activation_metrics["eligible_for_heldout_count"] >= 103,
        "max_doc_share_acceptable": float(concentration.get("max_doc_share", 1.0) or 1.0) <= 0.35,
        "max_page_share_acceptable": float(concentration.get("max_page_share", 1.0) or 1.0) <= 0.25,
        "no_qa": True,
        "not_merged_into_final_cumulative_artifacts": True,
    }
    decision = "proceed_to_r039_heldout_activation_subset" if all(checks.values()) else "stop_before_heldout_review_replay_gate"
    return {
        "schema_version": "r038c_repaired_20_to_30_full_replay_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": {
            "full_replay_gate": True,
            "replays_original_r028_20_to_30_subset": True,
            "temporary_cumulative_store_only": True,
            "diagnostic_activation_scan": True,
            "no_final_artifact_store_merge": True,
            "no_qa": True,
            "no_effectiveness_claim": True,
            "no_graph": True,
            "no_rerank_tuning": True,
            "uses_gold_fields": False,
        },
        "inputs": {
            "records": args.records,
            "extract_root": args.extract_root,
            "subset": args.subset,
            "base_artifacts": args.base_artifacts,
            "model_config": args.model_config,
            "model_name": args.model_name,
            "prompt_version": args.prompt_version,
            "page_count": page_count,
        },
        "quality": quality_metrics_from_report(quality),
        "delta_eligibility": eligibility_metrics_from_report(delta_eligibility),
        "merged_eligibility": eligibility_metrics_from_report(merged_eligibility),
        "atomic_eligibility": eligibility_metrics_from_report(atomic_eligibility),
        "delta_artifact_quality": artifact_quality,
        "r038a_comparison": {
            "expected_offline_artifacts": r038a_expected,
            "observed_provider_artifacts": observed,
            "artifact_growth_vs_r038a": growth,
        },
        "merge_report": merge_report,
        "activation": activation_metrics,
        "concentration": concentration_metrics_from_report(concentration),
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


def quality_metrics_from_report(quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider_call_success_count": int(quality.get("provider_call_success_count", 0) or 0),
        "provider_call_failed_count": int(quality.get("provider_call_failed_count", 0) or 0),
        "json_parse_success_count": int(quality.get("json_parse_success_count", 0) or 0),
        "parse_failure_count": int(quality.get("parse_failure_count", quality.get("provider_call_failed_count", 0)) or 0),
        "num_valid_artifacts": int(quality.get("num_valid_artifacts", quality.get("num_artifacts", 0)) or 0),
        "num_discarded_artifacts": int(quality.get("num_discarded_artifacts", 0) or 0),
        "table_cell_artifact_count": int(quality.get("table_cell_artifact_count", 0) or 0),
        "numeric_fact_count": int(quality.get("numeric_fact_count", 0) or 0),
    }


def eligibility_metrics_from_report(eligibility: dict[str, Any]) -> dict[str, int]:
    return {
        "total_artifacts": int(eligibility.get("total_artifacts", 0) or 0),
        "eligible_artifacts": int(eligibility.get("eligible_artifacts", 0) or 0),
        "atomic_strong_eligible_artifacts": int(eligibility.get("atomic_strong_eligible_artifacts", 0) or 0),
        "numeric_fact_count": int(eligibility.get("numeric_fact_count", 0) or 0),
        "table_cell_count": int(eligibility.get("table_cell_count", 0) or 0),
        "eligible_pages_with_atomic_artifact": int(eligibility.get("eligible_pages_with_atomic_artifact", 0) or 0),
        "mock_or_placeholder_content": int(eligibility.get("mock_or_placeholder_content", 0) or 0),
        "full_page_only_locator": int(eligibility.get("full_page_only_locator", 0) or 0),
        "broad_table_only_count": int(eligibility.get("broad_table_only_count", 0) or 0),
    }


def activation_metrics_from_report(activation: dict[str, Any]) -> dict[str, Any]:
    return {
        "activated_count": int(activation.get("activated_count", 0) or 0),
        "eligible_for_heldout_count": int(activation.get("eligible_for_heldout_count", 0) or 0),
        "changed_count": int(activation.get("changed_count", 0) or 0),
        "original_plus_changed_count": int(activation.get("original_plus_changed_count", 0) or 0),
        "strong_eligible_page_count": int(activation.get("strong_eligible_page_count", 0) or 0),
        "heldout_available": bool((activation.get("heldout_activation_rich_subset") or {}).get("available", False)),
    }


def concentration_metrics_from_report(concentration: dict[str, Any]) -> dict[str, Any]:
    return {
        "activated_by_doc": concentration.get("activated_by_doc", {}),
        "activated_by_page": concentration.get("activated_by_page", {}),
        "max_doc_share": float(concentration.get("max_doc_share", 0.0) or 0.0),
        "max_page_share": float(concentration.get("max_page_share", 0.0) or 0.0),
        "effective_num_docs": concentration.get("effective_num_docs", 0),
        "effective_num_pages": concentration.get("effective_num_pages", 0),
    }


def next_step(decision: str) -> str:
    if decision == "proceed_to_r039_heldout_activation_subset":
        return "Construct R039 held-out activation-rich subset; do not run QA/effectiveness until that subset is frozen."
    return "Review R038c gate failures before constructing held-out subset or running any QA/effectiveness gate."


def count_subset_pages(path: Path) -> int:
    count = 0
    for row in read_jsonl(path):
        page_indices = row.get("page_indices")
        if isinstance(page_indices, list):
            count += len(page_indices)
        elif row.get("page_index") is not None:
            count += 1
    if count <= 0:
        raise RuntimeError(f"No pages found in subset: {path}")
    return count


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        row = {"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]}
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return {"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-500:], "stderr_tail": completed.stderr[-500:]}


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    quality = report["quality"]
    artifact_quality = report["delta_artifact_quality"]
    activation = report["activation"]
    concentration = report["concentration"]
    comparison = report["r038a_comparison"]
    lines = [
        "# R038c Repaired 20 -> 30 Full Replay Gate",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Scope",
        "- Full replay of the original R028 20 -> 30 ten-page delta.",
        "- Temporary cumulative20 plus repaired-delta store for diagnostic activation only.",
        "- No final artifact-store merge, QA, graph, effectiveness claim, or rerank tuning.",
        "- Model key remains environment-only; no config or key file is written by this runner.",
        "",
        "## Provider Quality",
        f"- Provider success/fail: {quality['provider_call_success_count']} / {quality['provider_call_failed_count']}",
        f"- JSON parse success: {quality['json_parse_success_count']}",
        f"- Parse failures: {quality['parse_failure_count']}",
        f"- Valid/discarded artifacts: {quality['num_valid_artifacts']} / {quality['num_discarded_artifacts']}",
        "",
        "## Delta Artifact Quality",
        f"- Total artifacts: {artifact_quality['total_artifacts']}",
        f"- Atomic strong eligible: {artifact_quality['atomic_strong_eligible_artifacts']}",
        f"- Type counts: `{json.dumps(artifact_quality['type_counts'], sort_keys=True)}`",
        f"- Quality counts: `{json.dumps(artifact_quality['quality_counts'], sort_keys=True)}`",
        f"- R038a expected artifacts: {comparison['expected_offline_artifacts']}",
        f"- Artifact growth vs R038a: {comparison['artifact_growth_vs_r038a']}",
        "",
        "## Temporary Activation",
        f"- Activated records: {activation['activated_count']}",
        f"- Eligible for held-out: {activation['eligible_for_heldout_count']}",
        f"- Changed records, artifact_only: {activation['changed_count']}",
        f"- Changed records, original_plus_artifact: {activation['original_plus_changed_count']}",
        f"- Held-out available: `{activation['heldout_available']}`",
        f"- Max doc share: {concentration['max_doc_share']:.4f}",
        f"- Max page share: {concentration['max_page_share']:.4f}",
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
