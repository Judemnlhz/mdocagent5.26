#!/usr/bin/env python3
"""R038d no-provider activation attribution audit.

This audit compares R037 targeted coverage and R038c repaired 20 -> 30 replay
activation behavior by building temporary atomic-only stores and activation
scans. It does not call a provider, compile Stage 2, merge final artifacts, run
QA/effectiveness, use graph context, or tune reranking.
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
from mdocnexus.stage2.artifact_quality import is_atomic_strong_eligible  # noqa: E402


DEFAULT_BASE_ARTIFACTS = "outputs/stage2_structured_incremental/r028_10_to_20/cumulative/artifacts.jsonl"
DEFAULT_R037_ARTIFACTS = "outputs/stage2_structured_incremental/r037_budgeted_targeted_coverage/stage2_delta/artifacts.jsonl"
DEFAULT_R038C_ARTIFACTS = "outputs/stage2_structured_incremental/r038c_repaired_20_to_30_full_replay_gate/stage2_delta/artifacts.jsonl"
DEFAULT_R037_SCAN = "outputs/stage2_structured_incremental/r037_budgeted_targeted_coverage/activation_scan_review/activation_scan_atomic/real_structured_activation_scan_report.json"
DEFAULT_R038C_SCAN = "outputs/stage2_structured_incremental/r038c_repaired_20_to_30_full_replay_gate/activation_scan_review/activation_scan_atomic/real_structured_activation_scan_report.json"
DEFAULT_OUTPUT_ROOT = "outputs/stage2_structured_incremental/r038d_activation_attribution_audit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifacts", default=DEFAULT_BASE_ARTIFACTS)
    parser.add_argument("--r037-artifacts", default=DEFAULT_R037_ARTIFACTS)
    parser.add_argument("--r038c-artifacts", default=DEFAULT_R038C_ARTIFACTS)
    parser.add_argument("--r037-scan", default=DEFAULT_R037_SCAN)
    parser.add_argument("--r038c-scan", default=DEFAULT_R038C_SCAN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    variants = {
        "cumulative20_plus_r037": [Path(args.base_artifacts), Path(args.r037_artifacts)],
        "cumulative20_plus_r038c": [Path(args.base_artifacts), Path(args.r038c_artifacts)],
        "cumulative20_plus_r037_plus_r038c": [Path(args.base_artifacts), Path(args.r037_artifacts), Path(args.r038c_artifacts)],
    }
    commands = [
        ["python3", "scripts/scan_real_artifact_activation.py", "--artifacts", str(output_root / name / "atomic_only" / "artifacts.jsonl"), "--output-dir", str(output_root / name / "activation_scan")]
        for name in variants
    ]
    report_json = output_root / "r038d_activation_attribution_report.json"
    report_md = output_root / "r038d_activation_attribution_report.md"
    if not args.execute:
        print(json.dumps({"will_execute": False, "variants": list(variants), "commands": commands, "report_json": str(report_json)}, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    merge_reports: dict[str, Any] = {}
    for name, paths in variants.items():
        merge_reports[name] = build_variant_store(paths, output_root / name)
    command_results = [run_command(command, repo) for command in commands]
    scans = {
        name: read_json(output_root / name / "activation_scan" / "real_structured_activation_scan_report.json")
        for name in variants
    }
    source_scans = {
        "r037_existing": read_json(args.r037_scan),
        "r038c_existing": read_json(args.r038c_scan),
    }
    report = build_report(args, merge_reports, scans, source_scans, command_results)
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(json.dumps({"decision": report["decision"], "r037_unique_activated": report["overlap"]["r037_unique_activated_count"], "r038c_unique_activated": report["overlap"]["r038c_unique_activated_count"], "union_eligible_for_heldout": report["variant_metrics"]["cumulative20_plus_r037_plus_r038c"]["eligible_for_heldout_count"]}, ensure_ascii=False, indent=2))


def build_variant_store(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    merged_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    source_counts: dict[str, int] = {}
    for path in paths:
        rows = read_jsonl(path)
        source_counts[str(path)] = len(rows)
        for row in rows:
            key = artifact_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged_rows.append(row)
    atomic_rows = [row for row in merged_rows if is_atomic_strong_eligible(row, artifact_rerank_eligibility_reason(row))]
    write_jsonl(output_dir / "merged_all" / "artifacts.jsonl", merged_rows)
    write_jsonl(output_dir / "atomic_only" / "artifacts.jsonl", atomic_rows)
    return {
        "sources": [str(path) for path in paths],
        "source_counts": source_counts,
        "merged_artifact_count": len(merged_rows),
        "atomic_artifact_count": len(atomic_rows),
        "atomic_page_count": len({page_key(row) for row in atomic_rows}),
        "atomic_type_counts": dict(sorted(Counter(str(row.get("artifact_type") or "") for row in atomic_rows).items())),
        "not_final_cumulative_store": True,
    }


def build_report(args: argparse.Namespace, merge_reports: dict[str, Any], scans: dict[str, dict[str, Any]], source_scans: dict[str, dict[str, Any]], command_results: list[dict[str, Any]]) -> dict[str, Any]:
    r037_ids = activated_ids(scans["cumulative20_plus_r037"])
    r038c_ids = activated_ids(scans["cumulative20_plus_r038c"])
    union_ids = activated_ids(scans["cumulative20_plus_r037_plus_r038c"])
    r037_heldout = heldout_ids(scans["cumulative20_plus_r037"])
    r038c_heldout = heldout_ids(scans["cumulative20_plus_r038c"])
    union_heldout = heldout_ids(scans["cumulative20_plus_r037_plus_r038c"])
    overlap = {
        "r037_activated_count": len(r037_ids),
        "r038c_activated_count": len(r038c_ids),
        "union_activated_count": len(union_ids),
        "activated_overlap_count": len(r037_ids & r038c_ids),
        "r037_unique_activated_count": len(r037_ids - r038c_ids),
        "r038c_unique_activated_count": len(r038c_ids - r037_ids),
        "union_new_over_r037_count": len(union_ids - r037_ids),
        "r037_heldout_count": len(r037_heldout),
        "r038c_heldout_count": len(r038c_heldout),
        "union_heldout_count": len(union_heldout),
        "heldout_overlap_count": len(r037_heldout & r038c_heldout),
        "r037_unique_heldout_count": len(r037_heldout - r038c_heldout),
        "r038c_unique_heldout_count": len(r038c_heldout - r037_heldout),
        "union_new_heldout_over_r037_count": len(union_heldout - r037_heldout),
    }
    variant_metrics = {name: scan_metrics(scan) for name, scan in scans.items()}
    union = variant_metrics["cumulative20_plus_r037_plus_r038c"]
    r038c = variant_metrics["cumulative20_plus_r038c"]
    r037 = variant_metrics["cumulative20_plus_r037"]
    union_reliance_on_r037 = (overlap["r037_unique_activated_count"] + overlap["activated_overlap_count"]) / max(union["activated_count"], 1)
    checks = {
        "no_provider_calls": True,
        "no_stage2_compile": True,
        "no_qa": True,
        "union_heldout_available": bool(union["heldout_available"]),
        "union_eligible_for_heldout_at_least_30": union["eligible_for_heldout_count"] >= 30,
        "r038c_adds_activation_over_r037": overlap["union_new_over_r037_count"] > 0,
        "r038c_standalone_below_heldout_gate": r038c["eligible_for_heldout_count"] < 30,
        "r037_remains_primary_activation_source": union_reliance_on_r037 >= 0.75,
    }
    decision = "freeze_r037_based_heldout_with_targeted_coverage_caveat" if checks["union_heldout_available"] and checks["r037_remains_primary_activation_source"] else "design_next_targeted_coverage_batch_before_heldout"
    return {
        "schema_version": "r038d_activation_attribution_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": {
            "no_provider_calls": True,
            "no_stage2_compile": True,
            "temporary_stores_only": True,
            "activation_scan_diagnostic_only": True,
            "no_qa": True,
            "no_effectiveness_claim": True,
            "no_graph": True,
            "no_rerank_tuning": True,
            "uses_gold_fields": False,
        },
        "inputs": {
            "base_artifacts": args.base_artifacts,
            "r037_artifacts": args.r037_artifacts,
            "r038c_artifacts": args.r038c_artifacts,
            "r037_existing_scan": args.r037_scan,
            "r038c_existing_scan": args.r038c_scan,
        },
        "merge_reports": merge_reports,
        "variant_metrics": variant_metrics,
        "overlap": overlap,
        "union_reliance_on_r037": round(union_reliance_on_r037, 6),
        "top_pages": {name: top_counts((scan.get("concentration") or {}).get("activated_by_page", {}), 20) for name, scan in scans.items()},
        "top_docs": {name: top_counts((scan.get("concentration") or {}).get("activated_by_doc", {}), 20) for name, scan in scans.items()},
        "checks": checks,
        "source_scan_consistency": {
            "r037_existing_activated_count": int(source_scans["r037_existing"].get("activated_count", 0) or 0),
            "r037_recomputed_activated_count": r037["activated_count"],
            "r038c_existing_activated_count": int(source_scans["r038c_existing"].get("activated_count", 0) or 0),
            "r038c_recomputed_activated_count": r038c["activated_count"],
        },
        "commands": command_results,
        "next_step": next_step(decision),
    }


def scan_metrics(scan: dict[str, Any]) -> dict[str, Any]:
    concentration = scan.get("concentration") if isinstance(scan.get("concentration"), dict) else {}
    heldout = scan.get("heldout_activation_rich_subset") if isinstance(scan.get("heldout_activation_rich_subset"), dict) else {}
    return {
        "activated_count": int(scan.get("activated_count", 0) or 0),
        "eligible_for_heldout_count": int(scan.get("eligible_for_heldout_count", 0) or 0),
        "changed_count": int(scan.get("changed_count", 0) or 0),
        "original_plus_changed_count": int(scan.get("original_plus_changed_count", 0) or 0),
        "strong_eligible_page_count": int(scan.get("strong_eligible_page_count", 0) or 0),
        "heldout_available": bool(heldout.get("available", False)),
        "heldout_num_records": int(heldout.get("num_records", 0) or 0),
        "max_doc_share": float(concentration.get("max_doc_share", 0.0) or 0.0),
        "max_page_share": float(concentration.get("max_page_share", 0.0) or 0.0),
        "effective_num_docs": concentration.get("effective_num_docs", 0),
        "effective_num_pages": concentration.get("effective_num_pages", 0),
    }


def next_step(decision: str) -> str:
    if decision == "freeze_r037_based_heldout_with_targeted_coverage_caveat":
        return "Freeze a held-out subset using union attribution, but explicitly label targeted coverage as necessary; still run no QA until subset IDs are committed."
    return "Do another offline targeted selection/noise audit before any provider spend or held-out freeze."


def activated_ids(scan: dict[str, Any]) -> set[int]:
    return {int(value) for value in scan.get("activated_ids", [])}


def heldout_ids(scan: dict[str, Any]) -> set[int]:
    heldout = scan.get("heldout_activation_rich_subset")
    if not isinstance(heldout, dict):
        return set()
    return {int(value) for value in heldout.get("record_ids", [])}


def artifact_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("doc_id") or ""),
        int(row.get("page_index", -1)),
        str(row.get("artifact_type") or ""),
        str(row.get("content") or ""),
    )


def page_key(row: dict[str, Any]) -> str:
    return f"{row.get('doc_id')}#p{int(row.get('page_index', 0) or 0):03d}"


def top_counts(counts: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [{"key": str(key), "count": int(value)} for key, value in counts.items()]
    return sorted(rows, key=lambda row: (-row["count"], row["key"]))[:limit]


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
    metrics = report["variant_metrics"]
    overlap = report["overlap"]
    lines = [
        "# R038d Activation Attribution Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Scope",
        "- No-provider attribution audit over existing artifacts only.",
        "- Temporary atomic-only stores and diagnostic activation scans.",
        "- No Stage 2 compile, final artifact merge, QA, graph, effectiveness claim, or rerank tuning.",
        "",
        "## Variant Metrics",
    ]
    for name, row in metrics.items():
        lines.extend([
            f"### {name}",
            f"- Activated records: {row['activated_count']}",
            f"- Eligible for held-out: {row['eligible_for_heldout_count']}",
            f"- Held-out available: `{row['heldout_available']}` ({row['heldout_num_records']} records)",
            f"- Changed artifact_only / original_plus: {row['changed_count']} / {row['original_plus_changed_count']}",
            f"- Strong eligible pages: {row['strong_eligible_page_count']}",
            f"- Max doc/page share: {row['max_doc_share']:.4f} / {row['max_page_share']:.4f}",
            "",
        ])
    lines.extend([
        "## Overlap",
        f"- R037 activated: {overlap['r037_activated_count']}",
        f"- R038c activated: {overlap['r038c_activated_count']}",
        f"- Union activated: {overlap['union_activated_count']}",
        f"- Activated overlap: {overlap['activated_overlap_count']}",
        f"- R037 unique activated: {overlap['r037_unique_activated_count']}",
        f"- R038c unique activated: {overlap['r038c_unique_activated_count']}",
        f"- Union new over R037: {overlap['union_new_over_r037_count']}",
        f"- Union held-out records: {overlap['union_heldout_count']}",
        f"- Union new held-out over R037: {overlap['union_new_heldout_over_r037_count']}",
        f"- Union reliance on R037: {report['union_reliance_on_r037']}",
        "",
        "## Checks",
    ])
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Next Step", report["next_step"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
