#!/usr/bin/env python3
"""R031 bounded activation scan review.

This diagnostic constructs a temporary artifact store from cumulative20 plus
R030 repaired same-page artifacts, filters it to atomic strong-eligible
artifacts, and runs the activation scan. It does not run QA, graph expansion,
rerank tuning, or full ablation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdocnexus.integration.mdocagent_adapter import artifact_rerank_eligibility_reason  # noqa: E402
from mdocnexus.stage2.artifact_quality import classify_artifact_quality, is_atomic_strong_eligible  # noqa: E402


DEFAULT_BASE_ARTIFACTS = "outputs/stage2_structured_incremental/r028_10_to_20/cumulative/artifacts.jsonl"
DEFAULT_R030_ARTIFACTS = "outputs/stage2_structured_incremental/r028_20_to_30/r030_atomic_quality_replay_3/stage2_delta/artifacts.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/stage2_structured_incremental/r031_activation_scan_review"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifacts", default=DEFAULT_BASE_ARTIFACTS)
    parser.add_argument("--r030-artifacts", default=DEFAULT_R030_ARTIFACTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    merged_artifacts = output_root / "merged_cumulative20_plus_r030" / "artifacts.jsonl"
    atomic_artifacts = output_root / "atomic_only" / "artifacts.jsonl"
    boundary_json = output_root / "fallback_boundary_audit.json"
    boundary_md = output_root / "fallback_boundary_audit.md"
    merge_report_json = output_root / "merge_report.json"
    activation_all_dir = output_root / "activation_scan_merged_all"
    activation_dir = output_root / "activation_scan_atomic"
    eligibility_all_json = output_root / "merged_cumulative20_plus_r030" / "eligibility_audit.json"
    eligibility_all_md = output_root / "merged_cumulative20_plus_r030" / "eligibility_audit.md"
    eligibility_json = output_root / "atomic_only" / "eligibility_audit.json"
    eligibility_md = output_root / "atomic_only" / "eligibility_audit.md"
    report_json = output_root / "r031_activation_scan_review_report.json"
    report_md = output_root / "r031_activation_scan_review_report.md"

    commands = [
        [
            "python3",
            "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts",
            str(merged_artifacts),
            "--output-json",
            str(eligibility_all_json),
            "--output-md",
            str(eligibility_all_md),
        ],
        [
            "python3",
            "scripts/scan_real_artifact_activation.py",
            "--artifacts",
            str(merged_artifacts),
            "--output-dir",
            str(activation_all_dir),
        ],
        [
            "python3",
            "scripts/audit_artifact_rerank_eligibility.py",
            "--artifacts",
            str(atomic_artifacts),
            "--output-json",
            str(eligibility_json),
            "--output-md",
            str(eligibility_md),
        ],
        [
            "python3",
            "scripts/scan_real_artifact_activation.py",
            "--artifacts",
            str(atomic_artifacts),
            "--output-dir",
            str(activation_dir),
        ],
    ]
    if not args.execute:
        print(json.dumps({"will_execute": False, "commands": commands, "report": str(report_json)}, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    boundary = build_fallback_boundary_audit(repo / "scripts" / "stage2.py")
    write_json(boundary_json, boundary)
    write_boundary_markdown(boundary_md, boundary)

    merge_report = build_temporary_artifact_stores(
        base_path=Path(args.base_artifacts),
        r030_path=Path(args.r030_artifacts),
        merged_path=merged_artifacts,
        atomic_path=atomic_artifacts,
    )
    write_json(merge_report_json, merge_report)

    command_results = [run_command(command, repo) for command in commands]
    eligibility_all = read_json(eligibility_all_json)
    activation_all = read_json(activation_all_dir / "real_structured_activation_scan_report.json")
    eligibility = read_json(eligibility_json)
    activation = read_json(activation_dir / "real_structured_activation_scan_report.json")
    report = build_report(
        args=args,
        boundary=boundary,
        merge_report=merge_report,
        eligibility_all=eligibility_all,
        activation_all=activation_all,
        eligibility=eligibility,
        activation=activation,
        command_results=command_results,
    )
    write_json(report_json, report)
    write_markdown(report_md, report)
    print(
        json.dumps(
            {
        "decision": report["decision"],
        "atomic_activated_count": report["atomic_only_activation_metrics"]["activated_count"],
        "atomic_eligible_for_heldout_count": report["atomic_only_activation_metrics"]["eligible_for_heldout_count"],
        "merged_all_activated_count": report["merged_all_activation_metrics"]["activated_count"],
        "merged_all_eligible_for_heldout_count": report["merged_all_activation_metrics"]["eligible_for_heldout_count"],
        "atomic_max_doc_share": report["atomic_only_concentration"]["max_doc_share"],
        "atomic_max_page_share": report["atomic_only_concentration"]["max_page_share"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_fallback_boundary_audit(stage2_path: Path) -> dict[str, Any]:
    code = stage2_path.read_text(encoding="utf-8")
    marker = "def _build_deterministic_numeric_fact_artifacts"
    if marker not in code:
        return {
            "schema_version": "r031_fallback_boundary_audit_v3",
            "scope": "Stage 2 default artifact write path",
            "function_block": None,
            "uses_page_text": False,
            "requires_document_generic": False,
            "uses_layout_locator": False,
            "uses_selected_page_identity": False,
            "forbidden_term_hits": {},
            "reads_question_answer_gold_evidence": False,
            "decision": "pass_fallback_removed_from_default_stage2_path",
            "notes": [
                "The R030 deterministic numeric fallback is no longer present in scripts/stage2.py.",
                "Stage 2 now writes provider-validated artifacts only in the default path.",
                "This audit remains diagnostic only and does not run activation, QA, graph, or rerank tuning.",
            ],
        }
    start = code.index(marker)
    end = code.index("def _has_artifact_locator", start)
    block = code[start:end]
    forbidden_terms = ["question", "answer", "gold", "evidence_pages", "evidence_sources", "binary_correctness"]
    forbidden_hits = {term: block.count(term) for term in forbidden_terms if block.count(term)}
    return {
        "schema_version": "r031_fallback_boundary_audit_v2",
        "scope": "R030 deterministic OCR numeric fallback only",
        "function_block": "_build_deterministic_numeric_fact_artifacts through numeric helpers",
        "uses_page_text": "page_text = page_input.get" in block,
        "requires_document_generic": "if not document_generic:" in block,
        "uses_layout_locator": "_primary_text_locator(page_input)" in block,
        "uses_selected_page_identity": all(term in block for term in ['selected_page["doc_id"]', 'selected_page["page_index"]']),
        "forbidden_term_hits": forbidden_hits,
        "reads_question_answer_gold_evidence": bool(forbidden_hits),
        "decision": "pass_public_safe_ocr_only" if not forbidden_hits else "review_required",
        "notes": [
            "Fallback is gated to document_generic mode.",
            "Fallback reads OCR page_text and page layout locator only.",
            "selected_page is used for doc_id/page_index identity, not question or answer fields.",
            "No activation, QA, graph, or rerank tuning is performed by this audit.",
        ],
    }


def build_temporary_artifact_stores(base_path: Path, r030_path: Path, merged_path: Path, atomic_path: Path) -> dict[str, Any]:
    base_rows = read_jsonl(base_path)
    r030_rows = read_jsonl(r030_path)
    r030_pages = {(str(row.get("doc_id") or ""), int(row.get("page_index", -1))) for row in r030_rows}
    base_kept = [row for row in base_rows if (str(row.get("doc_id") or ""), int(row.get("page_index", -1))) not in r030_pages]
    merged_rows = base_kept + r030_rows
    atomic_rows = [row for row in merged_rows if is_atomic_strong_eligible(row, artifact_rerank_eligibility_reason(row))]

    write_jsonl(merged_path, merged_rows)
    write_jsonl(atomic_path, atomic_rows)
    return {
        "schema_version": "r031_temporary_artifact_store_v1",
        "base_artifacts": str(base_path),
        "r030_artifacts": str(r030_path),
        "merged_artifacts": str(merged_path),
        "atomic_artifacts": str(atomic_path),
        "base_artifact_count": len(base_rows),
        "base_kept_count": len(base_kept),
        "r030_artifact_count": len(r030_rows),
        "r030_replaced_pages": sorted(f"{doc_id}#p{page_index:03d}" for doc_id, page_index in r030_pages),
        "merged_artifact_count": len(merged_rows),
        "atomic_artifact_count": len(atomic_rows),
        "atomic_artifact_type_counts": dict(sorted(Counter(str(row.get("artifact_type") or "") for row in atomic_rows).items())),
        "atomic_pages": sorted(f"{row.get('doc_id')}#p{int(row.get('page_index')):03d}" for row in atomic_rows),
        "no_qa_run": True,
        "no_graph": True,
        "no_rerank_tuning": True,
        "not_merged_into_cumulative_artifacts": True,
    }


def build_report(
    *,
    args: argparse.Namespace,
    boundary: dict[str, Any],
    merge_report: dict[str, Any],
    eligibility_all: dict[str, Any],
    activation_all: dict[str, Any],
    eligibility: dict[str, Any],
    activation: dict[str, Any],
    command_results: list[dict[str, Any]],
) -> dict[str, Any]:
    concentration_all = activation_all.get("concentration") if isinstance(activation_all.get("concentration"), dict) else {}
    concentration = activation.get("concentration") if isinstance(activation.get("concentration"), dict) else {}
    activation_metrics_all = activation_metrics_from_report(activation_all)
    activation_metrics = activation_metrics_from_report(activation)
    distribution_ok = (
        activation_metrics["activated_count"] >= 30
        and float(concentration.get("max_doc_share", 1.0) or 1.0) <= 0.5
        and float(concentration.get("max_page_share", 1.0) or 1.0) <= 0.35
        and activation_metrics["eligible_for_heldout_count"] >= 30
    )
    decision = "proceed_to_repaired_20_to_30_gate" if distribution_ok else "continue_stage2_coverage_quality_no_qa"
    return {
        "schema_version": "r031_bounded_activation_scan_review_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "diagnostic_only": True,
            "artifact_store": "cumulative20_plus_r030_repaired_3_pages_atomic_only",
            "no_qa": True,
            "no_effectiveness_claim": True,
            "no_graph": True,
            "no_rerank_tuning": True,
            "not_merged_into_cumulative_artifacts": True,
            "uses_gold_fields": False,
        },
        "inputs": {"base_artifacts": args.base_artifacts, "r030_artifacts": args.r030_artifacts},
        "fallback_boundary_audit": boundary,
        "merge_report": merge_report,
        "merged_all_eligibility_metrics": eligibility_metrics_from_report(eligibility_all),
        "atomic_only_eligibility_metrics": {
            "total_artifacts": int(eligibility.get("total_artifacts", 0) or 0),
            "eligible_artifacts": int(eligibility.get("eligible_artifacts", 0) or 0),
            "atomic_strong_eligible_artifacts": int(eligibility.get("atomic_strong_eligible_artifacts", 0) or 0),
            "numeric_fact_count": int(eligibility.get("numeric_fact_count", 0) or 0),
            "table_cell_count": int(eligibility.get("table_cell_count", 0) or 0),
            "eligible_pages_with_atomic_artifact": int(eligibility.get("eligible_pages_with_atomic_artifact", 0) or 0),
            "broad_table_only_count": int(eligibility.get("broad_table_only_count", 0) or 0),
        },
        "merged_all_activation_metrics": activation_metrics_all,
        "atomic_only_activation_metrics": activation_metrics,
        "merged_all_concentration": concentration_metrics_from_report(concentration_all),
        "atomic_only_concentration": {
            "activated_by_doc": concentration.get("activated_by_doc", {}),
            "activated_by_page": concentration.get("activated_by_page", {}),
            "max_doc_share": float(concentration.get("max_doc_share", 0.0) or 0.0),
            "max_page_share": float(concentration.get("max_page_share", 0.0) or 0.0),
            "effective_num_docs": concentration.get("effective_num_docs", 0),
            "effective_num_pages": concentration.get("effective_num_pages", 0),
        },
        "decision": decision,
        "decision_reason": (
            "activation count and concentration are sufficient for repaired 20->30 generalization gate"
            if distribution_ok
            else "activation remains insufficient or too concentrated; do not run QA/effectiveness gate"
        ),
        "commands": command_results,
    }


def eligibility_metrics_from_report(eligibility: dict[str, Any]) -> dict[str, int]:
    return {
        "total_artifacts": int(eligibility.get("total_artifacts", 0) or 0),
        "eligible_artifacts": int(eligibility.get("eligible_artifacts", 0) or 0),
        "atomic_strong_eligible_artifacts": int(eligibility.get("atomic_strong_eligible_artifacts", 0) or 0),
        "numeric_fact_count": int(eligibility.get("numeric_fact_count", 0) or 0),
        "table_cell_count": int(eligibility.get("table_cell_count", 0) or 0),
        "eligible_pages_with_atomic_artifact": int(eligibility.get("eligible_pages_with_atomic_artifact", 0) or 0),
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


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    row = {"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]}
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows = []
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_boundary_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R031 Fallback Boundary Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        f"Uses page_text: `{report['uses_page_text']}`",
        f"Requires document_generic: `{report['requires_document_generic']}`",
        f"Uses layout locator: `{report['uses_layout_locator']}`",
        f"Reads question/answer/gold/evidence: `{report['reads_question_answer_gold_evidence']}`",
        f"Forbidden term hits: `{report['forbidden_term_hits']}`",
        "",
        "## Notes",
    ]
    lines.extend(f"- {note}" for note in report["notes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    activation_all = report["merged_all_activation_metrics"]
    activation = report["atomic_only_activation_metrics"]
    concentration_all = report["merged_all_concentration"]
    concentration = report["atomic_only_concentration"]
    eligibility_all = report["merged_all_eligibility_metrics"]
    eligibility = report["atomic_only_eligibility_metrics"]
    lines = [
        "# R031 Bounded Activation Scan Review",
        "",
        f"Decision: `{report['decision']}`",
        f"Reason: {report['decision_reason']}",
        "",
        "## Scope",
        "- Diagnostic only.",
        "- Temporary `cumulative20 + R030 repaired 3 pages` artifact store.",
        "- Activation scan uses atomic-only artifacts.",
        "- No QA, no graph, no rerank tuning, no effectiveness claim.",
        "",
        "## Fallback Boundary",
        f"- Boundary decision: `{report['fallback_boundary_audit']['decision']}`",
        f"- Reads question/answer/gold/evidence: `{report['fallback_boundary_audit']['reads_question_answer_gold_evidence']}`",
        "",
        "## Merged-All Eligibility",
        f"- Total artifacts: {eligibility_all['total_artifacts']}",
        f"- Eligible artifacts: {eligibility_all['eligible_artifacts']}",
        f"- Atomic strong eligible artifacts: {eligibility_all['atomic_strong_eligible_artifacts']}",
        f"- Numeric facts: {eligibility_all['numeric_fact_count']}",
        f"- Table cells: {eligibility_all['table_cell_count']}",
        f"- Broad table only: {eligibility_all['broad_table_only_count']}",
        "",
        "## Atomic-Only Eligibility",
        f"- Atomic artifacts: {eligibility['total_artifacts']}",
        f"- Atomic strong eligible artifacts: {eligibility['atomic_strong_eligible_artifacts']}",
        f"- Numeric facts: {eligibility['numeric_fact_count']}",
        f"- Table cells: {eligibility['table_cell_count']}",
        f"- Eligible pages with atomic artifact: {eligibility['eligible_pages_with_atomic_artifact']}",
        f"- Broad table only: {eligibility['broad_table_only_count']}",
        "",
        "## Merged-All Activation",
        f"- Activated records: {activation_all['activated_count']}",
        f"- Eligible for held-out: {activation_all['eligible_for_heldout_count']}",
        f"- Changed records, artifact_only: {activation_all['changed_count']}",
        f"- Changed records, original_plus_artifact: {activation_all['original_plus_changed_count']}",
        f"- Held-out available: `{activation_all['heldout_available']}`",
        "",
        "## Atomic-Only Activation",
        f"- Activated records: {activation['activated_count']}",
        f"- Eligible for held-out: {activation['eligible_for_heldout_count']}",
        f"- Changed records, artifact_only: {activation['changed_count']}",
        f"- Changed records, original_plus_artifact: {activation['original_plus_changed_count']}",
        f"- Held-out available: `{activation['heldout_available']}`",
        "",
        "## Merged-All Concentration",
        f"- Max doc share: {concentration_all['max_doc_share']:.4f}",
        f"- Max page share: {concentration_all['max_page_share']:.4f}",
        f"- Effective docs: {concentration_all['effective_num_docs']}",
        f"- Effective pages: {concentration_all['effective_num_pages']}",
        f"- Activated by doc: `{concentration_all['activated_by_doc']}`",
        f"- Activated by page: `{concentration_all['activated_by_page']}`",
        "",
        "## Atomic-Only Concentration",
        f"- Max doc share: {concentration['max_doc_share']:.4f}",
        f"- Max page share: {concentration['max_page_share']:.4f}",
        f"- Effective docs: {concentration['effective_num_docs']}",
        f"- Effective pages: {concentration['effective_num_pages']}",
        f"- Activated by doc: `{concentration['activated_by_doc']}`",
        f"- Activated by page: `{concentration['activated_by_page']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
