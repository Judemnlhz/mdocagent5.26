#!/usr/bin/env python3
"""R073 no-provider cross-dataset evidence-layer reuse audit.

R073 checks whether the R071/R072 evidence layer remains a lightweight,
dataset-agnostic interface across MMLB, LDU, PTAB, PTEXT, and FETA public
inputs. It runs the full retrieval/artifact/capsule audit only where equivalent
public retrieval and artifact bindings exist, and marks other datasets as
blocked rather than substituting gold evidence pages or answers. It does not
call providers, generate predictions, run QA, evaluate, or report an official
score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import statistics
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053
import run_r071_evidence_skill_graph_registry_gate as r071
from mdocnexus.integration.evidence_skill_registry import (
    REGISTRY,
    activated_skills,
    estimate_tokens,
    flat_artifact_context,
    raw_page_context,
    registry_contract,
    render_evidence_capsule,
    validate_registry_contract,
)
from mdocnexus.integration.guarded_prompt import build_question_profile, forbidden_public_fields, select_guarded_artifacts

DEFAULT_OUTPUT_ROOT = "outputs/heldout/r073_cross_dataset_evidence_layer_reuse_audit"
MAX_EXAMPLES = 20

DATASETS = [
    {
        "dataset": "MMLB",
        "sample_path": "data/MMLongBench/sample-with-retrieval-results.json",
        "extract_path": "tmp/MMLongBench",
        "artifact_path": r053.DEFAULT_ARTIFACTS,
        "mode": "full_public_retrieval_artifact_capsule",
    },
    {
        "dataset": "LDU",
        "sample_path": "data/LongDocURL/samples.json",
        "extract_path": "tmp/LongDocURL",
        "artifact_path": "",
        "mode": "question_only_availability",
    },
    {
        "dataset": "FETA",
        "sample_path": "data/FetaTab/samples.json",
        "extract_path": "tmp/FetaTab",
        "artifact_path": "",
        "mode": "question_only_availability",
    },
    {
        "dataset": "PTEXT",
        "sample_path": "data/PaperText/samples.json",
        "extract_path": "tmp/PaperText",
        "artifact_path": "",
        "mode": "question_only_availability",
    },
    {
        "dataset": "PTAB",
        "sample_path": "data/PaperTab/samples.json",
        "extract_path": "tmp/PaperTab",
        "artifact_path": "",
        "mode": "question_only_availability",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-records-per-dataset", type=int, default=0, help="Optional debug cap; 0 means all records.")
    parser.add_argument("--max-page-chars", type=int, default=2200)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--capsule-units", type=int, default=4)
    parser.add_argument("--flat-artifact-units", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "stage": "r073_cross_dataset_evidence_layer_reuse_audit",
            "datasets": [row["dataset"] for row in DATASETS],
            "no_provider_calls": True,
            "no_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(audit)
    report = build_report(audit, gate)
    r053.write_json(output_root / "r073_cross_dataset_evidence_layer_summary.json", audit["summary"])
    r053.write_jsonl(output_root / "r073_cross_dataset_evidence_layer_records.jsonl", audit["records"])
    r053.write_json(output_root / "r073_cross_dataset_evidence_layer_gate.json", gate)
    write_gate_markdown(output_root / "r073_cross_dataset_evidence_layer_gate.md", gate)
    r053.write_json(output_root / "r073_cross_dataset_evidence_layer_report.json", report)
    write_report_markdown(output_root / "r073_cross_dataset_evidence_layer_report.md", report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "datasets_reported": audit["summary"]["datasets_reported"],
        "mmlb_records_scanned": audit["summary"]["dataset_summaries"].get("MMLB", {}).get("records_scanned", 0),
        "mmlb_mean_guarded_capsule_raw_ratio": audit["summary"].get("mmlb_token_stats", {}).get("capsule_with_guard_vs_raw", {}).get("mean"),
        "blocked_datasets": audit["summary"]["blocked_datasets"],
        "report_md": str(output_root / "r073_cross_dataset_evidence_layer_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
        "not_official_score": True,
    }, ensure_ascii=False, indent=2))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for config in DATASETS:
        dataset, dataset_rows, dataset_summary = audit_dataset(config, args)
        rows.extend(dataset_rows)
        dataset_summaries[dataset] = dataset_summary
    summary = summarize(dataset_summaries, rows, args)
    public_payload = {"summary": summary, "records": rows}
    summary["forbidden_gold_fields_present"] = forbidden_public_fields(public_payload)
    return {
        "summary": summary,
        "records": rows,
        "registry_contract": registry_contract(),
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
        "not_full_qa": True,
        "not_official_score": True,
    }


def audit_dataset(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    dataset = str(config["dataset"])
    sample_path = Path(str(config["sample_path"]))
    extract_path = Path(str(config["extract_path"]))
    artifact_path = Path(str(config.get("artifact_path") or "")) if config.get("artifact_path") else None
    if not sample_path.exists():
        summary = empty_dataset_summary(dataset, sample_path, extract_path, artifact_path, "blocked_missing_sample_file")
        return dataset, [], summary
    records = r053.read_json(sample_path)
    total_records = len(records)
    if args.max_records_per_dataset and args.max_records_per_dataset > 0:
        records = records[: args.max_records_per_dataset]
    text_index = page_text_index(extract_path)
    public_retrieval_available = dataset_has_public_retrieval(records)
    artifacts_available = bool(artifact_path and artifact_path.exists())
    if dataset == "MMLB" and public_retrieval_available and artifacts_available:
        artifacts_by_page = r053.load_artifacts_by_page(artifact_path)
        rows = [audit_mmlb_record(i, record, artifacts_by_page, extract_path, args) for i, record in enumerate(records)]
        summary = summarize_full_dataset(dataset, sample_path, extract_path, artifact_path, rows, total_records, len(records), text_index)
        return dataset, rows, summary
    rows = [audit_question_only_record(dataset, i, record, text_index) for i, record in enumerate(records)]
    status = "blocked_missing_public_retrieval_or_artifacts"
    summary = summarize_question_only_dataset(dataset, sample_path, extract_path, artifact_path, rows, total_records, len(records), text_index, public_retrieval_available, artifacts_available, status)
    return dataset, rows, summary


def audit_mmlb_record(record_id: int, record: Mapping[str, Any], artifacts_by_page: Mapping[tuple[str, int], list[dict[str, Any]]], extract_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or "")
    question = str(record.get("question") or "")
    profile = build_question_profile(question)
    pages = r071.retrieval_pages(record, args.top_k)
    page_contexts = [r053.load_page_context(extract_path, doc_id, page, args.max_page_chars) for page in pages]
    current_artifacts = [artifact for page in pages for artifact in artifacts_by_page.get((doc_id, page), [])]
    scored = r071.score_artifacts(current_artifacts, question, profile, pages, args)
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)
    raw_text = raw_page_context(page_contexts, max_chars_per_page=args.max_page_chars)
    flat_text = flat_artifact_context(scored, max_units=args.flat_artifact_units, max_chars=args.max_artifact_chars)
    capsule = render_evidence_capsule(question, profile, selection, scored, max_units=args.capsule_units, include_guard_trace=True, max_chars=args.max_artifact_chars)
    raw_tokens = estimate_tokens(raw_text)
    flat_tokens = estimate_tokens(flat_text)
    capsule_tokens = int(capsule["token_estimate"])
    return {
        "schema_version": "r073_cross_dataset_record_v1",
        "dataset": "MMLB",
        "input_mode": "full_public_retrieval_artifact_capsule",
        "record_id": record_id,
        "doc_id": doc_id,
        "question": question,
        "retrieval_pages": pages,
        "page_text_exists_count": sum(1 for ctx in page_contexts if ctx.get("exists")),
        "candidate_artifact_count": len(scored),
        "selected_artifact_count": len(selection.get("selected_artifacts") or []),
        "activated_skill_names": capsule["activated_skill_names"],
        "guard_decision": selection.get("guard_decision"),
        "missing_requirements": capsule["missing_requirements"],
        "token_counts": {
            "raw_page": raw_tokens,
            "flat_artifact": flat_tokens,
            "capsule_with_guard": capsule_tokens,
        },
        "compression_ratios": {
            "flat_artifact_vs_raw": ratio(flat_tokens, raw_tokens),
            "capsule_with_guard_vs_raw": ratio(capsule_tokens, raw_tokens),
        },
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
    }


def audit_question_only_record(dataset: str, record_id: int, record: Mapping[str, Any], text_index: Mapping[str, int]) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or "")
    question = str(record.get("question") or "")
    profile = build_question_profile(question)
    skills = activated_skills(profile, question)
    return {
        "schema_version": "r073_cross_dataset_record_v1",
        "dataset": dataset,
        "input_mode": "question_only_availability",
        "record_id": record_id,
        "doc_id": doc_id,
        "question": question,
        "doc_page_text_available": page_text_count_for_doc(text_index, doc_id) > 0,
        "doc_page_text_count": page_text_count_for_doc(text_index, doc_id),
        "activated_skill_names": [skill.name for skill in skills],
        "question_profile_flags": {
            "requires_exact_code_selection": bool(profile.get("requires_exact_code_selection")),
            "is_numeric_or_table_question": bool(profile.get("is_numeric_or_table_question")),
            "is_computation_question": bool(profile.get("is_computation_question")),
            "is_document_metadata_lookup": bool(profile.get("is_document_metadata_lookup")),
        },
        "status": "blocked_missing_public_retrieval_or_artifacts",
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
    }


def summarize_full_dataset(dataset: str, sample_path: Path, extract_path: Path, artifact_path: Path | None, rows: list[Mapping[str, Any]], total_records: int, scanned_records: int, text_index: Mapping[str, int]) -> dict[str, Any]:
    skill_counts = Counter(skill for row in rows for skill in row.get("activated_skill_names") or [])
    guard_counts = Counter(str(row.get("guard_decision") or "") for row in rows)
    ratios = [row.get("compression_ratios", {}).get("capsule_with_guard_vs_raw", 0.0) for row in rows]
    return {
        "schema_version": "r073_dataset_summary_v1",
        "dataset": dataset,
        "status": "full_public_retrieval_artifact_capsule_audited",
        "sample_path": str(sample_path),
        "extract_path": str(extract_path),
        "artifact_path": str(artifact_path) if artifact_path else "",
        "total_records": total_records,
        "records_scanned": scanned_records,
        "public_sample_file_exists": sample_path.exists(),
        "page_text_dir_exists": extract_path.exists(),
        "docs_with_page_text_count": docs_with_text_count(rows),
        "public_retrieval_available": True,
        "artifact_store_available": bool(artifact_path and artifact_path.exists()),
        "full_capsule_audit_available": True,
        "activated_skill_counts": dict(sorted(skill_counts.items())),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "mean_capsule_with_guard_raw_ratio": number_stats(ratios)["mean"],
        "examples": compact_examples(rows),
        "text_file_count": sum(text_index.values()),
    }


def summarize_question_only_dataset(dataset: str, sample_path: Path, extract_path: Path, artifact_path: Path | None, rows: list[Mapping[str, Any]], total_records: int, scanned_records: int, text_index: Mapping[str, int], public_retrieval_available: bool, artifacts_available: bool, status: str) -> dict[str, Any]:
    skill_counts = Counter(skill for row in rows for skill in row.get("activated_skill_names") or [])
    docs_with_text = sum(1 for row in rows if row.get("doc_page_text_available"))
    return {
        "schema_version": "r073_dataset_summary_v1",
        "dataset": dataset,
        "status": status,
        "sample_path": str(sample_path),
        "extract_path": str(extract_path),
        "artifact_path": str(artifact_path) if artifact_path else "",
        "total_records": total_records,
        "records_scanned": scanned_records,
        "public_sample_file_exists": sample_path.exists(),
        "page_text_dir_exists": extract_path.exists(),
        "docs_with_page_text_count": docs_with_text,
        "docs_with_page_text_rate": ratio(docs_with_text, len(rows)),
        "public_retrieval_available": public_retrieval_available,
        "artifact_store_available": artifacts_available,
        "full_capsule_audit_available": False,
        "activated_skill_counts": dict(sorted(skill_counts.items())),
        "guard_decision_counts": {},
        "blocked_reason": "No public retrieval-page list and matching artifact store equivalent to the MMLB R038d store was found; gold answer/evidence fields were not used.",
        "examples": compact_examples(rows),
        "text_file_count": sum(text_index.values()),
    }


def empty_dataset_summary(dataset: str, sample_path: Path, extract_path: Path, artifact_path: Path | None, status: str) -> dict[str, Any]:
    return {
        "schema_version": "r073_dataset_summary_v1",
        "dataset": dataset,
        "status": status,
        "sample_path": str(sample_path),
        "extract_path": str(extract_path),
        "artifact_path": str(artifact_path) if artifact_path else "",
        "total_records": 0,
        "records_scanned": 0,
        "public_sample_file_exists": sample_path.exists(),
        "page_text_dir_exists": extract_path.exists(),
        "public_retrieval_available": False,
        "artifact_store_available": bool(artifact_path and artifact_path.exists()),
        "full_capsule_audit_available": False,
        "activated_skill_counts": {},
        "guard_decision_counts": {},
    }


def summarize(dataset_summaries: Mapping[str, Any], rows: list[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    contract = registry_contract()
    mmlb_rows = [row for row in rows if row.get("dataset") == "MMLB"]
    ratio_values = [row.get("compression_ratios", {}).get("capsule_with_guard_vs_raw", 0.0) for row in mmlb_rows]
    all_skill_counts = Counter(skill for row in rows for skill in row.get("activated_skill_names") or [])
    blocked = sorted(dataset for dataset, summary in dataset_summaries.items() if not summary.get("full_capsule_audit_available"))
    return {
        "schema_version": "r073_cross_dataset_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_reported": sorted(dataset_summaries),
        "dataset_summaries": dict(dataset_summaries),
        "registry_skill_count": len(REGISTRY),
        "registry_skill_names": sorted(skill.name for skill in REGISTRY),
        "registry_contract_failures": validate_registry_contract(contract),
        "cross_dataset_activated_skill_counts": dict(sorted(all_skill_counts.items())),
        "blocked_datasets": blocked,
        "full_capsule_audit_datasets": sorted(dataset for dataset, summary in dataset_summaries.items() if summary.get("full_capsule_audit_available")),
        "mmlb_token_stats": {
            "capsule_with_guard_vs_raw": number_stats(ratio_values),
        },
        "boundary": {
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_use_answer_or_evidence_pages": True,
            "does_not_substitute_gold_pages_for_retrieval": True,
            "not_large_skill_tree": True,
            "not_global_knowledge_graph": True,
        },
        "settings": {
            "top_k": args.top_k,
            "max_records_per_dataset": args.max_records_per_dataset,
            "capsule_units": args.capsule_units,
        },
    }


def build_gate(audit: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    dataset_summaries = summary["dataset_summaries"]
    target_datasets = {row["dataset"] for row in DATASETS}
    question_skill_ok = all(bool(dataset_summaries.get(dataset, {}).get("activated_skill_counts")) for dataset in target_datasets)
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "not_official_score": True,
        "all_target_datasets_reported": set(summary.get("datasets_reported") or []) == target_datasets,
        "registry_contract_valid": not summary.get("registry_contract_failures"),
        "registry_has_no_dataset_specific_skill_names": not any(skill_marker_in_name(name) for name in summary.get("registry_skill_names") or []),
        "mmlb_full_reuse_audit_available": dataset_summaries.get("MMLB", {}).get("full_capsule_audit_available") is True,
        "mmlb_capsule_mean_ratio_below_one": summary.get("mmlb_token_stats", {}).get("capsule_with_guard_vs_raw", {}).get("mean", 1.0) < 1.0,
        "cross_dataset_question_skill_activation_recorded": question_skill_ok,
        "blocked_inputs_are_explicit_not_silent": all(dataset_summaries.get(dataset, {}).get("status", "").startswith("blocked_") for dataset in summary.get("blocked_datasets") or []),
        "no_gold_fields_in_public_outputs": not summary.get("forbidden_gold_fields_present"),
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r073_cross_dataset_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r073_cross_dataset_reuse_audit_complete_with_input_gaps" if not hard_failures else "r073_cross_dataset_reuse_audit_invalid",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "blocked_datasets": summary.get("blocked_datasets") or [],
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    return {
        "schema_version": "r073_cross_dataset_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": gate["decision"],
        "scope": summary["boundary"],
        "summary": summary,
        "gate": gate,
        "recommended_next": recommendations(summary),
    }


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    blocked = summary.get("blocked_datasets") or []
    ratio_mean = summary.get("mmlb_token_stats", {}).get("capsule_with_guard_vs_raw", {}).get("mean", 1.0)
    return [
        f"Keep the same Evidence Skill Registry and capsule renderer; MMLB guarded capsule/raw ratio remains {ratio_mean} under the cross-dataset audit.",
        f"Do not claim cross-dataset token or citation gains until public retrieval/artifact bindings are built for blocked datasets: {blocked}.",
        "Next engineering step should be a small reusable public retrieval-to-artifact binding adapter, not new dataset-named skills or a large graph tree.",
    ]


def dataset_has_public_retrieval(records: list[Mapping[str, Any]]) -> bool:
    return any(isinstance(record.get("text-top-10-question"), list) or isinstance(record.get("image-top-10-question"), list) for record in records[:50])


def page_text_index(extract_path: Path) -> dict[str, int]:
    rows: dict[str, int] = defaultdict(int)
    if not extract_path.exists():
        return rows
    for path in extract_path.glob("*.txt"):
        stem = path.stem
        if "_" not in stem:
            continue
        doc_stem = stem.rsplit("_", 1)[0]
        rows[doc_stem] += 1
    return dict(rows)


def page_text_count_for_doc(text_index: Mapping[str, int], doc_id: str) -> int:
    stem = doc_id[:-4] if doc_id.lower().endswith(".pdf") else doc_id
    return int(text_index.get(stem, 0))


def docs_with_text_count(rows: list[Mapping[str, Any]]) -> int:
    return len({row.get("doc_id") for row in rows if row.get("page_text_exists_count", 0) > 0 or row.get("doc_page_text_available")})


def number_stats(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    nums = [float(value) for value in values]
    return {
        "mean": round(statistics.fmean(nums), 6),
        "median": round(statistics.median(nums), 6),
        "min": round(min(nums), 6),
        "max": round(max(nums), 6),
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0:
        return 1.0 if float(numerator) > 0 else 0.0
    return round(float(numerator) / float(denominator), 6)


def compact_examples(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows[:MAX_EXAMPLES]:
        output.append({
            "dataset": row.get("dataset"),
            "input_mode": row.get("input_mode"),
            "record_id": row.get("record_id"),
            "doc_id": row.get("doc_id"),
            "question": str(row.get("question") or "")[:180],
            "activated_skill_names": row.get("activated_skill_names"),
            "guard_decision": row.get("guard_decision"),
            "compression_ratios": row.get("compression_ratios"),
            "status": row.get("status"),
        })
    return output


def skill_marker_in_name(name: str) -> bool:
    text = str(name or "").lower()
    return any(marker in text for marker in ["mmlb", "mmlongbench", "ldu", "ptab", "ptext", "feta", "fetatab"])


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = [
        "# R073 Cross-Dataset Evidence Layer Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Does not use answers, evidence pages, or official scoring.",
        "- Missing retrieval/artifact bindings are reported as blocked inputs, not silently substituted.",
        "",
        "## Checks",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["blocked_datasets"]:
        lines.extend(["", "## Blocked Inputs"])
        lines.extend(f"- {dataset}" for dataset in gate["blocked_datasets"])
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# R073 Cross-Dataset Evidence Layer Reuse Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Public questions and page-text availability are used; answer/evidence fields are excluded from outputs.",
        "- Non-MMLB datasets without public retrieval/artifact bindings are marked blocked rather than scored.",
        "",
        "## Dataset Status",
    ]
    for dataset, item in summary["dataset_summaries"].items():
        lines.append(f"- `{dataset}`: {item['status']}; records scanned={item['records_scanned']}; full capsule audit={item['full_capsule_audit_available']}")
    lines.extend([
        "",
        "## MMLB Token Reuse",
        f"- mean guarded capsule/raw ratio: {summary['mmlb_token_stats']['capsule_with_guard_vs_raw']['mean']}",
        "",
        "## Cross-Dataset Skill Activation",
    ])
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("cross_dataset_activated_skill_counts", {}).items())
    lines.extend(["", "## Recommended Next"])
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()