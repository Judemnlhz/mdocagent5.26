#!/usr/bin/env python3
"""R069 no-provider dataset-level artifact health audit.

R069 audits whether current public retrieval candidates and artifact stores can
provide visible, citable evidence before any provider QA. It scans dataset
questions and retrieved public pages, buckets exact-code/key-value/table-numeric
cases, replays deterministic code/name extraction on retrieved pages, and checks
selector outcomes. It does not call providers, generate predictions, run eval,
run full QA, or report an official score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053
import run_r067_source_ocr_code_list_extraction_audit as r067
import run_r068_code_list_stage2_integration_audit as r068
from mdocnexus.integration.guarded_prompt import CODE_PATTERN, build_question_profile, forbidden_public_fields, score_guarded_artifact, select_guarded_artifacts
from mdocnexus.stage2.artifact_quality import classify_artifact_quality, quality_discard_reason
from mdocnexus.stage2.code_name_list_extractor import extract_code_name_list_artifacts

DEFAULT_OUTPUT_ROOT = "outputs/heldout/r069_dataset_artifact_health_audit"
NUMERIC_TABLE_TERMS = {
    "table", "code", "value", "number", "percent", "percentage", "difference", "total", "ratio", "rate",
    "market", "metric", "score", "amount", "year", "fy", "q1", "q2", "q3", "q4", "round", "average",
}
KEY_VALUE_TERMS = {"what is", "what's", "which", "who", "where", "name", "market", "code", "value"}
MAX_EXAMPLES_PER_BUCKET = 12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=r053.DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=r053.DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=r053.DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0, help="Optional debug cap; 0 means all records.")
    parser.add_argument("--max-page-chars", type=int, default=2200)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({"will_execute": False, "output_root": str(output_root), "records": args.records, "artifacts": args.artifacts, "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    gate = build_gate(args, audit)
    report = build_report(args, audit, gate)
    r053.write_json(output_root / "r069_dataset_artifact_health_summary.json", audit["summary"])
    r053.write_jsonl(output_root / "r069_dataset_artifact_health_records.jsonl", audit["records"])
    r053.write_json(output_root / "r069_dataset_artifact_health_gate.json", gate)
    write_gate_markdown(output_root / "r069_dataset_artifact_health_gate.md", gate)
    r053.write_json(output_root / "r069_dataset_artifact_health_report.json", report)
    write_report_markdown(output_root / "r069_dataset_artifact_health_report.md", report)
    print(json.dumps({"decision": gate["decision"], "gate_passed": gate["gate_passed"], "records_scanned": audit["summary"]["records_scanned"], "code_like_literal_records": audit["summary"]["by_type"].get("code_like_literal", {}).get("records", 0), "exact_code_lookup_records": audit["summary"]["by_type"].get("exact_code_lookup", {}).get("records", 0), "exact_code_lookup_public_text_present": audit["summary"]["by_type"].get("exact_code_lookup", {}).get("public_text_literal_present", 0), "exact_code_lookup_current_selector_selected": audit["summary"]["by_type"].get("exact_code_lookup", {}).get("current_selector_selected", 0), "exact_code_lookup_replay_selector_selected": audit["summary"]["by_type"].get("exact_code_lookup", {}).get("replay_selector_selected", 0), "report_md": str(output_root / "r069_dataset_artifact_health_report.md"), "no_provider_calls": True, "no_full_qa": True}, ensure_ascii=False, indent=2))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    records = r053.read_json(Path(args.records))
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    extract_path = Path(args.extract_path)
    rows = []
    for record_id, record in enumerate(records):
        rows.append(audit_record(record_id, record, artifacts_by_page, extract_path, args))
    summary = summarize(rows, args)
    public_payload = {"summary": summary, "records": rows}
    summary["forbidden_gold_fields_present"] = forbidden_public_fields(public_payload)
    return {"summary": summary, "records": rows, "no_provider_calls": True, "not_prediction_or_eval": True, "not_full_qa": True, "not_official_score": True, "not_artifact_lift_claim": True}


def audit_record(record_id: int, record: Mapping[str, Any], artifacts_by_page: Mapping[tuple[str, int], list[dict[str, Any]]], extract_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or "")
    question = str(record.get("question") or "")
    pages = retrieval_pages(record, args.top_k)
    profile = build_question_profile(question)
    types = classify_question(question, profile)
    literals = extract_literals(question, profile)
    page_contexts = [r053.load_page_context(extract_path, doc_id, page, args.max_page_chars) for page in pages]
    page_text_by_page = {ctx["page_index"]: str(ctx.get("text_preview") or "") for ctx in page_contexts}
    public_text_hits = literal_hits_in_text(literals, " ".join(page_text_by_page.values()))
    current_artifacts = [artifact for page in pages for artifact in artifacts_by_page.get((doc_id, page), [])]
    replay_artifacts = replay_code_name_artifacts(doc_id, pages, extract_path, artifacts_by_page)
    current_artifact_hits = literal_hits_in_artifacts(literals, current_artifacts)
    replay_artifact_hits = literal_hits_in_artifacts(literals, replay_artifacts)
    current_selection = selector_summary(current_artifacts, question, profile, pages, page_contexts, args)
    replay_selection = selector_summary(current_artifacts + replay_artifacts, question, profile, pages, page_contexts, args)
    failure_bucket = failure_bucket_for(types, literals, public_text_hits, current_artifact_hits, replay_artifact_hits, current_selection, replay_selection, current_artifacts, replay_artifacts)
    return {
        "schema_version": "r069_dataset_artifact_health_record_v1",
        "record_id": record_id,
        "doc_id": doc_id,
        "question": question,
        "question_types": types,
        "retrieval_pages": pages,
        "literal_requirements": literals,
        "page_text_exists_count": sum(1 for ctx in page_contexts if ctx.get("exists")),
        "public_text_literal_hits": public_text_hits,
        "current_candidate_artifact_count": len(current_artifacts),
        "current_artifact_literal_hits": current_artifact_hits,
        "replay_code_name_artifact_count": len(replay_artifacts),
        "replay_artifact_literal_hits": replay_artifact_hits,
        "current_selector": current_selection,
        "replay_selector": replay_selection,
        "failure_bucket": failure_bucket,
        "no_provider_calls": True,
        "not_prediction_or_eval": True,
    }


def retrieval_pages(record: Mapping[str, Any], top_k: int) -> list[int]:
    values = []
    for key in ["text-top-10-question", "image-top-10-question"]:
        raw = record.get(key)
        if isinstance(raw, list):
            values.extend(raw[:top_k])
    return r053.unique_ints(values)


def classify_question(question: str, profile: Mapping[str, Any]) -> list[str]:
    q_norm = r053.normalize(question)
    types = []
    if profile.get("codes"):
        types.append("code_like_literal")
    if is_exact_code_lookup_question(q_norm, profile):
        types.append("exact_code_lookup")
    if any(term in q_norm for term in NUMERIC_TABLE_TERMS) or profile.get("is_numeric_or_table_question"):
        types.append("table_numeric")
    if any(term in q_norm for term in KEY_VALUE_TERMS):
        types.append("key_value_lookup")
    if profile.get("is_computation_question"):
        types.append("computation")
    if not types:
        types.append("general")
    return types


def is_exact_code_lookup_question(q_norm: str, profile: Mapping[str, Any]) -> bool:
    if not actionable_codes(profile.get("codes") or []):
        return False
    lookup_terms = ["code", "eps code", "market", "geographic market", "designated area", "segment", "row", "column"]
    return any(term in q_norm for term in lookup_terms)


def actionable_codes(codes: list[Any]) -> list[str]:
    rows = []
    for code in codes:
        text = str(code or "").upper()
        if not text:
            continue
        if re.fullmatch(r"FY\d{4}", text):
            continue
        if re.fullmatch(r"Q[1-4]", text):
            continue
        if re.fullmatch(r"F\d+", text):
            continue
        if re.fullmatch(r"AP\d+", text):
            continue
        if re.fullmatch(r"GPT\d+", text):
            continue
        rows.append(text)
    return sorted(set(rows))


def extract_literals(question: str, profile: Mapping[str, Any]) -> dict[str, list[str]]:
    q_norm = r053.normalize(question)
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    numbers = sorted(set(str(item) for item in profile.get("numbers") or []))
    codes = sorted(set(str(item) for item in profile.get("codes") or CODE_PATTERN.findall(question)))
    actionable = actionable_codes(codes)
    quoted = sorted(set(match.strip() for match in re.findall(r"[\"']([^\"']{2,80})[\"']", question) if match.strip()))
    metric_phrases = []
    for phrase in ["gross profit", "net income", "total assets", "current liabilities", "total liabilities", "capital expenditure", "cash flow", "geographic market name", "f1", "ap50", "ebitda", "cash ratio", "current ratio", "quick ratio"]:
        if phrase in q_norm:
            metric_phrases.append(phrase)
    return {"codes": codes, "actionable_codes": actionable, "years": years, "numbers": numbers, "quoted": quoted, "metric_phrases": sorted(set(metric_phrases))}


def literal_hits_in_text(literals: Mapping[str, list[str]], text: str) -> dict[str, list[str]]:
    return {key: [item for item in values if literal_present(text, item)] for key, values in literals.items()}


def literal_hits_in_artifacts(literals: Mapping[str, list[str]], artifacts: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    text = " ".join(searchable_artifact_text(artifact) for artifact in artifacts)
    return literal_hits_in_text(literals, text)


def literal_present(text: str, literal: str) -> bool:
    if not literal:
        return False
    if CODE_PATTERN.fullmatch(literal) or re.fullmatch(r"\d+(?:\.\d+)?", literal):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None
    return r053.normalize(literal) in r053.normalize(text)


def searchable_artifact_text(artifact: Mapping[str, Any]) -> str:
    parts = [str(artifact.get("content") or "")]
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), Mapping) else {}
    for value in normalized.values():
        if isinstance(value, (str, int, float)):
            parts.append(str(value))
    return " ".join(parts)


def replay_code_name_artifacts(doc_id: str, pages: list[int], extract_path: Path, artifacts_by_page: Mapping[tuple[str, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for page in pages:
        page_text = r067.load_page_text(extract_path, doc_id, page)
        if not page_text.strip():
            continue
        page_input = r067.make_page_input(doc_id, page, page_text)
        extracted = extract_code_name_list_artifacts(selected_page={"doc_id": doc_id, "page_index": page}, page_input=page_input, existing_artifacts=artifacts_by_page.get((doc_id, page), []))
        for artifact in extracted:
            if artifact.get("artifact_type") != "text_span" or artifact.get("modality") != "text":
                continue
            quality_reason = quality_discard_reason(artifact)
            if quality_reason:
                continue
            rows.append(artifact)
    return rows


def selector_summary(artifacts: list[Mapping[str, Any]], question: str, profile: Mapping[str, Any], pages: list[int], page_contexts: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    scored = []
    page_set = set(pages)
    for artifact in artifacts:
        try:
            page = int(artifact.get("page_index"))
        except (TypeError, ValueError):
            page = -1
        scored.append(score_guarded_artifact(artifact, question, profile, page, artifact_pages=list(page_set), original_pages=list(page_set), max_chars=args.max_artifact_chars))
    selection = select_guarded_artifacts(scored, page_contexts, profile, max_artifacts=args.max_artifacts)
    selected = selection.get("selected_artifacts") or []
    return {
        "candidate_count": len(scored),
        "positive_candidate_count": selection.get("positive_candidate_count"),
        "guard_decision": selection.get("guard_decision"),
        "answer_policy": selection.get("answer_policy"),
        "selected_artifact_count": len(selected),
        "selected_artifact_ids": [row.get("artifact_id") for row in selected[: args.max_artifacts]],
        "selected_exact_code_matches": sorted({code for row in selected for code in row.get("exact_code_matches", [])}),
    }


def any_hits(hit_map: Mapping[str, list[str]]) -> bool:
    return any(bool(values) for values in hit_map.values())


def failure_bucket_for(types: list[str], literals: Mapping[str, list[str]], public_text_hits: Mapping[str, list[str]], current_artifact_hits: Mapping[str, list[str]], replay_artifact_hits: Mapping[str, list[str]], current_selection: Mapping[str, Any], replay_selection: Mapping[str, Any], current_artifacts: list[Mapping[str, Any]], replay_artifacts: list[Mapping[str, Any]]) -> str:
    if "code_like_literal" in types:
        required_codes = set(literals.get("actionable_codes") or literals.get("codes") or [])
        hit_key = "actionable_codes" if literals.get("actionable_codes") else "codes"
        text_codes = set(public_text_hits.get(hit_key) or [])
        current_codes = set(current_artifact_hits.get(hit_key) or [])
        replay_codes = set(replay_artifact_hits.get(hit_key) or [])
        selected_codes = set(replay_selection.get("selected_exact_code_matches") or [])
        if not literals.get("actionable_codes") and literals.get("codes"):
            if current_selection.get("guard_decision") == "exact_code_absence_guard" or replay_selection.get("guard_decision") == "exact_code_absence_guard":
                return "code_like_temporal_metric_literal_triggered_exact_code_guard"
            return "code_like_temporal_metric_literal_not_actionable_code"
        if not required_codes:
            return "code_like_no_required_literal_detected"
        if not required_codes <= text_codes:
            return "retrieval_or_public_text_missing_code_like_literal"
        if required_codes <= current_codes and current_selection.get("selected_artifact_count", 0) > 0:
            return "current_store_code_like_literal_selected"
        if required_codes <= replay_codes and required_codes <= selected_codes:
            return "replay_code_name_code_like_literal_selected"
        if required_codes <= replay_codes:
            return "replay_artifact_generated_but_selector_not_selecting"
        if required_codes <= current_codes:
            return "current_artifact_has_code_but_selector_not_selecting"
        return "artifact_extraction_missing_code_like_literal"
    if any_hits(public_text_hits) and not current_artifacts and not replay_artifacts:
        return "retrieved_text_has_literals_but_no_artifacts"
    if current_selection.get("candidate_count", 0) > 0 and current_selection.get("selected_artifact_count", 0) == 0:
        return str(current_selection.get("guard_decision") or "selector_rejected_candidates")
    if current_selection.get("selected_artifact_count", 0) > 0:
        return "current_store_selected_artifacts"
    if not any_hits(public_text_hits):
        return "retrieval_or_public_text_missing_literals"
    return "uncategorized_artifact_gap"


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_type: dict[str, Counter] = defaultdict(Counter)
    buckets = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["failure_bucket"]] += 1
        if len(examples[row["failure_bucket"]]) < MAX_EXAMPLES_PER_BUCKET:
            examples[row["failure_bucket"]].append(example_row(row))
        for qtype in row["question_types"]:
            bucket = by_type[qtype]
            bucket["records"] += 1
            if any_hits(row["public_text_literal_hits"]):
                bucket["public_text_literal_present"] += 1
            if any_hits(row["current_artifact_literal_hits"]):
                bucket["current_artifact_literal_present"] += 1
            if any_hits(row["replay_artifact_literal_hits"]):
                bucket["replay_artifact_literal_present"] += 1
            if row["current_selector"].get("selected_artifact_count", 0) > 0:
                bucket["current_selector_selected"] += 1
            if row["replay_selector"].get("selected_artifact_count", 0) > 0:
                bucket["replay_selector_selected"] += 1
    return {
        "schema_version": "r069_dataset_artifact_health_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records_scanned": len(rows),
        "top_k": args.top_k,
        "input_records": args.records,
        "input_artifacts": args.artifacts,
        "input_extract_path": args.extract_path,
        "by_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "failure_buckets": dict(sorted(buckets.items())),
        "failure_examples": dict(examples),
        "boundary": {"no_provider_calls": True, "no_prediction": True, "no_evaluation": True, "no_full_qa": True, "not_official_score": True, "not_artifact_lift_claim": True, "does_not_use_answer_or_evidence_pages": True},
    }


def example_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": row.get("record_id"),
        "doc_id": row.get("doc_id"),
        "question": str(row.get("question") or "")[:180],
        "question_types": row.get("question_types"),
        "retrieval_pages": row.get("retrieval_pages"),
        "literal_requirements": row.get("literal_requirements"),
        "public_text_literal_hits": row.get("public_text_literal_hits"),
        "current_artifact_literal_hits": row.get("current_artifact_literal_hits"),
        "replay_artifact_literal_hits": row.get("replay_artifact_literal_hits"),
        "current_selector_guard": row.get("current_selector", {}).get("guard_decision"),
        "replay_selector_guard": row.get("replay_selector", {}).get("guard_decision"),
    }


def build_gate(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "records_scanned_positive": summary.get("records_scanned", 0) > 0,
        "code_like_literal_bucket_present": summary.get("by_type", {}).get("code_like_literal", {}).get("records", 0) > 0,
        "failure_buckets_present": bool(summary.get("failure_buckets")),
        "no_gold_fields_in_public_outputs": not summary.get("forbidden_gold_fields_present"),
        "does_not_claim_artifact_lift": True,
        "not_official_score": True,
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == r053.DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {"schema_version": "r069_dataset_artifact_health_gate_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "r069_dataset_artifact_health_audit_complete" if not hard_failures else "r069_dataset_artifact_health_audit_invalid", "gate_passed": not hard_failures, "checks": checks, "hard_failures": hard_failures, "not_full_qa": True, "not_official_score": True, "not_artifact_lift_claim": True}


def build_report(args: argparse.Namespace, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = audit["summary"]
    return {"schema_version": "r069_dataset_artifact_health_report_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "decision": gate["decision"], "scope": summary["boundary"], "summary": summary, "gate": gate, "recommended_next": recommendations(summary)}


def recommendations(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    exact = summary.get("by_type", {}).get("exact_code_lookup", {})
    code_like = summary.get("by_type", {}).get("code_like_literal", {})
    if code_like:
        rows.append("Treat code-like literal rows as a broad health bucket; inspect exact-code lookup separately from years, quarters, and metric names.")
    if exact:
        rows.append("Use R069 exact-code lookup rows to separate retrieval/text absence from artifact extraction and selector failures before any provider run.")
        if exact.get("replay_selector_selected", 0) > exact.get("current_selector_selected", 0):
            rows.append("Rebuild a bounded Stage 2 artifact store with code/name extraction before testing exact-code positive provider cases.")
        else:
            rows.append("If replay does not improve exact-code lookup selection, inspect positive examples and repair extraction patterns before rebuilding the store.")
    rows.append("Do not run full QA until artifact health shows positive cases with visible text, generated artifacts, and selector-selected support.")
    return rows


def write_gate_markdown(path: Path, gate: Mapping[str, Any]) -> None:
    lines = ["# R069 Dataset Artifact Health Gate", "", f"Decision: `{gate['decision']}`", f"Gate passed: {gate['gate_passed']}", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Dataset-level public retrieval/artifact health audit only.", "- Not an official score and not an artifact-lift claim.", "", "## Checks"]
    lines.extend(f"- `{key}`: {value}" for key, value in gate["checks"].items())
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        lines.extend(f"- {item}" for item in gate["hard_failures"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    exact = summary.get("by_type", {}).get("exact_code_lookup", {})
    code_like = summary.get("by_type", {}).get("code_like_literal", {})
    lines = ["# R069 Dataset Artifact Health Audit", "", f"Decision: `{report['decision']}`", "", "## Boundary", "- No provider calls, no prediction, no evaluation, no full QA.", "- Uses public questions, public retrieved page text, and public artifacts only.", "- Does not use answers, evidence pages, official scoring, or artifact-lift claims.", "", "## Summary", f"- records scanned: {summary['records_scanned']}", f"- top-k per modality: {summary['top_k']}", f"- code-like literal records: {code_like.get('records', 0)}",
        f"- exact-code lookup records: {exact.get('records', 0)}", f"- exact-code lookup public text literal present: {exact.get('public_text_literal_present', 0)}", f"- exact-code lookup current artifact literal present: {exact.get('current_artifact_literal_present', 0)}", f"- exact-code lookup replay artifact literal present: {exact.get('replay_artifact_literal_present', 0)}", f"- exact-code lookup current selector selected: {exact.get('current_selector_selected', 0)}", f"- exact-code lookup replay selector selected: {exact.get('replay_selector_selected', 0)}", "", "## Failure Buckets"]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.get("failure_buckets", {}).items())
    lines.extend(["", "## Recommended Next"])
    lines.extend(f"- {item}" for item in report["recommended_next"])
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
