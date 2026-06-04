#!/usr/bin/env python3
"""R054 no-provider guarded selector repair gate.

This runner turns the manual R053 review notes for records 384, 508, and
569 into hard selector/prompt guards. It does not call providers, run
prediction, run evaluation, or report a score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_r053_question_aware_scaffold as r053

DEFAULT_R045_CASES = r053.DEFAULT_R045_CASES
DEFAULT_R044_REPORT = r053.DEFAULT_R044_REPORT
DEFAULT_R040_ROOT = r053.DEFAULT_R040_ROOT
DEFAULT_R039_RECORD_IDS = r053.DEFAULT_R039_RECORD_IDS
DEFAULT_RECORDS = r053.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r053.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r054_guarded_selector_repair"

METADATA_TERMS = {"produced", "producer", "revised", "revision", "document", "version"}
COMPUTE_TERMS = {"difference", "sum", "total", "ratio", "rate", "percentage", "calculate"}
NUMERIC_ARTIFACT_TYPES = {"numeric_fact", "table_cell", "table"}
CODE_FIELD_KEYS = {"code", "eps_code", "row_label", "row_header", "metric_name", "column_label", "column_header"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r045-cases", default=DEFAULT_R045_CASES)
    parser.add_argument("--r044-report", default=DEFAULT_R044_REPORT)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=1400)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--max-artifact-chars", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "no_provider_calls": True,
            "no_prediction_or_eval": True,
            "no_full_qa": True,
            "design_gate_only": True,
            "manual_feedback_records": [384, 508, 569],
        }, indent=2, ensure_ascii=False))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    r045_cases = r053.read_jsonl(Path(args.r045_cases))
    r044_report = r053.read_json(Path(args.r044_report))
    records = r053.read_json(Path(args.records))
    record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))
    previews = build_previews(args, r045_cases, r044_report, records, offsets, run_records, artifacts_by_page)
    gate = build_gate(args, r045_cases, previews)
    report = build_report(args, previews, gate)

    report_json = output_root / "r054_guarded_selector_repair_report.json"
    report_md = output_root / "r054_guarded_selector_repair_report.md"
    gate_json = output_root / "r054_guarded_selector_repair_gate.json"
    gate_md = output_root / "r054_guarded_selector_repair_gate.md"
    previews_path = output_root / "r054_guarded_prompt_previews.jsonl"
    compact_path = output_root / "r054_guarded_compact_index.jsonl"
    r053.write_jsonl(previews_path, previews)
    r053.write_jsonl(compact_path, build_compact_index(previews))
    r053.write_json(gate_json, gate)
    write_gate_markdown(gate_md, gate)
    r053.write_json(report_json, report)
    write_report_markdown(report_md, report)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_cases": len(previews),
        "report_md": str(report_md),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, indent=2, ensure_ascii=False))


def build_previews(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    r044_report: dict[str, Any],
    records: list[dict[str, Any]],
    offsets: dict[int, int],
    run_records: dict[str, list[dict[str, Any]]],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    r044_by_id = {int(row["record_id"]): row for row in r044_report["per_record"]}
    rows = []
    for case in cases:
        record_id = int(case["record_id"])
        if record_id not in offsets:
            raise ValueError(f"R045 case record_id not in R039 subset: {record_id}")
        source = records[record_id]
        doc_id = str(source["doc_id"])
        offset = offsets[record_id]
        original_record = run_records["top4_original_only"][offset]
        artifact_record = run_records["top4_artifact_only"][offset]
        original_pages = r053.combined_pages(original_record)
        artifact_pages = r053.combined_pages(artifact_record)
        candidate_pages = r053.unique_ints(artifact_pages + original_pages)
        question = str(source["question"])
        profile = guarded_question_profile(question)
        page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        artifact_candidates = []
        for page in candidate_pages:
            for artifact in artifacts_by_page.get((doc_id, page), []):
                artifact_candidates.append(score_guarded_artifact(artifact, question, profile, page, artifact_pages, original_pages, args.max_artifact_chars))
        selection = select_guarded_artifacts(artifact_candidates, page_contexts, profile, args.max_artifacts)
        prompt = render_prompt(question, page_contexts, selection, profile)
        public_payload = {
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "question_profile": profile,
            "retrieval_pages": {"artifact": artifact_pages, "original": original_pages, "candidate_union": candidate_pages},
            "selection": selection,
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
        }
        rows.append({
            "schema_version": "r054_guarded_selector_prompt_preview_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "case_type": case.get("case_type"),
            "r045_rubric_label": case.get("rubric_label"),
            "r045_artifact_evidence_status": case.get("artifact_evidence_status"),
            "r045_page_text_evidence_status": case.get("page_text_evidence_status"),
            "r044_transition_labels": r044_by_id.get(record_id, {}).get("transition_labels", []),
            "manual_feedback_applied": manual_feedback(record_id),
            "question_profile": profile,
            "retrieval_pages": {
                "top4_artifact_only_combined": artifact_pages,
                "top4_original_only_combined": original_pages,
                "candidate_union": candidate_pages,
            },
            "selection_policy": {
                "name": "guarded_selector_repair_v1",
                "not_first_n_per_page": True,
                "uses_question_tokens": True,
                "uses_metadata_refusal_route": True,
                "uses_exact_code_key_value_selection": True,
                "uses_operand_completeness_guard": True,
                "uses_retrieved_candidate_pages_only": True,
                "uses_gold_fields": False,
                "unsupported_answer_guard": True,
            },
            "candidate_artifact_count": len(artifact_candidates),
            "rejected_artifact_count": selection["rejected_artifact_count"],
            "selected_artifact_count": len(selection["selected_artifacts"]),
            "selected_artifacts": selection["selected_artifacts"],
            "guard_decision": selection["guard_decision"],
            "guard_reasons": selection["guard_reasons"],
            "answer_policy": selection["answer_policy"],
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
            "prompt_preview_sha256": r053.sha256(prompt),
            "forbidden_gold_fields_present": r053.forbidden_gold_fields(public_payload),
        })
    return rows


def guarded_question_profile(question: str) -> dict[str, Any]:
    base = r053.question_profile(question)
    tokens = set(base["tokens"])
    q_norm = r053.normalize(question)
    is_metadata_lookup = bool(tokens & METADATA_TERMS) and any(term in q_norm for term in ["revised", "produced", "producer", "document"])
    is_computation = bool(tokens & COMPUTE_TERMS) or any(term in q_norm for term in ["percentage difference", "return on asset", "round your answer"])
    required_operands = infer_required_operands(question)
    base.update({
        "is_document_metadata_lookup": is_metadata_lookup,
        "is_computation_question": is_computation,
        "requires_exact_code_selection": bool(base["codes"]),
        "required_operands": required_operands,
        "answer_policy": "cite_visible_support_or_refuse",
    })
    return base


def infer_required_operands(question: str) -> list[str]:
    q_norm = r053.normalize(question)
    operands = []
    if "older" in q_norm or "older age" in q_norm:
        operands.append("older_age_group")
    if "children" in q_norm:
        operands.append("children")
    if "received" in q_norm and "stem" in q_norm and "degree" in q_norm:
        operands.append("received_stem_degree")
    if "employed" in q_norm and "field" in q_norm:
        operands.append("employed_in_field")
    if "net income" in q_norm or "return on asset" in q_norm or "roa" in q_norm:
        operands.extend(["net_income", "total_assets"])
    return sorted(set(operands))


def score_guarded_artifact(
    artifact: dict[str, Any],
    question: str,
    profile: dict[str, Any],
    page: int,
    artifact_pages: list[int],
    original_pages: list[int],
    max_chars: int,
) -> dict[str, Any]:
    scored = r053.score_artifact(artifact, question, profile, page, artifact_pages, original_pages, max_chars)
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    searchable = searchable_artifact_text(artifact, normalized)
    exact_codes = [code for code in profile["codes"] if re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", searchable, re.IGNORECASE)]
    key_value_hits = sorted(set(profile["tokens"]) & r053.question_tokens(" ".join(str(normalized.get(key) or "") for key in CODE_FIELD_KEYS)))
    operand_hits = sorted(operand for operand in profile["required_operands"] if operand_covered(operand, searchable))
    scored.update({
        "exact_code_matches": exact_codes,
        "key_value_token_hits": key_value_hits,
        "operand_hits": operand_hits,
        "is_numeric_or_table_artifact": scored["artifact_type"] in NUMERIC_ARTIFACT_TYPES,
    })
    return scored


def searchable_artifact_text(artifact: dict[str, Any], normalized: dict[str, Any]) -> str:
    return " ".join([
        str(artifact.get("content") or ""),
        json.dumps(normalized, ensure_ascii=False, sort_keys=True),
    ])


def operand_covered(operand: str, searchable: str) -> bool:
    text = r053.normalize(searchable)
    checks = {
        "older_age_group": ["older", "age", "25"],
        "children": ["children", "child", "k-12", "student"],
        "received_stem_degree": ["stem", "degree"],
        "employed_in_field": ["employed", "occupation", "field", "working"],
        "net_income": ["net income"],
        "total_assets": ["total assets", "assets"],
    }
    return any(term in text for term in checks.get(operand, [operand]))


def select_guarded_artifacts(candidates: list[dict[str, Any]], page_contexts: list[dict[str, Any]], profile: dict[str, Any], max_artifacts: int) -> dict[str, Any]:
    reasons = []
    rejected = list(candidates)
    if profile["is_document_metadata_lookup"]:
        reasons.append("document_metadata_lookup_uses_page_text_not_numeric_artifacts")
        page_signal = metadata_page_signal(page_contexts, profile)
        return selection_result([], rejected, "document_metadata_refusal_guard", reasons + page_signal, "refuse_if_visible_metadata_mismatches_question")

    if profile["requires_exact_code_selection"]:
        exact = [row for row in candidates if row["exact_code_matches"]]
        if not exact:
            reasons.append("no_artifact_contains_exact_question_code")
            reasons.append("numeric_artifacts_rejected_without_exact_code_key")
            return selection_result([], rejected, "exact_code_absence_guard", reasons, "use_page_evidence_for_absence_or_refuse")
        selected = rank_artifacts(exact)[:max_artifacts]
        return selection_result(selected, [row for row in rejected if row not in selected], "exact_code_key_value_selection", ["selected_only_exact_code_artifacts"], "answer_only_if_exact_code_value_pair_supports_it")

    if profile["is_computation_question"] and profile["required_operands"]:
        selected = rank_artifacts(candidates)[:max_artifacts]
        covered = sorted({operand for row in selected for operand in row["operand_hits"]})
        missing = sorted(set(profile["required_operands"]) - set(covered))
        if missing:
            reasons.extend(["operand_completeness_failed", "missing_operands:" + ",".join(missing)])
            return selection_result([], rejected, "operand_completeness_guard", reasons, "not_answerable_due_to_incomplete_operands")
        return selection_result(selected, [row for row in rejected if row not in selected], "operand_complete_selection", ["all_required_operands_covered"], "calculate_only_from_cited_operands")

    eligible = [row for row in candidates if row["question_token_overlap"] or row["key_value_token_hits"]]
    selected = rank_artifacts(eligible)[:max_artifacts]
    if selected:
        return selection_result(selected, [row for row in rejected if row not in selected], "token_key_value_selection", ["selected_question_overlapping_artifacts"], "cite_visible_support_or_refuse")
    return selection_result([], rejected, "no_relevant_artifact_guard", ["no_question_overlapping_artifacts"], "use_page_evidence_or_refuse")


def metadata_page_signal(page_contexts: list[dict[str, Any]], profile: dict[str, Any]) -> list[str]:
    joined = " ".join(ctx["text_preview"] for ctx in page_contexts)
    joined_norm = r053.normalize(joined)
    signals = []
    if "produced" in joined_norm or "revised" in joined_norm:
        signals.append("visible_page_metadata_present")
    for number in [str(num) for num in profile["numbers"]]:
        if number not in joined:
            signals.append(f"question_date_not_visible:{number}")
    return signals


def rank_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row["selection_score"]), int(row["page_index"]), row["artifact_type"], row["artifact_id"]))
    result = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["selection_rank"] = rank
        result.append(item)
    return result


def selection_result(selected: list[dict[str, Any]], rejected: list[dict[str, Any]], decision: str, reasons: list[str], answer_policy: str) -> dict[str, Any]:
    return {
        "guard_decision": decision,
        "guard_reasons": sorted(set(reasons)),
        "answer_policy": answer_policy,
        "selected_artifacts": selected,
        "rejected_artifact_count": len(rejected),
    }


def render_prompt(question: str, page_contexts: list[dict[str, Any]], selection: dict[str, Any], profile: dict[str, Any]) -> str:
    lines = [
        "[R054 condition: guarded_selector_repair_prompt]",
        "Answer using only the visible page evidence and selected artifact evidence below.",
        "First list supporting evidence, then answer. Cite page ids and artifact ids for every factual claim.",
        "If the guard decision says metadata/refusal, exact-code absence, or operand incompleteness, do not compute or infer from partial artifact snippets.",
        "If the visible evidence does not fully support an answer, say Not answerable and cite what is missing.",
        f"Question: {question}",
        "",
        "[Question profile]",
        f"metadata_lookup={profile['is_document_metadata_lookup']}; computation={profile['is_computation_question']}; codes={profile['codes']}; required_operands={profile['required_operands']}",
        "",
        "[Guard decision]",
        f"decision={selection['guard_decision']}; answer_policy={selection['answer_policy']}; reasons={selection['guard_reasons']}",
        "",
        "[Page evidence]",
    ]
    if page_contexts:
        for ctx in page_contexts:
            lines.append(f"Page {ctx['page_index']} ({'present' if ctx['exists'] else 'missing'}): {ctx['text_preview']}")
    else:
        lines.append("No page evidence is visible.")
    lines.extend(["", "[Selected artifact evidence]"])
    if selection["selected_artifacts"]:
        for item in selection["selected_artifacts"]:
            lines.append(
                f"{item['artifact_id']} | page {item['page_index']} | type={item['artifact_type']} | score={item['selection_score']} | exact_codes={item['exact_code_matches']} | operands={item['operand_hits']} | {item['content_preview']}"
            )
    else:
        lines.append("No artifact evidence was selected because the guard rejected the available snippets as insufficient or irrelevant.")
    lines.extend([
        "",
        "[Required response format]",
        "Page evidence: cite page ids or state none.",
        "Artifact evidence: cite artifact ids or state none.",
        "Guard check: state whether metadata, exact-code, and operand requirements are satisfied.",
        "Unsupported-answer check: explain whether the visible evidence fully supports the answer.",
        "Final answer: answer or Not answerable.",
    ])
    return "\n".join(lines).strip() + "\n"


def manual_feedback(record_id: int) -> dict[str, Any] | None:
    notes = {
        384: {
            "source": "user_manual_r053_review",
            "finding": "fail artifact selection; route to document-metadata/refusal instead of numeric table artifacts",
            "required_guard": "document_metadata_refusal_guard",
        },
        508: {
            "source": "user_manual_r053_review",
            "finding": "page evidence supports unsupported answer; numeric artifacts are noise; require exact-code/key-value selection",
            "required_guard": "exact_code_absence_guard_or_exact_code_key_value_selection",
        },
        569: {
            "source": "user_manual_r053_review",
            "finding": "artifact snippets do not cover calculation operands; trigger operand-completeness guard",
            "required_guard": "operand_completeness_guard",
        },
    }
    return notes.get(record_id)


def build_gate(args: argparse.Namespace, cases: list[dict[str, Any]], previews: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["record_id"]: row for row in previews}
    prompt_texts = [row["prompt_preview"] for row in previews]
    selected_counts = [row["selected_artifact_count"] for row in previews]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_cases_match_r045": sorted(int(c["record_id"]) for c in cases) == sorted(row["record_id"] for row in previews),
        "all_prompts_have_citation_requirement": all("Cite page ids and artifact ids" in text for text in prompt_texts),
        "all_prompts_have_unsupported_answer_guard": all("Not answerable" in text and "do not compute or infer" in text for text in prompt_texts),
        "page_and_artifact_evidence_separated": all("[Page evidence]" in text and "[Selected artifact evidence]" in text for text in prompt_texts),
        "selected_artifact_budget_respected": all(count <= args.max_artifacts for count in selected_counts),
        "no_gold_fields_in_public_previews": all(not row["forbidden_gold_fields_present"] for row in previews),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "r384_metadata_refusal_guard": record_guard(by_id, 384, "document_metadata_refusal_guard") and by_id.get(384, {}).get("selected_artifact_count") == 0,
        "r384_no_numeric_table_artifacts_selected": not any(item["artifact_type"] in NUMERIC_ARTIFACT_TYPES for item in by_id.get(384, {}).get("selected_artifacts", [])),
        "r508_exact_code_or_absence_guard": by_id.get(508, {}).get("guard_decision") in {"exact_code_absence_guard", "exact_code_key_value_selection"},
        "r508_no_artifact_without_exact_ar03": all("AR03" in item.get("exact_code_matches", []) for item in by_id.get(508, {}).get("selected_artifacts", [])),
        "r569_operand_completeness_guard": record_guard(by_id, 569, "operand_completeness_guard") and by_id.get(569, {}).get("answer_policy") == "not_answerable_due_to_incomplete_operands",
        "design_gate_only": True,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r054_guarded_selector_repair_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r054_guarded_selector_repair_gate_pass" if not hard_failures else "r054_guarded_selector_repair_gate_fail",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_cases": len(previews),
        "selected_artifact_count_by_record": {str(row["record_id"]): row["selected_artifact_count"] for row in previews},
        "guard_decision_by_record": {str(row["record_id"]): row["guard_decision"] for row in previews},
        "not_full_qa": True,
        "not_official_score": True,
    }


def record_guard(by_id: dict[int, dict[str, Any]], record_id: int, expected: str) -> bool:
    return by_id.get(record_id, {}).get("guard_decision") == expected


def build_report(args: argparse.Namespace, previews: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    rubric_counts = Counter(row["r045_rubric_label"] for row in previews)
    guard_counts = Counter(row["guard_decision"] for row in previews)
    reason_counts = Counter(reason for row in previews for reason in row["guard_reasons"])
    return {
        "schema_version": "r054_guarded_selector_repair_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r054_guarded_selector_repair_complete" if gate["gate_passed"] else "r054_guarded_selector_repair_needs_fix",
        "scope": {
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "selector_and_prompt_guard_only": True,
        },
        "inputs": {
            "r045_cases": args.r045_cases,
            "r044_report": args.r044_report,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
        },
        "num_cases": len(previews),
        "rubric_label_counts": dict(sorted(rubric_counts.items())),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "guard_reason_counts": dict(sorted(reason_counts.items())),
        "manual_repair_summary": {
            "384": "document metadata/refusal route; numeric/table artifacts rejected",
            "508": "exact AR03 key-value selection required; otherwise use page evidence for unsupported/refusal",
            "569": "operand-completeness guard blocks calculation from partial snippets",
        },
        "gate": gate,
        "recommended_next": [
            "Do not run full QA from R054.",
            "Manually inspect R054 prompt previews for 384, 508, and 569 against the recorded guard decisions.",
            "If accepted, run only a tiny provider diagnostic on the guarded prompts; treat it as diagnostic attribution, not an official score.",
            "If rejected, repair selector guards again before any provider call.",
        ],
    }


def build_compact_index(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "r054_guarded_compact_index_v1",
        "record_id": row["record_id"],
        "doc_id": row["doc_id"],
        "case_type": row["case_type"],
        "r045_rubric_label": row["r045_rubric_label"],
        "r044_transition_labels": row["r044_transition_labels"],
        "guard_decision": row["guard_decision"],
        "guard_reasons": row["guard_reasons"],
        "answer_policy": row["answer_policy"],
        "selected_artifact_count": row["selected_artifact_count"],
        "selected_artifact_ids": [artifact["artifact_id"] for artifact in row["selected_artifacts"]],
        "selected_artifact_pages": sorted({artifact["page_index"] for artifact in row["selected_artifacts"]}),
        "prompt_preview_sha256": row["prompt_preview_sha256"],
        "question_profile": row["question_profile"],
    } for row in previews]


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R054 Guarded Selector Repair Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Selector and prompt guard only.",
        "- Not an official score.",
        "",
        "## Checks",
    ]
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        for item in gate["hard_failures"]:
            lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R054 Guarded Selector Repair",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Converts manual R053 feedback for records 384, 508, and 569 into hard selector/prompt guards.",
        "- Not an official score.",
        "",
        "## Summary",
        f"- cases: {report['num_cases']}",
        f"- rubric labels: `{json.dumps(report['rubric_label_counts'], sort_keys=True)}`",
        f"- guard decisions: `{json.dumps(report['guard_decision_counts'], sort_keys=True)}`",
        f"- guard reasons: `{json.dumps(report['guard_reason_counts'], sort_keys=True)}`",
        "",
        "## Manual Repair Summary",
    ]
    for key, value in report["manual_repair_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
