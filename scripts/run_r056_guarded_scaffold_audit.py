#!/usr/bin/env python3
"""R056 no-provider guarded selector/prompt scaffold audit.

R056 extracts the R054/R055 refusal guards into the reusable
``mdocnexus.integration.guarded_prompt`` scaffold and audits it on the same
R045 diagnostic cases. It does not call providers, run prediction, run
evaluation, or report a score.

The key audit is two-sided:
1. refusal/noise cases 384, 508, and 569 must be guarded and clear artifacts;
2. non-refusal cases with positive artifact signals must not be globally cleared.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053

from mdocnexus.integration.guarded_prompt import (
    build_question_profile,
    forbidden_public_fields,
    render_guarded_prompt,
    score_guarded_artifact,
    select_guarded_artifacts,
    sha256,
)

DEFAULT_R045_CASES = r053.DEFAULT_R045_CASES
DEFAULT_R044_REPORT = r053.DEFAULT_R044_REPORT
DEFAULT_R040_ROOT = r053.DEFAULT_R040_ROOT
DEFAULT_R039_RECORD_IDS = r053.DEFAULT_R039_RECORD_IDS
DEFAULT_RECORDS = r053.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r053.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r056_guarded_scaffold_audit"
REFUSAL_GUARD_RECORDS = {384, 508, 569}


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
            "scaffold_only": True,
            "audit_focus": "refusal guards plus positive-artifact preservation",
        }, ensure_ascii=False, indent=2))
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

    previews_path = output_root / "r056_guarded_prompt_previews.jsonl"
    compact_path = output_root / "r056_guarded_compact_index.jsonl"
    gate_json = output_root / "r056_guarded_scaffold_gate.json"
    gate_md = output_root / "r056_guarded_scaffold_gate.md"
    report_json = output_root / "r056_guarded_scaffold_report.json"
    report_md = output_root / "r056_guarded_scaffold_report.md"
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
        "positive_signal_cases": gate["positive_signal_case_count"],
        "positive_signal_cases_cleared": gate["positive_signal_cases_cleared"],
        "report_md": str(report_md),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, ensure_ascii=False, indent=2))


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
        profile = build_question_profile(question)
        page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        candidates = []
        for page in candidate_pages:
            for artifact in artifacts_by_page.get((doc_id, page), []):
                candidates.append(
                    score_guarded_artifact(
                        artifact,
                        question,
                        profile,
                        page,
                        artifact_pages=artifact_pages,
                        original_pages=original_pages,
                        max_chars=args.max_artifact_chars,
                    )
                )
        selection = select_guarded_artifacts(candidates, page_contexts, profile, max_artifacts=args.max_artifacts)
        prompt = render_guarded_prompt(question, page_contexts, selection, profile, condition_label="R056 condition: reusable_guarded_selector_prompt")
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
        positive_candidate_count = int(selection["positive_candidate_count"])
        selected_artifact_count = len(selection["selected_artifacts"])
        rows.append({
            "schema_version": "r056_guarded_scaffold_prompt_preview_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "case_type": case.get("case_type"),
            "r045_rubric_label": case.get("rubric_label"),
            "r045_artifact_evidence_status": case.get("artifact_evidence_status"),
            "r045_page_text_evidence_status": case.get("page_text_evidence_status"),
            "r044_transition_labels": r044_by_id.get(record_id, {}).get("transition_labels", []),
            "question_profile": profile,
            "retrieval_pages": {
                "top4_artifact_only_combined": artifact_pages,
                "top4_original_only_combined": original_pages,
                "candidate_union": candidate_pages,
            },
            "selection_policy": {
                "name": "reusable_guarded_selector_prompt_scaffold_v1",
                "module": "mdocnexus.integration.guarded_prompt",
                "not_first_n_per_page": True,
                "uses_metadata_refusal_route": True,
                "uses_exact_code_key_value_selection": True,
                "uses_operand_completeness_guard": True,
                "uses_positive_signal_preservation_audit": True,
                "uses_retrieved_candidate_pages_only": True,
                "uses_gold_fields": False,
                "unsupported_answer_guard": True,
            },
            "candidate_artifact_count": len(candidates),
            "positive_candidate_count": positive_candidate_count,
            "rejected_artifact_count": selection["rejected_artifact_count"],
            "selected_artifact_count": selected_artifact_count,
            "positive_signal_case": positive_candidate_count > 0,
            "positive_signal_cleared": record_id not in REFUSAL_GUARD_RECORDS and positive_candidate_count > 0 and selected_artifact_count == 0,
            "selected_artifacts": selection["selected_artifacts"],
            "guard_decision": selection["guard_decision"],
            "guard_reasons": selection["guard_reasons"],
            "answer_policy": selection["answer_policy"],
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
            "prompt_preview_sha256": sha256(prompt),
            "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        })
    return rows


def build_gate(args: argparse.Namespace, cases: list[dict[str, Any]], previews: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["record_id"]: row for row in previews}
    prompt_texts = [row["prompt_preview"] for row in previews]
    positive_signal_rows = [row for row in previews if row["record_id"] not in REFUSAL_GUARD_RECORDS and row["positive_signal_case"]]
    positive_signal_cases_cleared = [row["record_id"] for row in positive_signal_rows if row["positive_signal_cleared"]]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_cases_match_r045": sorted(int(c["record_id"]) for c in cases) == sorted(row["record_id"] for row in previews),
        "uses_reusable_guarded_prompt_module": all(row["selection_policy"]["module"] == "mdocnexus.integration.guarded_prompt" for row in previews),
        "all_prompts_have_citation_requirement": all("Cite page ids and artifact ids" in text for text in prompt_texts),
        "all_prompts_have_unsupported_answer_guard": all("Not answerable" in text and "do not compute or infer" in text for text in prompt_texts),
        "page_and_artifact_evidence_separated": all("[Page evidence]" in text and "[Selected artifact evidence]" in text for text in prompt_texts),
        "no_gold_fields_in_public_previews": all(not row["forbidden_gold_fields_present"] for row in previews),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "r384_metadata_refusal_guard": by_id.get(384, {}).get("guard_decision") == "document_metadata_refusal_guard" and by_id.get(384, {}).get("selected_artifact_count") == 0,
        "r508_exact_code_absence_guard": by_id.get(508, {}).get("guard_decision") == "exact_code_absence_guard" and by_id.get(508, {}).get("selected_artifact_count") == 0,
        "r569_operand_completeness_guard": by_id.get(569, {}).get("guard_decision") == "operand_completeness_guard" and by_id.get(569, {}).get("selected_artifact_count") == 0,
        "has_positive_signal_non_refusal_cases": len(positive_signal_rows) >= 1,
        "positive_signal_cases_not_all_cleared": len(positive_signal_cases_cleared) == 0,
        "not_artifact_lift_claim": True,
        "scaffold_only": True,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r056_guarded_scaffold_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r056_guarded_scaffold_gate_pass" if not hard_failures else "r056_guarded_scaffold_gate_fail",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_cases": len(previews),
        "positive_signal_case_count": len(positive_signal_rows),
        "positive_signal_cases_cleared": positive_signal_cases_cleared,
        "selected_artifact_count_by_record": {str(row["record_id"]): row["selected_artifact_count"] for row in previews},
        "guard_decision_by_record": {str(row["record_id"]): row["guard_decision"] for row in previews},
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, previews: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    guard_counts = Counter(row["guard_decision"] for row in previews)
    rubric_counts = Counter(row["r045_rubric_label"] for row in previews)
    positive_rows = [row for row in previews if row["record_id"] not in REFUSAL_GUARD_RECORDS and row["positive_signal_case"]]
    return {
        "schema_version": "r056_guarded_scaffold_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r056_guarded_scaffold_complete" if gate["gate_passed"] else "r056_guarded_scaffold_needs_fix",
        "scope": {
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "reusable_scaffold_only": True,
            "does_not_prove_artifact_positive_lift": True,
        },
        "inputs": {
            "r045_cases": args.r045_cases,
            "r044_report": args.r044_report,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
        },
        "module": "mdocnexus.integration.guarded_prompt",
        "num_cases": len(previews),
        "rubric_label_counts": dict(sorted(rubric_counts.items())),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "positive_signal_case_count": len(positive_rows),
        "positive_signal_case_records": [row["record_id"] for row in positive_rows],
        "positive_signal_cases_cleared": gate["positive_signal_cases_cleared"],
        "refusal_guard_summary": {
            "384": "document metadata/refusal; selected artifacts = 0",
            "508": "exact-code absence/refusal; selected artifacts = 0",
            "569": "operand-completeness/refusal; selected artifacts = 0",
        },
        "gate": gate,
        "recommended_next": [
            "Do not run full QA from R056.",
            "Review whether the reusable scaffold should be wired into the adapter/prompt path behind an explicit config flag.",
            "Before broad QA, run one more no-provider or tiny-provider diagnostic on explicit positive-evidence cases if the paper needs artifact-use evidence.",
            "Keep claims limited: R056 audits scaffold safety and positive-signal preservation, not QA improvement.",
        ],
    }


def build_compact_index(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "r056_guarded_compact_index_v1",
        "record_id": row["record_id"],
        "doc_id": row["doc_id"],
        "case_type": row["case_type"],
        "r045_rubric_label": row["r045_rubric_label"],
        "r044_transition_labels": row["r044_transition_labels"],
        "guard_decision": row["guard_decision"],
        "guard_reasons": row["guard_reasons"],
        "answer_policy": row["answer_policy"],
        "candidate_artifact_count": row["candidate_artifact_count"],
        "positive_candidate_count": row["positive_candidate_count"],
        "positive_signal_case": row["positive_signal_case"],
        "positive_signal_cleared": row["positive_signal_cleared"],
        "selected_artifact_count": row["selected_artifact_count"],
        "selected_artifact_ids": [artifact["artifact_id"] for artifact in row["selected_artifacts"]],
        "selected_artifact_pages": sorted({artifact["page_index"] for artifact in row["selected_artifacts"]}),
        "prompt_preview_sha256": row["prompt_preview_sha256"],
        "question_profile": row["question_profile"],
    } for row in previews]


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R056 Guarded Scaffold Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Reusable guarded selector/prompt scaffold only.",
        "- Audits refusal guards and positive-signal preservation.",
        "- Not an official score and not an artifact-lift claim.",
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
        "# R056 Guarded Selector/Prompt Scaffold Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Extracts R054/R055 refusal guards into `mdocnexus.integration.guarded_prompt`.",
        "- Checks that refusal cases are guarded and positive-signal cases are not all cleared.",
        "- Not an official score and not evidence of artifact positive lift.",
        "",
        "## Summary",
        f"- cases: {report['num_cases']}",
        f"- guard decisions: `{json.dumps(report['guard_decision_counts'], sort_keys=True)}`",
        f"- positive signal case records: `{report['positive_signal_case_records']}`",
        f"- positive signal cases cleared: `{report['positive_signal_cases_cleared']}`",
        "",
        "## Refusal Guard Summary",
    ]
    for key, value in report["refusal_guard_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()