#!/usr/bin/env python3
"""R060 no-provider page/artifact routing audit.

R060 audits the post-R059 prompt route for cases where page evidence is visible
and sufficient but artifact snippets are rejected by the dimension-support
guard. It does not call providers, run prediction, run evaluation, run full QA,
or report a score.
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
    audit_selected_artifact_support,
    build_question_profile,
    forbidden_public_fields,
    render_guarded_prompt,
    score_guarded_artifact,
    select_guarded_artifacts,
    sha256,
)

DEFAULT_R040_ROOT = r053.DEFAULT_R040_ROOT
DEFAULT_R039_RECORD_IDS = r053.DEFAULT_R039_RECORD_IDS
DEFAULT_RECORDS = r053.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r053.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r060_page_artifact_routing_audit"
DEFAULT_TARGET_RECORD_IDS = "223,227"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--target-record-ids", default=DEFAULT_TARGET_RECORD_IDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=1800)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    record_ids = parse_record_ids(args.target_record_ids)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "target_record_ids": record_ids,
            "no_provider_calls": True,
            "no_prediction_or_eval": True,
            "no_full_qa": True,
            "audit_focus": "page evidence routing when artifacts are dimension-guarded",
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    records = r053.read_json(Path(args.records))
    frozen_record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(frozen_record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))

    previews = build_previews(args, record_ids, records, offsets, run_records, artifacts_by_page)
    gate = build_gate(args, record_ids, previews)
    report = build_report(args, previews, gate)

    r053.write_jsonl(output_root / "r060_routing_prompt_previews.jsonl", previews)
    r053.write_jsonl(output_root / "r060_routing_compact_index.jsonl", build_compact_index(previews))
    r053.write_json(output_root / "r060_routing_gate.json", gate)
    write_gate_markdown(output_root / "r060_routing_gate.md", gate)
    r053.write_json(output_root / "r060_routing_report.json", report)
    write_report_markdown(output_root / "r060_routing_report.md", report)

    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_cases": len(previews),
        "page_routed_records": gate["page_routed_records"],
        "report_md": str(output_root / "r060_routing_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, ensure_ascii=False, indent=2))


def build_previews(
    args: argparse.Namespace,
    record_ids: list[int],
    records: list[dict[str, Any]],
    offsets: dict[int, int],
    run_records: dict[str, list[dict[str, Any]]],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for record_id in record_ids:
        if record_id not in offsets:
            raise ValueError(f"target record_id is not in R039 frozen subset: {record_id}")
        source = records[record_id]
        doc_id = str(source["doc_id"])
        question = str(source["question"])
        offset = offsets[record_id]
        original_record = run_records["top4_original_only"][offset]
        artifact_record = run_records["top4_artifact_only"][offset]
        original_pages = r053.combined_pages(original_record)
        artifact_pages = r053.combined_pages(artifact_record)
        candidate_pages = r053.unique_ints(artifact_pages + original_pages)
        profile = build_question_profile(question)
        page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        candidates = []
        for page in candidate_pages:
            for artifact in artifacts_by_page.get((doc_id, page), []):
                candidates.append(score_guarded_artifact(
                    artifact,
                    question,
                    profile,
                    page,
                    artifact_pages=artifact_pages,
                    original_pages=original_pages,
                    max_chars=args.max_artifact_chars,
                ))
        selection = select_guarded_artifacts(candidates, page_contexts, profile, max_artifacts=args.max_artifacts)
        support = audit_selected_artifact_support(selection.get("selected_artifacts") or [], page_contexts, profile)
        prompt = render_guarded_prompt(question, page_contexts, selection, profile, condition_label="R060 condition: page_artifact_routing_audit")
        routing = routing_checks(prompt, selection, support, page_contexts)
        public_payload = {
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "question_profile": profile,
            "selection": selection,
            "support_audit": support,
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
        }
        rows.append({
            "schema_version": "r060_routing_prompt_preview_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "retrieval_pages": {
                "top4_artifact_only_combined": artifact_pages,
                "top4_original_only_combined": original_pages,
                "candidate_union": candidate_pages,
            },
            "candidate_artifact_count": len(candidates),
            "positive_candidate_count": selection["positive_candidate_count"],
            "selected_artifact_count": len(selection["selected_artifacts"]),
            "selected_artifacts": selection["selected_artifacts"],
            "guard_decision": selection["guard_decision"],
            "guard_reasons": selection["guard_reasons"],
            "answer_policy": selection["answer_policy"],
            "support_audit": support,
            "routing_checks": routing,
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
            "prompt_preview_sha256": sha256(prompt),
            "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        })
    return rows


def routing_checks(prompt: str, selection: dict[str, Any], support: dict[str, Any], page_contexts: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "artifact_dimension_guard": selection.get("guard_decision") == "artifact_dimension_support_guard",
        "answer_policy_uses_page_or_refuse": selection.get("answer_policy") == "use_page_evidence_or_refuse",
        "selected_artifacts_empty": not selection.get("selected_artifacts"),
        "visible_page_support_sufficient": support.get("visible_support_sufficient") is True,
        "artifact_support_insufficient": support.get("artifact_support_sufficient") is False,
        "page_evidence_visible": any(ctx.get("exists") and str(ctx.get("text_preview") or "").strip() for ctx in page_contexts),
        "prompt_separates_page_and_artifact": "[Page evidence]" in prompt and "[Selected artifact evidence]" in prompt,
        "prompt_names_dimension_guard": "artifact_dimension_support_guard" in prompt,
        "prompt_blocks_rejected_artifact_citation": "do not cite rejected artifact ids" in prompt.lower() and "never cite rejected artifact ids" in prompt.lower(),
        "prompt_routes_to_page_only_when_sufficient": "answer from cited page ids only" in prompt.lower(),
        "prompt_preserves_not_answerable_fallback": "Not answerable" in prompt,
    }
    failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r060_routing_checks_v1",
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def build_gate(args: argparse.Namespace, record_ids: list[int], previews: list[dict[str, Any]]) -> dict[str, Any]:
    page_routed_records = [row["record_id"] for row in previews if row["routing_checks"]["passed"]]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_records_match_page_sufficient_artifact_insufficient_cases": sorted(record_ids) == sorted(row["record_id"] for row in previews),
        "all_cases_page_routed": sorted(page_routed_records) == sorted(record_ids),
        "all_cases_guard_artifacts": all(row["guard_decision"] == "artifact_dimension_support_guard" for row in previews),
        "all_cases_select_zero_artifacts": all(row["selected_artifact_count"] == 0 for row in previews),
        "all_cases_have_visible_page_support": all(row["support_audit"]["visible_support_sufficient"] for row in previews),
        "all_prompts_block_rejected_artifact_citation": all(row["routing_checks"]["checks"]["prompt_blocks_rejected_artifact_citation"] for row in previews),
        "all_prompts_route_to_page_only_when_sufficient": all(row["routing_checks"]["checks"]["prompt_routes_to_page_only_when_sufficient"] for row in previews),
        "no_gold_fields_in_public_previews": all(not row["forbidden_gold_fields_present"] for row in previews),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "not_provider_run": True,
        "not_artifact_lift_claim": True,
        "not_official_score": True,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r060_routing_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r060_routing_gate_pass" if not hard_failures else "r060_routing_gate_fail",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_cases": len(previews),
        "page_routed_records": page_routed_records,
        "routing_failures_by_record": {str(row["record_id"]): row["routing_checks"]["failures"] for row in previews},
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, previews: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    guard_counts = Counter(row["guard_decision"] for row in previews)
    return {
        "schema_version": "r060_routing_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r060_routing_complete" if gate["gate_passed"] else "r060_routing_needs_fix",
        "scope": {
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_prove_artifact_positive_lift": True,
            "routing_audit_only": True,
        },
        "inputs": {
            "records": args.records,
            "r040_root": args.r040_root,
            "r039_record_ids": args.r039_record_ids,
            "artifacts": args.artifacts,
            "target_record_ids": args.target_record_ids,
        },
        "num_cases": len(previews),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "page_routed_records": gate["page_routed_records"],
        "per_record_summary": build_per_record_summary(previews),
        "gate": gate,
        "recommended_next": [
            "Do not run full QA from R060.",
            "If manually accepted, the next bounded step can be a tiny provider diagnostic on page-routed prompts only.",
            "Keep claims limited: R060 validates prompt routing behavior, not artifact-aware retrieval lift.",
        ],
    }


def build_per_record_summary(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "record_id": row["record_id"],
        "question": row["question"],
        "guard_decision": row["guard_decision"],
        "answer_policy": row["answer_policy"],
        "selected_artifact_count": row["selected_artifact_count"],
        "visible_page_support_sufficient": row["support_audit"]["visible_support_sufficient"],
        "artifact_support_sufficient": row["support_audit"]["artifact_support_sufficient"],
        "routing_passed": row["routing_checks"]["passed"],
        "routing_failures": row["routing_checks"]["failures"],
    } for row in previews]


def build_compact_index(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "r060_routing_compact_index_v1",
        "record_id": row["record_id"],
        "guard_decision": row["guard_decision"],
        "answer_policy": row["answer_policy"],
        "selected_artifact_count": row["selected_artifact_count"],
        "visible_page_support_sufficient": row["support_audit"]["visible_support_sufficient"],
        "routing_passed": row["routing_checks"]["passed"],
        "routing_failures": row["routing_checks"]["failures"],
        "prompt_preview_sha256": row["prompt_preview_sha256"],
    } for row in previews]


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R060 Page/Artifact Routing Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Page/artifact prompt-routing audit only.",
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
    lines.extend(["", "## Summary", f"- page-routed records: `{gate['page_routed_records']}`"])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R060 Page/Artifact Routing Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Audits prompts where page evidence is sufficient but artifact evidence is rejected.",
        "- Not an official score and not evidence of artifact positive lift.",
        "",
        "## Summary",
        f"- cases: {report['num_cases']}",
        f"- guard decisions: `{json.dumps(report['guard_decision_counts'], sort_keys=True)}`",
        f"- page-routed records: `{report['page_routed_records']}`",
        "",
        "## Per-Record Routing",
    ]
    for row in report["per_record_summary"]:
        lines.extend([
            f"### Record {row['record_id']}",
            f"- guard decision: `{row['guard_decision']}`",
            f"- answer policy: `{row['answer_policy']}`",
            f"- selected artifacts: {row['selected_artifact_count']}",
            f"- visible page support sufficient: {row['visible_page_support_sufficient']}",
            f"- artifact support sufficient: {row['artifact_support_sufficient']}",
            f"- routing passed: {row['routing_passed']}",
            f"- routing failures: `{row['routing_failures']}`",
        ])
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


def parse_record_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
