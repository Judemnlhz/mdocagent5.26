#!/usr/bin/env python3
"""R058 no-provider positive-evidence diagnostic for guarded selection.

R058 audits the positive-signal records preserved by R056. It does not call a
provider, run prediction, run evaluation, run full QA, or report a score. The
goal is stricter than "selected artifacts are non-empty": selected artifacts
must provide visible, citable evidence that can support the public question.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from mdocnexus.integration.guarded_prompt import (
    build_question_profile,
    forbidden_public_fields,
    question_tokens,
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
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r058_positive_evidence_diagnostic"
DEFAULT_POSITIVE_RECORD_IDS = "69,223,224,227"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--positive-record-ids", default=DEFAULT_POSITIVE_RECORD_IDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=1600)
    parser.add_argument("--max-artifacts", type=int, default=8)
    parser.add_argument("--max-artifact-chars", type=int, default=360)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    record_ids = parse_record_ids(args.positive_record_ids)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "positive_record_ids": record_ids,
            "no_provider_calls": True,
            "no_prediction_or_eval": True,
            "no_full_qa": True,
            "audit_focus": "visible citable artifact evidence sufficiency, not score",
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

    r053.write_jsonl(output_root / "r058_positive_evidence_prompt_previews.jsonl", previews)
    r053.write_jsonl(output_root / "r058_positive_evidence_compact_index.jsonl", build_compact_index(previews))
    r053.write_json(output_root / "r058_positive_evidence_gate.json", gate)
    write_gate_markdown(output_root / "r058_positive_evidence_gate.md", gate)
    r053.write_json(output_root / "r058_positive_evidence_report.json", report)
    write_report_markdown(output_root / "r058_positive_evidence_report.md", report)

    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_cases": len(previews),
        "artifact_support_sufficient_records": gate["artifact_support_sufficient_records"],
        "artifact_support_insufficient_records": gate["artifact_support_insufficient_records"],
        "report_md": str(output_root / "r058_positive_evidence_report.md"),
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
            raise ValueError(f"positive record_id is not in R039 frozen subset: {record_id}")
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
        prompt = render_guarded_prompt(question, page_contexts, selection, profile, condition_label="R058 condition: positive_evidence_support_audit")
        support = audit_visible_support(question, selection.get("selected_artifacts") or [], page_contexts)
        public_payload = {
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "question_profile": profile,
            "retrieval_pages": {"artifact": artifact_pages, "original": original_pages, "candidate_union": candidate_pages},
            "selection": selection,
            "support_audit": support,
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
        }
        rows.append({
            "schema_version": "r058_positive_evidence_prompt_preview_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "question_profile": profile,
            "retrieval_pages": {
                "top4_artifact_only_combined": artifact_pages,
                "top4_original_only_combined": original_pages,
                "candidate_union": candidate_pages,
            },
            "selection_policy": {
                "name": "guarded_positive_evidence_support_audit_v1",
                "module": "mdocnexus.integration.guarded_prompt",
                "uses_gold_fields": False,
                "requires_non_empty_artifacts": True,
                "requires_citable_artifact_ids": True,
                "requires_question_dimension_coverage": True,
                "requires_numeric_value_coverage_when_numeric": True,
                "not_provider_run": True,
            },
            "candidate_artifact_count": len(candidates),
            "positive_candidate_count": selection["positive_candidate_count"],
            "selected_artifact_count": len(selection["selected_artifacts"]),
            "selected_artifacts": selection["selected_artifacts"],
            "guard_decision": selection["guard_decision"],
            "guard_reasons": selection["guard_reasons"],
            "answer_policy": selection["answer_policy"],
            "support_audit": support,
            "page_contexts": page_contexts,
            "prompt_preview": prompt,
            "prompt_preview_sha256": sha256(prompt),
            "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        })
    return rows


def audit_visible_support(question: str, selected_artifacts: list[dict[str, Any]], page_contexts: list[dict[str, Any]]) -> dict[str, Any]:
    requirements = evidence_requirements(question)
    artifact_text = normalize(" ".join(artifact_evidence_text(item) for item in selected_artifacts))
    page_text = normalize(" ".join(str(ctx.get("text_preview") or "") for ctx in page_contexts))
    all_visible_text = normalize(f"{artifact_text} {page_text}")
    artifact_dimension_checks = dimension_checks(requirements["dimensions"], artifact_text)
    page_dimension_checks = dimension_checks(requirements["dimensions"], page_text)
    visible_dimension_checks = dimension_checks(requirements["dimensions"], all_visible_text)
    artifact_values = extract_numeric_values(artifact_text)
    page_values = extract_numeric_values(page_text)
    citable_artifacts = [
        item
        for item in selected_artifacts
        if str(item.get("artifact_id") or "").strip()
        and item.get("page_index") is not None
        and str(item.get("content_preview") or "").strip()
    ]
    artifact_support_sufficient = (
        bool(selected_artifacts)
        and len(citable_artifacts) == len(selected_artifacts)
        and all(check["covered"] for check in artifact_dimension_checks)
        and len(artifact_values) >= requirements["min_numeric_values"]
    )
    visible_support_sufficient = (
        all(check["covered"] for check in visible_dimension_checks)
        and len(sorted(set(artifact_values + page_values))) >= requirements["min_numeric_values"]
    )
    failure_reasons = []
    if not selected_artifacts:
        failure_reasons.append("no_selected_artifacts")
    if len(citable_artifacts) != len(selected_artifacts):
        failure_reasons.append("selected_artifacts_not_all_citable")
    missing_artifact_dimensions = [check["dimension"] for check in artifact_dimension_checks if not check["covered"]]
    if missing_artifact_dimensions:
        failure_reasons.append("artifact_missing_dimensions:" + ",".join(missing_artifact_dimensions))
    if len(artifact_values) < requirements["min_numeric_values"]:
        failure_reasons.append(f"artifact_numeric_values_below_required:{len(artifact_values)}<{requirements['min_numeric_values']}")
    if not visible_support_sufficient:
        missing_visible_dimensions = [check["dimension"] for check in visible_dimension_checks if not check["covered"]]
        if missing_visible_dimensions:
            failure_reasons.append("visible_context_missing_dimensions:" + ",".join(missing_visible_dimensions))
    if not failure_reasons:
        failure_reasons.append("none")
    return {
        "schema_version": "r058_visible_support_audit_v1",
        "requirements": requirements,
        "artifact_dimension_checks": artifact_dimension_checks,
        "page_dimension_checks": page_dimension_checks,
        "visible_dimension_checks": visible_dimension_checks,
        "artifact_numeric_values": artifact_values,
        "page_numeric_values": page_values[:20],
        "citable_artifact_count": len(citable_artifacts),
        "artifact_support_sufficient": artifact_support_sufficient,
        "visible_support_sufficient": visible_support_sufficient,
        "support_class": "supporting_artifact_evidence_confirmed" if artifact_support_sufficient else "artifact_positive_signal_only_insufficient",
        "failure_reasons": failure_reasons,
        "audit_note": "Public-input diagnostic only; this is not a provider answer, score, or gold-answer comparison.",
    }


def evidence_requirements(question: str) -> dict[str, Any]:
    q_norm = normalize(question)
    dimensions = []
    min_numeric_values = 0
    if "figure 4" in q_norm and "raptor" in q_norm:
        dimensions.extend([
            requirement("figure_4", "figure 4", ["figure 4", "fig. 4", "fig 4"]),
            requirement("raptor", "RAPTOR", ["raptor"]),
            requirement("retrieved_nodes", "retrieved nodes", ["node", "nodes", "retrieved"]),
            requirement("both_questions", "both questions", ["both questions", "both"]),
        ])
    if "higher-income" in q_norm or "higher income" in q_norm:
        dimensions.extend([
            requirement("higher_income_seniors", "Higher-income seniors", ["higher-income seniors", "higher income seniors", "higher-income", "higher income"]),
            requirement("go_online", "go online", ["go online", "online"]),
            requirement("smartphone", "smartphone", ["smartphone"]),
            requirement("tablet_computer", "tablet computer", ["tablet computer", "tablet"]),
        ])
        min_numeric_values = max(min_numeric_values, 3)
    if "college graduate" in q_norm:
        dimensions.extend([
            requirement("age_65_plus", "65+ people", ["65+", "65 +", "65 and older", "65 or older"]),
            requirement("college_graduate", "College graduate", ["college graduate", "college"]),
            requirement("cell_phone", "cell phone", ["cell phone", "cellphone"]),
            requirement("tablet_computer", "tablet computer", ["tablet computer", "tablet"]),
            requirement("gap_operation", "gap", ["gap", "difference"]),
        ])
        min_numeric_values = max(min_numeric_values, 2)
    for year in re.findall(r"\b(?:19|20)\d{2}\b", question):
        dimensions.append(requirement(f"year_{year}", year, [year]))
    if not dimensions:
        keyword_terms = sorted(question_tokens(question))[:8]
        dimensions = [requirement(f"term_{term}", term, [term]) for term in keyword_terms]
    deduped = []
    seen = set()
    for item in dimensions:
        if item["dimension"] not in seen:
            seen.add(item["dimension"])
            deduped.append(item)
    return {
        "dimensions": deduped,
        "min_numeric_values": min_numeric_values,
        "strictly_public_question_only": True,
        "requires_artifact_ids_for_citation": True,
    }


def requirement(dimension: str, label: str, aliases: list[str]) -> dict[str, Any]:
    return {"dimension": dimension, "label": label, "aliases": aliases}


def dimension_checks(requirements: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    rows = []
    for req in requirements:
        aliases = list(req.get("aliases") or [])
        matched = [alias for alias in aliases if phrase_present(text, alias)]
        rows.append({
            "dimension": req["dimension"],
            "label": req["label"],
            "covered": bool(matched),
            "matched_aliases": matched,
        })
    return rows


def phrase_present(text: str, phrase: str) -> bool:
    text_norm = normalize(text).replace("-", " ")
    phrase_norm = normalize(phrase).replace("-", " ")
    if phrase_norm in text_norm:
        return True
    tokens = phrase_norm.split()
    if len(tokens) > 1:
        return all(token in text_norm for token in tokens)
    return False


def artifact_evidence_text(item: Mapping[str, Any]) -> str:
    normalized_content = item.get("normalized_content") if isinstance(item.get("normalized_content"), Mapping) else {}
    return " ".join([
        str(item.get("artifact_id") or ""),
        str(item.get("artifact_type") or ""),
        str(item.get("content_preview") or ""),
        json.dumps(dict(normalized_content), ensure_ascii=False, sort_keys=True),
    ])


def extract_numeric_values(text: str) -> list[str]:
    return sorted(set(re.findall(r"[-+]?\d+(?:\.\d+)?\s*%?", text)))


def build_gate(args: argparse.Namespace, record_ids: list[int], previews: list[dict[str, Any]]) -> dict[str, Any]:
    support_records = [row["record_id"] for row in previews if row["support_audit"]["artifact_support_sufficient"]]
    insufficient_records = [row["record_id"] for row in previews if not row["support_audit"]["artifact_support_sufficient"]]
    visible_support_records = [row["record_id"] for row in previews if row["support_audit"]["visible_support_sufficient"]]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_records_match_positive_signal_cases": sorted(record_ids) == sorted(row["record_id"] for row in previews),
        "all_cases_have_selected_artifacts": all(row["selected_artifact_count"] > 0 for row in previews),
        "all_selected_artifacts_are_citable": all(
            row["support_audit"]["citable_artifact_count"] == row["selected_artifact_count"] for row in previews
        ),
        "all_prompts_have_citation_requirement": all("Cite page ids and artifact ids" in row["prompt_preview"] for row in previews),
        "page_and_artifact_evidence_separated": all("[Page evidence]" in row["prompt_preview"] and "[Selected artifact evidence]" in row["prompt_preview"] for row in previews),
        "no_gold_fields_in_public_previews": all(not row["forbidden_gold_fields_present"] for row in previews),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "positive_signal_is_not_treated_as_support": True,
        "all_positive_cases_have_supporting_artifact_evidence": not insufficient_records,
        "not_artifact_lift_claim": True,
        "not_official_score": True,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r058_positive_evidence_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r058_positive_evidence_gate_pass" if not hard_failures else "r058_positive_evidence_needs_selector_fix",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_cases": len(previews),
        "artifact_support_sufficient_records": support_records,
        "artifact_support_insufficient_records": insufficient_records,
        "visible_support_sufficient_records": visible_support_records,
        "selected_artifact_count_by_record": {str(row["record_id"]): row["selected_artifact_count"] for row in previews},
        "support_class_by_record": {str(row["record_id"]): row["support_audit"]["support_class"] for row in previews},
        "failure_reasons_by_record": {str(row["record_id"]): row["support_audit"]["failure_reasons"] for row in previews},
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, previews: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    support_counts = Counter(row["support_audit"]["support_class"] for row in previews)
    guard_counts = Counter(row["guard_decision"] for row in previews)
    return {
        "schema_version": "r058_positive_evidence_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r058_positive_evidence_confirmed" if gate["gate_passed"] else "r058_positive_evidence_needs_selector_fix",
        "scope": {
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_prove_artifact_positive_lift": True,
            "positive_evidence_diagnostic_only": True,
        },
        "inputs": {
            "records": args.records,
            "r040_root": args.r040_root,
            "r039_record_ids": args.r039_record_ids,
            "artifacts": args.artifacts,
            "positive_record_ids": args.positive_record_ids,
        },
        "num_cases": len(previews),
        "guard_decision_counts": dict(sorted(guard_counts.items())),
        "support_class_counts": dict(sorted(support_counts.items())),
        "artifact_support_sufficient_records": gate["artifact_support_sufficient_records"],
        "artifact_support_insufficient_records": gate["artifact_support_insufficient_records"],
        "per_record_summary": build_per_record_summary(previews),
        "gate": gate,
        "recommended_next": recommended_next(gate),
    }


def build_per_record_summary(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in previews:
        audit = row["support_audit"]
        rows.append({
            "record_id": row["record_id"],
            "question": row["question"],
            "guard_decision": row["guard_decision"],
            "selected_artifact_count": row["selected_artifact_count"],
            "artifact_support_sufficient": audit["artifact_support_sufficient"],
            "visible_support_sufficient": audit["visible_support_sufficient"],
            "support_class": audit["support_class"],
            "failure_reasons": audit["failure_reasons"],
            "selected_artifact_ids": [artifact["artifact_id"] for artifact in row["selected_artifacts"]],
            "missing_artifact_dimensions": [
                check["dimension"] for check in audit["artifact_dimension_checks"] if not check["covered"]
            ],
        })
    return rows


def recommended_next(gate: dict[str, Any]) -> list[str]:
    if gate["gate_passed"]:
        return [
            "Keep R058 scoped as no-provider diagnostic evidence only.",
            "Optionally run a tiny provider diagnostic on these accepted positive-evidence prompts after manual review.",
            "Do not claim artifact lift without contrastive provider evidence.",
        ]
    return [
        "Do not run provider QA on R058 positives yet.",
        "Repair guarded selector ranking so positive-signal artifacts must cover question dimensions, not just token overlap.",
        "Add table/key-value dimension matching for demographic, time, metric, and operand constraints.",
        "Rerun R058 before wiring guarded selector into full QA.",
    ]


def build_compact_index(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "schema_version": "r058_positive_evidence_compact_index_v1",
        "record_id": row["record_id"],
        "doc_id": row["doc_id"],
        "guard_decision": row["guard_decision"],
        "selected_artifact_count": row["selected_artifact_count"],
        "selected_artifact_ids": [artifact["artifact_id"] for artifact in row["selected_artifacts"]],
        "selected_artifact_pages": sorted({artifact["page_index"] for artifact in row["selected_artifacts"]}),
        "artifact_support_sufficient": row["support_audit"]["artifact_support_sufficient"],
        "visible_support_sufficient": row["support_audit"]["visible_support_sufficient"],
        "support_class": row["support_audit"]["support_class"],
        "failure_reasons": row["support_audit"]["failure_reasons"],
        "prompt_preview_sha256": row["prompt_preview_sha256"],
    } for row in previews]


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R058 Positive Evidence Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Positive-evidence support diagnostic only.",
        "- Checks whether guarded selector keeps visible, citable, answer-supporting artifact evidence.",
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
    lines.extend([
        "",
        "## Support Summary",
        f"- artifact support sufficient records: `{gate['artifact_support_sufficient_records']}`",
        f"- artifact support insufficient records: `{gate['artifact_support_insufficient_records']}`",
    ])
    r053.write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R058 Positive-Evidence Diagnostic",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Audits R056 positive-signal records for visible, citable, answer-supporting artifact evidence.",
        "- Positive signal is not treated as support; selected artifacts must cover public question dimensions.",
        "- Not an official score and not evidence of artifact positive lift.",
        "",
        "## Summary",
        f"- cases: {report['num_cases']}",
        f"- guard decisions: `{json.dumps(report['guard_decision_counts'], sort_keys=True)}`",
        f"- support classes: `{json.dumps(report['support_class_counts'], sort_keys=True)}`",
        f"- artifact support sufficient records: `{report['artifact_support_sufficient_records']}`",
        f"- artifact support insufficient records: `{report['artifact_support_insufficient_records']}`",
        "",
        "## Per-Record Audit",
    ]
    for row in report["per_record_summary"]:
        lines.extend([
            f"### Record {row['record_id']}",
            f"- guard decision: `{row['guard_decision']}`",
            f"- selected artifacts: {row['selected_artifact_count']} `{row['selected_artifact_ids']}`",
            f"- artifact support sufficient: {row['artifact_support_sufficient']}",
            f"- visible support sufficient: {row['visible_support_sufficient']}",
            f"- missing artifact dimensions: `{row['missing_artifact_dimensions']}`",
            f"- failure reasons: `{row['failure_reasons']}`",
        ])
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


def parse_record_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


if __name__ == "__main__":
    main()
