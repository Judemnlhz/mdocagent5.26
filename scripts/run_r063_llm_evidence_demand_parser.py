#!/usr/bin/env python3
"""R063 LLM evidence-demand parser diagnostic.

R063 uses Qwen3-VL-8B-Instruct only as a question-only evidence-demand parser.
It does not ask the model to answer, select artifacts, evaluate predictions, run
full QA, or report an official score. The deterministic guarded selector remains
responsible for scoring and selecting public artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import run_r053_question_aware_scaffold as r053

from mdocnexus.integration.evidence_demand_parser import (
    build_evidence_demand_prompt,
    evidence_demand_contract,
    merge_evidence_demand_profile,
    parse_evidence_demand_response,
    validate_public_parser_payload,
)
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
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r063_llm_evidence_demand_parser"
DEFAULT_TARGET_RECORD_IDS = "384,508,569,69,223,224,227"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--target-record-ids", default=DEFAULT_TARGET_RECORD_IDS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-page-chars", type=int, default=1600)
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
            "provider_calls_planned": len(record_ids),
            "model": args.model,
            "parser_role_only": True,
            "no_answer_generation": True,
            "no_prediction_or_eval": True,
            "no_full_qa": True,
            "not_official_score": True,
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    records = r053.read_json(Path(args.records))
    frozen_record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(frozen_record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_page = r053.load_artifacts_by_page(Path(args.artifacts))

    parser_outputs_path = output_root / "provider" / "r063_evidence_demand_parser_outputs.jsonl"
    parser_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_parser_outputs(parser_outputs_path)
    parser_outputs = run_or_load_parser_outputs(args, record_ids, records, existing, parser_outputs_path)
    previews = build_selector_comparisons(args, record_ids, records, offsets, run_records, artifacts_by_page, parser_outputs)
    gate = build_gate(args, record_ids, parser_outputs, previews)
    report = build_report(args, parser_outputs, previews, gate)

    r053.write_jsonl(parser_outputs_path, parser_outputs)
    r053.write_jsonl(output_root / "r063_selector_comparisons.jsonl", previews)
    r053.write_json(output_root / "r063_llm_evidence_demand_gate.json", gate)
    write_gate_markdown(output_root / "r063_llm_evidence_demand_gate.md", gate)
    r053.write_json(output_root / "r063_llm_evidence_demand_report.json", report)
    write_report_markdown(output_root / "r063_llm_evidence_demand_report.md", report)
    r053.write_json(output_root / "r063_evidence_demand_contract.json", evidence_demand_contract())

    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_records": len(previews),
        "provider_calls_or_cached_outputs": len(parser_outputs),
        "parser_outputs_parseable": gate["checks"]["all_parser_outputs_parseable"],
        "no_full_qa": True,
        "not_official_score": True,
        "report_md": str(output_root / "r063_llm_evidence_demand_report.md"),
    }, ensure_ascii=False, indent=2))


def run_or_load_parser_outputs(
    args: argparse.Namespace,
    record_ids: list[int],
    records: list[dict[str, Any]],
    existing: dict[tuple[int, str], dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)
    outputs: list[dict[str, Any]] = []
    for record_id in record_ids:
        question = str(records[record_id]["question"])
        prompt = build_evidence_demand_prompt(question)
        prompt_hash = sha256(prompt)
        key = (record_id, prompt_hash)
        if key in existing:
            outputs.append(existing[key])
            continue
        response_text = call_provider(client, args, prompt)
        parsed = None
        parse_error = ""
        try:
            parsed = parse_evidence_demand_response(response_text)
        except Exception as exc:  # diagnostic row records provider/schema failure.
            parse_error = str(exc)
        row = {
            "schema_version": "r063_evidence_demand_parser_output_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "record_id": record_id,
            "doc_id": records[record_id]["doc_id"],
            "question": question,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "parser_prompt_sha256": prompt_hash,
            "parser_prompt_chars": len(prompt),
            "parser_role_only": True,
            "question_only_input": True,
            "raw_response_text": response_text,
            "parsed_evidence_demand": parsed,
            "parse_error": parse_error,
            "forbidden_gold_fields_present": validate_public_parser_payload({
                "record_id": record_id,
                "doc_id": records[record_id]["doc_id"],
                "question": question,
                "parsed_evidence_demand": parsed,
                "parser_prompt_sha256": prompt_hash,
            }),
            "not_answer_generation": True,
            "not_prediction": True,
            "not_evaluation": True,
            "not_full_qa": True,
            "not_official_score": True,
            "not_artifact_lift_claim": True,
        }
        outputs.append(row)
        r053.write_jsonl(output_path, outputs)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    return outputs


def build_selector_comparisons(
    args: argparse.Namespace,
    record_ids: list[int],
    records: list[dict[str, Any]],
    offsets: dict[int, int],
    run_records: dict[str, list[dict[str, Any]]],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    parser_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_by_record = {int(row["record_id"]): row for row in parser_outputs}
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
        page_contexts = [r053.load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        rule_profile = build_question_profile(question)
        parser_row = output_by_record[record_id]
        parsed = parser_row.get("parsed_evidence_demand")
        if parsed:
            llm_profile = merge_evidence_demand_profile(question, parsed)
        else:
            llm_profile = dict(rule_profile)
            llm_profile["profile_source"] = "rule_profile_fallback_parser_unparseable"
        rule = run_selector_variant(args, question, doc_id, candidate_pages, artifact_pages, original_pages, page_contexts, artifacts_by_page, rule_profile, "rule_only_profile")
        llm = run_selector_variant(args, question, doc_id, candidate_pages, artifact_pages, original_pages, page_contexts, artifacts_by_page, llm_profile, "llm_evidence_demand_profile")
        comparison = compare_variants(rule, llm, parsed)
        public_payload = {
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "parser_output": parser_row,
            "rule_variant": rule,
            "llm_variant": llm,
            "comparison": comparison,
        }
        rows.append({
            "schema_version": "r063_selector_comparison_v1",
            "record_id": record_id,
            "doc_id": doc_id,
            "question": question,
            "retrieval_pages": {
                "top4_artifact_only_combined": artifact_pages,
                "top4_original_only_combined": original_pages,
                "candidate_union": candidate_pages,
            },
            "parser_output_ref": {
                "model": parser_row["model"],
                "parser_prompt_sha256": parser_row["parser_prompt_sha256"],
                "parse_error": parser_row.get("parse_error", ""),
            },
            "parsed_evidence_demand": parsed,
            "rule_only": rule,
            "llm_evidence_demand": llm,
            "comparison": comparison,
            "forbidden_gold_fields_present": forbidden_public_fields(public_payload),
        })
    return rows


def run_selector_variant(
    args: argparse.Namespace,
    question: str,
    doc_id: str,
    candidate_pages: list[int],
    artifact_pages: list[int],
    original_pages: list[int],
    page_contexts: list[dict[str, Any]],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    profile: dict[str, Any],
    label: str,
) -> dict[str, Any]:
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
    prompt = render_guarded_prompt(question, page_contexts, selection, profile, condition_label=f"R063 condition: {label}")
    return {
        "variant": label,
        "profile_source": profile.get("profile_source", "rule_profile_only"),
        "question_profile": profile,
        "candidate_artifact_count": len(candidates),
        "positive_candidate_count": selection["positive_candidate_count"],
        "selected_artifact_count": len(selection["selected_artifacts"]),
        "selected_artifact_ids": [row.get("artifact_id") for row in selection.get("selected_artifacts") or []],
        "guard_decision": selection["guard_decision"],
        "guard_reasons": selection["guard_reasons"],
        "answer_policy": selection["answer_policy"],
        "support_audit": support,
        "prompt_preview_sha256": sha256(prompt),
        "prompt_preview": prompt,
    }


def compare_variants(rule: dict[str, Any], llm: dict[str, Any], parsed: dict[str, Any] | None) -> dict[str, Any]:
    rule_dims = dimensions(rule)
    llm_dims = dimensions(llm)
    selected_delta = int(llm["selected_artifact_count"]) - int(rule["selected_artifact_count"])
    return {
        "schema_version": "r063_selector_variant_comparison_v1",
        "parser_parseable": parsed is not None,
        "answer_type": (parsed or {}).get("answer_type"),
        "rule_guard_decision": rule["guard_decision"],
        "llm_guard_decision": llm["guard_decision"],
        "guard_decision_changed": rule["guard_decision"] != llm["guard_decision"],
        "rule_selected_artifact_count": rule["selected_artifact_count"],
        "llm_selected_artifact_count": llm["selected_artifact_count"],
        "selected_artifact_count_delta": selected_delta,
        "rule_dimension_count": len(rule_dims),
        "llm_dimension_count": len(llm_dims),
        "llm_added_dimensions": sorted(set(llm_dims) - set(rule_dims)),
        "llm_removed_dimensions": sorted(set(rule_dims) - set(llm_dims)),
        "llm_artifact_support_sufficient": bool(llm["support_audit"].get("artifact_support_sufficient")),
        "llm_visible_support_sufficient": bool(llm["support_audit"].get("visible_support_sufficient")),
        "interpretation": interpretation(rule, llm, parsed),
    }


def interpretation(rule: dict[str, Any], llm: dict[str, Any], parsed: dict[str, Any] | None) -> str:
    if parsed is None:
        return "parser_unparseable_no_selector_claim"
    if llm["guard_decision"] == "artifact_dimension_support_guard" and int(llm["selected_artifact_count"]) == 0:
        return "llm_requirements_tighten_or_confirm_artifact_rejection"
    if llm["support_audit"].get("artifact_support_sufficient"):
        return "llm_requirements_retained_citable_supporting_artifact_candidate"
    if int(llm["selected_artifact_count"]) > int(rule["selected_artifact_count"]):
        return "llm_requirements_expanded_selection_needs_manual_support_review"
    return "llm_requirements_no_positive_artifact_lift_claim"


def dimensions(row: dict[str, Any]) -> list[str]:
    req = row.get("question_profile", {}).get("evidence_requirements") or {}
    dims = req.get("dimensions") if isinstance(req, dict) else []
    return [str(item.get("dimension")) for item in dims if isinstance(item, dict) and item.get("dimension")]


def build_gate(args: argparse.Namespace, record_ids: list[int], parser_outputs: list[dict[str, Any]], previews: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in parser_outputs if row.get("parsed_evidence_demand")]
    no_gold_parser = all(not row.get("forbidden_gold_fields_present") for row in parser_outputs)
    no_gold_previews = all(not row.get("forbidden_gold_fields_present") for row in previews)
    llm_selected_positive = [row["record_id"] for row in previews if row["llm_evidence_demand"]["selected_artifact_count"] > 0]
    llm_supporting = [row["record_id"] for row in previews if row["llm_evidence_demand"]["support_audit"].get("artifact_support_sufficient")]
    checks = {
        "target_records_match_requested_small_set": sorted(record_ids) == sorted(row["record_id"] for row in previews),
        "provider_outputs_exactly_target_count": len(parser_outputs) == len(record_ids),
        "all_parser_outputs_parseable": len(parsed) == len(parser_outputs),
        "parser_inputs_question_only": all(row.get("question_only_input") is True and row.get("parser_role_only") is True for row in parser_outputs),
        "parser_outputs_have_no_gold_fields": no_gold_parser,
        "selector_previews_have_no_gold_fields": no_gold_previews,
        "deterministic_selector_still_used": all(row["llm_evidence_demand"]["variant"] == "llm_evidence_demand_profile" for row in previews),
        "llm_does_not_select_artifacts_directly": True,
        "scope_limited_to_evidence_demand_parser_diagnostic": True,
        "no_answer_generation": True,
        "no_prediction_or_eval": True,
        "no_full_qa": True,
        "not_official_score": True,
        "does_not_claim_artifact_lift": True,
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r063_llm_evidence_demand_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r063_llm_evidence_demand_parser_gate_pass" if not hard_failures else "r063_llm_evidence_demand_parser_needs_fix",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "target_record_ids": record_ids,
        "model": args.model,
        "llm_selected_positive_records": llm_selected_positive,
        "llm_artifact_supporting_records": llm_supporting,
        "guard_decision_counts": dict(Counter(row["llm_evidence_demand"]["guard_decision"] for row in previews)),
        "interpretation_counts": dict(Counter(row["comparison"]["interpretation"] for row in previews)),
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, parser_outputs: list[dict[str, Any]], previews: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    per_record = []
    for row in previews:
        parsed = row.get("parsed_evidence_demand") or {}
        per_record.append({
            "record_id": row["record_id"],
            "question": row["question"],
            "answer_type": parsed.get("answer_type"),
            "rule_guard": row["rule_only"]["guard_decision"],
            "llm_guard": row["llm_evidence_demand"]["guard_decision"],
            "rule_selected_artifact_count": row["rule_only"]["selected_artifact_count"],
            "llm_selected_artifact_count": row["llm_evidence_demand"]["selected_artifact_count"],
            "llm_artifact_support_sufficient": row["llm_evidence_demand"]["support_audit"].get("artifact_support_sufficient"),
            "llm_added_dimensions": row["comparison"].get("llm_added_dimensions"),
            "interpretation": row["comparison"].get("interpretation"),
        })
    return {
        "schema_version": "r063_llm_evidence_demand_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r063_llm_evidence_demand_parser_complete" if gate["gate_passed"] else "r063_llm_evidence_demand_parser_needs_fix",
        "scope": {
            "target_records_only": gate["target_record_ids"],
            "provider_calls": len(parser_outputs),
            "provider_role": "question-only evidence-demand parser",
            "selector_role": "deterministic guarded selector",
            "no_answer_generation": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_mmlongbench_result": True,
            "does_not_prove_artifact_positive_lift": True,
            "does_not_prove_retrieval_improvement": True,
        },
        "inputs": {
            "records": args.records,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
            "parser_contract": evidence_demand_contract(),
        },
        "model": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "gate": gate,
        "summary": {
            "parseable_outputs": sum(1 for row in parser_outputs if row.get("parsed_evidence_demand")),
            "num_records": len(previews),
            "guard_decision_counts": gate["guard_decision_counts"],
            "interpretation_counts": gate["interpretation_counts"],
            "llm_selected_positive_records": gate["llm_selected_positive_records"],
            "llm_artifact_supporting_records": gate["llm_artifact_supporting_records"],
        },
        "per_record": per_record,
        "recommended_next": [
            "Manually inspect R063 selector comparisons before changing the adapter path.",
            "If the parser adds useful dimensions but still selects no supporting artifacts, improve artifact store/coverage rather than running full QA.",
            "If the parser preserves or improves supporting artifact retention on positive cases, add a default-off integration gate for parser-assisted profiles before any provider QA run.",
        ],
    }


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R063 LLM Evidence-Demand Parser Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- Qwen3-VL-8B-Instruct is used only as a question-only evidence-demand parser.",
        "- The LLM does not answer questions and does not select artifacts directly.",
        "- Deterministic guarded selector still performs artifact scoring and selection.",
        "- No prediction, no evaluation, no full QA, no official score, and no artifact-lift claim.",
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
        "# R063 LLM Evidence-Demand Parser Diagnostic",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- Uses Qwen3-VL-8B-Instruct only to parse question evidence requirements.",
        "- Does not ask the model to answer, select artifacts, evaluate, or run QA.",
        "- Does not prove artifact positive lift, retrieval improvement, or official MMLongBench performance.",
        "",
        "## Summary",
        f"- model: `{report['model']['model']}`",
        f"- records: {report['summary']['num_records']}",
        f"- parseable outputs: {report['summary']['parseable_outputs']}",
        f"- guard decision counts: `{json.dumps(report['summary']['guard_decision_counts'], sort_keys=True)}`",
        f"- interpretation counts: `{json.dumps(report['summary']['interpretation_counts'], sort_keys=True)}`",
        f"- LLM-selected positive records: `{report['summary']['llm_selected_positive_records']}`",
        f"- LLM artifact-supporting records: `{report['summary']['llm_artifact_supporting_records']}`",
        "",
        "## Per Record",
    ]
    for row in report["per_record"]:
        lines.append(
            f"- {row['record_id']}: answer_type=`{row['answer_type']}`, rule_guard=`{row['rule_guard']}`, "
            f"llm_guard=`{row['llm_guard']}`, rule_selected={row['rule_selected_artifact_count']}, "
            f"llm_selected={row['llm_selected_artifact_count']}, artifact_support={row['llm_artifact_support_sufficient']}, "
            f"interpretation=`{row['interpretation']}`"
        )
    lines.extend(["", "## Recommended Next"])
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


def call_provider(client: Any, args: argparse.Namespace, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(max(int(args.max_retries), 1)):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(int(args.max_retries), 1):
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Provider call failed after {args.max_retries} attempts: {last_error}")


def parse_record_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def load_existing_parser_outputs(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    return {(int(row["record_id"]), str(row["parser_prompt_sha256"])): row for row in r053.read_jsonl(path)}


if __name__ == "__main__":
    main()
