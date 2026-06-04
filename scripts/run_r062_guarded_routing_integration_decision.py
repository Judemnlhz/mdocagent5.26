#!/usr/bin/env python3
"""R062 no-provider guarded routing integration decision gate.

R062 decides whether the R059/R060/R061 guarded selector and compact page-routing
scaffold are ready to sit behind the existing disabled-by-default integration
path. It does not call providers, run prediction, run evaluation, run full QA,
or report a score.
"""

from __future__ import annotations

import argparse
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
import run_r057_guarded_integration_gate as r057
import run_r061_page_routed_provider as r061

from mdocnexus.integration.guarded_integration import (
    GuardedPromptIntegrationConfig,
    apply_guarded_prompt_integration,
    guarded_prompt_integration_contract,
    write_integration_outputs,
)
from mdocnexus.integration.guarded_prompt import forbidden_public_fields, sha256
from mdocnexus.integration.mdocagent_adapter import canonical_json_hash

DEFAULT_R045_CASES = r057.DEFAULT_R045_CASES
DEFAULT_R044_REPORT = r057.DEFAULT_R044_REPORT
DEFAULT_R040_ROOT = r057.DEFAULT_R040_ROOT
DEFAULT_R039_RECORD_IDS = r057.DEFAULT_R039_RECORD_IDS
DEFAULT_RECORDS = r057.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r057.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r057.DEFAULT_EXTRACT_PATH
DEFAULT_R060_PREVIEWS = "outputs/heldout/r060_page_artifact_routing_audit/r060_routing_prompt_previews.jsonl"
DEFAULT_R061_REPORT = "outputs/heldout/r061_page_routed_provider_diagnostic/r061_page_routed_provider_report.json"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r062_guarded_routing_integration_decision"
TARGET_RECORD_IDS = [223, 227]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r045-cases", default=DEFAULT_R045_CASES)
    parser.add_argument("--r044-report", default=DEFAULT_R044_REPORT)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--r060-previews", default=DEFAULT_R060_PREVIEWS)
    parser.add_argument("--r061-report", default=DEFAULT_R061_REPORT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-contexts", type=int, default=4)
    parser.add_argument("--max-page-chars", type=int, default=700)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not args.execute:
        print(json.dumps({
            "will_execute": False,
            "output_root": str(output_root),
            "target_record_ids": TARGET_RECORD_IDS,
            "no_provider_calls": True,
            "no_prediction_or_eval": True,
            "no_full_qa": True,
            "integration_decision_gate_only": True,
            "default_enabled": False,
            "config_flag": "enable_guarded_prompt_scaffold",
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    public_records, page_contexts, artifacts_by_page = build_r062_public_inputs(args)
    disabled = apply_guarded_prompt_integration(public_records, artifacts_by_page, page_contexts, GuardedPromptIntegrationConfig())
    enabled = apply_guarded_prompt_integration(
        public_records,
        artifacts_by_page,
        page_contexts,
        GuardedPromptIntegrationConfig(enable_guarded_prompt_scaffold=True),
    )
    integration_paths = write_integration_outputs(enabled, output_root / "integration_outputs")
    r060_previews = load_target_r060_previews(Path(args.r060_previews))
    r061_report = r053.read_json(Path(args.r061_report))
    compact_scaffolds = build_compact_scaffolds(args, r060_previews, r061_report)
    decision = build_decision(args, public_records, disabled, enabled, r060_previews, r061_report, compact_scaffolds, integration_paths)

    r053.write_json(output_root / "r062_guarded_routing_integration_gate.json", decision["gate"])
    write_gate_markdown(output_root / "r062_guarded_routing_integration_gate.md", decision["gate"])
    r053.write_json(output_root / "r062_guarded_routing_integration_report.json", decision["report"])
    write_report_markdown(output_root / "r062_guarded_routing_integration_report.md", decision["report"])
    r053.write_json(output_root / "r062_integration_decision_manifest.json", decision["manifest"])
    r053.write_jsonl(output_root / "r062_compact_routing_scaffolds.jsonl", compact_scaffolds)
    r053.write_jsonl(output_root / "r062_public_records.jsonl", public_records)

    print(json.dumps({
        "decision": decision["gate"]["decision"],
        "gate_passed": decision["gate"]["gate_passed"],
        "num_records": len(public_records),
        "disabled_records_unchanged": decision["gate"]["checks"]["disabled_records_unchanged"],
        "enabled_records_unchanged": decision["gate"]["checks"]["enabled_records_unchanged"],
        "compact_scaffolds": len(compact_scaffolds),
        "report_md": str(output_root / "r062_guarded_routing_integration_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, ensure_ascii=False, indent=2))


def build_r062_public_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, dict[int, list[dict[str, Any]]]]]:
    r045_cases = r053.read_jsonl(Path(args.r045_cases))
    r044_report = r053.read_json(Path(args.r044_report))
    records = r053.read_json(Path(args.records))
    record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_tuple = r053.load_artifacts_by_page(Path(args.artifacts))
    all_records, all_page_contexts, all_artifacts_by_page = r057.build_public_inputs(
        args, r045_cases, r044_report, records, offsets, run_records, artifacts_by_tuple
    )
    selected_records = [row for row in all_records if int(row["record_index"]) in TARGET_RECORD_IDS]
    selected_records = sorted(selected_records, key=lambda row: TARGET_RECORD_IDS.index(int(row["record_index"])))
    found = [int(row["record_index"]) for row in selected_records]
    if found != TARGET_RECORD_IDS:
        raise ValueError(f"Expected R062 public records {TARGET_RECORD_IDS}, found {found}")
    needed_pages: set[tuple[str, int]] = set()
    for row in selected_records:
        doc_id = str(row["doc_id"])
        for key in ["text-top-10-question", "image-top-10-question"]:
            for page in row.get(key, []) or []:
                needed_pages.add((doc_id, int(page)))
    page_contexts = {key: value for key, value in all_page_contexts.items() if key in needed_pages}
    artifacts_by_page: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for doc_id, page_map in all_artifacts_by_page.items():
        for page, artifacts in page_map.items():
            if (doc_id, int(page)) in needed_pages:
                artifacts_by_page.setdefault(doc_id, {})[int(page)] = artifacts
    return selected_records, page_contexts, artifacts_by_page


def load_target_r060_previews(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in r053.read_jsonl(path) if int(row["record_id"]) in TARGET_RECORD_IDS]
    rows = sorted(rows, key=lambda row: TARGET_RECORD_IDS.index(int(row["record_id"])))
    found = [int(row["record_id"]) for row in rows]
    if found != TARGET_RECORD_IDS:
        raise ValueError(f"Expected R060 previews {TARGET_RECORD_IDS}, found {found}")
    return rows


def build_compact_scaffolds(args: argparse.Namespace, r060_previews: list[dict[str, Any]], r061_report: dict[str, Any]) -> list[dict[str, Any]]:
    provider_meta = r061_report.get("provider_prompt") or {}
    scaffold_args = argparse.Namespace(max_page_contexts=args.max_page_contexts, max_page_chars=args.max_page_chars)
    rows = []
    for preview in r060_previews:
        prompt = r061.build_provider_prompt(preview, scaffold_args)
        record_id = int(preview["record_id"])
        expected_sha = (provider_meta.get("sha256_by_record") or {}).get(str(record_id))
        rows.append({
            "schema_version": "r062_compact_routing_scaffold_v1",
            "record_id": record_id,
            "doc_id": preview["doc_id"],
            "question": preview["question"],
            "mode": "r060_derived_compact_page_routing_prompt",
            "source_r060_prompt_preview_sha256": preview["prompt_preview_sha256"],
            "compact_prompt_sha256": sha256(prompt),
            "r061_reported_compact_prompt_sha256": expected_sha,
            "compact_prompt_hash_matches_r061": sha256(prompt) == expected_sha,
            "compact_prompt_chars": len(prompt),
            "r061_reported_prompt_chars": (provider_meta.get("chars_by_record") or {}).get(str(record_id)),
            "max_page_contexts": args.max_page_contexts,
            "max_page_chars": args.max_page_chars,
            "guard_decision": preview["guard_decision"],
            "answer_policy": preview["answer_policy"],
            "selected_artifact_count": preview["selected_artifact_count"],
            "prompt_preview": prompt,
            "forbidden_gold_fields_present": forbidden_public_fields({"record_id": record_id, "prompt_preview": prompt}),
            "boundary": {
                "optional_provider_facing_scaffold": True,
                "not_default_full_prompt": True,
                "no_provider_call_in_r062": True,
                "not_full_qa": True,
                "not_official_score": True,
                "not_artifact_lift_claim": True,
            },
        })
    return rows


def build_decision(
    args: argparse.Namespace,
    public_records: list[dict[str, Any]],
    disabled: dict[str, Any],
    enabled: dict[str, Any],
    r060_previews: list[dict[str, Any]],
    r061_report: dict[str, Any],
    compact_scaffolds: list[dict[str, Any]],
    integration_paths: dict[str, str],
) -> dict[str, Any]:
    gate = build_gate(args, public_records, disabled, enabled, r060_previews, r061_report, compact_scaffolds)
    report = build_report(args, public_records, disabled, enabled, r060_previews, r061_report, compact_scaffolds, gate, integration_paths)
    manifest = {
        "schema_version": "r062_integration_decision_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "R062",
        "skill_context": ["ARIS run-experiment protocol", "experiment integrity", "output manifest discipline"],
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "scope": report["scope"],
        "inputs": report["inputs"],
        "outputs": report["output_paths"],
        "no_provider_calls": True,
        "no_prediction_or_eval": True,
        "no_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }
    return {"gate": gate, "report": report, "manifest": manifest}


def build_gate(
    args: argparse.Namespace,
    public_records: list[dict[str, Any]],
    disabled: dict[str, Any],
    enabled: dict[str, Any],
    r060_previews: list[dict[str, Any]],
    r061_report: dict[str, Any],
    compact_scaffolds: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = guarded_prompt_integration_contract()
    public_hash = canonical_json_hash(public_records)
    enabled_previews = list(enabled.get("prompt_previews") or [])
    r061_scope = r061_report.get("scope") or {}
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_records_exactly_223_227": [int(row["record_index"]) for row in public_records] == TARGET_RECORD_IDS,
        "contract_default_disabled": contract["default_enabled"] is False,
        "contract_has_config_flag": contract["config_flag"] == "enable_guarded_prompt_scaffold",
        "disabled_records_unchanged": disabled["input_records_sha256"] == disabled["output_records_sha256"] == public_hash,
        "disabled_generates_no_prompt_previews": len(disabled.get("prompt_previews") or []) == 0,
        "enabled_records_unchanged": enabled["input_records_sha256"] == enabled["output_records_sha256"] == public_hash,
        "enabled_emits_previews_and_manifest_only": enabled["manifest"]["num_prompt_previews"] == len(enabled_previews) == len(public_records),
        "enabled_previews_have_no_gold_fields": enabled["manifest"]["no_gold_fields_in_public_previews"] is True and all(not row.get("forbidden_gold_fields_present") for row in enabled_previews),
        "enabled_previews_page_routed_zero_artifact": all(
            row.get("guard_decision") == "artifact_dimension_support_guard"
            and int(row.get("selected_artifact_count") or 0) == 0
            and row.get("answer_policy") == "use_page_evidence_or_refuse"
            for row in enabled_previews
        ),
        "r060_previews_page_routed_zero_artifact": all(
            row.get("guard_decision") == "artifact_dimension_support_guard"
            and int(row.get("selected_artifact_count") or 0) == 0
            and row.get("answer_policy") == "use_page_evidence_or_refuse"
            for row in r060_previews
        ),
        "compact_scaffold_hashes_match_r061": all(row["compact_prompt_hash_matches_r061"] for row in compact_scaffolds),
        "compact_scaffold_provenance_from_r060": all(row.get("source_r060_prompt_preview_sha256") for row in compact_scaffolds),
        "compact_scaffold_marked_optional_provider_facing": all(row["boundary"]["optional_provider_facing_scaffold"] and row["boundary"]["not_default_full_prompt"] for row in compact_scaffolds),
        "compact_scaffolds_have_no_gold_fields": all(not row.get("forbidden_gold_fields_present") for row in compact_scaffolds),
        "r061_scope_is_tiny_page_routing_only": bool(r061_scope.get("tiny_provider_diagnostic")) and bool(r061_scope.get("tests_page_only_routing_behavior_only")),
        "does_not_claim_artifact_lift": True,
        "not_official_score": True,
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r062_guarded_routing_integration_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r062_guarded_routing_integration_gate_pass" if not hard_failures else "r062_guarded_routing_integration_needs_fix",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "target_record_ids": TARGET_RECORD_IDS,
        "num_records": len(public_records),
        "enabled_guard_decision_by_record": {str(row.get("record_index")): row.get("guard_decision") for row in enabled_previews},
        "compact_prompt_sha256_by_record": {str(row["record_id"]): row["compact_prompt_sha256"] for row in compact_scaffolds},
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(
    args: argparse.Namespace,
    public_records: list[dict[str, Any]],
    disabled: dict[str, Any],
    enabled: dict[str, Any],
    r060_previews: list[dict[str, Any]],
    r061_report: dict[str, Any],
    compact_scaffolds: list[dict[str, Any]],
    gate: dict[str, Any],
    integration_paths: dict[str, str],
) -> dict[str, Any]:
    output_paths = {
        "gate_json": str(Path(args.output_root) / "r062_guarded_routing_integration_gate.json"),
        "gate_md": str(Path(args.output_root) / "r062_guarded_routing_integration_gate.md"),
        "report_json": str(Path(args.output_root) / "r062_guarded_routing_integration_report.json"),
        "report_md": str(Path(args.output_root) / "r062_guarded_routing_integration_report.md"),
        "manifest_json": str(Path(args.output_root) / "r062_integration_decision_manifest.json"),
        "compact_scaffolds_jsonl": str(Path(args.output_root) / "r062_compact_routing_scaffolds.jsonl"),
        "public_records_jsonl": str(Path(args.output_root) / "r062_public_records.jsonl"),
        **{f"integration_{key}": value for key, value in integration_paths.items()},
    }
    return {
        "schema_version": "r062_guarded_routing_integration_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r062_guarded_routing_integration_ready_default_off" if gate["gate_passed"] else "r062_guarded_routing_integration_needs_fix",
        "scope": {
            "integration_decision_gate_only": True,
            "target_records_only": TARGET_RECORD_IDS,
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "does_not_prove_artifact_positive_lift": True,
            "does_not_prove_retrieval_improvement": True,
        },
        "inputs": {
            "r060_previews": args.r060_previews,
            "r061_report": args.r061_report,
            "r045_cases": args.r045_cases,
            "r044_report": args.r044_report,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
        },
        "contract": guarded_prompt_integration_contract(),
        "num_records": len(public_records),
        "disabled": {
            "enabled": disabled["enabled"],
            "records_unchanged": disabled["records_unchanged"],
            "num_prompt_previews": len(disabled.get("prompt_previews") or []),
        },
        "enabled": {
            "enabled": enabled["enabled"],
            "records_unchanged": enabled["records_unchanged"],
            "num_prompt_previews": len(enabled.get("prompt_previews") or []),
            "guard_decision_by_record": enabled["manifest"]["guard_decision_by_record"],
            "selected_artifact_count_by_record": enabled["manifest"]["selected_artifact_count_by_record"],
            "no_gold_fields_in_public_previews": enabled["manifest"]["no_gold_fields_in_public_previews"],
        },
        "compact_provider_scaffold": {
            "mode": "r060_derived_compact_page_routing_prompt",
            "boundary": "Optional provider-facing scaffold only; not default adapter prompt and not a full QA prompt.",
            "provenance": "Derived from R060 public page-routed previews and checked against R061 compact prompt hashes.",
            "record_ids": [row["record_id"] for row in compact_scaffolds],
            "hashes_match_r061": all(row["compact_prompt_hash_matches_r061"] for row in compact_scaffolds),
            "sha256_by_record": {str(row["record_id"]): row["compact_prompt_sha256"] for row in compact_scaffolds},
        },
        "r061_boundary_repeated": {
            "provider_diagnostic_records": (r061_report.get("scope") or {}).get("target_records_only"),
            "tests_page_only_routing_behavior_only": (r061_report.get("scope") or {}).get("tests_page_only_routing_behavior_only"),
            "does_not_prove_artifact_positive_lift": (r061_report.get("scope") or {}).get("does_not_prove_artifact_positive_lift"),
            "not_official_mmlongbench_result": (r061_report.get("scope") or {}).get("not_official_mmlongbench_result"),
        },
        "gate": gate,
        "output_paths": output_paths,
        "integration_decision": {
            "recommended_action": "Keep guarded selector and compact page-routing scaffold default-off behind enable_guarded_prompt_scaffold; emit previews/manifest for audit before any future provider run.",
            "adapter_path": "Do not alter official MDocAgent retrieval/eval path in R062; only expose the guarded scaffold as an opt-in preview/control surface.",
            "next_allowed_step": "If manually accepted, wire optional adapter preview hooks or run another no-provider compatibility audit on more positive controls. Do not run full QA from R062.",
        },
    }


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R062 Guarded Routing Integration Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Integration decision only for records 223 and 227.",
        "- Default-disabled adapter behavior must remain unchanged.",
        "- Compact prompt is optional provider-facing scaffold only, derived from R060 and checked against R061 hashes.",
        "- Not an official score, not artifact lift, and not retrieval improvement evidence.",
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
        "# R062 Guarded Routing Integration Decision",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Uses only records 223 and 227 from the R060/R061 page-routing diagnostics.",
        "- Does not prove artifact-aware retrieval lift, retrieval improvement, or official MMLongBench performance.",
        "",
        "## Integration Result",
        f"- disabled records unchanged: {report['disabled']['records_unchanged']}",
        f"- disabled prompt previews: {report['disabled']['num_prompt_previews']}",
        f"- enabled records unchanged: {report['enabled']['records_unchanged']}",
        f"- enabled prompt previews: {report['enabled']['num_prompt_previews']}",
        f"- enabled guard decisions: `{json.dumps(report['enabled']['guard_decision_by_record'], sort_keys=True)}`",
        f"- selected artifact counts: `{json.dumps(report['enabled']['selected_artifact_count_by_record'], sort_keys=True)}`",
        f"- no gold fields in public previews: {report['enabled']['no_gold_fields_in_public_previews']}",
        "",
        "## Compact Scaffold Provenance",
        f"- mode: `{report['compact_provider_scaffold']['mode']}`",
        f"- provenance: {report['compact_provider_scaffold']['provenance']}",
        f"- hashes match R061: {report['compact_provider_scaffold']['hashes_match_r061']}",
        f"- sha256 by record: `{json.dumps(report['compact_provider_scaffold']['sha256_by_record'], sort_keys=True)}`",
        "",
        "## Decision",
        f"- recommended action: {report['integration_decision']['recommended_action']}",
        f"- adapter path: {report['integration_decision']['adapter_path']}",
        f"- next allowed step: {report['integration_decision']['next_allowed_step']}",
    ]
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
