#!/usr/bin/env python3
"""R057 no-provider guarded integration design gate.

R057 turns the R056 reusable guarded prompt scaffold into an opt-in integration
contract. It does not call providers, run prediction, run evaluation, or report
scores. The key checks are:

1. default disabled integration leaves adapter records unchanged;
2. enabled integration builds prompt previews and a manifest only;
3. prompt previews use public inputs only and contain no gold fields;
4. the integration contract explicitly says this is not an artifact-lift claim.
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

from mdocnexus.integration.guarded_integration import (
    GuardedPromptIntegrationConfig,
    apply_guarded_prompt_integration,
    guarded_prompt_integration_contract,
    write_integration_outputs,
)
from mdocnexus.integration.mdocagent_adapter import canonical_json_hash

DEFAULT_R045_CASES = r053.DEFAULT_R045_CASES
DEFAULT_R044_REPORT = r053.DEFAULT_R044_REPORT
DEFAULT_R040_ROOT = r053.DEFAULT_R040_ROOT
DEFAULT_R039_RECORD_IDS = r053.DEFAULT_R039_RECORD_IDS
DEFAULT_RECORDS = r053.DEFAULT_RECORDS
DEFAULT_ARTIFACTS = r053.DEFAULT_ARTIFACTS
DEFAULT_EXTRACT_PATH = r053.DEFAULT_EXTRACT_PATH
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r057_guarded_integration_design_gate"


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
            "integration_design_gate_only": True,
            "default_enabled": False,
            "config_flag": "enable_guarded_prompt_scaffold",
        }, ensure_ascii=False, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    r045_cases = r053.read_jsonl(Path(args.r045_cases))
    r044_report = r053.read_json(Path(args.r044_report))
    records = r053.read_json(Path(args.records))
    record_ids = r053.read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(record_ids)}
    run_records = r053.load_r040_records(Path(args.r040_root))
    artifacts_by_tuple = r053.load_artifacts_by_page(Path(args.artifacts))
    public_records, page_contexts, artifacts_by_page = build_public_inputs(args, r045_cases, r044_report, records, offsets, run_records, artifacts_by_tuple)

    disabled = apply_guarded_prompt_integration(public_records, artifacts_by_page, page_contexts, GuardedPromptIntegrationConfig())
    enabled = apply_guarded_prompt_integration(
        public_records,
        artifacts_by_page,
        page_contexts,
        GuardedPromptIntegrationConfig(enable_guarded_prompt_scaffold=True),
    )
    output_paths = write_integration_outputs(enabled, output_root / "integration_outputs")
    gate = build_gate(args, public_records, disabled, enabled)
    report = build_report(args, public_records, disabled, enabled, gate, output_paths)

    r053.write_json(output_root / "r057_guarded_integration_gate.json", gate)
    write_gate_markdown(output_root / "r057_guarded_integration_gate.md", gate)
    r053.write_json(output_root / "r057_guarded_integration_report.json", report)
    write_report_markdown(output_root / "r057_guarded_integration_report.md", report)
    r053.write_json(output_root / "r057_integration_contract.json", guarded_prompt_integration_contract())
    r053.write_jsonl(output_root / "r057_public_records.jsonl", public_records)
    print(json.dumps({
        "decision": gate["decision"],
        "gate_passed": gate["gate_passed"],
        "num_records": len(public_records),
        "prompt_previews": len(enabled["prompt_previews"]),
        "records_unchanged_when_disabled": gate["checks"]["disabled_records_unchanged"],
        "records_unchanged_when_enabled": gate["checks"]["enabled_records_unchanged"],
        "report_md": str(output_root / "r057_guarded_integration_report.md"),
        "no_provider_calls": True,
        "no_full_qa": True,
    }, ensure_ascii=False, indent=2))


def build_public_inputs(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    r044_report: dict[str, Any],
    records: list[dict[str, Any]],
    offsets: dict[int, int],
    run_records: dict[str, list[dict[str, Any]]],
    artifacts_by_tuple: dict[tuple[str, int], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, dict[int, list[dict[str, Any]]]]]:
    r044_by_id = {int(row["record_id"]): row for row in r044_report["per_record"]}
    public_records = []
    page_contexts: dict[tuple[str, int], dict[str, Any]] = {}
    artifacts_by_page: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for case in cases:
        record_id = int(case["record_id"])
        if record_id not in offsets:
            raise ValueError(f"R045 case record_id not in R039 subset: {record_id}")
        source = records[record_id]
        doc_id = str(source["doc_id"])
        offset = offsets[record_id]
        artifact_record = run_records["top4_artifact_only"][offset]
        original_record = run_records["top4_original_only"][offset]
        artifact_pages = r053.combined_pages(artifact_record)
        original_pages = r053.combined_pages(original_record)
        candidate_pages = r053.unique_ints(artifact_pages + original_pages)
        public_record = {
            "record_index": record_id,
            "doc_id": doc_id,
            "question": str(source["question"]),
            "text-top-10-question": artifact_pages[:4],
            "text-top-10-question_score": descending_scores(artifact_pages[:4]),
            "image-top-10-question": original_pages[:4],
            "image-top-10-question_score": descending_scores(original_pages[:4]),
            "_r057_case_type": case.get("case_type"),
            "_r057_r045_rubric_label": case.get("rubric_label"),
            "_r057_r044_transition_labels": r044_by_id.get(record_id, {}).get("transition_labels", []),
        }
        public_records.append(public_record)
        for page in candidate_pages:
            page_contexts[(doc_id, page)] = r053.load_page_context(Path(args.extract_path), doc_id, page, 1400)
            artifacts = []
            for artifact in artifacts_by_tuple.get((doc_id, page), []):
                artifacts.append(public_artifact(artifact, doc_id, page))
            if artifacts:
                artifacts_by_page.setdefault(doc_id, {})[page] = artifacts
    return public_records, page_contexts, artifacts_by_page


def public_artifact(artifact: dict[str, Any], doc_id: str, page: int) -> dict[str, Any]:
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    return {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "artifact_type": str(artifact.get("artifact_type") or ""),
        "modality": str(artifact.get("modality") or ""),
        "doc_id": str(artifact.get("doc_id") or doc_id),
        "page_index": int(artifact.get("page_index", page)),
        "content": str(artifact.get("content") or ""),
        "normalized_content": dict(normalized),
        "source_anchored": bool(artifact.get("source_anchored")),
        "validation_status": artifact.get("validation_status"),
    }


def descending_scores(pages: list[int]) -> list[float]:
    return [round(1.0 / float(index + 1), 8) for index, _ in enumerate(pages)]


def build_gate(args: argparse.Namespace, public_records: list[dict[str, Any]], disabled: dict[str, Any], enabled: dict[str, Any]) -> dict[str, Any]:
    disabled_manifest = disabled["manifest"]
    enabled_manifest = enabled["manifest"]
    forbidden = list(enabled_manifest.get("forbidden_gold_fields_present") or [])
    prompt_previews = list(enabled.get("prompt_previews") or [])
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "contract_default_disabled": guarded_prompt_integration_contract()["default_enabled"] is False,
        "contract_has_config_flag": guarded_prompt_integration_contract()["config_flag"] == "enable_guarded_prompt_scaffold",
        "disabled_records_unchanged": disabled["input_records_sha256"] == disabled["output_records_sha256"] == canonical_json_hash(public_records),
        "disabled_generates_no_prompt_previews": disabled_manifest["num_prompt_previews"] == 0,
        "enabled_records_unchanged": enabled["input_records_sha256"] == enabled["output_records_sha256"] == canonical_json_hash(public_records),
        "enabled_generates_prompt_previews": enabled_manifest["num_prompt_previews"] == len(public_records) == len(prompt_previews),
        "enabled_manifest_no_gold": enabled_manifest["no_gold_fields_in_public_previews"] is True and not forbidden,
        "prompt_previews_have_no_gold_fields": all(not row.get("forbidden_gold_fields_present") for row in prompt_previews),
        "enabled_does_not_claim_artifact_lift": enabled_manifest["not_artifact_lift_claim"] is True,
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "integration_outputs_written": True,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r057_guarded_integration_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r057_guarded_integration_gate_pass" if not hard_failures else "r057_guarded_integration_gate_fail",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_records": len(public_records),
        "disabled_manifest": disabled_manifest,
        "enabled_manifest": enabled_manifest,
        "not_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def build_report(args: argparse.Namespace, public_records: list[dict[str, Any]], disabled: dict[str, Any], enabled: dict[str, Any], gate: dict[str, Any], output_paths: dict[str, str]) -> dict[str, Any]:
    decisions = Counter(row.get("guard_decision") for row in enabled.get("prompt_previews") or [])
    return {
        "schema_version": "r057_guarded_integration_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r057_guarded_integration_complete" if gate["gate_passed"] else "r057_guarded_integration_needs_fix",
        "scope": {
            "no_provider_calls": True,
            "no_prediction": True,
            "no_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "integration_design_gate_only": True,
            "default_disabled": True,
            "does_not_prove_artifact_positive_lift": True,
        },
        "module": "mdocnexus.integration.guarded_integration",
        "contract": guarded_prompt_integration_contract(),
        "inputs": {
            "r045_cases": args.r045_cases,
            "r044_report": args.r044_report,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
        },
        "num_records": len(public_records),
        "disabled": {
            "enabled": disabled["enabled"],
            "records_unchanged": disabled["records_unchanged"],
            "num_prompt_previews": len(disabled["prompt_previews"]),
        },
        "enabled": {
            "enabled": enabled["enabled"],
            "records_unchanged": enabled["records_unchanged"],
            "num_prompt_previews": len(enabled["prompt_previews"]),
            "guard_decision_counts": dict(sorted(decisions.items())),
            "no_gold_fields_in_public_previews": enabled["manifest"]["no_gold_fields_in_public_previews"],
        },
        "output_paths": output_paths,
        "gate": gate,
        "recommended_next": [
            "Do not run full QA from R057.",
            "If integration contract is accepted, either wire it behind the disabled-by-default config flag or run R058 tiny positive-evidence diagnostic first.",
            "Keep claims limited: R057 proves an opt-in integration contract and default-off safety, not artifact-aware QA improvement.",
        ],
    }


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R057 Guarded Integration Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Default-disabled integration contract only.",
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
        "# R057 Guarded Integration Design Gate",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Adds an opt-in integration contract for `mdocnexus.integration.guarded_prompt`.",
        "- Default config leaves adapter records unchanged and generates no prompt previews.",
        "- Not an official score and not evidence of artifact positive lift.",
        "",
        "## Summary",
        f"- records: {report['num_records']}",
        f"- disabled records unchanged: {report['disabled']['records_unchanged']}",
        f"- disabled prompt previews: {report['disabled']['num_prompt_previews']}",
        f"- enabled records unchanged: {report['enabled']['records_unchanged']}",
        f"- enabled prompt previews: {report['enabled']['num_prompt_previews']}",
        f"- enabled guard decisions: `{json.dumps(report['enabled']['guard_decision_counts'], sort_keys=True)}`",
        f"- no gold fields in public previews: {report['enabled']['no_gold_fields_in_public_previews']}",
        "",
        "## Recommended Next",
    ]
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    r053.write_text(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()