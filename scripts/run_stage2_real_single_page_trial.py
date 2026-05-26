"""Manual one-page Stage 2 real-provider trial runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mdocnexus.stage2.compiler_integration import run_stage2_single_page_real_api_smoke_test
from mdocnexus.stage2.mdocagent_compat import (
    build_api_run_config_from_mdocagent_yaml,
    find_record_by_id_or_doc_question,
    read_json_or_jsonl_records,
)
from mdocnexus.stage2.normalize_record import normalize_record
from mdocnexus.stage2.trial_candidate_selector import strip_eval_only_fields
from mdocnexus.stage2.page_range_validation import (
    OUT_OF_RANGE_ERROR,
    PAGE_COUNT_UNKNOWN_ERROR,
    apply_explicit_page_range_validation_to_canonical_record,
    infer_document_page_count,
)
from scripts.stage2_prepare_single_page_trial import build_preflight_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one guarded Stage 2 real-provider page trial.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--normalized-record-path", default=None)
    parser.add_argument("--sample-path", default=None)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--question-substring", default=None)
    parser.add_argument("--extract-path", default=None)
    parser.add_argument("--extract-root", default=None)
    parser.add_argument("--pdf-root", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--target-page-index", default=None, type=int)
    parser.add_argument("--candidate-report", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--enable-real-api", action="store_true")
    parser.add_argument("--run-real-trial", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env-var", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--raw-output-dir", default=None)
    parser.add_argument("--discard-log-dir", default=None)
    return parser.parse_args()


def apply_candidate_report_to_args(args: argparse.Namespace) -> argparse.Namespace:
    """Populate single-page trial args from an objective candidate report."""

    if not args.candidate_report:
        return args
    report = _load_candidate_report(args.candidate_report)
    selected = report["selected"]
    args.record_id = selected["record_id"]
    args.doc_id = selected["doc_id"]
    args.question_substring = selected.get("question")
    args.target_page_index = int(selected["page_index"])
    if not args.extract_root and report.get("extract_root"):
        args.extract_root = report["extract_root"]
    return args


def _load_candidate_report(candidate_report: str) -> Dict[str, Any]:
    report = json.loads(Path(candidate_report).read_text(encoding="utf-8"))
    if not report.get("selection_passed"):
        raise RuntimeError("Candidate report did not pass selection.")
    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("Candidate report does not contain a selected candidate.")
    required_fields = {"record_id", "doc_id", "page_index"}
    missing_fields = sorted(field for field in required_fields if selected.get(field) in (None, ""))
    if missing_fields:
        raise RuntimeError(f"Candidate report selected block is missing fields: {missing_fields}.")
    return report


def validate_real_trial_args(args: argparse.Namespace) -> None:
    """Reject any invocation that is not an explicit single-page real trial."""

    if getattr(args, "preflight_only", False):
        return
    if not getattr(args, "enable_real_api", False):
        raise RuntimeError("Refusing real provider trial without --enable-real-api.")
    if not getattr(args, "run_real_trial", False):
        raise RuntimeError("Refusing real provider trial without --run-real-trial.")
    if getattr(args, "target_page_index", None) is None:
        raise RuntimeError("A single --target-page-index is required.")


def load_canonical_record_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    record_path = args.normalized_record_path or args.sample_path
    if not record_path:
        raise RuntimeError("Either --normalized-record-path or --sample-path is required.")
    records = read_json_or_jsonl_records(record_path)
    if args.candidate_report:
        records = [strip_eval_only_fields(record) for record in records]
    record = find_record_by_id_or_doc_question(
        records=records,
        record_id=args.record_id,
        doc_id=args.doc_id,
        question_substring=args.question_substring,
    )
    normalized = record if "canonical_record" in record else normalize_record(record)
    canonical_record = normalized["canonical_record"]
    apply_page_range_validation_from_args(canonical_record, args)
    return canonical_record


def build_api_run_config(args: argparse.Namespace):
    return build_api_run_config_from_mdocagent_yaml(
        args.config,
        overrides=build_config_overrides(args, enable_real_api=bool(args.enable_real_api)),
    )


def build_config_overrides(args: argparse.Namespace, enable_real_api: bool) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "enable_real_api": enable_real_api,
        "timeout_seconds": args.timeout_seconds,
        "raw_output_dir": args.raw_output_dir,
        "discard_log_dir": args.discard_log_dir,
    }
    for arg_name, field_name in (
        ("provider", "provider"),
        ("model_name", "model_name"),
        ("api_base_url", "api_base_url"),
        ("api_key_env_var", "api_key_env_var"),
        ("temperature", "temperature"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    return overrides


def resolve_extract_root(args: argparse.Namespace) -> str:
    extract_root = args.extract_root or args.extract_path
    if not extract_root:
        raise RuntimeError("Either --extract-root or --extract-path is required.")
    return extract_root


def run_preflight_only(args: argparse.Namespace) -> Dict[str, Any]:
    record_path = args.sample_path or args.normalized_record_path
    if not record_path:
        raise RuntimeError("Either --sample-path or --normalized-record-path is required for preflight.")
    return build_preflight_report(
        sample_path=record_path,
        extract_root=resolve_extract_root(args),
        config_path=args.config,
        target_page_index=args.target_page_index,
        record_id=args.record_id,
        doc_id=args.doc_id,
        question_substring=args.question_substring,
        overrides=build_config_overrides(args, enable_real_api=False),
        pdf_root=args.pdf_root,
    )


def apply_page_range_validation_from_args(
    canonical_record: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    page_count_info = infer_document_page_count(
        doc_id=canonical_record["document"]["doc_id"],
        pdf_root=args.pdf_root,
        extract_root=resolve_extract_root(args),
    )
    validation = apply_explicit_page_range_validation_to_canonical_record(
        canonical_record,
        page_count_info,
    )
    invalid_target_refs = [
        ref
        for ref in validation["invalid_explicit_page_references"]
        if int(ref.get("page_index_zero_based", -1)) == int(args.target_page_index)
    ]
    if invalid_target_refs:
        error_types = sorted({ref.get("error_type") for ref in invalid_target_refs})
        if OUT_OF_RANGE_ERROR in error_types:
            raise RuntimeError(f"Target page is outside the document page range: {args.target_page_index}.")
        if PAGE_COUNT_UNKNOWN_ERROR in error_types:
            raise RuntimeError(f"Target page range could not be validated: {args.target_page_index}.")
        raise RuntimeError(f"Target page is not valid for compilation: {args.target_page_index}.")
    return validation


def main() -> None:
    args = apply_candidate_report_to_args(parse_args())
    if args.preflight_only:
        print(json.dumps(run_preflight_only(args), ensure_ascii=False, indent=2))
        return

    validate_real_trial_args(args)
    canonical_record = load_canonical_record_from_args(args)
    summary = run_stage2_single_page_real_api_smoke_test(
        canonical_record=canonical_record,
        extract_path=resolve_extract_root(args),
        output_path=args.output_path,
        api_config=build_api_run_config(args),
        target_page_index=args.target_page_index,
        run_real_trial=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
