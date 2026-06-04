#!/usr/bin/env python3
"""R053 no-provider question-aware artifact selection and citation prompt gate.

This runner consumes R044/R045 diagnostic cases and builds prompt previews for a
question-aware artifact selector. It does not call providers, run prediction,
run evaluation, or report scores. The goal is to turn R045 support attribution
into a concrete next design gate before any full QA run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_R045_CASES = "outputs/heldout/r045_support_rubric/r045_support_rubric_cases.jsonl"
DEFAULT_R044_REPORT = "outputs/heldout/r044_small_contrastive_provider/r044_diagnostic_attribution_report.json"
DEFAULT_R040_ROOT = "outputs/heldout/r040_targeted_activation_rich_qa/run_tags/r040_targeted_activation_rich_qa"
DEFAULT_R039_RECORD_IDS = "outputs/heldout/r039_targeted_activation_rich/record_ids.txt"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_ARTIFACTS = "outputs/stage2_structured_incremental/r038d_activation_attribution_audit/cumulative20_plus_r037_plus_r038c/atomic_only/artifacts.jsonl"
DEFAULT_EXTRACT_PATH = "tmp/MMLongBench"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r053_question_aware_scaffold"
FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "answers",
    "gold_answer",
    "gold_answer_for_posthoc_diagnostic_only",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}
NUMERIC_WORDS = {
    "percent",
    "percentage",
    "difference",
    "sum",
    "total",
    "ratio",
    "rate",
    "number",
    "amount",
    "value",
    "average",
}
CODE_PATTERN = re.compile(r"\b[A-Z]{1,4}\d{1,4}\b")
STOPWORDS = {
    "according",
    "document",
    "what",
    "which",
    "who",
    "where",
    "when",
    "this",
    "that",
    "with",
    "from",
    "into",
    "about",
    "using",
    "there",
    "their",
    "have",
    "been",
    "were",
    "will",
    "shall",
    "the",
    "and",
    "for",
    "are",
    "was",
    "not",
    "can",
    "you",
    "its",
    "his",
    "her",
}


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
    report_json = output_root / "r053_question_aware_scaffold_report.json"
    report_md = output_root / "r053_question_aware_scaffold_report.md"
    gate_json = output_root / "r053_question_aware_scaffold_gate.json"
    gate_md = output_root / "r053_question_aware_scaffold_gate.md"
    previews_path = output_root / "r053_question_aware_prompt_previews.jsonl"
    compact_path = output_root / "r053_question_aware_compact_index.jsonl"
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "output_root": str(output_root),
                    "no_provider_calls": True,
                    "no_full_qa": True,
                    "design_gate_only": True,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    r045_cases = read_jsonl(Path(args.r045_cases))
    r044_report = read_json(Path(args.r044_report))
    records = read_json(Path(args.records))
    record_ids = read_record_ids(Path(args.r039_record_ids))
    offsets = {record_id: offset for offset, record_id in enumerate(record_ids)}
    run_records = load_r040_records(Path(args.r040_root))
    artifacts_by_page = load_artifacts_by_page(Path(args.artifacts))
    previews = build_previews(args, r045_cases, r044_report, records, offsets, run_records, artifacts_by_page)
    gate = build_gate(args, r045_cases, previews)
    report = build_report(args, r045_cases, previews, gate)
    write_jsonl(previews_path, previews)
    write_jsonl(compact_path, build_compact_index(previews))
    write_json(gate_json, gate)
    write_gate_markdown(gate_md, gate)
    write_json(report_json, report)
    write_report_markdown(report_md, report)
    print(
        json.dumps(
            {
                "decision": gate["decision"],
                "gate_passed": gate["gate_passed"],
                "num_cases": len(previews),
                "report_md": str(report_md),
                "no_provider_calls": True,
                "no_full_qa": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


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
        original_pages = combined_pages(original_record)
        artifact_pages = combined_pages(artifact_record)
        candidate_pages = unique_ints(artifact_pages + original_pages)
        question = str(source["question"])
        profile = question_profile(question)
        artifact_candidates = []
        for page in candidate_pages:
            for artifact in artifacts_by_page.get((doc_id, page), []):
                artifact_candidates.append(score_artifact(artifact, question, profile, page, artifact_pages, original_pages, args.max_artifact_chars))
        selected_artifacts = select_question_aware_artifacts(artifact_candidates, args.max_artifacts)
        page_contexts = [load_page_context(Path(args.extract_path), doc_id, page, args.max_page_chars) for page in artifact_pages]
        prompt = render_prompt(question, page_contexts, selected_artifacts, profile)
        rows.append(
            {
                "schema_version": "r053_question_aware_prompt_preview_v1",
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
                    "name": "question_aware_artifact_selection_v1",
                    "not_first_n_per_page": True,
                    "uses_question_tokens": True,
                    "uses_artifact_type_priority": True,
                    "uses_metric_value_code_matching": True,
                    "uses_retrieved_candidate_pages_only": True,
                    "uses_gold_fields": False,
                    "unsupported_answer_guard": True,
                },
                "candidate_artifact_count": len(artifact_candidates),
                "selected_artifact_count": len(selected_artifacts),
                "selected_artifacts": selected_artifacts,
                "page_contexts": page_contexts,
                "prompt_preview": prompt,
                "prompt_preview_sha256": sha256(prompt),
                "forbidden_gold_fields_present": forbidden_gold_fields(
                    {
                        "record_id": record_id,
                        "doc_id": doc_id,
                        "question": question,
                        "question_profile": profile,
                        "retrieval_pages": {"artifact": artifact_pages, "original": original_pages, "candidate_union": candidate_pages},
                        "selected_artifacts": selected_artifacts,
                        "page_contexts": page_contexts,
                        "prompt_preview": prompt,
                    }
                ),
            }
        )
    return rows


def question_profile(question: str) -> dict[str, Any]:
    q_norm = normalize(question)
    tokens = sorted(question_tokens(question))
    codes = CODE_PATTERN.findall(question)
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", question)
    is_numeric = any(word in q_norm for word in NUMERIC_WORDS) or bool(numbers)
    is_code_or_table = bool(codes) or any(word in q_norm for word in ["code", "table", "row", "column", "market"])
    return {
        "tokens": tokens,
        "codes": codes,
        "numbers": numbers,
        "is_numeric_or_table_question": bool(is_numeric or is_code_or_table),
        "requires_unsupported_answer_guard": True,
        "answer_policy": "cite_visible_support_or_refuse",
    }


def score_artifact(
    artifact: dict[str, Any],
    question: str,
    profile: dict[str, Any],
    page: int,
    artifact_pages: list[int],
    original_pages: list[int],
    max_chars: int,
) -> dict[str, Any]:
    content = re.sub(r"\s+", " ", str(artifact.get("content") or "")).strip()
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    searchable = " ".join([content, json.dumps(normalized, ensure_ascii=False, sort_keys=True)])
    artifact_tokens = question_tokens(searchable)
    q_tokens = set(profile["tokens"])
    overlap = sorted(q_tokens & artifact_tokens)
    score = float(len(overlap))
    reasons = []
    if overlap:
        reasons.append("question_token_overlap")
    artifact_type = str(artifact.get("artifact_type") or "")
    modality = str(artifact.get("modality") or "")
    if profile["is_numeric_or_table_question"] and artifact_type in {"numeric_fact", "table_cell", "table"}:
        score += 3.0
        reasons.append("numeric_table_type_priority")
    if artifact_type in {"numeric_fact", "table_cell"}:
        score += 1.0
        reasons.append("atomic_artifact_priority")
    for code in profile["codes"]:
        if code.lower() in searchable.lower():
            score += 5.0
            reasons.append(f"code_match:{code}")
    for number in profile["numbers"]:
        if number and number in searchable:
            score += 2.0
            reasons.append(f"value_match:{number}")
    metric_text = " ".join(str(normalized.get(key) or "") for key in ["metric_name", "row_label", "row_header", "column_label", "column_header", "value_text", "unit"])
    metric_overlap = sorted(q_tokens & question_tokens(metric_text))
    if metric_overlap:
        score += float(len(metric_overlap)) * 1.5
        reasons.append("metric_label_overlap")
    if page in artifact_pages:
        score += 1.0
        reasons.append("artifact_reranked_page")
    if page in original_pages:
        score += 0.25
        reasons.append("original_candidate_page")
    if not bool(artifact.get("source_anchored")):
        score -= 2.0
        reasons.append("source_anchor_missing_penalty")
    if not reasons:
        reasons.append("low_question_match_retained_for_audit")
    return {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "artifact_type": artifact_type,
        "modality": modality,
        "doc_id": str(artifact.get("doc_id") or ""),
        "page_index": page,
        "content_preview": content[:max_chars],
        "normalized_content": compact_normalized(normalized),
        "source_anchored": bool(artifact.get("source_anchored")),
        "validation_status": artifact.get("validation_status"),
        "selection_score": round(score, 4),
        "selection_reasons": sorted(set(reasons)),
        "question_token_overlap": overlap,
    }


def select_question_aware_artifacts(candidates: list[dict[str, Any]], max_artifacts: int) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["selection_score"]),
            int(row["page_index"]),
            row["artifact_type"],
            row["artifact_id"],
        ),
    )
    selected = ranked[:max_artifacts]
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    return selected


def render_prompt(question: str, page_contexts: list[dict[str, Any]], artifacts: list[dict[str, Any]], profile: dict[str, Any]) -> str:
    lines = [
        "[R053 condition: question_aware_citation_prompt]",
        "Answer using only the visible page evidence and artifact evidence below.",
        "First list supporting evidence, then answer. Cite page ids and artifact ids for every factual claim.",
        "If the visible evidence does not fully support an answer, say Not answerable and cite what is missing. Do not infer from partial numeric overlap or unsupported artifact snippets.",
        f"Question: {question}",
        "",
        "[Question profile]",
        f"numeric_or_table_question={profile['is_numeric_or_table_question']}; codes={profile['codes']}; numbers={profile['numbers']}",
        "",
        "[Page evidence]",
    ]
    if page_contexts:
        for ctx in page_contexts:
            lines.append(f"Page {ctx['page_index']} ({'present' if ctx['exists'] else 'missing'}): {ctx['text_preview']}")
    else:
        lines.append("No page evidence is visible.")
    lines.extend(["", "[Artifact evidence]"])
    if artifacts:
        for item in artifacts:
            lines.append(
                f"{item['artifact_id']} | page {item['page_index']} | type={item['artifact_type']} | score={item['selection_score']} | reasons={','.join(item['selection_reasons'])} | {item['content_preview']}"
            )
    else:
        lines.append("No artifact evidence was selected by the question-aware selector.")
    lines.extend(
        [
            "",
            "[Required response format]",
            "Page evidence: cite page ids or state none.",
            "Artifact evidence: cite artifact ids or state none.",
            "Unsupported-answer check: explain whether the visible evidence fully supports the answer.",
            "Final answer: answer or Not answerable.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_gate(args: argparse.Namespace, cases: list[dict[str, Any]], previews: list[dict[str, Any]]) -> dict[str, Any]:
    selected_counts = [row["selected_artifact_count"] for row in previews]
    prompt_texts = [row["prompt_preview"] for row in previews]
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "no_full_qa": True,
        "target_cases_match_r045": sorted(int(c["record_id"]) for c in cases) == sorted(row["record_id"] for row in previews),
        "all_prompts_have_citation_requirement": all("Cite page ids and artifact ids" in text for text in prompt_texts),
        "all_prompts_have_unsupported_answer_guard": all("Not answerable" in text and "Do not infer" in text for text in prompt_texts),
        "page_and_artifact_evidence_separated": all("[Page evidence]" in text and "[Artifact evidence]" in text for text in prompt_texts),
        "question_aware_policy_not_first_n": all(row["selection_policy"]["not_first_n_per_page"] for row in previews),
        "selected_artifact_budget_respected": all(count <= args.max_artifacts for count in selected_counts),
        "at_least_one_case_has_selected_artifacts": any(count > 0 for count in selected_counts),
        "no_gold_fields_in_public_previews": all(not row["forbidden_gold_fields_present"] for row in previews),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "design_gate_only": True,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    return {
        "schema_version": "r053_question_aware_scaffold_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r053_question_aware_scaffold_gate_pass" if not hard_failures else "r053_question_aware_scaffold_gate_fail",
        "gate_passed": not hard_failures,
        "checks": checks,
        "hard_failures": hard_failures,
        "num_cases": len(previews),
        "selected_artifact_count_by_record": {str(row["record_id"]): row["selected_artifact_count"] for row in previews},
        "not_full_qa": True,
        "not_official_score": True,
    }


def build_report(args: argparse.Namespace, cases: list[dict[str, Any]], previews: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    rubric_counts = Counter(row["r045_rubric_label"] for row in previews)
    transition_counts = Counter(label for row in previews for label in row["r044_transition_labels"])
    selection_reason_counts = Counter(reason for row in previews for artifact in row["selected_artifacts"] for reason in artifact["selection_reasons"])
    return {
        "schema_version": "r053_question_aware_scaffold_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r053_question_aware_scaffold_complete" if gate["gate_passed"] else "r053_question_aware_scaffold_needs_fix",
        "scope": {
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "not_official_score": True,
            "prompt_and_selection_scaffold_only": True,
        },
        "inputs": {
            "r045_cases": args.r045_cases,
            "r044_report": args.r044_report,
            "r040_root": args.r040_root,
            "artifacts": args.artifacts,
        },
        "num_cases": len(previews),
        "rubric_label_counts": dict(sorted(rubric_counts.items())),
        "transition_label_counts": dict(sorted(transition_counts.items())),
        "selection_reason_counts": dict(sorted(selection_reason_counts.items())),
        "gate": gate,
        "recommended_next": [
            "Manually inspect R053 prompt previews for records 384, 508, and 569 before any provider run.",
            "If selected artifacts still lack necessary evidence, improve artifact selection with stronger question/value/metric matching.",
            "If selected artifacts contain evidence but prompts remain ambiguous, run a small prompt-template diagnostic with citation-required output only.",
            "Do not run full QA until this scaffold is manually accepted and recorded in the tracker.",
        ],
    }


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R053 Question-Aware Scaffold Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Prompt and artifact-selection scaffold only.",
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
    write_text(path, "\n".join(lines) + "\n")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# R053 Question-Aware Artifact Selection Scaffold",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Boundary",
        "- No provider calls, no prediction, no evaluation, no full QA.",
        "- Uses R045 cases to design a question-aware artifact selector and citation prompt preview.",
        "- Not an official score.",
        "",
        "## Summary",
        f"- cases: {report['num_cases']}",
        f"- rubric labels: `{json.dumps(report['rubric_label_counts'], sort_keys=True)}`",
        f"- transition labels: `{json.dumps(report['transition_label_counts'], sort_keys=True)}`",
        f"- selection reasons: `{json.dumps(report['selection_reason_counts'], sort_keys=True)}`",
        "",
        "## Recommended Next",
    ]
    for item in report["recommended_next"]:
        lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def build_compact_index(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "r053_question_aware_compact_index_v1",
            "record_id": row["record_id"],
            "doc_id": row["doc_id"],
            "case_type": row["case_type"],
            "r045_rubric_label": row["r045_rubric_label"],
            "r044_transition_labels": row["r044_transition_labels"],
            "selected_artifact_count": row["selected_artifact_count"],
            "selected_artifact_ids": [artifact["artifact_id"] for artifact in row["selected_artifacts"]],
            "selected_artifact_pages": sorted({artifact["page_index"] for artifact in row["selected_artifacts"]}),
            "prompt_preview_sha256": row["prompt_preview_sha256"],
            "question_profile": row["question_profile"],
        }
        for row in previews
    ]


def load_page_context(extract_path: Path, doc_id: str, page: int, max_chars: int) -> dict[str, Any]:
    doc_stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    path = extract_path / f"{doc_stem}_{page}.txt"
    text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
    text = re.sub(r"\s+", " ", text).strip()
    return {
        "page_index": page,
        "page_id": f"{doc_id}#p{page:03d}",
        "exists": path.is_file(),
        "char_count_full": len(text),
        "text_preview": text[:max_chars],
    }


def load_r040_records(r040_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "top4_original_only": read_json(r040_root / "reranked_records/top4_original_only.json"),
        "top4_artifact_only": read_json(r040_root / "reranked_records/top4_artifact_only.json"),
    }


def load_artifacts_by_page(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        try:
            page = int(row.get("page_index"))
        except (TypeError, ValueError):
            continue
        grouped[(str(row.get("doc_id") or ""), page)].append(row)
    return grouped


def combined_pages(record: dict[str, Any]) -> list[int]:
    return unique_ints(list(record.get("text-top-10-question", [])[:4]) + list(record.get("image-top-10-question", [])[:4]))


def compact_normalized(value: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in ["metric_name", "row_label", "row_header", "column_label", "column_header", "value_text", "unit", "normalized_value"]:
        if key in value:
            keep[key] = value[key]
    return keep


def forbidden_gold_fields(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_PUBLIC_KEYS:
                found.append(next_path)
            found.extend(forbidden_gold_fields(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_gold_fields(item, f"{path}[{index}]"))
    return found


def question_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", normalize(text)) if len(token) > 2 and token not in STOPWORDS}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def unique_ints(values: list[Any]) -> list[int]:
    rows = []
    seen = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item not in seen:
            seen.add(item)
            rows.append(item)
    return rows


def read_record_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
