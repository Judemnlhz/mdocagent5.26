#!/usr/bin/env python3
"""R043 no-provider contrastive prompt exposure scaffold.

This runner prepares prompt/input previews for four contrastive conditions:

* original_pages_only
* page_rerank_only
* original_pages_plus_artifact_snippets
* artifact_snippets_only

It does not call providers, predictions, evaluators, retrieval, or full QA. The
goal is to gate whether a later experiment can actually test prompt-visible
artifact context instead of only page reranking.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_R040_ROOT = "outputs/heldout/r040_targeted_activation_rich_qa/run_tags/r040_targeted_activation_rich_qa"
DEFAULT_R041_MATRIX = "outputs/heldout/r041_r040_identical_score_attribution/record_level_attribution_matrix.jsonl"
DEFAULT_R039_RECORD_IDS = "outputs/heldout/r039_targeted_activation_rich/record_ids.txt"
DEFAULT_RECORDS = "data/MMLongBench/sample-with-retrieval-results.json"
DEFAULT_ARTIFACTS = "outputs/stage2_structured_incremental/r038d_activation_attribution_audit/cumulative20_plus_r037_plus_r038c/atomic_only/artifacts.jsonl"
DEFAULT_EXTRACT_PATH = "tmp/MMLongBench"
DEFAULT_OUTPUT_ROOT = "outputs/heldout/r043_contrastive_prompt_exposure"
RUNS = {
    "original_pages_only": "top4_original_only",
    "page_rerank_only": "top4_artifact_only",
    "original_pages_plus_artifact_snippets": "top4_original_only",
    "artifact_snippets_only": "top4_artifact_only",
}
CONDITIONS = list(RUNS.keys())
BRANCH_FIELDS = ["text-top-10-question", "image-top-10-question"]
FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "answers",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r040-root", default=DEFAULT_R040_ROOT)
    parser.add_argument("--r041-matrix", default=DEFAULT_R041_MATRIX)
    parser.add_argument("--r039-record-ids", default=DEFAULT_R039_RECORD_IDS)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--extract-path", default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-page-chars", type=int, default=1800)
    parser.add_argument("--max-artifacts-per-page", type=int, default=6)
    parser.add_argument("--max-artifact-chars", type=int, default=260)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    manifest_path = output_root / "r043_prompt_exposure_manifest.json"
    gate_json_path = output_root / "r043_prompt_exposure_gate.json"
    gate_md_path = output_root / "r043_prompt_exposure_gate.md"
    preview_dir = output_root / "prompt_previews"
    if not args.execute:
        print(
            json.dumps(
                {
                    "will_execute": False,
                    "output_root": str(output_root),
                    "conditions": CONDITIONS,
                    "manifest": str(manifest_path),
                    "gate": str(gate_json_path),
                    "no_provider_calls": True,
                    "no_new_qa": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    record_ids = read_record_ids(Path(args.r039_record_ids))
    records = read_json(Path(args.records))
    selected_records = [records[index] for index in record_ids]
    matrix_rows = read_jsonl(Path(args.r041_matrix))
    artifacts_by_page = load_artifacts_by_page(Path(args.artifacts))
    run_records = load_r040_reranked_records(Path(args.r040_root))
    previews_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        previews = [
            build_preview(args, condition, record_id, record, matrix_rows[offset], run_records[RUNS[condition]][offset], artifacts_by_page)
            for offset, (record_id, record) in enumerate(zip(record_ids, selected_records))
        ]
        previews_by_condition[condition] = previews
        write_jsonl(preview_dir / f"{condition}.jsonl", previews)
    manifest = build_manifest(args, record_ids, previews_by_condition)
    gate = build_gate(args, manifest, previews_by_condition)
    write_json(manifest_path, manifest)
    write_json(gate_json_path, gate)
    write_gate_markdown(gate_md_path, gate)
    print(
        json.dumps(
            {
                "decision": gate["decision"],
                "gate_passed": gate["gate_passed"],
                "conditions": CONDITIONS,
                "num_records": len(record_ids),
                "manifest": str(manifest_path),
                "gate_md": str(gate_md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_r040_reranked_records(r040_root: Path) -> dict[str, list[dict[str, Any]]]:
    records = {}
    for run in {"top4_original_only", "top4_artifact_only"}:
        path = r040_root / "reranked_records" / f"{run}.json"
        records[run] = read_json(path)
    return records


def build_preview(
    args: argparse.Namespace,
    condition: str,
    record_id: int,
    source_record: dict[str, Any],
    matrix_row: dict[str, Any],
    retrieval_record: dict[str, Any],
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    doc_id = str(source_record["doc_id"])
    doc_stem = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
    text_pages = unique_ints(retrieval_record.get("text-top-10-question", [])[:4])
    image_pages = unique_ints(retrieval_record.get("image-top-10-question", [])[:4])
    combined_pages = unique_ints(text_pages + image_pages)
    include_page_text = condition != "artifact_snippets_only"
    include_artifacts = condition in {"original_pages_plus_artifact_snippets", "artifact_snippets_only"}
    page_contexts = []
    if include_page_text:
        for page in combined_pages:
            page_contexts.append(load_page_context(Path(args.extract_path), doc_stem, page, args.max_page_chars))
    artifact_contexts = []
    if include_artifacts:
        artifact_contexts = select_artifacts_for_pages(
            artifacts_by_page,
            doc_id,
            combined_pages,
            args.max_artifacts_per_page,
            args.max_artifact_chars,
        )
    prompt_preview = render_prompt(condition, source_record["question"], page_contexts, artifact_contexts)
    exposure = {
        "condition": condition,
        "record_id": record_id,
        "doc_id": doc_id,
        "prompt_contains_page_text": include_page_text and bool(page_contexts),
        "prompt_contains_artifacts": include_artifacts and bool(artifact_contexts),
        "artifact_ids": [item["artifact_id"] for item in artifact_contexts],
        "artifact_source_pages": sorted(set(item["page_index"] for item in artifact_contexts)),
        "artifact_snippet_count": len(artifact_contexts),
        "artifact_snippet_char_budget": args.max_artifact_chars,
        "page_text_char_budget": args.max_page_chars,
        "selected_text_pages": text_pages,
        "selected_image_pages": image_pages,
        "combined_pages": combined_pages,
        "missing_page_text_pages": [ctx["page_index"] for ctx in page_contexts if not ctx["exists"]],
        "page_order_changed_vs_original": bool(
            matrix_row["retrieval_deltas_vs_original"].get("top4_artifact_only", {}).get("any_branch_list_changed")
        )
        if condition in {"page_rerank_only", "artifact_snippets_only"}
        else False,
        "r041_binary_pattern": matrix_row["binary_pattern"],
        "r041_case_class": matrix_row["record_attribution"]["correctness_class"],
    }
    return {
        "schema_version": "r043_prompt_preview_v1",
        "condition": condition,
        "record_id": record_id,
        "doc_id": doc_id,
        "question": source_record["question"],
        "retrieval_source_run": RUNS[condition],
        "retrieval_pages": {
            "text": text_pages,
            "image": image_pages,
            "combined": combined_pages,
        },
        "exposure": exposure,
        "page_contexts": page_contexts,
        "artifact_contexts": artifact_contexts,
        "prompt_preview": prompt_preview,
        "prompt_preview_sha256": sha256(prompt_preview),
        "forbidden_gold_fields_present": forbidden_gold_fields({
            "condition": condition,
            "record_id": record_id,
            "doc_id": doc_id,
            "question": source_record["question"],
            "retrieval_pages": {"text": text_pages, "image": image_pages, "combined": combined_pages},
            "exposure": exposure,
            "page_contexts": page_contexts,
            "artifact_contexts": artifact_contexts,
            "prompt_preview": prompt_preview,
        }),
    }


def load_page_context(extract_path: Path, doc_stem: str, page: int, max_chars: int) -> dict[str, Any]:
    path = extract_path / f"{doc_stem}_{page}.txt"
    text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
    text = re.sub(r"\s+", " ", text).strip()
    return {
        "page_index": page,
        "text_path": str(path),
        "exists": path.is_file(),
        "char_count_full": len(text),
        "text_preview": text[:max_chars],
    }


def select_artifacts_for_pages(
    artifacts_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    doc_id: str,
    pages: list[int],
    max_per_page: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    rows = []
    for page in pages:
        for artifact in artifacts_by_page.get((doc_id, page), [])[:max_per_page]:
            content = str(artifact.get("content") or "")
            rows.append(
                {
                    "artifact_id": str(artifact.get("artifact_id") or ""),
                    "artifact_type": str(artifact.get("artifact_type") or ""),
                    "modality": str(artifact.get("modality") or ""),
                    "doc_id": doc_id,
                    "page_index": page,
                    "source_anchored": bool(artifact.get("source_anchored")),
                    "validation_status": artifact.get("validation_status"),
                    "content_preview": re.sub(r"\s+", " ", content).strip()[:max_chars],
                }
            )
    return rows


def render_prompt(condition: str, question: str, page_contexts: list[dict[str, Any]], artifact_contexts: list[dict[str, Any]]) -> str:
    lines = [
        f"[R043 condition: {condition}]",
        "Answer the question using only the visible context below.",
        f"Question: {question}",
        "",
    ]
    if page_contexts:
        lines.append("[Page text context]")
        for ctx in page_contexts:
            lines.append(f"Page {ctx['page_index']} ({'present' if ctx['exists'] else 'missing'}): {ctx['text_preview']}")
        lines.append("")
    if artifact_contexts:
        lines.append("[Artifact snippets]")
        for item in artifact_contexts:
            lines.append(
                f"{item['artifact_id']} | page {item['page_index']} | {item['artifact_type']} | {item['content_preview']}"
            )
        lines.append("")
    if not page_contexts and not artifact_contexts:
        lines.append("[No visible context was available for this condition.]")
    return "\n".join(lines).strip() + "\n"


def build_manifest(args: argparse.Namespace, record_ids: list[int], previews_by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    condition_rows = {}
    for condition, previews in previews_by_condition.items():
        condition_rows[condition] = {
            "num_records": len(previews),
            "retrieval_source_run": RUNS[condition],
            "prompt_contains_page_text_records": sum(1 for row in previews if row["exposure"]["prompt_contains_page_text"]),
            "prompt_contains_artifacts_records": sum(1 for row in previews if row["exposure"]["prompt_contains_artifacts"]),
            "total_artifact_snippets": sum(row["exposure"]["artifact_snippet_count"] for row in previews),
            "missing_page_text_records": sum(1 for row in previews if row["exposure"]["missing_page_text_pages"]),
            "preview_path": f"prompt_previews/{condition}.jsonl",
        }
    return {
        "schema_version": "r043_prompt_exposure_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r043_prompt_exposure_scaffold_prepared",
        "scope": {
            "no_provider_calls": True,
            "no_new_prediction": True,
            "no_new_evaluation": True,
            "no_full_qa": True,
            "prompt_preview_only": True,
            "gold_fields_excluded_from_previews": True,
            "not_full_data_generalization": True,
        },
        "inputs": {
            "r040_root": args.r040_root,
            "r041_matrix": args.r041_matrix,
            "r039_record_ids": args.r039_record_ids,
            "records": args.records,
            "artifacts": args.artifacts,
            "extract_path": args.extract_path,
        },
        "record_ids": record_ids,
        "num_records": len(record_ids),
        "conditions": condition_rows,
        "budgets": {
            "max_page_chars": args.max_page_chars,
            "max_artifacts_per_page": args.max_artifacts_per_page,
            "max_artifact_chars": args.max_artifact_chars,
        },
    }


def build_gate(args: argparse.Namespace, manifest: dict[str, Any], previews_by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    checks = {
        "no_provider_calls": True,
        "no_prediction_or_eval_invoked": True,
        "four_conditions_present": sorted(previews_by_condition) == sorted(CONDITIONS),
        "all_conditions_have_37_records": all(len(rows) == 37 for rows in previews_by_condition.values()),
        "no_gold_fields_in_previews": all(not row["forbidden_gold_fields_present"] for rows in previews_by_condition.values() for row in rows),
        "page_rerank_only_has_no_artifact_snippets": all(not row["exposure"]["prompt_contains_artifacts"] for row in previews_by_condition["page_rerank_only"]),
        "artifact_snippets_only_has_no_page_text": all(not row["exposure"]["prompt_contains_page_text"] for row in previews_by_condition["artifact_snippets_only"]),
        "plus_condition_has_artifact_exposure": all(row["exposure"]["prompt_contains_artifacts"] for row in previews_by_condition["original_pages_plus_artifact_snippets"]),
        "snippet_only_has_artifact_exposure": all(row["exposure"]["prompt_contains_artifacts"] for row in previews_by_condition["artifact_snippets_only"]),
        "original_pages_only_has_no_artifact_snippets": all(not row["exposure"]["prompt_contains_artifacts"] for row in previews_by_condition["original_pages_only"]),
        "artifact_store_bound_to_r038d_union_atomic_store": args.artifacts == DEFAULT_ARTIFACTS,
        "prompt_hashes_unique_by_condition": prompt_hash_uniqueness(previews_by_condition),
    }
    hard_failures = [key for key, value in checks.items() if not value]
    condition_summary = manifest["conditions"]
    return {
        "schema_version": "r043_prompt_exposure_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "r043_prompt_exposure_gate_pass" if not hard_failures else "r043_prompt_exposure_gate_fail",
        "gate_passed": not hard_failures,
        "hard_failures": hard_failures,
        "checks": checks,
        "condition_summary": condition_summary,
        "recommended_next_phase": "small_provider_contrastive_run" if not hard_failures else "fix_prompt_exposure_scaffold",
        "not_full_qa": True,
    }


def prompt_hash_uniqueness(previews_by_condition: dict[str, list[dict[str, Any]]]) -> bool:
    for index in range(37):
        hashes = {condition: previews_by_condition[condition][index]["prompt_preview_sha256"] for condition in CONDITIONS}
        if len(set(hashes.values())) < 3:
            return False
    return True


def write_gate_markdown(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# R043 Prompt Exposure Gate",
        "",
        f"Decision: `{gate['decision']}`",
        f"Gate passed: {gate['gate_passed']}",
        f"Recommended next phase: `{gate['recommended_next_phase']}`",
        "",
        "## Boundary",
        "- No provider calls, no new prediction, no new evaluation, no full QA.",
        "- Prompt previews only.",
        "- Gold answer/evidence fields are excluded from previews.",
        "",
        "## Conditions",
        "| condition | records | page-text records | artifact records | total artifact snippets | missing page-text records |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, row in gate["condition_summary"].items():
        lines.append(
            f"| {condition} | {row['num_records']} | {row['prompt_contains_page_text_records']} | {row['prompt_contains_artifacts_records']} | {row['total_artifact_snippets']} | {row['missing_page_text_records']} |"
        )
    lines.extend(["", "## Checks"])
    for key, value in gate["checks"].items():
        lines.append(f"- `{key}`: {value}")
    if gate["hard_failures"]:
        lines.extend(["", "## Hard Failures"])
        for key in gate["hard_failures"]:
            lines.append(f"- {key}")
    write_text(path, "\n".join(lines) + "\n")


def load_artifacts_by_page(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        doc_id = str(row.get("doc_id") or "")
        try:
            page = int(row.get("page_index"))
        except (TypeError, ValueError):
            continue
        grouped[(doc_id, page)].append(row)
    for key in list(grouped):
        grouped[key] = sorted(grouped[key], key=lambda row: (str(row.get("artifact_id") or ""), str(row.get("content") or "")))
    return grouped


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
