"""MDocAgent-aligned Stage 2 JSON augmentation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .constraint_parser import parse_question_constraints
from .layout_parser import build_basic_layout_blocks
from .mdocagent_compat import (
    build_mdocagent_extract_paths,
    read_json_or_jsonl_records,
)
from .page_loader import find_existing_file
from .page_range_validation import (
    validate_explicit_page_references_against_page_count,
    infer_document_page_count,
)
from .retrieval_processor import parse_sequence_field


TEXT_TOP_10_FIELD = "text-top-10-question"
IMAGE_TOP_10_FIELD = "image-top-10-question"
FORBIDDEN_STAGE2_FIELDS = {
    "answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "api_key",
    "proof_trace",
    "verified",
    "answer_supported",
    "proof_used",
}


SUMMARY_FORBIDDEN_FIELDS = FORBIDDEN_STAGE2_FIELDS | {
    "gold_annotation",
    "baseline_outputs",
}


def augment_retrieval_results_file(
    input_path: str | Path,
    output_path: str | Path,
    extract_root: str | Path,
    config_path: str | Path | None = None,
    max_records: int | None = None,
) -> List[Dict[str, Any]]:
    """Read MDocAgent retrieval results and write records with only a stage2 addition."""

    records = read_json_or_jsonl_records(input_path)
    augmented = augment_retrieval_records(
        records=records,
        extract_root=extract_root,
        config_path=config_path,
        max_records=max_records,
    )
    write_json(augmented, output_path)
    return augmented


def augment_retrieval_records(
    records: Iterable[Mapping[str, Any]],
    extract_root: str | Path,
    config_path: str | Path | None = None,
    max_records: int | None = None,
) -> List[Dict[str, Any]]:
    """Append a stage2 preflight block while preserving all original fields."""

    if config_path is not None and not Path(config_path).is_file():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    result: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if max_records is not None and index >= int(max_records):
            break
        augmented_record = dict(record)
        augmented_record["stage2"] = build_stage2_preflight(record, extract_root)
        result.append(augmented_record)
    return result


def build_stage2_preflight(record: Mapping[str, Any], extract_root: str | Path) -> Dict[str, Any]:
    """Build the compact Stage 2 page-route index for one retrieval result record."""

    doc_id = str(record["doc_id"])
    candidate_page_routes = build_candidate_page_routes(record)
    candidate_page_sources = [
        build_page_source(
            doc_id=doc_id,
            extract_root=extract_root,
            page_index=route["page_index"],
        )
        for route in candidate_page_routes
    ]
    blocking_reasons = build_preflight_blocking_reasons(
        candidate_page_routes=candidate_page_routes,
        candidate_page_sources=candidate_page_sources,
    )

    return {
        "preflight": {
            "passed": not blocking_reasons,
            "blocking_reasons": blocking_reasons,
        },
        "candidate_page_routes": candidate_page_routes,
    }


def build_candidate_page_routes(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build compact page routes from original MDocAgent text/image top-10 fields."""

    text_pages_raw = parse_sequence_field(record.get(TEXT_TOP_10_FIELD, []), TEXT_TOP_10_FIELD)
    image_pages_raw = parse_sequence_field(record.get(IMAGE_TOP_10_FIELD, []), IMAGE_TOP_10_FIELD)
    route_map: Dict[int, set[str]] = {}
    for page_index in text_pages_raw:
        route_map.setdefault(int(page_index), set()).add("text")
    for page_index in image_pages_raw:
        route_map.setdefault(int(page_index), set()).add("image")
    return [
        {
            "page_index": page_index,
            "routes": [route for route in ("text", "image") if route in routes],
        }
        for page_index, routes in sorted(route_map.items(), key=lambda item: item[0])
    ]


def build_page_source(
    doc_id: str,
    extract_root: str | Path,
    page_index: int,
) -> Dict[str, Any]:
    """Build page source metadata using BaseDataset-compatible paths first."""

    paths = build_mdocagent_extract_paths(extract_root, doc_id, page_index)
    text_path = find_existing_file(list(paths["text_candidate_paths"]))
    image_path = find_existing_file(list(paths["image_candidate_paths"]))
    page_text = text_path.read_text(encoding="utf-8", errors="replace") if text_path else None
    layout_blocks = build_basic_layout_blocks(
        doc_id=doc_id,
        page_index=page_index,
        page_text=page_text,
        has_page_image=image_path is not None,
    )
    return {
        "page_index": int(page_index),
        "page_text_path": str(text_path) if text_path else None,
        "page_image_path": str(image_path) if image_path else None,
        "has_page_text": text_path is not None,
        "has_page_image": image_path is not None,
        "layout_block_ids": [block["block_id"] for block in layout_blocks],
    }


def build_preflight_blocking_reasons(
    candidate_page_routes: List[Dict[str, Any]],
    candidate_page_sources: List[Dict[str, Any]],
) -> List[str]:
    reasons: set[str] = set()
    if not candidate_page_routes:
        reasons.add("no_candidate_page_routes")
    elif not any(source["has_page_text"] or source["has_page_image"] for source in candidate_page_sources):
        reasons.add("missing_all_candidate_sources")
    return sorted(reasons)


def select_trial_candidate_from_stage2_file(
    stage2_json: str | Path,
    output_path: str | Path | None = None,
    extract_root: str | Path = "tmp/MMLongBench",
) -> Dict[str, Any]:
    records = read_json_or_jsonl_records(stage2_json)
    report = select_trial_candidate_from_stage2_records(records, extract_root=extract_root)
    if output_path is not None:
        write_json(report, output_path)
    return report


def select_trial_candidate_from_stage2_records(
    records: Iterable[Mapping[str, Any]],
    extract_root: str | Path = "tmp/MMLongBench",
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        stage2 = record.get("stage2", {})
        if not isinstance(stage2, dict):
            continue
        candidates.extend(build_record_trial_candidates(record, record_index, extract_root))

    selected = select_best_trial_candidate(candidates)
    return {
        "selection_passed": selected is not None,
        "blocking_reasons": [] if selected is not None else ["no_valid_single_page_trial_candidate"],
        "selected": selected,
        "selection_policy": {
            "priority_order": [
                "valid_explicit_page_with_image",
                "image_top_10_first_available",
                "retrieval_union_first_available",
            ],
            "uses_answer": False,
            "uses_evidence_pages": False,
            "uses_binary_correctness": False,
        },
    }


def build_record_trial_candidates(
    record: Mapping[str, Any],
    record_index: int,
    extract_root: str | Path = "tmp/MMLongBench",
) -> List[Dict[str, Any]]:
    stage2 = record.get("stage2", {})
    if not isinstance(stage2, dict) or not stage2.get("preflight", {}).get("passed", False):
        return []
    route_pages = _candidate_route_pages(stage2)
    page_sources_by_index = {
        page_index: build_page_source(str(record.get("doc_id")), extract_root, page_index)
        for page_index in route_pages
    }
    valid_explicit_pages = get_valid_explicit_page_indices(record, extract_root)
    explicit_refs = parse_question_constraints(str(record.get("question", ""))).get("explicit_page_references", [])
    image_unique_pages = _candidate_route_pages(stage2, required_route="image")
    retrieval_candidate_pages = route_pages

    candidates: List[Dict[str, Any]] = []
    for order, page_index in enumerate(valid_explicit_pages):
        source = page_sources_by_index.get(page_index)
        if page_source_has_image(source):
            candidates.append(
                build_trial_candidate(record, record_index, page_index, source, "valid_explicit_page_with_image", 0, order)
            )

    if not explicit_refs:
        for order, page_index in enumerate(image_unique_pages):
            source = page_sources_by_index.get(page_index)
            if page_source_has_image(source):
                candidates.append(
                    build_trial_candidate(record, record_index, page_index, source, "image_top_10_first_available", 1, order)
                )
                break

    for order, page_index in enumerate(retrieval_candidate_pages):
        source = page_sources_by_index.get(page_index)
        if page_source_has_image(source):
            candidates.append(
                build_trial_candidate(record, record_index, page_index, source, "retrieval_union_first_available", 2, order)
            )
            break

    return candidates


def get_valid_explicit_page_indices(record: Mapping[str, Any], extract_root: str | Path) -> List[int]:
    """Compute valid explicit page indices at runtime without storing them in stage2."""

    doc_id = str(record.get("doc_id"))
    question_constraints = parse_question_constraints(str(record.get("question", "")))
    page_count_info = infer_document_page_count(doc_id=doc_id, pdf_root=None, extract_root=extract_root)
    explicit_validation = validate_explicit_page_references_against_page_count(
        {"question_constraints": question_constraints},
        page_count_info,
    )
    return [int(page_index) for page_index in explicit_validation.get("valid_explicit_page_indices", [])]


def _candidate_route_pages(stage2: Mapping[str, Any], required_route: str | None = None) -> List[int]:
    pages = []
    for item in stage2.get("candidate_page_routes", []) or []:
        if not isinstance(item, dict) or item.get("page_index") is None:
            continue
        routes = item.get("routes", [])
        if required_route is not None and required_route not in routes:
            continue
        pages.append(int(item["page_index"]))
    return pages


def build_trial_candidate(
    record: Mapping[str, Any],
    record_index: int,
    page_index: int,
    page_source: Mapping[str, Any],
    selection_reason: str,
    priority_rank: int,
    page_order: int,
) -> Dict[str, Any]:
    return {
        "record_index": int(record_index),
        "doc_id": record.get("doc_id"),
        "question": record.get("question"),
        "page_index": int(page_index),
        "page_number_one_based": int(page_index) + 1,
        "selection_reason": selection_reason,
        "page_image_path": page_source.get("page_image_path"),
        "layout_block_ids": list(page_source.get("layout_block_ids", [])),
        "_priority_rank": int(priority_rank),
        "_page_order": int(page_order),
    }


def select_best_trial_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["_priority_rank"]),
            int(item["record_index"]),
            int(item["_page_order"]),
        ),
    )
    if not ordered:
        return None
    selected = dict(ordered[0])
    selected.pop("_priority_rank", None)
    selected.pop("_page_order", None)
    return selected


def page_source_has_image(page_source: Mapping[str, Any] | None) -> bool:
    return bool(page_source and page_source.get("has_page_image") and page_source.get("page_image_path"))


def build_single_page_trial_summary(
    provider: str,
    model_name: str,
    record_id: str,
    doc_id: str,
    page_index: int,
    raw_output_log_path: str | Path,
    artifact_store_path: str | Path,
    num_raw_artifacts: int,
    num_valid_artifacts: int,
    num_validation_issues: int,
    single_page_smoke_test_passed: bool,
) -> Dict[str, Any]:
    return {
        "step": "7B",
        "provider": provider,
        "model_name": model_name,
        "record_id": record_id,
        "doc_id": doc_id,
        "page_index": int(page_index),
        "raw_output_log_path": str(raw_output_log_path),
        "artifact_store_path": str(artifact_store_path),
        "num_raw_artifacts": int(num_raw_artifacts),
        "num_valid_artifacts": int(num_valid_artifacts),
        "num_validation_issues": int(num_validation_issues),
        "single_page_smoke_test_passed": bool(single_page_smoke_test_passed),
    }


def write_single_page_trial_summary(summary: Mapping[str, Any], output_path: str | Path) -> None:
    forbidden = sorted(key for key in SUMMARY_FORBIDDEN_FIELDS if contains_key(summary, key))
    if forbidden:
        raise ValueError(f"Summary contains forbidden fields: {forbidden}")
    write_json(dict(summary), output_path)


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(contains_key(child, key) for child in value)
    return False


def write_json(value: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
