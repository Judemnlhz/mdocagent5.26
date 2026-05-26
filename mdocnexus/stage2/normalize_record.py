"""Canonical record normalization for Stage 2.

Step 1 converts raw MDocAgent retrieval records into a stable canonical record.
It does not compile evidence artifacts, call an LLM, or produce final answers.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Mapping

from .candidate_pool_builder import build_candidate_pool
from .constraint_parser import parse_question_constraints
from .field_aliases import COMPILER_EXCLUDED_FIELDS, get_source_value, require_source_value
from .retrieval_processor import deduplicate_ranked_pages, parse_sequence_field


def parse_list_field(value: Any) -> List[Any]:
    """Parse list-like source values for eval-only annotations."""

    return parse_sequence_field(value, field_name="list_field")


def make_record_id(doc_id: str, question: str) -> str:
    """Create a deterministic record id from document id and question text."""

    raw = f"{doc_id}::{question}"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    stem = os.path.basename(doc_id).removesuffix(".pdf")
    normalized_stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or "document"
    return f"{normalized_stem}__{suffix}"


def normalize_record(source_record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one raw retrieval record into the approved Stage 2 structure."""

    doc_id = str(require_source_value(source_record, "doc_id"))
    question_text = str(require_source_value(source_record, "question_text"))
    record_id = make_record_id(doc_id, question_text)

    evidence_pages_raw = parse_list_field(get_source_value(source_record, "evidence_pages", []))
    evidence_sources = parse_list_field(get_source_value(source_record, "evidence_sources", []))
    answer = get_source_value(source_record, "answer")

    text_pages = parse_sequence_field(
        get_source_value(source_record, "text_ranked_pages", []),
        field_name="text_ranked_pages",
    )
    text_scores = parse_sequence_field(
        get_source_value(source_record, "text_ranked_scores", []),
        field_name="text_ranked_scores",
    )
    image_pages = parse_sequence_field(
        get_source_value(source_record, "image_ranked_pages", []),
        field_name="image_ranked_pages",
    )
    image_scores = parse_sequence_field(
        get_source_value(source_record, "image_ranked_scores", []),
        field_name="image_ranked_scores",
    )

    text_unique = deduplicate_ranked_pages(text_pages, text_scores)
    image_unique = deduplicate_ranked_pages(image_pages, image_scores)
    question_constraints = parse_question_constraints(question_text)
    candidate_pool = build_candidate_pool(
        text_ranked_pages_unique=text_unique,
        image_ranked_pages_unique=image_unique,
        explicit_page_references=question_constraints["explicit_page_references"],
    )

    canonical_record = {
        "document": {
            "doc_id": doc_id,
            "doc_type": get_source_value(source_record, "doc_type"),
        },
        "question": {
            "text": question_text,
            "answer_format": get_source_value(source_record, "answer_format"),
        },
        "gold_annotation": {
            "answer": answer,
            "is_answerable": _infer_is_answerable(answer),
            "evidence_pages_raw": evidence_pages_raw,
            "evidence_pages_zero_based": _convert_one_based_pages_to_zero_based(evidence_pages_raw),
            "evidence_sources": evidence_sources,
            "eval_only": True,
        },
        "query_rewrites": {
            "mdocagent_retrieval_query": get_source_value(source_record, "mdocagent_retrieval_query"),
            "mdocagent_retrieval_key": get_source_value(source_record, "mdocagent_retrieval_key"),
            "qwen_retrieval_query": get_source_value(source_record, "qwen_retrieval_query"),
            "qwen_retrieval_key": get_source_value(source_record, "qwen_retrieval_key"),
        },
        "question_constraints": question_constraints,
        "retrieval": {
            "text": {
                "index_path": get_source_value(source_record, "text_index_path"),
                "ranked_pages_raw": text_pages,
                "ranked_scores_raw": text_scores,
                "ranked_pages_unique": text_unique,
            },
            "image": {
                "ranked_pages_raw": image_pages,
                "ranked_scores_raw": image_scores,
                "ranked_pages_unique": image_unique,
            },
        },
        "candidate_pool": candidate_pool,
        "compilation_plan": {
            "compile_scope": "retrieval_union_plus_explicit_page_constraints",
            "pages_to_compile": candidate_pool["required_pages_for_compilation"],
            "priority_pages": candidate_pool["explicit_constraint_pages"],
            "compilation_reasons": [
                {
                    "page_index": ref["page_index_zero_based"],
                    "reason_type": "explicit_page_reference",
                    "reason_text": ref["surface_text"],
                }
                for ref in question_constraints["explicit_page_references"]
            ],
            "excluded_fields_from_compiler": list(COMPILER_EXCLUDED_FIELDS),
        },
        "artifact_compilation": {
            "artifact_store_path": None,
            "validation_summary": None,
            "quality_gate": None,
        },
        "baseline_outputs": {
            "mdocagent": {
                "answer": get_source_value(source_record, "mdocagent_answer"),
                "binary_correctness": get_source_value(source_record, "binary_correctness"),
            },
            "eval_only": True,
        },
    }

    return {
        "record_id": record_id,
        "source_record": dict(source_record),
        "canonical_record": canonical_record,
    }


def _infer_is_answerable(answer: Any) -> bool:
    if answer is None:
        return False
    return str(answer).strip().lower() != "not answerable"


def _convert_one_based_pages_to_zero_based(evidence_pages_raw: List[Any]) -> List[int]:
    zero_based_pages: List[int] = []
    for raw_page in evidence_pages_raw:
        if isinstance(raw_page, bool):
            continue
        page_number = int(raw_page)
        if page_number >= 1:
            zero_based_pages.append(page_number - 1)
    return zero_based_pages
