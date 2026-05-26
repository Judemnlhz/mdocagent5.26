"""Prompt builders for the Stage 2 artifact compiler interface."""

from __future__ import annotations

import json
from typing import Any, Dict

from .schema_serialization import get_allowed_artifact_types


def build_artifact_compiler_system_prompt() -> str:
    """Build the system prompt for a constrained artifact compiler."""

    return "\n".join(
        [
            "You are an evidence artifact compiler.",
            "Convert one document page into typed evidence artifacts.",
            "Do not answer the question.",
            "Do not use gold answers.",
            "Do not infer unsupported facts.",
            "Every artifact must cite source_anchors from the provided layout_blocks.",
            "Artifact status must be candidate before validation.",
            "Do not generate supports / contradicts edges.",
            "Do not generate proof_trace.",
            "Do not output verified / answer_supported / proof_used.",
            "If uncertain, write uncertain_or_unreadable.",
            "Return JSON only.",
        ]
    )


def build_artifact_compiler_user_prompt(
    canonical_record: Dict[str, Any],
    page_input: Dict[str, Any],
    schema_dict: Dict[str, Any],
) -> str:
    """Build the user prompt from compiler-safe canonical fields and page input."""

    document = canonical_record["document"]
    question = canonical_record["question"]
    question_constraints = canonical_record.get("question_constraints", {})
    compilation_plan = canonical_record.get("compilation_plan", {})
    page_index = int(page_input["page_index"])
    is_explicit_page = _is_explicit_page_constraint(question_constraints, page_index)
    page_requirement = _build_page_requirement(page_input, is_explicit_page)
    page_modality_diagnosis = page_input.get("page_modality_diagnosis", {})

    prompt_payload = {
        "task": "Convert this single page into candidate evidence artifacts.",
        "document": {
            "doc_id": document["doc_id"],
            "page_index": page_index,
            "page_number_one_based": page_index + 1,
        },
        "question": {
            "text": question.get("text"),
            "answer_format": question.get("answer_format"),
        },
        "question_constraints": question_constraints,
        "compilation_plan": {
            "compile_scope": compilation_plan.get("compile_scope"),
            "priority_pages": compilation_plan.get("priority_pages", []),
            "compilation_reasons": compilation_plan.get("compilation_reasons", []),
        },
        "page_requirement": page_requirement,
        "page_modality_diagnosis": page_modality_diagnosis,
        "allowed_artifact_types": get_allowed_artifact_types(),
        "required_json_schema": schema_dict,
        "layout_blocks": page_input.get("layout_blocks", []),
        "page_text": page_input.get("page_text"),
        "artifact_coverage_instruction": [
            "If page_modality_diagnosis.requires_visual_reasoning is true and a full_page_image block exists, produce at least one visual_observation candidate artifact anchored to the full_page_image block.",
            "If page_modality_diagnosis.mentions_chart is true, produce figure and numeric_fact candidate artifacts when visible in the page.",
            "If page_modality_diagnosis.mentions_table is true, produce table and numeric_fact candidate artifacts when visible in the page.",
            "If numeric values are visible and relevant to the question, produce numeric_fact candidate artifacts.",
            "Do not answer the question.",
            "Do not infer hidden values.",
            "If the chart/table/visual detail is not readable, add uncertain_or_unreadable instead of guessing.",
        ],
        "rules": [
            "Return exactly one PageArtifactOutput JSON object.",
            "Use only source_id values present in layout_blocks.",
            "Every artifact must have validation_status set to candidate.",
            "Do not decide the final answer.",
            "Do not include gold answers, baseline outputs, or source records.",
            "Do not create supports or contradicts edges.",
            "Do not create proof_trace, verified, answer_supported, or proof_used fields.",
            "For visual questions, output candidate visual_observation artifacts only when anchored; set presence to undetermined unless explicitly visible.",
            "Return JSON only.",
        ],
    }

    return json.dumps(prompt_payload, ensure_ascii=False, indent=2)


def _is_explicit_page_constraint(question_constraints: Dict[str, Any], page_index: int) -> bool:
    for page_ref in question_constraints.get("explicit_page_references", []):
        if page_ref.get("page_index_zero_based") == page_index:
            return True
    return False


def _build_page_requirement(page_input: Dict[str, Any], is_explicit_page: bool) -> Dict[str, Any]:
    if not is_explicit_page:
        return {
            "explicit_page_constraint": False,
            "minimum_candidate_artifact": None,
        }

    block_types = {block.get("block_type") for block in page_input.get("layout_blocks", [])}
    if "full_page_image" in block_types:
        minimum_candidate_artifact = "visual_observation"
    elif "text_block" in block_types:
        minimum_candidate_artifact = "text_span"
    else:
        minimum_candidate_artifact = None

    return {
        "explicit_page_constraint": True,
        "minimum_candidate_artifact": minimum_candidate_artifact,
        "instruction": "If the required block type exists, include at least one page-level visual_observation or text_span candidate anchored to the page.",
    }
