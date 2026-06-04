"""Evidence-demand parser scaffold for guarded artifact selection.

The parser is intentionally narrow: it converts a public question into a
structured evidence-demand profile. It does not answer the question, inspect
artifact contents, call providers, evaluate predictions, or use gold fields.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .guarded_prompt import CODE_PATTERN, build_question_profile, forbidden_public_fields, normalize, question_tokens

SCHEMA_VERSION = "evidence_demand_parser_v1"
ALLOWED_ANSWER_TYPES = {
    "numeric_comparison",
    "metadata_lookup",
    "table_lookup",
    "visual_caption",
    "computation",
    "not_answerable_sensitive",
    "general_fact_lookup",
}


def build_evidence_demand_prompt(question: str) -> str:
    return "\n".join([
        "[Evidence-demand parser]",
        "Parse the question into evidence requirements for artifact/page selection.",
        "Do not answer the question. Do not infer a final answer.",
        "Use only the question text. Do not assume hidden gold answers or evidence pages.",
        "Return one JSON object only, with this schema:",
        "{",
        '  "answer_type": "numeric_comparison | metadata_lookup | table_lookup | visual_caption | computation | not_answerable_sensitive | general_fact_lookup",',
        '  "required_entities": ["entity names or groups that evidence must mention"],',
        '  "required_metrics": ["metrics, columns, fields, chart/table concepts evidence must mention"],',
        '  "required_values_or_codes": ["literal years, codes, labels, values that must match exactly when present"],',
        '  "required_operands": ["operands needed for computation questions"],',
        '  "evidence_dimensions": [{"dimension": "stable_snake_case", "label": "human label", "aliases": ["surface forms"]}],',
        '  "min_numeric_values": 0,',
        '  "requires_exact_code_selection": false,',
        '  "is_document_metadata_lookup": false,',
        '  "is_computation_question": false,',
        '  "is_numeric_or_table_question": false,',
        '  "answer_policy": "cite_visible_support_or_refuse"',
        "}",
        "Keep evidence_dimensions compact and necessary. Include years/codes in required_values_or_codes.",
        f"Question: {question}",
    ]).strip() + "\n"


def parse_evidence_demand_response(text: str) -> dict[str, Any]:
    parsed = json.loads(extract_json_object(text))
    if not isinstance(parsed, dict):
        raise ValueError("Evidence-demand parser response must be a JSON object")
    return normalize_evidence_demand(parsed)


def normalize_evidence_demand(value: Mapping[str, Any]) -> dict[str, Any]:
    answer_type = str(value.get("answer_type") or "general_fact_lookup").strip().lower()
    if answer_type not in ALLOWED_ANSWER_TYPES:
        answer_type = "general_fact_lookup"
    demand = {
        "schema_version": SCHEMA_VERSION,
        "answer_type": answer_type,
        "required_entities": clean_string_list(value.get("required_entities")),
        "required_metrics": clean_string_list(value.get("required_metrics")),
        "required_values_or_codes": clean_string_list(value.get("required_values_or_codes")),
        "required_operands": clean_string_list(value.get("required_operands")),
        "evidence_dimensions": normalize_dimensions(value.get("evidence_dimensions")),
        "min_numeric_values": safe_int(value.get("min_numeric_values"), 0),
        "requires_exact_code_selection": bool(value.get("requires_exact_code_selection")),
        "is_document_metadata_lookup": bool(value.get("is_document_metadata_lookup")) or answer_type == "metadata_lookup",
        "is_computation_question": bool(value.get("is_computation_question")) or answer_type in {"computation", "numeric_comparison"},
        "is_numeric_or_table_question": bool(value.get("is_numeric_or_table_question")) or answer_type in {"numeric_comparison", "table_lookup", "computation"},
        "answer_policy": "cite_visible_support_or_refuse",
        "strictly_public_question_only": True,
        "not_answer_generation": True,
    }
    literal_text = " ".join(demand["required_values_or_codes"])
    if CODE_PATTERN.search(literal_text):
        demand["requires_exact_code_selection"] = True
    if demand["is_computation_question"] and demand["min_numeric_values"] < 2:
        demand["min_numeric_values"] = 2
    return demand


def merge_evidence_demand_profile(question: str, demand: Mapping[str, Any]) -> dict[str, Any]:
    profile = build_question_profile(question)
    normalized = normalize_evidence_demand(demand)
    extra_text = " ".join(
        list(normalized["required_entities"])
        + list(normalized["required_metrics"])
        + list(normalized["required_values_or_codes"])
        + [dimension.get("label", "") for dimension in normalized["evidence_dimensions"]]
        + [" ".join(dimension.get("aliases") or []) for dimension in normalized["evidence_dimensions"]]
    )
    tokens = sorted(set(profile.get("tokens") or []) | question_tokens(extra_text))
    codes = sorted(set(profile.get("codes") or []) | set(CODE_PATTERN.findall(extra_text)))
    numbers = sorted(set(profile.get("numbers") or []) | set(re.findall(r"[-+]?\d+(?:\.\d+)?", extra_text)))
    requirements = profile.get("evidence_requirements") if isinstance(profile.get("evidence_requirements"), Mapping) else {}
    demand_dimensions = list(normalized["evidence_dimensions"])
    if demand_dimensions:
        requirements = {
            "dimensions": demand_dimensions,
            "min_numeric_values": max(int(requirements.get("min_numeric_values") or 0), int(normalized["min_numeric_values"] or 0)),
            "strictly_public_question_only": True,
            "requires_artifact_ids_for_citation": True,
            "source": "llm_evidence_demand_parser",
        }
    profile.update({
        "tokens": tokens,
        "codes": codes,
        "numbers": numbers,
        "evidence_requirements": requirements,
        "is_numeric_or_table_question": bool(profile.get("is_numeric_or_table_question") or normalized["is_numeric_or_table_question"]),
        "is_document_metadata_lookup": bool(profile.get("is_document_metadata_lookup") or normalized["is_document_metadata_lookup"]),
        "is_computation_question": bool(profile.get("is_computation_question") or normalized["is_computation_question"]),
        "requires_exact_code_selection": bool(profile.get("requires_exact_code_selection") or normalized["requires_exact_code_selection"]),
        "required_operands": sorted(set(profile.get("required_operands") or []) | set(normalized["required_operands"])),
        "answer_policy": "cite_visible_support_or_refuse",
        "evidence_demand_parser": normalized,
        "profile_source": "rule_profile_plus_llm_evidence_demand",
    })
    return profile


def build_rule_profile_with_parser_disabled(question: str) -> dict[str, Any]:
    profile = build_question_profile(question)
    profile["profile_source"] = "rule_profile_only_parser_disabled"
    profile["evidence_demand_parser_enabled"] = False
    return profile


def evidence_demand_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "default_enabled": False,
        "config_flag": "enable_llm_evidence_demand_parser",
        "allowed_inputs": ["question text only for parser", "public parser output merged into deterministic selector profile"],
        "forbidden_inputs": ["answer", "answers", "gold_answer", "evidence_pages", "gold_evidence", "binary_correctness", "provider answer generation"],
        "does_not_do": ["answer generation", "artifact selection by LLM", "prediction", "evaluation", "official score", "artifact lift claim"],
        "selector_boundary": "LLM parses evidence demand only; deterministic guarded selector still scores and selects public artifacts.",
    }


def validate_public_parser_payload(payload: Mapping[str, Any]) -> list[str]:
    return forbidden_public_fields(dict(payload))


def extract_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in evidence-demand parser response")
    return raw[start : end + 1]


def normalize_dimensions(value: Any) -> list[dict[str, Any]]:
    rows = []
    if not isinstance(value, list):
        return rows
    seen = set()
    for item in value:
        if isinstance(item, str):
            label = item.strip()
            dimension = slug(label)
            aliases = [label] if label else []
        elif isinstance(item, Mapping):
            label = str(item.get("label") or item.get("dimension") or "").strip()
            dimension = slug(str(item.get("dimension") or label))
            aliases = clean_string_list(item.get("aliases")) or ([label] if label else [])
        else:
            continue
        if not dimension or dimension in seen:
            continue
        seen.add(dimension)
        rows.append({"dimension": dimension, "label": label or dimension, "aliases": aliases})
    return rows[:12]


def clean_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    output = []
    seen = set()
    for item in items:
        text = re.sub(r"\s+", " ", str(item).strip())
        if not text:
            continue
        key = normalize(text)
        if key in seen:
            continue
        seen.add(key)
        output.append(text[:120])
    return output[:20]


def safe_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text[:80]
