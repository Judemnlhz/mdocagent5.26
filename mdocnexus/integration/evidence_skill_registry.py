"""Lightweight Evidence Skill Registry for guarded document QA.

The registry is a deterministic, public-input-only interface that turns existing
artifact/evidence-unit semantics into auditable evidence capabilities. It is not
a large skill tree or a dataset-specific rule set: skill names, unit types, and
edge types are document-native and bounded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from .guarded_prompt import actionable_exact_codes

SCHEMA_VERSION = "evidence_skill_registry_v1"
MAX_EVIDENCE_UNIT_TYPES = 6
MAX_EDGE_TYPES = 8

EVIDENCE_UNIT_TYPES = [
    "text_span",
    "table_cell",
    "numeric_fact",
    "key_value",
    "caption",
    "code_name_pair",
]

DOCUMENT_EDGE_TYPES = [
    "contains",
    "same_page",
    "same_table",
    "row_of",
    "column_of",
    "caption_of",
    "nearby",
    "code_maps_to",
]

DATASET_NAME_MARKERS = {"mmlb", "mmlongbench", "ldu", "ptab", "ptext", "feta", "fetatab"}


@dataclass(frozen=True)
class EvidenceSkill:
    name: str
    applies_if: str
    accepted_unit_types: tuple[str, ...]
    required_fields: tuple[str, ...]
    guard_rule: str
    capsule_render_policy: str
    answer_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REGISTRY: tuple[EvidenceSkill, ...] = (
    EvidenceSkill(
        name="exact_code_lookup",
        applies_if="profile.requires_exact_code_selection and profile.codes contains actionable exact code literals",
        accepted_unit_types=("code_name_pair", "table_cell", "key_value", "text_span"),
        required_fields=("artifact_id", "page_index", "content", "source_anchored", "exact_code_match"),
        guard_rule="exact_code_absence_guard",
        capsule_render_policy="render_required_code_value_or_missing_code",
        answer_policy="answer_only_if_exact_code_value_pair_supports_it",
    ),
    EvidenceSkill(
        name="key_value_lookup",
        applies_if="question asks for named field/value/entity and does not require exact code or computation",
        accepted_unit_types=("key_value", "table_cell", "text_span"),
        required_fields=("artifact_id", "page_index", "content", "source_anchored", "key_or_label"),
        guard_rule="artifact_dimension_support_guard",
        capsule_render_policy="render_key_value_with_locator",
        answer_policy="cite_visible_support_or_refuse",
    ),
    EvidenceSkill(
        name="table_numeric_lookup",
        applies_if="profile.is_numeric_or_table_question and no computation operands are required",
        accepted_unit_types=("table_cell", "numeric_fact", "key_value"),
        required_fields=("artifact_id", "page_index", "content", "source_anchored", "metric_or_column", "value_text"),
        guard_rule="artifact_dimension_support_guard",
        capsule_render_policy="render_metric_value_rows_with_locator",
        answer_policy="cite_visible_support_or_refuse",
    ),
    EvidenceSkill(
        name="numeric_computation",
        applies_if="profile.is_computation_question and profile.required_operands is non-empty",
        accepted_unit_types=("numeric_fact", "table_cell"),
        required_fields=("artifact_id", "page_index", "content", "source_anchored", "operand", "value_text"),
        guard_rule="operand_completeness_guard",
        capsule_render_policy="render_operand_set_or_missing_operands",
        answer_policy="calculate_only_from_cited_operands",
    ),
    EvidenceSkill(
        name="figure_caption_grounding",
        applies_if="question asks about figure, chart, image, caption, or visual content",
        accepted_unit_types=("caption", "text_span"),
        required_fields=("artifact_id", "page_index", "content", "source_anchored"),
        guard_rule="artifact_dimension_support_guard",
        capsule_render_policy="render_caption_or_visual_grounding_with_locator",
        answer_policy="cite_visible_support_or_refuse",
    ),
    EvidenceSkill(
        name="text_span_grounding",
        applies_if="general fact lookup or fallback when no specialized evidence skill applies",
        accepted_unit_types=("text_span", "key_value", "caption"),
        required_fields=("artifact_id", "page_index", "content", "source_anchored"),
        guard_rule="no_relevant_artifact_guard",
        capsule_render_policy="render_compact_text_span_with_locator",
        answer_policy="use_page_evidence_or_refuse",
    ),
)


def registry_contract() -> dict[str, Any]:
    skills = [skill.to_dict() for skill in REGISTRY]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_unit_types": list(EVIDENCE_UNIT_TYPES),
        "document_edge_types": list(DOCUMENT_EDGE_TYPES),
        "skills": skills,
        "boundaries": {
            "no_provider_calls": True,
            "not_prediction_or_eval": True,
            "not_full_qa": True,
            "not_official_score": True,
            "dataset_agnostic": True,
            "not_large_skill_tree": True,
            "not_global_knowledge_graph": True,
        },
    }


def validate_registry_contract(contract: Mapping[str, Any] | None = None) -> list[str]:
    contract = registry_contract() if contract is None else contract
    failures = []
    unit_types = list(contract.get("evidence_unit_types") or [])
    edge_types = list(contract.get("document_edge_types") or [])
    skills = list(contract.get("skills") or [])
    if len(unit_types) > MAX_EVIDENCE_UNIT_TYPES:
        failures.append("too_many_evidence_unit_types")
    if len(edge_types) > MAX_EDGE_TYPES:
        failures.append("too_many_edge_types")
    if len(set(unit_types)) != len(unit_types):
        failures.append("duplicate_evidence_unit_types")
    if len(set(edge_types)) != len(edge_types):
        failures.append("duplicate_document_edge_types")
    skill_names = [str(skill.get("name") or "") for skill in skills]
    if len(set(skill_names)) != len(skill_names):
        failures.append("duplicate_skill_names")
    for skill in skills:
        name = str(skill.get("name") or "")
        if not name:
            failures.append("skill_missing_name")
        if contains_dataset_marker(name):
            failures.append(f"dataset_specific_skill_name:{name}")
        for field in ["applies_if", "accepted_unit_types", "required_fields", "guard_rule", "capsule_render_policy", "answer_policy"]:
            if not skill.get(field):
                failures.append(f"skill_missing_{field}:{name}")
        for unit_type in skill.get("accepted_unit_types") or []:
            if unit_type not in unit_types:
                failures.append(f"skill_unknown_unit_type:{name}:{unit_type}")
    return sorted(set(failures))


def activated_skills(profile: Mapping[str, Any], question: str = "") -> list[EvidenceSkill]:
    q = str(question or "").lower()
    rows = []
    if profile.get("requires_exact_code_selection") and actionable_exact_codes(list(profile.get("codes") or [])):
        rows.append(skill_by_name("exact_code_lookup"))
    if profile.get("is_computation_question") and profile.get("required_operands"):
        rows.append(skill_by_name("numeric_computation"))
    if profile.get("is_numeric_or_table_question") and not rows:
        rows.append(skill_by_name("table_numeric_lookup"))
    if any(term in q for term in ["figure", "fig", "chart", "image", "caption", "visual"]):
        rows.append(skill_by_name("figure_caption_grounding"))
    if any(term in q for term in ["what", "which", "who", "where", "name", "value"]) and not rows:
        rows.append(skill_by_name("key_value_lookup"))
    if not rows:
        rows.append(skill_by_name("text_span_grounding"))
    return dedupe_skills(rows)


def build_skill_trace(
    profile: Mapping[str, Any],
    question: str,
    selection: Mapping[str, Any],
    scored_artifacts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scored_artifacts = [] if scored_artifacts is None else scored_artifacts
    skills = activated_skills(profile, question)
    guard_decision = str(selection.get("guard_decision") or "")
    selected = list(selection.get("selected_artifacts") or [])
    traces = []
    for skill in skills:
        accepted = [row for row in scored_artifacts if str(row.get("artifact_type") or "") in skill.accepted_unit_types]
        matched_selected = [row for row in selected if str(row.get("artifact_type") or "") in skill.accepted_unit_types]
        missing = missing_requirements(skill, profile, selection, matched_selected)
        traces.append({
            "skill": skill.name,
            "activated": True,
            "applies_if": skill.applies_if,
            "accepted_unit_types": list(skill.accepted_unit_types),
            "required_fields": list(skill.required_fields),
            "guard_rule": skill.guard_rule,
            "capsule_render_policy": skill.capsule_render_policy,
            "answer_policy": skill.answer_policy,
            "candidate_count": len(accepted),
            "selected_count": len(matched_selected),
            "selected_artifact_ids": [row.get("artifact_id") for row in matched_selected],
            "missing_requirements": missing,
            "guard_decision": guard_decision,
        })
    return {
        "schema_version": "evidence_skill_trace_v1",
        "activated_skill_names": [skill.name for skill in skills],
        "guard_decision": guard_decision,
        "answer_policy": selection.get("answer_policy"),
        "traces": traces,
        "boundary": {
            "no_provider_calls": True,
            "not_prediction_or_eval": True,
            "not_full_qa": True,
            "not_official_score": True,
        },
    }


def missing_requirements(skill: EvidenceSkill, profile: Mapping[str, Any], selection: Mapping[str, Any], selected: list[Mapping[str, Any]]) -> list[str]:
    missing = []
    guard_decision = str(selection.get("guard_decision") or "")
    if skill.name == "exact_code_lookup":
        selected_codes = sorted({code for row in selected for code in row.get("exact_code_matches", [])})
        for code in actionable_exact_codes(list(profile.get("codes") or [])):
            if code not in selected_codes:
                missing.append(f"exact_code:{code}")
    if skill.name == "numeric_computation":
        covered = sorted({operand for row in selected for operand in row.get("operand_hits", [])})
        for operand in profile.get("required_operands") or []:
            if operand not in covered:
                missing.append(f"operand:{operand}")
    if guard_decision in {"artifact_dimension_support_guard", "no_relevant_artifact_guard"} and not selected:
        missing.append("selected_supporting_artifact")
    return sorted(set(missing))


def render_evidence_capsule(
    question: str,
    profile: Mapping[str, Any],
    selection: Mapping[str, Any],
    scored_artifacts: list[Mapping[str, Any]] | None = None,
    max_units: int = 4,
    include_guard_trace: bool = True,
    max_chars: int = 180,
) -> dict[str, Any]:
    """Render a compact, auditable evidence capsule from selected/scored units."""
    scored_artifacts = [] if scored_artifacts is None else scored_artifacts
    max_units = max(0, int(max_units))
    trace = build_skill_trace(profile, question, selection, scored_artifacts)
    selected = list(selection.get("selected_artifacts") or [])
    if selected:
        unit_rows = selected[:max_units]
        source = "selected_artifacts"
    else:
        accepted_types = {unit for item in trace["traces"] for unit in item.get("accepted_unit_types", [])}
        candidates = [row for row in scored_artifacts if str(row.get("artifact_type") or "") in accepted_types]
        candidates = sorted(candidates, key=lambda row: (-float(row.get("selection_score", 0.0)), int(row.get("page_index", 0)), str(row.get("artifact_id") or "")))
        unit_rows = candidates[:max_units]
        source = "top_accepted_candidates"
    lines = [
        "[Evidence Capsule]",
        f"Skills: {', '.join(trace['activated_skill_names'])}",
        f"Guard: {trace.get('guard_decision')}; policy: {trace.get('answer_policy')}",
    ]
    if unit_rows:
        lines.append("Evidence:")
        for row in unit_rows:
            lines.append(render_capsule_unit(row, max_chars=max_chars))
    else:
        lines.append("Evidence: none selected")
    missing = sorted({item for skill_trace in trace["traces"] for item in skill_trace.get("missing_requirements", [])})
    if missing:
        lines.append("Missing: " + ", ".join(missing))
    if include_guard_trace:
        for skill_trace in trace["traces"]:
            lines.append(
                "Trace: "
                + f"skill={skill_trace['skill']}; candidates={skill_trace['candidate_count']}; "
                + f"selected={skill_trace['selected_count']}; guard={skill_trace['guard_rule']}"
            )
    text = "\n".join(lines).strip() + "\n"
    return {
        "schema_version": "evidence_capsule_v1",
        "text": text,
        "token_estimate": estimate_tokens(text),
        "unit_count": len(unit_rows),
        "unit_source": source,
        "include_guard_trace": bool(include_guard_trace),
        "missing_requirements": missing,
        "activated_skill_names": trace["activated_skill_names"],
        "guard_decision": trace.get("guard_decision"),
        "answer_policy": trace.get("answer_policy"),
        "selected_artifact_ids": [row.get("artifact_id") for row in unit_rows],
        "boundary": {
            "no_provider_calls": True,
            "not_prediction_or_eval": True,
            "not_full_qa": True,
            "not_official_score": True,
        },
    }


def render_capsule_unit(row: Mapping[str, Any], max_chars: int = 180) -> str:
    normalized = row.get("normalized_content") if isinstance(row.get("normalized_content"), Mapping) else {}
    labels = []
    for key in ["row_label", "row_header", "column_label", "column_header", "metric_name", "value_text", "unit", "normalized_value"]:
        if normalized.get(key) not in (None, ""):
            labels.append(f"{key}={normalized.get(key)}")
    exact = row.get("exact_code_matches") or []
    operands = row.get("operand_hits") or []
    parts = [
        f"- id={row.get('artifact_id')}",
        f"page={row.get('page_index')}",
        f"type={row.get('artifact_type')}",
    ]
    if exact:
        parts.append("codes=" + ",".join(str(item) for item in exact))
    if operands:
        parts.append("operands=" + ",".join(str(item) for item in operands))
    if labels:
        parts.append("fields=" + ";".join(labels[:5]))
    preview = compact_text(str(row.get("content_preview") or row.get("content") or ""), max_chars=max_chars)
    if preview:
        parts.append("text=" + preview)
    return " | ".join(parts)


def flat_artifact_context(scored_artifacts: list[Mapping[str, Any]], max_units: int = 8, max_chars: int = 240) -> str:
    rows = sorted(scored_artifacts, key=lambda row: (-float(row.get("selection_score", 0.0)), int(row.get("page_index", 0)), str(row.get("artifact_id") or "")))[: max(0, int(max_units))]
    lines = ["[Flat Artifact Context]"]
    for row in rows:
        lines.append(render_capsule_unit(row, max_chars=max_chars))
    return "\n".join(lines).strip() + "\n"


def raw_page_context(page_contexts: list[Mapping[str, Any]], max_chars_per_page: int = 2200) -> str:
    lines = ["[Raw Page Context]"]
    for ctx in page_contexts:
        text = compact_text(str(ctx.get("text_preview") or ""), max_chars=max_chars_per_page)
        lines.append(f"Page {ctx.get('page_index')} ({'present' if ctx.get('exists') else 'missing'}): {text}")
    return "\n".join(lines).strip() + "\n"


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", str(text or "")))


def compact_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    max_chars = max(0, int(max_chars))
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def skill_by_name(name: str) -> EvidenceSkill:
    for skill in REGISTRY:
        if skill.name == name:
            return skill
    raise KeyError(name)


def dedupe_skills(skills: list[EvidenceSkill]) -> list[EvidenceSkill]:
    rows = []
    seen = set()
    for skill in skills:
        if skill.name in seen:
            continue
        seen.add(skill.name)
        rows.append(skill)
    return rows


def contains_dataset_marker(value: str) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in DATASET_NAME_MARKERS)
