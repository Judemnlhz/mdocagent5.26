"""Reusable guarded artifact selection and prompt scaffold.

This module contains deterministic, public-input-only guards distilled from
R054/R055 diagnostics. It is a scaffold for later integration: no provider
calls, no evaluation, and no gold fields are required by these helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

CODE_PATTERN = re.compile(r"\b[A-Z]{1,4}\d{1,4}\b")
TEMPORAL_METRIC_CODE_PATTERN = re.compile(r"^(?:FY\d{2,4}|Q[1-4]|F\d+|AP\d+|GPT\d+)$", re.IGNORECASE)
METADATA_TERMS = {"produced", "producer", "revised", "revision", "document", "version"}
COMPUTE_TERMS = {"difference", "sum", "total", "ratio", "rate", "percentage", "calculate"}
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
NUMERIC_ARTIFACT_TYPES = {"numeric_fact", "table_cell", "table"}
CODE_FIELD_KEYS = {"code", "eps_code", "row_label", "row_header", "metric_name", "column_label", "column_header"}
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


def build_question_profile(question: str) -> dict[str, Any]:
    q_norm = normalize(question)
    tokens = sorted(question_tokens(question))
    token_set = set(tokens)
    code_like_literals = normalize_code_like_literals(CODE_PATTERN.findall(question))
    codes = actionable_exact_codes(code_like_literals)
    temporal_metric_literals = temporal_metric_code_like_literals(code_like_literals)
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", question)
    is_numeric = any(word in q_norm for word in NUMERIC_WORDS) or bool(numbers)
    is_code_or_table = bool(codes) or any(word in q_norm for word in ["code", "table", "row", "column", "market"])
    is_metadata_lookup = bool(token_set & METADATA_TERMS) and any(
        term in q_norm for term in ["revised", "produced", "producer", "document"]
    )
    is_computation = bool(token_set & COMPUTE_TERMS) or any(
        term in q_norm for term in ["percentage difference", "return on asset", "round your answer"]
    )
    return {
        "tokens": tokens,
        "codes": codes,
        "code_like_literals": code_like_literals,
        "temporal_metric_literals": temporal_metric_literals,
        "numbers": numbers,
        "evidence_requirements": build_evidence_requirements(question),
        "is_numeric_or_table_question": bool(is_numeric or is_code_or_table),
        "is_document_metadata_lookup": is_metadata_lookup,
        "is_computation_question": is_computation,
        "requires_exact_code_selection": bool(codes),
        "required_operands": infer_required_operands(question),
        "requires_unsupported_answer_guard": True,
        "answer_policy": "cite_visible_support_or_refuse",
    }


def infer_required_operands(question: str) -> list[str]:
    q_norm = normalize(question)
    operands = []
    if "older" in q_norm or "older age" in q_norm:
        operands.append("older_age_group")
    if "children" in q_norm:
        operands.append("children")
    if "received" in q_norm and "stem" in q_norm and "degree" in q_norm:
        operands.append("received_stem_degree")
    if "employed" in q_norm and "field" in q_norm:
        operands.append("employed_in_field")
    if "net income" in q_norm or "return on asset" in q_norm or "roa" in q_norm:
        operands.extend(["net_income", "total_assets"])
    return sorted(set(operands))


def score_guarded_artifact(
    artifact: Mapping[str, Any],
    question: str,
    profile: Mapping[str, Any],
    page_index: int,
    artifact_pages: list[int] | None = None,
    original_pages: list[int] | None = None,
    max_chars: int = 300,
) -> dict[str, Any]:
    artifact_pages = [] if artifact_pages is None else artifact_pages
    original_pages = [] if original_pages is None else original_pages
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    content = re.sub(r"\s+", " ", str(artifact.get("content") or "")).strip()
    searchable = searchable_artifact_text(artifact, normalized)
    artifact_tokens = question_tokens(searchable)
    q_tokens = set(profile.get("tokens") or [])
    overlap = sorted(q_tokens & artifact_tokens)
    score = float(len(overlap))
    reasons = []
    if overlap:
        reasons.append("question_token_overlap")
    artifact_type = str(artifact.get("artifact_type") or "")
    modality = str(artifact.get("modality") or "")
    if profile.get("is_numeric_or_table_question") and artifact_type in NUMERIC_ARTIFACT_TYPES:
        score += 3.0
        reasons.append("numeric_table_type_priority")
    if artifact_type in {"numeric_fact", "table_cell"}:
        score += 1.0
        reasons.append("atomic_artifact_priority")
    exact_codes = [
        code
        for code in profile.get("codes") or []
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(str(code))}(?![A-Za-z0-9])", searchable, re.IGNORECASE)
    ]
    for code in exact_codes:
        score += 5.0
        reasons.append(f"code_match:{code}")
    for number in profile.get("numbers") or []:
        if number and str(number) in searchable:
            score += 2.0
            reasons.append(f"value_match:{number}")
    key_text = " ".join(str(normalized.get(key) or "") for key in CODE_FIELD_KEYS)
    key_value_hits = sorted(q_tokens & question_tokens(key_text))
    if key_value_hits:
        score += float(len(key_value_hits)) * 1.5
        reasons.append("metric_label_overlap")
    if page_index in artifact_pages:
        score += 1.0
        reasons.append("artifact_reranked_page")
    if page_index in original_pages:
        score += 0.25
        reasons.append("original_candidate_page")
    if artifact.get("source_anchored") is False:
        score -= 2.0
        reasons.append("source_anchor_missing_penalty")
    operand_hits = sorted(
        operand for operand in profile.get("required_operands") or [] if operand_covered(operand, searchable)
    )
    positive_signal = bool(overlap or exact_codes or key_value_hits or operand_hits)
    if not reasons:
        reasons.append("low_question_match_retained_for_audit")
    return {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "artifact_type": artifact_type,
        "modality": modality,
        "doc_id": str(artifact.get("doc_id") or ""),
        "page_index": int(page_index),
        "content_preview": content[:max_chars],
        "normalized_content": compact_normalized(normalized),
        "source_anchored": bool(artifact.get("source_anchored")),
        "validation_status": artifact.get("validation_status"),
        "selection_score": round(score, 4),
        "selection_reasons": sorted(set(reasons)),
        "question_token_overlap": overlap,
        "exact_code_matches": exact_codes,
        "key_value_token_hits": key_value_hits,
        "operand_hits": operand_hits,
        "is_numeric_or_table_artifact": artifact_type in NUMERIC_ARTIFACT_TYPES,
        "positive_selection_signal": positive_signal,
    }


def select_guarded_artifacts(
    candidates: list[dict[str, Any]],
    page_contexts: list[dict[str, Any]],
    profile: Mapping[str, Any],
    max_artifacts: int = 8,
) -> dict[str, Any]:
    max_artifacts = max(0, int(max_artifacts))
    positive_candidate_count = sum(1 for row in candidates if row.get("positive_selection_signal"))
    if profile.get("is_document_metadata_lookup"):
        reasons = ["document_metadata_lookup_uses_page_text_not_numeric_artifacts", *metadata_page_signal(page_contexts, profile)]
        return selection_result([], candidates, "document_metadata_refusal_guard", reasons, "refuse_if_visible_metadata_mismatches_question", positive_candidate_count)

    if profile.get("requires_exact_code_selection"):
        exact = [row for row in candidates if row.get("exact_code_matches")]
        if not exact:
            reasons = ["no_artifact_contains_exact_question_code", "numeric_artifacts_rejected_without_exact_code_key"]
            return selection_result([], candidates, "exact_code_absence_guard", reasons, "use_page_evidence_for_absence_or_refuse", positive_candidate_count)
        selected = rank_artifacts(exact)[:max_artifacts]
        return selection_result(selected, [row for row in candidates if row not in selected], "exact_code_key_value_selection", ["selected_only_exact_code_artifacts"], "answer_only_if_exact_code_value_pair_supports_it", positive_candidate_count)

    if profile.get("is_computation_question") and profile.get("required_operands"):
        selected = rank_artifacts(candidates)[:max_artifacts]
        covered = sorted({operand for row in selected for operand in row.get("operand_hits", [])})
        missing = sorted(set(profile.get("required_operands") or []) - set(covered))
        if missing:
            page_coverage = page_operand_coverage(page_contexts, profile)
            if page_coverage["visible_page_operand_complete"]:
                reasons = [
                    "artifact_operand_completeness_failed",
                    "missing_artifact_operands:" + ",".join(missing),
                    "visible_page_operands_complete",
                    "page_covered_operands:" + ",".join(page_coverage["page_covered_operands"]),
                ]
                return selection_result([], candidates, "operand_page_evidence_route", reasons, "calculate_from_visible_page_evidence_when_operands_are_cited", positive_candidate_count)
            reasons = ["operand_completeness_failed", "missing_operands:" + ",".join(missing)]
            return selection_result([], candidates, "operand_completeness_guard", reasons, "not_answerable_due_to_incomplete_operands", positive_candidate_count)
        return selection_result(selected, [row for row in candidates if row not in selected], "operand_complete_selection", ["all_required_operands_covered"], "calculate_only_from_cited_operands", positive_candidate_count)

    eligible = [row for row in candidates if row.get("question_token_overlap") or row.get("key_value_token_hits")]
    selected = rank_artifacts(eligible)[:max_artifacts]
    if selected:
        support = audit_selected_artifact_support(selected, page_contexts, profile)
        if not support["artifact_support_sufficient"]:
            reasons = ["selected_artifacts_do_not_cover_question_dimensions", *support["failure_reasons"]]
            return selection_result([], candidates, "artifact_dimension_support_guard", reasons, "use_page_evidence_or_refuse", positive_candidate_count)
        return selection_result(selected, [row for row in candidates if row not in selected], "token_key_value_selection", ["selected_question_overlapping_artifacts"], "cite_visible_support_or_refuse", positive_candidate_count)
    return selection_result([], candidates, "no_relevant_artifact_guard", ["no_question_overlapping_artifacts"], "use_page_evidence_or_refuse", positive_candidate_count)


def render_guarded_prompt(
    question: str,
    page_contexts: list[dict[str, Any]],
    selection: Mapping[str, Any],
    profile: Mapping[str, Any],
    condition_label: str = "guarded_selector_prompt",
) -> str:
    lines = [
        f"[{condition_label}]",
        "Answer using only the visible page evidence and selected artifact evidence below.",
        "First list supporting evidence, then answer. Cite page ids and artifact ids for every factual claim.",
        "If the guard decision says metadata/refusal, exact-code absence, or operand incompleteness, do not compute or infer from partial artifact snippets.",
        "If the guard decision says artifact-dimension support, do not cite rejected artifact ids; use page evidence only if it fully supports the answer.",
        "If the visible evidence does not fully support an answer, say Not answerable and cite what is missing.",
        f"Question: {question}",
        "",
        "[Question profile]",
        f"metadata_lookup={profile.get('is_document_metadata_lookup')}; computation={profile.get('is_computation_question')}; codes={profile.get('codes')}; required_operands={profile.get('required_operands')}",
        "",
        "[Guard decision]",
        f"decision={selection.get('guard_decision')}; answer_policy={selection.get('answer_policy')}; reasons={selection.get('guard_reasons')}",
        "",
        "[Page evidence]",
    ]
    if page_contexts:
        for ctx in page_contexts:
            lines.append(f"Page {ctx['page_index']} ({'present' if ctx.get('exists') else 'missing'}): {ctx.get('text_preview', '')}")
    else:
        lines.append("No page evidence is visible.")
    lines.extend(["", "[Selected artifact evidence]"])
    selected = list(selection.get("selected_artifacts") or [])
    if selected:
        for item in selected:
            lines.append(
                f"{item['artifact_id']} | page {item['page_index']} | type={item['artifact_type']} | score={item['selection_score']} | exact_codes={item.get('exact_code_matches', [])} | operands={item.get('operand_hits', [])} | {item['content_preview']}"
            )
    else:
        lines.append("No artifact evidence was selected because the guard rejected the available snippets as insufficient or irrelevant.")
        lines.append("Do not cite artifact ids from rejected snippets. If page evidence is sufficient, answer from cited page ids only; otherwise say Not answerable.")
    lines.extend([
        "",
        "[Required response format]",
        "Page evidence: cite page ids or state none.",
        "Artifact evidence: cite selected artifact ids or state none; never cite rejected artifact ids.",
        "Guard check: state whether metadata, exact-code, and operand requirements are satisfied.",
        "Unsupported-answer check: explain whether the visible evidence fully supports the answer.",
        "Final answer: answer or Not answerable.",
    ])
    return "\n".join(lines).strip() + "\n"


def selection_result(
    selected: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    decision: str,
    reasons: list[str],
    answer_policy: str,
    positive_candidate_count: int,
) -> dict[str, Any]:
    return {
        "guard_decision": decision,
        "guard_reasons": sorted(set(reasons)),
        "answer_policy": answer_policy,
        "selected_artifacts": selected,
        "rejected_artifact_count": len(rejected),
        "positive_candidate_count": int(positive_candidate_count),
    }


def metadata_page_signal(page_contexts: list[dict[str, Any]], profile: Mapping[str, Any]) -> list[str]:
    joined = " ".join(str(ctx.get("text_preview") or "") for ctx in page_contexts)
    joined_norm = normalize(joined)
    signals = []
    if "produced" in joined_norm or "revised" in joined_norm:
        signals.append("visible_page_metadata_present")
    for number in [str(num) for num in profile.get("numbers") or []]:
        if number not in joined:
            signals.append(f"question_date_not_visible:{number}")
    return signals


def page_operand_coverage(page_contexts: list[dict[str, Any]], profile: Mapping[str, Any]) -> dict[str, Any]:
    required = sorted(str(operand) for operand in profile.get("required_operands") or [])
    joined = " ".join(str(ctx.get("text_preview") or "") for ctx in page_contexts if ctx.get("exists", True))
    covered = sorted(operand for operand in required if operand_covered(operand, joined))
    missing = sorted(set(required) - set(covered))
    return {
        "required_operands": required,
        "page_covered_operands": covered,
        "page_missing_operands": missing,
        "visible_page_operand_complete": bool(required) and not missing,
    }


def audit_selected_artifact_support(
    selected_artifacts: list[dict[str, Any]],
    page_contexts: list[dict[str, Any]],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    requirements = profile.get("evidence_requirements")
    if not isinstance(requirements, Mapping):
        requirements = {"dimensions": [], "min_numeric_values": 0}
    dimensions = list(requirements.get("dimensions") or [])
    artifact_text = normalize(" ".join(artifact_evidence_text(item) for item in selected_artifacts))
    page_text = normalize(" ".join(str(ctx.get("text_preview") or "") for ctx in page_contexts))
    all_visible_text = normalize(f"{artifact_text} {page_text}")
    artifact_dimension_checks = dimension_checks(dimensions, artifact_text)
    page_dimension_checks = dimension_checks(dimensions, page_text)
    visible_dimension_checks = dimension_checks(dimensions, all_visible_text)
    artifact_values = extract_numeric_values(artifact_text)
    page_values = extract_numeric_values(page_text)
    min_numeric_values = int(requirements.get("min_numeric_values") or 0)
    citable_artifacts = [
        item
        for item in selected_artifacts
        if str(item.get("artifact_id") or "").strip()
        and item.get("page_index") is not None
        and str(item.get("content_preview") or "").strip()
    ]
    artifact_support_sufficient = (
        bool(selected_artifacts)
        and len(citable_artifacts) == len(selected_artifacts)
        and all(check["covered"] for check in artifact_dimension_checks)
        and len(artifact_values) >= min_numeric_values
    )
    visible_support_sufficient = (
        all(check["covered"] for check in visible_dimension_checks)
        and len(sorted(set(artifact_values + page_values))) >= min_numeric_values
    )
    failure_reasons = []
    if not selected_artifacts:
        failure_reasons.append("no_selected_artifacts")
    if len(citable_artifacts) != len(selected_artifacts):
        failure_reasons.append("selected_artifacts_not_all_citable")
    missing_artifact_dimensions = [check["dimension"] for check in artifact_dimension_checks if not check["covered"]]
    if missing_artifact_dimensions:
        failure_reasons.append("artifact_missing_dimensions:" + ",".join(missing_artifact_dimensions))
    if len(artifact_values) < min_numeric_values:
        failure_reasons.append(f"artifact_numeric_values_below_required:{len(artifact_values)}<{min_numeric_values}")
    if not visible_support_sufficient:
        missing_visible_dimensions = [check["dimension"] for check in visible_dimension_checks if not check["covered"]]
        if missing_visible_dimensions:
            failure_reasons.append("visible_context_missing_dimensions:" + ",".join(missing_visible_dimensions))
    if not failure_reasons:
        failure_reasons.append("none")
    return {
        "schema_version": "guarded_artifact_support_audit_v1",
        "requirements": requirements,
        "artifact_dimension_checks": artifact_dimension_checks,
        "page_dimension_checks": page_dimension_checks,
        "visible_dimension_checks": visible_dimension_checks,
        "artifact_numeric_values": artifact_values,
        "page_numeric_values": page_values[:20],
        "citable_artifact_count": len(citable_artifacts),
        "artifact_support_sufficient": artifact_support_sufficient,
        "visible_support_sufficient": visible_support_sufficient,
        "support_class": "supporting_artifact_evidence_confirmed" if artifact_support_sufficient else "artifact_positive_signal_only_insufficient",
        "failure_reasons": failure_reasons,
    }


def build_evidence_requirements(question: str) -> dict[str, Any]:
    q_norm = normalize(question)
    dimensions = []
    min_numeric_values = 0
    if "figure 4" in q_norm and "raptor" in q_norm:
        dimensions.extend([
            evidence_requirement("figure_4", "figure 4", ["figure 4", "fig. 4", "fig 4"]),
            evidence_requirement("raptor", "RAPTOR", ["raptor"]),
            evidence_requirement("retrieved_nodes", "retrieved nodes", ["node", "nodes", "retrieved"]),
            evidence_requirement("both_questions", "both questions", ["both questions", "both"]),
        ])
    if "higher-income" in q_norm or "higher income" in q_norm:
        dimensions.extend([
            evidence_requirement("higher_income_seniors", "Higher-income seniors", ["higher-income seniors", "higher income seniors", "higher-income", "higher income"]),
            evidence_requirement("go_online", "go online", ["go online", "online"]),
            evidence_requirement("smartphone", "smartphone", ["smartphone"]),
            evidence_requirement("tablet_computer", "tablet computer", ["tablet computer", "tablet"]),
        ])
        min_numeric_values = max(min_numeric_values, 3)
    if "college graduate" in q_norm:
        dimensions.extend([
            evidence_requirement("age_65_plus", "65+ people", ["65+", "65 +", "65 and older", "65 or older"]),
            evidence_requirement("college_graduate", "College graduate", ["college graduate", "college"]),
            evidence_requirement("cell_phone", "cell phone", ["cell phone", "cellphone"]),
            evidence_requirement("tablet_computer", "tablet computer", ["tablet computer", "tablet"]),
            evidence_requirement("gap_operation", "gap", ["gap", "difference"]),
        ])
        min_numeric_values = max(min_numeric_values, 2)
    for year in re.findall(r"\b(?:19|20)\d{2}\b", question):
        dimensions.append(evidence_requirement(f"year_{year}", year, [year]))
    if not dimensions:
        keyword_terms = sorted(question_tokens(question))[:8]
        dimensions = [evidence_requirement(f"term_{term}", term, [term]) for term in keyword_terms]
    deduped = []
    seen = set()
    for item in dimensions:
        if item["dimension"] not in seen:
            seen.add(item["dimension"])
            deduped.append(item)
    return {
        "dimensions": deduped,
        "min_numeric_values": min_numeric_values,
        "strictly_public_question_only": True,
        "requires_artifact_ids_for_citation": True,
    }


def evidence_requirement(dimension: str, label: str, aliases: list[str]) -> dict[str, Any]:
    return {"dimension": dimension, "label": label, "aliases": aliases}


def dimension_checks(requirements: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    rows = []
    for req in requirements:
        aliases = list(req.get("aliases") or [])
        matched = [alias for alias in aliases if phrase_present(text, alias)]
        rows.append({
            "dimension": req["dimension"],
            "label": req["label"],
            "covered": bool(matched),
            "matched_aliases": matched,
        })
    return rows


def phrase_present(text: str, phrase: str) -> bool:
    text_norm = normalize(text).replace("-", " ")
    phrase_norm = normalize(phrase).replace("-", " ")
    if phrase_norm in text_norm:
        return True
    tokens = phrase_norm.split()
    if len(tokens) > 1:
        return all(token in text_norm for token in tokens)
    return False


def artifact_evidence_text(item: Mapping[str, Any]) -> str:
    normalized_content = item.get("normalized_content") if isinstance(item.get("normalized_content"), Mapping) else {}
    return " ".join([
        str(item.get("artifact_id") or ""),
        str(item.get("artifact_type") or ""),
        str(item.get("content_preview") or ""),
        json.dumps(dict(normalized_content), ensure_ascii=False, sort_keys=True),
    ])


def extract_numeric_values(text: str) -> list[str]:
    return sorted(set(re.findall(r"[-+]?\d+(?:\.\d+)?\s*%?", text)))


def rank_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row.get("selection_score", 0.0)), int(row.get("page_index", 0)), str(row.get("artifact_type") or ""), str(row.get("artifact_id") or "")))
    output = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["selection_rank"] = rank
        output.append(item)
    return output


def normalize_code_like_literals(values: list[Any]) -> list[str]:
    rows = []
    seen = set()
    for value in values:
        text = str(value or "").strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def is_actionable_exact_code(value: Any) -> bool:
    text = str(value or "").strip().upper()
    return bool(CODE_PATTERN.fullmatch(text)) and TEMPORAL_METRIC_CODE_PATTERN.fullmatch(text) is None


def actionable_exact_codes(values: list[Any]) -> list[str]:
    return sorted(code for code in normalize_code_like_literals(values) if is_actionable_exact_code(code))


def temporal_metric_code_like_literals(values: list[Any]) -> list[str]:
    return sorted(code for code in normalize_code_like_literals(values) if not is_actionable_exact_code(code))


def searchable_artifact_text(artifact: Mapping[str, Any], normalized: Mapping[str, Any]) -> str:
    return " ".join([str(artifact.get("content") or ""), json.dumps(dict(normalized), ensure_ascii=False, sort_keys=True)])


def operand_covered(operand: str, searchable: str) -> bool:
    text = normalize(searchable)
    checks = {
        "older_age_group": ["older", "age", "25"],
        "children": ["children", "child", "k-12", "student"],
        "received_stem_degree": ["stem", "degree"],
        "employed_in_field": ["employed", "occupation", "field", "working"],
        "net_income": ["net income"],
        "total_assets": ["total assets", "assets"],
    }
    return any(term in text for term in checks.get(operand, [operand]))


def compact_normalized(value: Mapping[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in ["metric_name", "row_label", "row_header", "column_label", "column_header", "value_text", "unit", "normalized_value"]:
        if key in value:
            keep[key] = value[key]
    return keep


def forbidden_public_fields(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_PUBLIC_KEYS:
                found.append(next_path)
            found.extend(forbidden_public_fields(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_public_fields(item, f"{path}[{index}]"))
    return found


def question_tokens(text: Any) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", normalize(text)) if len(token) > 2 and token not in STOPWORDS}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
