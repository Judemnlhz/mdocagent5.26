"""Compare Stage 2 cross-document audit reports before and after refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def compare_crossdoc_audits(
    baseline_audit: str | Path,
    refined_audit: str | Path,
) -> dict:
    baseline = _load_json(baseline_audit)
    refined = _load_json(refined_audit)
    baseline_summary = _extract_summary(baseline)
    refined_summary = _extract_summary(refined)
    delta = {
        "visual_artifact_page_coverage": (
            refined_summary["visual_artifact_page_coverage"]
            - baseline_summary["visual_artifact_page_coverage"]
        ),
        "table_or_figure_artifact_page_coverage": (
            refined_summary["table_or_figure_artifact_page_coverage"]
            - baseline_summary["table_or_figure_artifact_page_coverage"]
        ),
        "numeric_fact_count": (
            refined_summary["numeric_fact_count"] - baseline_summary["numeric_fact_count"]
        ),
        "schema_valid_rate": refined_summary["schema_valid_rate"] - baseline_summary["schema_valid_rate"],
        "discard_rate": refined_summary["discard_rate"] - baseline_summary["discard_rate"],
    }
    return {
        "baseline": baseline_summary,
        "refined": refined_summary,
        "delta": delta,
        "acceptance": _build_acceptance(refined, delta),
    }


def write_refinement_comparison(report: Mapping[str, Any], output_json: str | Path) -> None:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: str | Path) -> dict:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Audit report root must be an object: {path}")
    return loaded


def _extract_summary(report: Mapping[str, Any]) -> dict:
    modality = report.get("modality_coverage", {})
    artifact_types = report.get("num_artifacts_by_type", {})
    return {
        "num_artifacts_by_type": dict(artifact_types),
        "num_artifacts_by_modality": dict(report.get("num_artifacts_by_modality", {})),
        "visual_artifact_page_coverage": float(modality.get("visual_artifact_page_coverage", 0.0)),
        "table_or_figure_artifact_page_coverage": float(
            modality.get("table_or_figure_artifact_page_coverage", 0.0)
        ),
        "numeric_fact_count": int(artifact_types.get("numeric_fact", 0)),
        "schema_valid_rate": float(report.get("schema_valid_rate", 0.0)),
        "anchoring_rate": float(report.get("anchoring_rate", 0.0)),
        "discard_rate": float(report.get("discard_rate", 1.0)),
    }


def _build_acceptance(refined: Mapping[str, Any], delta: Mapping[str, float]) -> dict:
    reasons = []
    if float(refined.get("schema_valid_rate", 0.0)) < 0.90:
        reasons.append("schema_valid_rate_below_threshold")
    if float(refined.get("anchoring_rate", 0.0)) < 0.90:
        reasons.append("anchoring_rate_below_threshold")
    if float(refined.get("discard_rate", 1.0)) > 0.20:
        reasons.append("discard_rate_above_threshold")
    if int(refined.get("forbidden_field_violations", 0)) != 0:
        reasons.append("forbidden_field_violations")
    if int(refined.get("api_key_leaks", 0)) != 0:
        reasons.append("api_key_leaks")
    issue_types = refined.get("validation_issue_types", {})
    if int(issue_types.get("source_anchor_not_found", 0)) != 0:
        reasons.append("source_anchor_not_found")
    if int(issue_types.get("provenance_source_not_found", 0)) != 0:
        reasons.append("provenance_source_not_found")

    improved = (
        delta["visual_artifact_page_coverage"] > 0
        or delta["table_or_figure_artifact_page_coverage"] > 0
        or delta["numeric_fact_count"] > 0
    )
    if not improved:
        reasons.append("prompt_schema_refinement_insufficient")

    return {
        "passed": not reasons,
        "reasons": reasons,
    }
