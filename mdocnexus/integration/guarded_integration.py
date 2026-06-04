"""Integration contract for optional guarded prompt scaffolds.

The guarded prompt integration is deliberately opt-in. With the default config
it returns the adapter records unchanged. When enabled, it builds prompt-preview
artifacts from public retrieval, page text, and artifact fields only. It does
not run providers, predictions, evaluation, or official scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .guarded_prompt import (
    build_question_profile,
    forbidden_public_fields,
    render_guarded_prompt,
    score_guarded_artifact,
    select_guarded_artifacts,
    sha256,
)
from .mdocagent_adapter import canonical_json_hash

SCHEMA_VERSION = "guarded_prompt_integration_v1"
DEFAULT_MAX_ARTIFACTS = 8
DEFAULT_MAX_ARTIFACT_CHARS = 300
DEFAULT_MAX_PAGE_CHARS = 1400


@dataclass(frozen=True)
class GuardedPromptIntegrationConfig:
    enable_guarded_prompt_scaffold: bool = False
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS
    max_artifact_chars: int = DEFAULT_MAX_ARTIFACT_CHARS
    max_page_chars: int = DEFAULT_MAX_PAGE_CHARS

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GuardedPromptIntegrationConfig":
        if not value:
            return cls()
        return cls(
            enable_guarded_prompt_scaffold=bool(value.get("enable_guarded_prompt_scaffold", False)),
            max_artifacts=int(value.get("max_artifacts", DEFAULT_MAX_ARTIFACTS)),
            max_artifact_chars=int(value.get("max_artifact_chars", DEFAULT_MAX_ARTIFACT_CHARS)),
            max_page_chars=int(value.get("max_page_chars", DEFAULT_MAX_PAGE_CHARS)),
        )


def guarded_prompt_integration_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "default_enabled": False,
        "config_flag": "enable_guarded_prompt_scaffold",
        "allowed_inputs": [
            "record_index",
            "doc_id",
            "question",
            "text/image retrieval page ids and scores",
            "public page text previews",
            "public artifact fields: artifact_id, artifact_type, modality, content, normalized_content, page_index, source_anchored, validation_status",
        ],
        "forbidden_inputs": [
            "answer",
            "answers",
            "gold_answer",
            "evidence_pages",
            "evidence_sources",
            "binary_correctness",
            "gold_evidence",
            "gold_page",
            "gold_pages",
            "provider raw outputs",
            "API keys or secrets",
        ],
        "does_not_do": [
            "provider call",
            "prediction",
            "evaluation",
            "official score",
            "artifact positive-lift claim",
            "default adapter behavior change",
        ],
        "control_surface": "selection/prompt preview only; official retrieval/eval path is unchanged unless explicitly wired later",
    }


def apply_guarded_prompt_integration(
    records: list[dict[str, Any]],
    artifacts_by_page: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    page_contexts_by_doc_page: Mapping[tuple[str, int], Mapping[str, Any]],
    config: GuardedPromptIntegrationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, GuardedPromptIntegrationConfig) else GuardedPromptIntegrationConfig.from_mapping(config)
    input_hash = canonical_json_hash(records)
    if not cfg.enable_guarded_prompt_scaffold:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": False,
            "records": records,
            "records_unchanged": True,
            "input_records_sha256": input_hash,
            "output_records_sha256": canonical_json_hash(records),
            "prompt_previews": [],
            "manifest": build_manifest(cfg, input_hash, input_hash, [], []),
        }

    previews = []
    forbidden = []
    for record in records:
        preview = build_guarded_prompt_preview(record, artifacts_by_page, page_contexts_by_doc_page, cfg)
        previews.append(preview)
        forbidden.extend(preview.get("forbidden_gold_fields_present", []))
    output_hash = canonical_json_hash(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "records": records,
        "records_unchanged": True,
        "input_records_sha256": input_hash,
        "output_records_sha256": output_hash,
        "prompt_previews": previews,
        "manifest": build_manifest(cfg, input_hash, output_hash, previews, forbidden),
    }


def build_guarded_prompt_preview(
    record: Mapping[str, Any],
    artifacts_by_page: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    page_contexts_by_doc_page: Mapping[tuple[str, int], Mapping[str, Any]],
    config: GuardedPromptIntegrationConfig,
) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or "")
    question = str(record.get("question") or "")
    candidate_pages = candidate_pages_from_record(record)
    artifact_pages = list(candidate_pages)
    profile = build_question_profile(question)
    page_contexts = [public_page_context(page_contexts_by_doc_page.get((doc_id, page), {}), doc_id, page, config.max_page_chars) for page in artifact_pages]
    candidates = []
    doc_artifacts = artifacts_by_page.get(doc_id, {})
    for page in candidate_pages:
        for artifact in doc_artifacts.get(page, []):
            candidates.append(
                score_guarded_artifact(
                    public_artifact(artifact, doc_id, page),
                    question,
                    profile,
                    page,
                    artifact_pages=artifact_pages,
                    original_pages=list(candidate_pages),
                    max_chars=config.max_artifact_chars,
                )
            )
    selection = select_guarded_artifacts(candidates, page_contexts, profile, max_artifacts=config.max_artifacts)
    prompt = render_guarded_prompt(question, page_contexts, selection, profile, condition_label="guarded_prompt_integration_preview")
    payload = {
        "record_index": record.get("record_index"),
        "doc_id": doc_id,
        "question": question,
        "question_profile": profile,
        "candidate_pages": candidate_pages,
        "selection": selection,
        "page_contexts": page_contexts,
        "prompt_preview": prompt,
    }
    return {
        "schema_version": "guarded_prompt_integration_preview_v1",
        "record_index": record.get("record_index"),
        "doc_id": doc_id,
        "question": question,
        "candidate_pages": candidate_pages,
        "question_profile": profile,
        "candidate_artifact_count": len(candidates),
        "positive_candidate_count": selection["positive_candidate_count"],
        "selected_artifact_count": len(selection["selected_artifacts"]),
        "selected_artifacts": selection["selected_artifacts"],
        "guard_decision": selection["guard_decision"],
        "guard_reasons": selection["guard_reasons"],
        "answer_policy": selection["answer_policy"],
        "page_contexts": page_contexts,
        "prompt_preview": prompt,
        "prompt_preview_sha256": sha256(prompt),
        "forbidden_gold_fields_present": forbidden_public_fields(payload),
    }


def build_manifest(
    config: GuardedPromptIntegrationConfig,
    input_hash: str,
    output_hash: str,
    previews: list[dict[str, Any]],
    forbidden: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": guarded_prompt_integration_contract(),
        "enabled": config.enable_guarded_prompt_scaffold,
        "records_unchanged": input_hash == output_hash,
        "input_records_sha256": input_hash,
        "output_records_sha256": output_hash,
        "num_prompt_previews": len(previews),
        "selected_artifact_count_by_record": {str(row.get("record_index")): row.get("selected_artifact_count") for row in previews},
        "guard_decision_by_record": {str(row.get("record_index")): row.get("guard_decision") for row in previews},
        "no_gold_fields_in_public_previews": not forbidden,
        "forbidden_gold_fields_present": forbidden,
        "no_provider_calls": True,
        "no_prediction_or_eval": True,
        "no_full_qa": True,
        "not_official_score": True,
        "not_artifact_lift_claim": True,
    }


def candidate_pages_from_record(record: Mapping[str, Any]) -> list[int]:
    pages: list[int] = []
    seen = set()
    for key in sorted(record):
        if not (str(key).startswith("text-top-") or str(key).startswith("image-top-") or str(key).startswith("mix-top-")):
            continue
        if str(key).endswith("_score"):
            continue
        value = record.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            try:
                page = int(item)
            except (TypeError, ValueError):
                continue
            if page not in seen:
                seen.add(page)
                pages.append(page)
    return pages


def public_artifact(artifact: Mapping[str, Any], doc_id: str, page_index: int) -> dict[str, Any]:
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    return {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "artifact_type": str(artifact.get("artifact_type") or ""),
        "modality": str(artifact.get("modality") or ""),
        "doc_id": str(artifact.get("doc_id") or doc_id),
        "page_index": int(artifact.get("page_index", page_index)),
        "content": str(artifact.get("content") or ""),
        "normalized_content": dict(normalized),
        "source_anchored": bool(artifact.get("source_anchored")),
        "validation_status": artifact.get("validation_status"),
    }


def public_page_context(value: Mapping[str, Any], doc_id: str, page_index: int, max_chars: int) -> dict[str, Any]:
    text = ""
    if isinstance(value, Mapping):
        text = str(value.get("text_preview") or value.get("text") or "")
    return {
        "page_index": int(page_index),
        "page_id": f"{doc_id}#p{int(page_index):03d}",
        "exists": bool(text),
        "text_preview": " ".join(text.split())[:max_chars],
    }


def write_integration_outputs(result: Mapping[str, Any], output_root: str | Path) -> dict[str, str]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "guarded_prompt_integration_manifest.json"
    previews_path = root / "guarded_prompt_integration_previews.jsonl"
    manifest_path.write_text(json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with previews_path.open("w", encoding="utf-8") as handle:
        for row in result.get("prompt_previews") or []:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"manifest": str(manifest_path), "prompt_previews": str(previews_path)}