"""Quality report helpers for Stage 2 small-batch artifact compilation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


FORBIDDEN_FIELDS = {
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


CSV_FIELDS = [
    "record_index",
    "doc_id",
    "page_index",
    "selection_reason",
    "page_image_path",
    "num_raw_artifacts",
    "num_valid_artifacts",
    "num_validation_issues",
    "artifact_store_path",
    "raw_output_logged",
    "discard_logged",
    "passed",
]


def summarize_batch_results(page_results: list[dict]) -> dict:
    num_pages_attempted = len(page_results)
    num_api_calls = sum(1 for result in page_results if result.get("api_called"))
    num_raw_artifacts = sum(int(result.get("num_raw_artifacts", 0)) for result in page_results)
    num_valid_artifacts = sum(int(result.get("num_valid_artifacts", 0)) for result in page_results)
    num_validation_issues = sum(int(result.get("num_validation_issues", 0)) for result in page_results)
    num_artifact_stores_written = sum(1 for result in page_results if result.get("artifact_store_path"))
    forbidden_field_violations = sum(int(result.get("forbidden_field_violations", 0)) for result in page_results)
    denominator = max(1, num_raw_artifacts)
    return {
        "stage": "stage2_small_batch_artifact_compilation",
        "provider": page_results[0].get("provider") if page_results else None,
        "model_name": page_results[0].get("model_name") if page_results else None,
        "max_pages": page_results[0].get("max_pages") if page_results else 0,
        "num_pages_attempted": num_pages_attempted,
        "num_api_calls": num_api_calls,
        "num_raw_artifacts": num_raw_artifacts,
        "num_valid_artifacts": num_valid_artifacts,
        "num_validation_issues": num_validation_issues,
        "num_artifact_stores_written": num_artifact_stores_written,
        "schema_valid_rate": num_valid_artifacts / denominator,
        "anchoring_rate": num_valid_artifacts / denominator,
        "discard_rate": max(0, num_raw_artifacts - num_valid_artifacts) / denominator,
        "forbidden_field_violations": forbidden_field_violations,
        "uses_answer": False,
        "uses_evidence_pages": False,
        "uses_binary_correctness": False,
    }


def write_batch_quality_csv(page_results: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in page_results:
            writer.writerow({field: result.get(field) for field in CSV_FIELDS})


def write_batch_summary(summary: dict, path: str | Path) -> None:
    violations = [field for field in FORBIDDEN_FIELDS if contains_key(summary, field)]
    if violations:
        raise ValueError(f"Batch summary contains forbidden fields: {sorted(violations)}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(contains_key(child, key) for child in value)
    return False


def count_forbidden_fields(value: Any) -> int:
    return sum(1 for field in FORBIDDEN_FIELDS if contains_key(value, field))
