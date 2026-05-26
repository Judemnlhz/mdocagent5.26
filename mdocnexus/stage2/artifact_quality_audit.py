"""Offline quality audit for Stage 2 artifact compilation outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


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
API_KEY_FIELD = "api_key"


def load_batch_artifact_stores(artifact_store_dir: str | Path) -> list[dict]:
    """Load artifact_store JSON files in deterministic filename order."""

    root = Path(artifact_store_dir)
    stores: List[Dict[str, Any]] = []
    if not root.is_dir():
        return stores
    for path in sorted(root.glob("*.json")):
        store = json.loads(path.read_text(encoding="utf-8"))
        store["_artifact_store_path"] = str(path)
        stores.append(store)
    return stores


def audit_artifact_store(store: dict) -> dict:
    """Audit one artifact store for artifact quality and internal consistency."""

    doc_id = store.get("document", {}).get("doc_id")
    pages = store.get("pages", [])
    page_indices = [int(page.get("page_index")) for page in pages if page.get("page_index") is not None]
    artifacts = [artifact for page in pages for artifact in page.get("artifacts", [])]
    layout_block_ids = {
        block.get("block_id")
        for page in pages
        for block in page.get("layout_blocks", [])
        if block.get("block_id")
    }
    artifact_ids = [artifact.get("artifact_id") for artifact in artifacts]
    duplicate_artifact_ids = sorted(
        artifact_id for artifact_id, count in Counter(artifact_ids).items() if artifact_id and count > 1
    )

    num_missing_source_anchors = 0
    num_missing_provenance_sources = 0
    num_unanchored_artifacts = 0
    num_empty_content = 0
    page_index_consistency_passed = True
    source_anchor_consistency_passed = True
    provenance_consistency_passed = True

    for artifact in artifacts:
        artifact_page_index = artifact.get("page_index")
        if artifact_page_index not in page_indices:
            page_index_consistency_passed = False
        if not str(artifact.get("content", "")).strip():
            num_empty_content += 1
        source_anchors = artifact.get("source_anchors", [])
        if not source_anchors:
            num_unanchored_artifacts += 1
            num_missing_source_anchors += 1
            source_anchor_consistency_passed = False
        anchor_source_ids = [anchor.get("source_id") for anchor in source_anchors if isinstance(anchor, dict)]
        for source_id in anchor_source_ids:
            if source_id not in layout_block_ids:
                num_missing_source_anchors += 1
                source_anchor_consistency_passed = False
        provenance_sources = artifact.get("provenance", {}).get("sources", [])
        if not provenance_sources:
            num_missing_provenance_sources += 1
            provenance_consistency_passed = False
        for source_id in provenance_sources:
            if source_id not in layout_block_ids:
                num_missing_provenance_sources += 1
                provenance_consistency_passed = False

    num_artifacts_by_type = Counter(str(artifact.get("artifact_type")) for artifact in artifacts)
    num_artifacts_by_modality = Counter(str(artifact.get("modality")) for artifact in artifacts)
    forbidden_violations = count_forbidden_field_violations(store)

    return {
        "artifact_store_path": store.get("_artifact_store_path"),
        "doc_id": doc_id,
        "page_indices": page_indices,
        "num_pages": len(pages),
        "num_artifacts": len(artifacts),
        "num_artifacts_by_type": dict(sorted(num_artifacts_by_type.items())),
        "num_artifacts_by_modality": dict(sorted(num_artifacts_by_modality.items())),
        "num_anchored_artifacts": len(artifacts) - num_unanchored_artifacts,
        "num_unanchored_artifacts": num_unanchored_artifacts,
        "num_empty_content": num_empty_content,
        "num_missing_source_anchors": num_missing_source_anchors,
        "num_missing_provenance_sources": num_missing_provenance_sources,
        "num_forbidden_field_violations": forbidden_violations,
        "artifact_ids_unique": not duplicate_artifact_ids,
        "duplicate_artifact_ids": duplicate_artifact_ids,
        "page_index_consistency_passed": page_index_consistency_passed,
        "source_anchor_consistency_passed": source_anchor_consistency_passed,
        "provenance_consistency_passed": provenance_consistency_passed,
    }


def audit_batch_artifact_outputs(batch_dir: str | Path) -> dict:
    """Audit an existing small-batch output directory without model calls."""

    root = Path(batch_dir)
    stores = load_batch_artifact_stores(root / "artifact_stores")
    store_audits = [audit_artifact_store(store) for store in stores]
    batch_summary = read_json_if_exists(root / "reports" / "batch_summary.json")
    validation_issue_types = read_validation_issue_types(root / "discard" / "discard.jsonl")
    api_key_leaks = count_api_key_leaks(root)

    num_artifacts_by_type = Counter()
    num_artifacts_by_modality = Counter()
    for audit in store_audits:
        num_artifacts_by_type.update(audit["num_artifacts_by_type"])
        num_artifacts_by_modality.update(audit["num_artifacts_by_modality"])

    num_artifacts = sum(audit["num_artifacts"] for audit in store_audits)
    num_pages = sum(audit["num_pages"] for audit in store_audits)
    num_forbidden = sum(audit["num_forbidden_field_violations"] for audit in store_audits)
    all_artifacts_have_source_anchors = all(
        audit["num_unanchored_artifacts"] == 0 and audit["num_missing_source_anchors"] == 0
        for audit in store_audits
    )
    all_provenance_sources_resolve = all(
        audit["num_missing_provenance_sources"] == 0 and audit["provenance_consistency_passed"]
        for audit in store_audits
    )
    schema_valid_rate = float(batch_summary.get("schema_valid_rate", 0.0) or 0.0)
    anchoring_rate = float(batch_summary.get("anchoring_rate", 0.0) or 0.0)
    discard_rate = float(batch_summary.get("discard_rate", 0.0) or 0.0)
    num_validation_issues = int(batch_summary.get("num_validation_issues", sum(validation_issue_types.values())) or 0)

    quality_gate = build_quality_gate(
        num_artifact_stores=len(store_audits),
        num_artifacts=num_artifacts,
        schema_valid_rate=schema_valid_rate,
        anchoring_rate=anchoring_rate,
        discard_rate=discard_rate,
        forbidden_field_violations=num_forbidden,
        api_key_leaks=api_key_leaks,
        all_artifacts_have_source_anchors=all_artifacts_have_source_anchors,
        all_provenance_sources_resolve=all_provenance_sources_resolve,
    )

    return {
        "num_artifact_stores": len(store_audits),
        "num_documents": len({audit.get("doc_id") for audit in store_audits if audit.get("doc_id")}),
        "num_pages": num_pages,
        "num_artifacts": num_artifacts,
        "num_artifacts_by_type": dict(sorted(num_artifacts_by_type.items())),
        "num_artifacts_by_modality": dict(sorted(num_artifacts_by_modality.items())),
        "schema_valid_rate": schema_valid_rate,
        "anchoring_rate": anchoring_rate,
        "discard_rate": discard_rate,
        "num_validation_issues": num_validation_issues,
        "validation_issue_types": dict(sorted(validation_issue_types.items())),
        "num_forbidden_field_violations": num_forbidden,
        "num_api_key_leaks": api_key_leaks,
        "all_artifacts_have_source_anchors": all_artifacts_have_source_anchors,
        "all_provenance_sources_resolve": all_provenance_sources_resolve,
        "stage2_quality_gate": quality_gate,
        "artifact_store_audits": store_audits,
    }


def build_quality_gate(
    num_artifact_stores: int,
    num_artifacts: int,
    schema_valid_rate: float,
    anchoring_rate: float,
    discard_rate: float,
    forbidden_field_violations: int,
    api_key_leaks: int,
    all_artifacts_have_source_anchors: bool,
    all_provenance_sources_resolve: bool,
) -> dict:
    blocking_reasons: List[str] = []
    if num_artifact_stores <= 0:
        blocking_reasons.append("no_artifact_stores")
    if num_artifacts <= 0:
        blocking_reasons.append("no_artifacts")
    if schema_valid_rate < 0.90:
        blocking_reasons.append("schema_valid_rate_below_threshold")
    if anchoring_rate < 0.90:
        blocking_reasons.append("anchoring_rate_below_threshold")
    if discard_rate > 0.20:
        blocking_reasons.append("discard_rate_above_threshold")
    if forbidden_field_violations != 0:
        blocking_reasons.append("forbidden_field_violations")
    if api_key_leaks != 0:
        blocking_reasons.append("api_key_leaks")
    if not all_artifacts_have_source_anchors:
        blocking_reasons.append("source_anchor_consistency_failed")
    if not all_provenance_sources_resolve:
        blocking_reasons.append("provenance_consistency_failed")
    return {
        "passed": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }


def write_audit_csv(audit: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "artifact_store_path",
        "doc_id",
        "page_indices",
        "num_pages",
        "num_artifacts",
        "num_anchored_artifacts",
        "num_unanchored_artifacts",
        "num_empty_content",
        "num_missing_source_anchors",
        "num_missing_provenance_sources",
        "num_forbidden_field_violations",
        "artifact_ids_unique",
        "duplicate_artifact_ids",
        "page_index_consistency_passed",
        "source_anchor_consistency_passed",
        "provenance_consistency_passed",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in audit.get("artifact_store_audits", []):
            writer.writerow({field: serialize_csv_value(row.get(field)) for field in fields})


def write_audit_json(audit: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_validation_issue_types(discard_log_path: Path) -> Counter:
    issue_types: Counter = Counter()
    if not discard_log_path.is_file():
        return issue_types
    for line in discard_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        issue_types[str(row.get("error_type", "unknown"))] += 1
    return issue_types


def count_forbidden_field_violations(value: Any) -> int:
    return sum(1 for field in FORBIDDEN_FIELDS if contains_key(value, field))


def count_api_key_leaks(root: Path) -> int:
    leaks = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        leaks += text.count(API_KEY_FIELD)
    return leaks


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(contains_key(child, key) for child in value)
    return False


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value
