"""Offline quality audit for Stage 2 cross-document artifact batches."""

from __future__ import annotations

from collections import Counter
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


FORBIDDEN_FIELD_NAMES = {
    "answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "proof_trace",
    "verified",
    "answer_supported",
    "proof_used",
}

API_KEY_FIELD_NAMES = {"api_key", "apikey", "api-key", "authorization"}
SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}|[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,})"
)

TEXT_ARTIFACT_TYPES = {"text_span", "page_summary"}
VISUAL_ARTIFACT_TYPES = {
    "chart",
    "chart_summary",
    "diagram",
    "figure",
    "figure_summary",
    "graph",
    "image",
    "image_region",
    "plot",
    "table",
    "table_region",
    "table_summary",
    "visual_region",
    "visual_summary",
}
VISUAL_MODALITIES = {"chart", "figure", "image", "multimodal", "table", "visual"}
TABLE_OR_FIGURE_TYPES = {
    "chart",
    "chart_summary",
    "diagram",
    "figure",
    "figure_summary",
    "graph",
    "plot",
    "table",
    "table_region",
    "table_summary",
}
TABLE_OR_FIGURE_MODALITIES = {"chart", "figure", "table"}

READINESS_THRESHOLDS = {
    "schema_valid_rate": 0.90,
    "anchoring_rate": 0.90,
    "discard_rate": 0.20,
    "num_documents": 5,
    "num_pages": 10,
}


def audit_crossdoc_batch(batch_dir: str | Path, stage2_json: str | Path | None = None) -> dict:
    """Audit an existing Stage 2 cross-document batch without compiling artifacts."""

    return audit_crossdoc_batch_with_options(batch_dir=batch_dir, stage2_json=stage2_json)


def audit_crossdoc_batch_with_options(
    batch_dir: str | Path,
    stage2_json: str | Path | None = None,
) -> dict:
    batch_path = Path(batch_dir)
    reports_dir = batch_path / "reports"
    context = _load_context(batch_path, reports_dir)
    stage2_context = _load_stage2_context(stage2_json)
    page_infos = _build_page_infos(context, stage2_context)

    artifact_counter = Counter()
    modality_counter = Counter()
    num_artifacts = 0
    anchor_valid_count = 0
    source_anchor_failures = []
    provenance_failures = []
    pages_with_text_artifact = set()
    pages_with_visual_artifact = set()
    pages_with_numeric_fact = set()
    pages_with_table_or_figure_artifact = set()

    forbidden_field_violations = 0
    api_key_leaks = 0
    files_for_secret_scan = []
    files_for_secret_scan.extend(context["artifact_store_values"])
    files_for_secret_scan.extend(context["raw_output_entries"])
    files_for_secret_scan.extend(context["discard_entries"])
    for report_name in ("summary", "manifest"):
        report_value = context.get(report_name)
        if report_value is not None:
            files_for_secret_scan.append(report_value)
    for value in files_for_secret_scan:
        forbidden_field_violations += _count_keys(value, FORBIDDEN_FIELD_NAMES)
        api_key_leaks += _count_api_key_leaks(value)

    for page_ref, page_info in page_infos.items():
        artifacts = page_info["artifacts"]
        num_artifacts += len(artifacts)
        anchor_checks = []
        provenance_checks = []
        for artifact in artifacts:
            artifact_type = _normalize_token(artifact.get("artifact_type") or artifact.get("type"))
            modality = _normalize_token(artifact.get("modality"))
            if artifact_type:
                artifact_counter[artifact_type] += 1
            if modality:
                modality_counter[modality] += 1

            if _is_text_artifact(artifact):
                pages_with_text_artifact.add(page_ref)
            if _is_visual_artifact(artifact):
                pages_with_visual_artifact.add(page_ref)
            if _is_numeric_artifact(artifact):
                pages_with_numeric_fact.add(page_ref)
            if _is_table_or_figure_artifact(artifact):
                pages_with_table_or_figure_artifact.add(page_ref)

            anchor_valid = _artifact_source_anchors_resolve(artifact, page_info, stage2_context)
            provenance_valid = _artifact_provenance_sources_resolve(artifact, page_info, stage2_context)
            anchor_checks.append(anchor_valid)
            provenance_checks.append(provenance_valid)
            if anchor_valid:
                anchor_valid_count += 1
            else:
                source_anchor_failures.append(page_ref)
            if not provenance_valid:
                provenance_failures.append(page_ref)

        page_info["num_artifacts"] = len(artifacts)
        page_info["artifact_types"] = sorted(
            _normalize_token(artifact.get("artifact_type") or artifact.get("type"))
            for artifact in artifacts
            if _normalize_token(artifact.get("artifact_type") or artifact.get("type"))
        )
        page_info["modalities"] = sorted(
            _normalize_token(artifact.get("modality"))
            for artifact in artifacts
            if _normalize_token(artifact.get("modality"))
        )
        page_info["has_text_artifact"] = page_ref in pages_with_text_artifact
        page_info["has_visual_artifact"] = page_ref in pages_with_visual_artifact
        page_info["has_numeric_fact"] = page_ref in pages_with_numeric_fact
        page_info["has_table_or_figure_artifact"] = page_ref in pages_with_table_or_figure_artifact
        page_info["source_anchors_valid"] = all(anchor_checks) if artifacts else True
        page_info["provenance_sources_resolve"] = all(provenance_checks) if artifacts else True

    num_pages = _resolve_num_pages(context, page_infos)
    num_documents = _resolve_num_documents(context, page_infos)
    validation_issue_types = _collect_validation_issue_types(context, page_infos)
    discarded_issue_types = _collect_discarded_issue_types(context)
    num_raw_artifacts = _resolve_num_raw_artifacts(context, num_artifacts, validation_issue_types)
    num_valid_artifacts = _resolve_num_valid_artifacts(context, num_artifacts)
    schema_valid_rate = _resolve_schema_valid_rate(context, num_valid_artifacts, num_raw_artifacts)
    anchoring_rate = _rate(anchor_valid_count, num_artifacts) if num_artifacts else _summary_rate(context, "anchoring_rate", 1.0)
    discard_rate = _resolve_discard_rate(context, num_raw_artifacts, num_valid_artifacts, validation_issue_types)

    all_artifacts_have_source_anchors = not source_anchor_failures
    all_provenance_sources_resolve = not provenance_failures
    selection_reason_counts = _selection_reason_counts(context, page_infos)
    page_refs = sorted(page_infos)
    pages_without_visual_artifacts = [page_ref for page_ref in page_refs if page_ref not in pages_with_visual_artifact]
    only_text_span_pages = [
        page_ref
        for page_ref in page_refs
        if page_infos[page_ref]["artifacts"]
        and all(_artifact_type_is(artifact, "text_span") for artifact in page_infos[page_ref]["artifacts"])
    ]
    pages_with_page_summary_only = [
        page_ref
        for page_ref in page_refs
        if page_infos[page_ref]["artifacts"]
        and all(_artifact_type_is(artifact, "page_summary") for artifact in page_infos[page_ref]["artifacts"])
    ]

    modality_coverage = {
        "num_pages_with_image_input": sum(1 for page_ref in page_refs if page_infos[page_ref]["has_image_input"]),
        "num_pages_with_visual_artifact": len(pages_with_visual_artifact),
        "visual_artifact_page_coverage": _rate(len(pages_with_visual_artifact), num_pages),
        "num_pages_with_text_artifact": len(pages_with_text_artifact),
        "text_artifact_page_coverage": _rate(len(pages_with_text_artifact), num_pages),
        "num_pages_with_numeric_fact": len(pages_with_numeric_fact),
        "numeric_fact_page_coverage": _rate(len(pages_with_numeric_fact), num_pages),
        "num_pages_with_table_or_figure_artifact": len(pages_with_table_or_figure_artifact),
        "table_or_figure_artifact_page_coverage": _rate(len(pages_with_table_or_figure_artifact), num_pages),
    }

    artifact_type_diagnosis = {
        "only_text_span_pages": only_text_span_pages,
        "pages_without_visual_artifacts": pages_without_visual_artifacts,
        "pages_with_page_summary_only": pages_with_page_summary_only,
        "discarded_artifact_issue_types": discarded_issue_types,
    }
    selection_diagnosis = {
        "selection_reasons": selection_reason_counts,
        "num_valid_explicit_page_with_image": int(selection_reason_counts.get("valid_explicit_page_with_image", 0)),
        "num_image_top_10_first_available": int(selection_reason_counts.get("image_top_10_first_available", 0)),
        "num_retrieval_union_first_available": int(selection_reason_counts.get("retrieval_union_first_available", 0)),
    }
    report = {
        "num_documents": num_documents,
        "num_pages": num_pages,
        "num_artifacts": num_artifacts,
        "num_artifacts_by_type": dict(sorted(artifact_counter.items())),
        "num_artifacts_by_modality": dict(sorted(modality_counter.items())),
        "schema_valid_rate": schema_valid_rate,
        "anchoring_rate": anchoring_rate,
        "discard_rate": discard_rate,
        "validation_issue_types": validation_issue_types,
        "forbidden_field_violations": forbidden_field_violations,
        "api_key_leaks": api_key_leaks,
        "all_artifacts_have_source_anchors": all_artifacts_have_source_anchors,
        "all_provenance_sources_resolve": all_provenance_sources_resolve,
        "modality_coverage": modality_coverage,
        "selection_diagnosis": selection_diagnosis,
        "artifact_type_diagnosis": artifact_type_diagnosis,
        "stage2_readiness_gate": {},
    }
    report["stage2_readiness_gate"] = _build_readiness_gate(report)
    report["_page_rows"] = _build_page_rows(page_infos)
    return report


def write_audit_json(report: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    public_report = {key: value for key, value in report.items() if not key.startswith("_")}
    path.write_text(json.dumps(public_report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_page_quality_csv(report: Mapping[str, Any], output_path: str | Path) -> None:
    rows = list(report.get("_page_rows", []))
    fields = [
        "doc_id",
        "page_index",
        "page_ref",
        "selection_reason",
        "has_image_input",
        "num_artifacts",
        "artifact_types",
        "modalities",
        "has_text_artifact",
        "has_visual_artifact",
        "has_numeric_fact",
        "has_table_or_figure_artifact",
        "only_text_span",
        "page_summary_only",
        "source_anchors_valid",
        "provenance_sources_resolve",
        "validation_issue_types",
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _load_context(batch_path: Path, reports_dir: Path) -> Dict[str, Any]:
    artifact_store_values = _read_artifact_stores(batch_path / "artifact_stores")
    raw_output_entries = _read_jsonl(batch_path / "raw_outputs" / "raw_outputs.jsonl")
    discard_entries = _read_jsonl(batch_path / "discard" / "discard.jsonl")
    return {
        "batch_path": batch_path,
        "artifact_store_values": artifact_store_values,
        "raw_output_entries": raw_output_entries,
        "discard_entries": discard_entries,
        "summary": _read_json_if_exists(reports_dir / "crossdoc_batch_summary.json"),
        "quality_rows": _read_csv_if_exists(reports_dir / "crossdoc_batch_quality.csv"),
        "manifest": _read_json_if_exists(reports_dir / "run_manifest.json"),
    }


def _read_artifact_stores(path: Path) -> list[dict]:
    stores = []
    if not path.is_dir():
        return stores
    for store_path in sorted(path.glob("*.json")):
        loaded = _read_json_if_exists(store_path)
        if isinstance(loaded, dict):
            loaded["_artifact_store_path"] = str(store_path)
            stores.append(loaded)
    return stores


def _read_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_jsonl_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        loaded = json.loads(text)
        return [item for item in loaded if isinstance(item, dict)]
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                entries.append(loaded)
    return entries


def _read_csv_if_exists(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _load_stage2_context(stage2_json: str | Path | None) -> dict:
    if stage2_json is None:
        return {"records": [], "page_sources": {}, "selection_candidates": Counter()}
    stage2_path = Path(stage2_json)
    records = _read_json_or_jsonl_records(stage2_path)
    page_sources: dict[tuple[str, int], dict] = {}
    selection_candidates: Counter[str] = Counter()

    for record in records:
        doc_id = record.get("doc_id")
        if doc_id is None:
            continue
        stage2 = record.get("stage2", {})
        if not isinstance(stage2, dict):
            continue
        for reason in _record_selection_reasons(stage2):
            selection_candidates[reason] += 1
    return {
        "records": records,
        "page_sources": page_sources,
        "selection_candidates": selection_candidates,
    }


def _record_selection_reasons(stage2: Mapping[str, Any]) -> list[str]:
    reasons = []
    route_pages = []
    image_pages = []
    for route in stage2.get("candidate_page_routes", []) or []:
        if not isinstance(route, dict) or route.get("page_index") is None:
            continue
        route_pages.append(int(route["page_index"]))
        if "image" in (route.get("routes") or []):
            image_pages.append(int(route["page_index"]))
    if image_pages:
        reasons.append("image_top_10_first_available")
    if route_pages:
        reasons.append("retrieval_union_first_available")
    return reasons


def _coerce_ints(values: Iterable[Any]) -> list[int]:
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _build_page_infos(context: Mapping[str, Any], stage2_context: Mapping[str, Any]) -> dict[str, dict]:
    page_infos: dict[str, dict] = {}
    for store in context["artifact_store_values"]:
        store_doc_id = store.get("doc_id") or store.get("document_id")
        pages = store.get("pages")
        if not isinstance(pages, list):
            continue
        for page in pages:
            if not isinstance(page, dict):
                continue
            doc_id = str(page.get("doc_id") or store_doc_id or _doc_id_from_artifacts(page.get("artifacts", [])) or "")
            page_index = _coerce_page_index(page.get("page_index"))
            if page_index is None:
                page_index = _coerce_page_index(_first_artifact_value(page.get("artifacts", []), "page_index"))
            if not doc_id or page_index is None:
                continue
            page_ref = _page_ref(doc_id, page_index)
            info = _ensure_page_info(page_infos, doc_id, page_index)
            info["artifacts"].extend(_extract_artifacts(page))
            info["artifact_store_paths"].append(store.get("_artifact_store_path"))
            for issue in page.get("validation_issues", []) or []:
                if isinstance(issue, dict):
                    info["validation_issue_types"].append(str(issue.get("error_type") or issue.get("issue_type") or "unknown"))

    for row in context["quality_rows"]:
        doc_id = row.get("doc_id")
        page_index = _coerce_page_index(row.get("page_index"))
        if not doc_id or page_index is None:
            continue
        info = _ensure_page_info(page_infos, str(doc_id), page_index)
        info["selection_reason"] = row.get("selection_reason") or info.get("selection_reason")
        info["quality_row"] = row
        if row.get("page_image_path"):
            info["has_image_input"] = True
        if row.get("provider_error_type"):
            info["validation_issue_types"].append(str(row["provider_error_type"]))

    for (doc_id, page_index), source in stage2_context["page_sources"].items():
        page_ref = _page_ref(doc_id, page_index)
        if page_ref not in page_infos:
            continue
        info = page_infos[page_ref]
        info["page_source"] = source
        if source.get("has_page_image") and source.get("page_image_path"):
            info["has_image_input"] = True

    return page_infos


def _ensure_page_info(page_infos: dict[str, dict], doc_id: str, page_index: int) -> dict:
    page_ref = _page_ref(doc_id, page_index)
    if page_ref not in page_infos:
        page_infos[page_ref] = {
            "doc_id": doc_id,
            "page_index": int(page_index),
            "page_ref": page_ref,
            "artifacts": [],
            "artifact_store_paths": [],
            "validation_issue_types": [],
            "selection_reason": None,
            "has_image_input": False,
            "page_source": None,
        }
    return page_infos[page_ref]


def _extract_artifacts(page: Mapping[str, Any]) -> list[dict]:
    artifacts = page.get("artifacts", [])
    if not isinstance(artifacts, list):
        return []
    return [dict(artifact) for artifact in artifacts if isinstance(artifact, dict)]


def _doc_id_from_artifacts(artifacts: Any) -> str | None:
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("doc_id"):
            return str(artifact["doc_id"])
    return None


def _first_artifact_value(artifacts: Any, field: str) -> Any:
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get(field) is not None:
            return artifact.get(field)
    return None


def _coerce_page_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_ref(doc_id: str, page_index: int) -> str:
    return f"{doc_id}#p{int(page_index)}"


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _is_text_artifact(artifact: Mapping[str, Any]) -> bool:
    artifact_type = _normalize_token(artifact.get("artifact_type") or artifact.get("type"))
    modality = _normalize_token(artifact.get("modality"))
    return modality == "text" or artifact_type in TEXT_ARTIFACT_TYPES


def _is_visual_artifact(artifact: Mapping[str, Any]) -> bool:
    artifact_type = _normalize_token(artifact.get("artifact_type") or artifact.get("type"))
    modality = _normalize_token(artifact.get("modality"))
    return artifact_type in VISUAL_ARTIFACT_TYPES or modality in VISUAL_MODALITIES


def _is_numeric_artifact(artifact: Mapping[str, Any]) -> bool:
    artifact_type = _normalize_token(artifact.get("artifact_type") or artifact.get("type"))
    return (
        "numeric" in artifact_type
        or artifact_type in {"number", "quantity", "measurement"}
        or artifact.get("numeric_value") is not None
    )


def _is_table_or_figure_artifact(artifact: Mapping[str, Any]) -> bool:
    artifact_type = _normalize_token(artifact.get("artifact_type") or artifact.get("type"))
    modality = _normalize_token(artifact.get("modality"))
    return artifact_type in TABLE_OR_FIGURE_TYPES or modality in TABLE_OR_FIGURE_MODALITIES


def _artifact_type_is(artifact: Mapping[str, Any], artifact_type: str) -> bool:
    return _normalize_token(artifact.get("artifact_type") or artifact.get("type")) == artifact_type


def _artifact_source_anchors_resolve(
    artifact: Mapping[str, Any],
    page_info: Mapping[str, Any],
    stage2_context: Mapping[str, Any],
) -> bool:
    anchors = _artifact_source_anchors(artifact)
    if not anchors:
        return False
    return all(_source_ref_resolves(anchor, page_info, stage2_context) for anchor in anchors)


def _artifact_provenance_sources_resolve(
    artifact: Mapping[str, Any],
    page_info: Mapping[str, Any],
    stage2_context: Mapping[str, Any],
) -> bool:
    sources = _artifact_provenance_sources(artifact)
    if not sources:
        return True
    return all(_source_ref_resolves(source, page_info, stage2_context) for source in sources)


def _artifact_source_anchors(artifact: Mapping[str, Any]) -> list[Any]:
    anchors = []
    for field in ("source_anchors", "source_anchor", "anchors"):
        value = artifact.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            anchors.extend(value)
        else:
            anchors.append(value)
    provenance = artifact.get("provenance")
    if isinstance(provenance, dict):
        for field in ("source_anchors", "anchors"):
            value = provenance.get(field)
            if isinstance(value, list):
                anchors.extend(value)
            elif value is not None:
                anchors.append(value)
    return anchors


def _artifact_provenance_sources(artifact: Mapping[str, Any]) -> list[Any]:
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        return []
    sources = []
    for field in ("sources", "source_refs", "source_ids"):
        value = provenance.get(field)
        if isinstance(value, list):
            sources.extend(value)
        elif value is not None:
            sources.append(value)
    return sources


def _source_ref_resolves(
    source_ref: Any,
    page_info: Mapping[str, Any],
    stage2_context: Mapping[str, Any],
) -> bool:
    normalized = _normalize_source_ref(source_ref)
    if normalized is None:
        return False
    doc_id = normalized.get("doc_id")
    page_index = normalized.get("page_index")
    if doc_id is not None and str(doc_id) != str(page_info["doc_id"]):
        return False
    if page_index is not None and int(page_index) != int(page_info["page_index"]):
        return False
    if doc_id is None:
        doc_id = page_info["doc_id"]
    if page_index is None:
        page_index = page_info["page_index"]
    source = stage2_context["page_sources"].get((str(doc_id), int(page_index))) or page_info.get("page_source")
    if source is None:
        return str(doc_id) == str(page_info["doc_id"]) and int(page_index) == int(page_info["page_index"])
    block_id = normalized.get("block_id")
    if block_id is None:
        return True
    return str(block_id) in {str(item) for item in source.get("layout_block_ids", []) or []}


def _normalize_source_ref(source_ref: Any) -> dict | None:
    if isinstance(source_ref, dict):
        doc_id = source_ref.get("doc_id") or source_ref.get("document_id")
        page_value = (
            source_ref.get("page_index")
            if source_ref.get("page_index") is not None
            else source_ref.get("page")
        )
        page_index = _coerce_page_index(page_value)
        block_id = (
            source_ref.get("block_id")
            or source_ref.get("layout_block_id")
            or source_ref.get("source_id")
        )
        return {"doc_id": doc_id, "page_index": page_index, "block_id": block_id}
    if isinstance(source_ref, str):
        match = re.search(r"(?P<doc>[^#]+)#p(?P<page>\d+)(?:#(?P<block>.+))?$", source_ref)
        if match:
            return {
                "doc_id": match.group("doc"),
                "page_index": int(match.group("page")),
                "block_id": match.group("block"),
            }
        return {"doc_id": None, "page_index": None, "block_id": source_ref}
    return None


def _count_keys(value: Any, keys: set[str]) -> int:
    if isinstance(value, dict):
        return sum(1 for key in value if str(key) in keys) + sum(_count_keys(child, keys) for child in value.values())
    if isinstance(value, list):
        return sum(_count_keys(child, keys) for child in value)
    return 0


def _count_api_key_leaks(value: Any) -> int:
    if isinstance(value, dict):
        return sum(1 for key in value if str(key).lower() in API_KEY_FIELD_NAMES) + sum(
            _count_api_key_leaks(child) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_count_api_key_leaks(child) for child in value)
    if isinstance(value, str) and SECRET_VALUE_PATTERN.search(value):
        return 1
    return 0


def _collect_validation_issue_types(context: Mapping[str, Any], page_infos: Mapping[str, Mapping[str, Any]]) -> dict:
    counts = Counter()
    for page_info in page_infos.values():
        for issue_type in page_info.get("validation_issue_types", []) or []:
            if issue_type:
                counts[str(issue_type)] += 1
    for entry in context["discard_entries"]:
        issue_type = entry.get("error_type") or entry.get("issue_type") or entry.get("provider_error_type")
        if issue_type:
            counts[str(issue_type)] += 1
    return dict(sorted(counts.items()))


def _collect_discarded_issue_types(context: Mapping[str, Any]) -> dict:
    counts = Counter()
    for entry in context["discard_entries"]:
        issue_type = entry.get("error_type") or entry.get("issue_type") or "unknown"
        counts[str(issue_type)] += 1
    return dict(sorted(counts.items()))


def _resolve_num_pages(context: Mapping[str, Any], page_infos: Mapping[str, Any]) -> int:
    summary_value = _summary_int(context, "num_pages_attempted")
    return summary_value if summary_value is not None else len(page_infos)


def _resolve_num_documents(context: Mapping[str, Any], page_infos: Mapping[str, Any]) -> int:
    summary_value = _summary_int(context, "num_documents_attempted")
    if summary_value is not None:
        return summary_value
    return len({info["doc_id"] for info in page_infos.values()})


def _resolve_num_raw_artifacts(
    context: Mapping[str, Any],
    num_artifacts: int,
    validation_issue_types: Mapping[str, int],
) -> int:
    summary_value = _summary_int(context, "num_raw_artifacts")
    if summary_value is not None:
        return summary_value
    raw_count = 0
    for entry in context["raw_output_entries"]:
        raw_output = entry.get("raw_output", entry)
        if isinstance(raw_output, dict) and isinstance(raw_output.get("artifacts"), list):
            raw_count += len(raw_output["artifacts"])
    return raw_count or num_artifacts + sum(int(value) for value in validation_issue_types.values())


def _resolve_num_valid_artifacts(context: Mapping[str, Any], num_artifacts: int) -> int:
    summary_value = _summary_int(context, "num_valid_artifacts")
    return summary_value if summary_value is not None else num_artifacts


def _resolve_schema_valid_rate(context: Mapping[str, Any], num_valid_artifacts: int, num_raw_artifacts: int) -> float:
    summary_value = _summary_rate(context, "schema_valid_rate", None)
    if summary_value is not None:
        return summary_value
    return _rate(num_valid_artifacts, num_raw_artifacts) if num_raw_artifacts else 1.0


def _resolve_discard_rate(
    context: Mapping[str, Any],
    num_raw_artifacts: int,
    num_valid_artifacts: int,
    validation_issue_types: Mapping[str, int],
) -> float:
    summary_value = _summary_rate(context, "discard_rate", None)
    if summary_value is not None:
        return summary_value
    if num_raw_artifacts:
        return max(0.0, float(num_raw_artifacts - num_valid_artifacts) / float(num_raw_artifacts))
    total_issues = sum(int(value) for value in validation_issue_types.values())
    return _rate(total_issues, num_valid_artifacts + total_issues) if total_issues else 0.0


def _selection_reason_counts(context: Mapping[str, Any], page_infos: Mapping[str, Any]) -> dict:
    counts = Counter()
    for row in context["quality_rows"]:
        reason = row.get("selection_reason")
        if reason:
            counts[str(reason)] += 1
    if not counts:
        for info in page_infos.values():
            reason = info.get("selection_reason")
            if reason:
                counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _summary_int(context: Mapping[str, Any], field: str) -> int | None:
    summary = context.get("summary")
    if not isinstance(summary, dict) or summary.get(field) is None:
        return None
    try:
        return int(summary[field])
    except (TypeError, ValueError):
        return None


def _summary_rate(context: Mapping[str, Any], field: str, default: float | None) -> float | None:
    summary = context.get("summary")
    if not isinstance(summary, dict) or summary.get(field) is None:
        return default
    try:
        return float(summary[field])
    except (TypeError, ValueError):
        return default


def _rate(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _build_readiness_gate(report: Mapping[str, Any]) -> dict:
    warnings = []
    blocking_reasons = []

    if float(report["schema_valid_rate"]) < READINESS_THRESHOLDS["schema_valid_rate"]:
        blocking_reasons.append("schema_valid_rate_below_threshold")
    if float(report["anchoring_rate"]) < READINESS_THRESHOLDS["anchoring_rate"]:
        blocking_reasons.append("anchoring_rate_below_threshold")
    if float(report["discard_rate"]) > READINESS_THRESHOLDS["discard_rate"]:
        blocking_reasons.append("discard_rate_above_threshold")
    if int(report["forbidden_field_violations"]) != 0:
        blocking_reasons.append("forbidden_field_violations")
    if int(report["api_key_leaks"]) != 0:
        blocking_reasons.append("api_key_leaks")
    if not report["all_artifacts_have_source_anchors"]:
        blocking_reasons.append("source_anchor_resolution_failed")
    if not report["all_provenance_sources_resolve"]:
        blocking_reasons.append("provenance_source_resolution_failed")
    if int(report["num_documents"]) < READINESS_THRESHOLDS["num_documents"]:
        blocking_reasons.append("num_documents_below_threshold")
    if int(report["num_pages"]) < READINESS_THRESHOLDS["num_pages"]:
        blocking_reasons.append("num_pages_below_threshold")

    modality = report["modality_coverage"]
    if modality["visual_artifact_page_coverage"] == 0:
        warnings.append("visual_artifact_coverage_zero")
    if modality["table_or_figure_artifact_page_coverage"] == 0:
        warnings.append("table_or_figure_artifact_coverage_zero")
    artifact_types = set(report["num_artifacts_by_type"])
    if artifact_types and artifact_types.issubset({"text_span", "page_summary"}):
        warnings.append("artifact_types_text_only")
    modalities = set(report["num_artifacts_by_modality"])
    if modalities and modalities == {"text"}:
        warnings.append("artifact_modalities_text_only")

    return {
        "passed": not blocking_reasons,
        "warnings": warnings,
        "blocking_reasons": blocking_reasons,
    }


def _build_page_rows(page_infos: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    for page_ref in sorted(page_infos):
        info = page_infos[page_ref]
        artifact_types = info.get("artifact_types", [])
        modalities = info.get("modalities", [])
        rows.append(
            {
                "doc_id": info["doc_id"],
                "page_index": int(info["page_index"]),
                "page_ref": page_ref,
                "selection_reason": info.get("selection_reason") or "",
                "has_image_input": bool(info.get("has_image_input")),
                "num_artifacts": int(info.get("num_artifacts", 0)),
                "artifact_types": "|".join(artifact_types),
                "modalities": "|".join(modalities),
                "has_text_artifact": bool(info.get("has_text_artifact")),
                "has_visual_artifact": bool(info.get("has_visual_artifact")),
                "has_numeric_fact": bool(info.get("has_numeric_fact")),
                "has_table_or_figure_artifact": bool(info.get("has_table_or_figure_artifact")),
                "only_text_span": bool(
                    info.get("artifacts")
                    and all(_artifact_type_is(artifact, "text_span") for artifact in info.get("artifacts", []))
                ),
                "page_summary_only": bool(
                    info.get("artifacts")
                    and all(_artifact_type_is(artifact, "page_summary") for artifact in info.get("artifacts", []))
                ),
                "source_anchors_valid": bool(info.get("source_anchors_valid", True)),
                "provenance_sources_resolve": bool(info.get("provenance_sources_resolve", True)),
                "validation_issue_types": "|".join(sorted(set(info.get("validation_issue_types", [])))),
            }
        )
    return rows
