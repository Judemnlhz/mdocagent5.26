"""Retrieval-only evaluation metrics for Stage 3/4 outputs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote


SEMANTIC_EDGE_TYPES = {"supports", "contradicts", "derived_from", "semantic_relation", "entails", "refutes"}
DEFAULT_EXPANSION_EDGE_TYPES = {
    "located_on_page",
    "supported_by_anchor",
    "anchor_on_page",
    "adjacent_page",
    "next_block",
    "section_contains",
    "table_contains_cell",
    "row_contains_cell",
    "column_contains_cell",
    "caption_of",
    "figure_has_caption",
}


def evaluate_stage3_retrieval(
    retrieval_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    k_values: Iterable[int] = (1, 3, 5),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact_by_id = {str(artifact.get("artifact_id")): artifact for artifact in artifacts}
    records_by_index = {int(record.get("record_index", index)): record for index, record in enumerate(records) if isinstance(record, dict)}
    records_by_doc = {str(record.get("doc_id")): record for record in records if isinstance(record, dict) and record.get("doc_id")}
    per_query: list[dict[str, Any]] = []
    type_counter: Counter[str] = Counter()
    modality_counter: Counter[str] = Counter()
    zero_hit_count = 0
    no_gold_count = 0

    for row_index, row in enumerate(retrieval_rows):
        record = records_by_index.get(int(row.get("record_index", -1))) or records_by_doc.get(str(row.get("doc_id") or "")) or {}
        gold_pages = extract_gold_pages(record)
        retrieved_ids = [str(value) for value in row.get("retrieved_artifact_ids", [])]
        retrieved_pages = [artifact_page(artifact_by_id.get(artifact_id)) for artifact_id in retrieved_ids]
        retrieved_pages = [page for page in retrieved_pages if page is not None]
        for artifact_id in retrieved_ids:
            artifact = artifact_by_id.get(artifact_id)
            if not artifact:
                continue
            type_counter[str(artifact.get("artifact_type") or "unknown")] += 1
            modality_counter[str(artifact.get("modality") or "unknown")] += 1
        if not retrieved_ids:
            zero_hit_count += 1
        if not gold_pages:
            no_gold_count += 1
        per_query.append(
            {
                "query_key": row.get("query_hash") or row.get("query_id") or row_index,
                "record_index": row.get("record_index"),
                "doc_id": row.get("doc_id"),
                "num_gold_pages": len(gold_pages),
                "num_retrieved": len(retrieved_ids),
                "recall_at_k_by_page": {str(k): recall_at_k(gold_pages, retrieved_pages, k) for k in k_values},
                "coverage_at_k_by_page": {str(k): coverage_at_k(gold_pages, retrieved_pages, k) for k in k_values},
                "evaluation_only": True,
            }
        )

    report = {
        "num_queries": len(retrieval_rows),
        "num_queries_with_gold": sum(1 for row in per_query if int(row["num_gold_pages"]) > 0),
        "recall_at_k_by_page": average_metric(per_query, "recall_at_k_by_page", k_values),
        "coverage_at_k_by_page": average_metric(per_query, "coverage_at_k_by_page", k_values),
        "retrieved_artifact_type_distribution": dict(sorted(type_counter.items())),
        "retrieved_modality_distribution": dict(sorted(modality_counter.items())),
        "zero_hit_query_count": zero_hit_count,
        "no_gold_available_count": no_gold_count,
        "evaluation_only": True,
    }
    return report, per_query


def evaluate_stage4_graph_expansion(
    retrieval_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    formal_edges: list[dict[str, Any]],
    k_values: Iterable[int] = (1, 3, 5),
    allowed_edge_types: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed = set(allowed_edge_types or DEFAULT_EXPANSION_EDGE_TYPES)
    if allowed & SEMANTIC_EDGE_TYPES:
        raise ValueError("Semantic edge types are not allowed for graph expansion evaluation")
    adjacency, edge_types_used = build_formal_adjacency(formal_edges, allowed)
    artifact_by_id = {str(artifact.get("artifact_id")): artifact for artifact in artifacts}
    records_by_index = {int(record.get("record_index", index)): record for index, record in enumerate(records) if isinstance(record, dict)}
    node_to_artifact = {artifact_node_id(artifact): str(artifact.get("artifact_id")) for artifact in artifacts}
    per_query: list[dict[str, Any]] = []
    expansion_factors: list[float] = []

    for row_index, row in enumerate(retrieval_rows):
        record = records_by_index.get(int(row.get("record_index", -1)), {})
        gold_pages = extract_gold_pages(record)
        flat_ids = [str(value) for value in row.get("retrieved_artifact_ids", [])]
        expanded_ids = expand_artifact_ids(flat_ids, artifact_by_id, node_to_artifact, adjacency)
        expansion_factors.append(len(expanded_ids) / max(1, len(flat_ids)))
        flat_pages = [artifact_page(artifact_by_id.get(artifact_id)) for artifact_id in flat_ids]
        graph_pages = [artifact_page(artifact_by_id.get(artifact_id)) for artifact_id in expanded_ids]
        flat_pages = [page for page in flat_pages if page is not None]
        graph_pages = [page for page in graph_pages if page is not None]
        per_query.append(
            {
                "query_key": row.get("query_hash") or row.get("query_id") or row_index,
                "record_index": row.get("record_index"),
                "doc_id": row.get("doc_id"),
                "flat_num_retrieved": len(flat_ids),
                "graph_num_retrieved": len(expanded_ids),
                "flat_recall_at_k": {str(k): recall_at_k(gold_pages, flat_pages, k) for k in k_values},
                "graph_recall_at_k": {str(k): recall_at_k(gold_pages, graph_pages, k) for k in k_values},
                "flat_coverage_at_k": {str(k): coverage_at_k(gold_pages, flat_pages, k) for k in k_values},
                "graph_coverage_at_k": {str(k): coverage_at_k(gold_pages, graph_pages, k) for k in k_values},
                "evaluation_only": True,
            }
        )

    flat_recall = average_metric(per_query, "flat_recall_at_k", k_values)
    graph_recall = average_metric(per_query, "graph_recall_at_k", k_values)
    flat_coverage = average_metric(per_query, "flat_coverage_at_k", k_values)
    graph_coverage = average_metric(per_query, "graph_coverage_at_k", k_values)
    report = {
        "flat_recall_at_k": flat_recall,
        "graph_recall_at_k": graph_recall,
        "delta_recall_at_k": {str(k): graph_recall[str(k)] - flat_recall[str(k)] for k in k_values},
        "flat_coverage_at_k": flat_coverage,
        "graph_coverage_at_k": graph_coverage,
        "delta_coverage_at_k": {str(k): graph_coverage[str(k)] - flat_coverage[str(k)] for k in k_values},
        "expansion_factor": sum(expansion_factors) / max(1, len(expansion_factors)),
        "edge_types_used": sorted(edge_types_used),
        "used_debug_edges": False,
        "used_semantic_edges": False,
        "evaluation_only": True,
    }
    return report, per_query


def build_formal_adjacency(edges: list[dict[str, Any]], allowed_edge_types: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    adjacency: dict[str, set[str]] = {}
    edge_types_used: set[str] = set()
    for edge in edges:
        edge_type = str(edge.get("edge_type") or "")
        if edge_type in SEMANTIC_EDGE_TYPES or edge_type in {"same_record", "same_record_debug"}:
            raise ValueError(f"Forbidden edge type in graph expansion evaluation: {edge_type}")
        if edge_type not in allowed_edge_types:
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
        edge_types_used.add(edge_type)
    return adjacency, edge_types_used


def expand_artifact_ids(
    artifact_ids: list[str],
    artifact_by_id: dict[str, dict[str, Any]],
    node_to_artifact: dict[str, str],
    adjacency: dict[str, set[str]],
) -> list[str]:
    expanded = list(dict.fromkeys(artifact_ids))
    seen = set(expanded)
    for artifact_id in artifact_ids:
        artifact = artifact_by_id.get(artifact_id)
        if not artifact:
            continue
        start_node = artifact_node_id(artifact)
        for neighbor in sorted(adjacency.get(start_node, set())):
            neighbor_artifact = node_to_artifact.get(neighbor)
            if neighbor_artifact and neighbor_artifact not in seen:
                seen.add(neighbor_artifact)
                expanded.append(neighbor_artifact)
    return expanded


def extract_gold_pages(record: dict[str, Any]) -> set[int]:
    raw = record.get("evidence_pages")
    if raw in (None, "", []):
        return set()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [raw]
    if not isinstance(raw, list):
        raw = [raw]
    pages = set()
    for value in raw:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        pages.add(page - 1 if page > 0 else page)
    return pages


def recall_at_k(gold_pages: set[int], retrieved_pages: list[int], k: int) -> float:
    if not gold_pages:
        return 0.0
    return 1.0 if gold_pages & set(retrieved_pages[: int(k)]) else 0.0


def coverage_at_k(gold_pages: set[int], retrieved_pages: list[int], k: int) -> float:
    if not gold_pages:
        return 0.0
    return len(gold_pages & set(retrieved_pages[: int(k)])) / len(gold_pages)


def average_metric(rows: list[dict[str, Any]], field: str, k_values: Iterable[int]) -> dict[str, float]:
    return {
        str(k): sum(float(row[field][str(k)]) for row in rows) / max(1, len(rows))
        for k in k_values
    }


def artifact_page(artifact: dict[str, Any] | None) -> int | None:
    if not artifact:
        return None
    try:
        return int(artifact.get("page_index"))
    except (TypeError, ValueError):
        return None


def artifact_node_id(artifact: dict[str, Any]) -> str:
    from urllib.parse import quote

    return ":".join(
        [
            "artifact",
            quote(str(artifact.get("doc_id") or ""), safe=""),
            quote(str(int(artifact.get("page_index", -1))), safe=""),
            quote(str(artifact.get("artifact_id") or ""), safe=""),
        ]
    )


def parse_artifact_node_id(node_id: str) -> str | None:
    parts = str(node_id).split(":", 3)
    if len(parts) != 4 or parts[0] != "artifact":
        return None
    return unquote(parts[3])


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return read_jsonl(input_path)
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("records", "data", "items"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    return []


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
