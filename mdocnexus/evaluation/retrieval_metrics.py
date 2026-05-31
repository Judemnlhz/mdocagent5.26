"""Retrieval-only evaluation metrics for Stage 3/4 outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote


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
EXPANSION_MODES = {"flat", "direct_structural", "page_neighborhood", "source_anchor_neighborhood"}
DEBUG_EDGE_TYPES = {"same_record", "same_record_debug"}
CONTEXT_CLIQUE_EDGE_TYPES = {"same_page", "same_source_block"}


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
    blocked_edge_types: set[str] | None = None,
    expansion_mode: str = "direct_structural",
    include_edge_type_deltas: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if expansion_mode not in EXPANSION_MODES:
        raise ValueError(f"Unsupported expansion_mode: {expansion_mode}")
    allowed = set(allowed_edge_types or DEFAULT_EXPANSION_EDGE_TYPES)
    if blocked_edge_types:
        allowed -= set(blocked_edge_types)
    if allowed & SEMANTIC_EDGE_TYPES:
        raise ValueError("Semantic edge types are not allowed for graph expansion evaluation")
    graph_index = build_formal_edge_index(formal_edges, allowed | CONTEXT_CLIQUE_EDGE_TYPES)
    artifact_by_id = {str(artifact.get("artifact_id")): artifact for artifact in artifacts}
    records_by_index = {int(record.get("record_index", index)): record for index, record in enumerate(records) if isinstance(record, dict)}
    records_by_doc = {str(record.get("doc_id")): record for record in records if isinstance(record, dict) and record.get("doc_id")}
    node_to_artifact = {artifact_node_id(artifact): str(artifact.get("artifact_id")) for artifact in artifacts}
    artifact_to_node = {str(artifact.get("artifact_id")): artifact_node_id(artifact) for artifact in artifacts}
    per_query: list[dict[str, Any]] = []
    flat_counts: list[int] = []
    expanded_counts: list[int] = []
    added_counts: list[int] = []
    added_gold_hits: list[int] = []
    added_artifact_count_by_edge_type: Counter[str] = Counter()
    all_edge_types_used: set[str] = set()

    for row_index, row in enumerate(retrieval_rows):
        record = records_by_index.get(int(row.get("record_index", -1))) or records_by_doc.get(str(row.get("doc_id") or "")) or {}
        gold_pages = extract_gold_pages(record)
        flat_ids = [str(value) for value in row.get("retrieved_artifact_ids", [])]
        flat_unique_ids = list(dict.fromkeys(flat_ids))
        expanded_ids, edge_types_used = expand_artifact_ids(
            flat_ids,
            artifact_by_id=artifact_by_id,
            node_to_artifact=node_to_artifact,
            artifact_to_node=artifact_to_node,
            graph_index=graph_index,
            expansion_mode=expansion_mode,
        )
        all_edge_types_used.update(edge_types_used)
        added_ids = [artifact_id for artifact_id in expanded_ids if artifact_id not in set(flat_unique_ids)]
        added = len(added_ids)
        for edge_type in edge_types_used:
            added_artifact_count_by_edge_type[str(edge_type)] += added
        added_counts.append(added)
        flat_counts.append(len(flat_unique_ids))
        expanded_counts.append(len(expanded_ids))
        flat_pages_all = [artifact_page(artifact_by_id.get(artifact_id)) for artifact_id in flat_ids]
        flat_pages_all = [page for page in flat_pages_all if page is not None]
        added_pages = [artifact_page(artifact_by_id.get(artifact_id)) for artifact_id in added_ids]
        added_pages = [page for page in added_pages if page is not None]
        if added:
            added_gold_hits.append(1 if gold_pages & set(added_pages) else 0)
        flat_recall_at_k: dict[str, float] = {}
        expanded_recall_at_k: dict[str, float] = {}
        flat_coverage_at_k: dict[str, float] = {}
        expanded_coverage_at_k: dict[str, float] = {}
        for k in k_values:
            seed_ids = flat_ids[: int(k)]
            expanded_seed_ids, seed_edge_types = expand_artifact_ids(
                seed_ids,
                artifact_by_id=artifact_by_id,
                node_to_artifact=node_to_artifact,
                artifact_to_node=artifact_to_node,
                graph_index=graph_index,
                expansion_mode=expansion_mode,
            )
            all_edge_types_used.update(seed_edge_types)
            expanded_pages_for_k = [artifact_page(artifact_by_id.get(artifact_id)) for artifact_id in expanded_seed_ids]
            expanded_pages_for_k = [page for page in expanded_pages_for_k if page is not None]
            flat_recall_at_k[str(k)] = recall_at_k(gold_pages, flat_pages_all, k)
            flat_coverage_at_k[str(k)] = coverage_at_k(gold_pages, flat_pages_all, k)
            expanded_recall_at_k[str(k)] = recall_over_pages(gold_pages, expanded_pages_for_k)
            expanded_coverage_at_k[str(k)] = coverage_over_pages(gold_pages, expanded_pages_for_k)
        per_query.append(
            {
                "query_key": row.get("query_hash") or row.get("query_id") or row_index,
                "record_index": row.get("record_index"),
                "doc_id": row.get("doc_id"),
                "flat_num_retrieved": len(flat_ids),
                "expanded_num_retrieved": len(expanded_ids),
                "graph_num_retrieved": len(expanded_ids),
                "num_added_artifacts": added,
                "added_gold_page_hit": bool(added and gold_pages & set(added_pages)),
                "flat_recall_at_k": flat_recall_at_k,
                "expanded_recall_at_k": expanded_recall_at_k,
                "graph_recall_at_k": expanded_recall_at_k,
                "flat_coverage_at_k": flat_coverage_at_k,
                "expanded_coverage_at_k": expanded_coverage_at_k,
                "graph_coverage_at_k": expanded_coverage_at_k,
                "evaluation_only": True,
            }
        )

    flat_recall = average_metric(per_query, "flat_recall_at_k", k_values)
    expanded_recall = average_metric(per_query, "expanded_recall_at_k", k_values)
    flat_coverage = average_metric(per_query, "flat_coverage_at_k", k_values)
    expanded_coverage = average_metric(per_query, "expanded_coverage_at_k", k_values)
    avg_flat_artifacts = sum(flat_counts) / max(1, len(flat_counts))
    avg_added_artifacts = sum(added_counts) / max(1, len(added_counts))
    avg_expanded_artifacts = sum(expanded_counts) / max(1, len(expanded_counts))
    edge_type_delta_recall: dict[str, dict[str, float]] = {}
    edge_type_delta_coverage: dict[str, dict[str, float]] = {}
    if include_edge_type_deltas and expansion_mode != "flat":
        edge_type_delta_recall, edge_type_delta_coverage = evaluate_edge_type_deltas(
            retrieval_rows=retrieval_rows,
            artifacts=artifacts,
            records=records,
            formal_edges=formal_edges,
            k_values=k_values,
            edge_types=all_edge_types_used,
            expansion_mode=expansion_mode,
        )
    report = {
        "expansion_mode": expansion_mode,
        "flat_recall_at_k": flat_recall,
        "expanded_recall_at_k": expanded_recall,
        "graph_recall_at_k": expanded_recall,
        "delta_recall_at_k": {str(k): expanded_recall[str(k)] - flat_recall[str(k)] for k in k_values},
        "flat_coverage_at_k": flat_coverage,
        "expanded_coverage_at_k": expanded_coverage,
        "graph_coverage_at_k": expanded_coverage,
        "delta_coverage_at_k": {str(k): expanded_coverage[str(k)] - flat_coverage[str(k)] for k in k_values},
        "avg_flat_artifacts": avg_flat_artifacts,
        "avg_added_artifacts": avg_added_artifacts,
        "avg_expanded_artifacts": avg_expanded_artifacts,
        "expansion_ratio": avg_expanded_artifacts / max(1.0, avg_flat_artifacts),
        "added_ratio": avg_added_artifacts / max(1.0, avg_flat_artifacts),
        "expansion_factor": avg_expanded_artifacts / max(1.0, avg_flat_artifacts),
        "added_gold_page_hit_rate": sum(added_gold_hits) / max(1, len(added_gold_hits)),
        "added_artifact_count_by_edge_type": dict(sorted(added_artifact_count_by_edge_type.items())),
        "edge_type_delta_recall": edge_type_delta_recall,
        "edge_type_delta_coverage": edge_type_delta_coverage,
        "edge_types_used": sorted(all_edge_types_used),
        "used_debug_edges": False,
        "used_semantic_edges": False,
        "evaluation_only": True,
    }
    return report, per_query


def evaluate_edge_type_deltas(
    retrieval_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    formal_edges: list[dict[str, Any]],
    k_values: Iterable[int],
    edge_types: set[str],
    expansion_mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    delta_recall: dict[str, dict[str, float]] = {}
    delta_coverage: dict[str, dict[str, float]] = {}
    for edge_type in sorted(edge_types):
        if edge_type in SEMANTIC_EDGE_TYPES or edge_type in DEBUG_EDGE_TYPES:
            continue
        report, _ = evaluate_stage4_graph_expansion(
            retrieval_rows=retrieval_rows,
            artifacts=artifacts,
            records=records,
            formal_edges=formal_edges,
            k_values=k_values,
            allowed_edge_types={edge_type},
            expansion_mode=expansion_mode,
            include_edge_type_deltas=False,
        )
        delta_recall[edge_type] = dict(report.get("delta_recall_at_k", {}))
        delta_coverage[edge_type] = dict(report.get("delta_coverage_at_k", {}))
    return delta_recall, delta_coverage


def build_formal_edge_index(edges: list[dict[str, Any]], allowed_edge_types: set[str]) -> dict[str, Any]:
    out_by_type: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    in_by_type: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    undirected_by_type: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    edge_types_seen: set[str] = set()
    for edge in edges:
        edge_type = str(edge.get("edge_type") or "")
        if edge.get("debug") is True or edge_type in SEMANTIC_EDGE_TYPES or edge_type in DEBUG_EDGE_TYPES:
            raise ValueError(f"Forbidden edge type in graph expansion evaluation: {edge_type}")
        if edge_type not in allowed_edge_types:
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        out_by_type[edge_type][source].add(target)
        in_by_type[edge_type][target].add(source)
        undirected_by_type[edge_type][source].add(target)
        undirected_by_type[edge_type][target].add(source)
        edge_types_seen.add(edge_type)
    return {
        "out": out_by_type,
        "in": in_by_type,
        "undirected": undirected_by_type,
        "edge_types_seen": edge_types_seen,
    }


def build_formal_adjacency(edges: list[dict[str, Any]], allowed_edge_types: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    graph_index = build_formal_edge_index(edges, allowed_edge_types)
    adjacency: dict[str, set[str]] = {}
    edge_types_used: set[str] = set()
    for edge_type, by_source in graph_index["undirected"].items():
        for source, targets in by_source.items():
            adjacency.setdefault(source, set()).update(targets)
        edge_types_used.add(edge_type)
    return adjacency, edge_types_used


def expand_artifact_ids(
    artifact_ids: list[str],
    artifact_by_id: dict[str, dict[str, Any]],
    node_to_artifact: dict[str, str],
    artifact_to_node: dict[str, str] | None = None,
    graph_index: dict[str, Any] | None = None,
    expansion_mode: str = "direct_structural",
    adjacency: dict[str, set[str]] | None = None,
) -> tuple[list[str], set[str]] | list[str]:
    legacy_return = graph_index is None and adjacency is not None
    if graph_index is None:
        graph_index = {"undirected": {"direct": adjacency or {}}, "out": {}, "in": {}}
    artifact_to_node = artifact_to_node or {artifact_id: artifact_node_id(artifact) for artifact_id, artifact in artifact_by_id.items()}
    expanded: list[str] = []
    seen: set[str] = set()
    edge_types_used: set[str] = set()
    for artifact_id in artifact_ids:
        if artifact_id not in seen:
            seen.add(artifact_id)
            expanded.append(artifact_id)
        artifact = artifact_by_id.get(artifact_id)
        if not artifact:
            continue
        start_node = artifact_to_node.get(artifact_id) or artifact_node_id(artifact)
        neighbors, used = artifact_neighbors_for_mode(start_node, graph_index, node_to_artifact, expansion_mode)
        edge_types_used.update(used)
        for neighbor_artifact_id in sorted(neighbors):
            if neighbor_artifact_id not in seen:
                seen.add(neighbor_artifact_id)
                expanded.append(neighbor_artifact_id)
    if legacy_return:
        return expanded
    return expanded, edge_types_used


def artifact_neighbors_for_mode(
    start_node: str,
    graph_index: dict[str, Any],
    node_to_artifact: dict[str, str],
    expansion_mode: str,
) -> tuple[set[str], set[str]]:
    if expansion_mode == "flat":
        return set(), set()
    if expansion_mode == "direct_structural":
        return direct_structural_neighbors(start_node, graph_index, node_to_artifact)
    if expansion_mode == "page_neighborhood":
        return page_neighborhood_neighbors(start_node, graph_index, node_to_artifact)
    if expansion_mode == "source_anchor_neighborhood":
        return source_anchor_neighborhood_neighbors(start_node, graph_index, node_to_artifact)
    raise ValueError(f"Unsupported expansion_mode: {expansion_mode}")


def direct_structural_neighbors(start_node: str, graph_index: dict[str, Any], node_to_artifact: dict[str, str]) -> tuple[set[str], set[str]]:
    neighbors: set[str] = set()
    edge_types_used: set[str] = set()
    for edge_type in DEFAULT_EXPANSION_EDGE_TYPES:
        for node in graph_index["undirected"].get(edge_type, {}).get(start_node, set()):
            artifact_id = node_to_artifact.get(node)
            if artifact_id:
                neighbors.add(artifact_id)
                edge_types_used.add(edge_type)
    return neighbors, edge_types_used


def page_neighborhood_neighbors(start_node: str, graph_index: dict[str, Any], node_to_artifact: dict[str, str]) -> tuple[set[str], set[str]]:
    neighbors: set[str] = set()
    edge_types_used: set[str] = set()
    pages = set(graph_index["out"].get("located_on_page", {}).get(start_node, set()))
    pages.update(graph_index["in"].get("located_on_page", {}).get(start_node, set()))
    if pages:
        edge_types_used.add("located_on_page")
    for page in sorted(pages):
        same_page_nodes = graph_index["in"].get("located_on_page", {}).get(page, set())
        for node in same_page_nodes:
            artifact_id = node_to_artifact.get(node)
            if artifact_id:
                neighbors.add(artifact_id)
        adjacent_pages = set(graph_index["out"].get("adjacent_page", {}).get(page, set()))
        adjacent_pages.update(graph_index["in"].get("adjacent_page", {}).get(page, set()))
        if adjacent_pages:
            edge_types_used.add("adjacent_page")
        for neighbor_page in sorted(adjacent_pages):
            for node in graph_index["in"].get("located_on_page", {}).get(neighbor_page, set()):
                artifact_id = node_to_artifact.get(node)
                if artifact_id:
                    neighbors.add(artifact_id)
                    edge_types_used.add("located_on_page")
    return neighbors, edge_types_used


def source_anchor_neighborhood_neighbors(start_node: str, graph_index: dict[str, Any], node_to_artifact: dict[str, str]) -> tuple[set[str], set[str]]:
    neighbors: set[str] = set()
    edge_types_used: set[str] = set()
    anchors = set(graph_index["out"].get("supported_by_anchor", {}).get(start_node, set()))
    anchors.update(graph_index["in"].get("supported_by_anchor", {}).get(start_node, set()))
    if anchors:
        edge_types_used.add("supported_by_anchor")
    for anchor in sorted(anchors):
        for node in graph_index["in"].get("supported_by_anchor", {}).get(anchor, set()):
            artifact_id = node_to_artifact.get(node)
            if artifact_id:
                neighbors.add(artifact_id)
    return neighbors, edge_types_used


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


def recall_over_pages(gold_pages: set[int], retrieved_pages: list[int]) -> float:
    if not gold_pages:
        return 0.0
    return 1.0 if gold_pages & set(retrieved_pages) else 0.0


def coverage_over_pages(gold_pages: set[int], retrieved_pages: list[int]) -> float:
    if not gold_pages:
        return 0.0
    return len(gold_pages & set(retrieved_pages)) / len(gold_pages)


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
