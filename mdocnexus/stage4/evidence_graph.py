"""Build a deterministic rule-only document-native evidence graph.

Stage 4 consumes Stage 2 document-generic artifacts and emits only structural
relations grounded in artifact provenance and layout metadata. It never creates
answer-generation or semantic evidence edges.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable
from urllib.parse import quote

from mdocnexus.common.model_config import deterministic_stage_model_fields
from mdocnexus.stage4.locator_policy import classify_locator


DEFAULT_ARTIFACTS_JSONL = "outputs/stage2_doc/artifacts.jsonl"
DEFAULT_RETRIEVAL_JSONL = "outputs/stage3_doc_artifact_retrieval/retrieval.jsonl"
DEFAULT_STAGE2_JSON = "outputs/stage2/clean/sample-with-stage2-index.json"
DEFAULT_OUTPUT_DIR = "outputs/stage4/evidence_graph"
SCHEMA_VERSION = "stage4_evidence_graph_v2"
GRAPH_MODE = "rule_only_document_native_structural"
RULE_VERSION = "stage4b_rule_v1"

NODE_TYPES = {
    "artifact",
    "page",
    "source_anchor",
    "section",
    "figure",
    "caption",
    "table",
    "table_row",
    "table_column",
}
FORMAL_EDGE_TYPES = {
    "located_on_page",
    "supported_by_anchor",
    "anchor_on_page",
    "adjacent_page",
    "next_block",
    "section_contains",
    "caption_of",
    "figure_has_caption",
    "table_contains_cell",
    "row_contains_cell",
    "column_contains_cell",
}
CONTEXT_EDGE_TYPES = {"same_page", "same_source_block"}
FORMAL_RETRIEVAL_EDGE_TYPES = FORMAL_EDGE_TYPES - CONTEXT_EDGE_TYPES
DEBUG_EDGE_TYPES = {"same_record_debug"}
SEMANTIC_EDGE_TYPES = {
    "supports",
    "contradicts",
    "derived_from",
    "semantic_relation",
    "entails",
    "refutes",
    "answer_supports",
    "proof_supports",
    "cites",
}
FORBIDDEN_FORMAL_EDGE_TYPES = SEMANTIC_EDGE_TYPES | {"same_record", "same_record_debug"}
FORBIDDEN_OUTPUT_KEYS = {
    "answer",
    "answers",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
    "page_image_path",
    "page_text_path",
    "raw_output",
    "raw_response",
    "api_key",
    "local_path",
    "absolute_path",
    "image_path",
    "file_url",
    "provider_response",
    "raw_outputs",
    "secret",
}
FORBIDDEN_TEXT_FRAGMENTS = ("file://", "/home/", "data:image")
ELEMENT_LOCATOR_KEYS = {
    "block_id",
    "table_id",
    "figure_id",
    "caption_id",
    "section_id",
    "bbox",
    "row",
    "row_index",
    "col",
    "col_index",
    "column",
    "column_index",
    "text_span_offset",
    "text_span_start",
    "text_span_end",
    "char_start",
    "char_end",
    "start_offset",
    "end_offset",
}
BLOCK_SUFFIX_RE = re.compile(r"^(.*?)(\d+)$")


def run_evidence_graph_build(
    artifacts_jsonl_path: str | Path = DEFAULT_ARTIFACTS_JSONL,
    retrieval_jsonl_path: str | Path = DEFAULT_RETRIEVAL_JSONL,
    stage2_json_path: str | Path = DEFAULT_STAGE2_JSON,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build and write the Stage 4B rule-only document-native graph."""

    artifacts_path = Path(artifacts_jsonl_path)
    artifacts = load_jsonl(artifacts_path)
    retrieval_rows = load_jsonl_if_exists(retrieval_jsonl_path)
    stage2_records = load_stage2_record_metadata_if_exists(stage2_json_path)
    graph = build_evidence_graph(
        artifacts=artifacts,
        retrieval_rows=retrieval_rows,
        stage2_records=stage2_records,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    nodes_path = output_root / "nodes.jsonl"
    edges_path = output_root / "edges.jsonl"
    debug_edges_path = output_root / "debug_edges.jsonl"
    quality_report_path = output_root / "quality_report.json"
    manifest_path = output_root / "manifest.json"

    write_jsonl(nodes_path, graph["nodes"])
    write_jsonl(edges_path, graph["edges"])
    write_jsonl(debug_edges_path, graph["debug_edges"])
    quality_report_path.write_text(
        json.dumps(graph["quality_report"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = build_manifest(
        nodes=graph["nodes"],
        edges=graph["edges"],
        debug_edges=graph["debug_edges"],
        quality_report=graph["quality_report"],
        artifacts=artifacts,
        artifacts_jsonl_path=artifacts_path,
        command_args={
            "artifacts_jsonl": public_path(artifacts_jsonl_path),
            "retrieval_jsonl": public_path(retrieval_jsonl_path),
            "stage2_json": public_path(stage2_json_path),
            "output_dir": public_path(output_dir),
        },
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "debug_edges_path": str(debug_edges_path),
        "quality_report_path": str(quality_report_path),
        "manifest_path": str(manifest_path),
        "quality_report": graph["quality_report"],
        "manifest": manifest,
    }


def build_evidence_graph(
    artifacts: Iterable[dict[str, Any]],
    retrieval_rows: Iterable[dict[str, Any]] | None = None,
    stage2_records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return document-native structural graph rows and a quality report."""

    retrieval_rows = list(retrieval_rows or [])
    _ = list(stage2_records or [])
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    debug_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    artifact_refs: list[dict[str, Any]] = []
    anchor_refs: list[dict[str, Any]] = []
    page_refs: dict[tuple[str, int], str] = {}

    num_artifacts = 0
    num_artifacts_with_page_locator = 0
    num_artifacts_with_source_anchor = 0
    num_artifacts_with_element_locator = 0
    num_proof_trace_eligible = 0
    proof_trace_eligible_by_type: Counter[str] = Counter()
    locator_kind_counts: Counter[str] = Counter()

    for input_order, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        num_artifacts += 1
        ref = artifact_reference(artifact)
        if ref is None:
            skipped["artifact_missing_required_identity"] += 1
            continue

        artifact_node = ref["node_id"]
        page_node = page_node_id(ref["doc_id"], ref["page_index"])
        page_refs[(ref["doc_id"], ref["page_index"])] = page_node
        add_node(nodes, build_artifact_node(artifact, ref))
        add_node(nodes, {"node_id": page_node, "node_type": "page", "doc_id": ref["doc_id"], "page_id": page_node, "page_index": ref["page_index"]})
        add_edge(
            edges,
            artifact_node,
            page_node,
            "located_on_page",
            rule_name="artifact_page_locator",
            provenance=build_provenance(ref["doc_id"], ref["page_index"], artifact=artifact),
        )
        num_artifacts_with_page_locator += 1

        anchors = valid_source_anchors(artifact, fallback_page_index=ref["page_index"])
        if anchors:
            num_artifacts_with_source_anchor += 1
        else:
            skipped["artifact_missing_source_anchor"] += 1
        for anchor in anchors:
            anchor_page_index = anchor["page_index"]
            anchor_page_node = page_node_id(ref["doc_id"], anchor_page_index)
            anchor_node = source_anchor_node_id(ref["doc_id"], anchor_page_index, anchor["source_id"])
            page_refs[(ref["doc_id"], anchor_page_index)] = anchor_page_node
            add_node(nodes, {"node_id": anchor_page_node, "node_type": "page", "doc_id": ref["doc_id"], "page_id": anchor_page_node, "page_index": anchor_page_index})
            add_node(
                nodes,
                clean_optional_fields(
                    {
                        "node_id": anchor_node,
                        "node_type": "source_anchor",
                        "doc_id": ref["doc_id"],
                        "page_id": anchor_page_node,
                        "page_index": anchor_page_index,
                        "source_block_id": anchor["source_id"],
                        "bbox": anchor.get("bbox"),
                    }
                ),
            )
            add_edge(
                edges,
                artifact_node,
                anchor_node,
                "supported_by_anchor",
                rule_name="artifact_source_anchor",
                provenance=build_provenance(
                    ref["doc_id"],
                    anchor_page_index,
                    source_block_id=anchor["source_id"],
                    bbox=anchor.get("bbox"),
                    artifact=artifact,
                ),
            )
            add_edge(
                edges,
                anchor_node,
                anchor_page_node,
                "anchor_on_page",
                rule_name="source_anchor_page_locator",
                provenance=build_provenance(ref["doc_id"], anchor_page_index, source_block_id=anchor["source_id"], bbox=anchor.get("bbox")),
            )
            anchor_refs.append(
                {
                    "node_id": anchor_node,
                    "doc_id": ref["doc_id"],
                    "page_index": anchor_page_index,
                    "source_block_id": anchor["source_id"],
                }
            )

        locator_classification = classify_locator(artifact)
        locator_kind_counts[str(locator_classification["locator_kind"])] += 1
        if locator_classification["element_locatable"]:
            num_artifacts_with_element_locator += 1
        if locator_classification["proof_trace_eligible"]:
            num_proof_trace_eligible += 1
            proof_trace_eligible_by_type[str(artifact.get("artifact_type") or "unknown")] += 1
        ref.update(extract_layout_locator(artifact))
        source_block_id = first_source_block_id(artifact)
        if source_block_id:
            ref["source_block_id"] = source_block_id
        artifact_refs.append(ref)
        add_explicit_layout_edges(nodes, edges, artifact, ref, skipped)

    add_pairwise_document_edges(edges, artifact_refs, skipped)
    add_same_record_debug_edges_from_retrieval(debug_edges, artifact_refs, retrieval_rows)
    add_adjacent_page_edges(edges, page_refs)
    add_next_block_edges(edges, anchor_refs, skipped)

    node_rows = sorted(nodes.values(), key=lambda item: item["node_id"])
    edge_rows = sorted(edges.values(), key=edge_sort_key)
    debug_edge_rows = sorted(debug_edges.values(), key=edge_sort_key)
    quality_report = build_quality_report(
        nodes=node_rows,
        edges=edge_rows,
        debug_edges=debug_edge_rows,
        num_artifacts=num_artifacts,
        num_artifacts_with_page_locator=num_artifacts_with_page_locator,
        num_artifacts_with_source_anchor=num_artifacts_with_source_anchor,
        num_artifacts_with_element_locator=num_artifacts_with_element_locator,
        num_proof_trace_eligible=num_proof_trace_eligible,
        proof_trace_eligible_by_type=dict(sorted(proof_trace_eligible_by_type.items())),
        locator_kind_counts=dict(sorted(locator_kind_counts.items())),
        skipped_rule_edges_by_reason=dict(sorted(skipped.items())),
    )
    assert_graph_outputs(node_rows, edge_rows, quality_report, debug_edge_rows)
    return {"nodes": node_rows, "edges": edge_rows, "debug_edges": debug_edge_rows, "quality_report": quality_report}


def artifact_reference(artifact: dict[str, Any]) -> dict[str, Any] | None:
    doc_id = str(artifact.get("doc_id") or "")
    page_index = coerce_int(artifact.get("page_index"), fallback=-1)
    artifact_id = str(artifact.get("artifact_id") or "")
    if not doc_id or page_index < 0 or not artifact_id:
        return None
    return {
        "node_id": artifact_node_id(doc_id, page_index, artifact_id),
        "doc_id": doc_id,
        "page_index": page_index,
        "artifact_id": artifact_id,
    }


def build_artifact_node(artifact: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    locator = extract_layout_locator(artifact)
    node = {
        "node_id": ref["node_id"],
        "node_type": "artifact",
        "doc_id": ref["doc_id"],
        "page_id": page_node_id(ref["doc_id"], ref["page_index"]),
        "page_index": ref["page_index"],
        "artifact_id": ref["artifact_id"],
        "artifact_type": artifact.get("artifact_type"),
        "modality": artifact.get("modality"),
        "candidate_status": artifact.get("validation_status", "candidate"),
        "source_block_id": first_source_block_id(artifact),
        "bbox": first_bbox(artifact),
    }
    node.update({key: value for key, value in locator.items() if key in ELEMENT_LOCATOR_KEYS})
    return clean_optional_fields(node)


def add_explicit_layout_edges(
    nodes: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str, str], dict[str, Any]],
    artifact: dict[str, Any],
    ref: dict[str, Any],
    skipped: Counter[str],
) -> None:
    locator = extract_layout_locator(artifact)
    doc_id = ref["doc_id"]
    page_index = ref["page_index"]
    artifact_node = ref["node_id"]

    section_id = clean_id(locator.get("section_id"))
    if section_id:
        section_node = section_node_id(doc_id, page_index, section_id)
        add_node(nodes, {"node_id": section_node, "node_type": "section", "doc_id": doc_id, "page_id": page_node_id(doc_id, page_index), "page_index": page_index, "section_id": section_id})
        add_edge(
            edges,
            section_node,
            artifact_node,
            "section_contains",
            rule_name="explicit_section_locator",
            provenance=build_provenance(doc_id, page_index, source_block_id=first_source_block_id(artifact), artifact=artifact, section_id=section_id),
        )
    elif artifact_type_matches(artifact, {"section", "section_header", "header"}):
        skipped["section_contains_missing_section_id"] += 1

    figure_id = clean_id(locator.get("figure_id"))
    caption_id = clean_id(locator.get("caption_id"))
    if figure_id and caption_id:
        figure_node = figure_node_id(doc_id, page_index, figure_id)
        caption_node = caption_node_id(doc_id, page_index, caption_id)
        add_node(nodes, {"node_id": figure_node, "node_type": "figure", "doc_id": doc_id, "page_id": page_node_id(doc_id, page_index), "page_index": page_index, "figure_id": figure_id})
        add_node(nodes, {"node_id": caption_node, "node_type": "caption", "doc_id": doc_id, "page_id": page_node_id(doc_id, page_index), "page_index": page_index, "caption_id": caption_id, "figure_id": figure_id})
        provenance = build_provenance(doc_id, page_index, artifact=artifact, figure_id=figure_id, caption_id=caption_id)
        add_edge(edges, caption_node, figure_node, "caption_of", rule_name="explicit_figure_caption_locator", provenance=provenance)
        add_edge(edges, figure_node, caption_node, "figure_has_caption", rule_name="explicit_figure_caption_locator", provenance=provenance)
    elif artifact_type_matches(artifact, {"figure", "caption", "figure_caption"}) or figure_id or caption_id:
        skipped["caption_figure_missing_explicit_pair"] += 1

    table_id = clean_id(locator.get("table_id"))
    row_index = clean_id(locator.get("row_index"))
    column_index = clean_id(locator.get("column_index"))
    is_cell = artifact_type_matches(artifact, {"table_cell", "cell"}) or bool(table_id and (row_index or column_index))
    if table_id and is_cell:
        table_node = table_node_id(doc_id, page_index, table_id)
        add_node(nodes, {"node_id": table_node, "node_type": "table", "doc_id": doc_id, "page_id": page_node_id(doc_id, page_index), "page_index": page_index, "table_id": table_id})
        add_edge(
            edges,
            table_node,
            artifact_node,
            "table_contains_cell",
            rule_name="explicit_table_cell_locator",
            provenance=build_provenance(doc_id, page_index, artifact=artifact, table_id=table_id, row_index=row_index, column_index=column_index),
        )
        if row_index:
            row_node = table_row_node_id(doc_id, page_index, table_id, row_index)
            add_node(nodes, {"node_id": row_node, "node_type": "table_row", "doc_id": doc_id, "page_id": page_node_id(doc_id, page_index), "page_index": page_index, "table_id": table_id, "row_index": row_index})
            add_edge(edges, row_node, artifact_node, "row_contains_cell", rule_name="explicit_table_cell_locator", provenance=build_provenance(doc_id, page_index, artifact=artifact, table_id=table_id, row_index=row_index))
        else:
            skipped["row_contains_cell_missing_row_index"] += 1
        if column_index:
            column_node = table_column_node_id(doc_id, page_index, table_id, column_index)
            add_node(nodes, {"node_id": column_node, "node_type": "table_column", "doc_id": doc_id, "page_id": page_node_id(doc_id, page_index), "page_index": page_index, "table_id": table_id, "column_index": column_index})
            add_edge(edges, column_node, artifact_node, "column_contains_cell", rule_name="explicit_table_cell_locator", provenance=build_provenance(doc_id, page_index, artifact=artifact, table_id=table_id, column_index=column_index))
        else:
            skipped["column_contains_cell_missing_column_index"] += 1
    elif is_cell:
        skipped["table_contains_cell_missing_table_id"] += 1


def add_pairwise_document_edges(
    edges: dict[tuple[str, str, str], dict[str, Any]],
    artifact_refs: list[dict[str, Any]],
    skipped: Counter[str],
) -> None:
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_source_block: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for ref in artifact_refs:
        by_page[(str(ref["doc_id"]), int(ref["page_index"]))].append(ref)
        source_block_id = clean_id(ref.get("source_block_id"))
        if source_block_id:
            by_source_block[(str(ref["doc_id"]), int(ref["page_index"]), source_block_id)].append(ref)

    skipped["same_doc_pairwise_clique_disabled"] += 1
    for (doc_id, page_index), refs in by_page.items():
        for left, right in sorted_pairs(refs):
            add_edge(edges, left["node_id"], right["node_id"], "same_page", rule_name="same_page_artifact_pair", provenance=build_provenance(doc_id, page_index))
    for (doc_id, page_index, source_block_id), refs in by_source_block.items():
        for left, right in sorted_pairs(refs):
            add_edge(
                edges,
                left["node_id"],
                right["node_id"],
                "same_source_block",
                rule_name="same_source_block_artifact_pair",
                provenance=build_provenance(doc_id, page_index, source_block_id=source_block_id),
            )


def add_same_record_debug_edges_from_retrieval(
    debug_edges: dict[tuple[str, str, str], dict[str, Any]],
    artifact_refs: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> None:
    refs_by_artifact_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in artifact_refs:
        refs_by_artifact_id[str(ref["artifact_id"])].append(ref)
    for row in retrieval_rows:
        artifact_ids = row.get("retrieved_artifact_ids")
        if not isinstance(artifact_ids, list):
            continue
        refs: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        for artifact_id in artifact_ids:
            for ref in refs_by_artifact_id.get(str(artifact_id), []):
                if ref["node_id"] in seen_node_ids:
                    continue
                seen_node_ids.add(ref["node_id"])
                refs.append(ref)
        for left, right in sorted_pairs(refs):
            add_edge(
                debug_edges,
                left["node_id"],
                right["node_id"],
                "same_record_debug",
                rule_name="retrieval_row_debug_only",
                provenance=clean_optional_fields(
                    {
                        "doc_id": left["doc_id"],
                        "query_hash": row.get("query_hash"),
                        "query_id": row.get("query_id"),
                        "retrieval_method": row.get("retrieval_method"),
                    }
                ),
                debug=True,
            )


def add_adjacent_page_edges(edges: dict[tuple[str, str, str], dict[str, Any]], page_refs: dict[tuple[str, int], str]) -> None:
    by_doc: dict[str, list[int]] = defaultdict(list)
    for doc_id, page_index in page_refs:
        by_doc[doc_id].append(page_index)
    for doc_id, page_indices in by_doc.items():
        ordered = sorted(set(page_indices))
        for left, right in zip(ordered, ordered[1:]):
            if right - left != 1:
                continue
            add_edge(
                edges,
                page_node_id(doc_id, left),
                page_node_id(doc_id, right),
                "adjacent_page",
                rule_name="adjacent_page_index_same_doc",
                provenance={
                    "doc_id": doc_id,
                    "source_page_id": page_node_id(doc_id, left),
                    "target_page_id": page_node_id(doc_id, right),
                    "source_page_index": left,
                    "target_page_index": right,
                },
            )


def add_next_block_edges(
    edges: dict[tuple[str, str, str], dict[str, Any]],
    anchor_refs: list[dict[str, Any]],
    skipped: Counter[str],
) -> None:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for ref in anchor_refs:
        source_block_id = str(ref.get("source_block_id") or "")
        parsed = parse_block_sequence(source_block_id)
        if parsed is None:
            skipped["next_block_unparseable_source_block_id"] += 1
            continue
        prefix, number = parsed
        key = (str(ref["doc_id"]), int(ref["page_index"]), source_block_id)
        if key in seen:
            continue
        seen.add(key)
        row = dict(ref)
        row["block_prefix"] = prefix
        row["block_number"] = number
        grouped[(str(ref["doc_id"]), int(ref["page_index"]), prefix)].append(row)
    for (doc_id, page_index, _prefix), refs in grouped.items():
        ordered = sorted(refs, key=lambda item: (int(item["block_number"]), str(item["source_block_id"])))
        for left, right in zip(ordered, ordered[1:]):
            if int(right["block_number"]) - int(left["block_number"]) != 1:
                continue
            add_edge(
                edges,
                left["node_id"],
                right["node_id"],
                "next_block",
                rule_name="consecutive_source_block_locator",
                provenance={
                    "doc_id": doc_id,
                    "page_id": page_node_id(doc_id, page_index),
                    "page_index": page_index,
                    "source_block_id": left["source_block_id"],
                    "target_source_block_id": right["source_block_id"],
                },
            )


def add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    cleaned = clean_optional_fields(node)
    node_type = cleaned.get("node_type")
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unsupported node_type: {node_type}")
    assert_no_forbidden_keys(cleaned)
    nodes[str(cleaned["node_id"])] = cleaned


def add_edge(
    edges: dict[tuple[str, str, str], dict[str, Any]],
    source: str,
    target: str,
    edge_type: str,
    rule_name: str,
    provenance: dict[str, Any] | None = None,
    debug: bool = False,
) -> None:
    allowed_edge_types = DEBUG_EDGE_TYPES if debug else FORMAL_EDGE_TYPES | CONTEXT_EDGE_TYPES
    if edge_type not in allowed_edge_types:
        raise ValueError(f"Unsupported edge_type: {edge_type}")
    if not debug and edge_type in FORBIDDEN_FORMAL_EDGE_TYPES:
        raise ValueError(f"Forbidden formal edge_type: {edge_type}")
    key = (str(source), str(target), edge_type)
    edge = {
        "edge_id": edge_id(edge_type, source, target),
        "source": str(source),
        "target": str(target),
        "edge_type": edge_type,
        "provenance": clean_optional_fields(provenance or {}),
        "rule_name": rule_name,
        "rule_version": RULE_VERSION,
        "deterministic": True,
    }
    if debug:
        edge["debug"] = True
    assert_no_forbidden_keys(edge)
    edges[key] = edge


def edge_sort_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(edge.get("edge_type")), str(edge.get("source")), str(edge.get("target")), str(edge.get("edge_id")))


def sorted_pairs(refs: list[dict[str, Any]]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(refs, key=lambda item: str(item["node_id"]))
    return combinations(ordered, 2)


def valid_source_anchors(artifact: dict[str, Any], fallback_page_index: int) -> list[dict[str, Any]]:
    anchors = artifact.get("source_anchors") if isinstance(artifact.get("source_anchors"), list) else []
    valid: list[dict[str, Any]] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        source_id = anchor.get("source_id")
        if source_id in (None, ""):
            continue
        valid.append(
            {
                "source_id": str(source_id),
                "page_index": coerce_int(anchor.get("page_index"), fallback=fallback_page_index),
                "bbox": anchor.get("bbox"),
                "anchor_type": anchor.get("anchor_type"),
            }
        )
    return valid


def extract_layout_locator(artifact: dict[str, Any]) -> dict[str, Any]:
    locator: dict[str, Any] = {}
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    provenance = artifact.get("provenance") if isinstance(artifact.get("provenance"), dict) else {}
    nested_locators = []
    locators = artifact.get("locators")
    if isinstance(locators, list):
        nested_locators.extend(locator for locator in locators if isinstance(locator, dict))
    for key in ("locator", "element_locator", "layout", "metadata"):
        value = artifact.get(key)
        if isinstance(value, dict):
            nested_locators.append(value)
        value = normalized.get(key) if isinstance(normalized, dict) else None
        if isinstance(value, dict):
            nested_locators.append(value)
    containers = [artifact, normalized, provenance, *nested_locators]

    aliases = {
        "table_id": ("table_id", "table"),
        "figure_id": ("figure_id", "figure"),
        "caption_id": ("caption_id", "caption"),
        "section_id": ("section_id", "section"),
        "row_index": ("row_index", "row"),
        "column_index": ("column_index", "col_index", "col", "column"),
        "source_block_id": ("source_block_id", "source_id", "block_id"),
        "block_id": ("block_id", "source_block_id", "source_id"),
        "char_start": ("char_start", "start_offset", "text_span_start"),
        "char_end": ("char_end", "end_offset", "text_span_end"),
        "page_sha256": ("page_sha256",),
    }
    for output_key, input_keys in aliases.items():
        value = first_present_value(containers, input_keys)
        if value not in (None, "", []):
            locator[output_key] = value
    bbox = first_bbox(artifact)
    if bbox not in (None, "", []):
        locator["bbox"] = bbox
    return clean_optional_fields(locator)


def first_present_value(containers: Iterable[dict[str, Any]], keys: Iterable[str]) -> Any:
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if value not in (None, "", []):
                return value
    return None


def has_element_locator(artifact: dict[str, Any]) -> bool:
    return bool(classify_locator(artifact)["element_locatable"])


def first_source_block_id(artifact: dict[str, Any]) -> str | None:
    anchors = artifact.get("source_anchors") if isinstance(artifact.get("source_anchors"), list) else []
    for anchor in anchors:
        if isinstance(anchor, dict) and anchor.get("source_id") not in (None, ""):
            return str(anchor.get("source_id"))
    locator = extract_layout_locator_without_source_fallback(artifact)
    source_block_id = locator.get("source_block_id")
    return str(source_block_id) if source_block_id not in (None, "", []) else None


def extract_layout_locator_without_source_fallback(artifact: dict[str, Any]) -> dict[str, Any]:
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    provenance = artifact.get("provenance") if isinstance(artifact.get("provenance"), dict) else {}
    return clean_optional_fields(
        {
            "source_block_id": first_present_value([artifact, normalized, provenance], ("source_block_id", "block_id")),
        }
    )


def first_bbox(artifact: dict[str, Any]) -> Any:
    anchors = artifact.get("source_anchors") if isinstance(artifact.get("source_anchors"), list) else []
    for anchor in anchors:
        if isinstance(anchor, dict) and anchor.get("bbox") not in (None, [], ""):
            return anchor.get("bbox")
    return first_present_value([artifact], ("bbox",))


def build_provenance(
    doc_id: str,
    page_index: int,
    source_block_id: str | None = None,
    bbox: Any = None,
    artifact: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    artifact = artifact or {}
    locator = extract_layout_locator(artifact) if artifact else {}
    provenance = {
        "doc_id": doc_id,
        "page_id": page_node_id(doc_id, page_index),
        "page_index": int(page_index),
        "source_block_id": source_block_id or first_source_block_id(artifact),
        "bbox": bbox if bbox not in (None, [], "") else first_bbox(artifact),
        "page_sha256": locator.get("page_sha256"),
        "table_id": extra.get("table_id") or locator.get("table_id"),
        "figure_id": extra.get("figure_id") or locator.get("figure_id"),
        "caption_id": extra.get("caption_id") or locator.get("caption_id"),
        "section_id": extra.get("section_id") or locator.get("section_id"),
        "row_index": extra.get("row_index") or locator.get("row_index"),
        "column_index": extra.get("column_index") or locator.get("column_index"),
    }
    return clean_optional_fields(provenance)


def clean_optional_fields(value: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, child in value.items():
        if child in (None, "", []):
            continue
        if isinstance(child, dict):
            nested = clean_optional_fields(child)
            if nested:
                cleaned[key] = nested
        else:
            cleaned[key] = child
    return cleaned


def artifact_type_matches(artifact: dict[str, Any], names: set[str]) -> bool:
    artifact_type = str(artifact.get("artifact_type") or "").lower()
    return artifact_type in names


def clean_id(value: Any) -> str | None:
    if value in (None, "", []):
        return None
    return str(value)


def parse_block_sequence(source_block_id: str) -> tuple[str, int] | None:
    match = BLOCK_SUFFIX_RE.match(source_block_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def build_manifest(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    debug_edges: list[dict[str, Any]],
    quality_report: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    artifacts_jsonl_path: str | Path = DEFAULT_ARTIFACTS_JSONL,
    command_args: dict[str, Any] | None = None,
    input_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    if input_paths and command_args is None:
        command_args = {key: public_path(value) for key, value in input_paths.items()}
    quality_report = quality_report or build_quality_report(
        nodes=nodes,
        edges=edges,
        debug_edges=debug_edges,
        num_artifacts=sum(1 for node in nodes if node.get("node_type") == "artifact"),
        num_artifacts_with_page_locator=sum(1 for node in nodes if node.get("node_type") == "artifact" and node.get("page_id")),
        num_artifacts_with_source_anchor=0,
        num_artifacts_with_element_locator=0,
        num_proof_trace_eligible=0,
        proof_trace_eligible_by_type={},
        locator_kind_counts={},
        skipped_rule_edges_by_reason={},
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "graph_mode": GRAPH_MODE,
        **deterministic_stage_model_fields("none_rule_only", "stage4_rule_only_document_native_graph"),
        "input_artifacts_path": public_path(artifacts_jsonl_path),
        "input_artifacts_hash": stable_hash_json(artifacts if artifacts is not None else []),
        "nodes_hash": stable_hash_json(nodes),
        "edges_hash": stable_hash_json(edges),
        "debug_edges_hash": stable_hash_json(debug_edges),
        "quality_report_hash": stable_hash_json(quality_report),
        "semantic_edges_enabled": False,
        "formal_edge_types": sorted({edge["edge_type"] for edge in edges}),
        "formal_retrieval_edge_types": sorted(edge_type for edge_type in {edge["edge_type"] for edge in edges} if edge_type in FORMAL_RETRIEVAL_EDGE_TYPES),
        "context_edge_types": sorted(edge_type for edge_type in {edge["edge_type"] for edge in edges} if edge_type in CONTEXT_EDGE_TYPES),
        "debug_edge_types": sorted({edge["edge_type"] for edge in debug_edges}),
        "created_by_script": "scripts/stage4_build_evidence_graph.py",
        "command_args": command_args or {"artifacts_jsonl": public_path(artifacts_jsonl_path)},
    }
    commit = current_git_commit()
    if commit == "unknown":
        manifest["git_commit_unavailable_reason"] = "git_rev_parse_failed"
    else:
        manifest["git_commit"] = commit
    assert_no_forbidden_keys(manifest)
    return manifest


def build_quality_report(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    debug_edges: list[dict[str, Any]],
    num_artifacts: int,
    num_artifacts_with_page_locator: int,
    num_artifacts_with_source_anchor: int,
    num_artifacts_with_element_locator: int,
    num_proof_trace_eligible: int,
    proof_trace_eligible_by_type: dict[str, int],
    locator_kind_counts: dict[str, int],
    skipped_rule_edges_by_reason: dict[str, int],
) -> dict[str, Any]:
    node_type_counts = Counter(str(node.get("node_type")) for node in nodes)
    edge_type_counts = Counter(str(edge.get("edge_type")) for edge in edges)
    debug_edge_type_counts = Counter(str(edge.get("edge_type")) for edge in debug_edges)
    formal_edge_types = sorted(edge_type_counts)
    debug_edge_types = sorted(debug_edge_type_counts)
    denominator = max(1, int(num_artifacts))
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_debug_edges": len(debug_edges),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "debug_edge_type_counts": dict(sorted(debug_edge_type_counts.items())),
        "formal_edge_types": formal_edge_types,
        "formal_retrieval_edge_types": sorted(edge_type for edge_type in formal_edge_types if edge_type in FORMAL_RETRIEVAL_EDGE_TYPES),
        "context_edge_types": sorted(edge_type for edge_type in formal_edge_types if edge_type in CONTEXT_EDGE_TYPES),
        "debug_edge_types": debug_edge_types,
        "semantic_edges_enabled": False,
        "same_record_in_formal_edges": "same_record" in edge_type_counts,
        "same_record_debug_in_formal_edges": "same_record_debug" in edge_type_counts,
        "skipped_rule_edges_by_reason": dict(sorted(skipped_rule_edges_by_reason.items())),
        "pairwise_clique_edges_disabled": True,
        "num_artifacts": int(num_artifacts),
        "num_artifacts_with_page_locator": int(num_artifacts_with_page_locator),
        "num_artifacts_with_source_anchor": int(num_artifacts_with_source_anchor),
        "num_source_anchored": int(num_artifacts_with_source_anchor),
        "source_anchored_rate": int(num_artifacts_with_source_anchor) / denominator,
        "num_artifacts_with_element_locator": int(num_artifacts_with_element_locator),
        "num_element_locatable": int(num_artifacts_with_element_locator),
        "element_locator_rate": int(num_artifacts_with_element_locator) / denominator,
        "num_artifacts_without_element_locator": int(num_artifacts - num_artifacts_with_element_locator),
        "num_proof_trace_eligible": int(num_proof_trace_eligible),
        "proof_trace_eligible_rate": int(num_proof_trace_eligible) / denominator,
        "proof_trace_eligible_by_type": dict(sorted(proof_trace_eligible_by_type.items())),
        "locator_kind_counts": dict(sorted(locator_kind_counts.items())),
    }


def assert_graph_outputs(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    quality_report: dict[str, Any],
    debug_edges: list[dict[str, Any]] | None = None,
) -> None:
    for node in nodes:
        assert_no_forbidden_keys(node)
        if node.get("node_type") not in NODE_TYPES:
            raise ValueError(f"Unexpected node_type: {node.get('node_type')}")
    for edge in edges:
        assert_no_forbidden_keys(edge)
        edge_type = edge.get("edge_type")
        if edge_type not in FORMAL_EDGE_TYPES | CONTEXT_EDGE_TYPES:
            raise ValueError(f"Unexpected formal edge_type: {edge_type}")
        if edge_type in FORBIDDEN_FORMAL_EDGE_TYPES:
            raise ValueError(f"Forbidden formal edge_type: {edge_type}")
        for field in ("edge_id", "source", "target", "edge_type", "provenance", "rule_name", "rule_version", "deterministic"):
            if field not in edge:
                raise ValueError(f"Formal edge missing required field: {field}")
        if edge.get("deterministic") is not True:
            raise ValueError("Formal edge must be deterministic")
    for edge in debug_edges or []:
        assert_no_forbidden_keys(edge)
        edge_type = edge.get("edge_type")
        if edge_type not in DEBUG_EDGE_TYPES:
            raise ValueError(f"Unexpected debug edge_type: {edge_type}")
        if edge_type in SEMANTIC_EDGE_TYPES:
            raise ValueError(f"Semantic debug edge_type is disabled: {edge_type}")
    assert_no_forbidden_keys(quality_report)


def assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_OUTPUT_KEYS:
                raise ValueError(f"Forbidden output key present: {key_text}")
            if key_text.endswith("_token") or key_text in {"api_token", "access_token", "secret_token"}:
                raise ValueError(f"Forbidden token key present: {key_text}")
            assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_keys(child)
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in FORBIDDEN_TEXT_FRAGMENTS:
            if fragment in lowered:
                raise ValueError(f"Forbidden public text fragment present: {fragment}")


def load_evidence_graph(output_dir: str | Path = DEFAULT_OUTPUT_DIR, debug: bool = False) -> dict[str, Any]:
    root = Path(output_dir)
    graph = {
        "nodes": load_jsonl(root / "nodes.jsonl"),
        "edges": load_jsonl(root / "edges.jsonl"),
        "debug_edges": [],
    }
    if debug:
        graph["debug_edges"] = load_jsonl_if_exists(root / "debug_edges.jsonl")
    return graph


def stable_hash_json(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    serialized = serialized.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_input_files(input_paths: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path_value in sorted(input_paths.items()):
        path = Path(path_value)
        if not path.is_file():
            hashes[name] = "missing"
            continue
        hashes[name] = file_sha256(path)
    return hashes


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {input_path} is not a JSON object")
            rows.append(row)
    return rows


def load_jsonl_if_exists(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        return []
    return load_jsonl(input_path)


def load_stage2_record_metadata_if_exists(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        return []
    return load_stage2_record_metadata(input_path)


def load_stage2_record_metadata(path: str | Path) -> list[dict[str, Any]]:
    records = read_json_records(path)
    metadata: list[dict[str, Any]] = []
    for inferred_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        metadata.append({"record_index": coerce_int(record.get("record_index"), fallback=inferred_index), "doc_id": record.get("doc_id")})
    return metadata


def read_json_records(path: str | Path) -> list[Any]:
    input_path = Path(path)
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items"):
            records = value.get(key)
            if isinstance(records, list):
                return records
    raise ValueError(f"Expected a JSON array or object with records in {input_path}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def public_path(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path_obj.name


def question_node_id(record_index: int) -> str:
    return make_node_id("question", record_index)


def artifact_node_id(doc_id: str, page_index: int, artifact_id: str) -> str:
    return make_node_id("artifact", doc_id, page_index, artifact_id)


def page_node_id(doc_id: str, page_index: int) -> str:
    return make_node_id("page", doc_id, page_index)


def source_anchor_node_id(doc_id: str, page_index: int, source_id: str) -> str:
    return make_node_id("source_anchor", doc_id, page_index, source_id)


def section_node_id(doc_id: str, page_index: int, section_id: str) -> str:
    return make_node_id("section", doc_id, page_index, section_id)


def figure_node_id(doc_id: str, page_index: int, figure_id: str) -> str:
    return make_node_id("figure", doc_id, page_index, figure_id)


def caption_node_id(doc_id: str, page_index: int, caption_id: str) -> str:
    return make_node_id("caption", doc_id, page_index, caption_id)


def table_node_id(doc_id: str, page_index: int, table_id: str) -> str:
    return make_node_id("table", doc_id, page_index, table_id)


def table_row_node_id(doc_id: str, page_index: int, table_id: str, row_index: str) -> str:
    return make_node_id("table_row", doc_id, page_index, table_id, row_index)


def table_column_node_id(doc_id: str, page_index: int, table_id: str, column_index: str) -> str:
    return make_node_id("table_column", doc_id, page_index, table_id, column_index)


def edge_id(edge_type: str, source: str, target: str) -> str:
    return make_node_id("edge", edge_type, source, target)


def make_node_id(prefix: str, *parts: Any) -> str:
    encoded = [quote(str(part), safe="") for part in parts]
    return ":".join([prefix, *encoded])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 4B rule-only document-native structural evidence graph.")
    parser.add_argument("--artifacts", "--artifacts-jsonl", dest="artifacts_jsonl", default=DEFAULT_ARTIFACTS_JSONL)
    parser.add_argument("--retrieval", "--retrieval-jsonl", dest="retrieval_jsonl", default=DEFAULT_RETRIEVAL_JSONL)
    parser.add_argument("--stage2-json", default=DEFAULT_STAGE2_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_evidence_graph_build(
        artifacts_jsonl_path=args.artifacts_jsonl,
        retrieval_jsonl_path=args.retrieval_jsonl,
        stage2_json_path=args.stage2_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["quality_report"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
