"""Build a minimal structural evidence graph from Stage 2 artifacts.

The graph is intentionally limited to structure that is already present in
Stage 2 artifact rows and Stage 3A retrieval rows. It does not add semantic
relations or element-level nodes when stable element locators are absent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


DEFAULT_ARTIFACTS_JSONL = "outputs/stage2/clean/artifacts.jsonl"
DEFAULT_RETRIEVAL_JSONL = "outputs/stage3/artifact_retrieval_dryrun/results.jsonl"
DEFAULT_STAGE2_JSON = "outputs/stage2/clean/sample-with-stage2-index.json"
DEFAULT_OUTPUT_DIR = "outputs/stage4/evidence_graph"

NODE_TYPES = {"question", "artifact", "page", "source_anchor"}
EDGE_TYPES = {
    "retrieved_artifact",
    "located_on_page",
    "supported_by_anchor",
    "anchor_on_page",
    "same_record",
    "same_page",
}
SEMANTIC_EDGE_TYPES = {
    "supports",
    "contradicts",
    "derived_from",
    "cites",
    "entails",
    "refutes",
    "semantic_relation",
}
FORBIDDEN_OUTPUT_KEYS = {
    "answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "page_image_path",
    "page_text_path",
    "raw_output",
    "api_key",
    "model_name",
    "config",
}
ELEMENT_LOCATOR_KEYS = {
    "table_id",
    "figure_id",
    "caption_id",
    "bbox",
    "row",
    "row_index",
    "col",
    "col_index",
    "text_span_offset",
    "text_span_start",
    "text_span_end",
    "start_offset",
    "end_offset",
}


def run_evidence_graph_build(
    artifacts_jsonl_path: str | Path = DEFAULT_ARTIFACTS_JSONL,
    retrieval_jsonl_path: str | Path = DEFAULT_RETRIEVAL_JSONL,
    stage2_json_path: str | Path = DEFAULT_STAGE2_JSON,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build and write the Stage 4A structural evidence graph."""

    artifacts = load_jsonl(artifacts_jsonl_path)
    retrieval_rows = load_jsonl(retrieval_jsonl_path)
    stage2_records = load_stage2_record_metadata(stage2_json_path)
    graph = build_evidence_graph(
        artifacts=artifacts,
        retrieval_rows=retrieval_rows,
        stage2_records=stage2_records,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    nodes_path = output_root / "nodes.jsonl"
    edges_path = output_root / "edges.jsonl"
    quality_report_path = output_root / "quality_report.json"
    write_jsonl(nodes_path, graph["nodes"])
    write_jsonl(edges_path, graph["edges"])
    quality_report_path.write_text(
        json.dumps(graph["quality_report"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "quality_report_path": str(quality_report_path),
        "quality_report": graph["quality_report"],
    }


def build_evidence_graph(
    artifacts: Iterable[dict[str, Any]],
    retrieval_rows: Iterable[dict[str, Any]],
    stage2_records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return structural graph nodes, edges, and a compact quality report."""

    record_metadata = index_record_metadata(stage2_records or [])
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    artifact_refs: list[dict[str, Any]] = []
    artifact_by_key: dict[tuple[int, str, int, str], str] = {}
    artifact_by_record_page_id: dict[tuple[int, int, str], list[str]] = defaultdict(list)

    num_artifacts = 0
    num_artifacts_with_page_locator = 0
    num_artifacts_with_source_anchor = 0
    num_artifacts_with_element_locator = 0

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        num_artifacts += 1
        record_index = coerce_int(artifact.get("record_index"), fallback=-1)
        doc_id = str(artifact.get("doc_id") or "")
        page_index = coerce_int(artifact.get("page_index"), fallback=-1)
        artifact_id = str(artifact.get("artifact_id") or "")
        if not doc_id or page_index < 0 or not artifact_id:
            continue

        artifact_node = artifact_node_id(record_index, doc_id, page_index, artifact_id)
        add_node(
            nodes,
            {
                "node_id": artifact_node,
                "node_type": "artifact",
                "record_index": record_index,
                "doc_id": doc_id,
                "page_index": page_index,
                "artifact_id": artifact_id,
                "artifact_type": artifact.get("artifact_type"),
                "modality": artifact.get("modality"),
            },
        )
        page_node = page_node_id(doc_id, page_index)
        add_node(nodes, {"node_id": page_node, "node_type": "page", "doc_id": doc_id, "page_index": page_index})
        add_edge(
            edges,
            artifact_node,
            page_node,
            "located_on_page",
            evidence={"source": "artifact_field", "field": "page_index"},
        )
        num_artifacts_with_page_locator += 1

        anchors = artifact.get("source_anchors") if isinstance(artifact.get("source_anchors"), list) else []
        valid_anchor_seen = False
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            source_id = anchor.get("source_id")
            if source_id is None or str(source_id) == "":
                continue
            valid_anchor_seen = True
            anchor_page_index = coerce_int(anchor.get("page_index"), fallback=page_index)
            anchor_node = source_anchor_node_id(doc_id, anchor_page_index, str(source_id))
            anchor_page_node = page_node_id(doc_id, anchor_page_index)
            add_node(
                nodes,
                {
                    "node_id": anchor_node,
                    "node_type": "source_anchor",
                    "doc_id": doc_id,
                    "page_index": anchor_page_index,
                },
            )
            add_node(
                nodes,
                {
                    "node_id": anchor_page_node,
                    "node_type": "page",
                    "doc_id": doc_id,
                    "page_index": anchor_page_index,
                },
            )
            add_edge(
                edges,
                artifact_node,
                anchor_node,
                "supported_by_anchor",
                evidence={"source": "artifact_field", "field": "source_anchors"},
            )
            add_edge(
                edges,
                anchor_node,
                anchor_page_node,
                "anchor_on_page",
                evidence={"source": "artifact_field", "field": "source_anchors.page_index"},
            )
        if valid_anchor_seen:
            num_artifacts_with_source_anchor += 1
        if has_element_locator(artifact):
            num_artifacts_with_element_locator += 1

        artifact_refs.append(
            {
                "node_id": artifact_node,
                "record_index": record_index,
                "doc_id": doc_id,
                "page_index": page_index,
                "artifact_id": artifact_id,
            }
        )
        artifact_by_key[(record_index, doc_id, page_index, artifact_id)] = artifact_node
        artifact_by_record_page_id[(record_index, page_index, artifact_id)].append(artifact_node)

    add_retrieval_edges(
        nodes=nodes,
        edges=edges,
        retrieval_rows=retrieval_rows,
        artifact_by_key=artifact_by_key,
        artifact_by_record_page_id=artifact_by_record_page_id,
        record_metadata=record_metadata,
    )
    add_pair_edges(edges, artifact_refs)

    node_rows = sorted(nodes.values(), key=lambda item: item["node_id"])
    edge_rows = sorted(edges.values(), key=lambda item: (item["source"], item["target"], item["edge_type"]))
    quality_report = build_quality_report(
        nodes=node_rows,
        edges=edge_rows,
        num_artifacts=num_artifacts,
        num_artifacts_with_page_locator=num_artifacts_with_page_locator,
        num_artifacts_with_source_anchor=num_artifacts_with_source_anchor,
        num_artifacts_with_element_locator=num_artifacts_with_element_locator,
    )
    assert_graph_outputs(node_rows, edge_rows, quality_report)
    return {"nodes": node_rows, "edges": edge_rows, "quality_report": quality_report}


def add_retrieval_edges(
    nodes: dict[str, dict[str, Any]],
    edges: dict[tuple[str, str, str], dict[str, Any]],
    retrieval_rows: Iterable[dict[str, Any]],
    artifact_by_key: dict[tuple[int, str, int, str], str],
    artifact_by_record_page_id: dict[tuple[int, int, str], list[str]],
    record_metadata: dict[int, dict[str, Any]],
) -> None:
    for row in retrieval_rows:
        if not isinstance(row, dict):
            continue
        record_index = coerce_int(row.get("record_index"), fallback=-1)
        retrieved_artifacts = row.get("retrieved_artifacts")
        if not isinstance(retrieved_artifacts, list) or not retrieved_artifacts:
            continue
        row_doc_id = str(row.get("doc_id") or record_metadata.get(record_index, {}).get("doc_id") or "")
        question_node = question_node_id(record_index)
        question_added = False
        for retrieved in retrieved_artifacts:
            if not isinstance(retrieved, dict):
                continue
            artifact_id = retrieved.get("artifact_id")
            if artifact_id is None:
                continue
            page_index = coerce_int(retrieved.get("page_index"), fallback=-1)
            if page_index < 0:
                continue
            target = None
            if row_doc_id:
                target = artifact_by_key.get((record_index, row_doc_id, page_index, str(artifact_id)))
            if target is None:
                candidates = artifact_by_record_page_id.get((record_index, page_index, str(artifact_id)), [])
                if len(candidates) == 1:
                    target = candidates[0]
            if target is None:
                continue
            if not question_added:
                question_node_data = {"node_id": question_node, "node_type": "question", "record_index": record_index}
                if row_doc_id:
                    question_node_data["doc_id"] = row_doc_id
                add_node(nodes, question_node_data)
                question_added = True
            add_edge(
                edges,
                question_node,
                target,
                "retrieved_artifact",
                evidence={"source": "stage3a_retrieval", "field": "retrieved_artifacts"},
            )


def add_pair_edges(edges: dict[tuple[str, str, str], dict[str, Any]], artifact_refs: list[dict[str, Any]]) -> None:
    by_record: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for ref in artifact_refs:
        by_record[int(ref["record_index"])].append(ref)
        by_page[(str(ref["doc_id"]), int(ref["page_index"]))].append(ref)

    for refs in by_record.values():
        for left, right in sorted_pairs(refs):
            add_edge(
                edges,
                left["node_id"],
                right["node_id"],
                "same_record",
                evidence={"source": "artifact_field", "field": "record_index"},
            )
    for refs in by_page.values():
        for left, right in sorted_pairs(refs):
            add_edge(
                edges,
                left["node_id"],
                right["node_id"],
                "same_page",
                evidence={"source": "artifact_field", "field": "doc_id,page_index"},
            )


def sorted_pairs(refs: list[dict[str, Any]]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(refs, key=lambda item: item["node_id"])
    return combinations(ordered, 2)


def add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    cleaned = {key: value for key, value in node.items() if value is not None}
    node_type = cleaned.get("node_type")
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unsupported node_type: {node_type}")
    nodes[str(cleaned["node_id"])] = cleaned


def add_edge(
    edges: dict[tuple[str, str, str], dict[str, Any]],
    source: str,
    target: str,
    edge_type: str,
    evidence: dict[str, str],
) -> None:
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Unsupported edge_type: {edge_type}")
    if edge_type in SEMANTIC_EDGE_TYPES:
        raise ValueError(f"Semantic edge_type is disabled: {edge_type}")
    key = (source, target, edge_type)
    edges[key] = {"source": source, "target": target, "edge_type": edge_type, "evidence": dict(evidence)}


def has_element_locator(artifact: dict[str, Any]) -> bool:
    anchors = artifact.get("source_anchors") if isinstance(artifact.get("source_anchors"), list) else []
    for anchor in anchors:
        if isinstance(anchor, dict) and anchor.get("bbox") not in (None, [], ""):
            return True
    normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), dict) else {}
    for key in ELEMENT_LOCATOR_KEYS - {"bbox"}:
        if key in artifact and artifact.get(key) not in (None, ""):
            return True
        if key in normalized and normalized.get(key) not in (None, ""):
            return True
    return False


def build_quality_report(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    num_artifacts: int,
    num_artifacts_with_page_locator: int,
    num_artifacts_with_source_anchor: int,
    num_artifacts_with_element_locator: int,
) -> dict[str, Any]:
    node_type_counts = Counter(str(node.get("node_type")) for node in nodes)
    edge_type_counts = Counter(str(edge.get("edge_type")) for edge in edges)
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "num_artifacts": int(num_artifacts),
        "num_artifacts_with_page_locator": int(num_artifacts_with_page_locator),
        "num_artifacts_with_source_anchor": int(num_artifacts_with_source_anchor),
        "num_artifacts_with_element_locator": int(num_artifacts_with_element_locator),
        "num_artifacts_without_element_locator": int(num_artifacts - num_artifacts_with_element_locator),
        "semantic_edges_enabled": False,
    }


def assert_graph_outputs(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], quality_report: dict[str, Any]) -> None:
    for node in nodes:
        assert_no_forbidden_keys(node)
        if node.get("node_type") not in NODE_TYPES:
            raise ValueError(f"Unexpected node_type: {node.get('node_type')}")
    for edge in edges:
        assert_no_forbidden_keys(edge)
        edge_type = edge.get("edge_type")
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unexpected edge_type: {edge_type}")
        if edge_type in SEMANTIC_EDGE_TYPES:
            raise ValueError(f"Semantic edge_type is disabled: {edge_type}")
    assert_no_forbidden_keys(quality_report)


def assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_OUTPUT_KEYS:
                raise ValueError(f"Forbidden output key present: {key}")
            assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_keys(child)


def index_record_metadata(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for inferred_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        record_index = coerce_int(record.get("record_index"), fallback=inferred_index)
        doc_id = record.get("doc_id")
        indexed[record_index] = {"record_index": record_index, "doc_id": doc_id}
    return indexed


def load_stage2_record_metadata(path: str | Path) -> list[dict[str, Any]]:
    records = read_json_records(path)
    metadata: list[dict[str, Any]] = []
    for inferred_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        metadata.append(
            {
                "record_index": coerce_int(record.get("record_index"), fallback=inferred_index),
                "doc_id": record.get("doc_id"),
            }
        )
    return metadata


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


def question_node_id(record_index: int) -> str:
    return make_node_id("question", record_index)


def artifact_node_id(record_index: int, doc_id: str, page_index: int, artifact_id: str) -> str:
    return make_node_id("artifact", record_index, doc_id, page_index, artifact_id)


def page_node_id(doc_id: str, page_index: int) -> str:
    return make_node_id("page", doc_id, page_index)


def source_anchor_node_id(doc_id: str, page_index: int, source_id: str) -> str:
    return make_node_id("source_anchor", doc_id, page_index, source_id)


def make_node_id(prefix: str, *parts: Any) -> str:
    encoded = [quote(str(part), safe="") for part in parts]
    return ":".join([prefix, *encoded])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 4A minimal structural evidence graph.")
    parser.add_argument("--artifacts-jsonl", default=DEFAULT_ARTIFACTS_JSONL)
    parser.add_argument("--retrieval-jsonl", default=DEFAULT_RETRIEVAL_JSONL)
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
