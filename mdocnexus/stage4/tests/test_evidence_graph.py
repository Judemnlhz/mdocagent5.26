"""Tests for Stage 4B rule-only document-native graph construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdocnexus.stage4.evidence_graph import (
    CONTEXT_EDGE_TYPES,
    FORMAL_EDGE_TYPES,
    build_evidence_graph,
    build_manifest,
    load_evidence_graph,
    run_evidence_graph_build,
    stable_hash_json,
)
from mdocnexus.stage4.locator_policy import classify_locator, is_proof_trace_eligible


FORBIDDEN_FORMAL_TYPES = {
    "same_record",
    "same_record_debug",
    "supports",
    "contradicts",
    "derived_from",
    "semantic_relation",
    "entails",
    "refutes",
    "answer_supports",
    "proof_supports",
}


class EvidenceGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = [
            make_artifact("a1", record_index=0, doc_id="doc.pdf", page_index=0, source_id="p000_text_0000"),
            make_artifact("a2", record_index=0, doc_id="doc.pdf", page_index=0, source_id="p000_text_0001"),
            make_artifact("a1_dup", record_index=0, doc_id="doc.pdf", page_index=0, source_id="p000_text_0000"),
            make_artifact("a3", record_index=0, doc_id="doc.pdf", page_index=1, source_id="p001_text_0000"),
            make_artifact("b1", record_index=1, doc_id="other.pdf", page_index=1, source_id="p001_text_0000"),
            make_artifact(
                "cell1",
                record_index=0,
                doc_id="doc.pdf",
                page_index=0,
                source_id="p000_table_0002",
                artifact_type="table_cell",
                normalized_content={"table_id": "t1", "row_index": 2, "column_index": 3},
            ),
            make_artifact(
                "cap1",
                record_index=0,
                doc_id="doc.pdf",
                page_index=0,
                source_id="p000_caption_0003",
                artifact_type="caption",
                normalized_content={"figure_id": "fig1", "caption_id": "caption1"},
            ),
        ]
        self.retrieval_rows = [
            {
                "query_hash": "query_hash_1",
                "retrieval_method": "deterministic_lexical",
                "retrieved_artifact_ids": ["a1", "a2", "a1_dup"],
            }
        ]
        self.graph = build_evidence_graph(artifacts=self.artifacts, retrieval_rows=self.retrieval_rows, stage2_records=[])

    def test_formal_edges_use_only_allowed_rule_types(self) -> None:
        edge_types = {edge["edge_type"] for edge in self.graph["edges"]}
        self.assertTrue(edge_types <= FORMAL_EDGE_TYPES | CONTEXT_EDGE_TYPES)
        self.assertFalse(edge_types & FORBIDDEN_FORMAL_TYPES)
        self.assertFalse(self.graph["quality_report"]["semantic_edges_enabled"])

    def test_formal_edges_never_include_same_record_debug(self) -> None:
        formal_types = {edge["edge_type"] for edge in self.graph["edges"]}
        debug_types = {edge["edge_type"] for edge in self.graph["debug_edges"]}

        self.assertNotIn("same_record", formal_types)
        self.assertNotIn("same_record_debug", formal_types)
        self.assertIn("same_record_debug", debug_types)
        self.assertFalse(self.graph["quality_report"]["same_record_in_formal_edges"])
        self.assertFalse(self.graph["quality_report"]["same_record_debug_in_formal_edges"])

    def test_artifact_node_id_is_independent_of_record_or_input_order(self) -> None:
        first = build_evidence_graph(
            artifacts=[
                make_artifact("stable", record_index=7, doc_id="doc.pdf", page_index=3),
                make_artifact("other", record_index=1, doc_id="doc.pdf", page_index=3),
            ],
            retrieval_rows=[],
            stage2_records=[],
        )
        second = build_evidence_graph(
            artifacts=[
                make_artifact("other", record_index=99, doc_id="doc.pdf", page_index=3),
                make_artifact("stable", record_index=42, doc_id="doc.pdf", page_index=3),
            ],
            retrieval_rows=[],
            stage2_records=[],
        )
        first_ids = {node["node_id"] for node in first["nodes"] if node.get("artifact_id") == "stable"}
        second_ids = {node["node_id"] for node in second["nodes"] if node.get("artifact_id") == "stable"}

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_ids, {"artifact:doc.pdf:3:stable"})

    def test_formal_graph_contains_no_record_index(self) -> None:
        formal_text = json.dumps({"nodes": self.graph["nodes"], "edges": self.graph["edges"]}, ensure_ascii=False)

        self.assertNotIn("record_index", formal_text)
        self.assertNotIn("same_record", formal_text)
        self.assertNotIn("same_record_debug", formal_text)

    def test_formal_edges_include_required_audit_fields(self) -> None:
        required = {"edge_id", "source", "target", "edge_type", "provenance", "rule_name", "rule_version", "deterministic"}
        for edge in self.graph["edges"]:
            self.assertTrue(required <= set(edge))
            self.assertTrue(edge["deterministic"])

    def test_loader_reads_debug_edges_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts_path = root / "artifacts.jsonl"
            retrieval_path = root / "retrieval.jsonl"
            write_jsonl(artifacts_path, self.artifacts)
            write_jsonl(retrieval_path, self.retrieval_rows)
            output_dir = root / "graph"

            run_evidence_graph_build(artifacts_path, retrieval_jsonl_path=retrieval_path, output_dir=output_dir)
            formal_only = load_evidence_graph(output_dir)
            with_debug = load_evidence_graph(output_dir, debug=True)

        self.assertEqual(formal_only["debug_edges"], [])
        self.assertGreater(len(with_debug["debug_edges"]), 0)
        self.assertTrue(all(edge["edge_type"] != "same_record_debug" for edge in formal_only["edges"]))

    def test_missing_metadata_does_not_guess_caption_or_table_edges(self) -> None:
        graph = build_evidence_graph(
            artifacts=[
                make_artifact("figure_without_caption", artifact_type="figure"),
                make_artifact("cell_without_table", artifact_type="table_cell", normalized_content={"row_index": 1, "column_index": 2}),
            ],
            retrieval_rows=[],
            stage2_records=[],
        )
        edge_types = {edge["edge_type"] for edge in graph["edges"]}
        skipped = graph["quality_report"]["skipped_rule_edges_by_reason"]

        self.assertNotIn("caption_of", edge_types)
        self.assertNotIn("figure_has_caption", edge_types)
        self.assertNotIn("table_contains_cell", edge_types)
        self.assertGreaterEqual(skipped.get("caption_figure_missing_explicit_pair", 0), 1)
        self.assertGreaterEqual(skipped.get("table_contains_cell_missing_table_id", 0), 1)

    def test_explicit_table_cell_locator_generates_table_edges(self) -> None:
        self.assertTrue(has_edge(self.graph, "table_contains_cell"))
        self.assertTrue(has_edge(self.graph, "row_contains_cell"))
        self.assertTrue(has_edge(self.graph, "column_contains_cell"))

    def test_explicit_figure_caption_locator_generates_caption_edges(self) -> None:
        self.assertTrue(has_edge(self.graph, "caption_of"))
        self.assertTrue(has_edge(self.graph, "figure_has_caption"))

    def test_adjacent_page_edges_stay_within_same_doc(self) -> None:
        adjacent_edges = [edge for edge in self.graph["edges"] if edge["edge_type"] == "adjacent_page"]
        self.assertTrue(adjacent_edges)
        for edge in adjacent_edges:
            provenance = edge["provenance"]
            self.assertEqual(provenance["doc_id"], "doc.pdf")
            self.assertIn("doc.pdf", edge["source"])
            self.assertIn("doc.pdf", edge["target"])
            self.assertNotIn("other.pdf", edge["source"] + edge["target"])

    def test_same_source_and_next_block_edges_are_structural(self) -> None:
        self.assertTrue(has_edge(self.graph, "same_page"))
        self.assertTrue(has_edge(self.graph, "same_source_block"))
        self.assertTrue(has_edge(self.graph, "next_block"))

    def test_edge_hash_is_stable_when_json_key_order_changes(self) -> None:
        reordered = [dict(reversed(list(edge.items()))) for edge in self.graph["edges"]]
        self.assertEqual(stable_hash_json(self.graph["edges"]), stable_hash_json(reordered))

    def test_modifying_formal_edge_changes_edges_hash(self) -> None:
        modified = [dict(edge) for edge in self.graph["edges"]]
        modified[0] = dict(modified[0], rule_name="changed_rule")
        self.assertNotEqual(stable_hash_json(self.graph["edges"]), stable_hash_json(modified))

    def test_debug_edge_hash_changes_independently_from_edges_hash(self) -> None:
        modified_debug = [dict(edge) for edge in self.graph["debug_edges"]]
        modified_debug[0] = dict(modified_debug[0], rule_name="changed_debug_rule")

        self.assertEqual(stable_hash_json(self.graph["edges"]), stable_hash_json(self.graph["edges"]))
        self.assertNotEqual(stable_hash_json(self.graph["debug_edges"]), stable_hash_json(modified_debug))

    def test_manifest_contains_stage4b_hashes_and_modes(self) -> None:
        manifest = build_manifest(
            self.graph["nodes"],
            self.graph["edges"],
            self.graph["debug_edges"],
            quality_report=self.graph["quality_report"],
            artifacts=self.artifacts,
            artifacts_jsonl_path="artifacts.jsonl",
        )

        self.assertEqual(manifest["graph_mode"], "rule_only_document_native_structural")
        self.assertEqual(len(manifest["nodes_hash"]), 64)
        self.assertEqual(len(manifest["edges_hash"]), 64)
        self.assertEqual(len(manifest["debug_edges_hash"]), 64)
        self.assertEqual(len(manifest["quality_report_hash"]), 64)
        self.assertFalse(manifest["semantic_edges_enabled"])

    def test_quality_report_splits_retrieval_context_and_debug_edges(self) -> None:
        report = self.graph["quality_report"]

        self.assertTrue(report["pairwise_clique_edges_disabled"])
        self.assertNotIn("same_doc", report["formal_edge_types"])
        self.assertIn("same_page", report["context_edge_types"])
        self.assertNotIn("same_page", report["formal_retrieval_edge_types"])
        self.assertEqual(report["debug_edge_types"], ["same_record_debug"])

    def test_full_page_anchor_is_not_proof_trace_eligible(self) -> None:
        artifact = make_artifact("visual", artifact_type="visual_observation", source_id="p000_full_page_image")
        artifact["source_anchors"][0]["anchor_type"] = "full_page_image"

        self.assertTrue(classify_locator(artifact)["source_anchored"])
        self.assertFalse(is_proof_trace_eligible(artifact))

    def test_page_sha256_alone_is_not_element_locator(self) -> None:
        artifact = make_artifact("hash_only")
        artifact["source_anchors"] = []
        artifact["page_sha256"] = "abc"

        self.assertFalse(classify_locator(artifact)["element_locatable"])

    def test_text_span_needs_block_id_and_offset_or_bbox_for_proof(self) -> None:
        no_offset = make_artifact("text_no_offset", artifact_type="text_span")
        with_offset = make_artifact("text_with_offset", artifact_type="text_span", normalized_content={"start_offset": 1, "end_offset": 5})

        self.assertFalse(is_proof_trace_eligible(no_offset))
        self.assertTrue(is_proof_trace_eligible(with_offset))

    def test_table_cell_requires_table_row_and_column(self) -> None:
        incomplete = make_artifact("cell_bad", artifact_type="table_cell", normalized_content={"table_id": "t1", "row_index": 1})
        complete = make_artifact(
            "cell_good",
            artifact_type="table_cell",
            normalized_content={"table_id": "t1", "row_index": 1, "column_index": 2},
        )

        self.assertFalse(is_proof_trace_eligible(incomplete))
        self.assertTrue(is_proof_trace_eligible(complete))


def make_artifact(
    artifact_id: str,
    record_index: int = 0,
    doc_id: str = "doc.pdf",
    page_index: int = 0,
    source_id: str = "p000_text_0000",
    artifact_type: str = "text_span",
    normalized_content: dict | None = None,
) -> dict:
    return {
        "record_index": record_index,
        "artifact_id": artifact_id,
        "doc_id": doc_id,
        "page_index": page_index,
        "artifact_type": artifact_type,
        "modality": "text",
        "content": "evidence content",
        "normalized_content": normalized_content or {},
        "source_anchors": [
            {
                "source_id": source_id,
                "anchor_type": "text_block",
                "page_index": page_index,
                "bbox": None,
            }
        ],
        "provenance": {"op": "ATOM", "sources": [source_id]},
        "validation_status": "anchored",
        "page_image_path": "/tmp/should_not_leak.png",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def has_edge(graph: dict, edge_type: str) -> bool:
    return any(edge.get("edge_type") == edge_type for edge in graph["edges"])


if __name__ == "__main__":
    unittest.main()
