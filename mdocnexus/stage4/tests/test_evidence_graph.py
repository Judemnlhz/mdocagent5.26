"""Tests for Stage 4A minimal structural evidence graph."""

from __future__ import annotations

import unittest

from mdocnexus.stage4.evidence_graph import build_evidence_graph


class EvidenceGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = [
            make_artifact("a1", record_index=0, doc_id="doc.pdf", page_index=2, source_id="p002_text_0000"),
            make_artifact("a2", record_index=0, doc_id="doc.pdf", page_index=2, source_id="p002_text_0001"),
            make_artifact("a3", record_index=0, doc_id="doc.pdf", page_index=3, source_id="p003_image"),
            make_artifact("b1", record_index=1, doc_id="other.pdf", page_index=0, source_id="p000_text_0000"),
        ]
        self.retrieval_rows = [
            {
                "record_index": 0,
                "doc_id": "doc.pdf",
                "question": "What evidence is visible?",
                "retrieved_artifacts": [
                    {"artifact_id": "a1", "page_index": 2, "score": 1.0},
                    {"artifact_id": "a3", "page_index": 3, "score": 0.5},
                ],
            },
            {
                "record_index": 1,
                "doc_id": "other.pdf",
                "question": "What other evidence is visible?",
                "retrieved_artifacts": [{"artifact_id": "b1", "page_index": 0, "score": 1.0}],
            },
        ]
        self.graph = build_evidence_graph(
            artifacts=self.artifacts,
            retrieval_rows=self.retrieval_rows,
            stage2_records=[{"record_index": 0, "doc_id": "doc.pdf"}, {"record_index": 1, "doc_id": "other.pdf"}],
        )

    def test_artifact_generates_artifact_node(self) -> None:
        artifact_nodes = nodes_of_type(self.graph, "artifact")
        self.assertEqual(len(artifact_nodes), 4)
        self.assertTrue(any(node.get("artifact_id") == "a1" for node in artifact_nodes))

    def test_question_generates_question_node(self) -> None:
        question_nodes = nodes_of_type(self.graph, "question")
        self.assertEqual({node["record_index"] for node in question_nodes}, {0, 1})

    def test_page_generates_page_node(self) -> None:
        page_nodes = nodes_of_type(self.graph, "page")
        self.assertTrue(any(node.get("doc_id") == "doc.pdf" and node.get("page_index") == 2 for node in page_nodes))
        self.assertTrue(any(node.get("doc_id") == "doc.pdf" and node.get("page_index") == 3 for node in page_nodes))

    def test_source_anchors_generate_source_anchor_nodes(self) -> None:
        anchor_nodes = nodes_of_type(self.graph, "source_anchor")
        self.assertEqual(len(anchor_nodes), 4)
        self.assertTrue(any("p002_text_0000" in node["node_id"] for node in anchor_nodes))

    def test_artifact_to_page_edge(self) -> None:
        self.assertTrue(has_edge(self.graph, "located_on_page"))
        edge = first_edge(self.graph, "located_on_page")
        self.assertEqual(edge["evidence"], {"source": "artifact_field", "field": "page_index"})

    def test_artifact_to_source_anchor_edge(self) -> None:
        self.assertTrue(has_edge(self.graph, "supported_by_anchor"))
        edge = first_edge(self.graph, "supported_by_anchor")
        self.assertEqual(edge["evidence"], {"source": "artifact_field", "field": "source_anchors"})

    def test_source_anchor_to_page_edge(self) -> None:
        self.assertTrue(has_edge(self.graph, "anchor_on_page"))
        edge = first_edge(self.graph, "anchor_on_page")
        self.assertEqual(edge["evidence"], {"source": "artifact_field", "field": "source_anchors.page_index"})

    def test_same_record_and_same_page_edges(self) -> None:
        self.assertGreaterEqual(count_edges(self.graph, "same_record"), 3)
        self.assertGreaterEqual(count_edges(self.graph, "same_page"), 1)

    def test_no_precise_nodes_without_element_locator(self) -> None:
        node_types = {node["node_type"] for node in self.graph["nodes"]}
        self.assertEqual(node_types, {"question", "artifact", "page", "source_anchor"})
        self.assertEqual(self.graph["quality_report"]["num_artifacts_with_element_locator"], 0)
        self.assertEqual(self.graph["quality_report"]["num_artifacts_without_element_locator"], 4)

    def test_nodes_and_edges_do_not_contain_local_paths(self) -> None:
        serialized = repr(self.graph["nodes"]) + repr(self.graph["edges"])
        self.assertNotIn("/tmp/", serialized)
        self.assertNotIn("page_image_path", serialized)
        self.assertNotIn("page_text_path", serialized)

    def test_no_semantic_edges(self) -> None:
        forbidden = {"supports", "contradicts", "derived_from", "cites", "entails", "refutes", "semantic_relation"}
        edge_types = {edge["edge_type"] for edge in self.graph["edges"]}
        self.assertFalse(edge_types & forbidden)
        self.assertFalse(self.graph["quality_report"]["semantic_edges_enabled"])

    def test_retrieved_artifact_edge(self) -> None:
        self.assertEqual(count_edges(self.graph, "retrieved_artifact"), 3)


def make_artifact(artifact_id: str, record_index: int, doc_id: str, page_index: int, source_id: str) -> dict:
    return {
        "record_index": record_index,
        "artifact_id": artifact_id,
        "doc_id": doc_id,
        "page_index": page_index,
        "artifact_type": "text_span",
        "modality": "text",
        "content": "evidence content",
        "normalized_content": {},
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


def nodes_of_type(graph: dict, node_type: str) -> list[dict]:
    return [node for node in graph["nodes"] if node.get("node_type") == node_type]


def has_edge(graph: dict, edge_type: str) -> bool:
    return any(edge.get("edge_type") == edge_type for edge in graph["edges"])


def count_edges(graph: dict, edge_type: str) -> int:
    return sum(1 for edge in graph["edges"] if edge.get("edge_type") == edge_type)


def first_edge(graph: dict, edge_type: str) -> dict:
    for edge in graph["edges"]:
        if edge.get("edge_type") == edge_type:
            return edge
    raise AssertionError(f"missing edge_type={edge_type}")


if __name__ == "__main__":
    unittest.main()
