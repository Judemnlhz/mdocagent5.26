"""Tests for retrieval-only evaluation metrics."""

from __future__ import annotations

import unittest

from mdocnexus.evaluation.retrieval_metrics import (
    evaluate_stage3_retrieval,
    evaluate_stage4_graph_expansion,
)


class RetrievalMetricsTest(unittest.TestCase):
    def test_stage3_recall_uses_gold_only_in_evaluation(self) -> None:
        report, per_query = evaluate_stage3_retrieval(
            retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}],
            artifacts=[make_artifact("a1", page_index=1, artifact_type="text_span", modality="text")],
            records=[{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[2]"}],
            k_values=(1,),
        )

        self.assertTrue(report["evaluation_only"])
        self.assertEqual(report["num_queries_with_gold"], 1)
        self.assertEqual(report["recall_at_k_by_page"]["1"], 1.0)
        self.assertEqual(per_query[0]["coverage_at_k_by_page"]["1"], 1.0)

    def test_stage4_expansion_uses_formal_edges_only(self) -> None:
        report, _ = evaluate_stage4_graph_expansion(
            retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}],
            artifacts=[
                make_artifact("a1", page_index=0),
                make_artifact("a2", page_index=1),
            ],
            records=[{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[2]"}],
            formal_edges=[
                {
                    "source": "artifact:doc.pdf:0:a1",
                    "target": "artifact:doc.pdf:1:a2",
                    "edge_type": "adjacent_page",
                }
            ],
            k_values=(2,),
        )

        self.assertFalse(report["used_debug_edges"])
        self.assertFalse(report["used_semantic_edges"])
        self.assertEqual(report["flat_recall_at_k"]["2"], 0.0)
        self.assertEqual(report["graph_recall_at_k"]["2"], 1.0)
        self.assertEqual(report["delta_recall_at_k"]["2"], 1.0)
        self.assertEqual(report["edge_types_used"], ["adjacent_page"])

    def test_stage4_expansion_rejects_semantic_edges(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_stage4_graph_expansion(
                retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"]}],
                artifacts=[make_artifact("a1")],
                records=[{"record_index": 0, "evidence_pages": "[1]"}],
                formal_edges=[{"source": "artifact:doc.pdf:0:a1", "target": "artifact:doc.pdf:0:a2", "edge_type": "supports"}],
            )


def make_artifact(
    artifact_id: str,
    page_index: int = 0,
    artifact_type: str = "text_span",
    modality: str = "text",
) -> dict:
    return {
        "artifact_id": artifact_id,
        "doc_id": "doc.pdf",
        "page_index": page_index,
        "artifact_type": artifact_type,
        "modality": modality,
    }


if __name__ == "__main__":
    unittest.main()
