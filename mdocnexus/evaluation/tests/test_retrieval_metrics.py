"""Tests for retrieval-only evaluation metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdocnexus.evaluation.retrieval_metrics import (
    evaluate_stage3_retrieval,
    evaluate_stage4_graph_expansion,
)
from scripts.eval_stage3_retrieval import main as eval_stage3_main
from scripts.eval_stage4_graph_expansion import main as eval_stage4_main


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


    def test_page_neighborhood_ignores_same_page_clique_and_computes_delta(self) -> None:
        report, per_query = evaluate_stage4_graph_expansion(
            retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}],
            artifacts=[make_artifact("a1", page_index=0), make_artifact("a2", page_index=0), make_artifact("a3", page_index=1)],
            records=[{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[2]"}],
            formal_edges=[
                {"source": "artifact:doc.pdf:0:a1", "target": "page:doc.pdf:0", "edge_type": "located_on_page"},
                {"source": "artifact:doc.pdf:1:a3", "target": "page:doc.pdf:1", "edge_type": "located_on_page"},
                {"source": "page:doc.pdf:0", "target": "page:doc.pdf:1", "edge_type": "adjacent_page"},
                {"source": "artifact:doc.pdf:0:a1", "target": "artifact:doc.pdf:0:a2", "edge_type": "same_page"},
            ],
            k_values=(2,),
            expansion_mode="page_neighborhood",
        )

        self.assertEqual(report["expansion_mode"], "page_neighborhood")
        self.assertNotIn("same_page", report["edge_types_used"])
        self.assertEqual(per_query[0]["expanded_num_retrieved"], 2)
        self.assertEqual(report["flat_recall_at_k"]["2"], 0.0)
        self.assertEqual(report["expanded_recall_at_k"]["2"], 1.0)
        self.assertEqual(report["delta_recall_at_k"]["2"], 1.0)
        self.assertEqual(report["expansion_factor"], 2.0)
        self.assertEqual(report["avg_added_artifacts"], 1.0)
        self.assertEqual(report["avg_flat_artifacts"], 1.0)
        self.assertEqual(report["avg_expanded_artifacts"], 2.0)
        self.assertEqual(report["expansion_ratio"], 2.0)
        self.assertEqual(report["added_ratio"], 1.0)
        self.assertEqual(report["added_gold_page_hit_rate"], 1.0)

    def test_source_anchor_neighborhood_ignores_debug_edges(self) -> None:
        report, per_query = evaluate_stage4_graph_expansion(
            retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}],
            artifacts=[make_artifact("a1", page_index=0), make_artifact("a2", page_index=0)],
            records=[{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[1]"}],
            formal_edges=[
                {"source": "artifact:doc.pdf:0:a1", "target": "anchor:doc.pdf:0:s1", "edge_type": "supported_by_anchor"},
                {"source": "artifact:doc.pdf:0:a2", "target": "anchor:doc.pdf:0:s1", "edge_type": "supported_by_anchor"},
            ],
            k_values=(2,),
            expansion_mode="source_anchor_neighborhood",
        )

        self.assertFalse(report["used_debug_edges"])
        self.assertFalse(report["used_semantic_edges"])
        self.assertEqual(per_query[0]["expanded_num_retrieved"], 2)
        self.assertEqual(report["edge_types_used"], ["supported_by_anchor"])

    def test_stage4_expansion_defaults_are_reasonable_with_no_added_artifacts(self) -> None:
        report, per_query = evaluate_stage4_graph_expansion(
            retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}],
            artifacts=[make_artifact("a1", page_index=0)],
            records=[{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[1]"}],
            formal_edges=[],
            k_values=(1,),
            expansion_mode="page_neighborhood",
        )

        self.assertEqual(per_query[0]["num_added_artifacts"], 0)
        self.assertEqual(report["avg_added_artifacts"], 0.0)
        self.assertEqual(report["added_ratio"], 0.0)
        self.assertEqual(report["added_gold_page_hit_rate"], 0.0)
        self.assertEqual(report["expansion_ratio"], 1.0)

    def test_edge_type_delta_metrics_are_reported(self) -> None:
        report, _ = evaluate_stage4_graph_expansion(
            retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}],
            artifacts=[make_artifact("a1", page_index=0), make_artifact("a2", page_index=1)],
            records=[{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[2]"}],
            formal_edges=[
                {"source": "artifact:doc.pdf:0:a1", "target": "artifact:doc.pdf:1:a2", "edge_type": "adjacent_page"},
            ],
            k_values=(1,),
            expansion_mode="direct_structural",
        )

        self.assertIn("adjacent_page", report["edge_type_delta_recall"])
        self.assertEqual(report["edge_type_delta_recall"]["adjacent_page"]["1"], 1.0)
        self.assertEqual(report["added_artifact_count_by_edge_type"]["adjacent_page"], 1)

    def test_allowed_and_blocked_edge_types_control_expansion(self) -> None:
        formal_edges = [
            {"source": "artifact:doc.pdf:0:a1", "target": "artifact:doc.pdf:1:a2", "edge_type": "adjacent_page"},
        ]
        blocked, _ = evaluate_stage4_graph_expansion(
            retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}],
            artifacts=[make_artifact("a1", page_index=0), make_artifact("a2", page_index=1)],
            records=[{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[2]"}],
            formal_edges=formal_edges,
            k_values=(1,),
            allowed_edge_types={"adjacent_page"},
            blocked_edge_types={"adjacent_page"},
            expansion_mode="direct_structural",
        )

        self.assertEqual(blocked["delta_recall_at_k"]["1"], 0.0)
        self.assertEqual(blocked["avg_added_artifacts"], 0.0)

    def test_stage4_expansion_rejects_semantic_edges(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_stage4_graph_expansion(
                retrieval_rows=[{"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"]}],
                artifacts=[make_artifact("a1")],
                records=[{"record_index": 0, "evidence_pages": "[1]"}],
                formal_edges=[{"source": "artifact:doc.pdf:0:a1", "target": "artifact:doc.pdf:0:a2", "edge_type": "supports"}],
            )

    def test_eval_manifests_are_marked_evaluation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = root / "artifacts.jsonl"
            retrieval = root / "retrieval.jsonl"
            records = root / "records.json"
            graph = root / "graph"
            graph.mkdir()
            artifacts.write_text(json.dumps(make_artifact("a1")) + "\n", encoding="utf-8")
            retrieval.write_text(json.dumps({"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}) + "\n", encoding="utf-8")
            records.write_text(json.dumps([{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[1]"}]), encoding="utf-8")
            (graph / "edges.jsonl").write_text("", encoding="utf-8")
            stage3_out = root / "eval" / "stage3"
            stage4_out = root / "eval" / "stage4"

            eval_stage3_main(["--retrieval", str(retrieval), "--artifacts", str(artifacts), "--records", str(records), "--output-dir", str(stage3_out)])
            eval_stage4_main(["--retrieval", str(retrieval), "--artifacts", str(artifacts), "--records", str(records), "--graph", str(graph), "--output-dir", str(stage4_out)])

            stage3_manifest = json.loads((stage3_out / "manifest.json").read_text(encoding="utf-8"))
            stage4_manifest = json.loads((stage4_out / "manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(stage3_manifest["evaluation_only"])
        self.assertTrue(stage3_manifest["not_consumed_by_stage2_stage3_stage4"])
        self.assertTrue(stage4_manifest["evaluation_only"])
        self.assertTrue(stage4_manifest["not_consumed_by_stage2_stage3_stage4"])

    def test_stage4_edge_ablation_writes_per_setting_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = root / "artifacts.jsonl"
            retrieval = root / "retrieval.jsonl"
            records = root / "records.json"
            graph = root / "graph"
            graph.mkdir()
            artifacts.write_text(json.dumps(make_artifact("a1")) + "\n", encoding="utf-8")
            retrieval.write_text(json.dumps({"record_index": 0, "doc_id": "doc.pdf", "retrieved_artifact_ids": ["a1"], "query_hash": "q"}) + "\n", encoding="utf-8")
            records.write_text(json.dumps([{"record_index": 0, "doc_id": "doc.pdf", "evidence_pages": "[1]"}]), encoding="utf-8")
            (graph / "edges.jsonl").write_text("", encoding="utf-8")
            stage4_out = root / "eval" / "ablation"

            eval_stage4_main(["--edge-ablation", "--retrieval", str(retrieval), "--artifacts", str(artifacts), "--records", str(records), "--graph", str(graph), "--output-dir", str(stage4_out)])
            report = json.loads((stage4_out / "report.json").read_text(encoding="utf-8"))
            manifest = json.loads((stage4_out / "manifest.json").read_text(encoding="utf-8"))
            per_setting = [json.loads(line) for line in (stage4_out / "per_setting.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertTrue(report["edge_ablation"])
        self.assertTrue(manifest["evaluation_only"])
        self.assertGreaterEqual(len(per_setting), 7)
        self.assertTrue(all(row["used_debug_edges"] is False for row in per_setting))


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
