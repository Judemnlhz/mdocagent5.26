from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mdocnexus.integration.mdocagent_adapter import (
    build_mdocagent_adapter_manifest,
    canonical_json_hash,
    load_artifacts_by_page,
    load_mdocagent_retrieval_records,
    rerank_pages_with_artifacts,
    select_pages_with_graph,
    write_mdocagent_compatible_records,
    write_manifest,
)


FORBIDDEN_KEYS = {
    "answer",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
}


class MDocAgentAdapterTests(unittest.TestCase):
    def test_sanitizes_gold_fields_and_preserves_base_dataset_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            retrieval = root / "sample-with-retrieval-results.json"
            artifacts = root / "artifacts.jsonl"
            retrieval.write_text(json.dumps([sample_record()]), encoding="utf-8")
            artifacts.write_text(artifact_lines(), encoding="utf-8")

            records = load_mdocagent_retrieval_records(retrieval)
            artifacts_by_page = load_artifacts_by_page(artifacts)
            adapted = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=4, mode="original_plus_artifact")

            self.assertEqual(adapted[0]["doc_id"], "doc.pdf")
            self.assertEqual(adapted[0]["question"], sample_record()["question"])
            self.assertIn("text-top-10-question", adapted[0])
            self.assertIn("image-top-10-question", adapted[0])
            self.assert_no_forbidden_keys(adapted)

    def test_original_only_keeps_original_retrieval_order(self) -> None:
        records = [load_mdocagent_retrieval_records_from_value(sample_record())[0]]
        adapted = rerank_pages_with_artifacts(records, {}, top_k=4, mode="original_only")
        self.assertEqual(adapted[0]["text-top-10-question"], [2, 1, 3, 0])
        self.assertEqual(adapted[0]["image-top-10-question"], [3, 2, 1, 0])

    def test_artifact_only_and_original_plus_artifact_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            retrieval = root / "retrieval.json"
            artifacts = root / "artifacts.jsonl"
            retrieval.write_text(json.dumps([sample_record()]), encoding="utf-8")
            artifacts.write_text(artifact_lines(), encoding="utf-8")
            records = load_mdocagent_retrieval_records(retrieval)
            artifacts_by_page = load_artifacts_by_page(artifacts)

            artifact_a = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=4, mode="artifact_only")
            artifact_b = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=4, mode="artifact_only")
            hybrid_a = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=4, mode="original_plus_artifact")
            hybrid_b = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=4, mode="original_plus_artifact")

            self.assertEqual(artifact_a, artifact_b)
            self.assertEqual(hybrid_a, hybrid_b)
            self.assertEqual(canonical_json_hash(artifact_a), canonical_json_hash(artifact_b))
            self.assertEqual(canonical_json_hash(hybrid_a), canonical_json_hash(hybrid_b))

    def test_artifact_rerank_falls_back_without_anchored_positive_signal(self) -> None:
        records = load_mdocagent_retrieval_records_from_value(sample_record())
        artifacts = load_artifacts_by_page_from_lines(unanchored_artifact_lines())

        adapted = rerank_pages_with_artifacts(records, artifacts, top_k=4, mode="original_plus_artifact")

        self.assertEqual(adapted[0]["text-top-10-question"], [2, 1, 3, 0])
        self.assertEqual(adapted[0]["image-top-10-question"], [3, 2, 1, 0])
        self.assertFalse(adapted[0]["_nexus_meta"]["anchored_artifact_rerank_applied"])
        self.assertEqual(adapted[0]["_nexus_meta"]["fallback_reason"], "no_positive_anchored_artifact_score")

    def test_artifact_rerank_preserves_text_and_image_branches(self) -> None:
        records = load_mdocagent_retrieval_records_from_value(branch_diversity_record())
        artifacts = load_artifacts_by_page_from_lines(anchored_branch_artifact_lines())

        adapted = rerank_pages_with_artifacts(records, artifacts, top_k=4, mode="original_plus_artifact")

        self.assertEqual(adapted[0]["text-top-10-question"][0], 1)
        self.assertEqual(adapted[0]["image-top-10-question"], [3, 2, 0, 4])
        self.assertNotEqual(adapted[0]["text-top-10-question"], adapted[0]["image-top-10-question"])
        self.assertTrue(adapted[0]["_nexus_meta"]["anchored_artifact_rerank_applied"])
        self.assertTrue(adapted[0]["_nexus_meta"]["preserved_retrieval_branches"])

    def test_lambda_weight_manifest_and_top4_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            retrieval = root / "retrieval.json"
            artifacts = root / "artifacts.jsonl"
            output = root / "out.json"
            manifest_path = root / "manifest.json"
            retrieval.write_text(json.dumps([sample_record()]), encoding="utf-8")
            artifacts.write_text(artifact_lines(), encoding="utf-8")
            records = load_mdocagent_retrieval_records(retrieval)
            artifacts_by_page = load_artifacts_by_page(artifacts)
            adapted = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=4, mode="original_plus_artifact", lambda_weight=0.25)
            write_mdocagent_compatible_records(adapted, output)
            manifest = build_mdocagent_adapter_manifest(
                mode="original_plus_artifact",
                top_k=4,
                lambda_weight=0.25,
                input_retrieval=retrieval,
                artifacts=artifacts,
                output_retrieval=output,
                command_args={"lambda_weight": 0.25},
                repo_root=Path.cwd(),
            )
            write_manifest(manifest, manifest_path)
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(loaded_manifest["lambda_weight"], 0.25)
            self.assertTrue(loaded_manifest["same_page_budget_as_baseline"])
            self.assertLessEqual(len(adapted[0]["text-top-10-question"]), 4)
            self.assertLessEqual(len(adapted[0]["image-top-10-question"]), 4)

    def test_graph_context_ignores_debug_and_semantic_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_dir = root / "graph"
            graph_dir.mkdir()
            (graph_dir / "edges.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "edge_type": "adjacent_page",
                                "source": "page:doc.pdf:2",
                                "target": "page:doc.pdf:4",
                                "provenance": {"doc_id": "doc.pdf", "source_page_index": 2, "target_page_index": 4},
                            }
                        ),
                        json.dumps(
                            {
                                "edge_type": "same_record_debug",
                                "source": "page:doc.pdf:2",
                                "target": "page:doc.pdf:99",
                                "provenance": {"doc_id": "doc.pdf", "source_page_index": 2, "target_page_index": 99},
                            }
                        ),
                        json.dumps(
                            {
                                "edge_type": "supports",
                                "source": "page:doc.pdf:2",
                                "target": "page:doc.pdf:88",
                                "provenance": {"doc_id": "doc.pdf", "source_page_index": 2, "target_page_index": 88},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (graph_dir / "debug_edges.jsonl").write_text(
                json.dumps({"edge_type": "adjacent_page", "source": "page:doc.pdf:2", "target": "page:doc.pdf:77"}) + "\n",
                encoding="utf-8",
            )
            artifacts = load_artifacts_by_page_from_lines(artifact_lines())
            records = load_mdocagent_retrieval_records_from_value(sample_record())

            adapted = select_pages_with_graph(records, artifacts, graph_dir, top_k=4, expansion_mode="page_neighborhood")
            selected = set(adapted[0]["text-top-10-question"])
            self.assertNotIn(99, selected)
            self.assertNotIn(88, selected)
            self.assertNotIn(77, selected)
            self.assertFalse(adapted[0]["_nexus_meta"]["used_debug_edges"])
            self.assertFalse(adapted[0]["_nexus_meta"]["used_semantic_edges"])

    def test_write_records_can_be_reloaded_and_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "sample-with-retrieval-results-nexus.json"
            records = load_mdocagent_retrieval_records_from_value(sample_record())
            adapted = rerank_pages_with_artifacts(records, {}, top_k=4, mode="original_only")
            hash_a = write_mdocagent_compatible_records(adapted, output)
            reloaded = load_mdocagent_retrieval_records(output)
            hash_b = write_mdocagent_compatible_records(adapted, output)

            self.assertEqual(hash_a, hash_b)
            self.assertEqual(reloaded[0]["doc_id"], "doc.pdf")
            self.assertIn("text-top-10-question", reloaded[0])

    def test_deepseek_not_in_adapter_or_selector_manifest_and_no_absolute_path_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            retrieval = root / "retrieval.json"
            artifacts = root / "artifacts.jsonl"
            output = root / "out.json"
            retrieval.write_text(json.dumps([sample_record()]), encoding="utf-8")
            artifacts.write_text(artifact_lines(), encoding="utf-8")
            records = load_mdocagent_retrieval_records(retrieval)
            artifacts_by_page = load_artifacts_by_page(artifacts)
            adapted = rerank_pages_with_artifacts(records, artifacts_by_page, top_k=4, mode="artifact_only")
            write_mdocagent_compatible_records(adapted, output)
            manifest = build_mdocagent_adapter_manifest(
                mode="artifact_only",
                top_k=4,
                lambda_weight=0.5,
                input_retrieval=retrieval,
                artifacts=artifacts,
                output_retrieval=output,
                command_args={"input_retrieval": str(retrieval), "local_path": "/home/lhz/secret"},
                repo_root=Path.cwd(),
            )
            serialized = json.dumps({"records": adapted, "manifest": manifest}, sort_keys=True)
            self.assertNotIn("DeepSeek-V3", serialized)
            self.assertNotIn("/home/", serialized)
            self.assertNotIn("local_path", serialized)

    def assert_no_forbidden_keys(self, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                self.assertNotIn(key, FORBIDDEN_KEYS)
                self.assertFalse(str(key).startswith("gold_"))
                self.assert_no_forbidden_keys(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_no_forbidden_keys(child)


def sample_record() -> dict[str, object]:
    return {
        "record_index": 7,
        "doc_id": "doc.pdf",
        "question": "Which page discusses alpha revenue?",
        "answer": "page 4",
        "gold_answer": "page 4",
        "evidence_pages": [4],
        "evidence_sources": ["Table"],
        "binary_correctness": 1,
        "text-index-path-question": "/home/lhz/private/index",
        "text-top-10-question": [2, 1, 3, 0],
        "text-top-10-question_score": [0.2, 0.9, 0.1, 0.05],
        "image-top-10-question": [3, 2, 1, 0],
        "image-top-10-question_score": [0.8, 0.7, 0.6, 0.5],
    }


def artifact_lines() -> str:
    rows = [
        anchored_artifact(1, "a1", "text_span", "alpha overview"),
        anchored_artifact(2, "a2", "text_span", "beta notes"),
        anchored_artifact(3, "a3", "caption", "alpha chart"),
        anchored_artifact(4, "a4", "table", "alpha revenue revenue"),
    ]
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"


def unanchored_artifact_lines() -> str:
    rows = [
        {"doc_id": "doc.pdf", "page_index": 1, "artifact_id": "a1", "artifact_type": "text_span", "content": "alpha revenue revenue"},
        {"doc_id": "doc.pdf", "page_index": 3, "artifact_id": "a3", "artifact_type": "caption", "content": "alpha chart"},
    ]
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"


def branch_diversity_record() -> dict[str, object]:
    return {
        "doc_id": "doc.pdf",
        "question": "Which page discusses alpha revenue?",
        "text-top-10-question": [2, 1, 0, 4],
        "text-top-10-question_score": [1.0, 0.95, 0.5, 0.1],
        "image-top-10-question": [3, 2, 0, 4],
        "image-top-10-question_score": [1.0, 0.8, 0.5, 0.1],
    }


def anchored_branch_artifact_lines() -> str:
    rows = [
        anchored_artifact(1, "a1", "table", "alpha revenue revenue revenue"),
        anchored_artifact(9, "a9", "table", "alpha revenue revenue revenue"),
    ]
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"


def anchored_artifact(page_index: int, artifact_id: str, artifact_type: str, content: str) -> dict[str, object]:
    return {
        "doc_id": "doc.pdf",
        "page_index": page_index,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "content": content,
        "source_anchors": [
            {
                "anchor_type": "text_span",
                "source_id": f"p{page_index:03d}_text",
                "page_index": page_index,
            }
        ],
    }


def load_mdocagent_retrieval_records_from_value(record: dict[str, object]) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "retrieval.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        return load_mdocagent_retrieval_records(path)


def load_artifacts_by_page_from_lines(lines: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "artifacts.jsonl"
        path.write_text(lines, encoding="utf-8")
        return load_artifacts_by_page(path)


if __name__ == "__main__":
    unittest.main()
