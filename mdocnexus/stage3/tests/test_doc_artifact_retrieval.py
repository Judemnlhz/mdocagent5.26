"""Tests for Stage 3 document-generic artifact retrieval dry-run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdocnexus.stage3.doc_artifact_retrieval import (
    canonical_json_hash,
    load_artifacts_jsonl,
    run_doc_artifact_retrieval,
)


class DocArtifactRetrievalTest(unittest.TestCase):
    def test_retrieval_ignores_gold_fields_and_outputs_no_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts_path, query_path = write_fixture_inputs(root)
            output_dir = root / "out"

            result = run_doc_artifact_retrieval(artifacts_path, query_path, output_dir, top_k=2)
            retrieval_rows = read_jsonl(output_dir / "retrieval.jsonl")
            public_text = public_output_text(output_dir)
            public_values = read_public_json_values(output_dir)

        self.assertEqual(result["quality_report"]["num_gold_field_violations"], 0)
        self.assertEqual(result["quality_report"]["num_outputs_with_answer_field"], 0)
        self.assertTrue(all(row["no_gold_fields_used"] for row in retrieval_rows))
        for value in public_values:
            walk_public_value(self, value)
        for forbidden_value in ("SECRET_ANSWER", "SECRET_GOLD"):
            self.assertNotIn(forbidden_value, public_text)

    def test_retrieval_does_not_modify_stage2_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts_path, query_path = write_fixture_inputs(root)
            before = artifacts_path.read_bytes()

            run_doc_artifact_retrieval(artifacts_path, query_path, root / "out", top_k=3)

            after = artifacts_path.read_bytes()
        self.assertEqual(before, after)

    def test_same_inputs_produce_same_retrieval_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts_path, query_path = write_fixture_inputs(root)

            first = run_doc_artifact_retrieval(artifacts_path, query_path, root / "out1", top_k=2)
            second = run_doc_artifact_retrieval(artifacts_path, query_path, root / "out2", top_k=2)

        self.assertEqual(first["manifest"]["retrieval_hash"], second["manifest"]["retrieval_hash"])
        self.assertEqual(first["retrieval_hash"], second["retrieval_hash"])

    def test_tied_scores_sort_by_artifact_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = [
                make_artifact("z_artifact", content="same token"),
                make_artifact("a_artifact", content="same token"),
                make_artifact("m_artifact", content="same token"),
            ]
            artifacts_path = root / "artifacts.jsonl"
            write_jsonl(artifacts_path, artifacts)
            query_path = root / "queries.jsonl"
            write_jsonl(query_path, [{"record_id": "r1", "doc_id": "doc.pdf", "question": "same"}])

            run_doc_artifact_retrieval(artifacts_path, query_path, root / "out", top_k=3)
            row = read_jsonl(root / "out" / "retrieval.jsonl")[0]

        self.assertEqual(row["retrieved_artifact_ids"], ["a_artifact", "m_artifact", "z_artifact"])

    def test_does_not_read_debug_edges_and_marks_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts_path, query_path = write_fixture_inputs(root)
            debug_edges = root / "debug_edges.jsonl"
            debug_edges.write_text(json.dumps({"edge_type": "same_record_debug", "raw_response": "SHOULD_NOT_READ"}) + "\n", encoding="utf-8")

            run_doc_artifact_retrieval(artifacts_path, query_path, root / "out", top_k=2)
            rows = read_jsonl(root / "out" / "retrieval.jsonl")
            quality = json.loads((root / "out" / "quality_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))
            public_text = public_output_text(root / "out")

        self.assertTrue(all(row["used_debug_edges"] is False for row in rows))
        self.assertFalse(quality["used_debug_edges"])
        self.assertFalse(manifest["used_debug_edges"])
        self.assertNotIn("same_record_debug", public_text)
        self.assertNotIn("SHOULD_NOT_READ", public_text)

    def test_no_model_or_api_provider_dependency(self) -> None:
        source = Path("mdocnexus/stage3/doc_artifact_retrieval.py").read_text(encoding="utf-8")
        self.assertNotIn("RealApi", source)
        self.assertNotIn("ArtifactCompilerClient", source)
        self.assertNotIn("openai", source.lower())

    def test_query_hash_uses_sanitized_query_identity(self) -> None:
        clean = {"record_index": 0, "doc_id": "doc.pdf", "question": "What title?"}
        dirty = dict(clean)
        dirty.update({"answer": "SECRET", "evidence_pages": [1], "gold_answer": "SECRET"})
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts_path = root / "artifacts.jsonl"
            write_jsonl(artifacts_path, [make_artifact("a1", content="title text")])
            clean_query = root / "clean.jsonl"
            dirty_query = root / "dirty.jsonl"
            write_jsonl(clean_query, [clean])
            write_jsonl(dirty_query, [dirty])

            first = run_doc_artifact_retrieval(artifacts_path, clean_query, root / "clean_out", top_k=1)
            second = run_doc_artifact_retrieval(artifacts_path, dirty_query, root / "dirty_out", top_k=1)
            artifact_hash_len = len(canonical_json_hash(load_artifacts_jsonl(artifacts_path)))

        self.assertEqual(first["retrieval_hash"], second["retrieval_hash"])
        self.assertEqual(artifact_hash_len, 64)


def write_fixture_inputs(root: Path) -> tuple[Path, Path]:
    artifacts_path = root / "artifacts.jsonl"
    query_path = root / "queries.jsonl"
    write_jsonl(
        artifacts_path,
        [
            make_artifact("a1", content="revenue increased in the chart"),
            make_artifact("a2", content="caption describes a map"),
            make_artifact("b1", doc_id="other.pdf", content="unrelated document"),
        ],
    )
    write_jsonl(
        query_path,
        [
            {
                "record_id": "r1",
                "doc_id": "doc.pdf",
                "question": "What does the chart show about revenue?",
                "answer": "SECRET_ANSWER",
                "gold_answer": "SECRET_GOLD",
                "evidence_pages": [1],
                "evidence_sources": ["x"],
                "binary_correctness": True,
                "gold_evidence": "SECRET",
                "gold_page": 1,
                "gold_pages": [1],
            }
        ],
    )
    return artifacts_path, query_path


def make_artifact(artifact_id: str, doc_id: str = "doc.pdf", content: str = "content") -> dict:
    return {
        "record_index": 0,
        "doc_id": doc_id,
        "page_index": 0,
        "artifact_id": artifact_id,
        "artifact_type": "text_span",
        "modality": "text",
        "content": content,
        "normalized_content": {"text": content},
        "source_anchors": [{"source_id": "p000_text_0000", "page_index": 0, "bbox": None}],
        "provenance": {"op": "ATOM", "sources": ["p000_text_0000"]},
        "validation_status": "candidate",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def public_output_text(output_dir: Path) -> str:
    return "\n".join(
        (output_dir / name).read_text(encoding="utf-8")
        for name in ("retrieval.jsonl", "quality_report.json", "manifest.json")
    )


SAFE_PUBLIC_FIELD_NAMES = {"no_answer_generation", "no_gold_fields_used"}
FORBIDDEN_PUBLIC_FIELD_NAMES = {
    "answer",
    "answers",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}


def read_public_json_values(output_dir: Path) -> list[object]:
    values: list[object] = []
    for name in ("retrieval.jsonl", "quality_report.json", "manifest.json"):
        path = output_dir / name
        if path.suffix == ".jsonl":
            values.extend(read_jsonl(path))
        else:
            values.append(json.loads(path.read_text(encoding="utf-8")))
    return values


def walk_public_value(testcase: unittest.TestCase, value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in SAFE_PUBLIC_FIELD_NAMES:
                testcase.assertNotIn(key, FORBIDDEN_PUBLIC_FIELD_NAMES)
            walk_public_value(testcase, child)
    elif isinstance(value, list):
        for child in value:
            walk_public_value(testcase, child)


if __name__ == "__main__":
    unittest.main()
