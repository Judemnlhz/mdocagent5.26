"""Tests for deterministic artifact deduplication."""

from __future__ import annotations

import unittest

from mdocnexus.stage2.artifact_deduplicator import (
    build_artifact_dedup_key,
    deduplicate_page_artifacts,
    normalize_for_hash,
)


class ArtifactDeduplicatorTest(unittest.TestCase):
    def test_normalize_for_hash_is_stable(self) -> None:
        self.assertEqual(normalize_for_hash({"b": 2, "a": 1}), '{"a":1,"b":2}')
        self.assertEqual(normalize_for_hash("  Some   Text  "), "some text")

    def test_dedup_key_ignores_source_anchor_order(self) -> None:
        first = make_artifact("a1", source_ids=["p001_text_1", "p001_text_0"])
        second = make_artifact("a2", source_ids=["p001_text_0", "p001_text_1"])

        self.assertEqual(build_artifact_dedup_key(first), build_artifact_dedup_key(second))

    def test_deduplicate_keeps_first_and_records_removed_duplicate(self) -> None:
        original = {
            "doc_id": "doc.pdf",
            "page_index": 1,
            "artifacts": [
                make_artifact("a1"),
                make_artifact("a2"),
                make_artifact("a3", content="different"),
            ],
        }

        deduped, removed = deduplicate_page_artifacts(original)

        self.assertEqual([artifact["artifact_id"] for artifact in deduped["artifacts"]], ["a1", "a3"])
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["error_type"], "duplicate_artifact_deduplicated")
        self.assertEqual(removed[0]["artifact_id"], "a2")
        self.assertEqual(removed[0]["duplicate_of"], "a1")
        self.assertEqual(original["artifacts"][1]["content"], "same content")

    def test_different_source_anchor_is_not_deduplicated(self) -> None:
        original = {
            "doc_id": "doc.pdf",
            "page_index": 1,
            "artifacts": [
                make_artifact("a1", source_ids=["p001_text_0000"]),
                make_artifact("a2", source_ids=["p001_text_0001"]),
            ],
        }

        deduped, removed = deduplicate_page_artifacts(original)

        self.assertEqual(len(deduped["artifacts"]), 2)
        self.assertEqual(removed, [])

    def test_different_artifact_type_is_not_deduplicated(self) -> None:
        first = make_artifact("a1")
        second = make_artifact("a2")
        second["artifact_type"] = "numeric_fact"
        original = {"doc_id": "doc.pdf", "page_index": 1, "artifacts": [first, second]}

        deduped, removed = deduplicate_page_artifacts(original)

        self.assertEqual(len(deduped["artifacts"]), 2)
        self.assertEqual(removed, [])


def make_artifact(
    artifact_id: str,
    content: str = "same content",
    source_ids: list[str] | None = None,
) -> dict:
    source_ids = ["p001_text_0000"] if source_ids is None else source_ids
    return {
        "artifact_id": artifact_id,
        "doc_id": "doc.pdf",
        "page_index": 1,
        "artifact_type": "text_span",
        "modality": "text",
        "content": content,
        "normalized_content": {"text": content},
        "source_anchors": [
            {"source_id": source_id, "anchor_type": "text_block", "page_index": 1, "bbox": None}
            for source_id in source_ids
        ],
        "provenance": {"op": "ATOM", "sources": source_ids},
        "validation_status": "candidate",
        "compiler_metadata": {},
    }


if __name__ == "__main__":
    unittest.main()
