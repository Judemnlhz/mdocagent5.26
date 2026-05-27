"""Tests for refined validation error attribution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdocnexus.stage2.tests.test_artifact_deduplicator import make_artifact
from mdocnexus.stage2.reports import summarize_refined_validation_failures


class RefinedErrorAttributionTest(unittest.TestCase):
    def test_duplicate_artifact_is_dedup_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw_outputs.jsonl"
            discard_path = root / "discard.jsonl"
            write_jsonl(
                raw_path,
                [
                    {
                        "doc_id": "doc.pdf",
                        "page_index": 1,
                        "raw_output": {
                            "doc_id": "doc.pdf",
                            "page_index": 1,
                            "artifacts": [make_artifact("a1"), make_artifact("a2")],
                        },
                    }
                ],
            )
            write_jsonl(
                discard_path,
                [
                    {
                        "doc_id": "doc.pdf",
                        "page_index": 1,
                        "artifact_id": "a2",
                        "error_type": "duplicate_artifact",
                    }
                ],
            )

            summary = summarize_refined_validation_failures(discard_path, raw_path)

        self.assertEqual(summary["issue_types"], {"duplicate_artifact": 1})
        self.assertTrue(summary["duplicate_artifact"]["deduplication_applicable"])
        self.assertEqual(summary["duplicate_artifact"]["same_type_anchor_content_duplicates"], 1)
        self.assertEqual(summary["non_dedup_blocking_issues"], [])

    def test_blocking_anchor_issue_prevents_dedup_only_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw_outputs.jsonl"
            discard_path = root / "discard.jsonl"
            write_jsonl(raw_path, [{"doc_id": "doc.pdf", "page_index": 1, "raw_output": {"artifacts": []}}])
            write_jsonl(
                discard_path,
                [
                    {
                        "doc_id": "doc.pdf",
                        "page_index": 1,
                        "artifact_id": "a1",
                        "error_type": "source_anchor_not_found",
                    }
                ],
            )

            summary = summarize_refined_validation_failures(discard_path, raw_path)

        self.assertFalse(summary["duplicate_artifact"]["deduplication_applicable"])
        self.assertEqual(summary["non_dedup_blocking_issues"], ["source_anchor_not_found"])


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
