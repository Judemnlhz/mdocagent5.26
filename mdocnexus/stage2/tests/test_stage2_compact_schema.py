"""Tests for compact Stage 2 page-route schema."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from mdocnexus.stage2.selectors import select_pages_for_small_batch
from mdocnexus.stage2.index_builder import (
    augment_retrieval_records,
    build_candidate_page_routes,
    select_trial_candidate_from_stage2_records,
)


REMOVED_STAGE2_FIELDS = {
    "version",
    "doc_name",
    "page_count",
    "question_constraints",
    "retrieval_pages",
    "explicit_page_validation",
    "pages_to_compile",
    "page_sources",
    "preflight_ref",
    "artifact_store_refs",
    "quality_summary_ref",
    "raw_output",
    "artifacts",
    "layout_blocks",
    "should_call_api",
    "should_generate_artifact",
}


class Stage2CompactSchemaTest(unittest.TestCase):
    def test_candidate_page_routes_merge_text_and_image_routes(self) -> None:
        routes = build_candidate_page_routes(
            {
                "text-top-10-question": "[2, 4, 2]",
                "image-top-10-question": "[3, 4]",
            }
        )

        self.assertEqual(
            routes,
            [
                {"page_index": 2, "routes": ["text"]},
                {"page_index": 3, "routes": ["image"]},
                {"page_index": 4, "routes": ["text", "image"]},
            ],
        )

    def test_compact_page_route_schema_preserves_original_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "extract"
            create_extract_pages(extract_root, "compact_doc", [1, 2, 3, 4])
            original = make_original_record()
            compact_record = augment_retrieval_records([original], extract_root=extract_root)[0]

        self.assertEqual(set(compact_record) - set(original), {"stage2"})
        for key, value in original.items():
            self.assertEqual(compact_record[key], value)

        stage2 = compact_record["stage2"]
        self.assertEqual(set(stage2), {"preflight", "candidate_page_routes"})
        self.assertEqual(set(stage2["preflight"]), {"passed", "blocking_reasons"})
        self.assertTrue(stage2["preflight"]["passed"])
        self.assertEqual(stage2["preflight"]["blocking_reasons"], [])
        self.assertEqual(
            stage2["candidate_page_routes"],
            [
                {"page_index": 1, "routes": ["text"]},
                {"page_index": 2, "routes": ["text", "image"]},
                {"page_index": 3, "routes": ["image"]},
            ],
        )
        serialized_stage2 = json.dumps(stage2, ensure_ascii=False)
        for removed in REMOVED_STAGE2_FIELDS:
            self.assertNotIn(removed, serialized_stage2)
        for forbidden in ["answer", "evidence_pages", "evidence_sources", "binary_correctness", "api_key"]:
            self.assertNotIn(forbidden, serialized_stage2)

    def test_selectors_rebuild_page_sources_from_extract_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "extract"
            create_extract_pages(extract_root, "compact_doc", [1, 2, 3])
            compact_records = augment_retrieval_records([make_original_record()], extract_root=extract_root)
            trial_report = select_trial_candidate_from_stage2_records(compact_records, extract_root=extract_root)
            small_batch_pages = select_pages_for_small_batch(
                compact_records,
                max_pages=3,
                extract_root=str(extract_root),
            )

        self.assertTrue(trial_report["selection_passed"])
        self.assertEqual(trial_report["selected"]["page_index"], 2)
        self.assertEqual(len(small_batch_pages), 1)
        self.assertEqual(small_batch_pages[0]["page_index"], 2)
        self.assertIn("p002_full_page_image", small_batch_pages[0]["layout_block_ids"])


def make_original_record() -> Dict[str, Any]:
    return {
        "doc_id": "compact_doc.pdf",
        "doc_type": "report",
        "question": "What is shown on page 3?",
        "answer": "SECRET_GOLD",
        "evidence_pages": "[3]",
        "evidence_sources": "SECRET_SOURCE",
        "binary_correctness": True,
        "answer_format": "Str",
        "image-top-10-question": "[2, 3]",
        "image-top-10-question_score": "[0.9, 0.8]",
        "text-top-10-question": "[1, 2]",
        "text-top-10-question_score": "[0.7, 0.6]",
    }


def create_extract_pages(extract_root: Path, doc_stem: str, page_indices: list[int]) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    for page_index in page_indices:
        (extract_root / f"{doc_stem}_{page_index}.png").write_bytes(b"not-a-real-png")
        (extract_root / f"{doc_stem}_{page_index}.txt").write_text(
            f"text for page {page_index}",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
