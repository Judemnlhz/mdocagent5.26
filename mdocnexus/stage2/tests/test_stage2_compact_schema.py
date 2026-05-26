"""Tests for compact Stage 2 index plus preflight sidecar storage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from mdocnexus.stage2.batch_page_selector import select_pages_for_small_batch
from mdocnexus.stage2.mdocagent_aligned_stage2 import select_trial_candidate_from_stage2_records
from mdocnexus.stage2.stage2_sidecar_store import (
    COMPACT_STAGE2_ALLOWED_FIELDS,
    load_stage2_preflight_sidecar,
)
from scripts.stage2_augment_retrieval_results import compact_stage2_records


class Stage2CompactSchemaTest(unittest.TestCase):
    def test_compact_index_and_sidecar_boundaries(self) -> None:
        original = make_original_record()
        full_stage2 = make_full_stage2()
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_dir = Path(tmpdir) / "preflight"
            compact_records = compact_stage2_records([{**original, "stage2": full_stage2}], sidecar_dir)
            compact_record = compact_records[0]
            sidecar = load_stage2_preflight_sidecar(compact_record["stage2"]["preflight_ref"])

        self.assertEqual(set(compact_record) - set(original), {"stage2"})
        for key, value in original.items():
            self.assertEqual(compact_record[key], value)

        stage2 = compact_record["stage2"]
        self.assertLessEqual(set(stage2), COMPACT_STAGE2_ALLOWED_FIELDS)
        self.assertNotIn("page_sources", stage2)
        self.assertNotIn("layout_blocks", stage2)
        self.assertNotIn("question_constraints", stage2)
        self.assertNotIn("retrieval_pages", stage2)
        self.assertNotIn("raw_output", stage2)
        self.assertNotIn("artifacts", stage2)
        self.assertIsInstance(stage2.get("preflight_ref"), str)

        self.assertIn("question_constraints", sidecar)
        self.assertIn("retrieval_pages", sidecar)
        self.assertIn("page_sources", sidecar)
        self.assertIn("layout_blocks_by_page", sidecar)
        serialized_sidecar = json.dumps(sidecar, ensure_ascii=False)
        for forbidden in ["answer", "evidence_pages", "evidence_sources", "binary_correctness", "api_key"]:
            self.assertNotIn(forbidden, serialized_sidecar)

    def test_selectors_load_compact_sidecar(self) -> None:
        original = make_original_record()
        full_stage2 = make_full_stage2()
        with tempfile.TemporaryDirectory() as tmpdir:
            compact_records = compact_stage2_records(
                [{**original, "stage2": full_stage2}],
                Path(tmpdir) / "preflight",
            )
            trial_report = select_trial_candidate_from_stage2_records(compact_records)
            small_batch_pages = select_pages_for_small_batch(compact_records, max_pages=3)

        self.assertTrue(trial_report["selection_passed"])
        self.assertEqual(trial_report["selected"]["page_index"], 1)
        self.assertEqual(len(small_batch_pages), 1)
        self.assertEqual(small_batch_pages[0]["page_index"], 1)


def make_original_record() -> Dict[str, Any]:
    return {
        "doc_id": "compact_doc.pdf",
        "doc_type": "report",
        "question": "What is shown on page 2?",
        "answer": "SECRET_GOLD",
        "evidence_pages": "[2]",
        "evidence_sources": "SECRET_SOURCE",
        "binary_correctness": True,
        "answer_format": "Str",
        "image-top-10-question": "[1]",
        "image-top-10-question_score": "[0.9]",
        "text-top-10-question": "[1]",
        "text-top-10-question_score": "[0.8]",
    }


def make_full_stage2() -> Dict[str, Any]:
    return {
        "version": "stage2_preflight_v1",
        "doc_name": "compact_doc",
        "page_count": {
            "value": 3,
            "source": "extract_files",
            "available_page_indices": [0, 1, 2],
            "page_index_contiguous": True,
        },
        "question_constraints": {
            "explicit_page_references": [
                {
                    "surface_text": "page 2",
                    "page_number_one_based": 2,
                    "page_index_zero_based": 1,
                    "source": "question_text",
                }
            ]
        },
        "retrieval_pages": {
            "image_top_10_question_unique": [{"page_index": 1, "rank": 1, "score": 0.9}],
            "text_top_10_question_unique": [{"page_index": 1, "rank": 1, "score": 0.8}],
            "retrieval_candidate_pages": [1],
        },
        "explicit_page_validation": {
            "valid_explicit_page_indices": [1],
            "invalid_explicit_page_references": [],
        },
        "pages_to_compile": [1],
        "page_sources": [
            {
                "page_index": 1,
                "page_text_path": "/tmp/compact_doc_1.txt",
                "page_image_path": "/tmp/compact_doc_1.png",
                "has_page_text": True,
                "has_page_image": True,
                "layout_block_ids": ["p001_text_0000", "p001_full_page_image"],
            }
        ],
        "preflight": {
            "passed": True,
            "blocking_reasons": [],
            "should_call_api": False,
            "should_generate_artifact": False,
        },
    }


if __name__ == "__main__":
    unittest.main()
