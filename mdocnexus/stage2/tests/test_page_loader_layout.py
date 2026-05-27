"""Tests for Step 2 page loading and layout block construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.page_input import load_page_content
from mdocnexus.stage2.page_input import prepare_pages_for_compilation


class PageLoaderLayoutTest(unittest.TestCase):
    def test_page_30_explicit_reference_generates_full_page_image_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir)
            images_dir = extract_path / "images"
            images_dir.mkdir()
            (images_dir / "example_029.png").write_bytes(b"not-a-real-png")

            canonical_record = make_canonical_record(
                pages_to_compile=[29],
                explicit_pages=[29],
                retrieval_pages=[],
            )

            prepared = prepare_pages_for_compilation(canonical_record, extract_path)
            page = get_page(prepared["pages"], 29)
            block_ids = {block["block_id"] for block in page["layout_blocks"]}

            self.assertEqual(prepared["errors"], [])
            self.assertIn("p029_full_page_image", block_ids)
            self.assertTrue(page["has_page_image"])
            self.assertFalse(page["has_page_text"])

    def test_text_only_page_generates_text_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir)
            texts_dir = extract_path / "texts"
            texts_dir.mkdir()
            (texts_dir / "example_3.txt").write_text("A text-only page.", encoding="utf-8")

            canonical_record = make_canonical_record(
                pages_to_compile=[3],
                explicit_pages=[],
                retrieval_pages=[3],
            )

            prepared = prepare_pages_for_compilation(canonical_record, extract_path)
            page = get_page(prepared["pages"], 3)
            block_ids = {block["block_id"] for block in page["layout_blocks"]}

            self.assertEqual(prepared["errors"], [])
            self.assertIn("p003_text_0000", block_ids)
            self.assertTrue(page["has_page_text"])
            self.assertFalse(page["has_page_image"])

    def test_text_and_image_page_generates_both_block_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir)
            texts_dir = extract_path / "texts"
            images_dir = extract_path / "images"
            texts_dir.mkdir()
            images_dir.mkdir()
            (texts_dir / "example_4.txt").write_text("A page with text and image.", encoding="utf-8")
            (images_dir / "example_4.png").write_bytes(b"not-a-real-png")

            canonical_record = make_canonical_record(
                pages_to_compile=[4],
                explicit_pages=[],
                retrieval_pages=[4],
            )

            prepared = prepare_pages_for_compilation(canonical_record, extract_path)
            page = get_page(prepared["pages"], 4)
            block_types = {block["block_type"] for block in page["layout_blocks"]}
            block_ids = {block["block_id"] for block in page["layout_blocks"]}

            self.assertEqual(prepared["errors"], [])
            self.assertEqual(block_types, {"text_block", "full_page_image"})
            self.assertIn("p004_text_0000", block_ids)
            self.assertIn("p004_full_page_image", block_ids)

    def test_missing_page_records_error_without_fabricating_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_record = make_canonical_record(
                pages_to_compile=[5],
                explicit_pages=[5],
                retrieval_pages=[],
            )

            prepared = prepare_pages_for_compilation(canonical_record, tmpdir)
            page = get_page(prepared["pages"], 5)

            self.assertEqual(page["layout_blocks"], [])
            self.assertEqual(len(prepared["errors"]), 1)
            self.assertEqual(prepared["errors"][0]["error_type"], "missing_source_anchors")
            self.assertEqual(prepared["errors"][0]["page_index"], 5)
            self.assertEqual(prepared["errors"][0]["required_by"], ["explicit_page_reference"])

    def test_loader_and_preparer_do_not_depend_on_eval_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir)
            texts_dir = extract_path / "texts"
            texts_dir.mkdir()
            (texts_dir / "example_6.txt").write_text("Visible page text.", encoding="utf-8")

            canonical_record = make_canonical_record(
                pages_to_compile=[6],
                explicit_pages=[],
                retrieval_pages=[6],
            )
            canonical_record["gold_annotation"] = {
                "answer": "must not be used",
                "evidence_pages_raw": [1],
                "eval_only": True,
            }
            canonical_record["baseline_outputs"] = {
                "mdocagent": {"answer": "must not be used"},
                "eval_only": True,
            }
            canonical_record["source_record"] = {
                "doc_id": "wrong.pdf",
                "question": "must not be used",
            }

            page_content = load_page_content(canonical_record, extract_path, 6)
            prepared = prepare_pages_for_compilation(canonical_record, extract_path)
            page = get_page(prepared["pages"], 6)

            self.assertEqual(page_content["doc_id"], "example.pdf")
            self.assertTrue(page["has_page_text"])
            self.assertIn("Visible page text.", page["page_text"])
            self.assertEqual(prepared["errors"], [])


def make_canonical_record(
    pages_to_compile: List[int],
    explicit_pages: List[int],
    retrieval_pages: List[int],
) -> Dict[str, Any]:
    return {
        "document": {"doc_id": "example.pdf", "doc_type": "test"},
        "question": {"text": "What is on page 30?", "answer_format": "short_text"},
        "question_constraints": {"explicit_page_references": []},
        "candidate_pool": {
            "explicit_constraint_pages": explicit_pages,
            "retrieval_candidate_pages": retrieval_pages,
        },
        "compilation_plan": {"pages_to_compile": pages_to_compile},
    }


def get_page(pages: List[Dict[str, Any]], page_index: int) -> Dict[str, Any]:
    for page in pages:
        if page["page_index"] == page_index:
            return page
    raise AssertionError(f"Page not found: {page_index}")


if __name__ == "__main__":
    unittest.main()
