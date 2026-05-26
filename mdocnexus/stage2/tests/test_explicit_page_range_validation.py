"""Tests for generic explicit page-reference range validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.normalize_record import normalize_record
from mdocnexus.stage2.page_range_validation import (
    OUT_OF_RANGE_ERROR,
    apply_explicit_page_range_validation_to_canonical_record,
    infer_document_page_count,
)
from scripts.stage2_prepare_single_page_trial import build_preflight_report


class ExplicitPageRangeValidationTest(unittest.TestCase):
    def test_page_reference_out_of_range(self) -> None:
        canonical_record = normalize_record(make_raw_sample(question="What is on page 30?"))["canonical_record"]
        validation = apply_explicit_page_range_validation_to_canonical_record(
            canonical_record,
            make_page_count_info(page_count=20),
        )

        self.assertEqual(validation["invalid_explicit_page_references"][0]["error_type"], OUT_OF_RANGE_ERROR)
        self.assertIn(29, canonical_record["candidate_pool"]["explicit_constraint_pages_raw"])
        self.assertIn(29, [ref["page_index_zero_based"] for ref in canonical_record["candidate_pool"]["explicit_constraint_pages_invalid"]])
        self.assertNotIn(29, canonical_record["candidate_pool"]["required_pages_for_compilation"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_path, config_path, extract_root = write_preflight_inputs(root, "What is on page 30?")
            create_extract_pages(extract_root, count=20)
            report = build_preflight_report(
                sample_path=sample_path,
                extract_root=extract_root,
                config_path=config_path,
                target_page_index=29,
                doc_id="example.pdf",
                question_substring="page 30",
            )

        self.assertIn(OUT_OF_RANGE_ERROR, report["blocking_reasons"])
        self.assertFalse(report["should_call_api"])
        self.assertFalse(report["should_generate_artifact"])
        self.assertNotIn(29, report["pages_to_compile"])

    def test_page_reference_valid(self) -> None:
        canonical_record = normalize_record(make_raw_sample(question="What is on page 30?"))["canonical_record"]
        apply_explicit_page_range_validation_to_canonical_record(
            canonical_record,
            make_page_count_info(page_count=40),
        )

        self.assertIn(29, canonical_record["candidate_pool"]["explicit_constraint_pages_valid"])
        self.assertIn(29, canonical_record["candidate_pool"]["required_pages_for_compilation"])
        self.assertIn(29, canonical_record["compilation_plan"]["pages_to_compile"])
        self.assertEqual(canonical_record["candidate_pool"]["explicit_constraint_pages_invalid"], [])

    def test_no_explicit_page_uses_retrieval_candidates_only(self) -> None:
        canonical_record = normalize_record(
            make_raw_sample(
                question="What does the chart show?",
                text_pages=[4, 8],
                image_pages=[],
            )
        )["canonical_record"]
        apply_explicit_page_range_validation_to_canonical_record(
            canonical_record,
            make_page_count_info(page_count=20),
        )

        self.assertEqual(canonical_record["candidate_pool"]["explicit_constraint_pages_raw"], [])
        self.assertEqual(canonical_record["candidate_pool"]["explicit_constraint_pages_invalid"], [])
        self.assertEqual(canonical_record["candidate_pool"]["required_pages_for_compilation"], [4, 8])

    def test_valid_page_with_missing_files_reports_missing_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_path, config_path, extract_root = write_preflight_inputs(root, "What is on page 30?")
            create_sparse_extract_pages(extract_root, [0, 39])
            report = build_preflight_report(
                sample_path=sample_path,
                extract_root=extract_root,
                config_path=config_path,
                target_page_index=29,
                doc_id="example.pdf",
                question_substring="page 30",
            )

        self.assertIn("missing_source_anchors", report["blocking_reasons"])
        self.assertNotIn(OUT_OF_RANGE_ERROR, report["blocking_reasons"])
        self.assertIn(29, report["pages_to_compile"])
        self.assertFalse(report["should_call_api"])

    def test_extract_files_can_infer_out_of_range_without_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir)
            create_extract_pages(extract_root, count=20)
            page_count_info = infer_document_page_count(
                doc_id="example.pdf",
                pdf_root=None,
                extract_root=extract_root,
            )
            canonical_record = normalize_record(make_raw_sample(question="What is on page 30?"))["canonical_record"]
            validation = apply_explicit_page_range_validation_to_canonical_record(
                canonical_record,
                page_count_info,
            )

        self.assertEqual(page_count_info["page_count"], 20)
        self.assertEqual(page_count_info["source"], "extract_files")
        self.assertEqual(validation["invalid_explicit_page_references"][0]["error_type"], OUT_OF_RANGE_ERROR)
        self.assertNotIn(29, canonical_record["compilation_plan"]["pages_to_compile"])

    def test_no_hard_coded_page_30(self) -> None:
        out_of_range_record = normalize_record(make_raw_sample(question="What is on page 100?"))["canonical_record"]
        apply_explicit_page_range_validation_to_canonical_record(
            out_of_range_record,
            make_page_count_info(page_count=20),
        )
        self.assertIn(99, [ref["page_index_zero_based"] for ref in out_of_range_record["candidate_pool"]["explicit_constraint_pages_invalid"]])
        self.assertNotIn(99, out_of_range_record["compilation_plan"]["pages_to_compile"])

        valid_record = normalize_record(make_raw_sample(question="What is on page 2?"))["canonical_record"]
        apply_explicit_page_range_validation_to_canonical_record(
            valid_record,
            make_page_count_info(page_count=20),
        )
        self.assertIn(1, valid_record["candidate_pool"]["explicit_constraint_pages_valid"])
        self.assertIn(1, valid_record["compilation_plan"]["pages_to_compile"])


def make_page_count_info(page_count: int) -> Dict[str, Any]:
    return {
        "page_count": page_count,
        "source": "test",
        "available_page_indices": list(range(page_count)),
        "page_index_contiguous": True,
    }


def make_raw_sample(
    question: str,
    text_pages: List[int] | None = None,
    image_pages: List[int] | None = None,
) -> Dict[str, Any]:
    text_pages = [] if text_pages is None else text_pages
    image_pages = [] if image_pages is None else image_pages
    return {
        "doc_id": "example.pdf",
        "doc_type": "test",
        "question": question,
        "answer_format": "Str",
        "answer": "Not answerable",
        "evidence_pages": "[]",
        "evidence_sources": "[]",
        "text-top-10-question": text_pages,
        "text-top-10-question_score": [1.0 for _ in text_pages],
        "image-top-10-question": image_pages,
        "image-top-10-question_score": [1.0 for _ in image_pages],
    }


def write_preflight_inputs(root: Path, question: str) -> tuple[Path, Path, Path]:
    sample_path = root / "sample-with-retrieval-results.json"
    config_path = root / "qwen3vl.yaml"
    extract_root = root / "tmp" / "MMLongBench"
    extract_root.mkdir(parents=True)
    sample_path.write_text(json.dumps([make_raw_sample(question=question)]), encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "model_id: Qwen/Qwen3-VL-8B-Instruct",
                "model: Qwen/Qwen3-VL-8B-Instruct",
                "base_url: https://api.siliconflow.cn/v1",
                "api_key: SECRET_VALUE_SHOULD_NOT_APPEAR",
                "api_key_env: SILICONFLOW_API_KEY",
            ]
        ),
        encoding="utf-8",
    )
    return sample_path, config_path, extract_root


def create_extract_pages(extract_root: Path, count: int) -> None:
    create_sparse_extract_pages(extract_root, list(range(count)))


def create_sparse_extract_pages(extract_root: Path, page_indices: List[int]) -> None:
    for page_index in page_indices:
        (extract_root / f"example_{page_index}.png").write_bytes(b"not-a-real-png")


if __name__ == "__main__":
    unittest.main()
