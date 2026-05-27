"""Tests for objective Stage 2 single-page trial candidate selection."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from mdocnexus.stage2.selectors import select_single_page_trial_candidate
from scripts.stage2 import (
    apply_candidate_report_to_args,
    load_canonical_record_from_args,
)


class TrialCandidateSelectorTest(unittest.TestCase):
    def test_valid_explicit_page_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            records = [
                make_raw_sample("doc_a.pdf", "What is shown on page 30?", image_pages=[0]),
                make_raw_sample("doc_b.pdf", "What is shown on page 2?", image_pages=[0]),
            ]
            sample_path = write_sample(root, records)
            create_extract_pages(extract_root, "doc_a", [0])
            create_extract_pages(extract_root, "doc_b", [0, 1])

            report = select_single_page_trial_candidate(sample_path, "MMLongBench", extract_root)

        self.assertTrue(report["selection_passed"])
        self.assertEqual(report["selected"]["doc_id"], "doc_b.pdf")
        self.assertEqual(report["selected"]["page_index"], 1)
        self.assertEqual(report["selected"]["selection_reason"], "valid_explicit_page_with_image")

    def test_out_of_range_explicit_page_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            sample_path = write_sample(
                root,
                [make_raw_sample("doc_a.pdf", "What is shown on page 30?", image_pages=[])],
            )
            create_extract_pages(extract_root, "doc_a", list(range(20)))

            report = select_single_page_trial_candidate(sample_path, "MMLongBench", extract_root)

        self.assertFalse(report["selection_passed"])
        self.assertEqual(report["blocking_reasons"], ["no_valid_single_page_trial_candidate"])
        self.assertIsNone(report["selected"])
        self.assertEqual(report["num_out_of_range_explicit_pages"], 1)

    def test_no_explicit_page_selects_image_top1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            sample_path = write_sample(
                root,
                [make_raw_sample("doc_a.pdf", "What does the figure show?", image_pages=[5, 2])],
            )
            create_extract_pages(extract_root, "doc_a", [5])

            report = select_single_page_trial_candidate(sample_path, "MMLongBench", extract_root)

        self.assertTrue(report["selection_passed"])
        self.assertEqual(report["selected"]["page_index"], 5)
        self.assertEqual(report["selected"]["selection_reason"], "image_top1_with_image")

    def test_missing_image_top1_selects_retrieval_union_first_available_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            sample_path = write_sample(
                root,
                [
                    make_raw_sample(
                        "doc_a.pdf",
                        "What does the figure show?",
                        text_pages=[4],
                        image_pages=[2, 4],
                    )
                ],
            )
            create_extract_pages(extract_root, "doc_a", [4])

            report = select_single_page_trial_candidate(sample_path, "MMLongBench", extract_root)

        self.assertTrue(report["selection_passed"])
        self.assertEqual(report["selected"]["page_index"], 4)
        self.assertEqual(report["selected"]["selection_reason"], "retrieval_union_first_image")

    def test_candidate_report_does_not_write_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            sample_path = write_sample(
                root,
                [make_raw_sample("doc_a.pdf", "What is shown on page 1?", image_pages=[])],
            )
            create_extract_pages(extract_root, "doc_a", [0])

            report = select_single_page_trial_candidate(sample_path, "MMLongBench", extract_root)

        serialized = json.dumps(report, ensure_ascii=False)
        forbidden_keys = collect_keys(report)
        self.assertNotIn("answer", forbidden_keys)
        self.assertNotIn("evidence_pages", forbidden_keys)
        self.assertNotIn("binary_correctness", forbidden_keys)
        self.assertNotIn("GOLD_SECRET", serialized)
        self.assertNotIn("BASELINE_SECRET", serialized)

    def test_candidate_report_drives_single_page_trial_wrapper_without_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            sample_path = write_sample(
                root,
                [make_raw_sample("doc_a.pdf", "What is shown on page 2?", image_pages=[])],
            )
            create_extract_pages(extract_root, "doc_a", [0, 1])
            report = select_single_page_trial_candidate(sample_path, "MMLongBench", extract_root)
            candidate_report = root / "candidate.json"
            candidate_report.write_text(json.dumps(report), encoding="utf-8")
            args = make_wrapper_args(sample_path, candidate_report)

            populated_args = apply_candidate_report_to_args(args)
            canonical_record = load_canonical_record_from_args(populated_args)

        self.assertEqual(populated_args.record_id, report["selected"]["record_id"])
        self.assertEqual(populated_args.doc_id, "doc_a.pdf")
        self.assertEqual(populated_args.target_page_index, 1)
        self.assertEqual(populated_args.extract_root, str(extract_root))
        self.assertIn(1, canonical_record["compilation_plan"]["pages_to_compile"])


def make_raw_sample(
    doc_id: str,
    question: str,
    text_pages: List[int] | None = None,
    image_pages: List[int] | None = None,
) -> Dict[str, Any]:
    text_pages = [] if text_pages is None else text_pages
    image_pages = [] if image_pages is None else image_pages
    return {
        "doc_id": doc_id,
        "doc_type": "test",
        "question": question,
        "answer_format": "Str",
        "answer": "GOLD_SECRET",
        "evidence_pages": "[99]",
        "evidence_sources": "['secret']",
        "text-top-10-question": text_pages,
        "text-top-10-question_score": [1.0 for _ in text_pages],
        "image-top-10-question": image_pages,
        "image-top-10-question_score": [1.0 for _ in image_pages],
        "ans_mmlb-MDocAgent": "BASELINE_SECRET",
        "binary_correctness": True,
    }


def write_sample(root: Path, records: List[Dict[str, Any]]) -> Path:
    sample_path = root / "sample-with-retrieval-results.json"
    sample_path.write_text(json.dumps(records), encoding="utf-8")
    return sample_path


def create_extract_pages(extract_root: Path, doc_stem: str, page_indices: List[int]) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    for page_index in page_indices:
        (extract_root / f"{doc_stem}_{page_index}.png").write_bytes(b"not-a-real-png")
        (extract_root / f"{doc_stem}_{page_index}.txt").write_text(
            f"page {page_index} text",
            encoding="utf-8",
        )


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value.keys())
        for child in value.values():
            keys.update(collect_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(collect_keys(child))
        return keys
    return set()


def make_wrapper_args(sample_path: Path, candidate_report: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(sample_path),
        normalized_record_path=None,
        sample_path=str(sample_path),
        record_id=None,
        doc_id=None,
        question_substring=None,
        extract_path=None,
        extract_root=None,
        pdf_root=None,
        output_path=str(sample_path.parent / "artifact_store.json"),
        target_page_index=None,
        candidate_report=str(candidate_report),
        provider=None,
        model_name=None,
        enable_real_api=False,
        run_real_trial=False,
        preflight_only=False,
        api_base_url=None,
        api_key_env_var=None,
        temperature=None,
        timeout_seconds=120,
        raw_output_dir=None,
        discard_log_dir=None,
    )


if __name__ == "__main__":
    unittest.main()
