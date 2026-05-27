"""Tests for MDocAgent-aligned Stage 2 JSON integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.mdocagent_aligned_stage2 import (
    augment_retrieval_records,
    build_single_page_trial_summary,
    select_trial_candidate_from_stage2_records,
    write_single_page_trial_summary,
)


ORIGINAL_FIELDS = [
    "doc_id",
    "doc_type",
    "question",
    "answer",
    "evidence_pages",
    "evidence_sources",
    "answer_format",
    "retrieval-query",
    "retrieval-key",
    "qwen_retrieval-query",
    "qwen_retrieval-key",
    "text-index-path-question",
    "text-top-10-question",
    "text-top-10-question_score",
    "image-top-10-question",
    "image-top-10-question_score",
    "ans_mmlb-MDocAgent",
    "binary_correctness",
]

REMOVED_STAGE2_FIELDS = {
    "version",
    "doc_name",
    "page_count",
    "question_constraints",
    "retrieval_pages",
    "explicit_page_validation",
    "pages_to_compile",
    "page_sources",
    "should_call_api",
    "should_generate_artifact",
}


class MDocAgentAlignedStage2JsonTest(unittest.TestCase):
    def test_original_fields_preserved_and_only_stage2_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            record = make_record(question="What is on page 2?", text_pages=[0], image_pages=[1])
            create_extract_pages(extract_root, "example", [0, 1])

            augmented = augment_retrieval_records([record], extract_root)[0]

        for field in ORIGINAL_FIELDS:
            self.assertIn(field, augmented)
            self.assertEqual(augmented[field], record[field])
        self.assertEqual(set(augmented.keys()) - set(record.keys()), {"stage2"})
        self.assertIn("text-top-10-question", augmented)
        self.assertNotIn("canonical_record", augmented)
        self.assertNotIn("source_record", augmented)
        self.assertNotIn("record_id", augmented)

    def test_candidate_page_routes_come_from_original_top_10_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            record = make_record(text_pages=[3, 3, 1], image_pages=[2, 2])
            create_extract_pages(extract_root, "example", [1, 2, 3])

            stage2 = augment_retrieval_records([record], extract_root)[0]["stage2"]

        self.assertEqual(
            stage2["candidate_page_routes"],
            [
                {"page_index": 1, "routes": ["text"]},
                {"page_index": 2, "routes": ["image"]},
                {"page_index": 3, "routes": ["text"]},
            ],
        )
        self.assertEqual(set(stage2), {"preflight", "candidate_page_routes"})
        for removed in REMOVED_STAGE2_FIELDS:
            self.assertNotIn(removed, collect_keys(stage2))

    def test_out_of_range_explicit_page_not_in_candidate_page_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            record = make_record(question="What is on page 30?", text_pages=[0], image_pages=[])
            create_extract_pages(extract_root, "example", list(range(20)))

            stage2 = augment_retrieval_records([record], extract_root)[0]["stage2"]

        self.assertNotIn(29, [route["page_index"] for route in stage2["candidate_page_routes"]])
        self.assertIn("explicit_page_reference_out_of_range", stage2["preflight"]["blocking_reasons"])
        self.assertFalse(stage2["preflight"]["passed"])

    def test_valid_explicit_page_does_not_expand_candidate_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            record = make_record(question="What is on page 2?", text_pages=[0], image_pages=[])
            create_extract_pages(extract_root, "example", [0, 1])

            stage2 = augment_retrieval_records([record], extract_root)[0]["stage2"]

        self.assertEqual(stage2["candidate_page_routes"], [{"page_index": 0, "routes": ["text"]}])
        self.assertTrue(stage2["preflight"]["passed"])

    def test_forbidden_gold_fields_do_not_enter_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            record = make_record(question="What is on page 1?", text_pages=[], image_pages=[])
            create_extract_pages(extract_root, "example", [0])

            stage2 = augment_retrieval_records([record], extract_root)[0]["stage2"]

        keys = collect_keys(stage2)
        for forbidden in ["answer", "evidence_pages", "evidence_sources", "binary_correctness"]:
            self.assertNotIn(forbidden, keys)
        serialized = json.dumps(stage2, ensure_ascii=False)
        self.assertNotIn("GOLD_SECRET", serialized)
        self.assertNotIn("BASELINE_SECRET", serialized)

    def test_candidate_selector_does_not_use_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            records = [
                make_record(doc_id="doc_a.pdf", question="What is on page 30?", image_pages=[0]),
                make_record(doc_id="doc_b.pdf", question="What is on page 2?", image_pages=[1]),
            ]
            create_extract_pages(extract_root, "doc_a", [0])
            create_extract_pages(extract_root, "doc_b", [0, 1])
            augmented = augment_retrieval_records(records, extract_root)

            report = select_trial_candidate_from_stage2_records(augmented, extract_root=extract_root)

        self.assertTrue(report["selection_passed"])
        self.assertEqual(report["selected"]["doc_id"], "doc_b.pdf")
        self.assertEqual(report["selected"]["page_index"], 1)
        self.assertFalse(report["selection_policy"]["uses_answer"])
        self.assertFalse(report["selection_policy"]["uses_evidence_pages"])
        self.assertFalse(report["selection_policy"]["uses_binary_correctness"])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("GOLD_SECRET", serialized)
        self.assertNotIn("BASELINE_SECRET", serialized)
        self.assertNotIn("api_key", serialized)

    def test_summary_does_not_contain_api_key_or_forbidden_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "single_page_trial_summary.json"
            summary = build_single_page_trial_summary(
                provider="siliconflow",
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                record_id="record_1",
                doc_id="example.pdf",
                page_index=13,
                raw_output_log_path="raw.jsonl",
                artifact_store_path="artifact.json",
                num_raw_artifacts=1,
                num_valid_artifacts=1,
                num_validation_issues=0,
                single_page_smoke_test_passed=True,
            )
            write_single_page_trial_summary(summary, output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        keys = collect_keys(loaded)
        for forbidden in [
            "api_key",
            "answer",
            "evidence_pages",
            "binary_correctness",
            "proof_trace",
            "verified",
            "answer_supported",
            "proof_used",
        ]:
            self.assertNotIn(forbidden, keys)


def make_record(
    doc_id: str = "example.pdf",
    question: str = "What is visible?",
    text_pages: List[int] | None = None,
    image_pages: List[int] | None = None,
) -> Dict[str, Any]:
    text_pages = [] if text_pages is None else text_pages
    image_pages = [] if image_pages is None else image_pages
    return {
        "doc_id": doc_id,
        "doc_type": "test",
        "question": question,
        "answer": "GOLD_SECRET",
        "evidence_pages": "[99]",
        "evidence_sources": "['secret']",
        "answer_format": "Str",
        "retrieval-query": "query",
        "retrieval-key": "key",
        "qwen_retrieval-query": "qwen query",
        "qwen_retrieval-key": "qwen key",
        "text-index-path-question": "index/path",
        "text-top-10-question": text_pages,
        "text-top-10-question_score": [1.0 for _ in text_pages],
        "image-top-10-question": image_pages,
        "image-top-10-question_score": [1.0 for _ in image_pages],
        "ans_mmlb-MDocAgent": "BASELINE_SECRET",
        "binary_correctness": True,
    }


def create_extract_pages(extract_root: Path, doc_stem: str, page_indices: List[int]) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    for page_index in page_indices:
        (extract_root / f"{doc_stem}_{page_index}.png").write_bytes(b"not-a-real-png")
        (extract_root / f"{doc_stem}_{page_index}.txt").write_text(
            f"text for page {page_index}",
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


if __name__ == "__main__":
    unittest.main()
