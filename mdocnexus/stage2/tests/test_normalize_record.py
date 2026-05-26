"""Unit tests for Step 1 canonical record normalization."""

from __future__ import annotations

import unittest

from mdocnexus.stage2.normalize_record import normalize_record


APPROVED_TOP_LEVEL_KEYS = {"record_id", "source_record", "canonical_record"}
APPROVED_CANONICAL_KEYS = {
    "document",
    "question",
    "gold_annotation",
    "query_rewrites",
    "question_constraints",
    "retrieval",
    "candidate_pool",
    "compilation_plan",
    "artifact_compilation",
    "baseline_outputs",
}


class NormalizeRecordTest(unittest.TestCase):
    def test_explicit_page_reference_forces_page_to_compile(self) -> None:
        source_record = {
            "doc_id": "example.pdf",
            "doc_type": "slide",
            "question": "What are the blue handwritten words on page 30?",
            "answer_format": "short_text",
            "answer": "do not leak",
            "evidence_pages": "[30]",
            "evidence_sources": "['Page']",
            "retrieval-query": "rewritten query",
            "retrieval-key": "question",
            "qwen_retrieval-query": "qwen query",
            "qwen_retrieval-key": "question",
            "text-index-path-question": "/tmp/index",
            "text-top-10-question": [0, 1, 1],
            "text-top-10-question_score": [9.0, 8.5, 8.0],
            "image-top-10-question": [2],
            "image-top-10-question_score": [7.5],
            "ans_mmlb-MDocAgent": "baseline answer",
            "binary_correctness": 0,
        }

        normalized = normalize_record(source_record)
        canonical_record = normalized["canonical_record"]

        self.assertEqual(set(normalized.keys()), APPROVED_TOP_LEVEL_KEYS)
        self.assertEqual(set(canonical_record.keys()), APPROVED_CANONICAL_KEYS)
        self.assertEqual(normalized["source_record"], source_record)

        explicit_page_references = canonical_record["question_constraints"][
            "explicit_page_references"
        ]
        self.assertEqual(explicit_page_references[0]["page_number_one_based"], 30)
        self.assertEqual(explicit_page_references[0]["page_index_zero_based"], 29)
        self.assertEqual(canonical_record["candidate_pool"]["explicit_constraint_pages"], [29])
        self.assertEqual(canonical_record["candidate_pool"]["retrieval_missed_explicit_pages"], [29])
        self.assertIn(29, canonical_record["compilation_plan"]["pages_to_compile"])
        self.assertEqual(canonical_record["compilation_plan"]["priority_pages"], [29])

        self.assertTrue(canonical_record["gold_annotation"]["eval_only"])
        self.assertTrue(canonical_record["baseline_outputs"]["eval_only"])
        self.assertEqual(
            set(canonical_record["compilation_plan"]["excluded_fields_from_compiler"]),
            {"source_record", "gold_annotation", "baseline_outputs"},
        )
        self.assert_no_hyphenated_canonical_keys(canonical_record)

    def test_retrieval_deduplication_preserves_duplicate_ranks(self) -> None:
        source_record = {
            "doc_id": "example.pdf",
            "question": "What is shown in the chart?",
            "text-top-10-question": [4, 4, 12],
            "text-top-10-question_score": [25.6, 24.3, 24.0],
            "image-top-10-question": [],
            "image-top-10-question_score": [],
        }

        canonical_record = normalize_record(source_record)["canonical_record"]
        ranked_pages_unique = canonical_record["retrieval"]["text"]["ranked_pages_unique"]

        self.assertEqual(ranked_pages_unique[0]["page_index"], 4)
        self.assertEqual(ranked_pages_unique[0]["rank"], 1)
        self.assertEqual(ranked_pages_unique[0]["duplicate_ranks"], [1, 2])
        self.assertEqual(ranked_pages_unique[1]["page_index"], 12)
        self.assertNotIn("duplicate_ranks", ranked_pages_unique[1])

    def test_dynamic_top_k_aliases_are_supported(self) -> None:
        source_record = {
            "doc_id": "example.pdf",
            "question": "What is shown on page 2?",
            "text-top-4-question": [3, 1],
            "text-top-4-question_score": [4.0, 3.0],
            "image-top-4-question": [],
            "image-top-4-question_score": [],
        }

        canonical_record = normalize_record(source_record)["canonical_record"]

        self.assertEqual(canonical_record["retrieval"]["text"]["ranked_pages_raw"], [3, 1])
        self.assertIn(1, canonical_record["compilation_plan"]["pages_to_compile"])

    def assert_no_hyphenated_canonical_keys(self, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                self.assertNotIn("-", key)
                self.assert_no_hyphenated_canonical_keys(child)
        elif isinstance(value, list):
            for item in value:
                self.assert_no_hyphenated_canonical_keys(item)


if __name__ == "__main__":
    unittest.main()
