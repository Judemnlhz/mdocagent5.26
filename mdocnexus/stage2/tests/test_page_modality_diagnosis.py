"""Tests for non-gold page modality diagnosis."""

from __future__ import annotations

import unittest

from mdocnexus.stage2.selectors import diagnose_page_modality_from_question_and_preflight


class PageModalityDiagnosisTest(unittest.TestCase):
    def test_chart_question_recommends_visual_figure_and_numeric_fact(self) -> None:
        diagnosis = diagnose_page_modality_from_question_and_preflight(
            record={
                "question": "In the chart, what percentage value is shown for 2020?",
                "answer": "GOLD_POISON",
                "evidence_pages": "[99]",
                "binary_correctness": True,
            },
            page_context=make_page_context(has_image=True),
            page_index=2,
        )

        self.assertTrue(diagnosis["requires_visual_reasoning"])
        self.assertTrue(diagnosis["mentions_chart"])
        self.assertTrue(diagnosis["mentions_numeric_value"])
        self.assertTrue(diagnosis["has_page_image"])
        self.assertIn("visual_observation", diagnosis["recommended_artifact_types"])
        self.assertIn("figure", diagnosis["recommended_artifact_types"])
        self.assertIn("numeric_fact", diagnosis["recommended_artifact_types"])

    def test_table_question_recommends_table_and_numeric_fact(self) -> None:
        diagnosis = diagnose_page_modality_from_question_and_preflight(
            record={"question": "How many entries are listed in the table?"},
            page_context=make_page_context(has_image=True),
            page_index=2,
        )

        self.assertTrue(diagnosis["mentions_table"])
        self.assertIn("table", diagnosis["recommended_artifact_types"])
        self.assertIn("numeric_fact", diagnosis["recommended_artifact_types"])

    def test_visual_recommendation_requires_image_input(self) -> None:
        diagnosis = diagnose_page_modality_from_question_and_preflight(
            record={"question": "What color is visible in the image?"},
            page_context=make_page_context(has_image=False),
            page_index=2,
        )

        self.assertTrue(diagnosis["requires_visual_reasoning"])
        self.assertFalse(diagnosis["has_page_image"])
        self.assertNotIn("visual_observation", diagnosis["recommended_artifact_types"])

    def test_uses_question_text_not_gold_fields(self) -> None:
        diagnosis = diagnose_page_modality_from_question_and_preflight(
            record={
                "question": "What does the page state?",
                "answer": "chart table figure percent",
                "evidence_pages": "[1]",
                "binary_correctness": False,
            },
            page_context=make_page_context(has_image=True),
            page_index=2,
        )

        self.assertFalse(diagnosis["mentions_chart"])
        self.assertFalse(diagnosis["mentions_table"])
        self.assertFalse(diagnosis["mentions_figure"])
        self.assertFalse(diagnosis["mentions_numeric_value"])


def make_page_context(has_image: bool) -> dict:
    return {
        "page_sources": [
            {
                "page_index": 2,
                "has_page_image": has_image,
                "page_image_path": "/tmp/page_2.png" if has_image else None,
                "has_page_text": True,
                "page_text_path": "/tmp/page_2.txt",
                "layout_block_ids": ["p002_text_0000", "p002_full_page_image"],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
