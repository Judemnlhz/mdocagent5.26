from __future__ import annotations

import json
import unittest

from mdocnexus.integration.guarded_prompt import (
    audit_selected_artifact_support,
    build_question_profile,
    forbidden_public_fields,
    render_guarded_prompt,
    score_guarded_artifact,
    select_guarded_artifacts,
)


class GuardedPromptTests(unittest.TestCase):
    def test_metadata_lookup_rejects_numeric_noise(self) -> None:
        question = "Who produced the document that was revised on May 2018?"
        profile = build_question_profile(question)
        candidates = [score_guarded_artifact(numeric_artifact("noise", "Planning Team: 09"), question, profile, 10)]
        page_contexts = [page_context(1, "Version 1.3 Revised May 2016 Produced by: Florida Department of Health")]

        selection = select_guarded_artifacts(candidates, page_contexts, profile)

        self.assertEqual(selection["guard_decision"], "document_metadata_refusal_guard")
        self.assertEqual(selection["selected_artifacts"], [])

    def test_exact_code_absence_rejects_numeric_noise(self) -> None:
        question = "According to this document, what's the geographic market name for EPS Code AR03?"
        profile = build_question_profile(question)
        candidates = [score_guarded_artifact(numeric_artifact("ar01", "EPS Code AR01: Little Rock"), question, profile, 8)]

        selection = select_guarded_artifacts(candidates, [], profile)

        self.assertEqual(selection["guard_decision"], "exact_code_absence_guard")
        self.assertEqual(selection["selected_artifacts"], [])

    def test_exact_code_present_is_retained(self) -> None:
        question = "According to this document, what's the geographic market name for EPS Code AR03?"
        profile = build_question_profile(question)
        candidates = [
            score_guarded_artifact(
                {
                    "artifact_id": "kv_ar03",
                    "artifact_type": "table_cell",
                    "content": "EPS Code AR03 Geographic Market Name: Central Arkansas",
                    "normalized_content": {"row_label": "AR03", "column_label": "Geographic Market Name", "value_text": "Central Arkansas"},
                    "source_anchored": True,
                },
                question,
                profile,
                8,
            )
        ]

        selection = select_guarded_artifacts(candidates, [], profile)

        self.assertEqual(selection["guard_decision"], "exact_code_key_value_selection")
        self.assertEqual([row["artifact_id"] for row in selection["selected_artifacts"]], ["kv_ar03"])

    def test_operand_incomplete_blocks_calculation(self) -> None:
        question = "What is the percentage difference between older age group with STEM degree and children with the same status?"
        profile = build_question_profile(question)
        candidates = [score_guarded_artifact(numeric_artifact("older", "workers ages 25 and older with STEM degree employed in field: 52%"), question, profile, 40)]

        selection = select_guarded_artifacts(candidates, [], profile)

        self.assertEqual(selection["guard_decision"], "operand_completeness_guard")
        self.assertEqual(selection["selected_artifacts"], [])
        self.assertTrue(any(reason.startswith("missing_operands:") for reason in selection["guard_reasons"]))

    def test_non_guarded_positive_artifact_is_not_cleared(self) -> None:
        question = "Which figure shows both RAPTOR retrieved nodes and questions?"
        profile = build_question_profile(question)
        candidates = [
            score_guarded_artifact(
                {
                    "artifact_id": "fig4",
                    "artifact_type": "caption",
                    "content": "Figure 4 shows RAPTOR retrieved nodes for both questions.",
                    "normalized_content": {"metric_name": "Figure 4", "value_text": "RAPTOR retrieved nodes for both questions"},
                    "source_anchored": True,
                },
                question,
                profile,
                6,
            )
        ]

        selection = select_guarded_artifacts(candidates, [], profile)

        self.assertEqual(selection["guard_decision"], "token_key_value_selection")
        self.assertEqual(len(selection["selected_artifacts"]), 1)

    def test_raptor_metric_overlap_without_question_dimensions_is_guarded(self) -> None:
        question = "In figure 4, which nodes are retrieved by RAPTOR for both questions?"
        profile = build_question_profile(question)
        candidates = [
            score_guarded_artifact(
                {
                    "artifact_id": "raptor_metric",
                    "artifact_type": "numeric_fact",
                    "content": "SBERT with RAPTOR ROUGE: 30.87%",
                    "normalized_content": {
                        "metric_name": "SBERT with RAPTOR",
                        "column_label": "ROUGE",
                        "value_text": "30.87%",
                    },
                    "source_anchored": True,
                },
                question,
                profile,
                6,
            )
        ]

        selection = select_guarded_artifacts(candidates, [], profile)

        self.assertEqual(selection["guard_decision"], "artifact_dimension_support_guard")
        self.assertEqual(selection["selected_artifacts"], [])
        self.assertTrue(any("artifact_missing_dimensions:" in reason for reason in selection["guard_reasons"]))

    def test_raptor_artifact_with_all_question_dimensions_is_retained(self) -> None:
        question = "In figure 4, which nodes are retrieved by RAPTOR for both questions?"
        profile = build_question_profile(question)
        candidates = [
            score_guarded_artifact(
                {
                    "artifact_id": "fig4_nodes",
                    "artifact_type": "caption",
                    "content": "Figure 4 shows the RAPTOR retrieved nodes for both questions.",
                    "normalized_content": {
                        "metric_name": "Figure 4",
                        "value_text": "RAPTOR retrieved nodes for both questions",
                    },
                    "source_anchored": True,
                },
                question,
                profile,
                6,
            )
        ]

        selection = select_guarded_artifacts(candidates, [], profile)
        support = audit_selected_artifact_support(selection["selected_artifacts"], [], profile)

        self.assertEqual(selection["guard_decision"], "token_key_value_selection")
        self.assertEqual([row["artifact_id"] for row in selection["selected_artifacts"]], ["fig4_nodes"])
        self.assertTrue(support["artifact_support_sufficient"])

    def test_demographic_artifact_must_cover_group_metric_time_and_values(self) -> None:
        question = (
            "Among the Higher-income seniors, what are the percentage of them go online, "
            "has smartphone phone, and own a tablet computer in the Pew Research Center's "
            "Internet Project July 18-September 30, 2013 tracking survey?"
        )
        profile = build_question_profile(question)
        weak = score_guarded_artifact(
            {
                "artifact_id": "smartphone_only",
                "artifact_type": "numeric_fact",
                "content": "devices among older adults have high levels of smartphone: 18%",
                "normalized_content": {
                    "metric_name": "devices among older adults",
                    "column_label": "have high levels of smartphone",
                    "value_text": "18%",
                },
                "source_anchored": True,
            },
            question,
            profile,
            8,
        )
        strong = score_guarded_artifact(
            {
                "artifact_id": "higher_income_2013",
                "artifact_type": "table",
                "content": "2013 Higher-income seniors go online: 80%; smartphone: 50%; tablet computer: 30%",
                "normalized_content": {
                    "row_label": "Higher-income seniors",
                    "column_label": "2013 go online smartphone tablet computer",
                    "value_text": "80%; 50%; 30%",
                },
                "source_anchored": True,
            },
            question,
            profile,
            8,
        )

        weak_selection = select_guarded_artifacts([weak], [], profile)
        strong_selection = select_guarded_artifacts([strong], [], profile)

        self.assertEqual(weak_selection["guard_decision"], "artifact_dimension_support_guard")
        self.assertEqual(strong_selection["guard_decision"], "token_key_value_selection")
        self.assertEqual([row["artifact_id"] for row in strong_selection["selected_artifacts"]], ["higher_income_2013"])

    def test_prompt_and_public_payload_are_no_gold(self) -> None:
        question = "Which figure shows both RAPTOR retrieved nodes and questions?"
        profile = build_question_profile(question)
        artifact = score_guarded_artifact(
            {
                "artifact_id": "fig4",
                "artifact_type": "caption",
            "content": "Figure 4 shows RAPTOR retrieved nodes for both questions.",
                "source_anchored": True,
            },
            question,
            profile,
            6,
        )
        selection = select_guarded_artifacts([artifact], [], profile)
        prompt = render_guarded_prompt(question, [], selection, profile)
        payload = {"question": question, "selection": selection, "prompt_preview": prompt}

        self.assertEqual(forbidden_public_fields(payload), [])
        self.assertIn("[Selected artifact evidence]", prompt)
        self.assertIn("Final answer: answer or Not answerable.", prompt)


def numeric_artifact(artifact_id: str, content: str) -> dict:
    return {
        "artifact_id": artifact_id,
        "artifact_type": "numeric_fact",
        "modality": "numeric",
        "content": content,
        "normalized_content": {"metric_name": content, "value_text": "09"},
        "source_anchored": True,
    }


def page_context(page_index: int, text: str) -> dict:
    return {"page_index": page_index, "exists": True, "text_preview": text}


if __name__ == "__main__":
    unittest.main()
