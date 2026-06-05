from __future__ import annotations

import unittest

from mdocnexus.integration.evidence_skill_registry import (
    DOCUMENT_EDGE_TYPES,
    EVIDENCE_UNIT_TYPES,
    activated_skills,
    build_skill_trace,
    estimate_tokens,
    flat_artifact_context,
    raw_page_context,
    registry_contract,
    render_evidence_capsule,
    validate_registry_contract,
)
from mdocnexus.integration.guarded_prompt import build_question_profile, score_guarded_artifact, select_guarded_artifacts


class EvidenceSkillRegistryTests(unittest.TestCase):
    def test_registry_contract_is_lightweight_and_dataset_agnostic(self) -> None:
        contract = registry_contract()

        self.assertEqual(validate_registry_contract(contract), [])
        self.assertLessEqual(len(EVIDENCE_UNIT_TYPES), 6)
        self.assertLessEqual(len(DOCUMENT_EDGE_TYPES), 8)
        self.assertTrue(contract["boundaries"]["dataset_agnostic"])
        self.assertTrue(contract["boundaries"]["not_large_skill_tree"])
        self.assertTrue(contract["boundaries"]["not_global_knowledge_graph"])
        joined_names = " ".join(skill["name"] for skill in contract["skills"])
        for marker in ["mmlb", "ldu", "ptab", "ptext", "feta"]:
            self.assertNotIn(marker, joined_names)

    def test_exact_code_skill_trace_reports_missing_required_code(self) -> None:
        question = "According to this document, what's the geographic market name for EPS Code AR03?"
        profile = build_question_profile(question)
        artifact = score_guarded_artifact(
            {
                "artifact_id": "ar01",
                "artifact_type": "table_cell",
                "content": "EPS Code AR01 Geographic Market Name: Little Rock",
                "normalized_content": {"row_label": "AR01", "column_label": "Geographic Market Name", "value_text": "Little Rock"},
                "source_anchored": True,
            },
            question,
            profile,
            8,
        )
        selection = select_guarded_artifacts([artifact], [], profile)

        trace = build_skill_trace(profile, question, selection, [artifact])

        self.assertEqual(trace["activated_skill_names"], ["exact_code_lookup"])
        self.assertEqual(trace["traces"][0]["guard_rule"], "exact_code_absence_guard")
        self.assertIn("exact_code:AR03", trace["traces"][0]["missing_requirements"])
        self.assertEqual(trace["guard_decision"], "exact_code_absence_guard")


    def test_capsule_renderer_is_compact_and_auditable(self) -> None:
        question = "According to this document, what's the geographic market name for EPS Code AR03?"
        profile = build_question_profile(question)
        artifact = score_guarded_artifact(
            {
                "artifact_id": "ar01",
                "artifact_type": "table_cell",
                "content": "EPS Code AR01 Geographic Market Name: Little Rock",
                "normalized_content": {"row_label": "AR01", "column_label": "Geographic Market Name", "value_text": "Little Rock"},
                "source_anchored": True,
            },
            question,
            profile,
            8,
        )
        selection = select_guarded_artifacts([artifact], [], profile)

        capsule = render_evidence_capsule(question, profile, selection, [artifact], max_units=2)

        self.assertIn("[Evidence Capsule]", capsule["text"])
        self.assertIn("exact_code:AR03", capsule["missing_requirements"])
        self.assertGreater(capsule["token_estimate"], 0)
        self.assertLessEqual(capsule["unit_count"], 2)
        self.assertTrue(capsule["boundary"]["no_provider_calls"])

    def test_page_visible_operand_route_does_not_report_missing_operands(self) -> None:
        question = "what is Amazon's FY2017 return on asset ? round your answer to three decimal"
        profile = build_question_profile(question)
        artifact = score_guarded_artifact(
            {
                "artifact_id": "cash",
                "artifact_type": "numeric_fact",
                "content": "Cash and cash equivalents: 20522",
                "normalized_content": {"metric_name": "Cash and cash equivalents", "value_text": "20522"},
                "source_anchored": True,
            },
            question,
            profile,
            36,
        )
        pages = [{"page_index": 36, "exists": True, "text_preview": "Net income 2017 was $3,033 million. Total assets 2017 were $131,310 million."}]
        selection = select_guarded_artifacts([artifact], pages, profile)

        capsule = render_evidence_capsule(question, profile, selection, [artifact], max_units=2)

        self.assertEqual(selection["guard_decision"], "operand_page_evidence_route")
        self.assertEqual(capsule["missing_requirements"], [])
        self.assertNotIn("Missing: operand:", capsule["text"])

    def test_raw_flat_and_capsule_contexts_are_token_countable(self) -> None:
        page_text = "alpha beta gamma " * 100
        raw = raw_page_context([{"page_index": 1, "exists": True, "text_preview": page_text}], max_chars_per_page=500)
        flat = flat_artifact_context([
            {
                "artifact_id": "a1",
                "artifact_type": "text_span",
                "page_index": 1,
                "content_preview": "alpha beta gamma",
                "selection_score": 1.0,
                "normalized_content": {},
            }
        ])

        self.assertGreater(estimate_tokens(raw), estimate_tokens(flat))
        self.assertIn("[Raw Page Context]", raw)
        self.assertIn("[Flat Artifact Context]", flat)

    def test_numeric_computation_skill_precedes_table_lookup(self) -> None:
        question = "What is the percentage difference between older age group with STEM degree and children with the same status?"
        profile = build_question_profile(question)

        skills = activated_skills(profile, question)

        self.assertEqual([skill.name for skill in skills], ["numeric_computation"])

    def test_figure_caption_skill_activates_for_visual_question(self) -> None:
        question = "Which figure shows both RAPTOR retrieved nodes and questions?"
        profile = build_question_profile(question)

        skills = activated_skills(profile, question)

        self.assertIn("figure_caption_grounding", [skill.name for skill in skills])

    def test_general_question_falls_back_to_text_span_grounding(self) -> None:
        question = "Describe the policy context."
        profile = build_question_profile(question)

        skills = activated_skills(profile, question)

        self.assertEqual([skill.name for skill in skills], ["text_span_grounding"])


if __name__ == "__main__":
    unittest.main()
