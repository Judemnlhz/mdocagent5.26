from __future__ import annotations

import unittest

from mdocnexus.integration.evidence_skill_registry import (
    DOCUMENT_EDGE_TYPES,
    EVIDENCE_UNIT_TYPES,
    activated_skills,
    build_skill_trace,
    registry_contract,
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
