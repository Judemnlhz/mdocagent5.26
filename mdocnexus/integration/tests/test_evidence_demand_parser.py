from __future__ import annotations

import unittest

from mdocnexus.integration.evidence_demand_parser import (
    build_evidence_demand_prompt,
    evidence_demand_contract,
    merge_evidence_demand_profile,
    normalize_evidence_demand,
    parse_evidence_demand_response,
    validate_public_parser_payload,
)


class EvidenceDemandParserTests(unittest.TestCase):
    def test_prompt_is_question_only_parser_not_answer_generation(self) -> None:
        prompt = build_evidence_demand_prompt("What is EPS Code AR03?")

        self.assertIn("Do not answer the question", prompt)
        self.assertIn("Use only the question text", prompt)
        self.assertIn("required_values_or_codes", prompt)
        self.assertNotIn("gold_answer", prompt)

    def test_parse_response_normalizes_schema(self) -> None:
        parsed = parse_evidence_demand_response(
            "```json\n"
            "{\n"
            "  \"answer_type\": \"table_lookup\",\n"
            "  \"required_entities\": [\"EPS Code\"],\n"
            "  \"required_metrics\": [\"Geographic Market Name\"],\n"
            "  \"required_values_or_codes\": [\"AR03\"],\n"
            "  \"evidence_dimensions\": [{\"dimension\": \"eps_code\", \"label\": \"EPS Code\", \"aliases\": [\"EPS Code\"]}],\n"
            "  \"min_numeric_values\": 0\n"
            "}\n```"
        )

        self.assertEqual(parsed["answer_type"], "table_lookup")
        self.assertTrue(parsed["requires_exact_code_selection"])
        self.assertTrue(parsed["is_numeric_or_table_question"])
        self.assertEqual(parsed["required_values_or_codes"], ["AR03"])
        self.assertEqual(parsed["evidence_dimensions"][0]["dimension"], "eps_code")

    def test_merge_demand_profile_adds_dimensions_without_gold(self) -> None:
        question = "According to this document, what's the geographic market name for EPS Code AR03?"
        demand = normalize_evidence_demand({
            "answer_type": "table_lookup",
            "required_metrics": ["geographic market name"],
            "required_values_or_codes": ["AR03"],
            "evidence_dimensions": [
                {"dimension": "eps_code_ar03", "label": "EPS Code AR03", "aliases": ["AR03", "EPS Code AR03"]},
                {"dimension": "geographic_market_name", "label": "Geographic Market Name", "aliases": ["geographic market name"]},
            ],
        })

        profile = merge_evidence_demand_profile(question, demand)

        self.assertEqual(profile["profile_source"], "rule_profile_plus_llm_evidence_demand")
        self.assertTrue(profile["requires_exact_code_selection"])
        self.assertEqual(profile["evidence_requirements"]["source"], "llm_evidence_demand_parser")
        self.assertEqual(len(profile["evidence_requirements"]["dimensions"]), 2)
        self.assertEqual(validate_public_parser_payload({"question": question, "profile": profile}), [])

    def test_contract_is_default_off_and_forbids_gold(self) -> None:
        contract = evidence_demand_contract()

        self.assertFalse(contract["default_enabled"])
        self.assertEqual(contract["config_flag"], "enable_llm_evidence_demand_parser")
        self.assertIn("gold_answer", contract["forbidden_inputs"])
        self.assertIn("artifact selection by LLM", contract["does_not_do"])


if __name__ == "__main__":
    unittest.main()
