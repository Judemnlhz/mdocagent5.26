from __future__ import annotations

import unittest

from mdocnexus.integration.guarded_prompt import build_question_profile, score_guarded_artifact
from mdocnexus.stage2.code_name_list_extractor import extract_code_name_list_artifacts


class CodeNameListExtractorTest(unittest.TestCase):
    def test_extracts_eps_code_name_pairs_without_inventing_missing_code(self) -> None:
        text = "\n".join([
            "EPS",
            "Geographic Market Name",
            "Code",
            "Arkansas (AR)",
            "1.",
            "Little Rock",
            "AR01",
            "2.",
            "Northern Arkansas",
            "AR02",
        ])
        artifacts = extract_code_name_list_artifacts(
            selected_page={"doc_id": "eps.pdf", "page_index": 7},
            page_input=make_page_input(text),
            existing_artifacts=[],
        )

        by_code = {artifact["normalized_content"]["eps_code"]: artifact for artifact in artifacts}
        self.assertEqual(sorted(by_code), ["AR01", "AR02"])
        self.assertEqual(by_code["AR01"]["normalized_content"]["geographic_market_name"], "Little Rock")
        self.assertEqual(by_code["AR02"]["normalized_content"]["geographic_market_name"], "Northern Arkansas")
        self.assertNotIn("AR03", by_code)
        self.assertTrue(by_code["AR01"]["source_anchored"])
        self.assertTrue(by_code["AR01"]["element_locatable"])

    def test_exact_code_selector_can_match_extracted_eps_code(self) -> None:
        text = "\n".join(["EPS", "Geographic Market Name", "Code", "Arkansas (AR)", "1.", "Little Rock", "AR01"])
        artifact = extract_code_name_list_artifacts(
            selected_page={"doc_id": "eps.pdf", "page_index": 7},
            page_input=make_page_input(text),
            existing_artifacts=[],
        )[0]
        profile = build_question_profile("What is the geographic market name for EPS Code AR01?")
        scored = score_guarded_artifact(artifact, "What is the geographic market name for EPS Code AR01?", profile, 7)

        self.assertEqual(scored["exact_code_matches"], ["AR01"])
        self.assertIn("code_match:AR01", scored["selection_reasons"])


def make_page_input(text: str) -> dict:
    return {
        "doc_id": "eps.pdf",
        "page_index": 7,
        "page_text": text,
        "layout_blocks": [
            {
                "block_id": "p007_text_0000",
                "block_type": "text_block",
                "page_index": 7,
                "bbox": None,
                "text": text,
                "char_start": 0,
                "char_end": len(text),
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
