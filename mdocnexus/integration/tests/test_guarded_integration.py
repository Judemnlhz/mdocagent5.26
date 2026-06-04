from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mdocnexus.integration.guarded_integration import (
    GuardedPromptIntegrationConfig,
    apply_guarded_prompt_integration,
    guarded_prompt_integration_contract,
    write_integration_outputs,
)
from mdocnexus.integration.mdocagent_adapter import canonical_json_hash


class GuardedIntegrationTests(unittest.TestCase):
    def test_default_disabled_returns_records_unchanged_and_no_previews(self) -> None:
        records = [sample_record()]
        before_hash = canonical_json_hash(records)

        result = apply_guarded_prompt_integration(records, sample_artifacts_by_page(), sample_page_contexts())

        self.assertFalse(result["enabled"])
        self.assertTrue(result["records_unchanged"])
        self.assertEqual(result["records"], records)
        self.assertEqual(result["input_records_sha256"], before_hash)
        self.assertEqual(result["output_records_sha256"], before_hash)
        self.assertEqual(result["prompt_previews"], [])
        self.assertEqual(result["manifest"]["num_prompt_previews"], 0)
        self.assertTrue(result["manifest"]["no_provider_calls"])

    def test_enabled_builds_prompt_preview_without_changing_records(self) -> None:
        records = [sample_record()]
        config = GuardedPromptIntegrationConfig(enable_guarded_prompt_scaffold=True)

        result = apply_guarded_prompt_integration(records, sample_artifacts_by_page(), sample_page_contexts(), config)

        self.assertTrue(result["enabled"])
        self.assertTrue(result["records_unchanged"])
        self.assertEqual(result["records"], records)
        self.assertEqual(result["input_records_sha256"], result["output_records_sha256"])
        self.assertEqual(len(result["prompt_previews"]), 1)
        preview = result["prompt_previews"][0]
        self.assertEqual(preview["guard_decision"], "token_key_value_selection")
        self.assertGreater(preview["selected_artifact_count"], 0)
        self.assertEqual(preview["forbidden_gold_fields_present"], [])
        self.assertTrue(result["manifest"]["no_gold_fields_in_public_previews"])
        self.assertTrue(result["manifest"]["not_artifact_lift_claim"])

    def test_contract_declares_default_off_and_forbidden_inputs(self) -> None:
        contract = guarded_prompt_integration_contract()

        self.assertFalse(contract["default_enabled"])
        self.assertEqual(contract["config_flag"], "enable_guarded_prompt_scaffold")
        self.assertIn("answer", contract["forbidden_inputs"])
        self.assertIn("official score", contract["does_not_do"])
        self.assertIn("provider call", contract["does_not_do"])

    def test_write_outputs_contains_manifest_and_jsonl_without_gold_values(self) -> None:
        records = [sample_record()]
        result = apply_guarded_prompt_integration(
            records,
            sample_artifacts_by_page(),
            sample_page_contexts(),
            {"enable_guarded_prompt_scaffold": True},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_integration_outputs(result, tmpdir)
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            previews = [json.loads(line) for line in Path(paths["prompt_previews"]).read_text(encoding="utf-8").splitlines()]

        self.assertEqual(manifest["num_prompt_previews"], 1)
        self.assertEqual(len(previews), 1)
        serialized = json.dumps({"manifest": manifest, "previews": previews})
        self.assertNotIn("GOLD_SHOULD_NOT_APPEAR", serialized)
        self.assertNotIn("GOLD_ARTIFACT_SECRET", serialized)
        self.assertEqual(previews[0]["forbidden_gold_fields_present"], [])


def sample_record() -> dict[str, object]:
    return {
        "record_index": 12,
        "doc_id": "doc.pdf",
        "question": "Which figure shows alpha revenue?",
        "answer": "GOLD_SHOULD_NOT_APPEAR",
        "evidence_pages": [4],
        "binary_correctness": True,
        "text-top-10-question": [4, 1],
        "text-top-10-question_score": [1.0, 0.5],
        "image-top-10-question": [4, 2],
        "image-top-10-question_score": [1.0, 0.2],
    }


def sample_artifacts_by_page() -> dict[str, dict[int, list[dict[str, object]]]]:
    return {
        "doc.pdf": {
            4: [
                {
                    "artifact_id": "caption_4",
                    "artifact_type": "caption",
                    "modality": "text",
                    "doc_id": "doc.pdf",
                    "page_index": 4,
                    "content": "Figure 4 shows alpha revenue by segment.",
                    "normalized_content": {"metric_name": "Figure 4", "value_text": "alpha revenue"},
                    "source_anchored": True,
                    "validation_status": "anchored",
                    "answer": "GOLD_ARTIFACT_SECRET",
                }
            ]
        }
    }


def sample_page_contexts() -> dict[tuple[str, int], dict[str, object]]:
    return {
        ("doc.pdf", 4): {"text": "Figure 4 shows alpha revenue by segment."},
        ("doc.pdf", 1): {"text": "Unrelated page."},
        ("doc.pdf", 2): {"text": "Another unrelated page."},
    }


if __name__ == "__main__":
    unittest.main()