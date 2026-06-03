"""Tests for Step 3 schema serialization and validation skeleton."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from mdocnexus.stage2.artifact_pipeline import validate_page_artifact_output
from mdocnexus.stage2.artifact_quality import classify_artifact_quality, is_atomic_strong_eligible
from mdocnexus.stage2.logs import (
    issue_to_discard_log_entry,
    write_discard_log_entry,
)
from mdocnexus.stage2.artifact_schema import (
    build_page_artifact_output_schema_dict,
    get_allowed_validation_statuses,
)
from mdocnexus.stage2.artifact_schema import ValidationErrorType, ValidationIssue

class ArtifactSchemaValidationTest(unittest.TestCase):
    def test_schema_serialization(self) -> None:
        schema = build_page_artifact_output_schema_dict()
        validation_statuses = get_allowed_validation_statuses()
        artifact_type_enum = schema["properties"]["artifacts"]["items"]["properties"][
            "artifact_type"
        ]["enum"]

        self.assertIsInstance(schema, dict)
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("verified", validation_statuses)
        self.assertNotIn("proof_trace", validation_statuses)
        self.assertNotIn("answer_supported", validation_statuses)
        self.assertNotIn("proof_used", validation_statuses)
        self.assertIn("visual_observation", artifact_type_enum)
        self.assertIn("numeric_fact", artifact_type_enum)
        for removed_type in [
            "page_summary",
            "document_identity",
            "reference_section",
            "reference_entry",
            "organization_mention",
            "claim_candidate",
            "handwriting_observation",
            "color_observation",
        ]:
            self.assertNotIn(removed_type, artifact_type_enum)
        modality_enum = schema["properties"]["artifacts"]["items"]["properties"]["modality"]["enum"]
        self.assertEqual(modality_enum, ["text", "image", "table", "layout", "numeric"])

    def test_valid_artifact_passes(self) -> None:
        layout_blocks = [make_full_page_image_block()]
        raw_output = {
            "doc_id": "example.pdf",
            "page_index": 29,
            "artifacts": [make_artifact()],
            "uncertain_or_unreadable": [],
        }

        valid_artifacts, issues = validate_page_artifact_output(raw_output, layout_blocks)

        self.assertEqual(issues, [])
        self.assertEqual(len(valid_artifacts), 1)
        self.assertEqual(valid_artifacts[0]["validation_status"], "anchored")

    def test_missing_source_anchor_fails(self) -> None:
        layout_blocks = [make_full_page_image_block()]
        artifact = make_artifact()
        artifact["source_anchors"][0]["source_id"] = "not_exist"
        raw_output = {
            "doc_id": "example.pdf",
            "page_index": 29,
            "artifacts": [artifact],
        }

        valid_artifacts, issues = validate_page_artifact_output(raw_output, layout_blocks)

        self.assertEqual(valid_artifacts, [])
        self.assertIn(
            ValidationErrorType.source_anchor_not_found,
            {issue.error_type for issue in issues},
        )

    def test_invalid_enum_fails(self) -> None:
        layout_blocks = [make_full_page_image_block()]
        artifact = make_artifact()
        artifact["artifact_type"] = "proof_trace"
        raw_output = {
            "doc_id": "example.pdf",
            "page_index": 29,
            "artifacts": [artifact],
        }

        valid_artifacts, issues = validate_page_artifact_output(raw_output, layout_blocks)

        self.assertEqual(valid_artifacts, [])
        self.assertIn(
            ValidationErrorType.invalid_enum_value,
            {issue.error_type for issue in issues},
        )

    def test_non_empty_content_is_not_phrase_filtered(self) -> None:
        layout_blocks = [make_full_page_image_block()]
        artifact = make_artifact()
        artifact["content"] = "A model-supplied sentence is treated as content when structure is valid."
        raw_output = {
            "doc_id": "example.pdf",
            "page_index": 29,
            "artifacts": [artifact],
        }

        valid_artifacts, issues = validate_page_artifact_output(raw_output, layout_blocks)

        self.assertEqual(issues, [])
        self.assertEqual(len(valid_artifacts), 1)

    def test_forbidden_artifact_fields_are_rejected(self) -> None:
        forbidden_fields = [
            "answer",
            "prediction",
            "final_answer",
            "binary_correctness",
            "evidence_pages",
            "evidence_sources",
        ]
        layout_blocks = [make_full_page_image_block()]
        for field_name in forbidden_fields:
            with self.subTest(field_name=field_name):
                artifact = make_artifact()
                artifact[field_name] = "forbidden"
                raw_output = {
                    "doc_id": "example.pdf",
                    "page_index": 29,
                    "artifacts": [artifact],
                }

                valid_artifacts, issues = validate_page_artifact_output(raw_output, layout_blocks)

                self.assertEqual(valid_artifacts, [])
                self.assertIn(ValidationErrorType.schema_invalid, {issue.error_type for issue in issues})
                self.assertIn(field_name, {str(issue.details.get("unexpected_field")) for issue in issues})

    def test_evidence_numeric_fact_passes(self) -> None:
        layout_blocks = [make_full_page_image_block()]
        artifact = make_artifact()
        artifact["artifact_type"] = "numeric_fact"
        artifact["modality"] = "numeric"
        artifact["content"] = "The chart shows 8 Redacted signal labels."
        raw_output = {"doc_id": "example.pdf", "page_index": 29, "artifacts": [artifact]}

        valid_artifacts, issues = validate_page_artifact_output(raw_output, layout_blocks)

        self.assertEqual(issues, [])
        self.assertEqual(len(valid_artifacts), 1)

    def test_duplicate_artifact_detected(self) -> None:
        layout_blocks = [make_full_page_image_block()]
        artifact_1 = make_artifact(artifact_id="artifact_a")
        artifact_2 = make_artifact(artifact_id="artifact_b")
        raw_output = {
            "doc_id": "example.pdf",
            "page_index": 29,
            "artifacts": [artifact_1, artifact_2],
        }

        valid_artifacts, issues = validate_page_artifact_output(raw_output, layout_blocks)

        self.assertIn(
            ValidationErrorType.duplicate_artifact,
            {issue.error_type for issue in issues},
        )
        self.assertEqual([artifact["artifact_id"] for artifact in valid_artifacts], ["artifact_a"])

    def test_discard_log_serialization(self) -> None:
        issue = ValidationIssue(
            error_type=ValidationErrorType.source_anchor_not_found,
            message="Anchor missing.",
            doc_id="example.pdf",
            page_index=29,
            artifact_id="artifact_a",
            field_path="source_anchors[0].source_id",
            details={"source_id": "not_exist"},
        )
        entry = issue_to_discard_log_entry(
            issue=issue,
            stage="stage2_validation",
            compiler_version="test_version",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "discard.jsonl"
            write_discard_log_entry(log_path, entry)
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["error_type"], "source_anchor_not_found")
        self.assertEqual(rows[0]["doc_id"], "example.pdf")
        self.assertEqual(rows[0]["page_index"], 29)
        self.assertEqual(rows[0]["artifact_id"], "artifact_a")
        self.assertEqual(rows[0]["stage"], "stage2_validation")

    def test_artifact_quality_marks_table_title_only_as_broad(self) -> None:
        artifact = make_artifact()
        artifact["artifact_type"] = "table"
        artifact["modality"] = "table"
        artifact["content"] = "International Segment Performance Summary"
        artifact["normalized_content"] = {"table_id": "table_001", "table_title": "International Segment Performance Summary"}
        artifact["locators"] = [{"locator_kind": "text_offset", "block_id": "p029_full_page_image", "char_start": 0, "char_end": 10}]

        quality = classify_artifact_quality(artifact)

        self.assertTrue(quality["broad_table_only"])
        self.assertTrue(quality["caption_or_table_title_only"])
        self.assertTrue(quality["schema_valid_but_semantically_weak"])

    def test_artifact_quality_marks_complete_numeric_fact_as_atomic(self) -> None:
        artifact = make_artifact()
        artifact["artifact_type"] = "numeric_fact"
        artifact["modality"] = "numeric"
        artifact["content"] = "Revenue 2023: 3,504 USD millions"
        artifact["normalized_content"] = {
            "metric_name": "Revenue",
            "row_label": "Revenue",
            "column_label": "2023",
            "value_text": "3,504",
            "unit": "USD millions",
            "source_text": "Revenue 2023 3,504",
        }
        artifact["locators"] = [{"locator_kind": "text_offset", "block_id": "p029_full_page_image", "char_start": 0, "char_end": 10}]

        quality = classify_artifact_quality(artifact)

        self.assertTrue(quality["atomic_numeric_ok"])
        self.assertFalse(quality["schema_valid_but_semantically_weak"])
        self.assertTrue(is_atomic_strong_eligible(artifact, "eligible"))


def make_full_page_image_block() -> Dict[str, Any]:
    return {
        "block_id": "p029_full_page_image",
        "block_type": "full_page_image",
        "page_index": 29,
        "bbox": None,
        "text": None,
    }


def make_artifact(artifact_id: str = "artifact_a") -> Dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "doc_id": "example.pdf",
        "page_index": 29,
        "artifact_type": "visual_observation",
        "modality": "image",
        "content": "A page-level visual observation.",
        "normalized_content": {},
        "source_anchors": [
            {
                "source_id": "p029_full_page_image",
                "anchor_type": "full_page_image",
                "page_index": 29,
                "bbox": None,
            }
        ],
        "provenance": {
            "op": "ATOM",
            "sources": ["p029_full_page_image"],
        },
        "validation_status": "candidate",
        "compiler_metadata": {},
    }


if __name__ == "__main__":
    unittest.main()
