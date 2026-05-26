"""Tests for Step 5 artifact compiler interface and controlled dry run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.artifact_compiler import compile_page_with_client
from mdocnexus.stage2.artifact_validator import validate_page_artifact_output
from mdocnexus.stage2.compiler_client import (
    ArtifactCompilerClient,
    FakeArtifactCompilerClient,
    RealArtifactCompilerClient,
)
from mdocnexus.stage2.compiler_integration import run_stage2_compiler_dry_run
from mdocnexus.stage2.compiler_prompt import build_artifact_compiler_user_prompt
from mdocnexus.stage2.schema_serialization import (
    build_page_artifact_output_schema_dict,
    get_allowed_validation_statuses,
)
from mdocnexus.stage2.validation_errors import ValidationErrorType


class ArtifactCompilerInterfaceTest(unittest.TestCase):
    def test_prompt_does_not_leak_gold_or_baseline(self) -> None:
        canonical_record = make_canonical_record([29])
        canonical_record["gold_annotation"] = {"answer": "GOLD_SECRET", "eval_only": True}
        canonical_record["baseline_outputs"] = {"mdocagent": {"answer": "BASELINE_SECRET"}}
        page_input = make_page_input(29)

        prompt = build_artifact_compiler_user_prompt(
            canonical_record=canonical_record,
            page_input=page_input,
            schema_dict=build_page_artifact_output_schema_dict(),
        )

        self.assertNotIn("GOLD_SECRET", prompt)
        self.assertNotIn("BASELINE_SECRET", prompt)
        self.assertNotIn("gold_annotation", prompt)
        self.assertNotIn("baseline_outputs", prompt)
        self.assertIn("What is visible on page 30?", prompt)
        self.assertIn("question_constraints", prompt)
        self.assertIn("layout_blocks", prompt)

    def test_fake_client_output_validates(self) -> None:
        client = FakeArtifactCompilerClient()
        page_input = make_page_input(29)
        user_prompt = build_artifact_compiler_user_prompt(
            canonical_record=make_canonical_record([29]),
            page_input=page_input,
            schema_dict=build_page_artifact_output_schema_dict(),
        )

        raw_output = client.generate_page_artifacts(
            system_prompt="system",
            user_prompt=user_prompt,
            schema_dict=build_page_artifact_output_schema_dict(),
        )
        valid_artifacts, issues = validate_page_artifact_output(
            raw_output,
            page_input["layout_blocks"],
        )

        self.assertEqual(issues, [])
        self.assertGreater(len(valid_artifacts), 0)
        self.assertTrue(all(artifact["validation_status"] == "anchored" for artifact in valid_artifacts))

    def test_real_client_disabled_by_default(self) -> None:
        client = RealArtifactCompilerClient(enable_real_api=False)

        with self.assertRaises(RuntimeError):
            client.generate_page_artifacts(
                system_prompt="system",
                user_prompt="{}",
                schema_dict={},
            )

    def test_compile_page_with_client_filters_invalid_artifacts(self) -> None:
        page_input = make_page_input(29)
        client = InvalidAnchorClient()

        result = compile_page_with_client(
            canonical_record=make_canonical_record([29]),
            page_input=page_input,
            client=client,
            schema_dict=build_page_artifact_output_schema_dict(),
            compiler_metadata={},
        )

        self.assertEqual(result["valid_artifacts"], [])
        self.assertIn(
            "source_anchor_not_found",
            {issue["error_type"] for issue in result["validation_issues"]},
        )

    def test_compiler_dry_run_writes_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "example_29.png").write_bytes(b"not-a-real-png")
            output_path = root / "artifact_store.json"
            canonical_record = make_canonical_record([29])
            canonical_record["gold_annotation"] = {"answer": "GOLD_SECRET"}
            canonical_record["baseline_outputs"] = {"mdocagent": {"answer": "BASELINE_SECRET"}}
            canonical_record["source_record"] = {"answer": "SOURCE_SECRET"}

            summary = run_stage2_compiler_dry_run(
                canonical_record=canonical_record,
                extract_path=root,
                output_path=output_path,
                client=FakeArtifactCompilerClient(),
            )
            store_text = output_path.read_text(encoding="utf-8")
            store = json.loads(store_text)

            self.assertTrue(output_path.exists())
            self.assertIn("29", store["artifact_index"]["by_page_index"])
            self.assertGreater(summary["num_valid_artifacts"], 0)
            self.assertNotIn("GOLD_SECRET", store_text)
            self.assertNotIn("BASELINE_SECRET", store_text)
            self.assertNotIn("SOURCE_SECRET", store_text)
            self.assertNotIn("gold_annotation", store_text)
            self.assertNotIn("baseline_outputs", store_text)
            self.assertNotIn("source_record", store_text)

    def test_forbidden_terms_not_in_runtime_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "example_29.png").write_bytes(b"not-a-real-png")
            output_path = root / "artifact_store.json"

            run_stage2_compiler_dry_run(
                canonical_record=make_canonical_record([29]),
                extract_path=root,
                output_path=output_path,
            )
            store_text = output_path.read_text(encoding="utf-8")

        for forbidden in ["proof_trace", "verified", "answer_supported", "proof_used"]:
            self.assertNotIn(forbidden, store_text)
            self.assertNotIn(forbidden, get_allowed_validation_statuses())


class InvalidAnchorClient(ArtifactCompilerClient):
    def generate_page_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = system_prompt
        _ = user_prompt
        _ = schema_dict
        return {
            "doc_id": "example.pdf",
            "page_index": 29,
            "artifacts": [
                {
                    "artifact_id": "example_p029_visual_observation_0001",
                    "doc_id": "example.pdf",
                    "page_index": 29,
                    "artifact_type": "visual_observation",
                    "modality": "visual",
                    "content": "Invalid anchor test artifact.",
                    "normalized_content": {"presence": "undetermined"},
                    "source_anchors": [
                        {
                            "source_id": "not_exist",
                            "anchor_type": "full_page_image",
                            "page_index": 29,
                            "bbox": None,
                        }
                    ],
                    "provenance": {
                        "op": "ATOM",
                        "sources": ["not_exist"],
                    },
                    "validation_status": "candidate",
                    "compiler_metadata": {},
                }
            ],
            "uncertain_or_unreadable": [],
        }


def make_canonical_record(pages_to_compile: List[int]) -> Dict[str, Any]:
    return {
        "document": {"doc_id": "example.pdf", "doc_type": "test", "dataset": None},
        "question": {"text": "What is visible on page 30?", "answer_format": "short_text"},
        "question_constraints": {
            "explicit_page_references": [
                {
                    "surface_text": "page 30",
                    "page_number_one_based": 30,
                    "page_index_zero_based": 29,
                    "source": "question_text",
                }
            ]
        },
        "candidate_pool": {
            "explicit_constraint_pages": [29],
            "retrieval_candidate_pages": [],
            "retrieval_missed_explicit_pages": [29],
        },
        "compilation_plan": {
            "compile_scope": "retrieval_union_plus_explicit_page_constraints",
            "pages_to_compile": pages_to_compile,
            "priority_pages": [29],
            "compilation_reasons": [
                {
                    "page_index": 29,
                    "reason_type": "explicit_page_reference",
                    "reason_text": "page 30",
                }
            ],
        },
    }


def make_page_input(page_index: int) -> Dict[str, Any]:
    return {
        "doc_id": "example.pdf",
        "page_index": page_index,
        "page_text": None,
        "page_text_path": None,
        "page_image_path": f"/tmp/example_{page_index}.png",
        "has_page_text": False,
        "has_page_image": True,
        "layout_blocks": [
            {
                "block_id": f"p{page_index:03d}_full_page_image",
                "block_type": "full_page_image",
                "page_index": page_index,
                "bbox": None,
                "text": None,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
