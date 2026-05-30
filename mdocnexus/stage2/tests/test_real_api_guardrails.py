"""Tests for Step 6 real API guardrails and audit logs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from mdocnexus.stage2.provider import ApiRunConfig, assert_real_api_allowed
from mdocnexus.stage2.artifact_pipeline import compile_page_with_client
from mdocnexus.stage2.provider import ArtifactCompilerClient
from mdocnexus.stage2.artifact_pipeline import run_stage2_single_page_real_api_smoke_test
from mdocnexus.stage2.logs import (
    RawCompilerOutputLogEntry,
    hash_raw_output,
    write_raw_output_log,
)
from mdocnexus.stage2.provider import RealApiArtifactCompilerClient
from mdocnexus.stage2.artifact_schema import build_page_artifact_output_schema_dict


class RealApiGuardrailsTest(unittest.TestCase):
    def test_real_api_disabled_by_default(self) -> None:
        client = RealApiArtifactCompilerClient(ApiRunConfig(enable_real_api=False))

        with self.assertRaises(RuntimeError):
            client.generate_page_artifacts("system", "user", {})

    def test_real_api_requires_single_page_per_call(self) -> None:
        config = ApiRunConfig(
            enable_real_api=True,
            model_name="dummy",
            max_pages_total=3,
            max_pages_per_call=2,
        )

        with self.assertRaises(RuntimeError):
            assert_real_api_allowed(config)

    def test_real_api_allows_finite_total_with_single_page_calls(self) -> None:
        config = ApiRunConfig(
            enable_real_api=True,
            model_name="dummy",
            max_pages_total=3,
            max_pages_per_call=1,
        )

        assert_real_api_allowed(config)

    def test_real_api_requires_model_name(self) -> None:
        config = ApiRunConfig(enable_real_api=True, model_name=None, max_pages=1)

        with self.assertRaises(RuntimeError):
            assert_real_api_allowed(config)

    def test_raw_output_log_serialization(self) -> None:
        raw_output = {"doc_id": "example.pdf", "page_index": 29, "artifacts": []}
        entry = RawCompilerOutputLogEntry(
            doc_id="example.pdf",
            page_index=29,
            provider="test_provider",
            model_name="test_model",
            compiler_version="test_compiler",
            prompt_version="test_prompt",
            raw_output=raw_output,
            raw_output_hash=hash_raw_output(raw_output),
            stage="stage2_compiler",
            created_at="2026-01-01T00:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "raw.jsonl"
            write_raw_output_log(log_path, entry)
            log_text = log_path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in log_text.splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertIn("raw_output_hash", rows[0])
        self.assertEqual(rows[0]["doc_id"], "example.pdf")
        self.assertEqual(rows[0]["page_index"], 29)
        self.assertEqual(rows[0]["stage"], "stage2_compiler")
        self.assertNotIn("gold_annotation", log_text)
        self.assertNotIn("baseline_outputs", log_text)
        self.assertNotIn("source_record", log_text)

    def test_compile_page_writes_raw_and_discard_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_log = root / "raw.jsonl"
            discard_log = root / "discard.jsonl"
            result = compile_page_with_client(
                canonical_record=make_canonical_record([29]),
                page_input=make_page_input(29),
                client=BadClient(),
                schema_dict=build_page_artifact_output_schema_dict(),
                compiler_metadata={"provider": "test", "model_name": "fake"},
                raw_output_log_path=raw_log,
                discard_log_path=discard_log,
            )
            raw_text = raw_log.read_text(encoding="utf-8")
            discard_text = discard_log.read_text(encoding="utf-8")

            self.assertEqual(result["valid_artifacts"], [])
            self.assertTrue(raw_log.exists())
            self.assertTrue(discard_log.exists())
            self.assertIn("source_anchor_not_found", discard_text)
            self.assertNotIn("gold_annotation", raw_text)
            self.assertNotIn("baseline_outputs", raw_text)
            self.assertNotIn("source_record", raw_text)

    def test_single_page_smoke_test_uses_only_one_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "example_29.png").write_bytes(b"not-a-real-png")
            (images_dir / "example_30.png").write_bytes(b"not-a-real-png")
            output_path = root / "artifact_store.json"
            config = ApiRunConfig(
                enable_real_api=True,
                provider="fake_real",
                model_name="fake-real-model",
                max_pages=1,
                raw_output_dir=root,
                discard_log_dir=root,
            )

            with patch(
                "mdocnexus.stage2.artifact_pipeline.RealApiArtifactCompilerClient",
                FakeRealApiArtifactCompilerClient,
            ):
                summary = run_stage2_single_page_real_api_smoke_test(
                    canonical_record=make_canonical_record([29, 30]),
                    extract_path=root,
                    output_path=output_path,
                    api_config=config,
                    target_page_index=29,
                    run_real_trial=True,
                )
            store = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["target_page_index"], 29)
            self.assertEqual(summary["num_pages_compiled"], 1)
            self.assertEqual([page["page_index"] for page in store["pages"]], [29])
            self.assertIn("29", store["artifact_index"]["by_page_index"])

    def test_forbidden_fields_never_enter_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "example_29.png").write_bytes(b"not-a-real-png")
            output_path = root / "artifact_store.json"
            config = ApiRunConfig(
                enable_real_api=True,
                provider="fake_real",
                model_name="fake-real-model",
                max_pages=1,
                raw_output_dir=root,
                discard_log_dir=root,
            )
            canonical_record = make_canonical_record([29])
            canonical_record["gold_annotation"] = {"answer": "GOLD_SECRET"}
            canonical_record["baseline_outputs"] = {"answer": "BASELINE_SECRET"}
            canonical_record["source_record"] = {"answer": "SOURCE_SECRET"}

            with patch(
                "mdocnexus.stage2.artifact_pipeline.RealApiArtifactCompilerClient",
                FakeRealApiArtifactCompilerClient,
            ):
                run_stage2_single_page_real_api_smoke_test(
                    canonical_record=canonical_record,
                    extract_path=root,
                    output_path=output_path,
                    api_config=config,
                    target_page_index=29,
                    run_real_trial=True,
                )
            store_text = output_path.read_text(encoding="utf-8")

        for forbidden in [
            "gold_annotation",
            "baseline_outputs",
            "source_record",
            "proof_trace",
            "verified",
            "answer_supported",
            "proof_used",
            "GOLD_SECRET",
            "BASELINE_SECRET",
            "SOURCE_SECRET",
        ]:
            self.assertNotIn(forbidden, store_text)


class BadClient(ArtifactCompilerClient):
    def generate_page_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = system_prompt
        _ = user_prompt
        _ = schema_dict
        return make_bad_output()


class FakeRealApiArtifactCompilerClient(ArtifactCompilerClient):
    def __init__(self, api_config: ApiRunConfig) -> None:
        self.api_config = api_config

    def generate_page_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = system_prompt
        _ = schema_dict
        prompt_payload = json.loads(user_prompt)
        page_index = prompt_payload["document"]["page_index"]
        source_id = f"p{page_index:03d}_full_page_image"
        return {
            "doc_id": prompt_payload["document"]["doc_id"],
            "page_index": page_index,
            "artifacts": [
                {
                    "artifact_id": f"example_p{page_index:03d}_visual_observation_0001",
                    "doc_id": prompt_payload["document"]["doc_id"],
                    "page_index": page_index,
                    "artifact_type": "visual_observation",
                    "modality": "image",
                    "content": "Fake real API visual observation.",
                    "normalized_content": {"presence": "undetermined"},
                    "source_anchors": [
                        {
                            "source_id": source_id,
                            "anchor_type": "full_page_image",
                            "page_index": page_index,
                            "bbox": None,
                        }
                    ],
                    "provenance": {"op": "ATOM", "sources": [source_id]},
                    "validation_status": "candidate",
                    "compiler_metadata": {},
                }
            ],
            "uncertain_or_unreadable": [],
        }


def make_bad_output() -> Dict[str, Any]:
    return {
        "doc_id": "example.pdf",
        "page_index": 29,
        "artifacts": [
            {
                "artifact_id": "example_p029_visual_observation_0001",
                "doc_id": "example.pdf",
                "page_index": 29,
                "artifact_type": "visual_observation",
                "modality": "image",
                "content": "Invalid anchor artifact.",
                "normalized_content": {"presence": "undetermined"},
                "source_anchors": [
                    {
                        "source_id": "not_exist",
                        "anchor_type": "full_page_image",
                        "page_index": 29,
                        "bbox": None,
                    }
                ],
                "provenance": {"op": "ATOM", "sources": ["not_exist"]},
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
        "compilation_plan": {"pages_to_compile": pages_to_compile},
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
