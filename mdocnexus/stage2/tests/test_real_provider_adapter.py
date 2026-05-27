"""Tests for Step 7 generic real-provider adapter guardrails."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from mdocnexus.stage2.provider import ApiRunConfig
from mdocnexus.stage2.artifact_pipeline import run_stage2_single_page_real_api_smoke_test
from mdocnexus.stage2.provider import CompatibleChatJsonProvider
from mdocnexus.stage2.provider import ProviderNotConfiguredError, ProviderResponseFormatError
from mdocnexus.stage2.provider import RealApiArtifactCompilerClient
from scripts.run_stage2_real_single_page_trial import validate_real_trial_args


class RealProviderAdapterTest(unittest.TestCase):
    def test_provider_requires_explicit_run_real_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_page_image(root, 29)
            config = make_api_config(root)

            with self.assertRaises(RuntimeError):
                run_stage2_single_page_real_api_smoke_test(
                    canonical_record=make_canonical_record([29]),
                    extract_path=root,
                    output_path=root / "store.json",
                    api_config=config,
                    target_page_index=29,
                    run_real_trial=False,
                )

    def test_provider_configuration_missing_handled(self) -> None:
        config = ApiRunConfig(
            enable_real_api=True,
            provider="siliconflow",
            model_name="dummy-model",
            max_pages=1,
            api_key_env_var="STAGE2_TEST_MISSING_API_KEY",
        )
        old_value = os.environ.pop("STAGE2_TEST_MISSING_API_KEY", None)
        try:
            with self.assertRaises(ProviderNotConfiguredError):
                CompatibleChatJsonProvider(config).generate_json("system", "user", {})
        finally:
            if old_value is not None:
                os.environ["STAGE2_TEST_MISSING_API_KEY"] = old_value

    def test_provider_not_implemented_fails_safely(self) -> None:
        config = ApiRunConfig(
            enable_real_api=True,
            provider="unknown",
            model_name="dummy-model",
            max_pages=1,
        )
        client = RealApiArtifactCompilerClient(config)

        with self.assertRaises(ProviderNotConfiguredError):
            client.generate_page_artifacts("system", "user", {})

    def test_provider_non_json_response_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_page_image(root, 29)
            output_path = root / "store.json"
            config = make_api_config(root)

            with patch(
                "mdocnexus.stage2.provider.CompatibleChatJsonProvider",
                NonJsonProvider,
            ):
                summary = run_stage2_single_page_real_api_smoke_test(
                    canonical_record=make_canonical_record([29]),
                    extract_path=root,
                    output_path=output_path,
                    api_config=config,
                    target_page_index=29,
                    run_real_trial=True,
                )

            raw_log_text = Path(summary["raw_output_log_path"]).read_text(encoding="utf-8")
            store = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertFalse(summary["quality_gate"]["single_page_smoke_test_passed"])
        self.assertIn("provider_error", summary["quality_gate"]["blocking_reasons"])
        self.assertIn("ProviderResponseFormatError", raw_log_text)
        self.assertEqual(store["pages"][0]["artifacts"], [])

    def test_provider_valid_json_goes_through_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_page_image(root, 29)
            output_path = root / "store.json"
            config = make_api_config(root)

            with patch(
                "mdocnexus.stage2.provider.CompatibleChatJsonProvider",
                ValidJsonProvider,
            ):
                summary = run_stage2_single_page_real_api_smoke_test(
                    canonical_record=make_canonical_record([29]),
                    extract_path=root,
                    output_path=output_path,
                    api_config=config,
                    target_page_index=29,
                    run_real_trial=True,
                )
            store = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertTrue(summary["quality_gate"]["single_page_smoke_test_passed"])
        self.assertIn("29", store["artifact_index"]["by_page_index"])
        self.assertEqual(store["pages"][0]["artifacts"][0]["validation_status"], "anchored")

    def test_provider_invalid_anchor_goes_to_discard_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_page_image(root, 29)
            output_path = root / "store.json"
            config = make_api_config(root)

            with patch(
                "mdocnexus.stage2.provider.CompatibleChatJsonProvider",
                InvalidAnchorProvider,
            ):
                summary = run_stage2_single_page_real_api_smoke_test(
                    canonical_record=make_canonical_record([29]),
                    extract_path=root,
                    output_path=output_path,
                    api_config=config,
                    target_page_index=29,
                    run_real_trial=True,
                )
            store = json.loads(output_path.read_text(encoding="utf-8"))
            discard_text = Path(summary["discard_log_path"]).read_text(encoding="utf-8")

        self.assertFalse(summary["quality_gate"]["single_page_smoke_test_passed"])
        self.assertEqual(store["pages"][0]["artifacts"], [])
        self.assertIn("source_anchor_not_found", discard_text)

    def test_script_guardrail_rejects_missing_flags(self) -> None:
        args = argparse.Namespace(enable_real_api=False, run_real_trial=True, target_page_index=29)
        with self.assertRaises(RuntimeError):
            validate_real_trial_args(args)

        args = argparse.Namespace(enable_real_api=True, run_real_trial=False, target_page_index=29)
        with self.assertRaises(RuntimeError):
            validate_real_trial_args(args)

    def test_forbidden_fields_never_enter_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_page_image(root, 29)
            output_path = root / "store.json"
            config = make_api_config(root)
            canonical_record = make_canonical_record([29])
            canonical_record["gold_annotation"] = {"answer": "GOLD_SECRET"}
            canonical_record["baseline_outputs"] = {"answer": "BASELINE_SECRET"}
            canonical_record["source_record"] = {"answer": "SOURCE_SECRET"}

            with patch(
                "mdocnexus.stage2.provider.CompatibleChatJsonProvider",
                ValidJsonProvider,
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


class NonJsonProvider:
    def __init__(self, api_config: ApiRunConfig) -> None:
        self.api_config = api_config

    def generate_json(self, system_prompt: str, user_prompt: str, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
        _ = system_prompt
        _ = user_prompt
        _ = schema_dict
        raise ProviderResponseFormatError("Provider response was not valid JSON.", raw_text="not json")


class ValidJsonProvider:
    def __init__(self, api_config: ApiRunConfig) -> None:
        self.api_config = api_config

    def generate_json(self, system_prompt: str, user_prompt: str, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
        _ = system_prompt
        _ = schema_dict
        return make_provider_output(user_prompt, source_id="p029_full_page_image")


class InvalidAnchorProvider:
    def __init__(self, api_config: ApiRunConfig) -> None:
        self.api_config = api_config

    def generate_json(self, system_prompt: str, user_prompt: str, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
        _ = system_prompt
        _ = schema_dict
        return make_provider_output(user_prompt, source_id="not_exist")


def make_provider_output(user_prompt: str, source_id: str) -> Dict[str, Any]:
    prompt_payload = json.loads(user_prompt)
    doc_id = prompt_payload["document"]["doc_id"]
    page_index = int(prompt_payload["document"]["page_index"])
    return {
        "doc_id": doc_id,
        "page_index": page_index,
        "artifacts": [
            {
                "artifact_id": f"example_p{page_index:03d}_visual_observation_0001",
                "doc_id": doc_id,
                "page_index": page_index,
                "artifact_type": "visual_observation",
                "modality": "visual",
                "content": "Provider visual observation candidate.",
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
                "compiler_metadata": {"compiler_name": "provider_test"},
            }
        ],
        "uncertain_or_unreadable": [],
    }


def make_api_config(root: Path) -> ApiRunConfig:
    return ApiRunConfig(
        enable_real_api=True,
        provider="siliconflow",
        model_name="dummy-model",
        max_pages=1,
        raw_output_dir=root,
        discard_log_dir=root,
    )


def make_canonical_record(pages_to_compile: list[int]) -> Dict[str, Any]:
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


def create_page_image(root: Path, page_index: int) -> None:
    images_dir = root / "images"
    images_dir.mkdir(exist_ok=True)
    (images_dir / f"example_{page_index}.png").write_bytes(b"not-a-real-png")


if __name__ == "__main__":
    unittest.main()
