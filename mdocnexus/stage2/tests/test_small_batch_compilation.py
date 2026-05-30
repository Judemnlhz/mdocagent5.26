"""Tests for Stage 2 small-batch artifact compilation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.selectors import select_pages_for_small_batch
from mdocnexus.stage2.reports import summarize_batch_results, write_batch_summary
from mdocnexus.stage2.provider import ArtifactCompilerClient
from scripts.stage2 import document_generic_candidate_page_indices, run_doc_compile_command, run_small_batch, validate_small_batch_args as validate_args


class SmallBatchCompilationTest(unittest.TestCase):
    def test_selector_does_not_use_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_b", [1])
            records = [
                make_stage2_record(
                    "doc_a.pdf",
                    "What is on page 30?",
                    answer="GOLD_A",
                    binary_correctness=True,
                    explicit_valid=[],
                    invalid_explicit=[29],
                    image_pages=[],
                    pages_to_compile=[],
                ),
                make_stage2_record("doc_b.pdf", "What is on page 2?", answer="GOLD_B", explicit_valid=[1]),
            ]

            selected = select_pages_for_small_batch(records, max_pages=5, extract_root=extract_root)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["doc_id"], "doc_b.pdf")
        serialized = json.dumps(selected, ensure_ascii=False)
        self.assertNotIn("GOLD_A", serialized)
        self.assertNotIn("GOLD_B", serialized)
        self.assertNotIn("binary_correctness", serialized)
        self.assertNotIn("evidence_pages", serialized)

    def test_selector_requires_image_and_layout_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0], include_image=False)
            create_extract_pages(extract_root, "doc_c", [0])
            records = [
                make_stage2_record("doc_a.pdf", "q", image_pages=[0], has_image=False),
                make_stage2_record("doc_b.pdf", "q", image_pages=[0], has_image=True, layout_block_ids=[]),
                make_stage2_record("doc_c.pdf", "q", image_pages=[0], has_image=True),
            ]

            selected = select_pages_for_small_batch(records, max_pages=5, extract_root=extract_root)

        self.assertEqual([item["doc_id"] for item in selected], ["doc_c.pdf"])

    def test_out_of_range_explicit_page_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            records = [
                make_stage2_record(
                    "doc_a.pdf",
                    "What is on page 30?",
                    explicit_valid=[],
                    invalid_explicit=[29],
                    pages_to_compile=[],
                    page_sources=[],
                )
            ]

            selected = select_pages_for_small_batch(records, max_pages=5, extract_root=extract_root)

        self.assertEqual(selected, [])

    def test_max_pages_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            records = [make_stage2_record(f"doc_{index}.pdf", "q") for index in range(5)]
            for index in range(5):
                create_extract_pages(extract_root, f"doc_{index}", [0])

            selected = select_pages_for_small_batch(records, max_pages=2, extract_root=extract_root)

        self.assertEqual(len(selected), 2)
        self.assertEqual([item["record_index"] for item in selected], [0, 1])

    def test_script_rejects_without_enable_real_api(self) -> None:
        args = make_args(enable_real_api=False, run_real_trial=True)

        with self.assertRaises(RuntimeError):
            validate_args(args)

    def test_script_rejects_without_run_real_trial(self) -> None:
        args = make_args(enable_real_api=True, run_real_trial=False)

        with self.assertRaises(RuntimeError):
            validate_args(args)

    def test_batch_summary_omits_api_key_and_gold_fields(self) -> None:
        page_results = [
            {
                "api_called": True,
                "num_raw_artifacts": 2,
                "num_valid_artifacts": 1,
                "num_validation_issues": 1,
                "artifact_store_path": "store.json",
                "forbidden_field_violations": 0,
                "provider": "siliconflow",
                "model_name": "Qwen/Qwen3-VL-8B-Instruct",
                "max_pages": 5,
            }
        ]
        summary = summarize_batch_results(page_results)

        keys = collect_keys(summary)
        for forbidden in ["api_key", "answer", "evidence_pages", "binary_correctness"]:
            self.assertNotIn(forbidden, keys)
        self.assertEqual(summary["num_api_calls"], 1)
        self.assertEqual(summary["discard_rate"], 0.5)
        self.assertIn("deterministic_dedup_enabled", summary)

    def test_write_batch_summary_rejects_forbidden_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_batch_summary({"api_key": "SECRET"}, Path(tmpdir) / "summary.json")

    def test_fake_client_dry_run_writes_quality_csv_and_clean_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "outputs"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "What is on page 1?")]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\napi_key: SECRET_SHOULD_NOT_LEAK\n", encoding="utf-8")

            result = run_small_batch(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                )
            )
            quality_path = output_dir / "reports" / "batch_quality.csv"
            with quality_path.open(encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            store_path = Path(result["page_results"][0]["artifact_store_path"])
            store_text = store_path.read_text(encoding="utf-8")
            quality_exists = quality_path.is_file()

        self.assertTrue(quality_exists)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["doc_id"], "doc_a.pdf")
        for forbidden in ["proof_trace", "verified", "answer_supported", "proof_used", "api_key"]:
            self.assertNotIn(forbidden, store_text)
        self.assertEqual(result["summary"]["num_pages_attempted"], 1)
        self.assertEqual(result["summary"]["num_api_calls"], 0)
        self.assertTrue(result["summary"]["deterministic_dedup_enabled"])
        self.assertEqual(result["summary"]["num_deduplicated_artifacts"], 0)

    def test_dedup_keeps_raw_log_original_and_store_only_valid_unique_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "outputs"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "What is on page 1?")]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            result = run_small_batch(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                ),
                client=DuplicateArtifactClient(),
            )
            raw_entry = json.loads((output_dir / "raw_outputs" / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
            discard_entries = [
                json.loads(line)
                for line in (output_dir / "discard" / "discard.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            store = json.loads(Path(result["page_results"][0]["artifact_store_path"]).read_text(encoding="utf-8"))

        self.assertEqual(len(raw_entry["raw_output"]["artifacts"]), 2)
        self.assertEqual([entry["error_type"] for entry in discard_entries], ["duplicate_artifact_deduplicated"])
        self.assertEqual(len(store["pages"][0]["artifacts"]), 1)
        self.assertEqual(result["summary"]["num_raw_artifacts_before_dedup"], 2)
        self.assertEqual(result["summary"]["num_deduplicated_artifacts"], 1)
        self.assertEqual(result["summary"]["schema_valid_rate_before_dedup"], 0.5)
        self.assertEqual(result["summary"]["schema_valid_rate_after_dedup"], 1.0)

    def test_disable_dedup_reproduces_duplicate_artifact_validation_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "outputs"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "What is on page 1?")]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            result = run_small_batch(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    deterministic_dedup_enabled=False,
                ),
                client=DuplicateArtifactClient(),
            )
            discard_entries = [
                json.loads(line)
                for line in (output_dir / "discard" / "discard.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertFalse(result["summary"]["deterministic_dedup_enabled"])
        self.assertEqual(result["summary"]["num_deduplicated_artifacts"], 0)
        self.assertIn("duplicate_artifact", [entry["error_type"] for entry in discard_entries])

    def test_doc_compile_writes_document_generic_outputs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "stage2_doc"
            question = "QUESTION_TEXT_MUST_NOT_CONDITION_DOC_COMPILE"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", question)]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            result = run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                )
            )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            raw_text = (output_dir / "artifacts.jsonl").read_text(encoding="utf-8")

        self.assertTrue(result["manifest"]["document_generic"])
        self.assertTrue(manifest["document_generic"])
        self.assertNotIn(question, raw_text)
        self.assertEqual(len(manifest["artifacts_hash"]), 64)

    def test_no_page_29_special_case_in_stage2_core(self) -> None:
        source = Path("mdocnexus/stage2/artifact_pipeline.py").read_text(encoding="utf-8")

        self.assertNotIn("int(explicit_page) == 29", source)

    def test_document_generic_candidate_pages_ignore_question_conditioned_routes(self) -> None:
        record = make_stage2_record("doc_a.pdf", "What is on page 30?", pages_to_compile=None)
        record["stage2"]["candidate_page_routes"] = [{"page_index": 29, "routes": ["image"]}]

        self.assertEqual(document_generic_candidate_page_indices(record), [0])

    def test_doc_compile_ignores_question_conditioned_routes_for_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "stage2_doc"
            record = make_stage2_record("doc_a.pdf", "What is on page 30?")
            record["stage2"]["candidate_page_routes"] = [{"page_index": 0, "routes": ["image"]}]
            stage2_json.write_text(json.dumps([record]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            result = run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    max_pages=1,
                )
            )

        self.assertEqual(result["summary"]["artifact_type_counts"], {"text_span": 1, "visual_observation": 1})

    def test_doc_compile_image_route_records_payload_audit_without_public_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "stage2_doc"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "q")]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            result = run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    provider="fake",
                    image_payload_mode="base64",
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    max_pages=1,
                )
            )
            call_log = [
                json.loads(line)
                for line in (output_dir / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            leakage_violations = collect_public_leakage_marker_violations(
                output_dir,
                extra_forbidden=[str(extract_root)],
            )

        self.assertEqual(result["summary"]["pages_with_image_payload"], 1)
        self.assertEqual(call_log[0]["modality_route"], "mixed")
        self.assertTrue(call_log[0]["image_payload_sent"])
        self.assertEqual(call_log[0]["image_payload_mode"], "base64")
        self.assertEqual(len(call_log[0]["image_sha256"]), 64)
        self.assertEqual(manifest["provider_modes"], ["fake"])
        self.assertEqual(manifest["pages_with_image_payload"], 1)
        self.assertEqual(manifest["image_sha256_values"], [call_log[0]["image_sha256"]])
        self.assertFalse(manifest["public_provider_outputs_written"])
        self.assertFalse(manifest["private_debug_enabled"])
        self.assertFalse(manifest["private_debug_dir_recorded_as_public"])
        self.assertTrue(manifest["provider_body_redacted"])
        self.assertTrue(manifest["encoded_payload_redacted"])
        self.assertTrue(manifest["filesystem_locations_redacted"])
        self.assertTrue(manifest["credentials_redacted"])
        self.assertEqual([], leakage_violations)

    def test_doc_compile_text_route_records_no_image_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0], include_image=False)
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "stage2_doc"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "q", has_image=False)]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            result = run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    provider="fake",
                    image_payload_mode="image_url",
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    max_pages=1,
                )
            )
            call_log = [
                json.loads(line)
                for line in (output_dir / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result["summary"]["pages_without_image_payload"], 1)
        self.assertEqual(call_log[0]["modality_route"], "text")
        self.assertFalse(call_log[0]["image_payload_sent"])
        self.assertIsNone(call_log[0]["image_sha256"])

    def test_doc_compile_image_payload_mode_none_never_sends_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "stage2_doc"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "q")]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            result = run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    provider="fake",
                    image_payload_mode="none",
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    max_pages=1,
                )
            )
            call_log = read_jsonl(output_dir / "call_log.jsonl")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(result["summary"]["pages_with_image_payload"], 0)
        self.assertEqual(result["summary"]["pages_without_image_payload"], 1)
        self.assertFalse(call_log[0]["image_payload_sent"])
        self.assertEqual(call_log[0]["image_payload_mode"], "none")
        self.assertIsNone(call_log[0]["image_sha256"])
        self.assertEqual(call_log[0]["image_sha256_unavailable_reason"], "image_payload_mode_none")
        self.assertEqual(manifest["image_payload_modes"], ["none"])

    def test_doc_compile_public_outputs_exclude_private_payload_and_secret_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "stage2_doc"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "q")]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\napi_key: SECRET_SHOULD_NOT_LEAK\n", encoding="utf-8")

            run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=output_dir,
                    provider="fake",
                    image_payload_mode="base64",
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    max_pages=1,
                )
            )
            leakage_violations = collect_public_leakage_marker_violations(output_dir)

        self.assertEqual([], leakage_violations)

    def test_doc_compile_real_provider_requires_explicit_double_opt_in(self) -> None:
        for enable_real_api, run_real_trial in ((False, False), (False, True), (True, False)):
            with self.subTest(enable_real_api=enable_real_api, run_real_trial=run_real_trial):
                with self.assertRaises(RuntimeError):
                    run_doc_compile_command(
                        make_args(
                            provider="real",
                            enable_real_api=enable_real_api,
                            run_real_trial=run_real_trial,
                            max_pages=1,
                        )
                    )

    def test_doc_compile_real_provider_requires_finite_page_limit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "finite --max-pages"):
            run_doc_compile_command(
                make_args(
                    provider="real",
                    enable_real_api=True,
                    run_real_trial=True,
                    max_pages=None,
                )
            )

    def test_audit_real_provider_smoke_exits_zero_on_safe_fake_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_safe_fake_doc_compile_output(Path(tmpdir))
            completed = run_audit_script(output_dir)

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["call_log_rows"], 1)
        self.assertEqual(report["provider_modes"], ["fake"])

    def test_audit_real_provider_smoke_exits_one_on_public_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_safe_fake_doc_compile_output(Path(tmpdir))
            with (output_dir / "artifacts.jsonl").open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps({"api_key": "SECRET_SHOULD_FAIL"}))
                file_obj.write("\n")
            completed = run_audit_script(output_dir)

        self.assertEqual(completed.returncode, 1)
        self.assertIn("public_leakage_violations", completed.stdout)
        self.assertIn("api_key", completed.stdout)

    def test_audit_real_provider_smoke_exits_one_on_missing_call_log_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_safe_fake_doc_compile_output(Path(tmpdir))
            rows = read_jsonl(output_dir / "call_log.jsonl")
            rows[0].pop("call_id", None)
            write_jsonl(output_dir / "call_log.jsonl", rows)
            completed = run_audit_script(output_dir)

        self.assertEqual(completed.returncode, 1)
        self.assertIn("call_log[1].call_id", completed.stdout)

    def test_doc_compile_raw_output_only_private_when_explicitly_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            first_output_dir = root / "stage2_doc_first"
            second_output_dir = root / "stage2_doc_second"
            private_debug_dir = root / "outputs_private" / "stage2_debug"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "q")]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")

            run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=first_output_dir,
                    provider="fake",
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    max_pages=1,
                )
            )
            run_doc_compile_command(
                make_args(
                    stage2_json=stage2_json,
                    config=config_path,
                    extract_root=extract_root,
                    output_dir=second_output_dir,
                    provider="fake",
                    dry_run_fake_client=True,
                    enable_real_api=False,
                    run_real_trial=False,
                    max_pages=1,
                    save_private_debug=True,
                    private_debug_dir=private_debug_dir,
                )
            )
            first_public_raw_exists = (first_output_dir / "raw_outputs.jsonl").exists()
            second_public_raw_exists = (second_output_dir / "raw_outputs.jsonl").exists()
            private_raw_path = private_debug_dir / "raw_outputs.jsonl"
            private_raw_exists = private_raw_path.is_file()
            private_raw_size = private_raw_path.stat().st_size if private_raw_exists else 0
            second_manifest = json.loads((second_output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(first_public_raw_exists)
        self.assertFalse(second_public_raw_exists)
        self.assertTrue(private_raw_exists)
        self.assertGreater(private_raw_size, 0)
        self.assertTrue(second_manifest["private_debug_enabled"])
        self.assertFalse(second_manifest["private_debug_dir_recorded_as_public"])
        self.assertFalse(second_manifest["public_provider_outputs_written"])


def make_stage2_record(
    doc_id: str,
    question: str,
    answer: str = "GOLD_SECRET",
    binary_correctness: bool = False,
    explicit_valid: List[int] | None = None,
    invalid_explicit: List[int] | None = None,
    image_pages: List[int] | None = None,
    pages_to_compile: List[int] | None = None,
    page_sources: List[Dict[str, Any]] | None = None,
    has_image: bool = True,
    layout_block_ids: List[str] | None = None,
) -> Dict[str, Any]:
    explicit_valid = [0] if explicit_valid is None else explicit_valid
    invalid_explicit = [] if invalid_explicit is None else invalid_explicit
    pages_to_compile = explicit_valid if pages_to_compile is None and explicit_valid else pages_to_compile
    pages_to_compile = [0] if pages_to_compile is None else pages_to_compile
    image_pages = pages_to_compile if image_pages is None else image_pages
    layout_block_ids = ["p000_full_page_image"] if layout_block_ids is None else layout_block_ids
    route_pages = sorted({int(page_index) for page_index in pages_to_compile} | {int(page_index) for page_index in image_pages})
    candidate_page_routes = [
        {
            "page_index": page_index,
            "routes": [route for route in ("text", "image") if route == "text" or page_index in image_pages],
        }
        for page_index in route_pages
    ]
    return {
        "doc_id": doc_id,
        "question": question,
        "answer": answer,
        "evidence_pages": "[99]",
        "binary_correctness": binary_correctness,
        "answer_format": "Str",
        "stage2": {
            "preflight": {"passed": True, "blocking_reasons": []},
            "candidate_page_routes": candidate_page_routes,
        },
    }


def create_extract_pages(extract_root: Path, doc_stem: str, page_indices: List[int], include_image: bool = True) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    for page_index in page_indices:
        if include_image:
            (extract_root / f"{doc_stem}_{page_index}.png").write_bytes(b"not-a-real-png")
        (extract_root / f"{doc_stem}_{page_index}.txt").write_text(
            f"text for page {page_index}",
            encoding="utf-8",
        )


def make_args(
    stage2_json: Path | str = "stage2.json",
    config: Path | str = "config.yaml",
    extract_root: Path | str = "tmp/MMLongBench",
    output_dir: Path | str = "outputs",
    max_pages: int = 5,
    provider: str = "siliconflow",
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    enable_real_api: bool = True,
    run_real_trial: bool = True,
    dry_run_fake_client: bool = False,
    deterministic_dedup_enabled: bool = True,
    image_payload_mode: str = "image_url",
    save_private_debug: bool = False,
    private_debug_dir: Path | str = "outputs_private/stage2_debug/",
    max_pages_real: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        stage2_json=str(stage2_json),
        config=str(config),
        extract_root=str(extract_root),
        output_dir=str(output_dir),
        max_pages=max_pages,
        max_pages_real=max_pages_real,
        provider=provider,
        model_name=model_name,
        enable_real_api=enable_real_api,
        run_real_trial=run_real_trial,
        dry_run_fake_client=dry_run_fake_client,
        deterministic_dedup_enabled=deterministic_dedup_enabled,
        image_payload_mode=image_payload_mode,
        save_private_debug=save_private_debug,
        private_debug_dir=str(private_debug_dir),
        timeout_seconds=120,
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_public_stage2_doc_outputs(output_dir: Path) -> str:
    texts: List[str] = []
    for name in ("artifacts.jsonl", "discard.jsonl", "call_log.jsonl", "quality_report.json", "manifest.json"):
        path = output_dir / name
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
    return "\n".join(texts)


SAFE_PUBLIC_LEAKAGE_DECLARATION_KEYS = {
    "credentials_redacted",
    "provider_body_redacted",
    "encoded_payload_redacted",
    "filesystem_locations_redacted",
    "public_provider_outputs_written",
}
PUBLIC_FORBIDDEN_MARKERS = [
    "raw_response",
    "data:image",
    "base64,",
    "file://",
    "/home/",
    "image_path",
    "api_key",
    "secret",
]


def collect_public_leakage_marker_violations(output_dir: Path, extra_forbidden: List[str] | None = None) -> List[str]:
    markers = list(PUBLIC_FORBIDDEN_MARKERS)
    markers.extend(extra_forbidden or [])
    violations: List[str] = []
    for name in ("artifacts.jsonl", "discard.jsonl", "call_log.jsonl", "quality_report.json", "manifest.json"):
        path = output_dir / name
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            for index, row in enumerate(read_jsonl(path), start=1):
                collect_marker_violations(row, f"{name}[{index}]", markers, violations)
        else:
            collect_marker_violations(json.loads(path.read_text(encoding="utf-8")), name, markers, violations)
    return violations


def collect_marker_violations(value: Any, field_path: str, markers: List[str], violations: List[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_lower not in SAFE_PUBLIC_LEAKAGE_DECLARATION_KEYS:
                for marker in markers:
                    if marker.lower() in key_lower:
                        violations.append(f"{field_path}.{key_text}:{marker}")
            collect_marker_violations(child, f"{field_path}.{key_text}", markers, violations)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect_marker_violations(child, f"{field_path}[{index}]", markers, violations)
    elif isinstance(value, str):
        for marker in markers:
            if marker and marker.lower() in value.lower():
                violations.append(f"{field_path}:{marker}")


def create_safe_fake_doc_compile_output(root: Path) -> Path:
    extract_root = root / "tmp" / "MMLongBench"
    create_extract_pages(extract_root, "doc_a", [0])
    stage2_json = root / "stage2.json"
    config_path = root / "qwen3vl.yaml"
    output_dir = root / "stage2_doc"
    stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", "q")]), encoding="utf-8")
    config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")
    run_doc_compile_command(
        make_args(
            stage2_json=stage2_json,
            config=config_path,
            extract_root=extract_root,
            output_dir=output_dir,
            provider="fake",
            image_payload_mode="base64",
            dry_run_fake_client=True,
            enable_real_api=False,
            run_real_trial=False,
            max_pages=1,
        )
    )
    return output_dir


def run_audit_script(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/audit_real_provider_smoke.py", "--output-dir", str(output_dir)],
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


class DuplicateArtifactClient(ArtifactCompilerClient):
    def generate_page_artifacts(self, system_prompt: str, user_prompt: str, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
        _ = system_prompt
        _ = schema_dict
        prompt_payload = json.loads(user_prompt)
        doc_id = prompt_payload["document"]["doc_id"]
        page_index = int(prompt_payload["document"]["page_index"])
        source_id = f"p{page_index:03d}_text_0000"
        artifact = {
            "doc_id": doc_id,
            "page_index": page_index,
            "artifact_type": "text_span",
            "modality": "text",
            "content": "same content",
            "normalized_content": {"text": "same content"},
            "source_anchors": [
                {"source_id": source_id, "anchor_type": "text_block", "page_index": page_index, "bbox": None}
            ],
            "provenance": {"op": "ATOM", "sources": [source_id]},
            "validation_status": "candidate",
            "compiler_metadata": {},
        }
        first = dict(artifact)
        first["artifact_id"] = "artifact_001"
        second = dict(artifact)
        second["artifact_id"] = "artifact_002"
        return {"doc_id": doc_id, "page_index": page_index, "artifacts": [first, second]}


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value.keys())
        for child in value.values():
            keys.update(collect_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(collect_keys(child))
        return keys
    return set()


if __name__ == "__main__":
    unittest.main()
