"""Tests for Stage 2 small-batch artifact compilation."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.selectors import select_pages_for_small_batch
from mdocnexus.stage2.reports import summarize_batch_results, write_batch_summary
from mdocnexus.stage2.provider import ArtifactCompilerClient
from scripts.stage2_compile_small_batch import run_small_batch, validate_args


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
) -> argparse.Namespace:
    return argparse.Namespace(
        stage2_json=str(stage2_json),
        config=str(config),
        extract_root=str(extract_root),
        output_dir=str(output_dir),
        max_pages=max_pages,
        provider=provider,
        model_name=model_name,
        enable_real_api=enable_real_api,
        run_real_trial=run_real_trial,
        dry_run_fake_client=dry_run_fake_client,
        deterministic_dedup_enabled=deterministic_dedup_enabled,
        timeout_seconds=120,
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
