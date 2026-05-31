"""Tests for Stage 2 cross-document controlled batch compilation."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.selectors import select_crossdoc_pages_for_batch
from scripts.stage2 import run_crossdoc_batch, validate_crossdoc_args as validate_args


class CrossDocBatchCompilationTest(unittest.TestCase):
    def test_selector_respects_doc_page_and_total_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            records = [make_stage2_record(f"doc_{doc}.pdf", pages=[0, 1, 2]) for doc in range(8)]
            for doc in range(8):
                create_extract_pages(extract_root, f"doc_{doc}", [0, 1, 2])

            selected = select_crossdoc_pages_for_batch(
                records,
                max_docs=5,
                max_pages_per_doc=2,
                max_pages=10,
                extract_root=extract_root,
            )

        doc_ids = {item["doc_id"] for item in selected}
        self.assertLessEqual(len(doc_ids), 5)
        self.assertLessEqual(len(selected), 10)
        for doc_id in doc_ids:
            self.assertLessEqual(sum(1 for item in selected if item["doc_id"] == doc_id), 2)

    def test_selector_does_not_emit_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0])
            records = [make_stage2_record("doc_a.pdf", pages=[0], answer="SECRET", binary_correctness=True)]

            selected = select_crossdoc_pages_for_batch(records, extract_root=extract_root)
        serialized = json.dumps(selected, ensure_ascii=False)

        self.assertEqual(len(selected), 1)
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("answer", serialized)
        self.assertNotIn("evidence_pages", serialized)
        self.assertNotIn("binary_correctness", serialized)

    def test_selector_skips_out_of_range_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0, 1])
            records = [make_stage2_record("doc_a.pdf", pages=[99], page_count=2)]

            selected = select_crossdoc_pages_for_batch(records, extract_root=extract_root)

        self.assertEqual(selected, [])

    def test_selector_supports_structured_page_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0, 1, 2])
            records = [
                make_stage2_record(
                    "doc_a.pdf",
                    pages=[1],
                    page_count={"value": 3, "available_page_indices": [0, 1, 2]},
                )
            ]

            selected = select_crossdoc_pages_for_batch(records, extract_root=extract_root)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["page_index"], 1)

    def test_selector_requires_image_and_layout_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0], include_image=False)
            create_extract_pages(extract_root, "doc_c", [0])
            records = [
                make_stage2_record("doc_a.pdf", pages=[0], has_image=False),
                make_stage2_record("doc_b.pdf", pages=[0], layout_block_ids=[]),
                make_stage2_record("doc_c.pdf", pages=[0], layout_block_ids=["p000_full_page_image"]),
            ]

            selected = select_crossdoc_pages_for_batch(records, extract_root=extract_root)

        self.assertEqual([item["doc_id"] for item in selected], ["doc_c.pdf"])

    def test_script_rejects_without_enable_real_api(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_args(make_args(enable_real_api=False, run_real_trial=True))

    def test_script_rejects_without_run_real_trial(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_args(make_args(enable_real_api=True, run_real_trial=False))

    def test_dry_run_outputs_are_secret_free_and_manifest_records_runtime_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "tmp" / "MMLongBench"
            create_extract_pages(extract_root, "doc_a", [0, 1])
            stage2_json = root / "stage2.json"
            config_path = root / "qwen3vl.yaml"
            output_dir = root / "outputs"
            stage2_json.write_text(json.dumps([make_stage2_record("doc_a.pdf", pages=[0, 1])]), encoding="utf-8")
            config_path.write_text("model: Qwen/Qwen3-VL-8B-Instruct\napi_key: SECRET_SHOULD_NOT_LEAK\n", encoding="utf-8")

            result = run_crossdoc_batch(
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

            summary = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
            artifacts_text = (output_dir / "artifacts.jsonl").read_text(encoding="utf-8")
            discard_text = (output_dir / "discard.jsonl").read_text(encoding="utf-8")
            combined = json.dumps(summary, ensure_ascii=False) + artifacts_text + discard_text

        for forbidden in [
            "SECRET_SHOULD_NOT_LEAK",
            '"proof_trace"',
            "verified",
            "answer_supported",
            "proof_used",
        ]:
            self.assertNotIn(forbidden, combined)
        self.assertEqual(summary["storage_format"], "artifacts_jsonl")
        self.assertEqual(result["summary"]["num_pages_attempted"], 2)
        self.assertEqual(result["summary"]["num_documents_attempted"], 1)
        self.assertFalse((output_dir / "artifact_stores").exists())
        self.assertFalse((output_dir / "raw_outputs").exists())
        self.assertFalse((output_dir / "reports" / "run_manifest.json").exists())
        artifact_rows = [json.loads(line) for line in artifacts_text.splitlines() if line.strip()]
        self.assertTrue(artifact_rows)
        self.assertEqual(
            set(artifact_rows[0]),
            {
                "record_index",
                "doc_id",
                "page_id",
                "page_index",
                "artifact_id",
                "artifact_type",
                "modality",
                "content",
                "normalized_content",
                "source_anchors",
                "provenance",
                "status",
                "validation_status",
                "locators",
                "source_anchored",
                "element_locatable",
                "proof_trace_eligible",
            },
        )
        self.assertNotIn("compiler_metadata", combined)
        self.assertNotIn("page_text_path", combined)
        self.assertNotIn("page_image_path", combined)
        self.assertNotIn("layout_blocks", combined)


def make_stage2_record(
    doc_id: str,
    pages: List[int],
    answer: str = "GOLD_SECRET",
    binary_correctness: bool = False,
    has_image: bool = True,
    layout_block_ids: List[str] | None = None,
    page_count: Any = 120,
) -> Dict[str, Any]:
    layout_block_ids = ["p000_full_page_image"] if layout_block_ids is None else layout_block_ids
    candidate_page_routes = [
        {"page_index": int(page_index), "routes": ["text", "image"]}
        for page_index in pages
    ]
    return {
        "doc_id": doc_id,
        "question": "What is visible?",
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
    max_docs: int = 5,
    max_pages_per_doc: int = 2,
    max_pages: int = 10,
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
        sidecar_dir=None,
        selected_pages_csv=None,
        max_docs=max_docs,
        max_pages_per_doc=max_pages_per_doc,
        max_pages=max_pages,
        provider=provider,
        model_name=model_name,
        prompt_version="artifact_compiler_prompt_v1",
        enable_real_api=enable_real_api,
        run_real_trial=run_real_trial,
        dry_run_fake_client=dry_run_fake_client,
        deterministic_dedup_enabled=deterministic_dedup_enabled,
        timeout_seconds=120,
    )


if __name__ == "__main__":
    unittest.main()
