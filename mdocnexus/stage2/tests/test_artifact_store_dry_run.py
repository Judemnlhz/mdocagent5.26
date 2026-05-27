"""Tests for Step 4 artifact store and Stage 2 dry-run integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.artifact_pipeline import build_document_artifact_store
from mdocnexus.stage2.artifact_pipeline import validate_page_artifact_output
from mdocnexus.stage2.artifact_pipeline import run_stage2_dry_run
from mdocnexus.stage2.provider import build_mock_page_artifact_output
from mdocnexus.stage2.reports import summarize_artifact_store, write_quality_summary
from mdocnexus.stage2.artifact_schema import ValidationErrorType


class ArtifactStoreDryRunTest(unittest.TestCase):
    def test_artifact_store_builder_basic_structure(self) -> None:
        canonical_record = make_canonical_record(pages_to_compile=[29], explicit_pages=[29])
        prepared_pages = [make_prepared_page(29)]
        raw_output = build_mock_page_artifact_output(
            doc_id="example.pdf",
            page_index=29,
            layout_blocks=prepared_pages[0]["layout_blocks"],
        )
        valid_artifacts, issues = validate_page_artifact_output(
            raw_output,
            prepared_pages[0]["layout_blocks"],
        )
        self.assertEqual(issues, [])

        store = build_document_artifact_store(
            canonical_record=canonical_record,
            prepared_pages=prepared_pages,
            page_artifact_outputs={29: raw_output},
            validation_results={29: {"valid_artifacts": valid_artifacts, "validation_issues": []}},
            compiler_metadata={"compiler_name": "test", "compiler_version": "test"},
        )

        self.assertEqual(
            set(store.keys()),
            {"document", "compiler", "pages", "artifact_index", "compilation_statistics"},
        )
        self.assertIn("29", store["artifact_index"]["by_page_index"])
        self.assertEqual(store["document"]["page_index_base"], 0)

    def test_page_30_dry_run_closes_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "example_29.png").write_bytes(b"not-a-real-png")
            output_path = root / "artifact_store.json"
            canonical_record = make_canonical_record(pages_to_compile=[29], explicit_pages=[29])

            summary = run_stage2_dry_run(canonical_record, root, output_path)
            store = json.loads(output_path.read_text(encoding="utf-8"))
            page_29 = get_store_page(store, 29)
            artifact_types = {artifact["artifact_type"] for artifact in page_29["artifacts"]}

            self.assertTrue(output_path.exists())
            self.assertTrue(summary["quality_gate"]["stage2_dry_run_passed"])
            self.assertIn("visual_observation", artifact_types)
            self.assertIn("29", store["artifact_index"]["by_page_index"])

    def test_invalid_mock_artifact_is_filtered(self) -> None:
        canonical_record = make_canonical_record(pages_to_compile=[29], explicit_pages=[29])
        prepared_pages = [make_prepared_page(29)]
        raw_output = build_mock_page_artifact_output(
            doc_id="example.pdf",
            page_index=29,
            layout_blocks=prepared_pages[0]["layout_blocks"],
        )
        raw_output["artifacts"][0]["source_anchors"][0]["source_id"] = "not_exist"
        valid_artifacts, issues = validate_page_artifact_output(
            raw_output,
            prepared_pages[0]["layout_blocks"],
        )
        store = build_document_artifact_store(
            canonical_record=canonical_record,
            prepared_pages=prepared_pages,
            page_artifact_outputs={29: raw_output},
            validation_results={
                29: {
                    "valid_artifacts": valid_artifacts,
                    "validation_issues": [issue.to_dict() for issue in issues],
                }
            },
            compiler_metadata={},
        )

        self.assertEqual(store["pages"][0]["artifacts"], [])
        self.assertIn(
            ValidationErrorType.source_anchor_not_found,
            {issue.error_type for issue in issues},
        )

    def test_missing_required_page_fails_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "artifact_store.json"
            canonical_record = make_canonical_record(pages_to_compile=[29], explicit_pages=[29])

            summary = run_stage2_dry_run(canonical_record, tmpdir, output_path)

            self.assertFalse(summary["quality_gate"]["stage2_dry_run_passed"])
            self.assertTrue(
                {
                    "missing_source_anchors",
                    "missing_required_page_artifacts",
                    "no_valid_artifacts",
                }.intersection(summary["quality_gate"]["blocking_reasons"])
            )

    def test_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            canonical_record = make_canonical_record(pages_to_compile=[29], explicit_pages=[29])
            prepared_pages = [make_prepared_page(29)]
            raw_output = build_mock_page_artifact_output(
                doc_id="example.pdf",
                page_index=29,
                layout_blocks=prepared_pages[0]["layout_blocks"],
            )
            valid_artifacts, _ = validate_page_artifact_output(
                raw_output,
                prepared_pages[0]["layout_blocks"],
            )
            store = build_document_artifact_store(
                canonical_record=canonical_record,
                prepared_pages=prepared_pages,
                page_artifact_outputs={29: raw_output},
                validation_results={29: {"valid_artifacts": valid_artifacts, "validation_issues": []}},
                compiler_metadata={},
            )
            row = summarize_artifact_store(store)
            csv_path = Path(tmpdir) / "quality.csv"
            json_path = Path(tmpdir) / "quality.json"

            write_quality_summary([row], csv_path, json_path)
            json_data = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(row["doc_id"], "example.pdf")
            self.assertIn("num_artifacts", row)
            self.assertEqual(json_data["num_documents"], 1)

    def test_forbidden_fields_do_not_enter_artifact_store(self) -> None:
        canonical_record = make_canonical_record(pages_to_compile=[29], explicit_pages=[29])
        canonical_record["gold_annotation"] = {"answer": "do not leak"}
        canonical_record["baseline_outputs"] = {"mdocagent": {"answer": "do not leak"}}
        canonical_record["source_record"] = {"answer": "do not leak"}
        prepared_pages = [make_prepared_page(29)]
        raw_output = build_mock_page_artifact_output(
            doc_id="example.pdf",
            page_index=29,
            layout_blocks=prepared_pages[0]["layout_blocks"],
        )
        valid_artifacts, _ = validate_page_artifact_output(
            raw_output,
            prepared_pages[0]["layout_blocks"],
        )

        store = build_document_artifact_store(
            canonical_record=canonical_record,
            prepared_pages=prepared_pages,
            page_artifact_outputs={29: raw_output},
            validation_results={29: {"valid_artifacts": valid_artifacts, "validation_issues": []}},
            compiler_metadata={},
        )
        store_text = json.dumps(store, ensure_ascii=False)

        for forbidden in [
            "gold_annotation",
            "baseline_outputs",
            "source_record",
            "proof_trace",
            "verified",
            "answer_supported",
            "proof_used",
        ]:
            self.assertNotIn(forbidden, store_text)


def make_canonical_record(
    pages_to_compile: List[int],
    explicit_pages: List[int],
) -> Dict[str, Any]:
    return {
        "document": {
            "doc_id": "example.pdf",
            "doc_type": "test",
            "dataset": None,
        },
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
            "explicit_constraint_pages": explicit_pages,
            "retrieval_candidate_pages": [],
            "retrieval_missed_explicit_pages": explicit_pages,
        },
        "compilation_plan": {"pages_to_compile": pages_to_compile},
    }


def make_prepared_page(page_index: int) -> Dict[str, Any]:
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


def get_store_page(store: Dict[str, Any], page_index: int) -> Dict[str, Any]:
    for page in store["pages"]:
        if page["page_index"] == page_index:
            return page
    raise AssertionError(f"Store page not found: {page_index}")


if __name__ == "__main__":
    unittest.main()
