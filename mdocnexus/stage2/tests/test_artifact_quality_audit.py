"""Tests for offline artifact quality audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from mdocnexus.stage2.artifact_quality_audit import (
    audit_artifact_store,
    audit_batch_artifact_outputs,
    load_batch_artifact_stores,
)


class ArtifactQualityAuditTest(unittest.TestCase):
    def test_loads_multiple_artifact_stores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir)
            write_store(store_dir / "b.json", make_store(doc_id="b.pdf", page_index=1))
            write_store(store_dir / "a.json", make_store(doc_id="a.pdf", page_index=0))

            stores = load_batch_artifact_stores(store_dir)

        self.assertEqual(len(stores), 2)
        self.assertEqual([store["document"]["doc_id"] for store in stores], ["a.pdf", "b.pdf"])

    def test_counts_artifact_type_distribution(self) -> None:
        store = make_store(
            artifacts=[
                make_artifact("a1", artifact_type="text_span"),
                make_artifact("a2", artifact_type="visual_observation"),
                make_artifact("a3", artifact_type="visual_observation"),
            ]
        )

        audit = audit_artifact_store(store)

        self.assertEqual(audit["num_artifacts_by_type"], {"text_span": 1, "visual_observation": 2})

    def test_counts_modality_distribution(self) -> None:
        store = make_store(
            artifacts=[
                make_artifact("a1", modality="text"),
                make_artifact("a2", modality="visual"),
            ]
        )

        audit = audit_artifact_store(store)

        self.assertEqual(audit["num_artifacts_by_modality"], {"text": 1, "visual": 1})

    def test_missing_source_anchor_fails_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = make_batch_dir(
                Path(tmpdir),
                [make_store(artifacts=[make_artifact("a1", source_id="missing_block")])],
                schema_valid_rate=1.0,
                anchoring_rate=1.0,
                discard_rate=0.0,
            )

            audit = audit_batch_artifact_outputs(batch_dir)

        self.assertFalse(audit["all_artifacts_have_source_anchors"])
        self.assertFalse(audit["stage2_quality_gate"]["passed"])
        self.assertIn("source_anchor_consistency_failed", audit["stage2_quality_gate"]["blocking_reasons"])

    def test_missing_provenance_source_fails_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = make_artifact("a1", provenance_sources=["missing_block"])
            batch_dir = make_batch_dir(Path(tmpdir), [make_store(artifacts=[artifact])])

            audit = audit_batch_artifact_outputs(batch_dir)

        self.assertFalse(audit["all_provenance_sources_resolve"])
        self.assertFalse(audit["stage2_quality_gate"]["passed"])
        self.assertIn("provenance_consistency_failed", audit["stage2_quality_gate"]["blocking_reasons"])

    def test_forbidden_fields_fail_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_store()
            store["answer"] = "GOLD_SECRET"
            batch_dir = make_batch_dir(Path(tmpdir), [store])

            audit = audit_batch_artifact_outputs(batch_dir)

        self.assertEqual(audit["num_forbidden_field_violations"], 1)
        self.assertFalse(audit["stage2_quality_gate"]["passed"])
        self.assertIn("forbidden_field_violations", audit["stage2_quality_gate"]["blocking_reasons"])

    def test_api_key_field_fails_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_store()
            store["compiler"]["api_key"] = "SECRET"
            batch_dir = make_batch_dir(Path(tmpdir), [store])

            audit = audit_batch_artifact_outputs(batch_dir)

        self.assertGreaterEqual(audit["num_api_key_leaks"], 1)
        self.assertFalse(audit["stage2_quality_gate"]["passed"])
        self.assertIn("api_key_leaks", audit["stage2_quality_gate"]["blocking_reasons"])

    def test_normal_artifact_store_passes_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = make_batch_dir(Path(tmpdir), [make_store(), make_store(doc_id="b.pdf", page_index=1)])

            audit = audit_batch_artifact_outputs(batch_dir)

        self.assertTrue(audit["stage2_quality_gate"]["passed"])
        self.assertEqual(audit["num_artifact_stores"], 2)
        self.assertEqual(audit["num_artifacts"], 2)
        self.assertEqual(audit["num_documents"], 2)

    def test_audit_does_not_need_gold_fields(self) -> None:
        store = make_store()
        store["source_record_like"] = {
            "question": "q",
            "answer": "GOLD_SECRET",
            "evidence_pages": [99],
            "binary_correctness": True,
        }

        audit = audit_artifact_store(store)

        self.assertEqual(audit["doc_id"], "example.pdf")
        self.assertEqual(audit["num_forbidden_field_violations"], 3)


def make_batch_dir(
    root: Path,
    stores: List[Dict[str, Any]],
    schema_valid_rate: float = 1.0,
    anchoring_rate: float = 1.0,
    discard_rate: float = 0.0,
) -> Path:
    batch_dir = root / "batch"
    store_dir = batch_dir / "artifact_stores"
    reports_dir = batch_dir / "reports"
    discard_dir = batch_dir / "discard"
    store_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    discard_dir.mkdir(parents=True)
    for index, store in enumerate(stores):
        write_store(store_dir / f"store_{index}.json", store)
    summary = {
        "schema_valid_rate": schema_valid_rate,
        "anchoring_rate": anchoring_rate,
        "discard_rate": discard_rate,
        "num_validation_issues": 0,
    }
    (reports_dir / "batch_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return batch_dir


def make_store(
    doc_id: str = "example.pdf",
    page_index: int = 0,
    artifacts: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    artifacts = [make_artifact("artifact_1", page_index=page_index)] if artifacts is None else artifacts
    return {
        "document": {"doc_id": doc_id, "dataset": None, "page_index_base": 0},
        "compiler": {"model_name": "fake"},
        "pages": [
            {
                "page_index": page_index,
                "page_number_one_based": page_index + 1,
                "page_source": {"page_text_path": None, "page_image_path": "page.png"},
                "layout_blocks": [
                    {
                        "block_id": f"p{page_index:03d}_full_page_image",
                        "block_type": "full_page_image",
                        "page_index": page_index,
                        "bbox": None,
                        "text": None,
                    }
                ],
                "artifacts": artifacts,
            }
        ],
        "artifact_index": {},
        "compilation_statistics": {},
    }


def make_artifact(
    artifact_id: str,
    page_index: int = 0,
    artifact_type: str = "visual_observation",
    modality: str = "visual",
    source_id: str | None = None,
    provenance_sources: List[str] | None = None,
) -> Dict[str, Any]:
    source_id = f"p{page_index:03d}_full_page_image" if source_id is None else source_id
    provenance_sources = [source_id] if provenance_sources is None else provenance_sources
    return {
        "artifact_id": artifact_id,
        "doc_id": "example.pdf",
        "page_index": page_index,
        "artifact_type": artifact_type,
        "modality": modality,
        "content": "content",
        "normalized_content": {},
        "source_anchors": [
            {
                "source_id": source_id,
                "anchor_type": "full_page_image",
                "page_index": page_index,
                "bbox": None,
            }
        ],
        "provenance": {"op": "ATOM", "sources": provenance_sources},
        "validation_status": "anchored",
        "compiler_metadata": {},
    }


def write_store(path: Path, store: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
