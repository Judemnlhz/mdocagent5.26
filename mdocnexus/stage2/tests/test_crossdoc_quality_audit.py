"""Tests for offline cross-document Stage 2 quality auditing."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from mdocnexus.stage2.reports import audit_crossdoc_batch


class CrossDocQualityAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.batch_dir = self.root / "outputs" / "stage2" / "artifacts_real_crossdoc_batch"
        self.stage2_json = self.root / "outputs" / "stage2" / "MMLongBench" / "sample-with-stage2-index.json"
        self.store_dir = self.batch_dir / "artifact_stores"
        self.discard_path = self.batch_dir / "discard" / "discard.jsonl"
        self.raw_path = self.batch_dir / "raw_outputs" / "raw_outputs.jsonl"
        self.reports_dir = self.batch_dir / "reports"
        build_valid_text_only_fixture(
            batch_dir=self.batch_dir,
            stage2_json=self.stage2_json,
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def audit(self) -> dict:
        return audit_crossdoc_batch(self.batch_dir, self.stage2_json)

    def test_reads_multiple_artifact_stores(self) -> None:
        report = self.audit()

        self.assertEqual(report["num_documents"], 5)
        self.assertEqual(report["num_pages"], 10)
        self.assertEqual(report["num_artifacts"], 10)

    def test_counts_artifact_type_and_modality_distribution(self) -> None:
        report = self.audit()

        self.assertEqual(report["num_artifacts_by_type"], {"text_span": 10})
        self.assertEqual(report["num_artifacts_by_modality"], {"text": 10})

    def test_detects_zero_visual_artifact_page_coverage(self) -> None:
        report = self.audit()

        self.assertEqual(report["modality_coverage"]["visual_artifact_page_coverage"], 0.0)
        self.assertEqual(report["modality_coverage"]["num_pages_with_visual_artifact"], 0)

    def test_counts_only_text_span_pages(self) -> None:
        report = self.audit()

        self.assertEqual(len(report["artifact_type_diagnosis"]["only_text_span_pages"]), 10)

    def test_counts_validation_issue_types(self) -> None:
        self.discard_path.write_text(
            json.dumps({"doc_id": "doc_0.pdf", "page_index": 0, "error_type": "schema_error"}) + "\n",
            encoding="utf-8",
        )

        report = self.audit()

        self.assertEqual(report["validation_issue_types"], {"schema_error": 1})
        self.assertEqual(report["artifact_type_diagnosis"]["discarded_artifact_issue_types"], {"schema_error": 1})

    def test_forbidden_field_causes_gate_failure(self) -> None:
        mutate_artifact(self.store_dir / "doc_0_p0.json", "answer", "do_not_use")

        report = self.audit()

        self.assertGreater(report["forbidden_field_violations"], 0)
        self.assertFalse(report["stage2_readiness_gate"]["passed"])
        self.assertIn("forbidden_field_violations", report["stage2_readiness_gate"]["blocking_reasons"])

    def test_api_key_causes_gate_failure_without_exposing_value(self) -> None:
        secret = "sk-this-value-must-not-appear-in-audit-output"
        mutate_artifact(self.store_dir / "doc_0_p0.json", "api_key", secret)

        report = self.audit()
        serialized = json.dumps(report, ensure_ascii=False)

        self.assertGreater(report["api_key_leaks"], 0)
        self.assertFalse(report["stage2_readiness_gate"]["passed"])
        self.assertIn("api_key_leaks", report["stage2_readiness_gate"]["blocking_reasons"])
        self.assertNotIn(secret, serialized)

    def test_inconsistent_source_anchor_causes_gate_failure(self) -> None:
        replace_source_reference(self.store_dir / "doc_0_p0.json", "source_anchors", "doc_0.pdf", 9)

        report = self.audit()

        self.assertFalse(report["all_artifacts_have_source_anchors"])
        self.assertFalse(report["stage2_readiness_gate"]["passed"])
        self.assertIn("source_anchor_resolution_failed", report["stage2_readiness_gate"]["blocking_reasons"])

    def test_inconsistent_provenance_source_causes_gate_failure(self) -> None:
        replace_source_reference(self.store_dir / "doc_0_p0.json", "provenance", "other_doc.pdf", 0)

        report = self.audit()

        self.assertTrue(report["all_artifacts_have_source_anchors"])
        self.assertFalse(report["all_provenance_sources_resolve"])
        self.assertIn("provenance_source_resolution_failed", report["stage2_readiness_gate"]["blocking_reasons"])

    def test_audit_does_not_use_gold_input_fields(self) -> None:
        report = self.audit()
        serialized = json.dumps(report, ensure_ascii=False)

        self.assertNotIn("GOLD_POISON", serialized)
        self.assertNotIn("PAGE_POISON", serialized)
        self.assertNotIn("BINARY_POISON", serialized)

    def test_loads_compact_page_routes_for_image_input(self) -> None:
        report = self.audit()

        self.assertEqual(report["modality_coverage"]["num_pages_with_image_input"], 10)
        self.assertEqual(report["selection_diagnosis"]["num_image_top_10_first_available"], 10)

    def test_warnings_include_zero_visual_coverage(self) -> None:
        report = self.audit()

        self.assertTrue(report["stage2_readiness_gate"]["passed"])
        self.assertIn("visual_artifact_coverage_zero", report["stage2_readiness_gate"]["warnings"])
        self.assertIn("table_or_figure_artifact_coverage_zero", report["stage2_readiness_gate"]["warnings"])
        self.assertIn("artifact_types_text_only", report["stage2_readiness_gate"]["warnings"])
        self.assertIn("artifact_modalities_text_only", report["stage2_readiness_gate"]["warnings"])


def build_valid_text_only_fixture(batch_dir: Path, stage2_json: Path) -> None:
    store_dir = batch_dir / "artifact_stores"
    raw_dir = batch_dir / "raw_outputs"
    discard_dir = batch_dir / "discard"
    reports_dir = batch_dir / "reports"
    for path in (store_dir, raw_dir, discard_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    records = []
    quality_rows = []
    for doc_number in range(5):
        doc_id = f"doc_{doc_number}.pdf"
        for page_index in range(2):
            block_id = f"p{page_index:03d}_text_0000"
            artifact = {
                "artifact_id": f"{doc_number}_{page_index}",
                "doc_id": doc_id,
                "page_index": page_index,
                "artifact_type": "text_span",
                "modality": "text",
                "source_anchors": [
                    {"page_index": page_index, "source_id": block_id}
                ],
                "provenance": {
                    "sources": [block_id]
                },
            }
            store = {"doc_id": doc_id, "pages": [{"page_index": page_index, "artifacts": [artifact]}]}
            (store_dir / f"doc_{doc_number}_p{page_index}.json").write_text(
                json.dumps(store),
                encoding="utf-8",
            )
            quality_rows.append(
                {
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "selection_reason": "image_top_10_first_available",
                    "page_image_path": f"/tmp/{doc_id}_{page_index}.png",
                }
            )
        records.append(
            {
                "doc_id": doc_id,
                "question": "Audit only",
                "answer": "GOLD_POISON",
                "evidence_pages": "PAGE_POISON",
                "binary_correctness": "BINARY_POISON",
                "stage2": {
                    "preflight": {"passed": True, "blocking_reasons": []},
                    "candidate_page_routes": [
                        {"page_index": 0, "routes": ["text", "image"]},
                        {"page_index": 1, "routes": ["text", "image"]},
                    ],
                },
            }
        )

    stage2_json.parent.mkdir(parents=True, exist_ok=True)
    stage2_json.write_text(json.dumps(records), encoding="utf-8")
    (raw_dir / "raw_outputs.jsonl").write_text("", encoding="utf-8")
    (discard_dir / "discard.jsonl").write_text("", encoding="utf-8")
    (reports_dir / "crossdoc_batch_summary.json").write_text(
        json.dumps(
            {
                "num_documents_attempted": 5,
                "num_pages_attempted": 10,
                "num_raw_artifacts": 10,
                "num_valid_artifacts": 10,
                "schema_valid_rate": 1.0,
                "anchoring_rate": 1.0,
                "discard_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    with (reports_dir / "crossdoc_batch_quality.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["doc_id", "page_index", "selection_reason", "page_image_path"])
        writer.writeheader()
        writer.writerows(quality_rows)
    (reports_dir / "run_manifest.json").write_text(
        json.dumps({"stage": "stage2_crossdoc_small_batch_artifact_compilation", "real_api_called": True}),
        encoding="utf-8",
    )


def mutate_artifact(store_path: Path, key: str, value: str) -> None:
    store = json.loads(store_path.read_text(encoding="utf-8"))
    store["pages"][0]["artifacts"][0][key] = value
    store_path.write_text(json.dumps(store), encoding="utf-8")


def replace_source_reference(store_path: Path, field: str, doc_id: str, page_index: int) -> None:
    store = json.loads(store_path.read_text(encoding="utf-8"))
    artifact = store["pages"][0]["artifacts"][0]
    reference = {"doc_id": doc_id, "page_index": page_index, "block_id": "p000_text_0000"}
    if field == "provenance":
        artifact["provenance"] = {"sources": [reference]}
    else:
        artifact[field] = [reference]
    store_path.write_text(json.dumps(store), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
