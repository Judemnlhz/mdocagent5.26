"""Tests for MDocAgent compatibility helpers and preflight wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from mdocnexus.stage2.index_builder import (
    build_api_run_config_from_mdocagent_yaml,
    build_mdocagent_extract_paths,
    find_record_by_id_or_doc_question,
    normalize_doc_name_for_mdocagent,
    read_json_or_jsonl_records,
)
from mdocnexus.stage2.page_input import load_page_content
from mdocnexus.stage2.page_input import prepare_pages_for_compilation
from scripts.stage2 import build_preflight_report


class MDocAgentCompatTest(unittest.TestCase):
    def test_doc_name_normalization(self) -> None:
        self.assertEqual(
            normalize_doc_name_for_mdocagent("edb88a99670417f64a6b719646aed326.pdf"),
            "edb88a99670417f64a6b719646aed326",
        )
        self.assertEqual(normalize_doc_name_for_mdocagent("doc.name-v1_abc.pdf"), "doc.name-v1_abc")

    def test_mdocagent_extract_path_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            doc_id = "edb88a99670417f64a6b719646aed326.pdf"
            doc_name = "edb88a99670417f64a6b719646aed326"
            (root / f"{doc_name}_29.png").write_bytes(b"not-a-real-png")
            canonical_record = make_canonical_record(doc_id=doc_id, pages_to_compile=[29])

            built_paths = build_mdocagent_extract_paths(root, doc_id, 29)
            page_content = load_page_content(canonical_record, root, 29)
            prepared = prepare_pages_for_compilation(canonical_record, root)

        self.assertEqual(built_paths["image_candidate_paths"][0], root / f"{doc_name}_29.png")
        self.assertEqual(page_content["page_image_path"], str(root / f"{doc_name}_29.png"))
        self.assertIn(
            "p029_full_page_image",
            [block["block_id"] for block in prepared["pages"][0]["layout_blocks"]],
        )

    def test_qwen3vl_yaml_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "qwen3vl.yaml"
            write_test_yaml(config_path)

            api_config = build_api_run_config_from_mdocagent_yaml(config_path)
            serialized = repr(api_config)

        self.assertEqual(api_config.model_name, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(api_config.api_base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(api_config.api_key_env_var, "SILICONFLOW_API_KEY")
        self.assertFalse(api_config.enable_real_api)
        self.assertEqual(api_config.max_pages, 1)
        self.assertNotIn("SECRET_VALUE_SHOULD_NOT_APPEAR", serialized)

    def test_raw_sample_json_list_reading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / "sample-with-retrieval-results.json"
            sample_path.write_text(json.dumps([make_raw_sample()]), encoding="utf-8")
            records = read_json_or_jsonl_records(sample_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["doc_id"], "example.pdf")

    def test_find_record_by_doc_id_and_question_substring(self) -> None:
        records = [make_raw_sample(doc_id="other.pdf"), make_raw_sample()]
        record = find_record_by_id_or_doc_question(
            records,
            record_id=None,
            doc_id="example.pdf",
            question_substring="page 30",
        )

        self.assertEqual(record["doc_id"], "example.pdf")

    def test_preflight_report_no_api_and_no_key_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_path = root / "sample-with-retrieval-results.json"
            config_path = root / "qwen3vl.yaml"
            extract_root = root / "tmp" / "MMLongBench"
            extract_root.mkdir(parents=True)
            sample_path.write_text(json.dumps([make_raw_sample()]), encoding="utf-8")
            write_test_yaml(config_path)
            (extract_root / "example_29.png").write_bytes(b"not-a-real-png")

            report = build_preflight_report(
                sample_path=sample_path,
                extract_root=extract_root,
                config_path=config_path,
                target_page_index=29,
                doc_id="example.pdf",
                question_substring="page 30",
            )
            report_text = json.dumps(report, ensure_ascii=False)

        self.assertTrue(report["preflight_passed"])
        self.assertFalse(report["will_call_api"])
        self.assertEqual(report["target_page_index"], 29)
        self.assertIn("p029_full_page_image", report["layout_block_ids"])
        self.assertNotIn("SECRET_VALUE_SHOULD_NOT_APPEAR", report_text)
        self.assertNotIn(chr(34) + "api_key" + chr(34), report_text)


def write_test_yaml(config_path: Path) -> None:
    lines = [
        "model_id: Qwen/Qwen3-VL-8B-Instruct",
        "model: Qwen/Qwen3-VL-8B-Instruct",
        "base_url: https://api.siliconflow.cn/v1",
        "api_key: SECRET_VALUE_SHOULD_NOT_APPEAR",
        "api_key_env: SILICONFLOW_API_KEY",
        "module_name: models.siliconflow",
        "class_name: SiliconFlowVisionModel",
    ]
    config_path.write_text(chr(10).join(lines), encoding="utf-8")


def make_canonical_record(doc_id: str = "example.pdf", pages_to_compile: list[int] | None = None) -> Dict[str, Any]:
    pages_to_compile = pages_to_compile or [29]
    return {
        "document": {"doc_id": doc_id, "doc_type": "test", "dataset": "MMLongBench"},
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


def make_raw_sample(doc_id: str = "example.pdf") -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "doc_type": "test",
        "question": "Are there blue handwritten words on page 30?",
        "answer_format": "Str",
        "answer": "Not answerable",
        "evidence_pages": "[30]",
        "evidence_sources": "[]",
        "text-top-10-question": [3, 4],
        "text-top-10-question_score": [1.0, 0.9],
        "image-top-10-question": [29],
        "image-top-10-question_score": [2.0],
        "text-index-path-question": ".ragatouille/index",
    }


if __name__ == "__main__":
    unittest.main()
