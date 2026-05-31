"""Tests for document-only Stage 2 coverage subset selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage2_coverage_subset import build_coverage_subset, build_manifest
from scripts.stage2 import build_parser


class Stage2CoverageSubsetTest(unittest.TestCase):
    def test_subset_selects_by_doc_id_only_and_strips_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queries = root / "queries.jsonl"
            extract_root = root / "extract"
            extract_root.mkdir()
            input_rows = [
                {"doc_id": "b.pdf", "question": "SECRET QUESTION", "answer": "SECRET"},
                {"doc_id": "a.pdf", "question": "another", "evidence_pages": [99]},
                {"doc_id": "a.pdf", "question": "duplicate"},
            ]
            queries.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in input_rows),
                encoding="utf-8",
            )
            for doc_name in ("a", "b"):
                for page_index in (0, 1, 2):
                    (extract_root / f"{doc_name}_{page_index}.txt").write_text("page text", encoding="utf-8")

            rows = build_coverage_subset(
                public_query_input=queries,
                records_path=root / "missing.json",
                extract_root=extract_root,
                max_docs=2,
                max_pages_per_doc=2,
            )

        self.assertEqual([row["doc_id"] for row in rows], ["a.pdf", "b.pdf"])
        self.assertEqual(rows[0]["page_indices"], [0, 1])
        public_text = json.dumps(rows, ensure_ascii=False)
        for forbidden in ("SECRET", "question", "answer", "evidence_pages", "gold_answer", "binary_correctness"):
            self.assertNotIn(forbidden, public_text)



    def test_scope_modes_are_stable_and_do_not_emit_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queries = root / "queries.jsonl"
            records = root / "records.json"
            extract_root = root / "extract"
            extract_root.mkdir()
            query_rows = [
                {"doc_id": "b.pdf", "question": "keyword should not drive scope", "answer": "SECRET"},
                {"doc_id": "a.pdf", "question": "another", "gold_answer": "SECRET"},
                {"doc_id": "a.pdf", "question": "duplicate", "binary_correctness": True},
            ]
            queries.write_text("".join(json.dumps(row) + "\n" for row in query_rows), encoding="utf-8")
            records.write_text(json.dumps(query_rows), encoding="utf-8")
            for doc_name in ("a", "b"):
                for page_index in range(3):
                    (extract_root / f"{doc_name}_{page_index}.txt").write_text("page text", encoding="utf-8")

            doc_first = build_coverage_subset(queries, records, extract_root, max_docs=1, max_pages_per_doc=2, scope_mode="doc_first")
            query_all = build_coverage_subset(queries, records, extract_root, max_docs=None, max_pages_per_doc=1, scope_mode="query_doc_all")
            query_all_again = build_coverage_subset(queries, records, extract_root, max_docs=None, max_pages_per_doc=1, scope_mode="query_doc_all")

        self.assertEqual([row["doc_id"] for row in doc_first], ["a.pdf"])
        self.assertEqual([row["doc_id"] for row in query_all], ["a.pdf", "b.pdf"])
        self.assertEqual(query_all, query_all_again)
        public_text = json.dumps({"doc_first": doc_first, "query_all": query_all}, ensure_ascii=False)
        for forbidden in ("question", "answer", "gold_answer", "binary_correctness", "keyword"):
            self.assertNotIn(forbidden, public_text)

    def test_retrieval_topk_scope_uses_retrieval_pages_not_evidence_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            retrieval = root / "retrieval.json"
            extract_root = root / "extract"
            extract_root.mkdir()
            retrieval_rows = [
                {
                    "doc_id": "doc.pdf",
                    "question": "SECRET QUESTION",
                    "answer": "SECRET",
                    "evidence_pages": [99],
                    "text-top-10-question": [3, 1, 5],
                    "image-top-10-question": [2, 4],
                    "text-top-10-question_score": [1.0, 0.5],
                }
            ]
            retrieval.write_text(json.dumps(retrieval_rows), encoding="utf-8")
            for page_index in range(6):
                (extract_root / f"doc_{page_index}.txt").write_text("page text", encoding="utf-8")

            rows = build_coverage_subset(
                records_path=retrieval,
                retrieval_topk_file=retrieval,
                extract_root=extract_root,
                max_docs=1,
                max_pages_per_doc=2,
                scope_mode="retrieval_topk_scope",
                retrieval_topk=1,
            )
            manifest = build_manifest(
                subset_rows=rows,
                output=root / "subset.jsonl",
                input_path=retrieval,
                scope_mode="retrieval_topk_scope",
                max_docs=1,
                max_pages_per_doc=2,
                top_k=1,
            )

        self.assertEqual(rows[0]["selection_source"], "retrieval_topk_non_gold")
        self.assertEqual(rows[0]["compile_scope_source"], "mdocagent_retrieval_topk")
        self.assertEqual(rows[0]["page_indices"], [2, 3])
        self.assertTrue(manifest["no_gold_fields_used"])
        public_text = json.dumps({"rows": rows, "manifest": manifest}, ensure_ascii=False)
        for forbidden in ("SECRET", "question", "answer", "evidence_pages", "gold_answer", "binary_correctness"):
            self.assertNotIn(forbidden, public_text)


    def test_doc_compile_retrieval_topk_file_without_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "extract"
            extract_root.mkdir()
            (extract_root / "alpha_2.txt").write_text("alpha page text", encoding="utf-8")
            retrieval = root / "retrieval.json"
            retrieval.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": "alpha.pdf",
                            "text-top-10-question": [2],
                            "evidence_pages": [99],
                            "answer": "SECRET",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_dir = root / "coverage"

            parser = build_parser()
            args = parser.parse_args(
                [
                    "doc-compile",
                    "--scope-mode",
                    "retrieval_topk_scope",
                    "--retrieval-topk-file",
                    str(retrieval),
                    "--retrieval-topk",
                    "1",
                    "--extract-root",
                    str(extract_root),
                    "--output-dir",
                    str(output_dir),
                    "--provider",
                    "fake",
                    "--image-payload-mode",
                    "none",
                    "--max-docs",
                    "1",
                    "--max-pages-per-doc",
                    "1",
                    "--max-pages",
                    "1",
                ]
            )
            result = args.func(args)

        self.assertEqual(result["summary"]["num_artifacts"], 6)
        self.assertEqual(result["summary"]["compile_scope_mode"], "retrieval_topk_scope")
        self.assertEqual(result["summary"]["page_selection_source_counts"], {"retrieval_topk_non_gold": 1})

    def test_doc_compile_subset_does_not_require_stage2_index_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "extract"
            extract_root.mkdir()
            (extract_root / "alpha_0.txt").write_text("alpha page text", encoding="utf-8")
            (extract_root / "alpha_0.png").write_bytes(b"not-an-image-but-present")
            subset = root / "subset.jsonl"
            subset.write_text(json.dumps({"doc_id": "alpha.pdf", "page_indices": [0], "selection_source": "retrieval_topk_non_gold"}) + "\n", encoding="utf-8")
            output_dir = root / "coverage"

            parser = build_parser()
            args = parser.parse_args(
                [
                    "doc-compile",
                    "--subset-file",
                    str(subset),
                    "--extract-root",
                    str(extract_root),
                    "--output-dir",
                    str(output_dir),
                    "--scope-mode",
                    "retrieval_topk_scope",
                    "--provider",
                    "fake",
                    "--image-payload-mode",
                    "none",
                    "--max-docs",
                    "1",
                    "--max-pages-per-doc",
                    "1",
                    "--max-pages",
                    "1",
                ]
            )
            result = args.func(args)

            self.assertEqual(result["summary"]["num_artifacts"], 9)
            self.assertEqual(result["summary"]["compile_scope_mode"], "retrieval_topk_scope")
            self.assertEqual(result["summary"]["num_selected_docs"], 1)
            self.assertEqual(result["summary"]["num_selected_pages"], 1)
            self.assertEqual(result["summary"]["page_selection_source_counts"], {"retrieval_topk_non_gold": 1})
            self.assertTrue((output_dir / "artifacts.jsonl").is_file())
            self.assertIn("input_hash_unavailable_reason", result["manifest"])


if __name__ == "__main__":
    unittest.main()
