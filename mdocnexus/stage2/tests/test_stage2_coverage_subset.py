"""Tests for document-only Stage 2 coverage subset selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage2_coverage_subset import build_coverage_subset
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
        for forbidden in ("SECRET", "question", "answer", "evidence_pages", "gold"):
            self.assertNotIn(forbidden, public_text)


    def test_doc_compile_subset_does_not_require_stage2_index_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extract_root = root / "extract"
            extract_root.mkdir()
            (extract_root / "alpha_0.txt").write_text("alpha page text", encoding="utf-8")
            (extract_root / "alpha_0.png").write_bytes(b"not-an-image-but-present")
            subset = root / "subset.jsonl"
            subset.write_text(json.dumps({"doc_id": "alpha.pdf", "page_indices": [0]}) + "\n", encoding="utf-8")
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
            self.assertTrue((output_dir / "artifacts.jsonl").is_file())
            self.assertIn("input_hash_unavailable_reason", result["manifest"])


if __name__ == "__main__":
    unittest.main()
