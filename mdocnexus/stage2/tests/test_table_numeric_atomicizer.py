from __future__ import annotations

import unittest

from mdocnexus.stage2.artifact_quality import is_atomic_strong_eligible
from mdocnexus.stage2.table_numeric_atomicizer import atomicize_table_numeric_artifacts


class TableNumericAtomicizerTest(unittest.TestCase):
    def test_extracts_generic_multicolumn_numeric_rows(self) -> None:
        page_input = make_page_input(
            "\n".join(
                [
                    "Model",
                    "ROUGE",
                    "BLEU-1",
                    "BLEU-4",
                    "METEOR",
                    "Method A",
                    "30.87%",
                    "23.50%",
                    "6.42%",
                    "19.20%",
                ]
            )
        )

        artifacts = atomicize_table_numeric_artifacts(
            selected_page={"doc_id": "paper.pdf", "page_index": 2},
            page_input=page_input,
            existing_artifacts=[],
            max_cells=4,
        )

        table_cells = [artifact for artifact in artifacts if artifact["artifact_type"] == "table_cell"]
        numeric_facts = [artifact for artifact in artifacts if artifact["artifact_type"] == "numeric_fact"]
        self.assertEqual(len(table_cells), 4)
        self.assertEqual(len(numeric_facts), 4)
        self.assertEqual(table_cells[0]["normalized_content"]["row_header"], "Method A")
        self.assertEqual(table_cells[0]["normalized_content"]["column_header"], "ROUGE")
        self.assertEqual(table_cells[0]["normalized_content"]["value_text"], "30.87%")
        self.assertTrue(is_atomic_strong_eligible(table_cells[0], "eligible"))
        self.assertTrue(is_atomic_strong_eligible(numeric_facts[0], "eligible"))

    def test_extracts_generic_year_header_numeric_rows(self) -> None:
        page_input = make_page_input(
            "\n".join(
                [
                    "Performance Summary",
                    "2023",
                    "2022",
                    "2021",
                    "Total units",
                    "3,504",
                    "3,931",
                    "3,969",
                ]
            )
        )

        artifacts = atomicize_table_numeric_artifacts(
            selected_page={"doc_id": "report.pdf", "page_index": 4},
            page_input=page_input,
            existing_artifacts=[],
            max_cells=3,
        )

        table_cells = [artifact for artifact in artifacts if artifact["artifact_type"] == "table_cell"]
        self.assertEqual([cell["normalized_content"]["column_header"] for cell in table_cells], ["2023", "2022", "2021"])
        self.assertEqual([cell["normalized_content"]["value_text"] for cell in table_cells], ["3,504", "3,931", "3,969"])

    def test_deduplicates_existing_atomic_artifacts(self) -> None:
        page_input = make_page_input("\n".join(["2023", "2022", "Total units", "10", "20"]))
        existing = [
            {
                "artifact_id": "existing_cell",
                "artifact_type": "table_cell",
                "normalized_content": {"row_header": "Total units", "column_header": "2023", "value_text": "10"},
            }
        ]

        artifacts = atomicize_table_numeric_artifacts(
            selected_page={"doc_id": "report.pdf", "page_index": 4},
            page_input=page_input,
            existing_artifacts=existing,
            max_cells=2,
        )

        table_cells = [artifact for artifact in artifacts if artifact["artifact_type"] == "table_cell"]
        self.assertEqual(len(table_cells), 1)
        self.assertEqual(table_cells[0]["normalized_content"]["column_header"], "2022")


def make_page_input(text: str) -> dict:
    return {
        "doc_id": "example.pdf",
        "page_index": 2,
        "page_text": text,
        "layout_blocks": [
            {
                "block_id": "p002_text_0000",
                "block_type": "text_block",
                "page_index": 2,
                "bbox": None,
                "text": text,
                "char_start": 0,
                "char_end": len(text),
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
