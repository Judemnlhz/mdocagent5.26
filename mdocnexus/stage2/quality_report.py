"""Quality summary helpers for Stage 2 artifact stores."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


QUALITY_FIELDS = [
    "doc_id",
    "num_pages_compiled",
    "num_artifacts",
    "num_schema_valid_artifacts",
    "num_anchored_artifacts",
    "schema_valid_rate",
    "anchoring_rate",
    "discard_rate",
    "num_explicit_constraint_pages_compiled",
    "num_retrieval_missed_explicit_pages_compiled",
]


def summarize_artifact_store(store: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize one artifact store without evaluation-only metrics."""

    stats = store.get("compilation_statistics", {})
    return {
        "doc_id": store.get("document", {}).get("doc_id"),
        "num_pages_compiled": stats.get("num_pages_compiled", 0),
        "num_artifacts": stats.get("num_artifacts", 0),
        "num_schema_valid_artifacts": stats.get("num_schema_valid_artifacts", 0),
        "num_anchored_artifacts": stats.get("num_anchored_artifacts", 0),
        "schema_valid_rate": stats.get("schema_valid_rate", 0.0),
        "anchoring_rate": stats.get("anchoring_rate", 0.0),
        "discard_rate": stats.get("discard_rate", 0.0),
        "num_explicit_constraint_pages_compiled": stats.get(
            "num_explicit_constraint_pages_compiled",
            0,
        ),
        "num_retrieval_missed_explicit_pages_compiled": stats.get(
            "num_retrieval_missed_explicit_pages_compiled",
            0,
        ),
    }


def write_quality_summary(
    rows: List[Dict[str, Any]],
    output_csv: str | Path,
    output_json: str | Path,
) -> None:
    """Write per-document CSV rows and aggregate JSON summary."""

    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in QUALITY_FIELDS})

    json_path.write_text(
        json.dumps(
            {
                "num_documents": len(rows),
                "aggregate": _aggregate_rows(rows),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_documents = max(1, len(rows))
    return {
        "total_pages_compiled": sum(row.get("num_pages_compiled", 0) for row in rows),
        "total_artifacts": sum(row.get("num_artifacts", 0) for row in rows),
        "mean_schema_valid_rate": sum(row.get("schema_valid_rate", 0.0) for row in rows) / num_documents,
        "mean_anchoring_rate": sum(row.get("anchoring_rate", 0.0) for row in rows) / num_documents,
        "mean_discard_rate": sum(row.get("discard_rate", 0.0) for row in rows) / num_documents,
    }
