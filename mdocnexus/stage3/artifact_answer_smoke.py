"""Artifact-only answer smoke for Stage 3B.

This module consumes Stage 3A retrieved artifacts and writes conservative
extractive predictions. It intentionally uses only retrieved artifact content,
does not read raw page files, does not call model APIs, and does not use gold
evidence fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_STAGE2_JSON = "outputs/stage2/clean/sample-with-stage2-index.json"
DEFAULT_ARTIFACTS_JSONL = "outputs/stage2/clean/artifacts.jsonl"
DEFAULT_STAGE3A_RESULTS_JSONL = "outputs/stage3/artifact_retrieval_dryrun/results.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/stage3/artifact_answer_smoke"

FORBIDDEN_KEYS = {
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
}

NEGATIVE_CONTENT_RE = re.compile(
    r"\b("
    r"no relevant content|"
    r"not related(?: to the question)?|"
    r"irrelevant|"
    r"unrelated|"
    r"cannot determine|"
    r"not enough information|"
    r"no content"
    r")\b",
    re.IGNORECASE,
)

SENTENCE_END_RE = re.compile(r"[.;!?]\s+")


def run_artifact_answer_smoke(
    stage2_json_path: str | Path = DEFAULT_STAGE2_JSON,
    artifacts_jsonl_path: str | Path = DEFAULT_ARTIFACTS_JSONL,
    stage3a_results_jsonl_path: str | Path = DEFAULT_STAGE3A_RESULTS_JSONL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run artifact-only answer smoke and write result/quality JSON files."""

    records_by_index = load_stage2_records_by_index(stage2_json_path)
    artifact_ids_by_record = load_artifact_ids_by_record(artifacts_jsonl_path)
    stage3a_rows = load_jsonl(stage3a_results_jsonl_path)

    result_rows: list[dict[str, Any]] = []
    num_cross_record_artifacts_seen = 0

    for row in stage3a_rows:
        record_index = coerce_int(row.get("record_index"), fallback=-1)
        retrieved_artifacts = row.get("retrieved_artifacts") or []
        if not retrieved_artifacts:
            continue
        if not isinstance(retrieved_artifacts, list):
            raise ValueError(f"retrieved_artifacts is not a list for record_index={record_index}")

        same_record_artifacts, rejected_count = keep_same_record_retrieved_artifacts(
            record_index=record_index,
            retrieved_artifacts=retrieved_artifacts,
            artifact_ids_by_record=artifact_ids_by_record,
        )
        num_cross_record_artifacts_seen += rejected_count

        record = records_by_index.get(record_index, {})
        question = str(row.get("question") or record.get("question") or "")
        prediction, status = build_artifact_only_prediction(
            question=question,
            retrieved_artifacts=same_record_artifacts,
        )
        result = {
            "record_index": record_index,
            "doc_id": row.get("doc_id", record.get("doc_id")),
            "question": question,
            "retrieved_artifact_ids": [
                artifact.get("artifact_id") for artifact in same_record_artifacts
            ],
            "prediction": prediction,
            "answer_source": "artifact_only",
            "status": status,
        }
        assert_no_forbidden_keys(result)
        result_rows.append(result)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / "results.jsonl"
    quality_report_path = output_path / "quality_report.json"

    with results_path.open("w", encoding="utf-8") as file_obj:
        for result in result_rows:
            file_obj.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    quality_report = build_quality_report(
        result_rows=result_rows,
        num_cross_record_artifacts_seen=num_cross_record_artifacts_seen,
    )
    assert_no_forbidden_keys(quality_report)
    quality_report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "results_path": str(results_path),
        "quality_report_path": str(quality_report_path),
        "quality_report": quality_report,
    }


def load_stage2_records_by_index(path: str | Path) -> dict[int, dict[str, Any]]:
    records = read_json_records(path)
    indexed: dict[int, dict[str, Any]] = {}
    for inferred_index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Record at offset {inferred_index} is not a JSON object")
        stage2 = raw_record.get("stage2") if isinstance(raw_record.get("stage2"), dict) else {}
        record_index = coerce_int(
            raw_record.get("record_index", stage2.get("record_index")),
            fallback=inferred_index,
        )
        indexed[record_index] = {
            "record_index": record_index,
            "doc_id": raw_record.get("doc_id"),
            "question": raw_record.get("question") or "",
        }
    return indexed


def read_json_records(path: str | Path) -> list[Any]:
    input_path = Path(path)
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items"):
            records = value.get(key)
            if isinstance(records, list):
                return records
    raise ValueError(f"Expected a JSON array or object with records in {input_path}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {input_path} is not a JSON object")
            rows.append(row)
    return rows


def load_artifact_ids_by_record(path: str | Path) -> dict[int, set[str]]:
    artifact_ids_by_record: dict[int, set[str]] = {}
    for artifact in load_jsonl(path):
        record_index = coerce_int(artifact.get("record_index"), fallback=-1)
        artifact_id = artifact.get("artifact_id")
        if artifact_id is None:
            continue
        artifact_ids_by_record.setdefault(record_index, set()).add(str(artifact_id))
    return artifact_ids_by_record


def keep_same_record_retrieved_artifacts(
    record_index: int,
    retrieved_artifacts: Iterable[dict[str, Any]],
    artifact_ids_by_record: dict[int, set[str]],
) -> tuple[list[dict[str, Any]], int]:
    same_record_artifact_ids = artifact_ids_by_record.get(record_index, set())
    kept: list[dict[str, Any]] = []
    rejected_count = 0
    for artifact in retrieved_artifacts:
        if not isinstance(artifact, dict):
            rejected_count += 1
            continue
        artifact_id = artifact.get("artifact_id")
        if artifact_id is None or str(artifact_id) not in same_record_artifact_ids:
            rejected_count += 1
            continue
        kept.append(artifact)
    return kept, rejected_count


def build_artifact_only_prediction(
    question: str,
    retrieved_artifacts: list[dict[str, Any]],
) -> tuple[str, str]:
    usable_artifacts = [artifact for artifact in retrieved_artifacts if artifact_content(artifact)]
    if not usable_artifacts:
        return "", "insufficient_artifact"

    evidence_text = "\n".join(artifact_content(artifact) for artifact in usable_artifacts)
    if NEGATIVE_CONTENT_RE.search(evidence_text):
        return "", "insufficient_artifact"

    max_score = max(coerce_float(artifact.get("score"), fallback=0.0) for artifact in usable_artifacts)
    if max_score <= 0.0:
        return "", "insufficient_artifact"

    question_lower = question.lower()
    evidence_lower = evidence_text.lower()
    if asks_for_vote_or_election_percentage(question_lower, evidence_lower):
        return "", "insufficient_artifact"

    prediction = extract_direct_prediction(question=question, artifacts=usable_artifacts)
    if not prediction:
        return "", "insufficient_artifact"
    return prediction, "answered"


def asks_for_vote_or_election_percentage(question_lower: str, evidence_lower: str) -> bool:
    asks_percentage = "percentage" in question_lower or "percent" in question_lower
    asks_voting = "voted" in question_lower or "vote" in question_lower or "election" in question_lower
    if not (asks_percentage and asks_voting):
        return False
    has_voting_evidence = "voted" in evidence_lower or "vote" in evidence_lower or "election" in evidence_lower
    has_percentage_evidence = "%" in evidence_lower or "percent" in evidence_lower
    return not (has_voting_evidence and has_percentage_evidence)


def extract_direct_prediction(question: str, artifacts: list[dict[str, Any]]) -> str:
    question_lower = question.lower()
    primary_content = artifact_content(artifacts[0])
    evidence_text = "\n".join(artifact_content(artifact) for artifact in artifacts)

    if "where" in question_lower:
        prediction = extract_location_answer(evidence_text)
        if prediction:
            return prediction

    if "map" in question_lower or "diagram" in question_lower:
        if "map" in evidence_text.lower() or "diagram" in evidence_text.lower():
            return clean_prediction_text(primary_content)

    if "title" in question_lower:
        token_count = len(primary_content.split())
        if token_count <= 12:
            return clean_prediction_text(primary_content)
        return ""

    return clean_prediction_text(primary_content)


def extract_location_answer(text: str) -> str:
    patterns = (
        r"\bconceived\s+in\s+(?:the\s+)?([^.;\n]+)",
        r"\bwas\s+founded\s+in\s+(?:the\s+)?([^.;\n]+)",
        r"\bin\s+(?:the\s+)?([^.;\n]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_prediction_text(match.group(1))
    return ""


def clean_prediction_text(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    pieces = SENTENCE_END_RE.split(cleaned, maxsplit=1)
    if pieces:
        cleaned = pieces[0].strip()
    return cleaned.rstrip(".;")


def artifact_content(artifact: dict[str, Any]) -> str:
    content = artifact.get("content")
    return str(content).strip() if content is not None else ""


def build_quality_report(
    result_rows: list[dict[str, Any]],
    num_cross_record_artifacts_seen: int,
) -> dict[str, Any]:
    num_records_attempted = len(result_rows)
    num_answered = sum(1 for row in result_rows if row.get("status") == "answered")
    num_insufficient_artifact = sum(
        1 for row in result_rows if row.get("status") == "insufficient_artifact"
    )
    denominator = max(1, num_records_attempted)
    return {
        "num_records_attempted": num_records_attempted,
        "num_answered": num_answered,
        "num_insufficient_artifact": num_insufficient_artifact,
        "answer_rate": num_answered / denominator,
        "artifact_only": True,
        "raw_page_fallback": False,
        "retrieval_scope": "same_record_only",
        "num_cross_record_artifacts_used": 0,
        "num_retrieved_artifacts_rejected_by_scope": num_cross_record_artifacts_seen,
    }


def coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden key present: {key}")
            assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_keys(child)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3B artifact-only answer smoke.")
    parser.add_argument("--stage2-json", default=DEFAULT_STAGE2_JSON)
    parser.add_argument("--artifacts-jsonl", default=DEFAULT_ARTIFACTS_JSONL)
    parser.add_argument("--stage3a-results-jsonl", default=DEFAULT_STAGE3A_RESULTS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_artifact_answer_smoke(
        stage2_json_path=args.stage2_json,
        artifacts_jsonl_path=args.artifacts_jsonl,
        stage3a_results_jsonl_path=args.stage3a_results_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["quality_report"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
