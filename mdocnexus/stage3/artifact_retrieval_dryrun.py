"""Artifact-level retrieval dry run for Stage 3A.

This module reads Stage 2 compact records and clean artifact JSONL output,
then ranks artifacts for each question using only same-record artifacts and
question/content lexical evidence. It does not call model APIs and does not
generate answers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_STAGE2_JSON = "outputs/stage2/clean/sample-with-stage2-index.json"
DEFAULT_ARTIFACTS_JSONL = "outputs/stage2/clean/artifacts.jsonl"
DEFAULT_STAGE2_QUALITY_REPORT = "outputs/stage2/clean/quality_report.json"
DEFAULT_OUTPUT_DIR = "outputs/stage3/artifact_retrieval_dryrun"
DEFAULT_TOP_K = 5

FORBIDDEN_OUTPUT_KEYS = {
    "answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9%]+")


def run_artifact_retrieval_dryrun(
    stage2_json_path: str | Path = DEFAULT_STAGE2_JSON,
    artifacts_jsonl_path: str | Path = DEFAULT_ARTIFACTS_JSONL,
    stage2_quality_report_path: str | Path = DEFAULT_STAGE2_QUALITY_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run same-record artifact retrieval and write JSONL plus quality report."""

    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1")

    records = load_stage2_records(stage2_json_path)
    artifacts = load_artifacts_jsonl(artifacts_jsonl_path)
    load_stage2_quality_report(stage2_quality_report_path)

    artifacts_by_record = group_artifacts_by_record_index(artifacts)
    output_path = Path(output_dir)
    results_path = output_path / "results.jsonl"
    quality_report_path = output_path / "quality_report.json"
    output_path.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    retrieved_type_counts: Counter[str] = Counter()
    retrieved_modality_counts: Counter[str] = Counter()

    for record in records:
        record_index = int(record["record_index"])
        question = str(record.get("question") or "")
        candidate_artifacts = artifacts_by_record.get(record_index, [])
        retrieved_artifacts = retrieve_top_artifacts(
            question=question,
            candidate_artifacts=candidate_artifacts,
            top_k=int(top_k),
        )
        artifact_type_counts = Counter(
            str(item.get("artifact_type") or "unknown") for item in retrieved_artifacts
        )
        modality_counts = Counter(
            str(item.get("modality") or "unknown") for item in retrieved_artifacts
        )
        retrieved_type_counts.update(artifact_type_counts)
        retrieved_modality_counts.update(modality_counts)

        row = {
            "record_index": record_index,
            "doc_id": record.get("doc_id"),
            "question": question,
            "candidate_artifact_count": len(candidate_artifacts),
            "retrieved_artifacts": retrieved_artifacts,
            "num_retrieved_artifacts": len(retrieved_artifacts),
            "artifact_type_counts": dict(sorted(artifact_type_counts.items())),
            "modality_counts": dict(sorted(modality_counts.items())),
        }
        assert_no_forbidden_output_keys(row)
        result_rows.append(row)

    with results_path.open("w", encoding="utf-8") as file_obj:
        for row in result_rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    quality_report = build_quality_report(
        result_rows=result_rows,
        retrieved_type_counts=retrieved_type_counts,
        retrieved_modality_counts=retrieved_modality_counts,
    )
    assert_no_forbidden_output_keys(quality_report, allow_answer_generation=True)
    quality_report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "results_path": str(results_path),
        "quality_report_path": str(quality_report_path),
        "quality_report": quality_report,
    }


def load_stage2_records(path: str | Path) -> list[dict[str, Any]]:
    raw_records = read_json_records(path)
    records: list[dict[str, Any]] = []
    for inferred_index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Record at offset {inferred_index} is not a JSON object")
        stage2 = raw_record.get("stage2")
        if not isinstance(stage2, dict):
            stage2 = {}
        record_index = raw_record.get("record_index", stage2.get("record_index", inferred_index))
        records.append(
            {
                "record_index": coerce_int(record_index, inferred_index),
                "doc_id": raw_record.get("doc_id"),
                "question": raw_record.get("question") or "",
                "candidate_page_routes": stage2.get("candidate_page_routes") or [],
            }
        )
    return records


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


def load_artifacts_jsonl(path: str | Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            artifact = json.loads(line)
            if not isinstance(artifact, dict):
                raise ValueError(f"Artifact line {line_number} is not a JSON object")
            if "record_index" not in artifact:
                raise ValueError(f"Artifact line {line_number} is missing record_index")
            artifact = dict(artifact)
            artifact["record_index"] = coerce_int(artifact.get("record_index"), line_number)
            artifact["_input_order"] = len(artifacts)
            artifacts.append(artifact)
    return artifacts


def load_stage2_quality_report(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Stage 2 quality report not found: {input_path}")
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return value


def group_artifacts_by_record_index(artifacts: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        grouped[int(artifact["record_index"])].append(artifact)
    for record_artifacts in grouped.values():
        record_artifacts.sort(key=lambda item: int(item.get("_input_order", 0)))
    return dict(grouped)


def retrieve_top_artifacts(
    question: str,
    candidate_artifacts: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not candidate_artifacts:
        return []

    query_tokens = tokenize_for_retrieval(question)
    document_tokens = [
        tokenize_for_retrieval(build_artifact_retrieval_text(artifact))
        for artifact in candidate_artifacts
    ]
    scores = bm25_scores(query_tokens=query_tokens, document_tokens=document_tokens)
    scored_artifacts = []
    for artifact, score in zip(candidate_artifacts, scores):
        scored_artifacts.append(
            (
                round(float(score), 8),
                int(artifact.get("_input_order", 0)),
                artifact,
            )
        )
    scored_artifacts.sort(key=lambda item: (-item[0], item[1]))
    return [format_retrieved_artifact(score, artifact) for score, _, artifact in scored_artifacts[:top_k]]


def bm25_scores(query_tokens: list[str], document_tokens: list[list[str]]) -> list[float]:
    if not document_tokens:
        return []
    if not query_tokens:
        return [0.0 for _ in document_tokens]

    num_docs = len(document_tokens)
    doc_freq: Counter[str] = Counter()
    for tokens in document_tokens:
        doc_freq.update(set(tokens))

    doc_lengths = [len(tokens) for tokens in document_tokens]
    avg_doc_length = sum(doc_lengths) / max(1, num_docs)
    avg_doc_length = avg_doc_length or 1.0
    query_counter = Counter(query_tokens)
    scores: list[float] = []
    k1 = 1.5
    b = 0.75
    for tokens, doc_length in zip(document_tokens, doc_lengths):
        token_counts = Counter(tokens)
        score = 0.0
        for token, query_count in query_counter.items():
            term_frequency = token_counts.get(token, 0)
            if term_frequency <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            denominator = term_frequency + k1 * (1.0 - b + b * doc_length / avg_doc_length)
            score += query_count * idf * (term_frequency * (k1 + 1.0)) / denominator
        scores.append(score)
    return scores


def tokenize_for_retrieval(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def build_artifact_retrieval_text(artifact: dict[str, Any]) -> str:
    content = artifact.get("content")
    normalized_content = artifact.get("normalized_content")
    chunks: list[str] = []
    if content is not None:
        chunks.append(str(content))
    if normalized_content:
        if isinstance(normalized_content, str):
            chunks.append(normalized_content)
        else:
            chunks.append(json.dumps(normalized_content, ensure_ascii=False, sort_keys=True))
    return "\n".join(chunks)


def format_retrieved_artifact(score: float, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": artifact.get("artifact_id"),
        "page_index": artifact.get("page_index"),
        "artifact_type": artifact.get("artifact_type"),
        "modality": artifact.get("modality"),
        "score": score,
        "content": artifact.get("content"),
    }


def build_quality_report(
    result_rows: list[dict[str, Any]],
    retrieved_type_counts: Counter[str],
    retrieved_modality_counts: Counter[str],
) -> dict[str, Any]:
    num_records = len(result_rows)
    num_records_with_artifacts = sum(1 for row in result_rows if row["candidate_artifact_count"] > 0)
    num_records_without_artifacts = num_records - num_records_with_artifacts
    total_candidate_artifacts = sum(int(row["candidate_artifact_count"]) for row in result_rows)
    total_retrieved_artifacts = sum(int(row["num_retrieved_artifacts"]) for row in result_rows)
    num_records_with_retrieved_artifacts = sum(1 for row in result_rows if row["num_retrieved_artifacts"] > 0)
    denominator = max(1, num_records)
    return {
        "num_records": num_records,
        "num_records_with_artifacts": num_records_with_artifacts,
        "num_records_without_artifacts": num_records_without_artifacts,
        "artifact_coverage_rate": num_records_with_artifacts / denominator,
        "avg_candidate_artifacts_per_record": total_candidate_artifacts / denominator,
        "avg_retrieved_artifacts": total_retrieved_artifacts / denominator,
        "artifact_hit_rate": num_records_with_retrieved_artifacts / denominator,
        "retrieved_artifact_type_counts": dict(sorted(retrieved_type_counts.items())),
        "retrieved_modality_counts": dict(sorted(retrieved_modality_counts.items())),
        "retrieval_scope": "same_record_only",
        "num_cross_record_artifacts_used": 0,
        "answer_generation": False,
    }


def coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def assert_no_forbidden_output_keys(value: Any, allow_answer_generation: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_OUTPUT_KEYS:
                raise ValueError(f"Forbidden output key present: {key}")
            if key == "answer_generation" and allow_answer_generation:
                continue
            assert_no_forbidden_output_keys(child, allow_answer_generation=allow_answer_generation)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_output_keys(child, allow_answer_generation=allow_answer_generation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3A artifact retrieval dry run.")
    parser.add_argument("--stage2-json", default=DEFAULT_STAGE2_JSON)
    parser.add_argument("--artifacts-jsonl", default=DEFAULT_ARTIFACTS_JSONL)
    parser.add_argument("--stage2-quality-report", default=DEFAULT_STAGE2_QUALITY_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_artifact_retrieval_dryrun(
        stage2_json_path=args.stage2_json,
        artifacts_jsonl_path=args.artifacts_jsonl,
        stage2_quality_report_path=args.stage2_quality_report,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(result["quality_report"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
