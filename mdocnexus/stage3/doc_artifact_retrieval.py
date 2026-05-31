"""Deterministic document-generic artifact retrieval for Stage 3.

The retriever ranks Stage 2 document-generic artifacts with deterministic
lexical or hybrid scoring. It may read public query text, but it ignores
Gold/evaluation-only fields and never calls a model provider or generates an
answer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable
from urllib.parse import unquote

from mdocnexus.stage2.locator_enrichment import classify_artifact_locator


DEFAULT_ARTIFACTS_JSONL = "outputs/stage2_doc/artifacts.jsonl"
DEFAULT_QUERY_INPUT = "outputs/stage3_query/public_queries.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/stage3_doc_artifact_retrieval"
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL_METHOD = "deterministic_lexical"
RETRIEVAL_METHODS = {"deterministic_lexical", "deterministic_hybrid"}
DEFAULT_HYBRID_PRESET = "full_hybrid"
SCHEMA_VERSION = "stage3_doc_artifact_retrieval_v1"
SEMANTIC_EDGE_TYPES = {"supports", "contradicts", "derived_from", "semantic_relation", "entails", "refutes"}
DEBUG_EDGE_TYPES = {"same_record", "same_record_debug"}
COMPONENT_NAMES = ["lexical_score", "metadata_score", "locator_score", "type_modality_score", "graph_prior_score"]
HYBRID_PRESET_WEIGHTS: dict[str, dict[str, float]] = {
    "lexical_only": {
        "lexical_score": 1.0,
        "metadata_score": 0.0,
        "locator_score": 0.0,
        "type_modality_score": 0.0,
        "graph_prior_score": 0.0,
    },
    "lexical_metadata": {
        "lexical_score": 1.0,
        "metadata_score": 0.50,
        "locator_score": 0.0,
        "type_modality_score": 0.25,
        "graph_prior_score": 0.0,
    },
    "lexical_locator": {
        "lexical_score": 1.0,
        "metadata_score": 0.0,
        "locator_score": 0.50,
        "type_modality_score": 0.0,
        "graph_prior_score": 0.0,
    },
    "lexical_graph": {
        "lexical_score": 1.0,
        "metadata_score": 0.0,
        "locator_score": 0.0,
        "type_modality_score": 0.0,
        "graph_prior_score": 0.50,
    },
    "full_hybrid": {
        "lexical_score": 1.0,
        "metadata_score": 0.50,
        "locator_score": 0.25,
        "type_modality_score": 0.25,
        "graph_prior_score": 0.25,
    },
    "hybrid_no_graph": {
        "lexical_score": 1.0,
        "metadata_score": 0.50,
        "locator_score": 0.25,
        "type_modality_score": 0.25,
        "graph_prior_score": 0.0,
    },
    "graph_only_prior": {
        "lexical_score": 0.0,
        "metadata_score": 0.0,
        "locator_score": 0.0,
        "type_modality_score": 0.0,
        "graph_prior_score": 1.0,
    },
}
HYBRID_PRESETS = set(HYBRID_PRESET_WEIGHTS)

GOLD_FIELD_NAMES = {
    "answer",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}
FORBIDDEN_PUBLIC_FIELD_NAMES = GOLD_FIELD_NAMES | {
    "answers",
    "raw_response",
    "raw_output",
    "api_key",
    "local_path",
    "absolute_path",
    "image_path",
}
SAFE_PUBLIC_FIELD_NAMES = {
    "no_answer_generation",
    "no_gold_fields_used",
}
TOKEN_RE = re.compile(r"[a-z0-9%]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}


def run_doc_artifact_retrieval(
    artifacts_jsonl_path: str | Path = DEFAULT_ARTIFACTS_JSONL,
    query_input_path: str | Path = DEFAULT_QUERY_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_k: int = DEFAULT_TOP_K,
    retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
    graph_path: str | Path | None = None,
    hybrid_config_path: str | Path | None = None,
    hybrid_preset: str | None = None,
) -> dict[str, Any]:
    """Rank document-generic artifacts for each query and write public outputs."""

    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if retrieval_method not in RETRIEVAL_METHODS:
        raise ValueError(f"Unsupported retrieval_method: {retrieval_method}")

    artifacts_path = Path(artifacts_jsonl_path)
    query_path = Path(query_input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts_jsonl(artifacts_path)
    queries = load_query_records(query_path)
    artifacts_hash = file_sha256(artifacts_path)
    query_input_hash = file_sha256(query_path)
    hybrid_settings = resolve_hybrid_settings(retrieval_method, hybrid_config_path, hybrid_preset)
    graph_prior_requested = graph_path not in (None, "")
    graph_prior, graph_edges_hash = (
        load_graph_prior(graph_path)
        if graph_prior_requested and hybrid_settings["weights"].get("graph_prior_score", 0.0) > 0.0
        else ({}, None)
    )
    artifacts_by_doc = group_artifacts_by_doc_id(artifacts)

    retrieval_rows: list[dict[str, Any]] = []
    for query in queries:
        doc_id = str(query.get("doc_id") or "")
        candidate_artifacts = artifacts_by_doc.get(doc_id, []) if doc_id else list(artifacts)
        ranked = rank_artifacts(
            query_text=str(query.get("question") or ""),
            candidate_artifacts=candidate_artifacts,
            top_k=top_k,
            retrieval_method=retrieval_method,
            graph_prior=graph_prior,
            hybrid_weights=hybrid_settings["weights"],
        )
        row = build_retrieval_row(
            query=query,
            doc_id=doc_id,
            ranked_artifacts=ranked,
            candidate_count=len(candidate_artifacts),
            top_k=top_k,
            artifacts_hash=artifacts_hash,
            retrieval_method=retrieval_method,
        )
        assert_no_forbidden_public_fields(row)
        retrieval_rows.append(row)

    retrieval_path = output_root / "retrieval.jsonl"
    write_jsonl(retrieval_path, retrieval_rows)
    retrieval_hash = canonical_json_hash(retrieval_rows)

    quality_report = build_quality_report(
        retrieval_rows,
        top_k=top_k,
        retrieval_method=retrieval_method,
        graph_prior_enabled=bool(graph_prior),
        hybrid_settings=hybrid_settings,
    )
    assert_no_forbidden_public_fields(quality_report)
    quality_report_hash = canonical_json_hash(quality_report)
    quality_report_path = output_root / "quality_report.json"
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    manifest = build_manifest(
        artifacts_jsonl_path=artifacts_path,
        query_input_path=query_path,
        artifacts_hash=artifacts_hash,
        query_input_hash=query_input_hash,
        retrieval_hash=retrieval_hash,
        quality_report_hash=quality_report_hash,
        top_k=top_k,
        retrieval_method=retrieval_method,
        graph_path=graph_path,
        graph_edges_hash=graph_edges_hash,
        graph_prior_enabled=bool(graph_prior),
        hybrid_settings=hybrid_settings,
        hybrid_config_path=hybrid_config_path,
    )
    assert_no_forbidden_public_fields(manifest)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "retrieval_path": str(retrieval_path),
        "quality_report_path": str(quality_report_path),
        "manifest_path": str(manifest_path),
        "retrieval_hash": retrieval_hash,
        "quality_report": quality_report,
        "manifest": manifest,
    }


def load_query_records(path: str | Path) -> list[dict[str, Any]]:
    records = read_records(path)
    clean_records: list[dict[str, Any]] = []
    for inferred_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Query record at offset {inferred_index} is not a JSON object")
        clean_records.append(sanitize_query_record(record, inferred_index))
    return clean_records


def sanitize_query_record(record: dict[str, Any], inferred_index: int) -> dict[str, Any]:
    stage2 = record.get("stage2") if isinstance(record.get("stage2"), dict) else {}
    record_index = coerce_int(record.get("record_index", stage2.get("record_index")), inferred_index)
    clean: dict[str, Any] = {
        "record_index": record_index,
        "doc_id": record.get("doc_id"),
        "question": record.get("question") or record.get("query") or "",
    }
    for field in ("query_id", "record_id", "id"):
        if record.get(field) not in (None, ""):
            clean[field] = str(record.get(field))
    return clean


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
            artifact = dict(artifact)
            artifact["_input_order"] = len(artifacts)
            artifact["artifact_id"] = str(artifact.get("artifact_id") or f"artifact_{line_number:06d}")
            artifact["doc_id"] = str(artifact.get("doc_id") or "")
            artifacts.append(artifact)
    return artifacts


def group_artifacts_by_doc_id(artifacts: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        grouped[str(artifact.get("doc_id") or "")].append(artifact)
    for rows in grouped.values():
        rows.sort(key=lambda item: str(item.get("artifact_id") or ""))
    return dict(grouped)


def rank_artifacts(
    query_text: str,
    candidate_artifacts: list[dict[str, Any]],
    top_k: int,
    retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
    graph_prior: dict[str, float] | None = None,
    hybrid_weights: dict[str, float] | None = None,
) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
    if not candidate_artifacts:
        return []
    query_tokens = tokenize(query_text)
    artifact_tokens = [tokenize(artifact_retrieval_text(artifact)) for artifact in candidate_artifacts]
    lexical_scores = bm25_scores(query_tokens, artifact_tokens)
    scored: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    for artifact, lexical_score in zip(candidate_artifacts, lexical_scores):
        components = score_components(
            query_tokens=query_tokens,
            artifact=artifact,
            lexical_score=float(lexical_score),
            graph_prior=graph_prior or {},
            retrieval_method=retrieval_method,
        )
        total_score = components["lexical_score"]
        if retrieval_method == "deterministic_hybrid":
            weights = normalize_hybrid_weights(hybrid_weights or HYBRID_PRESET_WEIGHTS[DEFAULT_HYBRID_PRESET])
            total_score = sum(weights[name] * components[name] for name in COMPONENT_NAMES)
        scored.append((artifact, round(float(total_score), 8), {key: round(float(value), 8) for key, value in components.items()}))
    scored.sort(key=lambda item: (-item[1], str(item[0].get("artifact_id") or "")))
    return scored[:top_k]


def score_components(
    query_tokens: list[str],
    artifact: dict[str, Any],
    lexical_score: float,
    graph_prior: dict[str, float],
    retrieval_method: str,
) -> dict[str, float]:
    if retrieval_method == "deterministic_lexical":
        return {
            "lexical_score": float(lexical_score),
            "metadata_score": 0.0,
            "locator_score": 0.0,
            "type_modality_score": 0.0,
            "graph_prior_score": 0.0,
        }
    query_set = set(query_tokens)
    return {
        "lexical_score": float(lexical_score),
        "metadata_score": token_overlap_score(query_set, artifact_metadata_tokens(artifact)),
        "locator_score": locator_quality_score(artifact),
        "type_modality_score": token_overlap_score(query_set, tokenize(f"{artifact.get('artifact_type', '')} {artifact.get('modality', '')}")),
        "graph_prior_score": float(graph_prior.get(str(artifact.get("artifact_id") or ""), 0.0)),
    }


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
    avg_doc_length = sum(doc_lengths) / max(1, num_docs) or 1.0
    query_counter = Counter(query_tokens)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens, doc_length in zip(document_tokens, doc_lengths):
        token_counts = Counter(tokens)
        score = 0.0
        for token, query_count in query_counter.items():
            tf = token_counts.get(token, 0)
            if tf <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc_length / avg_doc_length)
            score += query_count * idf * (tf * (k1 + 1.0)) / denom
        scores.append(score)
    return scores


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(str(text).lower()) if token not in STOPWORDS]


def artifact_retrieval_text(artifact: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("artifact_type", "modality", "content"):
        if artifact.get(key) not in (None, ""):
            chunks.append(str(artifact.get(key)))
    normalized = artifact.get("normalized_content")
    if isinstance(normalized, dict):
        chunks.append(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
    elif normalized not in (None, ""):
        chunks.append(str(normalized))
    for anchor in artifact.get("source_anchors") or []:
        if isinstance(anchor, dict) and anchor.get("source_id"):
            chunks.append(str(anchor.get("source_id")))
    return "\n".join(chunks)


def artifact_metadata_tokens(artifact: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    for key in ("artifact_type", "modality", "artifact_id", "page_id", "status", "validation_status"):
        if artifact.get(key) not in (None, ""):
            chunks.append(str(artifact.get(key)))
    for key in ("normalized_content", "locators", "source_anchors", "provenance"):
        value = artifact.get(key)
        if value not in (None, "", []):
            chunks.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return tokenize("\n".join(chunks))


def token_overlap_score(query_tokens: set[str], candidate_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(candidate_tokens)) / len(query_tokens)


def locator_quality_score(artifact: dict[str, Any]) -> float:
    classification = classify_artifact_locator(artifact)
    score = 0.0
    if classification.get("source_anchored"):
        score += 0.20
    if classification.get("element_locatable"):
        score += 0.30
    if classification.get("proof_trace_eligible"):
        score += 0.50
    return score


def build_retrieval_row(
    query: dict[str, Any],
    doc_id: str,
    ranked_artifacts: list[tuple[dict[str, Any], float, dict[str, float]]],
    candidate_count: int,
    top_k: int,
    artifacts_hash: str,
    retrieval_method: str,
) -> dict[str, Any]:
    component_averages = average_components([components for _, _, components in ranked_artifacts])
    row: dict[str, Any] = {
        "record_index": int(query.get("record_index", -1)),
        "doc_id": doc_id,
        "retrieved_artifact_ids": [str(artifact.get("artifact_id")) for artifact, _, _ in ranked_artifacts],
        "retrieval_scores": [score for _, score, _ in ranked_artifacts],
        "retrieval_score_components": component_averages,
        "retrieval_method": retrieval_method,
        "top_k": int(top_k),
        "candidate_artifact_count": int(candidate_count),
        "used_debug_edges": False,
        "no_answer_generation": True,
        "no_gold_fields_used": True,
        "input_artifacts_hash": artifacts_hash,
        "query_hash": canonical_json_hash(query_public_identity(query)),
    }
    if query.get("query_id"):
        row["query_id"] = query["query_id"]
    if query.get("record_id"):
        row["record_id"] = query["record_id"]
    elif query.get("id"):
        row["query_id"] = query["id"]
    return row


def average_components(component_rows: list[dict[str, float]]) -> dict[str, float]:
    if not component_rows:
        return {name: 0.0 for name in COMPONENT_NAMES}
    return {name: round(sum(float(row.get(name, 0.0)) for row in component_rows) / len(component_rows), 8) for name in COMPONENT_NAMES}


def query_public_identity(query: dict[str, Any]) -> dict[str, Any]:
    return {
        key: query.get(key)
        for key in ("record_index", "query_id", "record_id", "id", "doc_id", "question")
        if query.get(key) not in (None, "")
    }


def build_quality_report(
    result_rows: list[dict[str, Any]],
    top_k: int,
    retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
    graph_prior_enabled: bool = False,
    hybrid_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    num_queries = len(result_rows)
    denominator = max(1, num_queries)
    with_artifacts = sum(1 for row in result_rows if int(row.get("candidate_artifact_count", 0)) > 0)
    total_candidates = sum(int(row.get("candidate_artifact_count", 0)) for row in result_rows)
    total_retrieved = sum(len(row.get("retrieved_artifact_ids", [])) for row in result_rows)
    settings = hybrid_settings or resolve_hybrid_settings(retrieval_method)
    weights = normalize_hybrid_weights(settings.get("weights", {}))
    component_names = active_scoring_components(retrieval_method, weights)
    component_averages = {
        name: sum(float(row.get("retrieval_score_components", {}).get(name, 0.0)) for row in result_rows) / denominator
        for name in COMPONENT_NAMES
    }
    nonzero_component_counts = {
        name: sum(1 for row in result_rows if float(row.get("retrieval_score_components", {}).get(name, 0.0)) > 0.0)
        for name in COMPONENT_NAMES
    }
    num_nonzero = sum(1 for row in result_rows if any(float(score) > 0.0 for score in row.get("retrieval_scores", [])))
    return {
        "num_queries": num_queries,
        "num_queries_with_doc_artifacts": with_artifacts,
        "num_queries_without_doc_artifacts": num_queries - with_artifacts,
        "artifact_coverage_rate": with_artifacts / denominator,
        "avg_candidate_artifacts_per_query": total_candidates / denominator,
        "avg_retrieved_artifacts_per_query": total_retrieved / denominator,
        "retrieval_method": retrieval_method,
        "scoring_components": component_names,
        "hybrid_preset": settings.get("preset"),
        "hybrid_weights": weights,
        "avg_lexical_score": component_averages["lexical_score"],
        "avg_metadata_score": component_averages["metadata_score"],
        "avg_locator_score": component_averages["locator_score"],
        "avg_graph_prior_score": component_averages["graph_prior_score"],
        "avg_type_modality_score": component_averages["type_modality_score"],
        "avg_component_scores": {name: component_averages[name] for name in COMPONENT_NAMES},
        "num_queries_with_nonzero_component_scores": nonzero_component_counts,
        "num_queries_with_nonzero_scores": num_nonzero,
        "graph_prior_enabled": bool(graph_prior_enabled),
        "top_k": int(top_k),
        "num_outputs_with_answer_field": 0,
        "num_gold_field_violations": 0,
        "used_debug_edges": False,
        "no_gold_fields_used": True,
    }


def resolve_hybrid_settings(
    retrieval_method: str,
    hybrid_config_path: str | Path | None = None,
    hybrid_preset: str | None = None,
) -> dict[str, Any]:
    config = load_hybrid_config(hybrid_config_path) if hybrid_config_path not in (None, "") else {}
    if retrieval_method == "deterministic_lexical":
        preset = "lexical_only"
        weights = dict(HYBRID_PRESET_WEIGHTS[preset])
    else:
        preset = str(hybrid_preset or config.get("hybrid_preset") or config.get("preset") or DEFAULT_HYBRID_PRESET)
        if preset not in HYBRID_PRESETS:
            raise ValueError(f"Unsupported hybrid_preset: {preset}")
        weights = dict(HYBRID_PRESET_WEIGHTS[preset])
        config_weights = config.get("weights")
        if isinstance(config_weights, dict):
            weights.update({str(key): float(value) for key, value in config_weights.items() if str(key) in COMPONENT_NAMES})
    weights = normalize_hybrid_weights(weights)
    return {
        "preset": preset,
        "weights": weights,
        "config_path": public_path(hybrid_config_path) if hybrid_config_path not in (None, "") else None,
        "config_hash": file_sha256(hybrid_config_path) if hybrid_config_path not in (None, "") else None,
    }


def normalize_hybrid_weights(weights: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name in COMPONENT_NAMES:
        try:
            normalized[name] = float(weights.get(name, 0.0))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid hybrid weight for {name}: {weights.get(name)!r}") from None
    return normalized


def active_scoring_components(retrieval_method: str, weights: dict[str, float]) -> list[str]:
    if retrieval_method == "deterministic_lexical":
        return ["lexical_score"]
    return [name for name in COMPONENT_NAMES if float(weights.get(name, 0.0)) != 0.0]


def load_hybrid_config(path: str | Path) -> dict[str, Any]:
    """Load the small retrieval YAML subset used by checked-in configs."""

    config_path = Path(path)
    result: dict[str, Any] = {}
    current_map: str | None = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_map = line[:-1].strip()
            result[current_map] = {}
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported hybrid config line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        target = result[current_map] if current_map and raw_line.startswith((" ", "\t")) else result
        target[key] = parse_config_scalar(value)
    return result


def parse_config_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_graph_prior(graph_path: str | Path | None) -> tuple[dict[str, float], str | None]:
    if graph_path in (None, ""):
        return {}, None
    edges_path = resolve_graph_edges_path(graph_path)
    if not edges_path.is_file():
        return {}, "missing"
    rows = read_jsonl(edges_path)
    degree: Counter[str] = Counter()
    for edge in rows:
        edge_type = str(edge.get("edge_type") or "")
        if edge.get("debug") is True or edge_type in SEMANTIC_EDGE_TYPES or edge_type in DEBUG_EDGE_TYPES:
            continue
        left = parse_artifact_node_id(edge.get("source"))
        right = parse_artifact_node_id(edge.get("target"))
        if left:
            degree[left] += 1
        if right:
            degree[right] += 1
    max_degree = max(degree.values(), default=0)
    if max_degree <= 0:
        return {}, file_sha256(edges_path)
    return {artifact_id: count / max_degree for artifact_id, count in sorted(degree.items())}, file_sha256(edges_path)


def resolve_graph_edges_path(graph_path: str | Path) -> Path:
    path = Path(graph_path)
    if path.is_dir():
        return path / "edges.jsonl"
    return path


def parse_artifact_node_id(node_id: Any) -> str | None:
    parts = str(node_id or "").split(":", 3)
    if len(parts) != 4 or parts[0] != "artifact":
        return None
    return unquote(parts[3])


def build_manifest(
    artifacts_jsonl_path: str | Path,
    query_input_path: str | Path,
    artifacts_hash: str,
    query_input_hash: str,
    retrieval_hash: str,
    quality_report_hash: str,
    top_k: int,
    retrieval_method: str,
    graph_path: str | Path | None,
    graph_edges_hash: str | None,
    graph_prior_enabled: bool,
    hybrid_settings: dict[str, Any],
    hybrid_config_path: str | Path | None = None,
) -> dict[str, Any]:
    commit = current_git_commit()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "stage3_doc_artifact_retrieval",
        "retrieval_mode": "query_conditioned_over_document_generic_artifacts",
        "retrieval_method": retrieval_method,
        "hybrid_preset": hybrid_settings.get("preset"),
        "hybrid_weights": normalize_hybrid_weights(hybrid_settings.get("weights", {})),
        "hybrid_config_path": public_path(hybrid_config_path) if hybrid_config_path not in (None, "") else None,
        "hybrid_config_hash": hybrid_settings.get("config_hash"),
        "input_artifacts_path": public_path(artifacts_jsonl_path),
        "input_artifacts_hash": artifacts_hash,
        "query_input_path": public_path(query_input_path),
        "query_input_hash": query_input_hash,
        "retrieval_hash": retrieval_hash,
        "quality_report_hash": quality_report_hash,
        "graph_prior_enabled": bool(graph_prior_enabled),
        "graph_prior_requested": graph_path not in (None, ""),
        "graph_edges_path": public_path(resolve_graph_edges_path(graph_path)) if graph_path not in (None, "") else None,
        "graph_edges_hash": graph_edges_hash,
        "no_answer_generation": True,
        "no_gold_fields_used": True,
        "used_debug_edges": False,
        "created_by_script": "scripts/stage3_doc_artifact_retrieval.py",
        "command_args": {
            "artifacts_jsonl": public_path(artifacts_jsonl_path),
            "query_input": public_path(query_input_path),
            "top_k": int(top_k),
            "retrieval_method": retrieval_method,
            "hybrid_preset": hybrid_settings.get("preset"),
            "hybrid_config_used": hybrid_config_path not in (None, ""),
            "graph_prior_enabled": bool(graph_prior_enabled),
        },
    }
    if commit == "unknown":
        manifest["git_commit_unavailable_reason"] = "git_rev_parse_failed"
    else:
        manifest["git_commit"] = commit
    return {key: value for key, value in manifest.items() if value is not None}


def read_records(path: str | Path) -> list[Any]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        records: list[Any] = []
        with input_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.strip():
                    records.append(json.loads(line))
        return records
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items", "queries"):
            records = value.get(key)
            if isinstance(records, list):
                return records
    raise ValueError(f"Expected query records in {input_path}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assert_no_forbidden_public_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text not in SAFE_PUBLIC_FIELD_NAMES and key_text in FORBIDDEN_PUBLIC_FIELD_NAMES:
                raise ValueError(f"Forbidden public field present: {key_text}")
            assert_no_forbidden_public_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_public_fields(child)


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = payload.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path_obj.name


def current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3 document-generic artifact retrieval dry-run.")
    parser.add_argument("--artifacts", "--artifacts-jsonl", dest="artifacts_jsonl", default=DEFAULT_ARTIFACTS_JSONL)
    parser.add_argument("--queries", "--query-input", "--records-jsonl", dest="query_input", default=DEFAULT_QUERY_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--retrieval-method", choices=sorted(RETRIEVAL_METHODS), default=DEFAULT_RETRIEVAL_METHOD)
    parser.add_argument("--graph", default=None, help="Optional Stage 4 graph directory or formal edges.jsonl for deterministic graph prior.")
    parser.add_argument("--hybrid-config", default=None, help="Optional fixed YAML weights for deterministic_hybrid scoring.")
    parser.add_argument("--hybrid-preset", choices=sorted(HYBRID_PRESETS), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_doc_artifact_retrieval(
        artifacts_jsonl_path=args.artifacts_jsonl,
        query_input_path=args.query_input,
        output_dir=args.output_dir,
        top_k=args.top_k,
        retrieval_method=args.retrieval_method,
        graph_path=args.graph,
        hybrid_config_path=args.hybrid_config,
        hybrid_preset=args.hybrid_preset,
    )
    print(json.dumps(result["quality_report"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
