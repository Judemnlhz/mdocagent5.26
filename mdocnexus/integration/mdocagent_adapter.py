"""Deterministic MDocAgent-compatible retrieval adapter.

This module connects Stage 2/3/4 diagnostic artifacts back to the original
MDocAgent prediction path by rewriting only sample-with-retrieval-results
records. It deliberately avoids answer generation, evaluation labels, debug
edges, semantic edges, and evaluator models.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping


DEFAULT_TOP_K = 4
DEFAULT_LAMBDA_WEIGHT = 0.5
ADAPTER_SCHEMA_VERSION = "mdocagent_adapter_v1"

RERANK_MODES = {"original_only", "artifact_only", "original_plus_artifact"}
ADAPT_MODES = RERANK_MODES | {"graph_context"}
EXPANSION_MODES = {"page_neighborhood", "source_anchor_neighborhood", "direct_structural"}

GOLD_FIELD_NAMES = {
    "answer",
    "answers",
    "gold_answer",
    "evidence_pages",
    "evidence_sources",
    "binary_correctness",
    "gold_evidence",
    "gold_page",
    "gold_pages",
}
PRIVATE_FIELD_NAMES = {
    "raw_output",
    "raw_response",
    "provider_response",
    "raw_outputs",
    "api_key",
    "api_token",
    "access_token",
    "secret",
    "secret_token",
    "local_path",
    "absolute_path",
    "image_path",
    "page_image_path",
    "page_text_path",
    "file_url",
    "base64",
    "image_base64",
    "base64_payload",
    "image_payload_base64",
}
FORBIDDEN_FIELD_NAMES = GOLD_FIELD_NAMES | PRIVATE_FIELD_NAMES
FORBIDDEN_TEXT_FRAGMENTS = ("file://", "/home/", "data:image")

TEXT_TOP_RE = re.compile(r"^text-top-\d+-.+")
IMAGE_TOP_RE = re.compile(r"^image-top-\d+-.+")
MIX_TOP_RE = re.compile(r"^mix-top-\d+-.+")
TOP_K_RE = re.compile(r"-top-(\d+)-")
TOKEN_RE = re.compile(r"[a-z0-9%]+")
PAGE_NODE_RE = re.compile(r"^page:(?P<doc>.*):(?P<page>-?\d+)$")
SOURCE_ANCHOR_NODE_RE = re.compile(r"^source_anchor:(?P<doc>.*):(?P<page>-?\d+):")
ARTIFACT_NODE_RE = re.compile(r"^artifact:(?P<doc>.*):(?P<page>-?\d+):(?P<artifact>.+)$")

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

SEMANTIC_EDGE_TYPES = {
    "supports",
    "contradicts",
    "derived_from",
    "semantic_relation",
    "entails",
    "refutes",
    "answer_supports",
    "proof_supports",
    "cites",
}
DEBUG_EDGE_TYPES = {"same_record", "same_record_debug"}
FORMAL_EDGE_TYPES = {
    "located_on_page",
    "supported_by_anchor",
    "anchor_on_page",
    "adjacent_page",
    "next_block",
    "section_contains",
    "caption_of",
    "figure_has_caption",
    "table_contains_cell",
    "row_contains_cell",
    "column_contains_cell",
}


def load_mdocagent_retrieval_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and sanitize MDocAgent sample-with-retrieval-results records.

    Only the public query identity and retrieval page/rank/score fields required
    by ``BaseDataset.load_sample_retrieval_data`` are propagated.
    """

    rows = read_records(path)
    records: list[dict[str, Any]] = []
    for inferred_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Retrieval record at offset {inferred_index} is not a JSON object")
        records.append(sanitize_retrieval_record(row, inferred_index))
    return records


def load_artifacts_by_page(path: str | Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Load Stage 2 artifacts indexed by doc_id and page index.

    Question, answer, gold, raw, and local-path fields are ignored. Missing
    artifact IDs are generated from page-local artifact content rather than
    record or question identifiers.
    """

    rows = read_records(path)
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for input_order, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Artifact record at offset {input_order} is not a JSON object")
        artifact = sanitize_artifact_record(row)
        doc_id = str(artifact.get("doc_id") or "")
        page_index = coerce_page_index(artifact.get("page_index", artifact.get("page_id")))
        if not doc_id or page_index is None:
            continue
        artifact["page_index"] = int(page_index)
        artifact.setdefault("page_id", f"page:{doc_id}:{page_index}")
        artifact["artifact_id"] = str(artifact.get("artifact_id") or generated_artifact_id(artifact))
        grouped[doc_id][int(page_index)].append(artifact)

    deterministic: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for doc_id in sorted(grouped):
        deterministic[doc_id] = {}
        for page_index in sorted(grouped[doc_id]):
            deterministic[doc_id][page_index] = sorted(
                grouped[doc_id][page_index],
                key=lambda item: (
                    str(item.get("artifact_id") or ""),
                    canonical_json_hash(item),
                ),
            )
    return deterministic


def rerank_pages_with_artifacts(
    records: Iterable[dict[str, Any]],
    artifacts_by_page: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    top_k: int = DEFAULT_TOP_K,
    mode: str = "original_plus_artifact",
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
) -> list[dict[str, Any]]:
    """Return MDocAgent-compatible records with artifact-aware page order."""

    if mode not in RERANK_MODES:
        raise ValueError(f"Unsupported rerank mode: {mode}")
    top_k = validate_top_k(top_k)
    lambda_weight = validate_lambda(lambda_weight)
    output: list[dict[str, Any]] = []
    for record in records:
        clean_record = sanitize_retrieval_record(record, 0)
        if mode == "original_only":
            output.append(truncate_original_record(clean_record, top_k, mode=mode, lambda_weight=lambda_weight))
            continue

        doc_id = str(clean_record.get("doc_id") or "")
        original_scores, original_order = original_page_scores(clean_record)
        doc_artifact_pages = set(artifacts_by_page.get(doc_id, {}).keys())
        candidate_pages = sorted(set(original_order) | doc_artifact_pages)
        if not candidate_pages:
            output.append(truncate_original_record(clean_record, top_k, mode=mode, lambda_weight=lambda_weight))
            continue

        artifact_scores = artifact_page_scores(str(clean_record.get("question") or ""), doc_id, candidate_pages, artifacts_by_page)
        scored_pages: list[tuple[int, float, float, float]] = []
        for page_index in candidate_pages:
            original_score = float(original_scores.get(page_index, 0.0))
            artifact_score = float(artifact_scores.get(page_index, 0.0))
            final_score = artifact_score if mode == "artifact_only" else lambda_weight * original_score + (1.0 - lambda_weight) * artifact_score
            scored_pages.append((int(page_index), round(final_score, 8), round(original_score, 8), round(artifact_score, 8)))
        scored_pages.sort(key=lambda item: (-item[1], doc_id, item[0]))
        selected = scored_pages[:top_k]
        output.append(
            apply_selected_pages(
                clean_record,
                selected_pages=[page for page, _, _, _ in selected],
                selected_scores=[score for _, score, _, _ in selected],
                meta={
                    "mode": mode,
                    "top_k": top_k,
                    "lambda_weight": lambda_weight,
                    "formula": formula_for_mode(mode),
                    "same_page_budget_as_baseline": True,
                    "no_gold_fields_used": True,
                    "used_debug_edges": False,
                    "used_semantic_edges": False,
                    "model_role": "none_deterministic",
                },
            )
        )
    return output


def select_pages_with_graph(
    records: Iterable[dict[str, Any]],
    artifacts_by_page: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    graph_dir: str | Path,
    top_k: int = DEFAULT_TOP_K,
    expansion_mode: str = "page_neighborhood",
) -> list[dict[str, Any]]:
    """Select final top-k pages using formal graph edges only."""

    if expansion_mode not in EXPANSION_MODES:
        raise ValueError(f"Unsupported expansion_mode: {expansion_mode}")
    top_k = validate_top_k(top_k)
    graph = load_formal_graph(graph_dir)
    output: list[dict[str, Any]] = []
    for record in records:
        clean_record = sanitize_retrieval_record(record, 0)
        doc_id = str(clean_record.get("doc_id") or "")
        original_scores, original_order = original_page_scores(clean_record)
        seed_pages = original_order[:top_k] or sorted(artifacts_by_page.get(doc_id, {}).keys())[:top_k]
        doc_artifact_pages = set(artifacts_by_page.get(doc_id, {}).keys())
        artifact_scores = artifact_page_scores(
            str(clean_record.get("question") or ""),
            doc_id,
            sorted(set(seed_pages) | doc_artifact_pages),
            artifacts_by_page,
        )
        candidate_scores: dict[int, float] = {page: float(original_scores.get(page, 0.0)) for page in seed_pages}

        if expansion_mode == "page_neighborhood":
            expand_page_neighborhood(candidate_scores, seed_pages, doc_id, graph, artifact_scores)
        elif expansion_mode == "source_anchor_neighborhood":
            expand_source_anchor_neighborhood(candidate_scores, seed_pages, doc_id, artifacts_by_page, graph, artifact_scores)
        else:
            expand_direct_structural(candidate_scores, seed_pages, doc_id, graph, artifact_scores, original_scores)

        for page_index in seed_pages:
            candidate_scores.setdefault(page_index, float(original_scores.get(page_index, 0.0)))
        if not candidate_scores:
            output.append(truncate_original_record(clean_record, top_k, mode="graph_context"))
            continue

        scored_pages = [
            (page_index, round(float(score), 8))
            for page_index, score in candidate_scores.items()
            if isinstance(page_index, int)
        ]
        scored_pages.sort(key=lambda item: (-item[1], doc_id, item[0]))
        selected = scored_pages[:top_k]
        output.append(
            apply_selected_pages(
                clean_record,
                selected_pages=[page for page, _ in selected],
                selected_scores=[score for _, score in selected],
                meta={
                    "mode": "graph_context",
                    "top_k": top_k,
                    "expansion_mode": expansion_mode,
                    "formula": formula_for_mode("graph_context"),
                    "same_page_budget_as_baseline": True,
                    "no_gold_fields_used": True,
                    "used_debug_edges": False,
                    "used_semantic_edges": False,
                    "formal_edges_path": "edges.jsonl",
                    "model_role": "none_deterministic",
                },
            )
        )
    return output


def write_mdocagent_compatible_records(records: Iterable[dict[str, Any]], output_path: str | Path) -> str:
    """Write sanitized sample-with-retrieval-results-nexus JSON records."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [sanitize_output_record(row) for row in records]
    assert_no_forbidden_public_fields(rows)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return canonical_json_hash(rows)


def build_mdocagent_adapter_manifest(
    *,
    mode: str,
    top_k: int,
    lambda_weight: float,
    input_retrieval: str | Path,
    artifacts: str | Path,
    output_retrieval: str | Path,
    graph_dir: str | Path | None = None,
    expansion_mode: str | None = None,
    command_args: Mapping[str, Any] | argparse.Namespace | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the public deterministic adapter manifest."""

    repo = Path(repo_root) if repo_root not in (None, "") else Path.cwd()
    graph_hash = graph_edges_hash(graph_dir) if graph_dir not in (None, "") else None
    git_commit = current_git_commit(repo)
    manifest: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "mode": mode,
        "top_k": validate_top_k(top_k),
        "lambda_weight": validate_lambda(lambda_weight),
        "same_page_budget_as_baseline": True,
        "no_gold_fields_used": True,
        "used_debug_edges": False,
        "used_semantic_edges": False,
        "formula": formula_for_mode(mode),
        "input_retrieval_hash": canonical_file_hash(input_retrieval),
        "artifacts_hash": canonical_file_hash(artifacts),
        "graph_hash": graph_hash,
        "output_hash": canonical_file_hash(output_retrieval),
        "model_role": "none_deterministic",
        "evaluator_model_used": False,
        "model_config_hash": combined_model_config_hash(repo),
        "command_args": sanitize_command_args(command_args or {}, repo),
    }
    if expansion_mode:
        manifest["expansion_mode"] = expansion_mode
    if git_commit:
        manifest["git_commit"] = git_commit
    else:
        manifest["git_commit_unavailable_reason"] = "git_command_failed"
    assert_no_forbidden_public_fields(manifest)
    return manifest


def write_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_manifest = sanitize_public_value(dict(manifest))
    assert_no_forbidden_public_fields(clean_manifest)
    output.write_text(json.dumps(clean_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return canonical_json_hash(clean_manifest)


def sanitize_retrieval_record(record: Mapping[str, Any], inferred_index: int) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    if record.get("record_index") not in (None, ""):
        clean["record_index"] = coerce_int(record.get("record_index"), inferred_index)
    for key in ("query_id", "record_id", "id"):
        if record.get(key) not in (None, "") and not is_forbidden_key(key):
            clean[key] = str(record.get(key))
    clean["doc_id"] = str(record.get("doc_id") or "")
    clean["question"] = record.get("question") if record.get("question") is not None else ""

    page_keys = [key for key in record.keys() if is_retrieval_page_key(key)]
    for key in sorted(page_keys, key=retrieval_key_sort):
        clean[key] = parse_page_list(record.get(key), key)
        score_key = f"{key}_score"
        if score_key in record:
            clean[score_key] = parse_score_list(record.get(score_key), score_key)
    assert_no_forbidden_public_fields(clean)
    return clean


def sanitize_artifact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "artifact_id",
        "artifact_type",
        "content",
        "doc_id",
        "locators",
        "modality",
        "normalized_content",
        "page_id",
        "page_index",
        "provenance",
        "source_anchors",
        "status",
        "validation_status",
    }
    clean: dict[str, Any] = {}
    for key in sorted(allowed_keys):
        if key in record and not is_forbidden_key(key):
            clean_value = sanitize_public_value(record.get(key))
            if clean_value is not None:
                clean[key] = clean_value
    return clean


def sanitize_output_record(record: Mapping[str, Any]) -> dict[str, Any]:
    clean = sanitize_public_value(dict(record))
    if not isinstance(clean, dict):
        raise ValueError("Output record must be a mapping")
    assert_no_forbidden_public_fields(clean)
    return clean


def truncate_original_record(
    record: Mapping[str, Any],
    top_k: int,
    *,
    mode: str,
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
) -> dict[str, Any]:
    top_k = validate_top_k(top_k)
    output = {key: value for key, value in record.items() if not is_retrieval_page_key(key) and not is_retrieval_score_key(key)}
    page_keys = retrieval_page_keys(record)
    if not page_keys:
        page_keys = default_retrieval_page_keys()
    for key in page_keys:
        pages = parse_page_list(record.get(key, []), key)[:top_k]
        output[key] = pages
        score_key = f"{key}_score"
        scores = parse_score_list(record.get(score_key, []), score_key)[: len(pages)]
        if not scores:
            scores = [round(1.0 / float(rank + 1), 8) for rank in range(len(pages))]
        output[score_key] = scores
    output["_nexus_meta"] = {
        "mode": mode,
        "top_k": top_k,
        "lambda_weight": validate_lambda(lambda_weight),
        "formula": formula_for_mode(mode),
        "same_page_budget_as_baseline": True,
        "no_gold_fields_used": True,
        "used_debug_edges": False,
        "used_semantic_edges": False,
        "model_role": "none_deterministic",
    }
    assert_no_forbidden_public_fields(output)
    return output


def apply_selected_pages(
    record: Mapping[str, Any],
    *,
    selected_pages: list[int],
    selected_scores: list[float],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    output = {key: value for key, value in record.items() if not is_retrieval_page_key(key) and not is_retrieval_score_key(key)}
    page_keys = retrieval_page_keys(record) or default_retrieval_page_keys()
    pages = [int(page) for page in selected_pages]
    scores = [round(float(score), 8) for score in selected_scores[: len(pages)]]
    while len(scores) < len(pages):
        scores.append(round(1.0 / float(len(scores) + 1), 8))
    for key in page_keys:
        output[key] = list(pages)
        output[f"{key}_score"] = list(scores)
    output["_nexus_meta"] = sanitize_public_value(dict(meta))
    assert_no_forbidden_public_fields(output)
    return output


def original_page_scores(record: Mapping[str, Any]) -> tuple[dict[int, float], list[int]]:
    observations: dict[int, float] = {}
    first_rank: dict[int, int] = {}
    page_order: list[int] = []
    running_rank = 0
    any_explicit_score = False
    for key in retrieval_page_keys(record):
        pages = parse_page_list(record.get(key, []), key)
        scores = parse_score_list(record.get(f"{key}_score", []), f"{key}_score")
        for rank, page_index in enumerate(pages):
            if page_index not in first_rank:
                first_rank[page_index] = running_rank
                page_order.append(page_index)
            running_rank += 1
            if rank < len(scores):
                score = float(scores[rank])
                any_explicit_score = True
            else:
                score = 1.0 / float(rank + 1)
            observations[page_index] = max(float(observations.get(page_index, float("-inf"))), score)
    if not observations:
        return {}, []
    if any_explicit_score:
        max_abs = max(abs(score) for score in observations.values()) or 1.0
        normalized = {page: round(max(0.0, score / max_abs), 8) for page, score in observations.items()}
    else:
        normalized = {page: round(1.0 / float(first_rank[page] + 1), 8) for page in observations}
    page_order.sort(key=lambda page: first_rank[page])
    return normalized, page_order


def artifact_page_scores(
    question: str,
    doc_id: str,
    candidate_pages: Iterable[int],
    artifacts_by_page: Mapping[str, Mapping[int, list[dict[str, Any]]]],
) -> dict[int, float]:
    pages = sorted({int(page) for page in candidate_pages})
    if not pages:
        return {}
    artifacts: list[dict[str, Any]] = []
    page_for_artifact: list[int] = []
    doc_pages = artifacts_by_page.get(doc_id, {})
    for page_index in pages:
        for artifact in doc_pages.get(page_index, []):
            artifacts.append(artifact)
            page_for_artifact.append(page_index)
    if not artifacts:
        return {page: 0.0 for page in pages}
    query_tokens = tokenize(question)
    document_tokens = [tokenize(artifact_retrieval_text(artifact)) for artifact in artifacts]
    artifact_scores = bm25_scores(query_tokens, document_tokens)
    raw_page_scores: dict[int, float] = {page: 0.0 for page in pages}
    for page_index, score in zip(page_for_artifact, artifact_scores):
        raw_page_scores[page_index] = max(raw_page_scores.get(page_index, 0.0), float(score))
    max_score = max(raw_page_scores.values(), default=0.0)
    if max_score <= 0.0:
        return {page: 0.0 for page in pages}
    return {page: round(float(score) / max_score, 8) for page, score in raw_page_scores.items()}


def load_formal_graph(graph_dir: str | Path) -> dict[str, Any]:
    edges_path = Path(graph_dir) / "edges.jsonl"
    edges: list[dict[str, Any]] = []
    if edges_path.is_file():
        for edge in read_jsonl(edges_path):
            if not isinstance(edge, dict):
                continue
            edge_type = str(edge.get("edge_type") or "")
            if edge_type in DEBUG_EDGE_TYPES or edge_type in SEMANTIC_EDGE_TYPES:
                continue
            if edge_type not in FORMAL_EDGE_TYPES:
                continue
            edges.append(edge)

    page_neighbors: dict[tuple[str, int], set[int]] = defaultdict(set)
    artifact_pages: dict[str, set[tuple[str, int]]] = defaultdict(set)
    artifact_anchors: dict[str, set[str]] = defaultdict(set)
    anchor_pages: dict[str, set[tuple[str, int]]] = defaultdict(set)
    structural_pages: dict[str, set[int]] = defaultdict(set)

    for edge in edges:
        edge_type = str(edge.get("edge_type") or "")
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        pages = edge_pages(edge)
        for doc_id, page_index in pages:
            structural_pages[doc_id].add(page_index)
        if edge_type == "adjacent_page":
            for left_doc, left_page in pages:
                for right_doc, right_page in pages:
                    if left_doc == right_doc and left_page != right_page:
                        page_neighbors[(left_doc, left_page)].add(right_page)
        if edge_type == "located_on_page":
            artifact_id = parse_artifact_node(source)
            page_ref = parse_page_node(target) or first_page_from_edge(edge)
            if artifact_id and page_ref:
                artifact_pages[artifact_id].add(page_ref)
        if edge_type == "supported_by_anchor":
            artifact_id = parse_artifact_node(source)
            if artifact_id and target.startswith("source_anchor:"):
                artifact_anchors[artifact_id].add(target)
        if edge_type == "anchor_on_page":
            page_ref = parse_page_node(target) or first_page_from_edge(edge)
            if page_ref:
                anchor_pages[source].add(page_ref)

    return {
        "edges": edges,
        "page_neighbors": {key: sorted(value) for key, value in page_neighbors.items()},
        "artifact_pages": {key: sorted(value) for key, value in artifact_pages.items()},
        "artifact_anchors": {key: sorted(value) for key, value in artifact_anchors.items()},
        "anchor_pages": {key: sorted(value) for key, value in anchor_pages.items()},
        "structural_pages": {key: sorted(value) for key, value in structural_pages.items()},
        "used_debug_edges": False,
        "used_semantic_edges": False,
    }


def expand_page_neighborhood(
    candidate_scores: dict[int, float],
    seed_pages: list[int],
    doc_id: str,
    graph: Mapping[str, Any],
    artifact_scores: Mapping[int, float],
) -> None:
    page_neighbors = graph.get("page_neighbors", {})
    for seed in seed_pages:
        seed_score = float(candidate_scores.get(seed, 0.0))
        for neighbor in page_neighbors.get((doc_id, seed), []):
            score = 0.85 * seed_score + 0.15 * float(artifact_scores.get(neighbor, 0.0))
            candidate_scores[int(neighbor)] = max(float(candidate_scores.get(int(neighbor), 0.0)), round(score, 8))


def expand_source_anchor_neighborhood(
    candidate_scores: dict[int, float],
    seed_pages: list[int],
    doc_id: str,
    artifacts_by_page: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    graph: Mapping[str, Any],
    artifact_scores: Mapping[int, float],
) -> None:
    artifact_anchors = graph.get("artifact_anchors", {})
    anchor_pages = graph.get("anchor_pages", {})
    artifact_pages = graph.get("artifact_pages", {})
    doc_pages = artifacts_by_page.get(doc_id, {})
    for seed in seed_pages:
        seed_score = float(candidate_scores.get(seed, 0.0))
        for artifact in doc_pages.get(seed, []):
            artifact_id = str(artifact.get("artifact_id") or "")
            linked_pages = set(artifact_pages.get(artifact_id, []))
            for anchor in artifact_anchors.get(artifact_id, []):
                linked_pages.update(anchor_pages.get(anchor, []))
            for linked_doc, linked_page in linked_pages:
                if linked_doc != doc_id:
                    continue
                score = 0.80 * seed_score + 0.20 * float(artifact_scores.get(int(linked_page), 0.0))
                candidate_scores[int(linked_page)] = max(float(candidate_scores.get(int(linked_page), 0.0)), round(score, 8))


def expand_direct_structural(
    candidate_scores: dict[int, float],
    seed_pages: list[int],
    doc_id: str,
    graph: Mapping[str, Any],
    artifact_scores: Mapping[int, float],
    original_scores: Mapping[int, float],
) -> None:
    structural_pages = graph.get("structural_pages", {}).get(doc_id, [])
    candidate_pool = sorted(set(seed_pages) | set(structural_pages))
    for page_index in candidate_pool:
        score = 0.50 * float(original_scores.get(int(page_index), 0.0)) + 0.50 * float(artifact_scores.get(int(page_index), 0.0))
        if int(page_index) in seed_pages:
            score = max(score, float(original_scores.get(int(page_index), 0.0)))
        candidate_scores[int(page_index)] = max(float(candidate_scores.get(int(page_index), 0.0)), round(score, 8))


def edge_pages(edge: Mapping[str, Any]) -> list[tuple[str, int]]:
    pages: list[tuple[str, int]] = []
    for key in ("source", "target"):
        parsed = parse_page_node(edge.get(key)) or parse_source_anchor_node(edge.get(key))
        if parsed:
            pages.append(parsed)
    provenance = edge.get("provenance") if isinstance(edge.get("provenance"), dict) else {}
    doc_id = str(provenance.get("doc_id") or "")
    for page_key in ("page_index", "source_page_index", "target_page_index"):
        if doc_id and provenance.get(page_key) not in (None, ""):
            page_index = coerce_page_index(provenance.get(page_key))
            if page_index is not None:
                pages.append((doc_id, int(page_index)))
    return sorted(set(pages), key=lambda item: (item[0], item[1]))


def first_page_from_edge(edge: Mapping[str, Any]) -> tuple[str, int] | None:
    pages = edge_pages(edge)
    return pages[0] if pages else None


def parse_page_node(value: Any) -> tuple[str, int] | None:
    match = PAGE_NODE_RE.match(str(value or ""))
    if not match:
        return None
    return match.group("doc"), int(match.group("page"))


def parse_source_anchor_node(value: Any) -> tuple[str, int] | None:
    match = SOURCE_ANCHOR_NODE_RE.match(str(value or ""))
    if not match:
        return None
    return match.group("doc"), int(match.group("page"))


def parse_artifact_node(value: Any) -> str | None:
    match = ARTIFACT_NODE_RE.match(str(value or ""))
    if match:
        return match.group("artifact")
    return None


def retrieval_page_keys(record: Mapping[str, Any]) -> list[str]:
    return sorted([key for key in record.keys() if is_retrieval_page_key(key)], key=retrieval_key_sort)


def default_retrieval_page_keys() -> list[str]:
    return ["text-top-10-question", "image-top-10-question"]


def is_retrieval_page_key(key: Any) -> bool:
    key_text = str(key)
    if is_forbidden_key(key_text) or key_text.endswith("_score"):
        return False
    return bool(TEXT_TOP_RE.match(key_text) or IMAGE_TOP_RE.match(key_text) or MIX_TOP_RE.match(key_text))


def is_retrieval_score_key(key: Any) -> bool:
    key_text = str(key)
    if not key_text.endswith("_score"):
        return False
    return is_retrieval_page_key(key_text[: -len("_score")])


def retrieval_key_sort(key: Any) -> tuple[int, int, str]:
    key_text = str(key)
    if key_text.startswith("text-"):
        family = 0
    elif key_text.startswith("image-"):
        family = 1
    elif key_text.startswith("mix-"):
        family = 2
    else:
        family = 3
    match = TOP_K_RE.search(key_text)
    top = int(match.group(1)) if match else 0
    return (family, -top, key_text)


def parse_page_list(value: Any, field_name: str) -> list[int]:
    values = parse_sequence(value, field_name)
    pages: list[int] = []
    for item in values:
        page_index = coerce_page_index(item)
        if page_index is not None:
            pages.append(int(page_index))
    return pages


def parse_score_list(value: Any, field_name: str) -> list[float]:
    values = parse_sequence(value, field_name)
    scores: list[float] = []
    for item in values:
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            continue
    return scores


def parse_sequence(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            raise ValueError(f"Cannot parse list field {field_name!r}") from None
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    raise ValueError(f"Cannot parse list field {field_name!r}: {value!r}")


def coerce_page_index(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def validate_top_k(top_k: int) -> int:
    value = int(top_k)
    if value < 1:
        raise ValueError("top_k must be at least 1")
    return value


def validate_lambda(lambda_weight: float) -> float:
    value = float(lambda_weight)
    if value < 0.0 or value > 1.0:
        raise ValueError("lambda_weight must be between 0 and 1")
    return value


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(str(text).lower()) if token not in STOPWORDS]


def artifact_retrieval_text(artifact: Mapping[str, Any]) -> str:
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


def generated_artifact_id(artifact: Mapping[str, Any]) -> str:
    identity = {
        key: artifact.get(key)
        for key in (
            "doc_id",
            "page_index",
            "page_id",
            "artifact_type",
            "modality",
            "content",
            "normalized_content",
            "source_anchors",
            "locators",
        )
        if artifact.get(key) not in (None, "", [], {})
    }
    return f"artifact_{canonical_json_hash(identity)[:16]}"


def formula_for_mode(mode: str) -> str:
    if mode == "original_only":
        return "original_retrieval_order"
    if mode == "artifact_only":
        return "artifact_score=max_bm25(question,page_artifacts)"
    if mode == "original_plus_artifact":
        return "final_score=lambda_weight*normalized_original_score+(1-lambda_weight)*normalized_artifact_score"
    if mode == "graph_context":
        return "formal_edges_only_graph_expansion_then_top_k_selection"
    return "unknown"


def read_records(path: str | Path) -> list[Any]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        return read_jsonl(input_path)
    value = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items", "queries"):
            if isinstance(value.get(key), list):
                return list(value[key])
    raise ValueError(f"Expected a JSON list or JSONL records in {input_path}")


def read_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def canonical_file_hash(path: str | Path) -> str:
    target = Path(path)
    if target.suffix.lower() == ".jsonl":
        return canonical_json_hash(read_jsonl(target))
    if target.suffix.lower() == ".json":
        return canonical_json_hash(json.loads(target.read_text(encoding="utf-8")))
    return file_sha256(target)


def graph_edges_hash(graph_dir: str | Path | None) -> str | None:
    if graph_dir in (None, ""):
        return None
    edges_path = Path(graph_dir) / "edges.jsonl"
    if not edges_path.is_file():
        return "missing"
    formal_edges = []
    for edge in read_jsonl(edges_path):
        if not isinstance(edge, dict):
            continue
        edge_type = str(edge.get("edge_type") or "")
        if edge_type in FORMAL_EDGE_TYPES and edge_type not in DEBUG_EDGE_TYPES and edge_type not in SEMANTIC_EDGE_TYPES:
            formal_edges.append(edge)
    return canonical_json_hash(formal_edges)


def canonical_json_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    serialized = serialized.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_model_config_hash(repo_root: str | Path | None = None) -> str:
    repo = Path(repo_root) if repo_root not in (None, "") else Path.cwd()
    config_paths = [
        repo / "config/model/deepseekv3.yaml",
        repo / "config/model/qwen3.yaml",
        repo / "config/model/qwen3vl.yaml",
    ]
    hashes = {
        path.name: file_sha256(path)
        for path in config_paths
        if path.is_file()
    }
    return canonical_json_hash(hashes)


def current_git_commit(repo_root: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def sanitize_command_args(command_args: Mapping[str, Any] | argparse.Namespace, repo_root: str | Path) -> dict[str, Any]:
    raw = vars(command_args) if isinstance(command_args, argparse.Namespace) else dict(command_args)
    return {
        str(key): sanitize_argument_value(value, repo_root)
        for key, value in sorted(raw.items())
        if not is_forbidden_key(key)
    }


def sanitize_argument_value(value: Any, repo_root: str | Path) -> Any:
    if isinstance(value, (str, Path)):
        text = str(value)
        if looks_like_path(text):
            return public_path(text, repo_root)
        return text
    if isinstance(value, list):
        return [sanitize_argument_value(item, repo_root) for item in value]
    if isinstance(value, tuple):
        return [sanitize_argument_value(item, repo_root) for item in value]
    if isinstance(value, dict):
        return {
            str(key): sanitize_argument_value(child, repo_root)
            for key, child in sorted(value.items())
            if not is_forbidden_key(key)
        }
    return value


def public_path(path: str | Path | None, repo_root: str | Path | None = None) -> str | None:
    if path in (None, ""):
        return None
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    repo = Path(repo_root) if repo_root not in (None, "") else Path.cwd()
    try:
        return str(path_obj.resolve().relative_to(repo.resolve()))
    except ValueError:
        return path_obj.name


def looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or value.endswith((".json", ".jsonl", ".yaml", ".yml", ".txt", ".md"))


def sanitize_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if is_forbidden_key(key_text):
                continue
            clean_child = sanitize_public_value(child)
            clean[key_text] = clean_child
        return clean
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_TEXT_FRAGMENTS):
            return None
        return value
    return value


def assert_no_forbidden_public_fields(value: Any) -> None:
    violations: list[str] = []
    find_forbidden_public_fields(value, "$", violations)
    if violations:
        raise ValueError("Forbidden public fields or values found: " + ", ".join(violations[:10]))


def find_forbidden_public_fields(value: Any, path: str, violations: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if is_forbidden_key(key_text):
                violations.append(f"{path}.{key_text}")
            find_forbidden_public_fields(child, f"{path}.{key_text}", violations)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            find_forbidden_public_fields(child, f"{path}[{index}]", violations)
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in FORBIDDEN_TEXT_FRAGMENTS:
            if fragment in lowered:
                violations.append(f"{path}:{fragment}")


def is_forbidden_key(key: Any) -> bool:
    key_text = str(key)
    lowered = key_text.lower()
    return lowered in FORBIDDEN_FIELD_NAMES or lowered.startswith("gold_")
