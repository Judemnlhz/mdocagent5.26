"""Generic code/name list extractor for Stage 2 page text.

This module derives text_span artifacts for public code-to-name lists such as
"EPS Geographic Market Name Code". It is deterministic, page-local, and does
not inspect questions, answers, gold fields, evidence pages, or document ids for
extraction decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from mdocnexus.stage2.locator_enrichment import classify_artifact_locator, page_id_for

CODE_RE = re.compile(r"^[A-Z]{2,4}\d{2,4}$")
STATE_RE = re.compile(r"^(.+?)\s*\(([A-Z]{2,4})\)$")
MAX_CODE_NAME_PAIRS_PER_PAGE = 80


@dataclass(frozen=True)
class TextLine:
    text: str
    block_id: str
    char_start: int
    char_end: int
    line_index: int


@dataclass(frozen=True)
class CodeNamePair:
    code: str
    name: str
    group_label: str
    group_code: str
    ordinal: str
    source_text: str
    block_id: str
    char_start: int
    char_end: int
    pair_index: int


def extract_code_name_list_artifacts(
    *,
    selected_page: Mapping[str, Any],
    page_input: Mapping[str, Any],
    existing_artifacts: list[Mapping[str, Any]],
    max_pairs: int = MAX_CODE_NAME_PAIRS_PER_PAGE,
) -> list[dict[str, Any]]:
    lines = _page_text_lines(page_input)
    if not lines:
        return []
    pairs = _extract_pairs(lines, max_pairs=max_pairs)
    if not pairs:
        return []

    doc_id = str(selected_page.get("doc_id") or page_input.get("doc_id") or "")
    page_index = int(selected_page.get("page_index", page_input.get("page_index", 0)) or 0)
    page_id = page_id_for(doc_id, page_index)
    existing_keys = _existing_code_keys(existing_artifacts)
    existing_ids = {str(artifact.get("artifact_id")) for artifact in existing_artifacts if artifact.get("artifact_id") not in (None, "")}
    artifacts: list[dict[str, Any]] = []
    local_index = 1
    for pair in pairs:
        key = (_compact_lower(pair.code), _compact_lower(pair.name), page_index)
        if key in existing_keys:
            continue
        artifact_id = _next_artifact_id("code_name_pair", local_index, existing_ids)
        local_index += 1
        existing_ids.add(artifact_id)
        artifacts.append(_build_artifact(artifact_id, doc_id, page_id, page_index, pair))
    return artifacts


def _extract_pairs(lines: list[TextLine], max_pairs: int) -> list[CodeNamePair]:
    pairs: list[CodeNamePair] = []
    current_group = ""
    current_group_code = ""
    pending_ordinal = ""
    pending_name_parts: list[TextLine] = []
    seen: set[tuple[str, str]] = set()

    for line in lines:
        if _is_heading(line.text):
            pending_ordinal = ""
            pending_name_parts = []
            continue
        group = STATE_RE.match(line.text)
        if group and not CODE_RE.match(line.text):
            current_group = group.group(1).strip()
            current_group_code = group.group(2).strip()
            pending_ordinal = ""
            pending_name_parts = []
            continue
        ordinal = re.match(r"^(\d+)\.\s*(.*)$", line.text)
        if ordinal:
            pending_ordinal = ordinal.group(1)
            rest = ordinal.group(2).strip()
            pending_name_parts = [TextLine(rest, line.block_id, line.char_start, line.char_end, line.line_index)] if rest else []
            continue
        if CODE_RE.match(line.text):
            name = _compact(" ".join(part.text for part in pending_name_parts))
            code = line.text.strip()
            if name and current_group:
                key = (_compact_lower(code), _compact_lower(name))
                if key not in seen:
                    seen.add(key)
                    source_lines = [*pending_name_parts, line]
                    pairs.append(CodeNamePair(
                        code=code,
                        name=name,
                        group_label=current_group,
                        group_code=current_group_code,
                        ordinal=pending_ordinal,
                        source_text=_compact(" ".join(part.text for part in source_lines)),
                        block_id=line.block_id,
                        char_start=min(part.char_start for part in source_lines),
                        char_end=max(part.char_end for part in source_lines),
                        pair_index=len(pairs),
                    ))
                    if len(pairs) >= max_pairs:
                        break
            pending_ordinal = ""
            pending_name_parts = []
            continue
        if pending_ordinal and _is_name_continuation(line.text):
            pending_name_parts.append(line)
    return pairs


def _build_artifact(artifact_id: str, doc_id: str, page_id: str, page_index: int, pair: CodeNamePair) -> dict[str, Any]:
    content = f"{pair.code}: {pair.name}"
    artifact = {
        "artifact_id": artifact_id,
        "doc_id": doc_id,
        "page_id": page_id,
        "page_index": page_index,
        "artifact_type": "text_span",
        "modality": "text",
        "content": content,
        "normalized_content": {
            "eps_code": pair.code,
            "code": pair.code,
            "geographic_market_name": pair.name,
            "name": pair.name,
            "group_label": pair.group_label,
            "group_code": pair.group_code,
            "ordinal": pair.ordinal,
            "source_text": pair.source_text,
            "extraction_method": "generic_code_name_list_extractor",
        },
        "source_anchors": [{"anchor_type": "text_block", "source_id": pair.block_id, "page_index": page_index, "bbox": None}],
        "provenance": {"op": "ATOM", "sources": [pair.block_id], "method": "generic_code_name_list_extractor"},
        "status": "anchored",
        "validation_status": "anchored",
        "locators": [
            {"locator_kind": "text_offset", "block_id": pair.block_id, "char_start": pair.char_start, "char_end": pair.char_end},
            {"locator_kind": "source_block", "block_id": pair.block_id, "source_id": pair.block_id},
        ],
    }
    classification = classify_artifact_locator(artifact)
    artifact["source_anchored"] = bool(classification["source_anchored"])
    artifact["element_locatable"] = bool(classification["element_locatable"])
    artifact["proof_trace_eligible"] = bool(classification["proof_trace_eligible"])
    return artifact


def _page_text_lines(page_input: Mapping[str, Any]) -> list[TextLine]:
    lines: list[TextLine] = []
    layout_blocks = page_input.get("layout_blocks")
    if isinstance(layout_blocks, list):
        for block in layout_blocks:
            if not isinstance(block, Mapping):
                continue
            text = block.get("text")
            block_id = str(block.get("block_id") or block.get("source_id") or "")
            if not block_id or not isinstance(text, str) or not text.strip():
                continue
            lines.extend(_split_text_block(text, block_id, _int_or_zero(block.get("char_start")), len(lines)))
    if lines:
        return lines
    page_text = page_input.get("page_text")
    if isinstance(page_text, str) and page_text.strip():
        page_index = int(page_input.get("page_index", 0) or 0)
        return _split_text_block(page_text, f"p{page_index:03d}_text_0000", 0, 0)
    return []


def _split_text_block(text: str, block_id: str, block_start: int, base_line_index: int) -> list[TextLine]:
    lines: list[TextLine] = []
    offset = 0
    for raw_line in text.splitlines():
        raw_start = offset
        raw_end = raw_start + len(raw_line)
        offset = raw_end + 1
        compact = _compact(raw_line)
        if not compact:
            continue
        leading = len(raw_line) - len(raw_line.lstrip())
        lines.append(TextLine(compact, block_id, block_start + raw_start + leading, block_start + raw_end, base_line_index + len(lines)))
    return lines


def _is_heading(text: str) -> bool:
    lowered = text.lower()
    return lowered in {"eps", "geographic market name", "code"} or lowered.startswith("enrollment planning service") or lowered.startswith("major metropolitan area")


def _is_name_continuation(text: str) -> bool:
    if CODE_RE.match(text) or STATE_RE.match(text) or re.match(r"^\d+\.\s*", text):
        return False
    lowered = text.lower()
    if lowered in {"eps", "geographic market name", "code"}:
        return False
    if len(text) > 90:
        return False
    return any(ch.isalpha() for ch in text)


def _existing_code_keys(existing_artifacts: list[Mapping[str, Any]]) -> set[tuple[str, str, int]]:
    keys = set()
    for artifact in existing_artifacts:
        normalized = artifact.get("normalized_content") if isinstance(artifact.get("normalized_content"), Mapping) else {}
        code = str(normalized.get("eps_code") or normalized.get("code") or "")
        name = str(normalized.get("geographic_market_name") or normalized.get("name") or "")
        try:
            page_index = int(artifact.get("page_index"))
        except (TypeError, ValueError):
            page_index = -1
        if code and name:
            keys.add((_compact_lower(code), _compact_lower(name), page_index))
    return keys


def _next_artifact_id(prefix: str, local_index: int, existing_ids: set[str]) -> str:
    while True:
        candidate = f"{prefix}_{local_index:03d}"
        if candidate not in existing_ids:
            return candidate
        local_index += 1


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _compact_lower(text: str) -> str:
    return _compact(text).lower()


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
