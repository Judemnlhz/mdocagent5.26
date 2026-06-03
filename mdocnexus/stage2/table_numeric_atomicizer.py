"""Generic table/numeric atomicizer for Stage 2 page text.

This module derives atomic table_cell and numeric_fact artifacts only from
page-local OCR/layout text and existing Stage 2 artifacts. It does not inspect
questions, answers, gold fields, evidence pages, dataset names, or document ids
for extraction decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from mdocnexus.stage2.locator_enrichment import classify_artifact_locator, page_id_for


NUMERIC_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9-])\(?\s*-?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?\s*(?:%|percent|percentage|bps|bp|million|billion|thousand|m\b|bn\b)?",
    re.I,
)
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
CURRENCY_ONLY_RE = re.compile(r"^\$+$")
MAX_ATOMIC_CELLS_PER_PAGE = 12


@dataclass(frozen=True)
class TextLine:
    text: str
    block_id: str
    char_start: int
    char_end: int
    line_index: int


@dataclass(frozen=True)
class AtomicCell:
    row_label: str
    column_label: str
    value_text: str
    source_text: str
    block_id: str
    char_start: int
    char_end: int
    row_index: int
    column_index: int
    table_id: str


def atomicize_table_numeric_artifacts(
    *,
    selected_page: Mapping[str, Any],
    page_input: Mapping[str, Any],
    existing_artifacts: list[Mapping[str, Any]],
    max_cells: int = MAX_ATOMIC_CELLS_PER_PAGE,
) -> list[dict[str, Any]]:
    """Return conservative atomic artifacts from generic table-like OCR text."""

    lines = _page_text_lines(page_input)
    if not lines:
        return []

    cells = _extract_atomic_cells(lines, max_cells=max_cells)
    if not cells:
        return []

    doc_id = str(selected_page.get("doc_id") or page_input.get("doc_id") or "")
    page_index = int(selected_page.get("page_index", page_input.get("page_index", 0)) or 0)
    page_id = page_id_for(doc_id, page_index)
    existing_keys = _existing_atomic_keys(existing_artifacts)
    existing_ids = {str(artifact.get("artifact_id")) for artifact in existing_artifacts if artifact.get("artifact_id") not in (None, "")}

    artifacts: list[dict[str, Any]] = []
    local_index = 1
    for cell in cells:
        cell_key = ("table_cell", _compact_lower(cell.row_label), _compact_lower(cell.column_label), _compact_lower(cell.value_text))
        fact_key = ("numeric_fact", _compact_lower(cell.row_label), _compact_lower(cell.column_label), _compact_lower(cell.value_text))
        if cell_key not in existing_keys:
            artifact_id = _next_artifact_id("atomicizer_table_cell", local_index, existing_ids)
            local_index += 1
            existing_ids.add(artifact_id)
            artifacts.append(_build_table_cell_artifact(artifact_id, doc_id, page_id, page_index, cell))
        if fact_key not in existing_keys:
            artifact_id = _next_artifact_id("atomicizer_numeric_fact", local_index, existing_ids)
            local_index += 1
            existing_ids.add(artifact_id)
            artifacts.append(_build_numeric_fact_artifact(artifact_id, doc_id, page_id, page_index, cell))
        if len(artifacts) >= max_cells * 2:
            break
    return artifacts


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
            block_start = _int_or_zero(block.get("char_start"))
            lines.extend(_split_text_block(text, block_id, block_start, len(lines)))
    if lines:
        return lines

    page_text = page_input.get("page_text")
    if isinstance(page_text, str) and page_text.strip():
        page_index = int(page_input.get("page_index", 0) or 0)
        block_id = f"p{page_index:03d}_text_0000"
        return _split_text_block(page_text, block_id, 0, 0)
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
        lines.append(
            TextLine(
                text=compact,
                block_id=block_id,
                char_start=block_start + raw_start + leading,
                char_end=block_start + raw_end,
                line_index=base_line_index + len(lines),
            )
        )
    return lines


def _extract_atomic_cells(lines: list[TextLine], max_cells: int) -> list[AtomicCell]:
    cells: list[AtomicCell] = []
    seen: set[tuple[str, str, str]] = set()
    row_index = 0

    for index, line in enumerate(lines):
        if len(cells) >= max_cells:
            break

        if _has_numeric_value(line.text):
            continue
        values = _collect_following_value_lines(lines, index + 1)
        if len(values) < 2:
            continue
        row_label = _row_label_at(lines, index)
        if not _is_usable_row_label(row_label):
            continue
        headers = _column_headers_for(lines, index, len(values))
        if _uses_generated_headers(headers):
            continue
        table_id = _table_id_for(line, row_index)
        source_text = " ".join([row_label, *[value.text for value in values]])
        for column_index, value_line in enumerate(values):
            value_text = _first_numeric_value(value_line.text)
            if not value_text:
                continue
            column_label = _column_label_for_value(headers, column_index, value_line.text)
            cell = AtomicCell(
                row_label=row_label,
                column_label=column_label,
                value_text=value_text,
                source_text=source_text,
                block_id=value_line.block_id,
                char_start=min(line.char_start, value_line.char_start),
                char_end=max(line.char_end, value_line.char_end),
                row_index=row_index,
                column_index=column_index,
                table_id=table_id,
            )
            key = (_compact_lower(cell.row_label), _compact_lower(cell.column_label), _compact_lower(cell.value_text))
            if key in seen:
                continue
            seen.add(key)
            cells.append(cell)
            if len(cells) >= max_cells:
                break
        row_index += 1
    return cells


def _collect_following_value_lines(lines: list[TextLine], start_index: int) -> list[TextLine]:
    values: list[TextLine] = []
    for line in lines[start_index : start_index + 14]:
        if _is_currency_only(line.text):
            continue
        if _is_year_only(line.text) and len(values) >= 2:
            break
        if _has_numeric_value(line.text):
            values.append(line)
            if len(values) >= 4:
                break
            continue
        break
    if values and all(_is_year_only(line.text) for line in values):
        return []
    return values


def _row_label_at(lines: list[TextLine], index: int) -> str:
    return _compact(lines[index].text)[:180]


def _column_headers_for(lines: list[TextLine], row_index: int, value_count: int) -> list[str]:
    prior = lines[max(0, row_index - 30) : row_index]
    year_headers = [_compact(line.text) for line in prior if _is_year_only(line.text)]
    if len(year_headers) >= value_count:
        return _dedupe_headers(year_headers[-value_count:])

    short_headers = [
        _compact(line.text)
        for line in prior
        if _is_probable_column_header(line.text) and not _is_currency_only(line.text)
    ]
    if len(short_headers) >= value_count:
        return _dedupe_headers(short_headers[-value_count:])
    return [f"value_{index + 1}" for index in range(value_count)]


def _column_label_for_value(headers: list[str], column_index: int, value_text: str) -> str:
    derived = _label_from_value_line(value_text)
    if derived:
        return derived
    return headers[column_index] if column_index < len(headers) else f"value_{column_index + 1}"


def _label_from_value_line(text: str) -> str:
    if not _has_numeric_value(text):
        return ""
    label = NUMERIC_VALUE_RE.sub(" ", text)
    label = _compact(label.strip(":-–—,;()[]"))
    if not label or len(label) > 60:
        return ""
    if not any(ch.isalpha() for ch in label):
        return ""
    if len(label.split()) > 6:
        return ""
    return label


def _build_table_cell_artifact(artifact_id: str, doc_id: str, page_id: str, page_index: int, cell: AtomicCell) -> dict[str, Any]:
    artifact = {
        "artifact_id": artifact_id,
        "doc_id": doc_id,
        "page_id": page_id,
        "page_index": page_index,
        "artifact_type": "table_cell",
        "modality": "table",
        "content": _cell_content(cell),
        "normalized_content": {
            "table_id": cell.table_id,
            "row_index": cell.row_index,
            "column_index": cell.column_index,
            "row_header": cell.row_label,
            "column_header": cell.column_label,
            "value_text": cell.value_text,
            "unit": _infer_unit(cell.value_text, cell.source_text),
            "source_text": cell.source_text,
            "extraction_method": "generic_table_numeric_atomicizer",
        },
        "source_anchors": [_source_anchor(cell, page_index)],
        "provenance": {"op": "ATOM", "sources": [cell.block_id], "method": "generic_table_numeric_atomicizer"},
        "status": "anchored",
        "validation_status": "anchored",
        "locators": _locators(cell),
    }
    return _with_locator_flags(artifact)


def _build_numeric_fact_artifact(artifact_id: str, doc_id: str, page_id: str, page_index: int, cell: AtomicCell) -> dict[str, Any]:
    artifact = {
        "artifact_id": artifact_id,
        "doc_id": doc_id,
        "page_id": page_id,
        "page_index": page_index,
        "artifact_type": "numeric_fact",
        "modality": "numeric",
        "content": _cell_content(cell),
        "normalized_content": {
            "metric_name": cell.row_label,
            "row_label": cell.row_label,
            "column_label": cell.column_label,
            "value_text": cell.value_text,
            "unit": _infer_unit(cell.value_text, cell.source_text),
            "normalized_value": _normalize_numeric_value(cell.value_text),
            "source_text": cell.source_text,
            "extraction_method": "generic_table_numeric_atomicizer",
        },
        "source_anchors": [_source_anchor(cell, page_index)],
        "provenance": {"op": "ATOM", "sources": [cell.block_id], "method": "generic_table_numeric_atomicizer"},
        "status": "anchored",
        "validation_status": "anchored",
        "locators": _locators(cell),
    }
    return _with_locator_flags(artifact)


def _source_anchor(cell: AtomicCell, page_index: int) -> dict[str, Any]:
    return {"anchor_type": "text_block", "source_id": cell.block_id, "page_index": page_index, "bbox": None}


def _locators(cell: AtomicCell) -> list[dict[str, Any]]:
    return [
        {"locator_kind": "text_offset", "block_id": cell.block_id, "char_start": cell.char_start, "char_end": cell.char_end},
        {"locator_kind": "source_block", "block_id": cell.block_id, "source_id": cell.block_id},
        {
            "locator_kind": "table_cell",
            "table_id": cell.table_id,
            "row_index": cell.row_index,
            "column_index": cell.column_index,
        },
    ]


def _with_locator_flags(artifact: dict[str, Any]) -> dict[str, Any]:
    classification = classify_artifact_locator(artifact)
    artifact["source_anchored"] = bool(classification["source_anchored"])
    artifact["element_locatable"] = bool(classification["element_locatable"])
    artifact["proof_trace_eligible"] = bool(classification["proof_trace_eligible"])
    return artifact


def _existing_atomic_keys(existing_artifacts: list[Mapping[str, Any]]) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for artifact in existing_artifacts:
        artifact_type = str(artifact.get("artifact_type") or "")
        if artifact_type not in {"table_cell", "numeric_fact"}:
            continue
        normalized = artifact.get("normalized_content")
        if not isinstance(normalized, Mapping):
            normalized = {}
        row = _compact_lower(normalized.get("row_label") or normalized.get("row_header") or normalized.get("metric_name"))
        column = _compact_lower(normalized.get("column_label") or normalized.get("column_header") or normalized.get("context"))
        value = _compact_lower(normalized.get("value_text") or normalized.get("value") or normalized.get("metric_value"))
        if row and column and value:
            keys.add((artifact_type, row, column, value))
    return keys


def _next_artifact_id(prefix: str, local_index: int, existing_ids: set[str]) -> str:
    candidate_index = local_index
    artifact_id = f"{prefix}_{candidate_index:03d}"
    while artifact_id in existing_ids:
        candidate_index += 1
        artifact_id = f"{prefix}_{candidate_index:03d}"
    return artifact_id


def _cell_content(cell: AtomicCell) -> str:
    return f"{cell.row_label} {cell.column_label}: {cell.value_text}"


def _table_id_for(line: TextLine, row_index: int) -> str:
    return f"atomicizer_{line.block_id}_table_{row_index // 12 + 1:03d}"


def _numeric_values(text: str) -> list[str]:
    return [_clean_numeric(match.group(0)) for match in NUMERIC_VALUE_RE.finditer(text) if _clean_numeric(match.group(0))]


def _first_numeric_value(text: str) -> str:
    values = _numeric_values(text)
    return values[0] if values else ""


def _has_numeric_value(text: str) -> bool:
    return bool(_first_numeric_value(text))


def _clean_numeric(value: str) -> str:
    return _compact(value.replace("$", ""))


def _normalize_numeric_value(value_text: str) -> float | None:
    text = value_text.strip()
    negative = text.startswith("(")
    text = text.replace("(", "").replace(")", "").replace(",", "").replace("%", "").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def _infer_unit(value_text: str, source_text: str = "") -> str:
    lower = f"{value_text} {source_text}".lower()
    if "%" in lower or "percent" in lower or "percentage" in lower:
        return "percent"
    if "minute" in lower:
        return "minutes"
    if "hour" in lower:
        return "hours"
    if "day" in lower:
        return "days"
    if any(token in lower for token in ("million", "billion", "thousand", " m", " bn")):
        return "scaled_number"
    return "numeric"


def _is_probable_column_header(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    if _is_year_only(compact):
        return True
    if _has_numeric_value(compact) or _is_currency_only(compact):
        return False
    words = compact.split()
    if len(words) > 5:
        return False
    return any(any(ch.isalpha() for ch in word) for word in words)


def _is_usable_row_label(text: str) -> bool:
    compact = _compact(text)
    if len(compact) < 2 or len(compact) > 180:
        return False
    if _is_year_only(compact) or _is_currency_only(compact):
        return False
    if not any(ch.isalpha() for ch in compact):
        return False
    words = compact.split()
    if len(words) > 24:
        return False
    if compact.endswith(".") and len(words) > 14:
        return False
    return True


def _is_year_only(text: str) -> bool:
    return bool(YEAR_RE.fullmatch(_compact(text)))


def _is_currency_only(text: str) -> bool:
    return bool(CURRENCY_ONLY_RE.fullmatch(_compact(text)))


def _dedupe_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for header in headers:
        key = _compact(header)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1:
            result.append(key)
        else:
            result.append(f"{key} #{counts[key]}")
    return result


def _uses_generated_headers(headers: list[str]) -> bool:
    return bool(headers) and all(header.startswith("value_") for header in headers)


def _compact(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").strip().split())


def _compact_lower(value: Any) -> str:
    return _compact(value).lower()


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
