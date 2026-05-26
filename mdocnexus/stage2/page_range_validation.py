"""Generic explicit page-reference range validation for Stage 2."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .mdocagent_compat import normalize_doc_name_for_mdocagent


PAGE_RANGE_POLICY = "retrieval_union_plus_valid_explicit_page_constraints"
OUT_OF_RANGE_ERROR = "explicit_page_reference_out_of_range"
PAGE_COUNT_UNKNOWN_ERROR = "document_page_count_unknown"


def infer_document_page_count(
    doc_id: str,
    pdf_root: str | Path | None = None,
    extract_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Infer document page count from a PDF first, then extracted page files."""

    pdf_path = _find_pdf_path(doc_id, pdf_root)
    if pdf_path is not None:
        page_count = _read_pdf_page_count(pdf_path)
        if page_count is not None:
            return {
                "page_count": page_count,
                "source": "pdf",
                "available_page_indices": list(range(page_count)),
                "page_index_contiguous": True,
            }

    available_page_indices = _find_extract_page_indices(doc_id, extract_root)
    if available_page_indices:
        page_count = max(available_page_indices) + 1
        return {
            "page_count": page_count,
            "source": "extract_files",
            "available_page_indices": available_page_indices,
            "page_index_contiguous": available_page_indices == list(range(page_count)),
        }

    return {
        "page_count": None,
        "source": "unknown",
        "available_page_indices": [],
        "page_index_contiguous": None,
    }


def validate_explicit_page_references_against_page_count(
    canonical_record: Dict[str, Any],
    page_count_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate all explicit page references against known document bounds."""

    explicit_refs = canonical_record.get("question_constraints", {}).get("explicit_page_references", [])
    page_count = page_count_info.get("page_count")
    available_indices = sorted({int(index) for index in page_count_info.get("available_page_indices", [])})
    valid_indices: Set[int] = set()
    invalid_refs: List[Dict[str, Any]] = []

    for raw_ref in explicit_refs:
        ref = _normalize_explicit_ref(raw_ref)
        page_index = ref["page_index_zero_based"]
        if page_count is not None:
            if 0 <= page_index < int(page_count):
                valid_indices.add(page_index)
            else:
                invalid_refs.append(_build_out_of_range_ref(ref, int(page_count), page_count_info))
        elif available_indices:
            if page_index in available_indices:
                valid_indices.add(page_index)
            else:
                invalid_refs.append(_build_out_of_range_ref(ref, None, page_count_info))
        else:
            unknown_ref = dict(ref)
            unknown_ref.update(
                {
                    "error_type": PAGE_COUNT_UNKNOWN_ERROR,
                    "page_count": None,
                    "max_valid_page_index": None,
                }
            )
            invalid_refs.append(unknown_ref)

    return {
        "valid_explicit_page_indices": sorted(valid_indices),
        "invalid_explicit_page_references": invalid_refs,
        "page_count_info": page_count_info,
    }


def apply_explicit_page_range_validation_to_canonical_record(
    canonical_record: Dict[str, Any],
    page_count_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Update candidate_pool and compilation_plan with only valid explicit pages."""

    validation = validate_explicit_page_references_against_page_count(canonical_record, page_count_info)
    candidate_pool = canonical_record.setdefault("candidate_pool", {})
    raw_explicit_pages = _extract_raw_explicit_page_indices(canonical_record)
    valid_explicit_pages = validation["valid_explicit_page_indices"]
    invalid_explicit_refs = validation["invalid_explicit_page_references"]
    invalid_explicit_pages = {
        int(ref["page_index_zero_based"])
        for ref in invalid_explicit_refs
        if ref.get("page_index_zero_based") is not None
    }
    retrieval_pages = _coerce_page_list(candidate_pool.get("retrieval_candidate_pages", []))
    required_pages = sorted((set(retrieval_pages) | set(valid_explicit_pages)) - invalid_explicit_pages)

    candidate_pool.update(
        {
            "retrieval_candidate_pages": retrieval_pages,
            "explicit_constraint_pages_raw": raw_explicit_pages,
            "explicit_constraint_pages_valid": valid_explicit_pages,
            "explicit_constraint_pages_invalid": invalid_explicit_refs,
            "explicit_constraint_pages": valid_explicit_pages,
            "required_pages_for_compilation": required_pages,
            "retrieval_missed_explicit_pages": [
                page for page in valid_explicit_pages if page not in retrieval_pages
            ],
            "candidate_pool_policy": PAGE_RANGE_POLICY,
            "explicit_page_range_validation": validation,
        }
    )

    compilation_plan = canonical_record.setdefault("compilation_plan", {})
    compilation_plan.update(
        {
            "compile_scope": PAGE_RANGE_POLICY,
            "pages_to_compile": required_pages,
            "priority_pages": valid_explicit_pages,
            "compilation_reasons": _build_valid_compilation_reasons(canonical_record, valid_explicit_pages),
        }
    )
    return validation


def _find_pdf_path(doc_id: str, pdf_root: str | Path | None) -> Optional[Path]:
    doc_path = Path(doc_id)
    doc_name = normalize_doc_name_for_mdocagent(doc_path.name)
    candidates: List[Path] = []
    if doc_path.is_file():
        candidates.append(doc_path)
    if pdf_root is not None:
        root = Path(pdf_root)
        candidates.extend(
            [
                root / doc_id,
                root / doc_path.name,
                root / f"{doc_name}.pdf",
            ]
        )
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            return candidate
    return None


def _read_pdf_page_count(pdf_path: Path) -> Optional[int]:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(pdf_path))
            return len(reader.pages)
        except Exception:
            continue
    try:
        import fitz  # type: ignore

        with fitz.open(str(pdf_path)) as document:
            return int(document.page_count)
    except Exception:
        return None


def _find_extract_page_indices(doc_id: str, extract_root: str | Path | None) -> List[int]:
    if extract_root is None:
        return []
    root = Path(extract_root)
    if not root.exists():
        return []
    doc_name = normalize_doc_name_for_mdocagent(Path(doc_id).name)
    pattern = re.compile(rf"^{re.escape(doc_name)}_(\d+)\.(?:png|txt)$", re.IGNORECASE)
    indices: Set[int] = set()
    for scan_root in (root, root / "texts", root / "images"):
        if not scan_root.is_dir():
            continue
        for path in scan_root.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match:
                indices.add(int(match.group(1)))
    return sorted(indices)


def _normalize_explicit_ref(raw_ref: Dict[str, Any]) -> Dict[str, Any]:
    page_number = raw_ref.get("page_number_one_based")
    page_index = raw_ref.get("page_index_zero_based")
    if page_index is None and page_number is not None:
        page_index = int(page_number) - 1
    if page_number is None and page_index is not None:
        page_number = int(page_index) + 1
    page_index = int(page_index) if page_index is not None else -1
    page_number = int(page_number) if page_number is not None else page_index + 1
    return {
        "surface_text": raw_ref.get("surface_text", f"page {page_number}"),
        "page_number_one_based": page_number,
        "page_index_zero_based": page_index,
        "source": raw_ref.get("source", "question_text"),
    }


def _build_out_of_range_ref(
    ref: Dict[str, Any],
    page_count: Optional[int],
    page_count_info: Dict[str, Any],
) -> Dict[str, Any]:
    available_indices = page_count_info.get("available_page_indices", [])
    max_valid_page_index = page_count - 1 if page_count is not None else (max(available_indices) if available_indices else None)
    result = dict(ref)
    result.update(
        {
            "error_type": OUT_OF_RANGE_ERROR,
            "page_count": page_count,
            "max_valid_page_index": max_valid_page_index,
        }
    )
    return result


def _extract_raw_explicit_page_indices(canonical_record: Dict[str, Any]) -> List[int]:
    pages = []
    for ref in canonical_record.get("question_constraints", {}).get("explicit_page_references", []):
        page_index = _normalize_explicit_ref(ref)["page_index_zero_based"]
        if page_index >= 0:
            pages.append(page_index)
    return sorted(set(pages))


def _coerce_page_list(raw_pages: Any) -> List[int]:
    pages: Set[int] = set()
    for raw_page in raw_pages or []:
        if isinstance(raw_page, bool):
            continue
        page_index = int(raw_page)
        if page_index >= 0:
            pages.add(page_index)
    return sorted(pages)


def _build_valid_compilation_reasons(
    canonical_record: Dict[str, Any],
    valid_explicit_pages: List[int],
) -> List[Dict[str, Any]]:
    valid_set = set(valid_explicit_pages)
    reasons: List[Dict[str, Any]] = []
    for raw_ref in canonical_record.get("question_constraints", {}).get("explicit_page_references", []):
        ref = _normalize_explicit_ref(raw_ref)
        if ref["page_index_zero_based"] in valid_set:
            reasons.append(
                {
                    "page_index": ref["page_index_zero_based"],
                    "reason_type": "explicit_page_reference",
                    "reason_text": ref["surface_text"],
                }
            )
    return reasons
