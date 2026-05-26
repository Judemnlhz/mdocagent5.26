"""Page text and image loading for Stage 2 page compilation inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .mdocagent_compat import build_mdocagent_extract_paths


def normalize_doc_name(doc_id: str) -> str:
    """Return the document name used by extracted page files."""

    return doc_id[:-4] if doc_id.endswith(".pdf") else doc_id


def find_existing_file(candidate_paths: List[Path]) -> Optional[Path]:
    """Return the first existing file from candidate paths."""

    for path in candidate_paths:
        if path.is_file():
            return path
    return None


def load_page_content(
    canonical_record: Dict[str, Any],
    extract_path: str | Path,
    page_index: int,
) -> Dict[str, Any]:
    """Load page text and image paths for one page index.

    This function intentionally reads only canonical document metadata and the
    requested page index. It never derives content from evaluation-only fields.
    """

    doc_id = canonical_record["document"]["doc_id"]
    doc_name = normalize_doc_name(doc_id)
    root = Path(extract_path)
    page_index = int(page_index)

    mdocagent_paths = build_mdocagent_extract_paths(root, doc_id, page_index)
    text_candidate_paths = list(mdocagent_paths["text_candidate_paths"])
    image_candidate_paths = list(mdocagent_paths["image_candidate_paths"])

    text_path = find_existing_file(text_candidate_paths)
    image_path = find_existing_file(image_candidate_paths)
    page_text = text_path.read_text(encoding="utf-8", errors="replace") if text_path else None

    return {
        "doc_id": doc_id,
        "page_index": page_index,
        "page_text": page_text,
        "page_text_path": str(text_path) if text_path else None,
        "page_image_path": str(image_path) if image_path else None,
        "has_page_text": page_text is not None,
        "has_page_image": image_path is not None,
        "text_paths_checked": [str(path) for path in text_candidate_paths],
        "image_paths_checked": [str(path) for path in image_candidate_paths],
    }


def build_text_candidate_paths(root: Path, doc_name: str, page_index: int) -> List[Path]:
    """Build supported candidate text paths in priority order."""

    return [
        root / f"{doc_name}_{page_index}.txt",
        root / f"{doc_name}_{page_index:03d}.txt",
        root / "texts" / f"{doc_name}_{page_index}.txt",
        root / "texts" / f"{doc_name}_{page_index:03d}.txt",
        root / "texts" / f"page_{page_index}.txt",
        root / "texts" / f"page_{page_index:03d}.txt",
    ]


def build_image_candidate_paths(root: Path, doc_name: str, page_index: int) -> List[Path]:
    """Build supported candidate image paths in priority order."""

    return [
        root / f"{doc_name}_{page_index}.png",
        root / f"{doc_name}_{page_index:03d}.png",
        root / "images" / f"{doc_name}_{page_index}.png",
        root / "images" / f"{doc_name}_{page_index:03d}.png",
        root / "images" / f"page_{page_index}.png",
        root / "images" / f"page_{page_index:03d}.png",
    ]
