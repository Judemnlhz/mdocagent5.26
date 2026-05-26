"""Basic layout block construction for Stage 2 page inputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TEXT_BLOCK_SIZE = 1000


def build_basic_layout_blocks(
    doc_id: str,
    page_index: int,
    page_text: Optional[str],
    has_page_image: bool,
) -> List[Dict[str, Any]]:
    """Create deterministic source blocks for page text and full-page images."""

    _ = doc_id
    page_index = int(page_index)
    blocks: List[Dict[str, Any]] = []

    if page_text:
        chunks = [
            page_text[index : index + TEXT_BLOCK_SIZE]
            for index in range(0, len(page_text), TEXT_BLOCK_SIZE)
        ]
        for chunk_index, chunk in enumerate(chunks):
            normalized_chunk = chunk.strip()
            if normalized_chunk:
                blocks.append(
                    {
                        "block_id": f"p{page_index:03d}_text_{chunk_index:04d}",
                        "block_type": "text_block",
                        "page_index": page_index,
                        "bbox": None,
                        "text": normalized_chunk,
                    }
                )

    if has_page_image:
        blocks.append(
            {
                "block_id": f"p{page_index:03d}_full_page_image",
                "block_type": "full_page_image",
                "page_index": page_index,
                "bbox": None,
                "text": None,
            }
        )

    return blocks
