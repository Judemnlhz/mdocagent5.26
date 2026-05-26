"""Stage 2 schema definitions for candidate evidence artifacts and page inputs.

The project environment does not require a third-party schema library here. The
dataclasses below keep the approved field names and enforce the Stage 2 status
boundary with standard-library runtime checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class ArtifactType(str, Enum):
    page_summary = "page_summary"
    text_span = "text_span"
    numeric_fact = "numeric_fact"
    table = "table"
    table_cell = "table_cell"
    figure = "figure"
    caption = "caption"
    claim_candidate = "claim_candidate"
    document_identity = "document_identity"
    reference_section = "reference_section"
    reference_entry = "reference_entry"
    organization_mention = "organization_mention"
    visual_observation = "visual_observation"
    handwriting_observation = "handwriting_observation"
    color_observation = "color_observation"


class Modality(str, Enum):
    text = "text"
    visual = "visual"
    image = "image"
    table = "table"
    figure = "figure"
    multimodal = "multimodal"
    metadata = "metadata"


class ProvenanceOp(str, Enum):
    atom = "ATOM"
    and_op = "AND"
    or_op = "OR"


ValidationStatus = Literal["candidate", "schema_valid", "anchored", "discarded"]
AnchorType = Literal[
    "text_block",
    "layout_block",
    "full_page_image",
    "table_cell",
    "figure_region",
]
LayoutBlockType = Literal["text_block", "full_page_image"]

ALLOWED_VALIDATION_STATUSES = {"candidate", "schema_valid", "anchored", "discarded"}
ALLOWED_ANCHOR_TYPES = {
    "text_block",
    "layout_block",
    "full_page_image",
    "table_cell",
    "figure_region",
}
ALLOWED_LAYOUT_BLOCK_TYPES = {"text_block", "full_page_image"}


@dataclass
class Provenance:
    op: ProvenanceOp
    sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.op = _coerce_enum(self.op, ProvenanceOp, "op")
        if not self.sources:
            raise ValueError("provenance.sources must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return _enum_to_value(asdict(self))


@dataclass
class SourceAnchor:
    source_id: str
    anchor_type: AnchorType
    page_index: int
    bbox: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.anchor_type not in ALLOWED_ANCHOR_TYPES:
            raise ValueError(f"Unsupported anchor_type: {self.anchor_type!r}")
        self.page_index = _coerce_page_index(self.page_index)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceArtifact:
    artifact_id: str
    doc_id: str
    page_index: int
    artifact_type: ArtifactType
    modality: Modality
    content: str
    normalized_content: Dict[str, Any] = field(default_factory=dict)
    source_anchors: List[SourceAnchor] = field(default_factory=list)
    provenance: Optional[Provenance] = None
    validation_status: ValidationStatus = "candidate"
    compiler_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.page_index = _coerce_page_index(self.page_index)
        self.artifact_type = _coerce_enum(self.artifact_type, ArtifactType, "artifact_type")
        self.modality = _coerce_enum(self.modality, Modality, "modality")
        if self.validation_status not in ALLOWED_VALIDATION_STATUSES:
            raise ValueError(f"Unsupported validation_status: {self.validation_status!r}")
        if not self.content:
            raise ValueError("content must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return _enum_to_value(asdict(self))


@dataclass
class PageArtifactOutput:
    doc_id: str
    page_index: int
    artifacts: List[EvidenceArtifact] = field(default_factory=list)
    uncertain_or_unreadable: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.page_index = _coerce_page_index(self.page_index)

    def to_dict(self) -> Dict[str, Any]:
        return _enum_to_value(asdict(self))


@dataclass
class LayoutBlock:
    block_id: str
    block_type: LayoutBlockType
    page_index: int
    bbox: Optional[List[float]] = None
    text: Optional[str] = None

    def __post_init__(self) -> None:
        if self.block_type not in ALLOWED_LAYOUT_BLOCK_TYPES:
            raise ValueError(f"Unsupported block_type: {self.block_type!r}")
        self.page_index = _coerce_page_index(self.page_index)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageCompilationInput:
    doc_id: str
    page_index: int
    page_text: Optional[str]
    page_text_path: Optional[str]
    page_image_path: Optional[str]
    has_page_text: bool
    has_page_image: bool
    layout_blocks: List[LayoutBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.page_index = _coerce_page_index(self.page_index)

    def to_dict(self) -> Dict[str, Any]:
        return _enum_to_value(asdict(self))


def _coerce_page_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean page_index: {value!r}")
    page_index = int(value)
    if page_index < 0:
        raise ValueError(f"Invalid negative page_index: {value!r}")
    return page_index


def _coerce_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported {field_name}: {value!r}") from exc


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_to_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_enum_to_value(child) for child in value]
    return value
