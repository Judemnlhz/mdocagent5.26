"""MDocAgent integration adapters for MDocNexus outputs."""

from .mdocagent_adapter import (
    build_mdocagent_adapter_manifest,
    load_artifacts_by_page,
    load_mdocagent_retrieval_records,
    rerank_pages_with_artifacts,
    select_pages_with_graph,
    write_mdocagent_compatible_records,
)

__all__ = [
    "build_mdocagent_adapter_manifest",
    "load_artifacts_by_page",
    "load_mdocagent_retrieval_records",
    "rerank_pages_with_artifacts",
    "select_pages_with_graph",
    "write_mdocagent_compatible_records",
]
