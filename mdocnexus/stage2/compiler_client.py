"""Artifact compiler client abstractions for Stage 2."""

from __future__ import annotations

import json
from typing import Any, Dict

from .mock_artifact_outputs import build_mock_page_artifact_output


class ArtifactCompilerClient:
    """Abstract compiler client interface."""

    def generate_page_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class FakeArtifactCompilerClient(ArtifactCompilerClient):
    """Offline fake compiler client used by default tests and dry runs."""

    def generate_page_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = system_prompt
        _ = schema_dict
        prompt_payload = json.loads(user_prompt)
        document = prompt_payload["document"]
        return build_mock_page_artifact_output(
            doc_id=document["doc_id"],
            page_index=document["page_index"],
            layout_blocks=prompt_payload.get("layout_blocks", []),
        )


class RealArtifactCompilerClient(ArtifactCompilerClient):
    """Real compiler client skeleton guarded by enable_real_api."""

    def __init__(self, enable_real_api: bool = False, provider_config: Dict[str, Any] | None = None) -> None:
        self.enable_real_api = enable_real_api
        self.provider_config = provider_config or {}

    def generate_page_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.enable_real_api:
            raise RuntimeError("Real artifact compiler API is disabled. Pass enable_real_api=True explicitly.")
        raise NotImplementedError("Real provider integration is intentionally not implemented in Step 5.")
