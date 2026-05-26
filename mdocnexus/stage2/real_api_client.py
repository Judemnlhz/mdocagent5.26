"""Controlled real API adapter for Stage 2 artifact compilation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .api_config import ApiRunConfig, assert_real_api_allowed
from .compiler_client import ArtifactCompilerClient
from .provider_client import CompatibleChatJsonProvider
from .provider_errors import ProviderNotConfiguredError


class RealApiArtifactCompilerClient(ArtifactCompilerClient):
    """Real provider client guarded by ApiRunConfig and single-page policy."""

    def __init__(self, api_config: ApiRunConfig, provider: Optional[Any] = None) -> None:
        self.api_config = api_config
        self.provider = provider

    def generate_page_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert_real_api_allowed(self.api_config)
        provider = self.provider or self._build_provider()
        return provider.generate_json(system_prompt, user_prompt, schema_dict)

    def _build_provider(self) -> Any:
        if self.api_config.provider in {"siliconflow", "custom", "compatible_chat"}:
            return CompatibleChatJsonProvider(self.api_config)
        raise ProviderNotConfiguredError(f"Provider {self.api_config.provider!r} is not implemented.")
