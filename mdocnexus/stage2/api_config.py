"""Guardrails and configuration for controlled real API usage in Stage 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ApiRunConfig:
    enable_real_api: bool = False
    provider: str = "siliconflow"
    model_name: Optional[str] = None
    max_pages: int = 1
    temperature: float = 0.0
    timeout_seconds: int = 120
    raw_output_dir: str | Path | None = None
    discard_log_dir: str | Path | None = None
    api_base_url: Optional[str] = None
    api_key_env_var: str = "SILICONFLOW_API_KEY"
    api_key: Optional[str] = field(default=None, repr=False)
    max_tokens: Optional[int] = None


def assert_real_api_allowed(config: ApiRunConfig) -> None:
    """Enforce explicit single-page opt-in before any real API call."""

    if not config.enable_real_api:
        raise RuntimeError("Real API usage is disabled. Set enable_real_api=True explicitly.")
    if config.max_pages != 1:
        raise RuntimeError("Real API usage is limited to max_pages=1 for Stage 2 smoke tests.")
    if not config.model_name:
        raise RuntimeError("Real API usage requires a non-empty model_name.")
