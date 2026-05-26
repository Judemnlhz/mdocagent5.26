"""Provider errors for Stage 2 real API adapters."""

from __future__ import annotations

from typing import Optional


class ProviderError(Exception):
    """Base class for provider adapter failures."""

    def __init__(self, message: str, raw_text: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text


class ProviderNotConfiguredError(ProviderError):
    """Raised when the requested provider is not configured or supported."""


class ProviderResponseFormatError(ProviderError):
    """Raised when a provider response cannot be parsed as a JSON object."""


class ProviderDependencyError(ProviderError):
    """Raised when a required provider runtime dependency is unavailable."""


class ProviderDisabledError(ProviderError):
    """Raised when real API usage is not explicitly enabled."""
