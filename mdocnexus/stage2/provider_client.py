"""Generic chat-completions JSON provider for Stage 2 artifact compilation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .api_config import ApiRunConfig, assert_real_api_allowed
from .provider_errors import ProviderError, ProviderNotConfiguredError, ProviderResponseFormatError


DEFAULT_PROVIDER_BASE_URLS = {
    "siliconflow": "https://api.siliconflow.cn/v1",
}


class CompatibleChatJsonProvider:
    """Small HTTP adapter for providers exposing chat-completions style JSON responses."""

    def __init__(self, api_config: ApiRunConfig) -> None:
        self.api_config = api_config

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Request one JSON object from a configured provider without validation or repair."""

        assert_real_api_allowed(self.api_config)
        api_key = _read_api_key(self.api_config)
        request_body = _build_request_body(
            model_name=str(self.api_config.model_name),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_dict=schema_dict,
            temperature=self.api_config.temperature,
            max_tokens=self.api_config.max_tokens,
        )
        request = Request(
            _resolve_chat_completions_url(self.api_config),
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.api_config.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8")
        except HTTPError as exc:
            raw_text = _read_error_body(exc)
            raise ProviderResponseFormatError(
                f"Provider HTTP error: {exc.code}",
                raw_text=raw_text,
            ) from exc
        except URLError as exc:
            raise ProviderError(f"Provider request failed: {exc.reason}") from exc

        return _parse_chat_completion_json(raw_text)


def _build_request_body(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    schema_dict: Dict[str, Any],
    temperature: float,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    _ = schema_dict
    request_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
        "metadata": {
            "stage": "stage2_artifact_compilation",
            "schema_name": "page_artifact_output",
            "schema_version": "stage2_artifact_schema_v1",
        },
    }
    if max_tokens is not None:
        request_body["max_tokens"] = max_tokens
    return request_body


def _read_api_key(api_config: ApiRunConfig) -> str:
    direct_api_key = getattr(api_config, "api_key", None)
    if direct_api_key:
        return str(direct_api_key)
    api_key = os.environ.get(api_config.api_key_env_var, "")
    if not api_key.strip():
        raise ProviderNotConfiguredError(
            f"Provider API key environment variable is not set: {api_config.api_key_env_var}"
        )
    return api_key


def _resolve_chat_completions_url(api_config: ApiRunConfig) -> str:
    base_url = api_config.api_base_url or os.environ.get("STAGE2_API_BASE_URL")
    if not base_url:
        base_url = DEFAULT_PROVIDER_BASE_URLS.get(api_config.provider)
    if not base_url:
        raise ProviderNotConfiguredError(
            f"Provider base URL is not configured for provider {api_config.provider!r}."
        )
    normalized_base = base_url.rstrip("/")
    if normalized_base.endswith("/chat/completions"):
        return normalized_base
    return f"{normalized_base}/chat/completions"


def _parse_chat_completion_json(raw_text: str) -> Dict[str, Any]:
    try:
        response_object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProviderResponseFormatError("Provider response was not valid JSON.", raw_text=raw_text) from exc

    if isinstance(response_object, dict) and _looks_like_page_artifact_output(response_object):
        return response_object

    content = _extract_message_content(response_object)
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ProviderResponseFormatError("Provider response did not contain JSON text.", raw_text=raw_text)

    json_text = _strip_json_fence(content)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ProviderResponseFormatError("Provider message content was not valid JSON.", raw_text=content) from exc
    if not isinstance(parsed, dict):
        raise ProviderResponseFormatError("Provider message JSON was not an object.", raw_text=content)
    return parsed


def _extract_message_content(response_object: Any) -> Optional[Any]:
    if not isinstance(response_object, dict):
        return None
    choices = response_object.get("choices")
    if not isinstance(choices, list) or not choices:
        return response_object.get("content")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            return _join_content_parts(content)
        return content
    return first_choice.get("text")


def _join_content_parts(content_parts: list[Any]) -> str:
    text_parts = []
    for part in content_parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif isinstance(part, str):
            text_parts.append(part)
    return "".join(text_parts)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    fence = chr(96) * 3
    if stripped.startswith(fence):
        lines = stripped.splitlines()
        if lines and lines[0].startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _looks_like_page_artifact_output(value: Dict[str, Any]) -> bool:
    return {"doc_id", "page_index", "artifacts"}.issubset(value.keys())


def _read_error_body(error: HTTPError) -> str:
    try:
        return error.read().decode("utf-8")
    except Exception:
        return ""
