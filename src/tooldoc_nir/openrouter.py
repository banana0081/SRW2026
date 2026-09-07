from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import requests


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot return a valid completion."""


def load_env_file(path: Path = Path(".env")) -> None:
    """Load a minimal dotenv file without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 180.0,
        max_retries: int = 6,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is empty.")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._explicit_session = session
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        if self._explicit_session is not None:
            return self._explicit_session
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    @property
    def session(self) -> requests.Session:
        return self._get_session()

    @classmethod
    def from_env(
        cls,
        *,
        env_path: Path = Path(".env"),
        timeout_seconds: float = 180.0,
        max_retries: int = 6,
    ) -> "OpenRouterClient":
        load_env_file(env_path)
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Add it to .env or the environment."
            )
        return cls(
            api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        provider: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = 42,
        reasoning_effort: str | None = "high",
        response_format: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None:
            payload.update(
                {
                    "tools": list(tools),
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if provider:
            payload["provider"] = {
                "order": [provider],
                "allow_fallbacks": False,
            }
        if reasoning_effort:
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": True,
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "ToolDoc NIR controlled evaluation",
        }
        retryable = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_session().post(
                    OPENROUTER_CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                time.sleep(2**attempt)
                continue

            if response.ok:
                try:
                    result = response.json()
                except ValueError as exc:
                    raise OpenRouterError(
                        "OpenRouter returned a non-JSON success response."
                    ) from exc
                if not isinstance(result, dict) or not result.get("choices"):
                    raise OpenRouterError(
                        f"OpenRouter response has no choices: {result!r}"
                    )
                result["_transport"] = {
                    "status_code": response.status_code,
                    "request_id": response.headers.get("x-request-id", ""),
                }
                return result

            body = response.text[:1000]
            last_error = f"HTTP {response.status_code}: {body}"
            if response.status_code not in retryable or attempt >= self.max_retries:
                break
            retry_after = response.headers.get("retry-after")
            if not retry_after:
                try:
                    error_payload = response.json().get("error", {})
                    metadata = error_payload.get("metadata", {})
                    retry_after = metadata.get("retry_after_seconds")
                except (AttributeError, ValueError):
                    retry_after = None
            try:
                delay = min(60.0, max(2.0, float(retry_after or 2**attempt)))
            except (TypeError, ValueError):
                delay = float(2**attempt)
            time.sleep(delay)

        raise OpenRouterError(f"OpenRouter request failed: {last_error}")
