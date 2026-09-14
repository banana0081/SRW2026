from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import requests

from tooldoc_nir.provider_route import (
    DEFAULT_ROUTE,
    ROUTE_BALANCED,
    provider_preferences,
    rank_provider_tags,
)


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
ROUTE_CACHE_SECONDS = 300.0


def normalize_provider(name: str | None) -> str:
    """Fold the routing slug and the served display name onto one key.

    `provider.order` takes a slug (`novita`, `deepinfra`), while the response
    reports a display name (`Novita`, `DeepInfra`, `Sail Research`). Pinning
    can only be verified if both sides are compared in the same shape.
    """
    return "".join(character for character in (name or "").lower() if character.isalnum())


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


def _chat_url(base_url: str | None) -> str:
    if not base_url:
        return OPENROUTER_CHAT_URL
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 180.0,
        max_retries: int = 6,
        session: requests.Session | None = None,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is empty.")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._explicit_session = session
        self._local = threading.local()
        self.base_url = (base_url or "").rstrip("/") or None
        self.chat_url = _chat_url(self.base_url)
        self.local = bool(self.base_url)
        self.route = DEFAULT_ROUTE
        self._route_lock = threading.Lock()
        self._route_cache: dict[str, tuple[float, list[str]]] = {}

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
        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or ""
        ).strip()
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        )
        if base_url and not api_key:
            api_key = "lm-studio"
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Add it to .env or the environment."
            )
        if base_url and timeout_seconds == 180.0:
            timeout_seconds = 300.0
        client = cls(
            api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_url=base_url or None,
        )
        client.route = (
            os.environ.get("OPENROUTER_ROUTE") or DEFAULT_ROUTE
        ).strip().lower() or DEFAULT_ROUTE
        return client

    def _ranked_providers(self, model: str) -> list[str]:
        now = time.time()
        with self._route_lock:
            cached = self._route_cache.get(model)
            if cached and cached[0] > now:
                return list(cached[1])
        try:
            response = self._get_session().get(
                OPENROUTER_ENDPOINTS_URL.format(model=model),
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=min(30.0, self.timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError, TypeError):
            return []
        data = payload.get("data") if isinstance(payload, dict) else None
        endpoints = (data or payload or {}).get("endpoints") if isinstance(data or payload, dict) else []
        ranked = rank_provider_tags(endpoints or [])
        with self._route_lock:
            self._route_cache[model] = (now + ROUTE_CACHE_SECONDS, ranked)
        return ranked

    def _provider_payload(
        self, model: str, pin: str | None
    ) -> dict[str, Any] | None:
        ranked = None
        if not pin and self.route == ROUTE_BALANCED:
            ranked = self._ranked_providers(model)
        return provider_preferences(route=self.route, pin=pin, ranked=ranked)

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
        if seed is not None and not self.local:
            payload["seed"] = seed
        if response_format is not None and not self.local:
            payload["response_format"] = dict(response_format)
        if not self.local:
            prefs = self._provider_payload(model, provider)
            if prefs:
                payload["provider"] = prefs
        if reasoning_effort and not self.local:
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
        label = "local chat" if self.local else "OpenRouter"
        timeout: float | tuple[float, float]
        if self.local:
            timeout = (10.0, self.timeout_seconds)
        else:
            timeout = self.timeout_seconds
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_session().post(
                    self.chat_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # A closed local box is not a 429. Do not sit on connect
                # retries while the rest of the seed writes transport misses.
                if self.local and isinstance(exc, requests.exceptions.ConnectionError):
                    break
                if attempt >= self.max_retries:
                    break
                time.sleep(2**attempt)
                continue

            if response.ok:
                try:
                    result = response.json()
                except ValueError as exc:
                    raise OpenRouterError(
                        f"{label} returned a non-JSON success response."
                    ) from exc
                if not isinstance(result, dict) or not result.get("choices"):
                    raise OpenRouterError(
                        f"{label} response has no choices: {result!r}"
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

        raise OpenRouterError(f"{label} request failed: {last_error}")
