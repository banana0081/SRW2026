from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from time import perf_counter
from typing import Any, Mapping
from urllib.parse import urlparse

import requests

from .dynamic_models import ExecutionObservation, ObservationSource
from .models import ApiDocument


_PATH_PARAMETER_RE = re.compile(r"\{([^{}]+)\}")
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _coerce_default(value: Any, declared_type: str) -> Any:
    declared = (declared_type or "").casefold()
    if value in (None, ""):
        if "number" in declared or "int" in declared:
            return 1
        if "bool" in declared:
            return False
        return "test"
    if "number" in declared or "int" in declared:
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
    if "bool" in declared and isinstance(value, str):
        return value.casefold() in {"1", "true", "yes"}
    return value


def default_arguments(document: ApiDocument) -> dict[str, Any]:
    required = {
        parameter.name: _coerce_default(parameter.default, parameter.type)
        for parameter in document.required_parameters
        if parameter.name
    }
    optional = {
        parameter.name: _coerce_default(parameter.default, parameter.type)
        for parameter in document.optional_parameters
        if parameter.name and parameter.default not in (None, "")
    }
    return {**required, **optional}


def _materialize_url(
    url: str, arguments: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    remaining = dict(arguments)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = remaining.pop(name, "test")
        return str(value)

    return _PATH_PARAMETER_RE.sub(replace, url), remaining


class HttpProber:
    """Bounded HTTP probe that records evidence and avoids unsafe calls by default."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 256_000,
        allow_unsafe_methods: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.allow_unsafe_methods = allow_unsafe_methods
        self.session = session or requests.Session()

    def probe(
        self,
        document: ApiDocument,
        *,
        arguments: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        source: ObservationSource = ObservationSource.REAL,
    ) -> ExecutionObservation:
        method = (document.method or "GET").upper()
        if method not in _SAFE_METHODS and not self.allow_unsafe_methods:
            raise ValueError(
                f"Refusing potentially state-changing method {method}; "
                "set allow_unsafe_methods=True explicitly."
            )
        parsed = urlparse(document.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Unsupported endpoint URL: {document.url!r}")

        supplied_arguments = (
            dict(arguments)
            if arguments is not None
            else default_arguments(document)
        )
        url, request_arguments = _materialize_url(
            document.url, supplied_arguments
        )
        request_headers = dict(headers or {})
        started_at = datetime.now(timezone.utc)
        timer = perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "headers": request_headers,
                "timeout": self.timeout_seconds,
                "stream": True,
            }
            if method in {"GET", "HEAD", "OPTIONS"}:
                kwargs["params"] = request_arguments
            else:
                kwargs["json"] = request_arguments
            response = self.session.request(method, url, **kwargs)
            raw = response.raw.read(
                self.max_response_bytes + 1, decode_content=True
            )
            truncated = len(raw) > self.max_response_bytes
            raw = raw[: self.max_response_bytes]
            text = raw.decode(response.encoding or "utf-8", errors="replace")
            try:
                body: Any = json.loads(text) if text else None
            except json.JSONDecodeError:
                body = text
            response_headers = {
                key.casefold(): value for key, value in response.headers.items()
            }
            if truncated:
                response_headers["x-tooldoc-body-truncated"] = "true"
            return ExecutionObservation(
                api_key=document.key,
                source=source,
                observed_at=started_at,
                request_method=method,
                request_arguments=supplied_arguments,
                status_code=response.status_code,
                latency_ms=round((perf_counter() - timer) * 1000, 3),
                response_headers=response_headers,
                response_body=body,
            )
        except requests.Timeout as exc:
            error_type = "timeout"
            error_message = str(exc)
        except requests.ConnectionError as exc:
            error_type = "connection_error"
            error_message = str(exc)
        except requests.RequestException as exc:
            error_type = "request_error"
            error_message = str(exc)

        return ExecutionObservation(
            api_key=document.key,
            source=source,
            observed_at=started_at,
            request_method=method,
            request_arguments=supplied_arguments,
            latency_ms=round((perf_counter() - timer) * 1000, 3),
            error_type=error_type,
            error_message=error_message,
        )

