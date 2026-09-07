"""Live TMDB backend with the same return shape as DRAFT's get_rapidapi_response."""

from __future__ import annotations

import json
from typing import Any, Mapping

from tooldoc_nir.restbench_http import TmdbClient, http_method_from_tool

RESPONSE_CHARS = 2048


class TmdbDfsdtBackend:
    """Drop-in for Inference_DFSDT.get_rapidapi_response on RestBench-TMDB."""

    def __init__(
        self,
        tmdb: TmdbClient,
        urls: Mapping[str, str],
    ) -> None:
        self.tmdb = tmdb
        self.urls = dict(urls)
        self.calls = 0
        self.live = 0
        self.errors = 0

    def __call__(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        tool_name = str(input_dict.get("tool_name") or "")
        api_name = str(input_dict.get("api_name") or "")
        raw = input_dict.get("tool_input") or {}
        if isinstance(raw, str):
            try:
                parameters = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return {"error": "Tool input parse error...\n", "response": ""}
        elif isinstance(raw, dict):
            parameters = dict(raw)
        else:
            parameters = {}
        url = self.urls.get(tool_name) or self.urls.get(api_name) or ""
        if not url:
            self.errors += 1
            return {
                "error": f"unknown TMDB tool {tool_name}/{api_name}",
                "response": "",
            }
        try:
            payload = self.tmdb.call(
                url_template=url,
                arguments=parameters,
                method=http_method_from_tool(api_name or tool_name),
            )
        except (ValueError, OSError, TypeError) as exc:
            self.errors += 1
            return {"error": str(exc), "response": ""}
        if payload.get("source") == "live":
            self.live += 1
        error = str(payload.get("error") or "")
        if error:
            self.errors += 1
        body = str(payload.get("response") or "")[:RESPONSE_CHARS]
        return {"error": error, "response": body}
