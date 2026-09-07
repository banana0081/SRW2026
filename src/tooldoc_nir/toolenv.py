from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterator

from .models import ApiDocument


def _endpoint_document(
    payload: dict[str, Any],
    api: dict[str, Any],
    *,
    category: str,
) -> ApiDocument:
    schema = api.get("schema")
    return ApiDocument(
        category_name=category,
        tool_name=str(
            payload.get("tool_name")
            or payload.get("name")
            or payload.get("title")
            or ""
        ),
        api_name=str(api.get("name") or ""),
        api_description=str(api.get("description") or ""),
        required_parameters=api.get("required_parameters") or [],
        optional_parameters=api.get("optional_parameters") or [],
        method=str(api.get("method") or ""),
        template_response=schema if schema not in ("", {}) else None,
        url=str(api.get("url") or ""),
        host=str(payload.get("host") or ""),
        historical_status_code=api.get("statuscode"),
        historical_schema=schema,
    )


def iter_toolenv_documents(root: Path) -> Iterator[ApiDocument]:
    """Yield endpoint-level documents from a ToolBench-style tool snapshot."""
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(
            payload.get("api_list"), list
        ):
            continue
        try:
            category = path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            category = path.parent.name
        for api in payload["api_list"]:
            if not isinstance(api, dict):
                continue
            document = _endpoint_document(payload, api, category=category)
            if document.tool_name and document.api_name:
                yield document


def load_toolenv(root: Path) -> list[ApiDocument]:
    return list(iter_toolenv_documents(root))


def inventory(documents: list[ApiDocument]) -> dict[str, Any]:
    status_codes = Counter(str(item.historical_status_code) for item in documents)
    return {
        "endpoints": len(documents),
        "tools": len({item.key[0] for item in documents}),
        "categories": len({item.category_name for item in documents}),
        "with_url": sum(bool(item.url) for item in documents),
        "with_historical_status": sum(
            item.historical_status_code is not None for item in documents
        ),
        "with_historical_schema": sum(
            item.historical_schema not in (None, "", {}) for item in documents
        ),
        "historical_status_codes": dict(status_codes.most_common()),
    }

