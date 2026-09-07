"""Wrap RestBench TMDB docs/queries into the ToolBench shape Inference_DFSDT expects.

DRAFT never released a RestBench agent. The table's DFSDT and DRAFT rows on
ToolBench are `Inference_DFSDT.py` with `--method Initial|DRAFT`. This adapter
is the smallest change that lets that same driver consume RestBench-TMDB:
each REST endpoint becomes a one-API tool, RapidAPI exec is replaced elsewhere
by live TMDB HTTP.
"""

from __future__ import annotations

import re
from typing import Any


def change_name(name: str) -> str:
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name


def standardize(string: str) -> str:
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while string.startswith("_"):
        string = string[1:]
    while string.endswith("_"):
        string = string[:-1]
    if string and string[0].isdigit():
        string = "get_" + string
    return string


def process_name(name: str) -> str:
    return change_name(standardize(name))


def wrap_document(document: dict[str, Any], *, category: str = "TMDB") -> dict[str, Any]:
    tool_name = str(document.get("tool_name") or "")
    description = str(document.get("description") or "")
    tool_description = str(document.get("tool_description") or description)
    url = str(document.get("url") or "")
    guideline: dict[str, Any] = {
        "name": tool_name,
        "description": description,
        "url": url,
        "required_parameters": list(document.get("required_parameters") or []),
        "optional_parameters": list(document.get("optional_parameters") or []),
    }
    if document.get("example") is not None:
        guideline["example"] = document["example"]
    return {
        "ID": str(document.get("ID")),
        "category": category,
        "tool_name": tool_name,
        "tool_description": tool_description,
        "url": url,
        "tool_guidelines": {tool_name: guideline},
    }


def wrap_instructions(
    instructions: dict[str, dict[str, Any]], *, category: str = "TMDB"
) -> dict[str, dict[str, Any]]:
    return {key: wrap_document(value, category=category) for key, value in instructions.items()}


def wrap_query(query: dict[str, Any]) -> dict[str, Any]:
    api_list = []
    for item in query.get("api_list") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name") or item.get("api_name") or "")
        row = dict(item)
        row["tool_name"] = name
        row["api_name"] = name
        row["category_name"] = "TMDB"
        api_list.append(row)
    gold = [str(name) for name in (query.get("relevant APIs") or [])]
    return {
        "query": query.get("query"),
        "query_id": query.get("query_id"),
        "relevant APIs": [[name, name] for name in gold],
        "Tool_dic": list(query.get("Tool_dic") or []),
        "api_list": api_list,
    }


def url_index(instructions: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Map every name form the released driver might send to the HTTP template."""
    index: dict[str, str] = {}
    for document in instructions.values():
        url = str(document.get("url") or "")
        if not url:
            continue
        original = str(document.get("tool_name") or "")
        for key in (original, standardize(original), process_name(original)):
            if key:
                index[key] = url
    return index
