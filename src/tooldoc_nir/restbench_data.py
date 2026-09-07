"""RestBench queries and documentation overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tooldoc_nir.draft_agent_reproduction import _load_json
from tooldoc_nir.restbench_http import path_param_names

DEFAULT_DRAFT_ROOT = Path("external/DRAFT")
DATASETS = ("TMDB", "Spotify")
DEFAULT_MODEL = "inclusionai/ling-3.0-flash"


def instruction_path(draft_root: Path, dataset: str, condition: str) -> Path:
    return (
        draft_root
        / "dataset"
        / "RestBench"
        / "tool_instruction"
        / f"{dataset}_{condition}.json"
    )


def test_path(draft_root: Path, dataset: str) -> Path:
    return draft_root / "dataset" / "RestBench" / "test_data" / f"{dataset}.json"


def load_queries(draft_root: Path, dataset: str) -> list[dict[str, Any]]:
    queries = _load_json(test_path(draft_root, dataset))
    if not isinstance(queries, list):
        raise ValueError(f"{dataset} test data is not a list.")
    return queries


def load_instructions(
    draft_root: Path, dataset: str, condition: str
) -> dict[str, dict[str, Any]]:
    payload = _load_json(instruction_path(draft_root, dataset, condition))
    if not isinstance(payload, dict):
        raise ValueError(f"{dataset} {condition} instructions are not an object.")
    return {str(key): value for key, value in payload.items()}


def candidate_docs(
    query: dict[str, Any],
    instructions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in query.get("Tool_dic") or []:
        key = str(item.get("ID"))
        document = instructions.get(key)
        if not isinstance(document, dict):
            continue
        tool_name = str(document.get("tool_name") or "")
        if not tool_name or tool_name in seen:
            continue
        seen.add(tool_name)
        docs.append(document)
    return docs


def gold_apis(query: dict[str, Any]) -> list[str]:
    return [str(name) for name in (query.get("relevant APIs") or [])]


def parameter_entries(document: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for field in ("required_parameters", "optional_parameters"):
        for item in document.get(field) or []:
            if isinstance(item, dict):
                entries.append(item)
    names = {str(item.get("name") or "") for item in entries}
    for placeholder in path_param_names(str(document.get("url") or "")):
        if placeholder not in names:
            entries.append(
                {
                    "name": placeholder,
                    "required": True,
                    "schema": {"type": "string"},
                    "description": f"Path parameter {placeholder}.",
                }
            )
    return entries


def openai_tools(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tools: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for document in documents:
        api_id = str(document.get("ID"))
        function_name = f"api_{api_id}"
        mapping[function_name] = document
        properties: dict[str, Any] = {}
        required: list[str] = []
        path_names = set(path_param_names(str(document.get("url") or "")))
        documented_required = {
            str(item.get("name") or "")
            for item in document.get("required_parameters") or []
            if isinstance(item, dict)
        }
        for item in parameter_entries(document):
            name = str(item.get("name") or "")
            if not name:
                continue
            schema = item.get("schema") if isinstance(item.get("schema"), dict) else {}
            type_name = str(schema.get("type") or item.get("type") or "string")
            if type_name not in {"string", "integer", "number", "boolean", "array"}:
                type_name = "string"
            description = str(
                item.get("description") or schema.get("description") or name
            )
            properties[name] = {"type": type_name, "description": description[:400]}
            if name in path_names or name in documented_required or item.get("required") in {
                True,
                "true",
            }:
                required.append(name)
        description = str(document.get("description") or "")
        example = document.get("example")
        if example:
            description += "\nExample: " + json_preview(example)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": f"{document.get('tool_name')}: {description}"[:8000],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": sorted(set(required)),
                    },
                },
            }
        )
    return tools, mapping


def json_preview(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)[:500]


def correct_path(executed: list[str], gold: list[str]) -> bool:
    remaining = iter(executed)
    return all(any(candidate == target for candidate in remaining) for target in gold)
