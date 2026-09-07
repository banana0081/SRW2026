"""Build the Ours condition's tool documentation.

Raw is the released `Initial.json` and DRAFT is the released `DRAFT.json`,
which rewrites prose and appends a worked example while leaving every schema
untouched. Ours instead canonicalizes the pinned ToolEnv snapshot that the
evaluation backend actually serves: parameter names, types and defaults are
copied verbatim so a call stays valid, while the purpose is normalized,
constraints buried in prose are surfaced, and the response contract is stated
explicitly.

Two rules keep this honest:

- No G3 outcome, gold path or query is read here. The only inputs are the
  released documentation and the pinned snapshot.
- Identifiers are never rewritten. Canonicalization may add or reorganize
  description material, never change what the agent must send.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from .canonicalize import _extract_constraints, _shorten_description, canonicalize
from .draft_agent_reproduction import (
    DEFAULT_DRAFT_ROOT,
    DEFAULT_STABLE_ROOT,
    StableExecutionAdapter,
    _dump_json,
    _load_json,
    _process_name,
)
from .models import ApiDocument

MAX_CONSTRAINTS = 6


def _tool_document_path(stable_root: Path, category: str, tool_name: str) -> Path:
    return (
        stable_root
        / "tools"
        / StableExecutionAdapter._category(category)
        / f"{_process_name(tool_name)}.json"
    )


def _snapshot_api(
    payload: dict[str, Any], api_name: str
) -> dict[str, Any] | None:
    target = _process_name(api_name)
    for api in payload.get("api_list") or []:
        if isinstance(api, dict) and _process_name(str(api.get("name") or "")) == target:
            return api
    return None


def _canonical_parameters(
    documented: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep every identifier verbatim; only clarify the prose."""
    canonical: list[dict[str, Any]] = []
    for parameter in documented:
        if not isinstance(parameter, dict):
            continue
        description = str(parameter.get("description") or "")
        entry: dict[str, Any] = {
            "name": parameter.get("name"),
            "type": parameter.get("type"),
            "description": _shorten_description(description, max_chars=360),
        }
        if parameter.get("default") not in (None, ""):
            entry["default"] = parameter["default"]
        constraints = _extract_constraints(description)
        if constraints:
            entry["constraints"] = constraints
        canonical.append(entry)
    return canonical


def _response_contract(document: ApiDocument) -> dict[str, str]:
    profile = canonicalize(document)
    return {field: profile.output_types[field] for field in profile.output_fields}


def build_guideline(
    *,
    released: dict[str, Any],
    snapshot: dict[str, Any] | None,
    category: str,
    tool_name: str,
) -> tuple[dict[str, Any], str]:
    """Return the Ours guideline and how much of it came from the snapshot."""
    source = snapshot if snapshot is not None else released
    provenance = "snapshot" if snapshot is not None else "released_only"
    description = str(
        source.get("description") or released.get("description") or ""
    )
    guideline: dict[str, Any] = {
        "name": released.get("name"),
        "description": _shorten_description(description),
        "required_parameters": _canonical_parameters(
            source.get("required_parameters") or []
        ),
        "optional_parameters": _canonical_parameters(
            source.get("optional_parameters") or []
        ),
    }
    method = str(source.get("method") or "").upper()
    if method:
        guideline["method"] = method

    api_constraints = _extract_constraints(description, limit=MAX_CONSTRAINTS)
    if api_constraints:
        guideline["constraints"] = api_constraints

    if snapshot is not None:
        document = ApiDocument(
            category_name=category,
            tool_name=tool_name,
            api_name=str(released.get("name") or ""),
            api_description=description,
            required_parameters=snapshot.get("required_parameters") or [],
            optional_parameters=snapshot.get("optional_parameters") or [],
            method=method,
            template_response=snapshot.get("schema") or None,
        )
        contract = _response_contract(document)
        if contract:
            guideline["response_contract"] = contract
    return guideline, provenance


def _identifier_names(parameters: list[dict[str, Any]]) -> set[str]:
    return {
        str(parameter.get("name"))
        for parameter in parameters
        if isinstance(parameter, dict) and parameter.get("name")
    }


def build_documentation(
    *,
    draft_root: Path,
    stable_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    released = _load_json(
        draft_root
        / "dataset"
        / "ToolBench"
        / "tool_instruction"
        / "Initial.json"
    )
    documentation: dict[str, Any] = {}
    provenance = Counter()
    identifier_changes: list[dict[str, Any]] = []
    contracts = 0
    apis = 0

    for tool_id, entry in released.items():
        category = str(entry.get("category") or "")
        tool_name = str(entry.get("tool_name") or "")
        path = _tool_document_path(stable_root, category, tool_name)
        payload = _load_json(path) if path.exists() else None
        if payload is None:
            provenance["tool_without_snapshot"] += 1
        guidelines: dict[str, Any] = {}
        for api_name, guideline in (entry.get("tool_guidelines") or {}).items():
            apis += 1
            snapshot = (
                _snapshot_api(payload, api_name) if payload is not None else None
            )
            built, source = build_guideline(
                released=guideline,
                snapshot=snapshot,
                category=category,
                tool_name=tool_name,
            )
            provenance[source] += 1
            if "response_contract" in built:
                contracts += 1
            if snapshot is not None:
                for field in ("required_parameters", "optional_parameters"):
                    before = _identifier_names(guideline.get(field) or [])
                    after = _identifier_names(built[field])
                    if before != after:
                        identifier_changes.append(
                            {
                                "tool_id": tool_id,
                                "api": api_name,
                                "field": field,
                                "released_only": sorted(before - after),
                                "snapshot_only": sorted(after - before),
                            }
                        )
            guidelines[api_name] = built
        documentation[tool_id] = {
            "ID": entry.get("ID"),
            "category": entry.get("category"),
            "tool_name": entry.get("tool_name"),
            "tool_description": _shorten_description(
                str(entry.get("tool_description") or "")
            ),
            "tool_guidelines": guidelines,
        }

    report = {
        "tools": len(documentation),
        "apis": apis,
        "provenance": dict(provenance),
        "apis_with_response_contract": contracts,
        "schema_identifier_differences": len(identifier_changes),
        "schema_identifier_examples": identifier_changes[:20],
        "policy": (
            "Parameter and API identifiers are copied verbatim from the "
            "documentation the backend serves. Ours adds normalized purpose "
            "text, surfaced constraints and an explicit response contract; it "
            "reads no G3 query, gold path or outcome."
        ),
    }
    return documentation, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Ours tool documentation from pinned inputs."
    )
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--stable-root", type=Path, default=DEFAULT_STABLE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/documentation/Ours.json"),
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    documentation, report = build_documentation(
        draft_root=args.draft_root,
        stable_root=args.stable_root,
    )
    _dump_json(args.output, documentation)
    report["output"] = str(args.output)
    _dump_json(args.report or args.output.with_name("Ours_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
