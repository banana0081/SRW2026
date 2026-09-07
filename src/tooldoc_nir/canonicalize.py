from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .models import (
    ApiDocument,
    ApiParameter,
    CanonicalParameter,
    CanonicalProfile,
    StructuralDelta,
    normalize_identifier,
)


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CONSTRAINT_RE = re.compile(
    r"\b("
    r"must|should|required|only|cannot|can't|do not|don't|ignored|"
    r"maximum|minimum|at least|at most|valid values?|one of|"
    r"supported|not supported|defaults?|if .+ then|unless|"
    r"mutually exclusive|instead of|in combination with"
    r")\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "api",
    "are",
    "been",
    "before",
    "being",
    "can",
    "data",
    "default",
    "defaults",
    "each",
    "for",
    "from",
    "get",
    "given",
    "has",
    "have",
    "into",
    "its",
    "may",
    "method",
    "not",
    "only",
    "optional",
    "parameter",
    "parameters",
    "request",
    "required",
    "response",
    "result",
    "results",
    "return",
    "returns",
    "that",
    "the",
    "their",
    "this",
    "tool",
    "use",
    "used",
    "using",
    "value",
    "values",
    "when",
    "where",
    "which",
    "will",
    "with",
    "you",
    "your",
}
_JSON_SCHEMA_TYPES = {
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
}


def clean_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value or "").strip()


def _shorten_description(value: str, max_chars: int = 900) -> str:
    cleaned = clean_text(value)
    if len(cleaned) <= max_chars:
        return cleaned
    boundary = cleaned.rfind(".", 0, max_chars)
    if boundary >= max_chars // 2:
        return cleaned[: boundary + 1]
    return cleaned[: max_chars - 1].rstrip() + "…"


def _extract_constraints(description: str, limit: int = 4) -> list[str]:
    constraints: list[str] = []
    for sentence in _SENTENCE_RE.split(description or ""):
        cleaned = clean_text(sentence).strip("-* ")
        if cleaned and _CONSTRAINT_RE.search(cleaned):
            constraints.append(cleaned)
        if len(constraints) >= limit:
            break
    return constraints


def _canonical_parameter(
    parameter: ApiParameter, *, required: bool
) -> CanonicalParameter:
    return CanonicalParameter(
        name=normalize_identifier(parameter.name),
        type=clean_text(parameter.type).lower(),
        description=_shorten_description(parameter.description, max_chars=360),
        required=required,
        default=parameter.default,
        constraints=_extract_constraints(parameter.description),
    )


def _output_fields(value: Any, prefix: str = "", depth: int = 0) -> list[str]:
    if value is None or depth > 2:
        return []
    if isinstance(value, dict):
        fields: list[str] = []
        for key, nested in value.items():
            name = normalize_identifier(str(key))
            path = f"{prefix}.{name}" if prefix else name
            fields.append(path)
            fields.extend(_output_fields(nested, path, depth + 1))
        return fields
    if isinstance(value, list) and value:
        return _output_fields(value[0], prefix, depth + 1)
    return []


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        declared = value.casefold().strip()
        aliases = {
            "bool": "boolean",
            "boolean": "boolean",
            "dict": "object",
            "float": "number",
            "int": "integer",
            "integer": "integer",
            "list": "array",
            "number": "number",
            "object": "object",
            "str": "string",
            "string": "string",
        }
        return aliases.get(declared, "string")
    return type(value).__name__.casefold()


def _is_json_schema(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    declared_type = str(value.get("type", "")).casefold()
    return bool(
        "$schema" in value
        or "properties" in value
        or (
            declared_type in _JSON_SCHEMA_TYPES
            and ("items" in value or "required" in value)
        )
    )


def _schema_output_type_map(
    schema: dict[str, Any],
    prefix: str = "",
    depth: int = 0,
) -> dict[str, str]:
    if depth > 5:
        return {}
    declared_type = str(schema.get("type", "unknown")).casefold()
    if "properties" in schema:
        declared_type = "object"
    result: dict[str, str] = {}
    if declared_type == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, nested in properties.items():
                path = (
                    f"{prefix}.{normalize_identifier(str(key))}"
                    if prefix
                    else normalize_identifier(str(key))
                )
                nested_schema = nested if isinstance(nested, dict) else {}
                nested_type = str(
                    nested_schema.get("type", "unknown")
                ).casefold()
                if "properties" in nested_schema:
                    nested_type = "object"
                elif "items" in nested_schema:
                    nested_type = "array"
                result[path] = normalize_type_name(nested_type)
                result.update(
                    _schema_output_type_map(
                        nested_schema,
                        path,
                        depth + 1,
                    )
                )
    elif declared_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            result.update(
                _schema_output_type_map(items, prefix, depth + 1)
            )
    return result


def normalize_type_name(value: str) -> str:
    aliases = {
        "bool": "boolean",
        "dict": "object",
        "float": "number",
        "int": "integer",
        "list": "array",
        "str": "string",
    }
    cleaned = (value or "unknown").casefold().strip()
    return aliases.get(cleaned, cleaned or "unknown")


def _output_type_map(
    value: Any, prefix: str = "", depth: int = 0
) -> dict[str, str]:
    if value is None or depth > 2:
        return {}
    if _is_json_schema(value):
        return _schema_output_type_map(value, prefix, depth)
    if isinstance(value, dict):
        fields: dict[str, str] = {}
        for key, nested in value.items():
            name = normalize_identifier(str(key))
            path = f"{prefix}.{name}" if prefix else name
            fields[path] = _value_type(nested)
            fields.update(_output_type_map(nested, path, depth + 1))
        return fields
    if isinstance(value, list) and value:
        return _output_type_map(value[0], prefix, depth + 1)
    return {}


def _functional_terms(values: Iterable[str], limit: int = 36) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        expanded = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value or "")
        expanded = expanded.replace("_", " ").replace("-", " ")
        for match in _TOKEN_RE.finditer(expanded):
            term = match.group(0).casefold()
            if term in _STOPWORDS or term in seen or term.isdigit():
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= limit:
                return terms
    return terms


def canonicalize(document: ApiDocument) -> CanonicalProfile:
    required = [
        _canonical_parameter(parameter, required=True)
        for parameter in document.required_parameters
    ]
    optional = [
        _canonical_parameter(parameter, required=False)
        for parameter in document.optional_parameters
    ]

    constraints: list[str] = []
    for parameter in [*required, *optional]:
        constraints.extend(
            f"{parameter.name}: {constraint}"
            for constraint in parameter.constraints
        )

    purpose = _shorten_description(document.api_description)
    if not purpose:
        purpose = clean_text(
            f"{document.api_name} operation from {document.tool_name}"
        )

    output_types = _output_type_map(document.template_response)
    return CanonicalProfile(
        source_hash=document.source_hash,
        category=clean_text(document.category_name),
        tool_name=clean_text(document.tool_name),
        api_name=clean_text(document.api_name),
        method=clean_text(document.method).upper(),
        purpose=purpose,
        required_inputs=required,
        optional_inputs=optional,
        output_fields=sorted(output_types),
        output_types=output_types,
        constraints=list(dict.fromkeys(constraints))[:24],
        functional_terms=_functional_terms(
            [
                document.tool_name,
                document.api_name,
                document.api_description,
            ]
        ),
    )


def raw_text(document: ApiDocument) -> str:
    """Faithful baseline: serialize the supplied ToolBench document."""
    return json.dumps(
        document.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=False,
    )


def _parameter_text(parameter: CanonicalParameter) -> str:
    pieces = [parameter.name]
    if parameter.type:
        pieces.append(f"type={parameter.type}")
    if parameter.default not in (None, ""):
        pieces.append(f"default={parameter.default}")
    if parameter.description:
        pieces.append(parameter.description)
    return " | ".join(pieces)


def canonical_text(profile: CanonicalProfile) -> str:
    sections = [
        f"Tool: {profile.tool_name}",
        f"API: {profile.api_name}",
        f"Category: {profile.category}",
        f"HTTP method: {profile.method or 'unspecified'}",
        f"Purpose: {profile.purpose}",
    ]
    if profile.required_inputs:
        sections.append(
            "Required inputs: "
            + "; ".join(_parameter_text(item) for item in profile.required_inputs)
        )
    if profile.optional_inputs:
        sections.append(
            "Optional inputs: "
            + "; ".join(_parameter_text(item) for item in profile.optional_inputs)
        )
    if profile.output_fields:
        sections.append(
            "Output fields: "
            + ", ".join(
                f"{field}:{profile.output_types.get(field, 'unknown')}"
                for field in profile.output_fields
            )
        )
    if profile.constraints:
        sections.append("Constraints: " + "; ".join(profile.constraints))
    if profile.functional_terms:
        sections.append(
            "Functional terms: " + ", ".join(profile.functional_terms)
        )
    return "\n".join(sections)


def structural_delta(
    target: CanonicalProfile, competitor: CanonicalProfile
) -> StructuralDelta:
    def names(parameters: list[CanonicalParameter]) -> set[str]:
        return {parameter.name for parameter in parameters}

    target_required = names(target.required_inputs)
    competitor_required = names(competitor.required_inputs)
    target_optional = names(target.optional_inputs)
    competitor_optional = names(competitor.optional_inputs)

    return StructuralDelta(
        target_key=target.key,
        competitor_key=competitor.key,
        unique_required_inputs=sorted(target_required - competitor_required),
        unique_optional_inputs=sorted(target_optional - competitor_optional),
        unique_output_fields=sorted(
            set(target.output_fields) - set(competitor.output_fields)
        ),
        unique_functional_terms=[
            term
            for term in target.functional_terms
            if term not in set(competitor.functional_terms)
        ][:18],
        differing_method=(
            target.method
            if target.method
            and competitor.method
            and target.method != competitor.method
            else ""
        ),
    )


def delta_text(delta: StructuralDelta) -> str:
    sections: list[str] = []
    if delta.unique_functional_terms:
        sections.append(
            "Distinct functions: " + ", ".join(delta.unique_functional_terms)
        )
    if delta.unique_required_inputs:
        sections.append(
            "Distinct required inputs: "
            + ", ".join(delta.unique_required_inputs)
        )
    if delta.unique_optional_inputs:
        sections.append(
            "Distinct optional inputs: "
            + ", ".join(delta.unique_optional_inputs)
        )
    if delta.unique_output_fields:
        sections.append(
            "Distinct output fields: " + ", ".join(delta.unique_output_fields)
        )
    if delta.differing_method:
        sections.append("Distinct HTTP method: " + delta.differing_method)
    return "\n".join(sections)

