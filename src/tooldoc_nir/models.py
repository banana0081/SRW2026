from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_identifier(value: str) -> str:
    """Match identifiers despite ToolBench casing and separator variation."""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


class ApiParameter(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: str = ""
    description: str = ""
    default: Any = None

    @field_validator("name", "type", "description", mode="before")
    @classmethod
    def coerce_text(cls, value: Any) -> str:
        return "" if value is None else str(value)


class ApiDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    category_name: str = ""
    tool_name: str
    api_name: str
    api_description: str = ""
    required_parameters: list[ApiParameter] = Field(default_factory=list)
    optional_parameters: list[ApiParameter] = Field(default_factory=list)
    method: str = ""
    template_response: Any = None
    url: str = ""
    host: str = ""
    historical_status_code: int | str | None = None
    historical_schema: Any = None

    @field_validator(
        "category_name",
        "tool_name",
        "api_name",
        "api_description",
        "method",
        "url",
        "host",
        mode="before",
    )
    @classmethod
    def coerce_text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @property
    def key(self) -> tuple[str, str]:
        return (
            normalize_identifier(self.tool_name),
            normalize_identifier(self.api_name),
        )

    @property
    def source_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()


class BenchmarkExample(BaseModel):
    query_id: str
    split: str
    query: str
    candidates: list[ApiDocument]
    relevant_apis: list[tuple[str, str]]

    @property
    def relevant_keys(self) -> set[tuple[str, str]]:
        return {
            (normalize_identifier(tool), normalize_identifier(api))
            for tool, api in self.relevant_apis
        }


class CanonicalParameter(BaseModel):
    name: str
    type: str = ""
    description: str = ""
    required: bool
    default: Any = None
    constraints: list[str] = Field(default_factory=list)


class CanonicalProfile(BaseModel):
    schema_version: str = "0.1.0"
    source_hash: str
    category: str = ""
    tool_name: str
    api_name: str
    method: str = ""
    purpose: str
    required_inputs: list[CanonicalParameter] = Field(default_factory=list)
    optional_inputs: list[CanonicalParameter] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    output_types: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    functional_terms: list[str] = Field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (
            normalize_identifier(self.tool_name),
            normalize_identifier(self.api_name),
        )


class StructuralDelta(BaseModel):
    target_key: tuple[str, str]
    competitor_key: tuple[str, str]
    unique_required_inputs: list[str] = Field(default_factory=list)
    unique_optional_inputs: list[str] = Field(default_factory=list)
    unique_output_fields: list[str] = Field(default_factory=list)
    unique_functional_terms: list[str] = Field(default_factory=list)
    differing_method: str = ""

