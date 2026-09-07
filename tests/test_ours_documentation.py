"""Gates for the Ours documentation builder.

The two properties that make the condition comparable at all: it must not
rewrite any identifier the agent has to send, and it must not read anything
about the evaluation queries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooldoc_nir.ours_documentation import build_documentation, build_guideline

RELEASED_GUIDELINE = {
    "name": "Search Videos",
    "description": "Search videos. The query must be at least 3 characters.",
    "required_parameters": [
        {"name": "query", "type": "STRING", "description": "Search text."}
    ],
    "optional_parameters": [
        {
            "name": "page",
            "type": "NUMBER",
            "description": "Page number. Maximum 100.",
            "default": "1",
        }
    ],
}

SNAPSHOT_API = {
    "name": "Search Videos",
    "description": "Search videos across the catalogue.",
    "method": "get",
    "required_parameters": [
        {"name": "query", "type": "STRING", "description": "Search text."},
        {"name": "apikey", "type": "STRING", "description": "Account key."},
    ],
    "optional_parameters": [
        {
            "name": "page",
            "type": "NUMBER",
            "description": "Page number. Maximum 100.",
            "default": "1",
        }
    ],
    "schema": {"items": [{"id": 1, "title": "clip", "duration": 1.5}]},
}


@pytest.fixture()
def pinned_roots(tmp_path: Path) -> tuple[Path, Path]:
    draft_root = tmp_path / "draft"
    instruction = draft_root / "dataset" / "ToolBench" / "tool_instruction"
    instruction.mkdir(parents=True)
    (instruction / "Initial.json").write_text(
        json.dumps(
            {
                "1": {
                    "ID": 1,
                    "category": "Data Feed",
                    "tool_name": "Vimeo",
                    "tool_description": "Video platform.",
                    "tool_guidelines": {"Search Videos": RELEASED_GUIDELINE},
                }
            }
        ),
        encoding="utf-8",
    )
    stable_root = tmp_path / "toolenv"
    tools = stable_root / "tools" / "Data_Feed"
    tools.mkdir(parents=True)
    (tools / "vimeo.json").write_text(
        json.dumps({"tool_name": "Vimeo", "api_list": [SNAPSHOT_API]}),
        encoding="utf-8",
    )
    return draft_root, stable_root


def test_identifiers_are_never_rewritten(pinned_roots: tuple[Path, Path]) -> None:
    draft_root, stable_root = pinned_roots
    documentation, report = build_documentation(
        draft_root=draft_root, stable_root=stable_root
    )
    guideline = documentation["1"]["tool_guidelines"]["Search Videos"]

    assert guideline["name"] == "Search Videos"
    assert [item["name"] for item in guideline["required_parameters"]] == [
        "query",
        "apikey",
    ]
    assert [item["name"] for item in guideline["optional_parameters"]] == ["page"]
    assert guideline["optional_parameters"][0]["default"] == "1"
    assert report["provenance"] == {"snapshot": 1}


def test_the_response_contract_and_constraints_are_stated_explicitly(
    pinned_roots: tuple[Path, Path]
) -> None:
    draft_root, stable_root = pinned_roots
    documentation, _ = build_documentation(
        draft_root=draft_root, stable_root=stable_root
    )
    guideline = documentation["1"]["tool_guidelines"]["Search Videos"]

    assert guideline["method"] == "GET"
    assert guideline["response_contract"] == {
        "items": "array",
        "items.duration": "number",
        "items.id": "integer",
        "items.title": "string",
    }
    # Only the sentence that states a constraint is surfaced.
    page = guideline["optional_parameters"][0]
    assert page["constraints"] == ["Maximum 100."]


def test_a_snapshot_required_parameter_missing_upstream_is_reported(
    pinned_roots: tuple[Path, Path]
) -> None:
    draft_root, stable_root = pinned_roots
    _, report = build_documentation(
        draft_root=draft_root, stable_root=stable_root
    )

    assert report["schema_identifier_differences"] == 1
    assert report["schema_identifier_examples"][0]["snapshot_only"] == ["apikey"]


def test_building_the_documentation_reads_no_evaluation_query(
    pinned_roots: tuple[Path, Path]
) -> None:
    draft_root, stable_root = pinned_roots
    # There is deliberately no G3.json, DRAFT.json or run artifact in the
    # fixture, so a build that succeeds cannot have consulted one.
    assert not (draft_root / "dataset" / "ToolBench" / "test_data").exists()
    documentation, _ = build_documentation(
        draft_root=draft_root, stable_root=stable_root
    )
    assert set(documentation) == {"1"}


def test_a_missing_snapshot_falls_back_to_the_released_document() -> None:
    guideline, provenance = build_guideline(
        released=RELEASED_GUIDELINE,
        snapshot=None,
        category="Data Feed",
        tool_name="Vimeo",
    )

    assert provenance == "released_only"
    assert "response_contract" not in guideline
    assert [item["name"] for item in guideline["required_parameters"]] == [
        "query"
    ]
