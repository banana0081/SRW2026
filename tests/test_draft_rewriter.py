"""Offline gates for the released DRAFT Explorer/Analyzer/Rewriter port."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from tooldoc_nir.draft_rewriter import (
    DraftPrompts,
    DraftRewriteFailure,
    change_name,
    cosine_similarity,
    rewrite_api_description,
    rewrite_from_observations,
    rewrite_tool,
    standardize,
)

API_INFO = {
    "name": "Search Videos",
    "description": (
        "Search videos across the catalogue by keyword and return clip "
        "identifiers, titles and durations for matching items."
    ),
    "required_parameters": [
        {"name": "query", "type": "STRING", "description": "Search text."}
    ],
    "optional_parameters": [],
}


def fake_embed(text: str) -> list[float]:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest]


class ScriptedRespond:
    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, messages: list[dict[str, Any]], stage: str
    ) -> dict[str, Any]:
        self.calls.append((stage, messages[-1]["content"]))
        if not self.answers:
            raise AssertionError(f"No scripted answer left for {stage}")
        return self.answers.pop(0)


def test_standardize_keeps_the_released_caret_quirk() -> None:
    assert standardize("search-videos") == "search_videos"
    # The released character class treats `^` as a kept character.
    assert standardize("search^videos") == "search^videos"


def test_change_name_matches_the_reserved_list() -> None:
    assert change_name("id") == "is_id"
    assert change_name("Search") == "Search"


def test_analyzer_prompt_still_contains_the_unfilled_placeholder() -> None:
    prompts = DraftPrompts.load()
    filled = prompts.analyzer.replace("{Tool Description}", "DESC").replace(
        "{usage_example}", "EXAMPLE"
    )
    assert "{tool_description}" in filled


def test_rewrite_stops_after_the_first_episode_when_delta_is_high() -> None:
    prompts = DraftPrompts.load()
    respond = ScriptedRespond(
        [
            {"User Query": "find clips about cats", "Parameters": {"query": "cats"}},
            {
                "Suggestions for tool description": "Mention video results.",
            },
            {
                "Rewritten description": (
                    "Search videos across the catalogue by keyword and return clip "
                    "identifiers, titles and durations for matching items."
                ),
                "Suggestions for exploring": "Try another topic.",
            },
        ]
    )
    executed: list[dict[str, Any]] = []

    def execute(payload: dict[str, Any]) -> dict[str, Any]:
        executed.append(payload)
        return {"error": "", "response": {"items": []}}

    result = rewrite_api_description(
        prompts=prompts,
        category="Data Feed",
        tool_name="Vimeo",
        api_info=API_INFO,
        execute=execute,
        respond=respond,
        embed=fake_embed,
        episodes=5,
    )

    assert result.stopped_early
    assert len(result.episodes) == 1
    assert result.episodes[0].delta is not None
    assert result.episodes[0].delta > 0.75
    assert executed[0]["api_name"] == "search_videos"


def test_second_episode_sends_the_mangled_api_name() -> None:
    prompts = DraftPrompts.load()
    # Force a low delta on the first rewrite by changing the text substantially,
    # then stop on the second.
    respond = ScriptedRespond(
        [
            {"User Query": "find clips about cats", "Parameters": {"query": "cats"}},
            {"Suggestions for tool description": "add result fields"},
            {
                "Rewritten description": (
                    "Completely different wording about retrieving clips "
                    "by keyword from the catalogue with paging."
                ),
                "Suggestions for exploring": "page numbers",
            },
            {
                "User Query": "find clips about dogs in london",
                "Parameters": {"query": "dogs"},
            },
            {"Suggestions for tool description": "keep going"},
            {
                "Rewritten description": (
                    "Completely different wording about retrieving clips "
                    "by keyword from the catalogue with paging."
                ),
                "Suggestions for exploring": "stop",
            },
        ]
    )
    names: list[str] = []

    def execute(payload: dict[str, Any]) -> dict[str, Any]:
        names.append(payload["api_name"])
        return {"error": "", "response": {"items": []}}

    result = rewrite_api_description(
        prompts=prompts,
        category="Data Feed",
        tool_name="Vimeo",
        api_info=API_INFO,
        execute=execute,
        respond=respond,
        embed=fake_embed,
        episodes=5,
    )

    explorer_prompts = [prompt for stage, prompt in respond.calls if stage == "explorer"]
    assert "Search Videos" in explorer_prompts[0]
    assert "search_videos" in explorer_prompts[1]
    assert names == ["search_videos", "search_videos"]
    assert result.episodes[0].delta is not None


def test_missing_keys_are_not_silently_dropped() -> None:
    prompts = DraftPrompts.load()

    def respond(messages: list[dict[str, Any]], stage: str) -> dict[str, Any]:
        del messages, stage
        return {"wrong": True}

    with pytest.raises(DraftRewriteFailure):
        rewrite_api_description(
            prompts=prompts,
            category="Data Feed",
            tool_name="Vimeo",
            api_info=API_INFO,
            execute=lambda payload: payload,
            respond=respond,
            embed=fake_embed,
            episodes=1,
        )


def test_rewrite_from_observations_never_executes_and_leaves_schema() -> None:
    prompts = DraftPrompts.load()
    respond = ScriptedRespond(
        [
            {"Suggestions for tool description": "mention empty results"},
            {
                "Rewritten description": "Search videos and return items.",
                "Suggestions for exploring": "try paging",
            },
        ]
    )
    result = rewrite_from_observations(
        prompts=prompts,
        category="Data Feed",
        tool_name="Vimeo",
        api_info=dict(API_INFO),
        observations=[
            {
                "query": "probe",
                "parameters": {"query": "cats"},
                "api_response": {"error": "", "response": {"items": []}},
            }
        ],
        respond=respond,
        embed=fake_embed,
    )
    assert [stage for stage, _ in respond.calls] == ["analyzer", "rewriter"]
    assert "Search videos" in result.history()[0]
    tool = {
        "category": "Data Feed",
        "tool_name": "Vimeo",
        "tool_description": "Video platform.",
        "tool_guidelines": {"Search Videos": dict(API_INFO)},
    }

    def execute(payload: dict[str, Any]) -> dict[str, Any]:
        return {"error": "", "response": payload}

    updated, _ = rewrite_tool(
        prompts=prompts,
        tool=tool,
        execute=execute,
        respond=ScriptedRespond(
            [
                {
                    "User Query": "find clips about cats",
                    "Parameters": {"query": "cats"},
                },
                {"Suggestions for tool description": "ok"},
                {
                    "Rewritten description": "Search videos.",
                    "Suggestions for exploring": "ok",
                },
                {"tool_description": "Video platform for clips."},
            ]
        ),
        embed=fake_embed,
        episodes=1,
    )
    guideline = updated["tool_guidelines"]["Search Videos"]
    assert guideline["name"] == "Search Videos"
    assert guideline["required_parameters"] == API_INFO["required_parameters"]


def test_cosine_similarity_is_zero_for_empty_vectors() -> None:
    assert cosine_similarity([], [1.0]) == 0.0
