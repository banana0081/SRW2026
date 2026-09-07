"""Offline gates for the held-out temporal pilot."""

from __future__ import annotations

import json

from tooldoc_nir.controlled import ControlledApiServer
from tooldoc_nir.draft_agent_reproduction import CostBudget
from tooldoc_nir.dynamic_models import LivenessState
from tooldoc_nir.temporal_pilot import (
    CONTRACT_FAMILIES,
    DEFAULT_AGENT_MODEL,
    DOWN_FAMILIES,
    build_temporal_cases,
    collect_observations,
    execute_call,
    gold_endpoint,
    held_out_query,
    is_callable_now,
    make_json_respond,
    oracle_call,
    run_pilot,
    scripted_oracle_calls,
    synthetic_documents,
)


def test_held_out_query_does_not_name_the_outage() -> None:
    document = synthetic_documents(1)[0]
    query = held_out_query(document, has_fallback=True)
    lowered = query.lower()
    for token in ("unavailable", "down", "503", "404", "fallback", "outage"):
        assert token not in lowered


def test_dev_and_test_do_not_share_an_api() -> None:
    cases = build_temporal_cases(
        synthetic_documents(), seeds_per_mutation=1, random_seed=0
    )
    dev = {case.primary.baseline.key for case in cases if case.split == "dev"}
    test = {case.primary.baseline.key for case in cases if case.split == "test"}
    assert dev
    assert test
    assert dev.isdisjoint(test)


def test_down_cases_have_a_live_fallback() -> None:
    cases = build_temporal_cases(
        synthetic_documents(), seeds_per_mutation=1, random_seed=0
    )
    down = [case for case in cases if case.family in DOWN_FAMILIES]
    assert down
    for case in down:
        assert case.fallback is not None
        assert case.fallback.expected_liveness is LivenessState.HEALTHY
        assert gold_endpoint(case) == "fallback"


def test_oracle_succeeds_on_callable_dev_cases() -> None:
    cases = build_temporal_cases(
        synthetic_documents(), seeds_per_mutation=1, random_seed=0
    )
    selected = [case for case in cases if case.split == "dev"]
    scripted = scripted_oracle_calls(selected, ("raw", "ours"))
    report = run_pilot(
        cases=selected,
        split="dev",
        scripted_calls=scripted,
        run_draft=False,
        run_ours=True,
    )
    callable_traces = [
        row
        for row in report["traces"]
        if row["callable_now"] and row["condition"] == "raw"
    ]
    assert callable_traces
    assert all(row["http_success"] for row in callable_traces)
    for row in report["traces"]:
        assert "expected_events" not in row["query"]


def test_server_reset_replays_the_availability_pattern() -> None:
    cases = build_temporal_cases(
        synthetic_documents(), seeds_per_mutation=1, random_seed=0
    )
    down = next(case for case in cases if case.family == "unavailable")
    with ControlledApiServer([down.primary]) as server:
        first = collect_observations(server, down, budget=3)
        server.reset_counts()
        second = collect_observations(server, down, budget=3)
    assert [item.status_code for item in first] == [
        item.status_code for item in second
    ]
    assert {item.status_code for item in first} == {503}


def test_add_required_oracle_uses_runtime_not_stale_docs() -> None:
    cases = build_temporal_cases(
        synthetic_documents(), seeds_per_mutation=1, random_seed=0
    )
    case = next(item for item in cases if item.family == "add_required")
    call = oracle_call(case.primary)
    assert "tooldoc_locale" in call.arguments
    with ControlledApiServer([case.primary]) as server:
        status, _detail, schema_valid = execute_call(server, case, call)
    assert schema_valid
    assert status == 200
    assert is_callable_now(case.primary)


def test_remove_parameter_oracle_does_not_resend_deleted_field() -> None:
    cases = build_temporal_cases(
        synthetic_documents(24), seeds_per_mutation=3, random_seed=0
    )
    case = next(item for item in cases if item.case_id == "remove_parameter_002")
    call = oracle_call(case.primary)
    assert "q" not in call.arguments
    with ControlledApiServer([case.primary]) as server:
        status, detail, schema_valid = execute_call(server, case, call)
    assert schema_valid, detail
    assert status == 200
    cases = build_temporal_cases(
        synthetic_documents(), seeds_per_mutation=1, random_seed=0
    )
    case = next(item for item in cases if item.family == "add_required")
    call = oracle_call(case.primary)
    assert "tooldoc_locale" in call.arguments
    with ControlledApiServer([case.primary]) as server:
        status, _detail, schema_valid = execute_call(server, case, call)
    assert schema_valid
    assert status == 200
    assert is_callable_now(case.primary)


def test_default_agent_is_deepseek_not_gpt_mini() -> None:
    assert DEFAULT_AGENT_MODEL.startswith("deepseek/")
    assert "gpt-4o-mini" not in DEFAULT_AGENT_MODEL


def test_parallel_workers_match_sequential_oracle_success() -> None:
    cases = build_temporal_cases(
        synthetic_documents(), seeds_per_mutation=1, random_seed=0
    )
    selected = [case for case in cases if case.split == "dev"]
    scripted = scripted_oracle_calls(selected, ("raw", "ours"))
    sequential = run_pilot(
        cases=selected,
        split="dev",
        scripted_calls=scripted,
        run_draft=False,
        run_ours=True,
        workers=1,
    )
    parallel = run_pilot(
        cases=selected,
        split="dev",
        scripted_calls=scripted,
        run_draft=False,
        run_ours=True,
        workers=4,
    )
    assert sequential["cases"] == parallel["cases"]
    assert sequential["conditions"]["ours"]["http_success"] == (
        parallel["conditions"]["ours"]["http_success"]
    )
    assert "ours_minus_raw" in parallel["paired"]["all"]


class _FakeOpenRouter:
    def chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "endpoint": "primary",
                                "method": "GET",
                                "arguments": {"q": "cats"},
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 8},
            "model": "deepseek/deepseek-v4-flash-0731",
            "provider": "Baidu",
            "_transport": {"request_id": "fake"},
        }


def test_json_respond_parses_deepseek_payload() -> None:
    budget = CostBudget(1.0)
    respond = make_json_respond(
        client=_FakeOpenRouter(),  # type: ignore[arg-type]
        model=DEFAULT_AGENT_MODEL,
        budget=budget,
    )
    parsed = respond(
        [{"role": "user", "content": "return json"}],
        "temporal_eval",
    )
    assert parsed["endpoint"] == "primary"
    assert parsed["arguments"]["q"] == "cats"
    assert budget.spent_usd == 0.001
