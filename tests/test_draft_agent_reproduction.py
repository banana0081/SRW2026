"""Offline gates for the G3 calibration harness.

Every test here runs without an OpenRouter key. They exist because the first
version of this harness produced a number that could not be defended: it
patched the released driver for no reason, substituted invented decisions for
failed model calls, and reported a crash that never happened.
"""

from __future__ import annotations

import json
from pathlib import Path
import types

import pytest

from tooldoc_nir import draft_agent_reproduction as dar
from tooldoc_nir.draft_agent_reproduction import (
    COST_CAP,
    TRANSPORT,
    CallContext,
    ConditionState,
    CostBudget,
    CostCapReached,
    FailureSink,
    HarnessFailure,
    LoggedOpenRouter,
    ReleasedDriverFailure,
    SimulatorCacheMiss,
    StableExecutionAdapter,
    _execute_one,
    _extract_json,
    _make_agent_response,
    _read_concatenated_json,
    build_query_plan,
    correct_path,
    correct_path_released,
    load_released_draft_module,
    model_alias,
    record_offsets,
    record_positions,
    truncate_records,
)
from tooldoc_nir.g3_metrics import backend_success, classify_call, parameter_validity
from tooldoc_nir.openrouter import OpenRouterError
from tooldoc_nir.provenance import (
    ManifestMismatch,
    build_manifest,
    file_digest,
    fingerprint,
)
from tooldoc_nir.selection_decompose import (
    decompose_g3,
    exact_sign_test,
    paired_bootstrap,
)

DRAFT_ROOT = Path("external/DRAFT")
LEGACY_ROOT = Path("artifacts/results/legacy/draft_gpt4omini_full")


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class FakeClient:
    """Returns scripted completions and records every request."""

    def __init__(self, contents: list[str], *, cost: float = 0.001) -> None:
        self.contents = list(contents)
        self.cost = cost
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        content = self.contents.pop(0) if self.contents else "{}"
        return {
            "model": "fake/model-2026",
            "provider": "FakeProvider",
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "cost": self.cost,
            },
            "_transport": {"status_code": 200, "request_id": "req-1"},
        }


class DeadClient:
    def chat_completion(self, **kwargs: object) -> dict[str, object]:
        raise OpenRouterError("HTTP 503: upstream is down")


def make_transport(
    client: object,
    tmp_path: Path,
    *,
    budget: CostBudget | None = None,
    failures: FailureSink | None = None,
    context: CallContext | None = None,
    role: str = "agent",
) -> LoggedOpenRouter:
    return LoggedOpenRouter(
        client=client,
        model="fake/model",
        role=role,
        usage_path=tmp_path / f"usage_{role}.jsonl",
        budget=budget or CostBudget(1.0),
        context=context or CallContext(),
        failures=failures or FailureSink(),
    )


def write_fake_draft_root(root: Path, queries: list[dict[str, object]]) -> Path:
    """A minimal released-layout tree so manifests hash real files."""
    (root / "dataset" / "ToolBench" / "test_data").mkdir(
        parents=True, exist_ok=True
    )
    (root / "dataset" / "ToolBench" / "tool_instruction").mkdir(
        parents=True, exist_ok=True
    )
    (root / "Inference_DFSDT.py").write_text(
        "temperature = 0.2\ntop_p = 1\nmax_tokens = 2000\n", encoding="utf-8"
    )
    (root / "dataset" / "ToolBench" / "test_data" / "G3.json").write_text(
        json.dumps(queries), encoding="utf-8"
    )
    for condition in ("Initial", "DRAFT"):
        path = (
            root
            / "dataset"
            / "ToolBench"
            / "tool_instruction"
            / f"{condition}.json"
        )
        path.write_text(
            json.dumps({"1": {"tool_name": condition, "category": "Tools"}}),
            encoding="utf-8",
        )
    return root


def stub_draft_module(*, on_call=None) -> types.ModuleType:
    """A stand-in for the released driver that writes one record per query."""
    module = types.ModuleType("stub_released_inference")

    def task_execution(
        data_type,
        base_path,
        index,
        dataset,
        test_data,
        progress_file,
        start_index,
        total_files,
        retrieval_num,
        ind,
        model_name,
        method,
    ):
        if on_call is not None:
            on_call(method, ind)
        record = {
            "ID": ind + 1,
            "question": test_data[0]["query"],
            "final_answer": f"{method} answer",
            "check_index": 1,
            "execute_log": {
                "api_result_ls": [
                    [
                        {
                            "categoty": "Tools",
                            "tool_name": "Vimeo",
                            "api_name": "SearchVideos",
                            "parameters": {"query": "cats"},
                        }
                    ]
                ],
                "parameter_ls": [],
                "call_result_ls": ["{}"],
            },
        }
        target = Path(f"ToolBench_{data_type}_DFS_{model_name}_{method}.jsonl")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")

    module.task_execution = task_execution
    return module


# --------------------------------------------------------------------------
# The released driver is used verbatim
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DRAFT_ROOT / "Inference_DFSDT.py").exists(),
    reason="the released DRAFT checkout is not present",
)
def test_released_driver_is_loaded_without_a_behavioural_patch() -> None:
    module = load_released_draft_module(DRAFT_ROOT)
    source_path = DRAFT_ROOT / "Inference_DFSDT.py"

    assert module.__released_digest__ == file_digest(source_path)
    # The earlier "empty candidate list" patch renamed a local variable that is
    # already distinct from the `api_list` parameter, so it changed nothing.
    assert "API_list" in module.retrieval.__code__.co_varnames
    assert "candidate_api_list" not in module.retrieval.__code__.co_varnames
    assert {
        key: module.__dict__[key] for key in dar.RELEASED_DECODING
    } == dar.RELEASED_DECODING


def test_decoding_drift_in_the_released_driver_is_refused() -> None:
    module = types.ModuleType("drifted")
    module.__dict__.update({"temperature": 0.7, "top_p": 1, "max_tokens": 2000})
    with pytest.raises(HarnessFailure, match="published decoding"):
        dar.assert_released_decoding(module)


def test_no_fabricated_decision_helper_remains() -> None:
    assert not hasattr(dar, "_fallback_response")


def test_model_alias_follows_the_released_filename_convention() -> None:
    assert model_alias("openai/gpt-4o-mini-2024-07-18") == (
        "gpt-4o-mini-2024-07-18"
    )
    assert model_alias("deepseek/deepseek-chat") == "deepseek-chat"


# --------------------------------------------------------------------------
# Failures never become results
# --------------------------------------------------------------------------


def test_extract_json_accepts_markdown_fence() -> None:
    assert _extract_json('```json\n{"ID": "7"}\n```') == {"ID": "7"}
    assert _extract_json('```json{"ID": "195"} ```') == {"ID": "195"}


def test_transport_failure_stops_the_run_instead_of_returning_a_stub(
    tmp_path: Path,
) -> None:
    failures = FailureSink()
    responder = _make_agent_response(
        transport=make_transport(DeadClient(), tmp_path, failures=failures),
        seed=0,
        states={},
        context=CallContext(condition="Initial", query_index=0),
    )
    with pytest.raises(HarnessFailure, match="transport failed"):
        responder([], 0.2, 1, 2000, "fake/model", False)
    assert failures.kind == TRANSPORT


def test_unparseable_completion_reproduces_the_released_none(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    context = CallContext(condition="Initial", query_index=0)
    responder = _make_agent_response(
        transport=make_transport(
            FakeClient(["I cannot help with that."]), tmp_path, context=context
        ),
        seed=0,
        states={"Initial": state},
        context=context,
    )

    # No format contract on this caller, so the released None is preserved.
    assert responder([], 0.2, 1, 2000, "fake/model", False) is None
    assert state.unparsed_completions == 1
    assert state.format_resamples == 0


def test_string_stages_pass_the_completion_through(tmp_path: Path) -> None:
    responder = _make_agent_response(
        transport=make_transport(FakeClient(["a free-form answer"]), tmp_path),
        seed=0,
        states={},
        context=CallContext(),
    )
    assert responder([], 0.2, 1, 2000, "fake/model", True) == (
        "a free-form answer"
    )


def test_cost_cap_is_recorded_before_any_spend(tmp_path: Path) -> None:
    failures = FailureSink()
    transport = make_transport(
        FakeClient(["{}"]),
        tmp_path,
        budget=CostBudget(0.01, spent_usd=0.01),
        failures=failures,
    )
    with pytest.raises(CostCapReached):
        transport.complete(
            messages=[],
            temperature=0.2,
            top_p=1.0,
            max_tokens=10,
            stage="choose_tool",
            structured=False,
            seed=0,
        )
    assert failures.kind == COST_CAP
    assert transport.request_count == 0


def test_usage_log_captures_provider_and_latency(tmp_path: Path) -> None:
    transport = make_transport(FakeClient(['{"ID": "1"}']), tmp_path)
    transport.complete(
        messages=[],
        temperature=0.2,
        top_p=1.0,
        max_tokens=10,
        stage="choose_tool",
        structured=False,
        seed=3,
    )
    row = dar.read_jsonl(tmp_path / "usage_agent.jsonl")[0]

    assert row["served_model"] == "fake/model-2026"
    assert row["provider"] == "FakeProvider"
    assert row["request_id"] == "req-1"
    assert row["seed"] == 3
    assert row["latency_ms"] >= 0


# --------------------------------------------------------------------------
# Rollback of records written while a failure was pending
# --------------------------------------------------------------------------


def make_state(tmp_path: Path, name: str = "Initial") -> ConditionState:
    root = tmp_path / name.lower()
    root.mkdir(parents=True, exist_ok=True)
    return ConditionState(
        name=name,
        dataset={"1": {"tool_name": name}},
        root=root,
        output_path=root / f"ToolBench_G3_DFS_alias_{name}.jsonl",
        progress_path=root / "progress.txt",
        log_path=root / "console.log",
        failures_path=root / "driver_failures.json",
    )


def run_one(
    tmp_path: Path,
    state: ConditionState,
    failures: FailureSink,
    *,
    on_call=None,
    attempts: int = 1,
) -> None:
    _execute_one(
        draft=stub_draft_module(on_call=on_call),
        state=state,
        query={"query": "find a video", "relevant APIs": []},
        position=0,
        total=1,
        stable_root=tmp_path,
        retrieval_num=5,
        alias="alias",
        attempts=attempts,
        previous_cwd=Path.cwd(),
        failures=failures,
    )


def test_clean_query_writes_exactly_one_record(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    run_one(tmp_path, state, FailureSink())

    assert state.completed() == 1
    assert state.rolled_back == 0


def test_record_written_during_a_pending_failure_is_rolled_back(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    failures = FailureSink()

    def swallow(method: str, ind: int) -> None:
        # Mimics the released driver's bare `except:` absorbing our error.
        failures.record(TRANSPORT, "simulated transport loss")

    with pytest.raises(HarnessFailure, match="simulated transport loss"):
        run_one(tmp_path, state, failures, on_call=swallow)

    assert state.completed() == 0
    assert state.rolled_back == 1


def test_cost_cap_swallowed_by_the_driver_still_stops_the_run(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    failures = FailureSink()

    def swallow(method: str, ind: int) -> None:
        failures.record(COST_CAP, "cap reached")

    with pytest.raises(CostCapReached):
        run_one(tmp_path, state, failures, on_call=swallow)
    assert state.completed() == 0


def test_driver_crash_excludes_the_query_and_survives_a_resume(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)

    def crash(method: str, ind: int) -> None:
        raise TypeError("list indices must be integers or slices, not str")

    run_one(tmp_path, state, FailureSink(), on_call=crash)

    assert state.completed() == 0
    assert state.driver_failures == {0: "TypeError: list indices must be integers or slices, not str"}
    assert state.done(0), "an excluded query must not be re-rolled on resume"

    reloaded = make_state(tmp_path)
    reloaded.load_driver_failures()
    assert reloaded.done(0)


def test_a_high_exclusion_rate_stops_the_run(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.driver_failures = {0: "boom", 1: "boom", 2: "boom", 3: "boom"}
    with pytest.raises(HarnessFailure, match="above the tolerated"):
        dar._assert_driver_failures_within_tolerance(
            states={"Initial": state},
            planned=100,
            attempted=dar.MIN_DRIVER_FAILURE_SAMPLE,
            tolerance=0.15,
        )
    # Two isolated crashes early in a run are not evidence of a harness fault.
    state.driver_failures = {0: "boom"}
    dar._assert_driver_failures_within_tolerance(
        states={"Initial": state}, planned=100, attempted=2, tolerance=0.15
    )


def test_resume_does_not_treat_old_exclusions_as_a_rate_of_infinity(
    tmp_path: Path,
) -> None:
    """A restart at position 0 already has exclusions on disk."""
    state = make_state(tmp_path)
    state.driver_failures = {7: "TypeError: NoneType", 12: "TypeError: NoneType"}
    dar._assert_driver_failures_within_tolerance(
        states={"Initial": state},
        planned=100,
        attempted=1,
        tolerance=0.15,
    )


def test_unparseable_decompose_is_resampled_then_accepted(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    context = CallContext(condition="Initial", query_index=0)
    client = FakeClient(
        ["I cannot help with that.", '{"Tasks": ["step one"]}']
    )
    responder = _make_agent_response(
        transport=make_transport(client, tmp_path, context=context),
        seed=0,
        states={"Initial": state},
        context=context,
    )

    def task_decompose() -> object:
        return responder([], 0.2, 1, 2000, "fake/model", False)

    assert task_decompose() == {"Tasks": ["step one"]}
    assert state.unparsed_completions == 1
    assert state.format_resamples == 1
    assert [call["seed"] for call in client.calls] == [0, None]


def test_persistent_unparseable_decompose_raises_a_driver_failure(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    context = CallContext(condition="Initial", query_index=0)
    responder = _make_agent_response(
        transport=make_transport(
            FakeClient(["nope", "still not json", "sorry"]),
            tmp_path,
            context=context,
        ),
        seed=0,
        states={"Initial": state},
        context=context,
    )

    def task_decompose() -> object:
        return responder([], 0.2, 1, 2000, "fake/model", False)

    with pytest.raises(ReleasedDriverFailure, match="Tasks"):
        task_decompose()
    assert state.unparsed_completions == 3
    assert state.format_resamples == 3


def test_malformed_shape_is_resampled_then_accepted(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    context = CallContext(condition="Initial", query_index=0)
    client = FakeClient(
        ['["just", "a", "list"]', '{"Tasks": ["step one"]}']
    )
    responder = _make_agent_response(
        transport=make_transport(client, tmp_path, context=context),
        seed=0,
        states={"Initial": state},
        context=context,
    )

    def task_decompose() -> object:
        return responder([], 0.2, 1, 2000, "fake/model", False)

    assert task_decompose() == {"Tasks": ["step one"]}
    assert state.format_resamples == 1
    # The retry drops the fixed seed so the resample is not the same draw.
    assert [call["seed"] for call in client.calls] == [0, None]


def test_persistent_shape_violation_raises_a_driver_failure(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    context = CallContext(condition="Initial", query_index=0)
    responder = _make_agent_response(
        transport=make_transport(
            FakeClient(['["a"]', '["b"]', '["c"]']), tmp_path, context=context
        ),
        seed=0,
        states={"Initial": state},
        context=context,
    )

    def task_decompose() -> object:
        return responder([], 0.2, 1, 2000, "fake/model", False)

    with pytest.raises(ReleasedDriverFailure, match="Tasks"):
        task_decompose()
    assert state.format_resamples == 3


def test_stages_without_a_format_contract_are_not_resampled(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    context = CallContext(condition="Initial", query_index=0)
    client = FakeClient(['[{"ID": "1"}]'])
    responder = _make_agent_response(
        transport=make_transport(client, tmp_path, context=context),
        seed=0,
        states={"Initial": state},
        context=context,
    )

    def choose_tool() -> object:
        return responder([], 0.2, 1, 2000, "fake/model", False)

    # The released driver guards this one itself, so its output is passed on
    # unchanged even though it is the wrong shape.
    assert choose_tool() == [{"ID": "1"}]
    assert state.format_resamples == 0
    assert len(client.calls) == 1


def test_records_are_matched_by_driver_stamped_id(tmp_path: Path) -> None:
    path = tmp_path / "gapped.jsonl"
    path.write_text(
        json.dumps({"ID": 1}, indent=4)
        + "\n"
        + json.dumps({"ID": 3}, indent=4)
        + "\n",
        encoding="utf-8",
    )
    assert sorted(record_positions(path)) == [0, 2]

    duplicated = tmp_path / "duplicated.jsonl"
    duplicated.write_text(
        json.dumps({"ID": 1}) + "\n" + json.dumps({"ID": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(HarnessFailure, match="two records"):
        record_positions(duplicated)


def test_truncate_records_keeps_only_the_requested_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"ID": 1}, indent=4)
        + "\n"
        + json.dumps({"ID": 2}, indent=4)
        + "\n",
        encoding="utf-8",
    )
    assert len(record_offsets(path)) == 2

    truncate_records(path, 1)
    assert [row["ID"] for row in _read_concatenated_json(path)] == [1]


def test_truncated_jsonl_keeps_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    path.write_text('{"ID": 1}\n{"ID": 2, "truncated":', encoding="utf-8")
    assert [row["ID"] for row in _read_concatenated_json(path)] == [1]


# --------------------------------------------------------------------------
# The simulated environment is identical for both conditions
# --------------------------------------------------------------------------


def make_adapter(
    tmp_path: Path,
    *,
    context: CallContext,
    transport: LoggedOpenRouter | None,
    replay_only: bool = False,
    failures: FailureSink | None = None,
    schema_gate: bool = False,
    live_api_key: str = "",
    live_session: object | None = None,
) -> StableExecutionAdapter:
    return StableExecutionAdapter(
        root=tmp_path / "toolenv",
        transport=transport,
        generated_cache_path=tmp_path / "stable_simulator_cache.json",
        call_log_path=tmp_path / "backend_calls.jsonl",
        context=context,
        failures=failures or FailureSink(),
        replay_only=replay_only,
        schema_gate=schema_gate,
        live_api_key=live_api_key,
        live_session=live_session,
    )


def backend_request() -> dict[str, object]:
    return {
        "category": "Data",
        "tool_name": "Vimeo",
        "api_name": "SearchVideos",
        "tool_input": {"query": "cats"},
    }


def test_simulated_response_does_not_depend_on_the_condition(
    tmp_path: Path,
) -> None:
    context = CallContext(condition="Initial", query_index=0)
    client = FakeClient(['{"error": "", "response": "video 42"}'])
    adapter = make_adapter(
        tmp_path,
        context=context,
        transport=make_transport(
            client, tmp_path, context=context, role="simulator"
        ),
    )

    first = adapter(backend_request())
    context.condition = "DRAFT"
    second = adapter(backend_request())

    assert first == second == {"error": "", "response": "video 42"}
    # One simulator request only: the second condition reuses the response.
    assert len(client.calls) == 1
    assert adapter.stats["simulated"] == 1
    assert adapter.stats["generated_cache"] == 1
    sources = [
        row["source"]
        for row in dar.read_jsonl(tmp_path / "backend_calls.jsonl")
    ]
    assert sources == ["simulated", "generated_cache"]


def test_exact_cache_hit_bypasses_the_simulator(tmp_path: Path) -> None:
    cache = (
        tmp_path
        / "toolenv"
        / "tool_response_cache"
        / "Data"
        / "vimeo_for_Data"
        / "searchvideos.json"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"{'query': 'cats'}": {"error": "", "response": "cached"}}),
        encoding="utf-8",
    )
    client = FakeClient(['{"error": "", "response": "simulated"}'])
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=make_transport(client, tmp_path, role="simulator"),
    )

    assert adapter(backend_request()) == {"error": "", "response": "cached"}
    assert adapter.stats["exact_cache"] == 1
    assert client.calls == []


def _write_vimeo_tool(tmp_path: Path) -> None:
    path = tmp_path / "toolenv" / "tools" / "Data" / "vimeo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tool_name": "vimeo",
                "tool_description": "Search Vimeo.",
                "api_list": [
                    {
                        "name": "SearchVideos",
                        "required_parameters": [
                            {"name": "query", "type": "STRING"}
                        ],
                        "optional_parameters": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_schema_gate_rejects_missing_required_parameters(tmp_path: Path) -> None:
    _write_vimeo_tool(tmp_path)
    client = FakeClient(['{"error": "", "response": "should not run"}'])
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=make_transport(client, tmp_path, role="simulator"),
        schema_gate=True,
    )
    result = adapter(
        {
            "category": "Data",
            "tool_name": "Vimeo",
            "api_name": "SearchVideos",
            "tool_input": {},
        }
    )
    assert "missing required parameters: query" in result["error"]
    assert result["response"] == ""
    assert adapter.stats["schema_rejected"] == 1
    assert client.calls == []


def test_schema_gate_allows_a_valid_call_to_reach_the_simulator(
    tmp_path: Path,
) -> None:
    _write_vimeo_tool(tmp_path)
    client = FakeClient(['{"error": "", "response": "video 42"}'])
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=make_transport(client, tmp_path, role="simulator"),
        schema_gate=True,
    )
    result = adapter(backend_request())
    assert result == {"error": "", "response": "video 42"}
    assert adapter.stats["schema_rejected"] == 0
    assert adapter.stats["simulated"] == 1
    assert len(client.calls) == 1


class _FakeHttpResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeHttpSession:
    def __init__(self, responses: list[_FakeHttpResponse]) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = list(responses)

    def request(self, method: str, url: str, **kwargs: object) -> _FakeHttpResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            return _FakeHttpResponse(500, "no scripted responses left")
        return self.responses.pop(0)


class _BoomSimulator:
    def complete(self, **kwargs: object) -> str:
        raise AssertionError("live RapidAPI mode must not call the simulator")


def _write_live_vimeo_tool(tmp_path: Path) -> None:
    path = tmp_path / "toolenv" / "tools" / "Data" / "vimeo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tool_name": "vimeo",
                "tool_description": "Search Vimeo.",
                "host": "vimeo.p.rapidapi.com",
                "api_list": [
                    {
                        "name": "SearchVideos",
                        "url": "https://vimeo.p.rapidapi.com/search",
                        "method": "GET",
                        "required_parameters": [
                            {"name": "query", "type": "STRING"}
                        ],
                        "optional_parameters": [
                            {"name": "id", "type": "STRING"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_live_backend_ignores_frozen_cache_and_the_simulator(
    tmp_path: Path,
) -> None:
    _write_live_vimeo_tool(tmp_path)
    cache = (
        tmp_path
        / "toolenv"
        / "tool_response_cache"
        / "Data"
        / "vimeo_for_Data"
        / "searchvideos.json"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"{'query': 'cats'}": {"error": "", "response": "cached"}}),
        encoding="utf-8",
    )
    session = _FakeHttpSession([_FakeHttpResponse(200, '{"videos": 1}')])
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=_BoomSimulator(),
        live_api_key="test-key",
        live_session=session,
    )

    result = adapter(backend_request())

    assert result == {"error": "", "response": '{"videos": 1}'}
    assert adapter.stats["exact_cache"] == 0
    assert adapter.stats["simulated"] == 0
    assert adapter.stats["live"] == 1
    assert len(session.calls) == 1
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["params"] == {"query": "cats"}
    assert session.calls[0]["headers"]["X-RapidAPI-Key"] == "test-key"
    assert session.calls[0]["headers"]["X-RapidAPI-Host"] == "vimeo.p.rapidapi.com"


def test_live_identical_calls_are_served_from_the_run_cache(
    tmp_path: Path,
) -> None:
    _write_live_vimeo_tool(tmp_path)
    session = _FakeHttpSession([_FakeHttpResponse(200, '{"videos": 1}')])
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=_BoomSimulator(),
        live_api_key="test-key",
        live_session=session,
    )

    first = adapter(backend_request())
    adapter.context.condition = "DRAFT"
    second = adapter(backend_request())

    assert first == second == {"error": "", "response": '{"videos": 1}'}
    assert adapter.stats["live"] == 1
    assert adapter.stats["live_cache"] == 1
    assert len(session.calls) == 1
    sources = [
        row["source"]
        for row in dar.read_jsonl(tmp_path / "backend_calls.jsonl")
    ]
    assert sources == ["live", "live_cache"]


def test_live_maps_reserved_parameter_names_back_to_the_toolenv_original(
    tmp_path: Path,
) -> None:
    _write_live_vimeo_tool(tmp_path)
    session = _FakeHttpSession([_FakeHttpResponse(200, '{"ok": true}')])
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=_BoomSimulator(),
        live_api_key="test-key",
        live_session=session,
    )

    adapter(
        {
            "category": "Data",
            "tool_name": "Vimeo",
            "api_name": "SearchVideos",
            "tool_input": {"query": "cats", "is_id": "42"},
        }
    )

    assert session.calls[0]["params"] == {"query": "cats", "id": "42"}


def test_live_http_error_is_an_api_error_not_a_simulator_call(
    tmp_path: Path,
) -> None:
    _write_live_vimeo_tool(tmp_path)
    session = _FakeHttpSession(
        [_FakeHttpResponse(404, "Endpoint '/' does not exist")]
    )
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=_BoomSimulator(),
        live_api_key="test-key",
        live_session=session,
    )

    result = adapter(backend_request())

    assert result["response"] == ""
    assert result["error"].startswith("HTTP 404:")
    assert adapter.stats["live"] == 1
    assert adapter.stats["live_http_error"] == 1
    assert adapter.stats["simulated"] == 0


def test_replay_mode_refuses_to_invent_a_missing_response(
    tmp_path: Path,
) -> None:
    failures = FailureSink()
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="DRAFT", query_index=1),
        transport=None,
        replay_only=True,
        failures=failures,
    )
    with pytest.raises(SimulatorCacheMiss, match="frozen"):
        adapter(backend_request())
    assert failures.kind == dar.CACHE_MISS


def test_simulator_transport_failure_is_infrastructure(tmp_path: Path) -> None:
    failures = FailureSink()
    adapter = make_adapter(
        tmp_path,
        context=CallContext(condition="Initial", query_index=0),
        transport=make_transport(
            DeadClient(), tmp_path, role="simulator", failures=failures
        ),
        failures=failures,
    )
    with pytest.raises(HarnessFailure):
        adapter(backend_request())
    assert failures.kind == TRANSPORT


# --------------------------------------------------------------------------
# Paired plan, resume and provenance
# --------------------------------------------------------------------------


def test_condition_order_is_seeded_and_balanced_over_queries() -> None:
    queries = [{"query": f"q{index}", "relevant APIs": []} for index in range(20)]
    plan = build_query_plan(
        queries=queries,
        query_indices=list(range(20)),
        conditions=("Initial", "DRAFT"),
        seed=0,
    )
    repeat = build_query_plan(
        queries=queries,
        query_indices=list(range(20)),
        conditions=("Initial", "DRAFT"),
        seed=0,
    )
    other = build_query_plan(
        queries=queries,
        query_indices=list(range(20)),
        conditions=("Initial", "DRAFT"),
        seed=1,
    )
    orders = [entry["condition_order"] for entry in plan]

    assert orders == [entry["condition_order"] for entry in repeat]
    assert orders != [entry["condition_order"] for entry in other]
    assert 1 <= sum(order[0] == "DRAFT" for order in orders) <= 19


def run_paired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    draft_root: Path,
    agent_model: str = "fake/agent-1",
    resume: bool = True,
    calls: list[str] | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(
        dar,
        "load_released_draft_module",
        lambda root: stub_draft_module(
            on_call=(lambda method, ind: calls.append(f"{method}:{ind}"))
            if calls is not None
            else None
        ),
    )
    monkeypatch.setattr(
        dar.OpenRouterClient,
        "from_env",
        classmethod(lambda cls, **kwargs: FakeClient([])),
    )
    return dar.run_paired_g3(
        draft_root=draft_root,
        stable_root=tmp_path / "toolenv",
        run_root=tmp_path / "run",
        conditions=["Initial", "DRAFT"],
        agent_model=agent_model,
        simulator_model="fake/simulator-1",
        agent_provider=None,
        query_indices=[0, 1],
        retrieval_num=5,
        seed=0,
        budget=CostBudget(1.0),
        resume=resume,
        replay_only=False,
        query_attempts=1,
    )


@pytest.fixture()
def fake_draft_root(tmp_path: Path) -> Path:
    return write_fake_draft_root(
        tmp_path / "draft",
        [
            {
                "query_id": 101,
                "query": "find a cat video",
                "relevant APIs": [["Vimeo", "SearchVideos"]],
            },
            {
                "query_id": 102,
                "query": "find a dog video",
                "relevant APIs": [["Vimeo", "SearchVideos"]],
            },
        ],
    )


def test_resume_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_draft_root: Path
) -> None:
    calls: list[str] = []
    first = run_paired(
        monkeypatch, tmp_path, draft_root=fake_draft_root, calls=calls
    )
    assert len(calls) == 4
    assert first["conditions"]["Initial"]["records"] == 2
    assert first["conditions"]["DRAFT"]["complete"] is True

    calls.clear()
    second = run_paired(
        monkeypatch, tmp_path, draft_root=fake_draft_root, calls=calls
    )

    assert calls == [], "a resumed run must not re-execute finished queries"
    assert second["conditions"]["Initial"]["records"] == 2
    assert second["conditions"]["DRAFT"]["records"] == 2
    assert second["stop_reason"] == "completed"


def test_resume_is_blocked_when_the_manifest_no_longer_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_draft_root: Path
) -> None:
    run_paired(monkeypatch, tmp_path, draft_root=fake_draft_root)
    with pytest.raises(ManifestMismatch, match="different harness"):
        run_paired(
            monkeypatch,
            tmp_path,
            draft_root=fake_draft_root,
            agent_model="fake/agent-2",
        )


def test_manifest_pins_upstream_documents_and_harness_code(
    tmp_path: Path, fake_draft_root: Path
) -> None:
    manifest = build_manifest(
        draft_root=fake_draft_root,
        stable_root=tmp_path,
        agent={"model": "fake/agent-1"},
        simulator={"model": "fake/simulator-1"},
        decoding=dar.RELEASED_DECODING,
        protocol={"dataset": "ToolBench G3"},
        queries={"count": 2},
    )
    digests = manifest["upstream_digests"]

    assert digests["Inference_DFSDT.py"] == file_digest(
        fake_draft_root / "Inference_DFSDT.py"
    )
    assert digests["dataset/ToolBench/test_data/G3.json"] != "missing"
    assert set(manifest["harness_digests"]) >= {
        "draft_agent_reproduction.py",
        "openrouter.py",
    }
    assert fingerprint(manifest) == manifest["fingerprint"]


def test_summary_reports_no_stub_and_no_rollback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_draft_root: Path
) -> None:
    summary = run_paired(monkeypatch, tmp_path, draft_root=fake_draft_root)
    for condition in summary["conditions"].values():
        assert condition["rolled_back_records"] == 0
        assert condition["unparsed_completions"] == 0
        assert condition["records"] == summary["planned_queries"]


# --------------------------------------------------------------------------
# Metric definitions
# --------------------------------------------------------------------------


def test_correct_path_matches_released_set_containment_metric() -> None:
    record = {
        "execute_log": {
            "api_result_ls": [
                [
                    {"tool_name": "Vimeo", "api_name": "SearchVideos"},
                    {"tool_name": "Extra", "api_name": "Unneeded"},
                ],
                [{"tool_name": "Vimeo", "api_name": "GetRelatedPeople"}],
            ]
        }
    }
    query = {
        "relevant APIs": [
            ["Vimeo", "SearchVideos"],
            ["Vimeo", "GetRelatedPeople"],
        ]
    }

    assert correct_path(record, query)


def test_paper_metric_is_ordered_but_released_script_is_not() -> None:
    record = {
        "execute_log": {
            "api_result_ls": [
                [{"tool_name": "Vimeo", "api_name": "GetRelatedPeople"}],
                [{"tool_name": "Vimeo", "api_name": "SearchVideos"}],
            ]
        }
    }
    query = {
        "relevant APIs": [
            ["Vimeo", "SearchVideos"],
            ["Vimeo", "GetRelatedPeople"],
        ]
    }

    assert not correct_path(record, query)
    assert correct_path_released(record, query)


def test_duplicated_gold_entry_requires_two_executions() -> None:
    query = {
        "relevant APIs": [
            ["Vimeo", "SearchVideos"],
            ["Vimeo", "SearchVideos"],
        ]
    }
    once = {
        "execute_log": {
            "api_result_ls": [
                [{"tool_name": "Vimeo", "api_name": "SearchVideos"}]
            ]
        }
    }
    twice = {
        "execute_log": {
            "api_result_ls": [
                [{"tool_name": "Vimeo", "api_name": "SearchVideos"}],
                [{"tool_name": "Vimeo", "api_name": "SearchVideos"}],
            ]
        }
    }

    assert not correct_path(once, query)
    assert correct_path(twice, query)
    # Set containment cannot see the duplicate, which is why both are reported.
    assert correct_path_released(once, query)


def test_exact_sign_test_endpoints() -> None:
    assert exact_sign_test(0, 0) == 1.0
    assert exact_sign_test(8, 8) == 1.0
    assert exact_sign_test(0, 8) == pytest.approx(2 / 256)


def test_paired_bootstrap_brackets_a_known_difference() -> None:
    interval = paired_bootstrap([1] * 10, resamples=500, seed=1)
    assert interval["mean"] == 1.0
    assert interval["low"] == 1.0 and interval["high"] == 1.0


def test_parameter_validity_flags_missing_and_unknown_arguments() -> None:
    schemas = {
        ("Data", "vimeo", "searchvideos"): {
            "name": "SearchVideos",
            "required_parameters": [{"name": "query"}],
            "optional_parameters": [{"name": "page"}],
        }
    }
    valid = classify_call(
        {
            "categoty": "Data",
            "tool_name": "Vimeo",
            "api_name": "SearchVideos",
            "parameters": {"query": "cats", "page": 2},
        },
        schemas,
    )
    invalid = classify_call(
        {
            "categoty": "Data",
            "tool_name": "Vimeo",
            "api_name": "SearchVideos",
            "parameters": {"q": "cats"},
        },
        schemas,
    )
    unknown_api = classify_call(
        {
            "categoty": "Data",
            "tool_name": "Vimeo",
            "api_name": "Nonexistent",
            "parameters": {},
        },
        schemas,
    )

    assert valid["fully_valid"]
    assert invalid["missing_required"] == ["query"]
    assert invalid["unknown_parameters"] == ["q"]
    assert unknown_api["schema_found"] is False

    report = parameter_validity(
        [
            {
                "execute_log": {
                    "api_result_ls": [
                        [
                            {
                                "categoty": "Data",
                                "tool_name": "Vimeo",
                                "api_name": "SearchVideos",
                                "parameters": {"query": "cats"},
                            },
                            {
                                "categoty": "Data",
                                "tool_name": "Vimeo",
                                "api_name": "SearchVideos",
                                "parameters": {"q": "cats"},
                            },
                        ]
                    ]
                }
            }
        ],
        schemas,
    )

    assert report["calls"] == 2
    assert report["fully_valid_rate"] == 0.5
    assert report["queries_with_all_calls_valid"] == 0.0


def test_backend_success_is_split_by_response_source() -> None:
    report = backend_success(
        [
            {"condition": "Initial", "source": "exact_cache", "error": "", "response_chars": 10},
            {"condition": "Initial", "source": "simulated", "error": "boom", "response_chars": 0},
            {"condition": "DRAFT", "source": "generated_cache", "error": "", "response_chars": 20},
            {"condition": "Ours", "source": "live", "error": "", "response_chars": 40},
            {"condition": "Ours", "source": "live_cache", "error": "HTTP 404", "response_chars": 0},
        ]
    )

    assert report["Initial"]["by_source"]["exact_cache"]["error_free_rate"] == 1.0
    assert report["Initial"]["by_source"]["simulated"]["error_free_rate"] == 0.0
    assert report["Initial"]["error_free_rate"] == 0.5
    assert report["DRAFT"]["calls"] == 1
    assert report["Ours"]["by_source"]["live"]["error_free_rate"] == 1.0
    assert report["Ours"]["by_source"]["live_cache"]["error_free_rate"] == 0.0


# --------------------------------------------------------------------------
# The withdrawn run is still reproducible as a diagnostic
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (LEGACY_ROOT / "initial").exists(),
    reason="the legacy GPT-4o-mini run is not present in this checkout",
)
def test_legacy_run_recomputes_to_the_audited_numbers() -> None:
    raw_jsonl = (
        LEGACY_ROOT
        / "initial"
        / "ToolBench_G3_DFS_gpt-4o-mini-2024-07-18_Initial.jsonl"
    )
    draft_jsonl = (
        LEGACY_ROOT
        / "draft"
        / "ToolBench_G3_DFS_gpt-4o-mini-2024-07-18_DRAFT.jsonl"
    )
    result = decompose_g3(
        draft_root=DRAFT_ROOT, raw_jsonl=raw_jsonl, draft_jsonl=draft_jsonl
    )
    queries = json.loads(
        (
            DRAFT_ROOT / "dataset" / "ToolBench" / "test_data" / "G3.json"
        ).read_text(encoding="utf-8")
    )
    raw_records = _read_concatenated_json(raw_jsonl)
    draft_records = _read_concatenated_json(draft_jsonl)

    assert result["queries"] == 100
    assert len(raw_records) == len(draft_records) == 100
    # Every record must sit at the position of the query it answered.
    for index in range(100):
        assert raw_records[index]["question"] == queries[index]["query"]
        assert draft_records[index]["question"] == queries[index]["query"]
    assert result["paper_cp"] == {"raw": 0.31, "draft": 0.31}
    assert result["released_cp"] == {"raw": 0.33, "draft": 0.33}
    assert result["paired"]["raw_only"] == 8
    assert result["paired"]["draft_only"] == 8
    assert result["paired_statistics"]["exact_sign_test_p"] == 1.0
    # No record in that run was a harness stub.
    for record in raw_records + draft_records:
        assert "runner_error" not in record
