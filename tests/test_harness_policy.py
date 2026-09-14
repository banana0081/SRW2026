"""Provider pinning, prompt provenance and the split retry/error policy.

Three failure modes these lock down, each of which already produced a number
that could not be defended:

- the same model id served by two providers, compared as one measurement;
- a documentation payload digest that says nothing about the prompt the model
  actually received;
- a retry loop that re-rolls a malformed completion, which quietly favours the
  arm whose text breaks the driver most often.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooldoc_nir.draft_agent_reproduction import (
    CallContext,
    CostBudget,
    FailureSink,
    LoggedOpenRouter,
    PROVIDER_DRIFT,
    ProviderDrift,
    usage_totals,
)
from tooldoc_nir.openrouter import normalize_provider
from tooldoc_nir.provenance import payload_digest
from tooldoc_nir.restbench_table import (
    DRIVER_ERROR,
    error_kind,
    is_credits_exhausted,
    is_endpoint_dead,
    is_transport_error,
    refuse_openrouter_for_local_model,
)


class ScriptedClient:
    """Answers with a fixed provider name, whatever was requested."""

    def __init__(self, provider: str, content: str = '{"Tasks": ["a"]}') -> None:
        self.provider = provider
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "model": "vendor/model",
            "provider": self.provider,
            "choices": [
                {"message": {"content": self.content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "cost": 0.0},
            "_transport": {"status_code": 200, "request_id": "req"},
        }


def _transport(client: object, tmp_path: Path, provider: str | None) -> tuple:
    failures = FailureSink()
    logged = LoggedOpenRouter(
        client=client,
        model="vendor/model",
        role="agent",
        usage_path=tmp_path / "usage.jsonl",
        budget=CostBudget(1.0),
        context=CallContext(condition="P", query_index=7),
        failures=failures,
        provider=provider,
    )
    return logged, failures


MESSAGES = [{"role": "user", "content": "catalog"}]


def _complete(logged: LoggedOpenRouter) -> str:
    return logged.complete(
        messages=MESSAGES,
        temperature=0.2,
        top_p=1.0,
        max_tokens=2000,
        stage="task_decompose",
        structured=False,
        seed=21,
    )


def test_provider_slug_and_display_name_compare_equal() -> None:
    assert normalize_provider("novita") == normalize_provider("Novita")
    assert normalize_provider("deepinfra") == normalize_provider("DeepInfra")
    assert normalize_provider("sailresearch") == normalize_provider("Sail Research")
    assert normalize_provider("novita") != normalize_provider("deepinfra")


def test_a_pinned_provider_that_did_not_answer_fails_the_call(tmp_path: Path) -> None:
    logged, failures = _transport(
        ScriptedClient("Sail Research"), tmp_path, "novita"
    )
    with pytest.raises(ProviderDrift):
        _complete(logged)
    assert failures.kind == PROVIDER_DRIFT
    assert "novita" in failures.message
    assert "Sail Research" in failures.message


def test_the_pin_is_sent_and_the_served_provider_is_accepted(tmp_path: Path) -> None:
    client = ScriptedClient("Novita")
    logged, failures = _transport(client, tmp_path, "novita")
    assert _complete(logged) == '{"Tasks": ["a"]}'
    assert failures.kind is None
    assert client.calls[0]["provider"] == "novita"


def test_an_unpinned_run_accepts_whoever_answered(tmp_path: Path) -> None:
    logged, failures = _transport(ScriptedClient("Wafer"), tmp_path, None)
    assert _complete(logged)
    assert failures.kind is None


def test_the_usage_log_records_the_prompt_that_was_actually_sent(
    tmp_path: Path,
) -> None:
    logged, _failures = _transport(ScriptedClient("Novita"), tmp_path, "novita")
    _complete(logged)
    record = json.loads(
        (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["prompt_digest"] == payload_digest(MESSAGES)
    assert record["prompt_chars"] == len("catalog")
    assert record["pinned_provider"] == "novita"
    assert record["provider"] == "Novita"
    totals = usage_totals(tmp_path / "usage.jsonl")
    assert totals["pinned_providers"] == ["novita"]
    assert totals["prompt_digests"] == 1
    assert totals["length_truncated"] == 0


def test_the_prompt_digest_separates_two_documentation_variants(
    tmp_path: Path,
) -> None:
    """Two arms that differ only in prose must differ in the prompt digest."""
    logged, _failures = _transport(ScriptedClient("Novita"), tmp_path, "novita")
    _complete(logged)
    logged.complete(
        messages=[{"role": "user", "content": "catalog with a role clause"}],
        temperature=0.2,
        top_p=1.0,
        max_tokens=2000,
        stage="task_decompose",
        structured=False,
        seed=21,
    )
    assert usage_totals(tmp_path / "usage.jsonl")["prompt_digests"] == 2


def test_transport_faults_retry_and_model_results_do_not() -> None:
    transport = {"error": "HarnessFailure: HTTP 503", "error_kind": "transport"}
    drift = {"error": "ProviderDrift: served by Wafer", "error_kind": PROVIDER_DRIFT}
    driver = {
        "error": "ReleasedDriverFailure: task_topology did not return a list",
        "error_kind": DRIVER_ERROR,
    }
    clean = {"error": "", "correct_path": True}

    assert is_transport_error(transport)
    assert is_transport_error(drift)
    assert not is_transport_error(driver)
    assert not is_transport_error(clean)
    assert error_kind(clean) == ""


def test_an_error_kind_missing_from_an_older_trace_defaults_to_a_miss() -> None:
    """Legacy roots have no `error_kind`; they must not become retryable."""
    legacy = {"error": "TypeError: 'NoneType' object is not subscriptable"}
    assert error_kind(legacy) == DRIVER_ERROR
    assert not is_transport_error(legacy)


def test_a_connect_timeout_is_a_dead_endpoint_and_a_402_is_not() -> None:
    dead = (
        "HarnessFailure: agent transport failed during task_decompose: "
        "local chat request failed: ConnectTimeout: HTTPConnectionPool"
    )
    assert is_endpoint_dead(dead)
    assert not is_endpoint_dead("HarnessFailure: HTTP 503")
    assert is_credits_exhausted("OpenRouter HTTP 402")
    assert not is_credits_exhausted(dead)


def test_qwen_refuses_openrouter_when_the_base_url_is_missing() -> None:
    class Cloud:
        local = False

    class Local:
        local = True

    with pytest.raises(SystemExit, match="refusing OpenRouter"):
        refuse_openrouter_for_local_model("qwen/qwen3.8-27b", Cloud())
    refuse_openrouter_for_local_model("qwen/qwen3.8-27b", Local())
    refuse_openrouter_for_local_model("inclusionai/ling-3.0-flash", Cloud())
