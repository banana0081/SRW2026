"""Held-out temporal pilot: shared observations, independent updaters, fresh eval.

This is not Experiment 3a. The withdrawn comparison derived the evaluation call
from the updater's own document and scored a refusal phrase the updater wrote.
Here:

- observations are collected once, with a shared policy, and frozen;
- Raw leaves the document alone, DRAFT runs the released Analyzer/Rewriter on
  those traces, Ours applies evidence-gated contract and liveness patches;
- the evaluation agent sees only documentation and a held-out query, never a
  mutation label or oracle status;
- the call is executed against the runtime server and scored against the
  runtime schema, not against the document the updater just wrote;
- each condition starts from a reset server so availability patterns do not
  leak across methods;
- down cases include a live fallback whose query does not name the outage;
- Ours may be iterated on the dev split; the test split is opened once.

Oracle calls (runtime schema, not an updater) must succeed on every case that
is callable on a fresh server. If they do not, the harness is broken and the
run must stop.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .benchmark import build_scenarios
from .canonicalize import canonicalize, canonical_text
from .controlled import (
    ControlledApiServer,
    ControlledScenario,
    _example_for_type,
    _error_payload,
)
from .draft_agent_reproduction import (
    DEFAULT_STABLE_ROOT,
    CostBudget,
    CostCapReached,
    _append_jsonl,
    _dump_json,
    _extract_json,
)
from .draft_rewriter import (
    DEFAULT_EMBEDDING_MODEL,
    DraftPrompts,
    DraftRewriteFailure,
    local_embedder,
    rewrite_from_observations,
)
from .openrouter import OpenRouterClient, OpenRouterError
from .dynamic_models import (
    ExecutionObservation,
    LivenessState,
    ObservationSource,
)
from .models import ApiDocument, normalize_identifier
from .ours_update import OursUpdate, update_from_observations
from .probe import HttpProber, default_arguments
from .toolenv import inventory, load_toolenv

OBSERVATION_BUDGET = 5
DOWN_FAMILIES = frozenset({"unavailable", "not_found"})
CALLABLE_NOW_STATUSES = {200}
DEFAULT_DEV_FRACTION = 0.3
DEFAULT_AGENT_MODEL = "deepseek/deepseek-v4-flash-0731"
EVAL_ATTEMPTS = 3
DEFAULT_WORKERS = 8
CONTRACT_FAMILIES = frozenset(
    {
        "add_required",
        "remove_parameter",
        "requiredness",
        "parameter_type",
        "rename_parameter",
        "method",
    }
)
RESPONSE_FAMILIES = frozenset(
    {"response_add", "response_remove", "response_type"}
)
LIVENESS_FAMILIES = frozenset(
    {
        "unavailable",
        "intermittent",
        "not_found",
        "auth_blocked",
        "rate_limited",
        "recovery",
    }
)
CONTROL_FAMILIES = frozenset({"unchanged"})
FAMILY_GROUPS = {
    "contract": CONTRACT_FAMILIES,
    "response": RESPONSE_FAMILIES,
    "liveness": LIVENESS_FAMILIES,
    "control": CONTROL_FAMILIES,
}


class TemporalHarnessFailure(RuntimeError):
    """The controlled scenarios themselves cannot be solved by an oracle."""


@dataclass
class TemporalCase:
    case_id: str
    split: str
    family: str
    primary: ControlledScenario
    fallback: ControlledScenario | None
    query: str
    observations: list[ExecutionObservation] = field(default_factory=list)


@dataclass
class EndpointView:
    endpoint_id: str
    scenario_id: str
    document: ApiDocument
    text: str


@dataclass
class AgentCall:
    endpoint: str
    method: str
    arguments: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionOutcome:
    case_id: str
    condition: str
    call: AgentCall | None
    status_code: int | None
    path_ok: bool
    schema_valid: bool
    http_success: bool
    fallback_selected: bool
    false_down: bool
    unsupported_patch: int
    liveness_state: str = ""
    error: str = ""


RespondFn = Callable[[list[dict[str, Any]], str], Any]
EmbedFn = Callable[[str], Sequence[float]]


def mutation_family(scenario_id: str) -> str:
    return scenario_id.rsplit("_", 1)[0]


def held_out_query(document: ApiDocument, *, has_fallback: bool) -> str:
    purpose = document.api_description.strip() or (
        f"complete a {document.api_name} lookup"
    )
    if has_fallback:
        return (
            f"I need to {purpose.rstrip('.')} . Use whichever available tool "
            "can satisfy the request. Do not explain."
        )
    return (
        f"I need to {purpose.rstrip('.')} . Use the available tool. Do not explain."
    )


def gold_endpoint(case: TemporalCase) -> str:
    if (
        case.fallback is not None
        and case.primary.expected_liveness is LivenessState.UNAVAILABLE
    ):
        return "fallback"
    return "primary"


def is_callable_now(scenario: ControlledScenario) -> bool:
    return bool(scenario.availability_pattern) and scenario.availability_pattern[
        0
    ] in CALLABLE_NOW_STATUSES


def _similar_document(
    target: ApiDocument,
    pool: Sequence[ApiDocument],
    used: set[tuple[str, str]],
) -> ApiDocument | None:
    same_category = [
        document
        for document in pool
        if document.category_name == target.category_name
        and document.key not in used
        and document.key != target.key
        and (document.method or "GET").upper() == "GET"
    ]
    if same_category:
        return same_category[0]
    others = [
        document
        for document in pool
        if document.key not in used and document.key != target.key
    ]
    return others[0] if others else None


def build_temporal_cases(
    documents: list[ApiDocument],
    *,
    seeds_per_mutation: int,
    random_seed: int,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
) -> list[TemporalCase]:
    scenarios = build_scenarios(
        documents,
        seeds_per_mutation=seeds_per_mutation,
        random_seed=random_seed,
    )
    rng = random.Random(random_seed + 17)
    by_api: dict[tuple[str, str], list[ControlledScenario]] = {}
    for scenario in scenarios:
        by_api.setdefault(scenario.baseline.key, []).append(scenario)
    api_keys = sorted(by_api)
    rng.shuffle(api_keys)
    dev_count = max(1, int(len(api_keys) * dev_fraction))
    dev_keys = set(api_keys[:dev_count])

    cases: list[TemporalCase] = []
    for scenario in scenarios:
        family = mutation_family(scenario.scenario_id)
        fallback = None
        if family in DOWN_FAMILIES:
            sibling = _similar_document(
                scenario.baseline, documents, {scenario.baseline.key}
            )
            if sibling is not None:
                fallback = ControlledScenario.unchanged(
                    f"{scenario.scenario_id}_fallback", sibling
                )
        split = "dev" if scenario.baseline.key in dev_keys else "test"
        cases.append(
            TemporalCase(
                case_id=scenario.scenario_id,
                split=split,
                family=family,
                primary=scenario,
                fallback=fallback,
                query=held_out_query(
                    scenario.baseline, has_fallback=fallback is not None
                ),
            )
        )
    return cases


def collect_observations(
    server: ControlledApiServer,
    case: TemporalCase,
    *,
    budget: int = OBSERVATION_BUDGET,
) -> list[ExecutionObservation]:
    """Shared exploration policy. Never uses the held-out evaluation query."""
    prober = HttpProber(timeout_seconds=2.0, allow_unsafe_methods=True)
    document = server.document(case.primary.scenario_id)
    arguments = default_arguments(document)
    current = document
    observations: list[ExecutionObservation] = []
    for _ in range(budget):
        observation = prober.probe(
            current,
            arguments=arguments,
            source=ObservationSource.CONTROLLED,
        )
        observations.append(observation)
        error = _error_payload(observation)
        if observation.status_code == 405:
            allowed = observation.response_headers.get("allow")
            if allowed:
                current = current.model_copy(
                    update={"method": allowed.split(",")[0].strip()}
                )
            continue
        if error.get("code") == "missing_parameter":
            name = str(error.get("parameter") or "")
            expected = str(error.get("expected_type") or "string")
            if name:
                arguments[name] = _example_for_type(expected)
            continue
        if error.get("code") == "unknown_parameter":
            name = str(error.get("parameter") or "")
            arguments.pop(name, None)
            arguments.pop(normalize_identifier(name), None)
    return observations


def oracle_call(scenario: ControlledScenario) -> AgentCall:
    arguments = {
        name: _example_for_type(field_type)
        for name, field_type in scenario.runtime_required.items()
    }
    return AgentCall(
        endpoint="primary",
        method=scenario.runtime_method,
        arguments=arguments,
        raw={"oracle": True},
    )


def execute_call(
    server: ControlledApiServer,
    case: TemporalCase,
    call: AgentCall,
) -> tuple[int | None, dict[str, Any], bool]:
    scenario = (
        case.fallback
        if call.endpoint == "fallback" and case.fallback is not None
        else case.primary
    )
    if call.endpoint == "fallback" and case.fallback is None:
        return None, {"error": "no_fallback"}, False
    accepted = {**scenario.runtime_required, **scenario.runtime_optional}
    provided = {
        normalize_identifier(str(key)): value
        for key, value in call.arguments.items()
    }
    missing = [name for name in scenario.runtime_required if name not in provided]
    unknown = [name for name in provided if name not in accepted]
    schema_valid = (
        not missing and not unknown and call.method == scenario.runtime_method
    )
    document = server.document(scenario.scenario_id).model_copy(
        update={"method": call.method}
    )
    prober = HttpProber(timeout_seconds=2.0, allow_unsafe_methods=True)
    observation = prober.probe(
        document,
        arguments=provided,
        source=ObservationSource.CONTROLLED,
    )
    return observation.status_code, {
        "body": observation.response_body,
        "missing": missing,
        "unknown": unknown,
    }, schema_valid


def _assert_oracle(
    server: ControlledApiServer, cases: Sequence[TemporalCase]
) -> None:
    failures: list[str] = []
    for case in cases:
        if not is_callable_now(case.primary):
            continue
        server.reset_counts()
        call = oracle_call(case.primary)
        status, _, schema_valid = execute_call(server, case, call)
        if status != 200 or not schema_valid:
            failures.append(
                f"{case.case_id}: oracle status={status} schema_valid={schema_valid}"
            )
    if failures:
        raise TemporalHarnessFailure(
            "Oracle below 100% on callable cases: " + "; ".join(failures[:8])
        )


def render_view(
    endpoint_id: str, scenario_id: str, document: ApiDocument
) -> EndpointView:
    return EndpointView(
        endpoint_id=endpoint_id,
        scenario_id=scenario_id,
        document=document,
        text=canonical_text(canonicalize(document)),
    )


def agent_prompt(case: TemporalCase, views: Sequence[EndpointView]) -> str:
    blocks = []
    for view in views:
        blocks.append(
            f"### {view.endpoint_id}\n"
            f"tool={view.document.tool_name}\n"
            f"api={view.document.api_name}\n"
            f"{view.text}"
        )
    catalog = "\n\n".join(blocks)
    allowed = ", ".join(view.endpoint_id for view in views)
    return (
        "Select one endpoint and the arguments needed to satisfy the user "
        "query. Return JSON with keys endpoint, method, arguments. endpoint "
        f"must be one of: {allowed}.\n\n"
        f"User query:\n{case.query}\n\nAvailable tools:\n{catalog}\n"
    )


def parse_agent_call(payload: Any, views: Sequence[EndpointView]) -> AgentCall:
    if not isinstance(payload, dict):
        raise ValueError("Agent did not return a JSON object.")
    allowed = {view.endpoint_id for view in views}
    endpoint = str(payload.get("endpoint") or "")
    if endpoint not in allowed:
        raise ValueError(f"endpoint must be one of {sorted(allowed)}")
    method = str(payload.get("method") or "GET").upper()
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object.")
    return AgentCall(
        endpoint=endpoint,
        method=method,
        arguments={str(key): value for key, value in arguments.items()},
        raw=dict(payload),
    )


def update_raw(document: ApiDocument) -> ApiDocument:
    return document


def update_draft(
    document: ApiDocument,
    observations: Sequence[ExecutionObservation],
    *,
    prompts: DraftPrompts,
    respond: RespondFn,
    embed: EmbedFn,
) -> ApiDocument:
    api_info = {
        "name": document.api_name,
        "description": document.api_description,
        "required_parameters": [
            item.model_dump(mode="json") for item in document.required_parameters
        ],
        "optional_parameters": [
            item.model_dump(mode="json") for item in document.optional_parameters
        ],
    }
    examples = [
        {
            "query": (
                f"Probe {document.api_name} with "
                f"{json.dumps(item.request_arguments, ensure_ascii=False)}"
            ),
            "parameters": item.request_arguments,
            "api_response": {
                "status_code": item.status_code,
                "error": item.error_message or item.error_type,
                "response": item.response_body,
            },
        }
        for item in observations
    ]
    result = rewrite_from_observations(
        prompts=prompts,
        category=document.category_name,
        tool_name=document.tool_name,
        api_info=api_info,
        observations=examples,
        respond=respond,
        embed=embed,
    )
    return document.model_copy(update={"api_description": result.description})


def update_ours(
    document: ApiDocument, observations: Sequence[ExecutionObservation]
) -> tuple[ApiDocument, OursUpdate]:
    update = update_from_observations(document, observations)
    return update.document, update


def score_outcome(
    *,
    case: TemporalCase,
    condition: str,
    call: AgentCall | None,
    status_code: int | None,
    schema_valid: bool,
    unsupported_patch: int,
    liveness_state: str = "",
    error: str = "",
) -> ConditionOutcome:
    gold = gold_endpoint(case)
    selected = call.endpoint if call is not None else ""
    fallback_selected = selected == "fallback"
    path_ok = selected == gold
    false_down = fallback_selected and gold == "primary"
    if liveness_state == LivenessState.UNAVAILABLE.value and gold == "primary":
        false_down = True
    return ConditionOutcome(
        case_id=case.case_id,
        condition=condition,
        call=call,
        status_code=status_code,
        path_ok=path_ok,
        schema_valid=schema_valid,
        http_success=status_code == 200,
        fallback_selected=fallback_selected,
        false_down=false_down,
        unsupported_patch=unsupported_patch,
        liveness_state=liveness_state,
        error=error,
    )


def evaluate_condition(
    *,
    server: ControlledApiServer,
    case: TemporalCase,
    condition: str,
    views: Sequence[EndpointView],
    respond: RespondFn | None,
    scripted: Mapping[str, Any] | None = None,
    unsupported_patch: int = 0,
    liveness_state: str = "",
) -> ConditionOutcome:
    server.reset_counts()
    prompt = agent_prompt(case, views)
    try:
        if scripted is not None:
            payload = scripted
            call = parse_agent_call(payload, views)
        elif respond is not None:
            last_error: Exception | None = None
            call = None
            for _ in range(EVAL_ATTEMPTS):
                try:
                    payload = respond(
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temporal_eval",
                    )
                    call = parse_agent_call(payload, views)
                    last_error = None
                    break
                except ValueError as exc:
                    last_error = exc
            if last_error is not None or call is None:
                raise last_error or ValueError("Agent did not return a JSON object.")
        else:
            raise TemporalHarnessFailure("No evaluation agent is configured.")
        status, _detail, schema_valid = execute_call(server, case, call)
        error = ""
    except (ValueError, TemporalHarnessFailure, OpenRouterError, CostCapReached) as exc:
        call = None
        status = None
        schema_valid = False
        error = str(exc)
    return score_outcome(
        case=case,
        condition=condition,
        call=call,
        status_code=status,
        schema_valid=schema_valid,
        unsupported_patch=unsupported_patch,
        liveness_state=liveness_state,
        error=error,
    )


def gold_needed_fallback(case: TemporalCase) -> bool:
    return gold_endpoint(case) == "fallback"


def _rate(rows: Sequence[ConditionOutcome], attr: str) -> float | None:
    if not rows:
        return None
    return round(sum(getattr(row, attr) for row in rows) / len(rows), 4)


def _condition_block(
    rows: Sequence[ConditionOutcome], cases: Sequence[TemporalCase]
) -> dict[str, Any]:
    by_id = {case.case_id: case for case in cases}
    fallback_gold = [
        row for row in rows if gold_needed_fallback(by_id[row.case_id])
    ]
    by_family: dict[str, list[ConditionOutcome]] = {}
    for row in rows:
        by_family.setdefault(by_id[row.case_id].family, []).append(row)
    by_group: dict[str, list[ConditionOutcome]] = {}
    for family, family_rows in by_family.items():
        group = next(
            (name for name, families in FAMILY_GROUPS.items() if family in families),
            "other",
        )
        by_group.setdefault(group, []).extend(family_rows)
    return {
        "cases": len(rows),
        "path": _rate(rows, "path_ok"),
        "schema_valid": _rate(rows, "schema_valid"),
        "http_success": _rate(rows, "http_success"),
        "fallback_selection": round(
            sum(row.fallback_selected and row.path_ok for row in fallback_gold)
            / max(1, len(fallback_gold)),
            4,
        ),
        "false_down": _rate(rows, "false_down"),
        "unsupported_patch_total": sum(row.unsupported_patch for row in rows),
        "errors": sum(1 for row in rows if row.error),
        "by_family": {
            family: {
                "n": len(family_rows),
                "path": _rate(family_rows, "path_ok"),
                "schema_valid": _rate(family_rows, "schema_valid"),
                "http_success": _rate(family_rows, "http_success"),
            }
            for family, family_rows in sorted(by_family.items())
        },
        "by_group": {
            group: {
                "n": len(group_rows),
                "path": _rate(group_rows, "path_ok"),
                "schema_valid": _rate(group_rows, "schema_valid"),
                "http_success": _rate(group_rows, "http_success"),
            }
            for group, group_rows in sorted(by_group.items())
        },
    }


def _summarize(
    outcomes: Sequence[ConditionOutcome], cases: Sequence[TemporalCase]
) -> dict[str, Any]:
    grouped: dict[str, list[ConditionOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.condition, []).append(outcome)
    return {
        condition: _condition_block(rows, cases)
        for condition, rows in grouped.items()
    }


def _sign_test(wins: int, losses: int) -> dict[str, Any]:
    n = wins + losses
    payload: dict[str, Any] = {
        "n_discordant": n,
        "wins": wins,
        "losses": losses,
        "p_two_sided": 1.0,
    }
    if n == 0:
        return payload
    try:
        from scipy.stats import binomtest

        payload["p_two_sided"] = float(
            binomtest(wins, n, 0.5, alternative="two-sided").pvalue
        )
    except Exception:
        payload["p_two_sided"] = None
    return payload


def _paired(
    outcomes: Sequence[ConditionOutcome],
    cases: Sequence[TemporalCase],
    families: frozenset[str] | None = None,
) -> dict[str, Any]:
    selected = [
        case for case in cases if families is None or case.family in families
    ]
    by_key = {(row.case_id, row.condition): row for row in outcomes}
    report: dict[str, Any] = {}
    for left, right in (("raw", "draft"), ("raw", "ours"), ("draft", "ours")):
        rows = []
        for case in selected:
            first = by_key.get((case.case_id, left))
            second = by_key.get((case.case_id, right))
            if first is None or second is None:
                continue
            rows.append(
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    f"{left}_path": first.path_ok,
                    f"{right}_path": second.path_ok,
                    f"{left}_schema": first.schema_valid,
                    f"{right}_schema": second.schema_valid,
                    f"{left}_http": first.http_success,
                    f"{right}_http": second.http_success,
                    "delta_path": int(second.path_ok) - int(first.path_ok),
                    "delta_schema": int(second.schema_valid)
                    - int(first.schema_valid),
                    "delta_http": int(second.http_success)
                    - int(first.http_success),
                }
            )
        if not rows:
            continue
        schema_wins = sum(
            1
            for row in rows
            if row[f"{right}_schema"] and not row[f"{left}_schema"]
        )
        schema_losses = sum(
            1
            for row in rows
            if row[f"{left}_schema"] and not row[f"{right}_schema"]
        )
        http_wins = sum(
            1
            for row in rows
            if row[f"{right}_http"] and not row[f"{left}_http"]
        )
        http_losses = sum(
            1
            for row in rows
            if row[f"{left}_http"] and not row[f"{right}_http"]
        )
        path_wins = sum(
            1
            for row in rows
            if row[f"{right}_path"] and not row[f"{left}_path"]
        )
        path_losses = sum(
            1
            for row in rows
            if row[f"{left}_path"] and not row[f"{right}_path"]
        )
        report[f"{right}_minus_{left}"] = {
            "n": len(rows),
            "path_mean_delta": round(
                sum(row["delta_path"] for row in rows) / len(rows), 4
            ),
            "schema_mean_delta": round(
                sum(row["delta_schema"] for row in rows) / len(rows), 4
            ),
            "http_mean_delta": round(
                sum(row["delta_http"] for row in rows) / len(rows), 4
            ),
            f"{right}_only_path": path_wins,
            f"{left}_only_path": path_losses,
            f"{right}_only_schema": schema_wins,
            f"{left}_only_schema": schema_losses,
            f"{right}_only_http": http_wins,
            f"{left}_only_http": http_losses,
            "sign_test_schema": _sign_test(schema_wins, schema_losses),
            "sign_test_http": _sign_test(http_wins, http_losses),
            "sign_test_path": _sign_test(path_wins, path_losses),
        }
    return report


def _paired_groups(
    outcomes: Sequence[ConditionOutcome], cases: Sequence[TemporalCase]
) -> dict[str, Any]:
    return {
        "all": _paired(outcomes, cases),
        **{
            name: _paired(outcomes, cases, families)
            for name, families in FAMILY_GROUPS.items()
        },
    }


def locked_embedder(embed: EmbedFn) -> EmbedFn:
    lock = threading.Lock()

    def wrapped(text: str) -> Sequence[float]:
        with lock:
            return embed(text)

    return wrapped


def make_json_respond(
    *,
    client: OpenRouterClient,
    model: str,
    budget: CostBudget,
    usage_path: Path | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    seed: int | None = 42,
) -> RespondFn:
    use_json_object = True

    def respond(messages: list[dict[str, Any]], stage: str) -> Any:
        nonlocal use_json_object
        budget.reserve_check()
        started = time.perf_counter()
        payload_format = (
            {"type": "json_object"} if use_json_object else None
        )
        try:
            result = client.chat_completion(
                model=model,
                messages=messages,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                seed=seed,
                reasoning_effort=None,
                response_format=payload_format,
            )
        except OpenRouterError as exc:
            message = str(exc).lower()
            if use_json_object and (
                "response_format" in message or "json_object" in message
            ):
                use_json_object = False
                result = client.chat_completion(
                    model=model,
                    messages=messages,
                    provider=provider,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1.0,
                    seed=seed,
                    reasoning_effort=None,
                    response_format=None,
                )
            else:
                raise
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        choice = result["choices"][0] or {}
        content = str((choice.get("message") or {}).get("content") or "")
        usage = result.get("usage") or {}
        cost_usd = float(usage.get("cost") or 0.0)
        budget.record(cost_usd)
        if usage_path is not None:
            transport = result.get("_transport") or {}
            _append_jsonl(
                usage_path,
                {
                    "stage": stage,
                    "requested_model": model,
                    "served_model": result.get("model", ""),
                    "provider": result.get("provider", ""),
                    "request_id": transport.get("request_id", ""),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                    "finish_reason": choice.get("finish_reason", ""),
                },
            )
        try:
            parsed = _extract_json(content)
        except ValueError:
            return content
        return parsed

    return respond


def _trace_row(
    *,
    case: TemporalCase,
    condition: str,
    document: ApiDocument,
    outcome: ConditionOutcome,
    draft_rewrite_error: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "family": case.family,
        "condition": condition,
        "query": case.query,
        "gold": gold_endpoint(case),
        "callable_now": is_callable_now(case.primary),
        "observation_statuses": [item.status_code for item in case.observations],
        "document_method": document.method,
        "document_required": [item.name for item in document.required_parameters],
        "path_ok": outcome.path_ok,
        "schema_valid": outcome.schema_valid,
        "http_success": outcome.http_success,
        "fallback_selected": outcome.fallback_selected,
        "false_down": outcome.false_down,
        "unsupported_patch": outcome.unsupported_patch,
        "status_code": outcome.status_code,
        "liveness_state": outcome.liveness_state,
        "error": outcome.error,
        "draft_rewrite_error": draft_rewrite_error,
        "call": None
        if outcome.call is None
        else {
            "endpoint": outcome.call.endpoint,
            "method": outcome.call.method,
            "arguments": outcome.call.arguments,
        },
    }


def _process_case(
    case: TemporalCase,
    *,
    respond: RespondFn | None,
    embed: EmbedFn | None,
    prompts: DraftPrompts | None,
    scripted_calls: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    run_draft: bool,
    run_ours: bool,
    budget: CostBudget | None,
) -> tuple[list[ConditionOutcome], list[dict[str, Any]]]:
    outcomes: list[ConditionOutcome] = []
    traces: list[dict[str, Any]] = []
    scenarios = [case.primary]
    if case.fallback is not None:
        scenarios.append(case.fallback)
    with ControlledApiServer(scenarios) as server:
        _assert_oracle(server, [case])
        server.reset_counts()
        case.observations = collect_observations(server, case)
        primary_doc = case.primary.baseline
        fallback_doc = case.fallback.baseline if case.fallback else None
        condition_docs: list[tuple[str, ApiDocument, int, str, str]] = [
            ("raw", update_raw(primary_doc), 0, "", "")
        ]
        if run_draft:
            if prompts is None or embed is None or respond is None:
                raise TemporalHarnessFailure(
                    "DRAFT rewrite needs prompts, embed and respond."
                )
            if budget is not None and budget.exhausted:
                condition_docs.append(
                    ("draft", primary_doc, 0, "", "cost cap before DRAFT rewrite")
                )
            else:
                try:
                    draft_doc = update_draft(
                        primary_doc,
                        case.observations,
                        prompts=prompts,
                        respond=respond,
                        embed=embed,
                    )
                    condition_docs.append(("draft", draft_doc, 0, "", ""))
                except (DraftRewriteFailure, OpenRouterError, CostCapReached) as exc:
                    condition_docs.append(("draft", primary_doc, 0, "", str(exc)))
        if run_ours:
            ours_doc, ours_update = update_ours(primary_doc, case.observations)
            condition_docs.append(
                (
                    "ours",
                    ours_doc,
                    len(ours_update.unsupported_operations),
                    ours_update.liveness.state.value,
                    "",
                )
            )

        for condition, document, unsupported, liveness_state, rewrite_error in condition_docs:
            views = [render_view("primary", case.primary.scenario_id, document)]
            if fallback_doc is not None and case.fallback is not None:
                views.append(
                    render_view(
                        "fallback",
                        case.fallback.scenario_id,
                        fallback_doc,
                    )
                )
            scripted = (scripted_calls or {}).get((case.case_id, condition))
            outcome = evaluate_condition(
                server=server,
                case=case,
                condition=condition,
                views=views,
                respond=respond,
                scripted=scripted,
                unsupported_patch=unsupported,
                liveness_state=liveness_state,
            )
            outcomes.append(outcome)
            traces.append(
                _trace_row(
                    case=case,
                    condition=condition,
                    document=document,
                    outcome=outcome,
                    draft_rewrite_error=rewrite_error,
                )
            )
    return outcomes, traces


def run_pilot(
    *,
    cases: Sequence[TemporalCase],
    split: str,
    respond: RespondFn | None = None,
    embed: EmbedFn | None = None,
    prompts: DraftPrompts | None = None,
    scripted_calls: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    run_draft: bool = True,
    run_ours: bool = True,
    workers: int = 1,
    budget: CostBudget | None = None,
    model: str = "",
) -> dict[str, Any]:
    selected = [case for case in cases if case.split == split]
    if not selected:
        raise ValueError(f"No temporal cases in split {split!r}.")

    outcomes: list[ConditionOutcome] = []
    traces: list[dict[str, Any]] = []
    worker_count = max(1, workers)
    kwargs = {
        "respond": respond,
        "embed": embed,
        "prompts": prompts,
        "scripted_calls": scripted_calls,
        "run_draft": run_draft,
        "run_ours": run_ours,
        "budget": budget,
    }
    if worker_count == 1:
        for case in selected:
            case_outcomes, case_traces = _process_case(case, **kwargs)
            outcomes.extend(case_outcomes)
            traces.extend(case_traces)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(_process_case, case, **kwargs): case.case_id
                for case in selected
            }
            for future in as_completed(futures):
                case_outcomes, case_traces = future.result()
                outcomes.extend(case_outcomes)
                traces.extend(case_traces)

    traces.sort(key=lambda row: (row["case_id"], row["condition"]))
    outcomes.sort(key=lambda row: (row.case_id, row.condition))
    return {
        "split": split,
        "cases": len(selected),
        "families": dict(Counter(case.family for case in selected)),
        "conditions": _summarize(outcomes, selected),
        "paired": _paired_groups(outcomes, selected),
        "traces": traces,
        "model": model,
        "workers": worker_count,
        "cost_usd": None if budget is None else round(budget.spent_usd, 6),
        "policy": (
            "Shared five observations, independent documentation updates, "
            "held-out query, runtime scoring, reset server per condition. "
            "Cases may run in parallel against isolated servers. "
            "Ours may be iterated on split=dev only."
        ),
    }


def synthetic_documents(count: int = 8) -> list[ApiDocument]:
    documents = []
    for index in range(count):
        documents.append(
            ApiDocument(
                category_name="Search",
                tool_name=f"Catalog {index}",
                api_name="find",
                api_description=(
                    f"Find records matching a query in catalog {index}."
                ),
                method="GET",
                required_parameters=[
                    {
                        "name": "q",
                        "type": "STRING",
                        "description": "Search query.",
                    }
                ],
                optional_parameters=[
                    {
                        "name": "limit",
                        "type": "NUMBER",
                        "description": "Maximum number of records.",
                        "default": 10,
                    },
                    {
                        "name": "locale",
                        "type": "STRING",
                        "description": "Optional locale tag.",
                    },
                ],
                template_response={"result": "str", "count": "int"},
            )
        )
    return documents


def scripted_oracle_calls(
    cases: Sequence[TemporalCase], conditions: Sequence[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    scripted: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        gold = gold_endpoint(case)
        target = case.primary if gold == "primary" else case.fallback
        if target is None:
            continue
        call = oracle_call(target)
        payload = {
            "endpoint": gold,
            "method": call.method,
            "arguments": call.arguments,
        }
        for condition in conditions:
            scripted[(case.case_id, condition)] = payload
    return scripted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Held-out temporal documentation pilot."
    )
    parser.add_argument("--stable-root", type=Path, default=DEFAULT_STABLE_ROOT)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--seeds-per-mutation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--documents", type=int, default=24)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument(
        "--source",
        choices=["synthetic", "toolenv"],
        default="synthetic",
    )
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--max-cost-usd", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=Path("external/DRAFT/prompts"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--usage-log",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run Raw and Ours only, with scripted evaluation calls. No LLM.",
    )
    args = parser.parse_args()
    output = args.output or Path(
        f"artifacts/results/temporal_pilot_v2/{args.split}_"
        f"{'offline' if args.offline else 'deepseek'}.json"
    )
    usage_log = args.usage_log or (
        None
        if args.offline
        else output.with_name(output.stem + "_usage.jsonl")
    )

    if args.source == "toolenv" and not args.offline:
        documents = load_toolenv(args.stable_root / "tools")
        print(json.dumps(inventory(documents), indent=2)[:1000])
    else:
        documents = synthetic_documents(args.documents if not args.offline else 8)

    cases = build_temporal_cases(
        documents,
        seeds_per_mutation=args.seeds_per_mutation if not args.offline else 1,
        random_seed=args.seed,
        dev_fraction=args.dev_fraction,
    )
    respond: RespondFn | None = None
    embed: EmbedFn | None = None
    prompts: DraftPrompts | None = None
    budget: CostBudget | None = None
    scripted = None
    workers = 1 if args.offline else max(1, args.workers)
    selected_n = sum(1 for case in cases if case.split == args.split)
    print(
        json.dumps(
            {
                "status": "starting",
                "split": args.split,
                "cases": selected_n,
                "model": "" if args.offline else args.model,
                "workers": workers,
                "source": "synthetic" if args.offline else args.source,
                "output": str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.offline:
        scripted = scripted_oracle_calls(cases, ("raw", "ours"))
    else:
        client = OpenRouterClient.from_env()
        budget = CostBudget(args.max_cost_usd)
        respond = make_json_respond(
            client=client,
            model=args.model,
            budget=budget,
            usage_path=usage_log,
            provider=args.provider or None,
        )
        embed = locked_embedder(local_embedder())
        prompts = DraftPrompts.load(args.prompt_root)

    report = run_pilot(
        cases=cases,
        split=args.split,
        respond=respond,
        embed=embed,
        prompts=prompts,
        scripted_calls=scripted,
        run_draft=not args.offline,
        run_ours=True,
        workers=workers,
        budget=budget,
        model="" if args.offline else args.model,
    )
    report["source"] = "synthetic" if args.offline else args.source
    _dump_json(output, report)
    printable = {key: value for key, value in report.items() if key != "traces"}
    print(json.dumps(printable, ensure_ascii=False, indent=2)[:6000])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
