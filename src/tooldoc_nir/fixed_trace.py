"""Integration test for the controlled server, liveness and patch mechanics.

Not an evaluator, and not a comparison against DRAFT. `reports/experiment_audit.md`
withdraws the ranking this module once produced: `draft_recompute` is a
hand-written string builder rather than DRAFT, its scored refusal is triggered
by a phrase it writes itself, and the evaluation call is derived from the
document the updater just wrote, so the condition that edits the schema also
chooses the call. Removing that one refusal rule collapses its score onto Raw's.

What the module still shows is that the local pipeline recovers 54 of the 60
request-changing mutations exactly on synthetic input. Use it for that
regression check only; documentation comparisons belong in the DeepSeek harness
with a real DRAFT recomputation and held-out queries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .benchmark import _event, _set_metrics, build_scenarios
from .canonicalize import canonicalize
from .contract import detect_contract_drift, snapshot_from_profile
from .controlled import (
    ControlledApiServer,
    ControlledScenario,
    _error_payload,
    _example_for_type,
)
from .dynamic_models import (
    ChangeKind,
    ChangeOperation,
    ContractDelta,
    ExecutionObservation,
    LivenessAssessment,
    LivenessState,
    ObservationSource,
)
from .liveness import assess_liveness
from .models import ApiDocument, ApiParameter, normalize_identifier
from .patching import build_contract_patch, validate_patch
from .probe import HttpProber, default_arguments
from .toolenv import load_toolenv


FIXED_TRACE_BUDGET = 5
DOWN_PREFIXES = {"unavailable", "not_found"}


def mutation_family(scenario_id: str) -> str:
    return scenario_id.rsplit("_", 1)[0]


def collect_fixed_trace(
    document: ApiDocument,
    *,
    budget: int = FIXED_TRACE_BUDGET,
) -> list[ExecutionObservation]:
    """Take exactly `budget` probes with a shared exploration policy."""
    prober = HttpProber(timeout_seconds=2.0, allow_unsafe_methods=True)
    current = document
    arguments = default_arguments(document)
    observations: list[ExecutionObservation] = []
    working: dict[str, Any] | None = None
    working_document = current

    for _ in range(budget):
        if working is not None:
            observation = prober.probe(
                working_document,
                arguments=working,
                source=ObservationSource.CONTROLLED,
            )
            observations.append(observation)
            continue
        observation = prober.probe(
            current,
            arguments=arguments,
            source=ObservationSource.CONTROLLED,
        )
        observations.append(observation)
        if observation.status_code == 200:
            working = dict(arguments)
            working_document = current
            continue
        if observation.status_code == 405:
            allowed = observation.response_headers.get("allow")
            if allowed:
                current = current.model_copy(update={"method": allowed})
            continue
        error = _error_payload(observation)
        code = error.get("code")
        name = normalize_identifier(str(error.get("parameter", "")))
        if code == "missing_parameter" and name:
            arguments[name] = _example_for_type(
                str(error.get("expected_type", "string"))
            )
        elif code == "unknown_parameter" and name:
            arguments.pop(name, None)
            for original in list(arguments):
                if normalize_identifier(original) == name:
                    arguments.pop(original, None)
        elif code == "invalid_type" and name:
            coerced = _example_for_type(str(error.get("expected_type", "string")))
            arguments[name] = coerced
            for original in list(arguments):
                if normalize_identifier(original) == name:
                    arguments[original] = coerced
    return observations


def draft_recompute(
    document: ApiDocument,
    observations: list[ExecutionObservation],
) -> ApiDocument:
    """Free-form enrichment that never edits the executable schema."""
    notes: list[str] = []
    example = ""
    for observation in observations:
        status = observation.status_code
        if status is None or status in {404, 408, 410, 500, 502, 503, 504}:
            notes.append(
                "The API appears unavailable or frequently returns server errors."
            )
        elif status in {401, 403, 429}:
            notes.append(
                "The API appears unavailable because requests are blocked."
            )
        elif status is not None and status >= 400:
            notes.append(
                "A recent invocation failed with HTTP "
                f"{status}: {observation.response_body!r}."
            )
        elif status is not None and 200 <= status < 300:
            example = (
                "Example call: "
                f"arguments={observation.request_arguments} "
                f"response={str(observation.response_body)[:400]}"
            )
    extra = " ".join(dict.fromkeys(notes))
    if example:
        extra = f"{extra} {example}".strip()
    description = document.api_description
    if extra:
        description = f"{description} {extra}".strip()
    return document.model_copy(update={"api_description": description})


def _refuses_call(document: ApiDocument) -> bool:
    text = document.api_description.casefold()
    return (
        "[liveness: unavailable]" in text
        or "appears unavailable" in text
    )


def apply_ours_update(
    document: ApiDocument,
    delta: ContractDelta,
    liveness: LivenessAssessment,
) -> ApiDocument:
    updated = document.model_copy(deep=True)
    required = {
        normalize_identifier(item.name): item
        for item in updated.required_parameters
        if item.name
    }
    optional = {
        normalize_identifier(item.name): item
        for item in updated.optional_parameters
        if item.name
    }

    def upsert(bag: dict[str, ApiParameter], name: str, **fields: Any) -> None:
        current = bag.get(name)
        if current is None:
            bag[name] = ApiParameter(name=name, **fields)
        else:
            bag[name] = current.model_copy(update=fields)

    for change in delta.confirmed_changes:
        if change.kind is ChangeKind.METHOD:
            updated.method = str(change.new_value)
        elif change.kind is not ChangeKind.REQUEST_FIELD:
            continue
        if change.operation is ChangeOperation.ADD:
            payload = (
                change.new_value
                if isinstance(change.new_value, dict)
                else {"type": "string", "required": True}
            )
            target = required if payload.get("required") else optional
            other = optional if payload.get("required") else required
            other.pop(change.path, None)
            upsert(
                target,
                change.path,
                type=str(payload.get("type") or "string"),
                description="Observed at runtime; evidence-gated update.",
            )
        elif change.operation is ChangeOperation.REMOVE:
            required.pop(change.path, None)
            optional.pop(change.path, None)
        elif change.operation is ChangeOperation.REPLACE:
            if change.path.endswith(".required"):
                name = change.path[: -len(".required")]
                if change.new_value:
                    parameter = optional.pop(name, None) or required.get(name)
                    if parameter is not None:
                        required[name] = parameter
                else:
                    parameter = required.pop(name, None) or optional.get(name)
                    if parameter is not None:
                        optional[name] = parameter
            elif change.path.endswith(".type"):
                name = change.path[: -len(".type")]
                for bag in (required, optional):
                    if name in bag:
                        bag[name] = bag[name].model_copy(
                            update={
                                "type": str(change.new_value),
                                "default": _example_for_type(str(change.new_value)),
                            }
                        )

    updated.required_parameters = list(required.values())
    updated.optional_parameters = list(optional.values())
    marker = f"[LIVENESS: {liveness.state.value}]"
    updated.api_description = f"{marker} {updated.api_description}".strip()
    return updated


def oracle_document(
    document: ApiDocument,
    scenario: ControlledScenario,
) -> ApiDocument:
    updated = document.model_copy(deep=True)
    updated.method = scenario.runtime_method
    required: list[ApiParameter] = []
    optional: list[ApiParameter] = []
    known = {
        normalize_identifier(item.name): item
        for item in [*document.required_parameters, *document.optional_parameters]
        if item.name
    }
    for name, field_type in scenario.runtime_required.items():
        previous = known.get(name)
        required.append(
            ApiParameter(
                name=name,
                type=field_type,
                description=(
                    previous.description if previous is not None else "Oracle runtime field."
                ),
                default=_example_for_type(field_type),
            )
        )
    for name, field_type in scenario.runtime_optional.items():
        previous = known.get(name)
        optional.append(
            ApiParameter(
                name=name,
                type=field_type,
                description=(
                    previous.description if previous is not None else "Oracle runtime field."
                ),
                default=_example_for_type(field_type),
            )
        )
    updated.required_parameters = required
    updated.optional_parameters = optional
    updated.api_description = (
        f"[LIVENESS: {scenario.expected_liveness.value}] {updated.api_description}"
    )
    return updated


def evaluate_call(
    document: ApiDocument,
    *,
    expected_liveness: LivenessState,
) -> dict[str, Any]:
    refused = _refuses_call(document)
    expected_down = expected_liveness is LivenessState.UNAVAILABLE
    if refused:
        return {
            "refused": True,
            "status_code": None,
            "executable": False,
            "correct_down": expected_down,
            "false_down": not expected_down,
            "missed_down": False,
        }
    prober = HttpProber(timeout_seconds=2.0, allow_unsafe_methods=True)
    observation = prober.probe(
        document,
        arguments=default_arguments(document),
        source=ObservationSource.CONTROLLED,
    )
    executable = observation.status_code == 200
    return {
        "refused": False,
        "status_code": observation.status_code,
        "executable": executable,
        "correct_down": False,
        "false_down": False,
        "missed_down": expected_down,
    }


def _condition_success(
    family: str,
    expected_liveness: LivenessState,
    result: dict[str, Any],
) -> bool:
    if family in DOWN_PREFIXES or expected_liveness is LivenessState.UNAVAILABLE:
        return bool(result["correct_down"])
    if expected_liveness in {
        LivenessState.AUTH_BLOCKED,
        LivenessState.RATE_LIMITED,
    }:
        return _auth_or_rate_success(result)
    return bool(result["executable"])


def _auth_or_rate_success(result: dict[str, Any]) -> bool:
    return (not result["false_down"]) and result["status_code"] in {401, 403, 429}


def run_fixed_trace(
    scenarios: list[ControlledScenario],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with ControlledApiServer(scenarios) as server:
        for scenario in scenarios:
            live = server.document(scenario.scenario_id)
            stale = live.model_copy(deep=True)
            observations = collect_fixed_trace(stale)
            liveness = assess_liveness(
                observations, source=ObservationSource.CONTROLLED
            )
            profile = canonicalize(stale)
            _, delta = detect_contract_drift(profile, observations)
            patch = build_contract_patch(delta, auto_applicable_only=True)
            validated, _ = validate_patch(patch, snapshot_from_profile(profile))
            ours_doc = apply_ours_update(stale, delta, liveness)
            draft_doc = draft_recompute(stale, observations)
            oracle_doc = oracle_document(stale, scenario)

            family = mutation_family(scenario.scenario_id)
            evaluations = {
                "raw": evaluate_call(
                    stale, expected_liveness=scenario.expected_liveness
                ),
                "draft": evaluate_call(
                    draft_doc, expected_liveness=scenario.expected_liveness
                ),
                "ours": evaluate_call(
                    ours_doc, expected_liveness=scenario.expected_liveness
                ),
                "oracle": evaluate_call(
                    oracle_doc, expected_liveness=scenario.expected_liveness
                ),
            }
            expected_events = set(scenario.expected_events)
            ours_events = {
                _event(change)
                for change in delta.confirmed_changes
                if change.kind is not ChangeKind.RESPONSE_FIELD
            }
            expected_request = {
                event
                for event in expected_events
                if not event.startswith("response_field:")
            }
            case = {
                "scenario_id": scenario.scenario_id,
                "family": family,
                "tool": stale.tool_name,
                "api": stale.api_name,
                "expected_events": sorted(expected_events),
                "expected_request_events": sorted(expected_request),
                "ours_request_events": sorted(ours_events),
                "expected_liveness": scenario.expected_liveness.value,
                "ours_liveness": liveness.state.value,
                "draft_refuses": _refuses_call(draft_doc),
                "ours_refuses": _refuses_call(ours_doc),
                "patch_status": validated.status.value,
                "observation_statuses": [
                    item.status_code for item in observations
                ],
                "evaluations": evaluations,
                "success": {
                    name: _condition_success(
                        family, scenario.expected_liveness, result
                    )
                    for name, result in evaluations.items()
                },
            }
            cases.append(case)

    def mean(flag: str, condition: str) -> float:
        values = [
            int(case["evaluations"][condition][flag])
            if flag in case["evaluations"][condition]
            else int(case["success"][condition])
            for case in cases
        ]
        return round(sum(values) / max(1, len(values)), 4)

    success_by_condition = {
        condition: round(
            sum(case["success"][condition] for case in cases) / max(1, len(cases)),
            4,
        )
        for condition in ("raw", "draft", "ours", "oracle")
    }
    by_family: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["family"]].append(case)
    for family, group in grouped.items():
        by_family[family] = {
            condition: round(
                sum(case["success"][condition] for case in group) / len(group),
                4,
            )
            for condition in ("raw", "draft", "ours", "oracle")
        }
    ours_vs_draft = [
        int(case["success"]["ours"]) - int(case["success"]["draft"])
        for case in cases
    ]
    return {
        "scenario_count": len(cases),
        "trace_budget": FIXED_TRACE_BUDGET,
        "success_rate": success_by_condition,
        "ours_minus_draft": {
            "wins": sum(delta == 1 for delta in ours_vs_draft),
            "losses": sum(delta == -1 for delta in ours_vs_draft),
            "ties": sum(delta == 0 for delta in ours_vs_draft),
            "mean_delta": round(
                sum(ours_vs_draft) / max(1, len(ours_vs_draft)), 4
            ),
        },
        "false_down_rate": {
            "raw": mean("false_down", "raw"),
            "draft": mean("false_down", "draft"),
            "ours": mean("false_down", "ours"),
            "oracle": mean("false_down", "oracle"),
        },
        "request_contract": _set_metrics(
            [set(case["expected_request_events"]) for case in cases],
            [set(case["ours_request_events"]) for case in cases],
        ),
        "by_family": by_family,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 3a: fixed-trace Raw/DRAFT/Ours/Oracle comparison."
    )
    parser.add_argument(
        "--toolenv",
        type=Path,
        default=Path(
            "data/raw/stabletoolbench/toolenv2404/toolenv2404_filtered"
        ),
    )
    parser.add_argument("--seeds-per-mutation", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/experiment3a_fixed_trace.json"),
    )
    args = parser.parse_args()
    documents = load_toolenv(args.toolenv)
    scenarios = build_scenarios(
        documents,
        seeds_per_mutation=args.seeds_per_mutation,
        random_seed=args.seed,
    )
    result = run_fixed_trace(scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in result.items() if key != "cases"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
