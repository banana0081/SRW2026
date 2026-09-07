from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable

from .canonicalize import canonicalize
from .contract import detect_contract_drift, normalize_type
from .controlled import (
    ControlledApiServer,
    ControlledScenario,
    explore_controlled_api,
    materialize_response,
    mutate_add_required,
    mutate_availability,
    mutate_method,
    mutate_parameter_type,
    mutate_remove_parameter,
    mutate_rename_parameter,
    mutate_requiredness,
    mutate_response,
)
from .dynamic_models import LivenessState, ObservationSource
from .liveness import assess_liveness
from .models import ApiDocument, normalize_identifier
from .toolenv import load_toolenv


def _take(
    pool: list[ApiDocument],
    count: int,
    rng: random.Random,
) -> list[ApiDocument]:
    if not pool:
        raise ValueError("No eligible ToolBench documents for a mutation.")
    if len(pool) >= count:
        return rng.sample(pool, count)
    return [pool[index % len(pool)] for index in range(count)]


def _response_paths(document: ApiDocument, prefix: str) -> list[str]:
    profile = canonicalize(document)
    return [
        field
        for field in profile.output_fields
        if field == prefix or field.startswith(f"{prefix}.")
    ]


def build_scenarios(
    documents: list[ApiDocument],
    *,
    seeds_per_mutation: int,
    random_seed: int,
) -> list[ControlledScenario]:
    rng = random.Random(random_seed)
    base_pool = [
        document
        for document in documents
        if document.method.upper() == "GET"
        and isinstance(document.template_response, dict)
        and document.template_response
        and len(document.required_parameters) + len(document.optional_parameters)
        <= 10
    ]
    required_pool = [
        document for document in base_pool if document.required_parameters
    ]
    optional_pool = [
        document
        for document in base_pool
        if any(
            parameter.default in (None, "")
            for parameter in document.optional_parameters
        )
    ]
    string_required_pool = [
        document
        for document in required_pool
        if any(
            normalize_type(parameter.type) == "string"
            and not str(parameter.default or "test").lstrip("-").isdigit()
            for parameter in document.required_parameters
        )
    ]
    object_response_pool = [
        document
        for document in base_pool
        if isinstance(
            materialize_response(document.template_response), dict
        )
        and bool(materialize_response(document.template_response))
    ]
    scalar_response_pool = [
        document
        for document in object_response_pool
        if any(
            "." not in path
            and field_type
            in {"boolean", "integer", "number", "string"}
            for path, field_type in canonicalize(document).output_types.items()
        )
    ]

    scenarios: list[ControlledScenario] = []

    def add_group(
        name: str,
        pool: list[ApiDocument],
        mutate: Callable[[ControlledScenario, ApiDocument], ControlledScenario],
    ) -> None:
        for index, document in enumerate(
            _take(pool, seeds_per_mutation, rng)
        ):
            scenario = ControlledScenario.unchanged(
                f"{name}_{index:03d}", document
            )
            scenarios.append(mutate(scenario, document))

    add_group("unchanged", base_pool, lambda scenario, _: scenario)
    add_group(
        "add_required",
        base_pool,
        lambda scenario, _: mutate_add_required(
            scenario, "tooldoc_locale", "string"
        ),
    )
    add_group(
        "remove_parameter",
        required_pool,
        lambda scenario, document: mutate_remove_parameter(
            scenario, document.required_parameters[0].name
        ),
    )

    def make_required(
        scenario: ControlledScenario, document: ApiDocument
    ) -> ControlledScenario:
        parameter = next(
            item
            for item in document.optional_parameters
            if item.default in (None, "")
        )
        return mutate_requiredness(
            scenario, parameter.name, required=True
        )

    add_group("requiredness", optional_pool, make_required)

    def change_parameter_type(
        scenario: ControlledScenario, document: ApiDocument
    ) -> ControlledScenario:
        parameter = next(
            item
            for item in document.required_parameters
            if normalize_type(item.type) == "string"
            and not str(item.default or "test").lstrip("-").isdigit()
        )
        return mutate_parameter_type(scenario, parameter.name, "integer")

    add_group("parameter_type", string_required_pool, change_parameter_type)

    def rename_parameter(
        scenario: ControlledScenario, document: ApiDocument
    ) -> ControlledScenario:
        return mutate_rename_parameter(
            scenario,
            document.required_parameters[0].name,
            "tooldoc_renamed_field",
        )

    add_group("rename_parameter", required_pool, rename_parameter)
    add_group(
        "method",
        base_pool,
        lambda scenario, _: mutate_method(scenario, "POST"),
    )

    def add_response_field(
        scenario: ControlledScenario, document: ApiDocument
    ) -> ControlledScenario:
        response = deepcopy(scenario.response_body)
        if not isinstance(response, dict):
            response = {"result": response}
        response["tooldoc_drift_marker"] = "value"
        return mutate_response(
            scenario,
            response,
            "response_field:add:tooldoc_drift_marker",
        )

    add_group("response_add", base_pool, add_response_field)

    def remove_response_field(
        scenario: ControlledScenario, document: ApiDocument
    ) -> ControlledScenario:
        response = deepcopy(scenario.response_body)
        if not isinstance(response, dict) or not response:
            raise ValueError("Response-removal seed has no object fields.")
        key = str(next(iter(response)))
        response.pop(key)
        normalized = normalize_identifier(key)
        events = [
            f"response_field:remove:{path}"
            for path in _response_paths(document, normalized)
        ]
        return mutate_response(scenario, response, *events)

    add_group(
        "response_remove",
        object_response_pool,
        remove_response_field,
    )

    def change_response_type(
        scenario: ControlledScenario, document: ApiDocument
    ) -> ControlledScenario:
        response = deepcopy(scenario.response_body)
        if not isinstance(response, dict):
            raise ValueError("Response-type seed is not an object.")
        key = next(
            name
            for name, value in response.items()
            if not isinstance(value, (dict, list))
        )
        old_type = canonicalize(document).output_types.get(
            normalize_identifier(str(key)), "unknown"
        )
        response[key] = (
            1 if old_type not in {"integer", "number"} else "changed"
        )
        path = normalize_identifier(str(key))
        return mutate_response(
            scenario,
            response,
            f"response_field:replace:{path}.type",
        )

    add_group("response_type", scalar_response_pool, change_response_type)
    add_group(
        "unavailable",
        base_pool,
        lambda scenario, _: mutate_availability(
            scenario,
            [503, 503, 503],
            LivenessState.UNAVAILABLE,
        ),
    )
    add_group(
        "intermittent",
        base_pool,
        lambda scenario, _: mutate_availability(
            scenario,
            [200, 503, 200, 503, 200],
            LivenessState.DEGRADED,
        ),
    )
    add_group(
        "not_found",
        base_pool,
        lambda scenario, _: mutate_availability(
            scenario,
            [404, 404, 404],
            LivenessState.UNAVAILABLE,
        ),
    )
    add_group(
        "auth_blocked",
        base_pool,
        lambda scenario, _: mutate_availability(
            scenario,
            [401, 401],
            LivenessState.AUTH_BLOCKED,
        ),
    )
    add_group(
        "rate_limited",
        base_pool,
        lambda scenario, _: mutate_availability(
            scenario,
            [429, 429],
            LivenessState.RATE_LIMITED,
        ),
    )
    add_group(
        "recovery",
        base_pool,
        lambda scenario, _: mutate_availability(
            scenario,
            [503, 503, 503, 200, 200, 200],
            LivenessState.HEALTHY,
        ),
    )
    return scenarios


def _event(change: Any) -> str:
    return f"{change.kind.value}:{change.operation.value}:{change.path}"


def _set_metrics(
    expected: Iterable[set[str]], predicted: Iterable[set[str]]
) -> dict[str, float | int]:
    expected_list = list(expected)
    predicted_list = list(predicted)
    true_positive = sum(
        len(gold & found)
        for gold, found in zip(expected_list, predicted_list, strict=True)
    )
    false_positive = sum(
        len(found - gold)
        for gold, found in zip(expected_list, predicted_list, strict=True)
    )
    false_negative = sum(
        len(gold - found)
        for gold, found in zip(expected_list, predicted_list, strict=True)
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    exact = sum(
        gold == found
        for gold, found in zip(expected_list, predicted_list, strict=True)
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_case_rate": round(exact / max(1, len(expected_list)), 4),
    }


def _liveness_metrics(
    expected: list[str], predicted: list[str]
) -> dict[str, Any]:
    labels = sorted(set(expected) | set(predicted))
    by_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(
            gold == label and found == label
            for gold, found in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            gold != label and found == label
            for gold, found in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            gold == label and found != label
            for gold, found in zip(expected, predicted, strict=True)
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(f1)
        by_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(item == label for item in expected),
        }
    accuracy = sum(
        gold == found
        for gold, found in zip(expected, predicted, strict=True)
    ) / max(1, len(expected))
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 4),
        "by_class": by_class,
        "confusion": dict(
            Counter(
                f"{gold}->{found}"
                for gold, found in zip(expected, predicted, strict=True)
            )
        ),
    }


def run_controlled_benchmark(
    scenarios: list[ControlledScenario],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with ControlledApiServer(scenarios) as server:
        for scenario in scenarios:
            document = server.document(scenario.scenario_id)
            observations = explore_controlled_api(document)
            liveness = assess_liveness(
                observations,
                source=ObservationSource.CONTROLLED,
            )
            _, delta = detect_contract_drift(
                canonicalize(document), observations
            )
            cases.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "tool": document.tool_name,
                    "api": document.api_name,
                    "expected_events": sorted(set(scenario.expected_events)),
                    "detected_events": sorted(
                        {_event(change) for change in delta.changes}
                    ),
                    "confirmed_events": sorted(
                        {_event(change) for change in delta.confirmed_changes}
                    ),
                    "expected_liveness": scenario.expected_liveness.value,
                    "detected_liveness": liveness.state.value,
                    "observation_statuses": [
                        item.status_code for item in observations
                    ],
                    "observation_count": len(observations),
                }
            )

    expected_events = [set(case["expected_events"]) for case in cases]
    detected_events = [set(case["detected_events"]) for case in cases]
    confirmed_events = [set(case["confirmed_events"]) for case in cases]
    expected_liveness = [case["expected_liveness"] for case in cases]
    detected_liveness = [case["detected_liveness"] for case in cases]
    return {
        "scenario_count": len(cases),
        "contract_all_changes": _set_metrics(
            expected_events, detected_events
        ),
        "contract_auto_applicable": _set_metrics(
            expected_events, confirmed_events
        ),
        "liveness": _liveness_metrics(
            expected_liveness, detected_liveness
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled ToolBench contract-drift benchmark."
    )
    parser.add_argument(
        "--toolenv",
        type=Path,
        default=Path(
            "data/raw/stabletoolbench/toolenv2404/"
            "toolenv2404_filtered"
        ),
    )
    parser.add_argument("--seeds-per-mutation", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/controlled_benchmark.json"),
    )
    args = parser.parse_args()

    documents = load_toolenv(args.toolenv)
    scenarios = build_scenarios(
        documents,
        seeds_per_mutation=args.seeds_per_mutation,
        random_seed=args.seed,
    )
    result = run_controlled_benchmark(scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in result.items() if key != "cases"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

