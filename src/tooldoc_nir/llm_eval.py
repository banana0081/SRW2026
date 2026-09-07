from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from enum import StrEnum
import json
from math import comb, sqrt
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable

from pydantic import BaseModel, Field

from .canonicalize import canonicalize, clean_text
from .contract import (
    detect_contract_drift,
    normalize_type,
    snapshot_from_profile,
)
from .controlled import (
    ControlledApiServer,
    ControlledScenario,
    explore_controlled_api,
    mutate_add_required,
    mutate_availability,
)
from .dynamic_models import LivenessAssessment, LivenessState, ObservationSource
from .liveness import assess_liveness
from .models import ApiDocument, CanonicalProfile
from .openrouter import OpenRouterClient, OpenRouterError
from .patching import (
    PatchStatus,
    build_contract_patch,
    contract_state,
    validate_patch,
)
from .probe import HttpProber
from .toolenv import load_toolenv


DEFAULT_MODEL = "z-ai/glm-5.2:free"
DEFAULT_PROVIDER = "Decart"
SYNTHETIC_REQUIRED_FIELD = "execution_locale"
_FUNCTION_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class CaseKind(StrEnum):
    CONTRACT = "contract"
    LIVENESS = "liveness"


class Condition(StrEnum):
    STALE = "stale"
    UPDATED = "updated"


class EvaluationCase(BaseModel):
    case_id: str
    kind: CaseKind
    scenario_id: str
    api_key: tuple[str, str]
    function_name: str
    query: str
    requested_arguments: dict[str, Any] = Field(default_factory=dict)
    stale_description: str
    updated_description: str
    stale_schema: dict[str, Any]
    updated_schema: dict[str, Any]
    expected_action: str
    evidence_ids: list[str] = Field(default_factory=list)
    patch_operations: list[dict[str, Any]] = Field(default_factory=list)
    liveness: dict[str, Any] | None = None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _function_name(case_id: str) -> str:
    cleaned = _FUNCTION_NAME_RE.sub("_", case_id).strip("_")
    return f"call_{cleaned}"[:64]


def _json_schema_type(value: str) -> str:
    normalized = normalize_type(value)
    return (
        normalized
        if normalized
        in {"array", "boolean", "integer", "number", "object", "string"}
        else "string"
    )


def _json_safe_default(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return value


def _profile_parameter_metadata(
    profile: CanonicalProfile,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for parameter in [*profile.required_inputs, *profile.optional_inputs]:
        result[parameter.name] = {
            "description": clean_text(parameter.description)[:500],
            "default": _json_safe_default(parameter.default),
        }
    return result


def _tool_schema(
    state: dict[str, Any],
    profile: CanonicalProfile,
) -> dict[str, Any]:
    metadata = _profile_parameter_metadata(profile)
    properties: dict[str, Any] = {}
    required: list[str] = []
    request_fields = state.get("request_fields", {})
    for name, field in sorted(request_fields.items()):
        field_type = _json_schema_type(str(field.get("type", "string")))
        property_schema: dict[str, Any] = {"type": field_type}
        details = metadata.get(name, {})
        if details.get("description"):
            property_schema["description"] = details["description"]
        default = details.get("default")
        if default not in (None, ""):
            property_schema["default"] = default
        properties[name] = property_schema
        if field.get("required") is True:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tool_description(
    profile: CanonicalProfile,
    *,
    liveness: LivenessAssessment | None = None,
    execution_updated: bool,
) -> str:
    pieces = [
        f"API: {profile.api_name}.",
        f"Tool: {profile.tool_name}.",
        f"Purpose: {profile.purpose}",
        f"HTTP method: {profile.method or 'unspecified'}.",
    ]
    if execution_updated:
        pieces.append("Contract status: updated from execution evidence.")
    else:
        pieces.append("Contract status: original static documentation.")
    if liveness is not None:
        pieces.append(
            "Liveness: "
            f"{liveness.state.value.upper()} "
            f"(confidence={liveness.confidence:.3f}, "
            f"evidence_count={len(liveness.evidence_ids)})."
        )
        pieces.append(f"Liveness evidence: {liveness.reason}")
    return clean_text(" ".join(pieces))[:1800]


def _example_value(name: str, field_type: str) -> Any:
    if name == SYNTHETIC_REQUIRED_FIELD:
        return "en-US"
    normalized = normalize_type(field_type)
    return {
        "array": ["sample"],
        "boolean": True,
        "integer": 7,
        "number": 7.5,
        "object": {"key": "sample"},
    }.get(normalized, f"sample_{name}")


def _runtime_arguments(scenario: ControlledScenario) -> dict[str, Any]:
    return {
        name: _example_value(name, field_type)
        for name, field_type in sorted(scenario.runtime_required.items())
    }


def _eligible_documents(documents: Iterable[ApiDocument]) -> list[ApiDocument]:
    selected: list[ApiDocument] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        profile = canonicalize(document)
        parameters = [*profile.required_inputs, *profile.optional_inputs]
        names = [parameter.name for parameter in parameters]
        if (
            document.key in seen
            or (document.method or "GET").upper() != "GET"
            or len(clean_text(document.api_description)) < 30
            or len(profile.required_inputs) > 4
            or len(profile.optional_inputs) > 5
            or not names
            or len(names) != len(set(names))
            or SYNTHETIC_REQUIRED_FIELD in names
        ):
            continue
        seen.add(document.key)
        selected.append(document)
    return selected


def build_initial_scenarios(
    documents: list[ApiDocument],
    *,
    cases_per_kind: int,
    seed: int,
) -> list[ControlledScenario]:
    eligible = _eligible_documents(documents)
    required = cases_per_kind * 2
    if len(eligible) < required:
        raise ValueError(
            f"Need {required} eligible documents, found {len(eligible)}."
        )
    rng = random.Random(seed)
    chosen = rng.sample(eligible, required)
    scenarios: list[ControlledScenario] = []
    for index, document in enumerate(chosen[:cases_per_kind]):
        baseline = ControlledScenario.unchanged(
            f"contract_{index:03d}", document
        )
        scenarios.append(
            mutate_add_required(
                baseline,
                SYNTHETIC_REQUIRED_FIELD,
                "string",
            )
        )
    for index, document in enumerate(chosen[cases_per_kind:]):
        baseline = ControlledScenario.unchanged(
            f"liveness_{index:03d}", document
        )
        scenarios.append(
            mutate_availability(
                baseline,
                [503, 503, 503],
                LivenessState.UNAVAILABLE,
            )
        )
    return scenarios


def _contract_case(
    server: ControlledApiServer,
    scenario: ControlledScenario,
) -> EvaluationCase:
    document = server.document(scenario.scenario_id)
    profile = canonicalize(document)
    observations = explore_controlled_api(document)
    _, delta = detect_contract_drift(profile, observations)
    patch = build_contract_patch(delta)
    baseline = snapshot_from_profile(profile)
    validated_patch, updated_state = validate_patch(patch, baseline)
    if (
        validated_patch.status is not PatchStatus.VALIDATED
        or updated_state is None
    ):
        raise RuntimeError(
            f"Generated patch failed validation for {scenario.scenario_id}."
        )
    expected_path = f"/request_fields/{SYNTHETIC_REQUIRED_FIELD}"
    if not any(item.path == expected_path for item in patch.operations):
        raise RuntimeError(
            f"Required-field drift was not detected for {scenario.scenario_id}."
        )

    requested_arguments = _runtime_arguments(scenario)
    query = (
        f"Use the supplied API to do the following: {profile.purpose} "
        "Call the tool with these exact argument values: "
        f"{json.dumps(requested_arguments, ensure_ascii=False, sort_keys=True)}"
    )
    return EvaluationCase(
        case_id=scenario.scenario_id,
        kind=CaseKind.CONTRACT,
        scenario_id=scenario.scenario_id,
        api_key=document.key,
        function_name=_function_name(scenario.scenario_id),
        query=query,
        requested_arguments=requested_arguments,
        stale_description=_tool_description(
            profile,
            execution_updated=False,
        ),
        updated_description=_tool_description(
            profile,
            execution_updated=True,
        ),
        stale_schema=_tool_schema(contract_state(baseline), profile),
        updated_schema=_tool_schema(updated_state, profile),
        expected_action="call_runtime_valid",
        evidence_ids=patch.evidence_ids,
        patch_operations=[
            item.model_dump(mode="json") for item in patch.operations
        ],
    )


def _liveness_case(
    server: ControlledApiServer,
    scenario: ControlledScenario,
) -> EvaluationCase:
    document = server.document(scenario.scenario_id)
    profile = canonicalize(document)
    observations = explore_controlled_api(document)
    assessment = assess_liveness(
        observations,
        source=ObservationSource.CONTROLLED,
    )
    if assessment.state is not LivenessState.UNAVAILABLE:
        raise RuntimeError(
            f"Unavailable state was not established for {scenario.scenario_id}."
        )
    requested_arguments = _runtime_arguments(scenario)
    query = (
        f"Use the supplied API to do the following: {profile.purpose} "
        "If it is usable, call it with these exact argument values: "
        f"{json.dumps(requested_arguments, ensure_ascii=False, sort_keys=True)}"
    )
    state = contract_state(snapshot_from_profile(profile))
    return EvaluationCase(
        case_id=scenario.scenario_id,
        kind=CaseKind.LIVENESS,
        scenario_id=scenario.scenario_id,
        api_key=document.key,
        function_name=_function_name(scenario.scenario_id),
        query=query,
        requested_arguments=requested_arguments,
        stale_description=_tool_description(
            profile,
            execution_updated=False,
        ),
        updated_description=_tool_description(
            profile,
            liveness=assessment,
            execution_updated=True,
        ),
        stale_schema=_tool_schema(state, profile),
        updated_schema=_tool_schema(state, profile),
        expected_action="abstain_unavailable",
        evidence_ids=assessment.evidence_ids,
        liveness=assessment.model_dump(mode="json"),
    )


def prepare_cases(
    server: ControlledApiServer,
    scenarios: list[ControlledScenario],
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for scenario in scenarios:
        if scenario.scenario_id.startswith("contract_"):
            cases.append(_contract_case(server, scenario))
        elif scenario.scenario_id.startswith("liveness_"):
            cases.append(_liveness_case(server, scenario))
        else:
            raise ValueError(f"Unknown scenario kind: {scenario.scenario_id}")
    return cases


def _tool(case: EvaluationCase, condition: Condition) -> dict[str, Any]:
    if condition is Condition.UPDATED:
        description = case.updated_description
        schema = case.updated_schema
    else:
        description = case.stale_description
        schema = case.stale_schema
    return {
        "type": "function",
        "function": {
            "name": case.function_name,
            "description": description,
            "parameters": schema,
        },
    }


def _messages(case: EvaluationCase) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an API tool-use agent. Treat tools without an explicit "
                "liveness warning as usable. Never call a tool explicitly marked "
                "UNAVAILABLE. Use only parameters declared in the supplied current "
                "tool schema; never invent undocumented parameters. Copy requested "
                "argument values exactly. Use at most one tool call."
            ),
        },
        {"role": "user", "content": case.query},
    ]


def _extract_call(response: dict[str, Any]) -> dict[str, Any]:
    message = response["choices"][0].get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        return {
            "called": False,
            "name": "",
            "arguments": {},
            "parse_error": "",
            "call_count": 0,
            "assistant_content": message.get("content") or "",
        }
    call = calls[0]
    function = call.get("function") or {}
    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = (
            raw_arguments
            if isinstance(raw_arguments, dict)
            else json.loads(raw_arguments)
        )
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments are not a JSON object.")
        parse_error = ""
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        arguments = {}
        parse_error = str(exc)
    return {
        "called": True,
        "name": str(function.get("name") or ""),
        "arguments": arguments,
        "parse_error": parse_error,
        "call_count": len(calls),
        "assistant_content": message.get("content") or "",
    }


def _evaluate_response(
    *,
    case: EvaluationCase,
    condition: Condition,
    response: dict[str, Any],
    document: ApiDocument,
) -> dict[str, Any]:
    call = _extract_call(response)
    runtime_status: int | None = None
    runtime_error = ""
    if (
        call["called"]
        and not call["parse_error"]
        and call["name"] == case.function_name
        and call["call_count"] == 1
    ):
        observation = HttpProber(
            timeout_seconds=2.0,
            allow_unsafe_methods=True,
        ).probe(
            document,
            arguments=call["arguments"],
            source=ObservationSource.CONTROLLED,
        )
        runtime_status = observation.status_code
        runtime_error = observation.error_type

    if case.kind is CaseKind.CONTRACT:
        success = bool(runtime_status and 200 <= runtime_status < 300)
    else:
        success = not call["called"]

    usage = response.get("usage") or {}
    return {
        "case_id": case.case_id,
        "kind": case.kind.value,
        "condition": condition.value,
        "success": success,
        "called": call["called"],
        "called_name": call["name"],
        "call_count": call["call_count"],
        "arguments": call["arguments"],
        "parse_error": call["parse_error"],
        "assistant_content": call["assistant_content"],
        "runtime_status": runtime_status,
        "runtime_error": runtime_error,
        "usage": usage,
        "response_id": response.get("id", ""),
        "response_model": response.get("model", ""),
        "provider": response.get("provider", ""),
        "transport": response.get("_transport", {}),
        "completed_at": _utc_iso(),
    }


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _mcnemar_exact(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    lower = min(improved, regressed)
    probability = sum(comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if not record.get("error")]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in completed:
        grouped[(record["kind"], record["condition"])].append(record)

    condition_metrics: dict[str, Any] = {}
    for (kind, condition), rows in sorted(grouped.items()):
        successes = sum(bool(row["success"]) for row in rows)
        calls = sum(bool(row["called"]) for row in rows)
        runtime_successes = sum(
            row.get("runtime_status") is not None
            and 200 <= row["runtime_status"] < 300
            for row in rows
        )
        condition_metrics[f"{kind}/{condition}"] = {
            "cases": len(rows),
            "successes": successes,
            "success_rate": round(successes / len(rows), 4),
            "success_rate_95ci": _wilson_interval(successes, len(rows)),
            "call_rate": round(calls / len(rows), 4),
            "runtime_2xx_rate": round(runtime_successes / len(rows), 4),
        }

    by_case: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in completed:
        by_case[(record["kind"], record["case_id"])][record["condition"]] = record

    paired_by_kind: dict[str, Any] = {}
    for kind in CaseKind:
        pairs = [
            conditions
            for (case_kind, _), conditions in by_case.items()
            if case_kind == kind.value
            and Condition.STALE.value in conditions
            and Condition.UPDATED.value in conditions
        ]
        improved = sum(
            not pair[Condition.STALE.value]["success"]
            and pair[Condition.UPDATED.value]["success"]
            for pair in pairs
        )
        regressed = sum(
            pair[Condition.STALE.value]["success"]
            and not pair[Condition.UPDATED.value]["success"]
            for pair in pairs
        )
        stale_successes = sum(
            bool(pair[Condition.STALE.value]["success"]) for pair in pairs
        )
        updated_successes = sum(
            bool(pair[Condition.UPDATED.value]["success"]) for pair in pairs
        )
        paired_by_kind[kind.value] = {
            "pairs": len(pairs),
            "stale_successes": stale_successes,
            "updated_successes": updated_successes,
            "absolute_gain": round(
                (updated_successes - stale_successes) / max(1, len(pairs)),
                4,
            ),
            "improved_pairs": improved,
            "regressed_pairs": regressed,
            "mcnemar_exact_p": round(_mcnemar_exact(improved, regressed), 6),
        }

    supported = bool(paired_by_kind) and all(
        metric["pairs"] > 0
        and metric["absolute_gain"] > 0
        and metric["mcnemar_exact_p"] < 0.05
        for metric in paired_by_kind.values()
    )
    token_usage: dict[str, int | float] = {
        key: sum(
            int((record.get("usage") or {}).get(key) or 0)
            for record in completed
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    token_usage["reasoning_tokens"] = sum(
        int(
            ((record.get("usage") or {}).get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            or 0
        )
        for record in completed
    )
    token_usage["cost_usd"] = round(
        sum(float((record.get("usage") or {}).get("cost") or 0) for record in completed),
        8,
    )
    return {
        "completed_calls": len(completed),
        "error_calls": len(records) - len(completed),
        "condition_metrics": condition_metrics,
        "paired_effects": paired_by_kind,
        "token_usage": token_usage,
        "initial_h4_supported": supported,
        "claim_scope": (
            "Controlled ToolBench-seeded contract and liveness cases only; "
            "this is an initial downstream result, not a live-API generalization."
        ),
    }


def _write_artifact(
    path: Path,
    *,
    metadata: dict[str, Any],
    cases: list[EvaluationCase],
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "metadata": metadata,
        "cases": [case.model_dump(mode="json") for case in cases],
        "records": records,
        "summary": summarize_records(records),
    }
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def run_initial_evaluation(
    *,
    client: OpenRouterClient,
    documents: list[ApiDocument],
    model: str,
    provider: str | None,
    cases_per_kind: int,
    seed: int,
    output: Path,
    max_tokens: int,
    min_interval_seconds: float,
    resume: bool,
) -> dict[str, Any]:
    scenarios = build_initial_scenarios(
        documents,
        cases_per_kind=cases_per_kind,
        seed=seed,
    )
    metadata = {
        "experiment": "execution_backed_documentation_initial_h4",
        "model": model,
        "provider": provider,
        "seed": seed,
        "cases_per_kind": cases_per_kind,
        "started_at": _utc_iso(),
        "tool_choice": "auto",
        "temperature": 0.0,
        "reasoning_effort": "high",
        "max_tokens": max_tokens,
    }
    records = _load_records(output) if resume else []
    completed_keys = {
        (record.get("case_id"), record.get("condition"))
        for record in records
        if not record.get("error")
    }
    last_request_at = 0.0

    with ControlledApiServer(scenarios) as server:
        cases = prepare_cases(server, scenarios)
        order: list[tuple[EvaluationCase, Condition]] = []
        rng = random.Random(seed)
        for case in cases:
            conditions = [Condition.STALE, Condition.UPDATED]
            rng.shuffle(conditions)
            order.extend((case, condition) for condition in conditions)

        for case, condition in order:
            key = (case.case_id, condition.value)
            if key in completed_keys:
                continue
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval_seconds:
                time.sleep(min_interval_seconds - elapsed)
            try:
                response = client.chat_completion(
                    model=model,
                    provider=provider,
                    messages=_messages(case),
                    tools=[_tool(case, condition)],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=seed,
                    reasoning_effort="high",
                )
                last_request_at = time.monotonic()
                document = server.document(case.scenario_id)
                record = _evaluate_response(
                    case=case,
                    condition=condition,
                    response=response,
                    document=document,
                )
            except OpenRouterError as exc:
                last_request_at = time.monotonic()
                record = {
                    "case_id": case.case_id,
                    "kind": case.kind.value,
                    "condition": condition.value,
                    "error": str(exc),
                    "completed_at": _utc_iso(),
                }
                records.append(record)
                _write_artifact(
                    output,
                    metadata=metadata,
                    cases=cases,
                    records=records,
                )
                raise
            records.append(record)
            _write_artifact(
                output,
                metadata=metadata,
                cases=cases,
                records=records,
            )

    metadata["completed_at"] = _utc_iso()
    _write_artifact(
        output,
        metadata=metadata,
        cases=cases,
        records=records,
    )
    return summarize_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired OpenRouter evaluation of stale versus "
            "execution-updated tool documentation."
        )
    )
    parser.add_argument(
        "--toolenv",
        type=Path,
        default=Path(
            "data/raw/stabletoolbench/toolenv2404/"
            "toolenv2404_filtered"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--cases-per-kind", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--min-interval", type=float, default=3.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/glm52_free_initial.json"),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any completed records in the output artifact.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    documents = load_toolenv(args.toolenv)
    client = OpenRouterClient.from_env(env_path=args.env_file)
    summary = run_initial_evaluation(
        client=client,
        documents=documents,
        model=args.model,
        provider=args.provider or None,
        cases_per_kind=args.cases_per_kind,
        seed=args.seed,
        output=args.output,
        max_tokens=args.max_tokens,
        min_interval_seconds=args.min_interval,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
