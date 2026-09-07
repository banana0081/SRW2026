from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .dynamic_models import (
    ChangeKind,
    ChangeOperation,
    ContractChange,
    ContractDelta,
    ContractField,
    ContractSnapshot,
    ExecutionObservation,
)
from .models import CanonicalProfile, normalize_identifier


_TYPE_ALIASES = {
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "double": "number",
    "float": "number",
    "int": "integer",
    "integer": "integer",
    "list": "array",
    "long": "integer",
    "number": "number",
    "object": "object",
    "str": "string",
    "string": "string",
}


def normalize_type(value: str) -> str:
    cleaned = (value or "unknown").casefold().strip()
    cleaned = cleaned.split(",", 1)[0].strip()
    if any(token in cleaned for token in ("string", "str", "date", "time", "url", "uuid", "email", "enum")):
        return "string"
    if any(token in cleaned for token in ("integer", "int", "long")):
        return "integer"
    if any(token in cleaned for token in ("number", "float", "double", "decimal")):
        return "number"
    if any(token in cleaned for token in ("boolean", "bool")):
        return "boolean"
    if any(token in cleaned for token in ("array", "list", "tuple")):
        return "array"
    if any(token in cleaned for token in ("object", "dict", "map")):
        return "object"
    return _TYPE_ALIASES.get(cleaned, cleaned or "unknown")


def _is_schema_noise_path(path: str) -> bool:
    parts = path.split(".")
    if not parts:
        return True
    if any(part.startswith("$") for part in parts):
        return True
    if "properties" in parts or "additionalproperties" in parts:
        return True
    if "items" in parts[1:]:
        return True
    return parts[0] == "type" and len(parts) > 1


def types_compatible(documented: str, observed: str) -> bool:
    documented = normalize_type(documented)
    observed = normalize_type(observed)
    return documented == observed or {
        documented,
        observed,
    } <= {"integer", "number"}


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    return type(value).__name__.casefold()


def _flatten_response(
    value: Any, prefix: str = "", depth: int = 0
) -> dict[str, str]:
    if depth > 4:
        return {}
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, nested in value.items():
            name = normalize_identifier(str(key))
            path = f"{prefix}.{name}" if prefix else name
            result[path] = value_type(nested)
            result.update(_flatten_response(nested, path, depth + 1))
        return result
    if isinstance(value, list) and value:
        return _flatten_response(value[0], prefix, depth + 1)
    return {}


def snapshot_from_profile(profile: CanonicalProfile) -> ContractSnapshot:
    request_fields = [
        ContractField(
            path=parameter.name,
            type=normalize_type(parameter.type),
            required=parameter.required,
            enum=parameter.constraints,
        )
        for parameter in [*profile.required_inputs, *profile.optional_inputs]
    ]
    response_fields = [
        ContractField(
            path=path,
            type=normalize_type(profile.output_types.get(path, "unknown")),
        )
        for path in profile.output_fields
    ]
    return ContractSnapshot(
        api_key=profile.key,
        method=profile.method,
        request_fields=request_fields,
        response_fields=response_fields,
    )


def _validation_error(observation: ExecutionObservation) -> dict[str, Any] | None:
    body = observation.response_body
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict) and error.get("code"):
        return error
    if body.get("code") and body.get("parameter"):
        return body
    return None


def snapshot_from_observations(
    profile: CanonicalProfile,
    observations: Iterable[ExecutionObservation],
) -> ContractSnapshot:
    selected = list(observations)
    request_types: dict[str, Counter[str]] = defaultdict(Counter)
    request_evidence: dict[str, list[str]] = defaultdict(list)
    request_required: dict[str, bool] = {}
    response_types: dict[str, Counter[str]] = defaultdict(Counter)
    response_evidence: dict[str, list[str]] = defaultdict(list)
    successful_count = 0
    method_votes: Counter[str] = Counter()
    explicit_method = ""

    for observation in selected:
        method_votes[observation.request_method.upper()] += 1
        if observation.status_code == 405:
            allowed = observation.response_headers.get("allow", "")
            if allowed:
                explicit_method = allowed.split(",", 1)[0].strip().upper()

        error = _validation_error(observation)
        if error and error.get("code") == "missing_parameter":
            path = normalize_identifier(str(error.get("parameter", "")))
            if path:
                request_required[path] = True
                expected = normalize_type(str(error.get("expected_type", "unknown")))
                request_types[path][expected] += 1
                request_evidence[path].append(observation.observation_id)

        if observation.status_code is None or not (
            200 <= observation.status_code < 300
        ):
            continue
        successful_count += 1
        for name, value in observation.request_arguments.items():
            path = normalize_identifier(name)
            request_types[path][value_type(value)] += 1
            request_evidence[path].append(observation.observation_id)
        for path, inferred_type in _flatten_response(
            observation.response_body
        ).items():
            response_types[path][inferred_type] += 1
            response_evidence[path].append(observation.observation_id)

    def majority(counter: Counter[str]) -> str:
        return counter.most_common(1)[0][0] if counter else "unknown"

    request_fields = [
        ContractField(
            path=path,
            type=majority(types),
            required=request_required.get(path),
            observed_count=sum(types.values()),
            successful_count=successful_count,
            evidence_ids=list(dict.fromkeys(request_evidence[path])),
        )
        for path, types in sorted(request_types.items())
    ]
    response_fields = [
        ContractField(
            path=path,
            type=majority(types),
            observed_count=sum(types.values()),
            successful_count=successful_count,
            evidence_ids=list(dict.fromkeys(response_evidence[path])),
        )
        for path, types in sorted(response_types.items())
    ]
    method = explicit_method
    if not method and method_votes:
        method = method_votes.most_common(1)[0][0]
    return ContractSnapshot(
        api_key=profile.key,
        method=method,
        request_fields=request_fields,
        response_fields=response_fields,
        evidence_ids=[item.observation_id for item in selected],
    )


def _structural_changes(
    baseline: ContractSnapshot,
    observed: ContractSnapshot,
    *,
    min_response_observations: int,
) -> list[ContractChange]:
    changes: list[ContractChange] = []
    evidence = observed.evidence_ids
    if baseline.method and observed.method and baseline.method != observed.method:
        changes.append(
            ContractChange(
                kind=ChangeKind.METHOD,
                operation=ChangeOperation.REPLACE,
                path="method",
                old_value=baseline.method,
                new_value=observed.method,
                confidence=0.8,
                evidence_ids=evidence,
                auto_applicable=False,
                rationale="Observed method differs from documented method.",
            )
        )

    baseline_request = {field.path: field for field in baseline.request_fields}
    for field in observed.request_fields:
        previous = baseline_request.get(field.path)
        if previous is None:
            confidence = 0.95 if field.required else 0.75
            changes.append(
                ContractChange(
                    kind=ChangeKind.REQUEST_FIELD,
                    operation=ChangeOperation.ADD,
                    path=field.path,
                    new_value=field.model_dump(mode="json"),
                    confidence=confidence,
                    evidence_ids=field.evidence_ids,
                    auto_applicable=bool(field.required),
                    rationale="Parameter is accepted or explicitly required at runtime.",
                )
            )
            continue
        if (
            previous.type != "unknown"
            and field.type != "unknown"
            and not types_compatible(previous.type, field.type)
        ):
            changes.append(
                ContractChange(
                    kind=ChangeKind.REQUEST_FIELD,
                    operation=ChangeOperation.REPLACE,
                    path=f"{field.path}.type",
                    old_value=previous.type,
                    new_value=field.type,
                    confidence=min(0.95, 0.65 + 0.1 * field.observed_count),
                    evidence_ids=field.evidence_ids,
                    auto_applicable=field.observed_count >= 2,
                    rationale="Runtime values consistently use a different type.",
                )
            )
        if field.required is not None and field.required != previous.required:
            changes.append(
                ContractChange(
                    kind=ChangeKind.REQUEST_FIELD,
                    operation=ChangeOperation.REPLACE,
                    path=f"{field.path}.required",
                    old_value=previous.required,
                    new_value=field.required,
                    confidence=0.95,
                    evidence_ids=field.evidence_ids,
                    auto_applicable=True,
                    rationale="Validation response explicitly changed requiredness.",
                )
            )

    observed_response = {
        field.path: field
        for field in observed.response_fields
        if not _is_schema_noise_path(field.path)
    }
    baseline_response = {
        field.path: field
        for field in baseline.response_fields
        if not _is_schema_noise_path(field.path)
    }
    for path, field in observed_response.items():
        previous = baseline_response.get(path)
        if previous is None:
            enough = field.observed_count >= min_response_observations
            changes.append(
                ContractChange(
                    kind=ChangeKind.RESPONSE_FIELD,
                    operation=ChangeOperation.ADD,
                    path=path,
                    new_value=field.type,
                    confidence=min(0.98, 0.55 + 0.12 * field.observed_count),
                    evidence_ids=field.evidence_ids,
                    auto_applicable=enough,
                    rationale="Previously undocumented field appears in responses.",
                )
            )
        elif (
            previous.type != "unknown"
            and field.type != "unknown"
            and not types_compatible(previous.type, field.type)
        ):
            enough = field.observed_count >= min_response_observations
            changes.append(
                ContractChange(
                    kind=ChangeKind.RESPONSE_FIELD,
                    operation=ChangeOperation.REPLACE,
                    path=f"{path}.type",
                    old_value=previous.type,
                    new_value=field.type,
                    confidence=min(0.98, 0.55 + 0.12 * field.observed_count),
                    evidence_ids=field.evidence_ids,
                    auto_applicable=enough,
                    rationale="Response field consistently has a different type.",
                )
            )

    successful_count = max(
        (field.successful_count for field in observed.response_fields),
        default=0,
    )
    if successful_count >= min_response_observations:
        for path, previous in baseline_response.items():
            if path not in observed_response:
                changes.append(
                    ContractChange(
                        kind=ChangeKind.RESPONSE_FIELD,
                        operation=ChangeOperation.REMOVE,
                        path=path,
                        old_value=previous.type,
                        confidence=min(0.8, 0.35 + 0.1 * successful_count),
                        evidence_ids=evidence,
                        auto_applicable=False,
                        rationale=(
                            "Field was absent from repeated responses; removal "
                            "remains unconfirmed because fields may be optional."
                        ),
                    )
                )
    return changes


def _explicit_validation_changes(
    baseline: ContractSnapshot,
    observations: Iterable[ExecutionObservation],
) -> list[ContractChange]:
    changes: list[ContractChange] = []
    baseline_fields = {field.path: field for field in baseline.request_fields}
    for observation in observations:
        if observation.status_code == 405:
            allowed = observation.response_headers.get("allow", "")
            if allowed:
                method = allowed.split(",", 1)[0].strip().upper()
                changes.append(
                    ContractChange(
                        kind=ChangeKind.METHOD,
                        operation=ChangeOperation.REPLACE,
                        path="method",
                        old_value=baseline.method,
                        new_value=method,
                        confidence=0.99,
                        evidence_ids=[observation.observation_id],
                        auto_applicable=True,
                        rationale="HTTP 405 response explicitly declares Allow.",
                    )
                )

        error = _validation_error(observation)
        if not error:
            continue
        code = str(error.get("code", ""))
        path = normalize_identifier(str(error.get("parameter", "")))
        if not path:
            continue
        previous = baseline_fields.get(path)
        if code == "missing_parameter":
            if previous is None:
                changes.append(
                    ContractChange(
                        kind=ChangeKind.REQUEST_FIELD,
                        operation=ChangeOperation.ADD,
                        path=path,
                        new_value={
                            "type": normalize_type(
                                str(error.get("expected_type", "unknown"))
                            ),
                            "required": True,
                        },
                        confidence=0.99,
                        evidence_ids=[observation.observation_id],
                        auto_applicable=True,
                        rationale="Validation error explicitly names a new requirement.",
                    )
                )
            elif previous.required is not True:
                changes.append(
                    ContractChange(
                        kind=ChangeKind.REQUEST_FIELD,
                        operation=ChangeOperation.REPLACE,
                        path=f"{path}.required",
                        old_value=previous.required,
                        new_value=True,
                        confidence=0.99,
                        evidence_ids=[observation.observation_id],
                        auto_applicable=True,
                        rationale="Validation error explicitly requires the parameter.",
                    )
                )
        elif code == "unknown_parameter" and previous is not None:
            changes.append(
                ContractChange(
                    kind=ChangeKind.REQUEST_FIELD,
                    operation=ChangeOperation.REMOVE,
                    path=path,
                    old_value=previous.model_dump(mode="json"),
                    confidence=0.99,
                    evidence_ids=[observation.observation_id],
                    auto_applicable=True,
                    rationale="Validation error explicitly rejects the parameter.",
                )
            )
        elif code == "invalid_type" and previous is not None:
            expected = normalize_type(str(error.get("expected_type", "unknown")))
            changes.append(
                ContractChange(
                    kind=ChangeKind.REQUEST_FIELD,
                    operation=ChangeOperation.REPLACE,
                    path=f"{path}.type",
                    old_value=previous.type,
                    new_value=expected,
                    confidence=0.99,
                    evidence_ids=[observation.observation_id],
                    auto_applicable=expected != "unknown",
                    rationale="Validation error explicitly declares the expected type.",
                )
            )
    return changes


def _deduplicate(changes: Iterable[ContractChange]) -> list[ContractChange]:
    selected: dict[tuple[str, str, str], ContractChange] = {}
    for change in changes:
        key = (change.kind.value, change.operation.value, change.path)
        previous = selected.get(key)
        if previous is None or change.confidence > previous.confidence:
            selected[key] = change
        elif change.confidence == previous.confidence:
            previous.evidence_ids = list(
                dict.fromkeys([*previous.evidence_ids, *change.evidence_ids])
            )
    return sorted(
        selected.values(),
        key=lambda item: (item.kind.value, item.path, item.operation.value),
    )


def detect_contract_drift(
    profile: CanonicalProfile,
    observations: Iterable[ExecutionObservation],
    *,
    min_response_observations: int = 2,
) -> tuple[ContractSnapshot, ContractDelta]:
    selected = list(observations)
    baseline = snapshot_from_profile(profile)
    observed = snapshot_from_observations(profile, selected)
    changes = _deduplicate(
        [
            *_structural_changes(
                baseline,
                observed,
                min_response_observations=min_response_observations,
            ),
            *_explicit_validation_changes(baseline, selected),
        ]
    )
    return observed, ContractDelta(
        api_key=profile.key,
        baseline_hash=baseline.source_hash,
        observed_hash=observed.source_hash,
        changes=changes,
    )

