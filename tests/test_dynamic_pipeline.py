from datetime import datetime, timedelta, timezone

from tooldoc_nir.canonicalize import canonicalize
from tooldoc_nir.contract import detect_contract_drift
from tooldoc_nir.controlled import (
    ControlledApiServer,
    ControlledScenario,
    explore_controlled_api,
    mutate_add_required,
    mutate_method,
    mutate_remove_parameter,
    mutate_response,
)
from tooldoc_nir.dynamic_models import (
    ChangeKind,
    ChangeOperation,
    ExecutionObservation,
    LivenessState,
    ObservationSource,
)
from tooldoc_nir.liveness import assess_liveness
from tooldoc_nir.models import ApiDocument


def _document() -> ApiDocument:
    return ApiDocument(
        category_name="Search",
        tool_name="Example Search",
        api_name="find",
        api_description="Find records matching a query.",
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
            }
        ],
        template_response={"result": "str", "count": "int"},
    )


def _observation(
    index: int,
    *,
    status: int | None,
    body: object = None,
    headers: dict[str, str] | None = None,
    source: ObservationSource = ObservationSource.CONTROLLED,
) -> ExecutionObservation:
    return ExecutionObservation(
        api_key=("example_search", "find"),
        source=source,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        + timedelta(minutes=index),
        request_method="GET",
        request_arguments={"q": "weather", "locale": "en"},
        status_code=status,
        response_headers=headers or {},
        response_body=body,
        error_type="timeout" if status is None else "",
    )


def test_liveness_requires_scoped_evidence() -> None:
    simulator_observation = _observation(
        0,
        status=200,
        body={"result": "ok"},
        source=ObservationSource.STABLE_SIMULATOR,
    )
    assessment = assess_liveness(
        [simulator_observation],
        source=ObservationSource.REAL,
    )
    assert assessment.state is LivenessState.UNKNOWN
    assert assessment.confidence == 0.0


def test_repeated_failures_establish_unavailable() -> None:
    observations = [
        _observation(index, status=503, body={"error": "maintenance"})
        for index in range(3)
    ]
    assessment = assess_liveness(
        observations,
        source=ObservationSource.CONTROLLED,
    )
    assert assessment.state is LivenessState.UNAVAILABLE
    assert assessment.confidence == 1.0


def test_mixed_success_and_failure_is_degraded() -> None:
    observations = [
        _observation(0, status=200, body={"result": "ok"}),
        _observation(1, status=503, body={"error": "maintenance"}),
    ]
    assessment = assess_liveness(
        observations,
        source=ObservationSource.CONTROLLED,
    )
    assert assessment.state is LivenessState.DEGRADED


def test_auth_and_rate_limits_are_not_reported_as_downtime() -> None:
    auth = [
        _observation(index, status=401, body={"error": "unauthorized"})
        for index in range(2)
    ]
    rate_limited = [
        _observation(index, status=429, body={"error": "slow down"})
        for index in range(2)
    ]
    assert (
        assess_liveness(
            auth,
            source=ObservationSource.CONTROLLED,
        ).state
        is LivenessState.AUTH_BLOCKED
    )
    assert (
        assess_liveness(
            rate_limited,
            source=ObservationSource.CONTROLLED,
        ).state
        is LivenessState.RATE_LIMITED
    )


def test_two_successes_confirm_recovery() -> None:
    observations = [
        _observation(0, status=503),
        _observation(1, status=503),
        _observation(2, status=503),
        _observation(3, status=200, body={"result": "ok"}),
        _observation(4, status=200, body={"result": "ok"}),
    ]
    assessment = assess_liveness(
        observations,
        source=ObservationSource.CONTROLLED,
    )
    assert assessment.state is LivenessState.HEALTHY
    assert "recovered" in assessment.reason


def test_contract_drift_is_typed_and_guarded() -> None:
    profile = canonicalize(_document())
    observations = [
        _observation(
            0,
            status=405,
            body={"error": "method changed"},
            headers={"allow": "POST"},
        ),
        _observation(
            1,
            status=422,
            body={
                "error": {
                    "code": "missing_parameter",
                    "parameter": "locale",
                    "expected_type": "string",
                }
            },
        ),
        _observation(
            2,
            status=200,
            body={"result": "sunny", "items": [{"id": 1}]},
        ),
        _observation(
            3,
            status=200,
            body={"result": "cloudy", "items": [{"id": 2}]},
        ),
    ]

    _, delta = detect_contract_drift(profile, observations)
    index = {
        (change.kind, change.operation, change.path): change
        for change in delta.changes
    }

    method = index[
        (ChangeKind.METHOD, ChangeOperation.REPLACE, "method")
    ]
    added_parameter = index[
        (ChangeKind.REQUEST_FIELD, ChangeOperation.ADD, "locale")
    ]
    added_output = index[
        (ChangeKind.RESPONSE_FIELD, ChangeOperation.ADD, "items")
    ]
    suspected_removal = index[
        (ChangeKind.RESPONSE_FIELD, ChangeOperation.REMOVE, "count")
    ]

    assert method.new_value == "POST"
    assert method.auto_applicable
    assert added_parameter.auto_applicable
    assert added_output.auto_applicable
    assert not suspected_removal.auto_applicable


def test_end_to_end_controlled_exploration_detects_drift() -> None:
    scenario = ControlledScenario.unchanged("combined", _document())
    scenario = mutate_add_required(scenario, "locale", "string")
    scenario = mutate_remove_parameter(scenario, "limit")
    scenario = mutate_method(scenario, "POST")
    scenario = mutate_response(
        scenario,
        {"result": "sunny", "items": [{"id": 1}]},
        "response:add:items",
        "response:remove:count",
    )

    with ControlledApiServer([scenario]) as server:
        document = server.document("combined")
        observations = explore_controlled_api(document)

    assert sum(item.status_code == 200 for item in observations) == 3
    profile = canonicalize(document)
    _, delta = detect_contract_drift(profile, observations)
    detected = {
        (change.kind.value, change.operation.value, change.path)
        for change in delta.changes
    }
    assert ("method", "replace", "method") in detected
    assert ("request_field", "add", "locale") in detected
    assert ("request_field", "remove", "limit") in detected
    assert ("response_field", "add", "items") in detected
    assert ("response_field", "remove", "count") in detected

