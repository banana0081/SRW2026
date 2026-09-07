from tooldoc_nir.controlled import (
    ControlledScenario,
    mutate_add_required,
    mutate_availability,
    mutate_method,
)
from tooldoc_nir.dynamic_models import LivenessState
from tooldoc_nir.fixed_trace import run_fixed_trace
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
        optional_parameters=[],
        template_response={"result": "ok"},
    )


def test_ours_beats_draft_after_required_field_is_added() -> None:
    scenario = mutate_add_required(
        ControlledScenario.unchanged("add_required_000", _document()),
        "locale",
        "string",
    )
    result = run_fixed_trace([scenario])
    case = result["cases"][0]
    assert case["success"]["ours"]
    assert case["success"]["oracle"]
    assert not case["success"]["raw"]
    assert not case["success"]["draft"]
    assert "request_field:add:locale" in case["ours_request_events"]


def test_draft_false_down_after_recovery_ours_calls() -> None:
    scenario = mutate_availability(
        ControlledScenario.unchanged("recovery_000", _document()),
        [503, 503, 503, 200, 200, 200],
        LivenessState.HEALTHY,
    )
    result = run_fixed_trace([scenario])
    case = result["cases"][0]
    assert case["draft_refuses"]
    assert not case["ours_refuses"]
    assert case["success"]["ours"]
    assert not case["success"]["draft"]
    assert case["evaluations"]["draft"]["false_down"]


def test_ours_and_draft_both_mark_persistent_outage() -> None:
    scenario = mutate_availability(
        ControlledScenario.unchanged("unavailable_000", _document()),
        [503, 503, 503],
        LivenessState.UNAVAILABLE,
    )
    result = run_fixed_trace([scenario])
    case = result["cases"][0]
    assert case["success"]["ours"]
    assert case["success"]["draft"]
    assert not case["success"]["raw"]


def test_method_change_is_applied_by_ours_only() -> None:
    scenario = mutate_method(
        ControlledScenario.unchanged("method_000", _document()),
        "POST",
    )
    result = run_fixed_trace([scenario])
    case = result["cases"][0]
    assert case["success"]["ours"]
    assert not case["success"]["raw"]
    assert not case["success"]["draft"]
    assert "method:replace:method" in case["ours_request_events"]
