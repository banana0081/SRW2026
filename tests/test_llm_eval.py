from tooldoc_nir.controlled import ControlledApiServer
from tooldoc_nir.llm_eval import (
    CaseKind,
    Condition,
    _extract_call,
    build_initial_scenarios,
    prepare_cases,
    summarize_records,
)
from tooldoc_nir.models import ApiDocument


def _documents(count: int) -> list[ApiDocument]:
    return [
        ApiDocument(
            category_name="Search",
            tool_name=f"Example Search {index}",
            api_name="find",
            api_description="Find records matching a supplied search query.",
            method="GET",
            required_parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "description": "Search query.",
                }
            ],
            template_response={"result": "string"},
        )
        for index in range(count)
    ]


def test_prepared_cases_are_derived_from_execution_evidence() -> None:
    scenarios = build_initial_scenarios(
        _documents(4),
        cases_per_kind=2,
        seed=7,
    )
    with ControlledApiServer(scenarios) as server:
        cases = prepare_cases(server, scenarios)

    contract_cases = [case for case in cases if case.kind is CaseKind.CONTRACT]
    liveness_cases = [case for case in cases if case.kind is CaseKind.LIVENESS]
    assert len(contract_cases) == 2
    assert len(liveness_cases) == 2
    for case in contract_cases:
        assert "execution_locale" not in case.stale_schema["properties"]
        assert "execution_locale" in case.updated_schema["properties"]
        assert "execution_locale" in case.updated_schema["required"]
        assert case.evidence_ids
    for case in liveness_cases:
        assert "UNAVAILABLE" not in case.stale_description
        assert "UNAVAILABLE" in case.updated_description
        assert case.evidence_ids


def test_extract_call_parses_openrouter_tool_arguments() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "call_contract_000",
                                "arguments": '{"query":"weather"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    call = _extract_call(response)
    assert call["called"]
    assert call["arguments"] == {"query": "weather"}
    assert call["parse_error"] == ""


def test_summary_uses_paired_exact_test() -> None:
    records = []
    for kind in CaseKind:
        for index in range(8):
            records.extend(
                [
                    {
                        "case_id": f"{kind.value}_{index:03d}",
                        "kind": kind.value,
                        "condition": Condition.STALE.value,
                        "success": False,
                        "called": True,
                        "runtime_status": 422,
                        "usage": {},
                    },
                    {
                        "case_id": f"{kind.value}_{index:03d}",
                        "kind": kind.value,
                        "condition": Condition.UPDATED.value,
                        "success": True,
                        "called": kind is CaseKind.CONTRACT,
                        "runtime_status": (
                            200 if kind is CaseKind.CONTRACT else None
                        ),
                        "usage": {},
                    },
                ]
            )

    summary = summarize_records(records)
    assert summary["initial_h4_supported"]
    assert summary["paired_effects"]["contract"]["absolute_gain"] == 1.0
    assert summary["paired_effects"]["liveness"]["mcnemar_exact_p"] == 0.007812
