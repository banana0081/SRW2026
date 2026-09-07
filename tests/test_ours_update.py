"""Gates for evidence-gated Ours updates."""

from __future__ import annotations

from datetime import datetime, timezone

from tooldoc_nir.dynamic_models import ExecutionObservation, ObservationSource
from tooldoc_nir.models import ApiDocument
from tooldoc_nir.ours_update import update_from_observations


def _document() -> ApiDocument:
    return ApiDocument(
        category_name="Search",
        tool_name="Catalog",
        api_name="find",
        api_description="Find records.",
        method="GET",
        required_parameters=[
            {"name": "q", "type": "STRING", "description": "query"}
        ],
    )


def test_missing_parameter_evidence_is_applied() -> None:
    document = _document()
    observations = [
        ExecutionObservation(
            api_key=document.key,
            source=ObservationSource.CONTROLLED,
            observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            request_method="GET",
            request_arguments={"q": "cats"},
            status_code=422,
            response_body={
                "error": {
                    "code": "missing_parameter",
                    "parameter": "tooldoc_locale",
                    "expected_type": "string",
                }
            },
        )
    ]
    update = update_from_observations(document, observations)
    names = {item.name for item in update.document.required_parameters}
    assert "tooldoc_locale" in names
    assert update.applied_operations >= 1
    assert update.unsupported_operations == []


def test_unevidenced_response_removal_is_not_applied() -> None:
    document = _document().model_copy(
        update={"template_response": {"result": "str", "count": "int"}}
    )
    observations = [
        ExecutionObservation(
            api_key=document.key,
            source=ObservationSource.CONTROLLED,
            observed_at=datetime(2026, 1, 1, minute=index, tzinfo=timezone.utc),
            request_method="GET",
            request_arguments={"q": "cats"},
            status_code=200,
            response_body={"result": "ok"},
        )
        for index in range(3)
    ]
    update = update_from_observations(document, observations)
    assert "count" not in {item.name for item in update.document.required_parameters}
    # Response-field removal is recorded as not auto-applicable.
    assert update.patch.operations == [] or all(
        "response_fields" not in operation.path
        or operation.op.value != "remove"
        for operation in update.patch.operations
    )
