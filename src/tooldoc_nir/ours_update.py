"""Update one API document from a frozen observation trace.

This is the Ours updater for the temporal pilot. It reuses contract drift,
liveness and JSON-patch validation. It never reads an evaluation query, a
mutation label or an oracle status. Schema identifiers are taken from the
validated patch; unsupported (unevidenced) operations are counted and not
applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonicalize import canonicalize, canonical_text
from .contract import detect_contract_drift, snapshot_from_profile
from .dynamic_models import (
    ContractDelta,
    ExecutionObservation,
    LivenessAssessment,
    ObservationSource,
)
from .liveness import assess_liveness
from .models import ApiDocument, ApiParameter
from .patching import (
    DocumentationPatch,
    PatchStatus,
    build_contract_patch,
    validate_patch,
)


@dataclass
class OursUpdate:
    document: ApiDocument
    liveness: LivenessAssessment
    delta: ContractDelta
    patch: DocumentationPatch
    applied_operations: int = 0
    unsupported_operations: list[str] = field(default_factory=list)
    documentation_text: str = ""


def _document_from_state(
    baseline: ApiDocument,
    state: dict[str, Any],
    liveness: LivenessAssessment,
) -> ApiDocument:
    required: list[ApiParameter] = []
    optional: list[ApiParameter] = []
    for path, spec in (state.get("request_fields") or {}).items():
        if not isinstance(spec, dict):
            continue
        parameter = ApiParameter(
            name=str(path),
            type=str(spec.get("type") or ""),
            description="",
        )
        if spec.get("required"):
            required.append(parameter)
        else:
            optional.append(parameter)
    note = (
        f"Current liveness: {liveness.state.value} "
        f"(confidence {liveness.confidence:.2f}). {liveness.reason}"
    )
    description = " ".join(
        part for part in (baseline.api_description.strip(), note) if part
    )
    return baseline.model_copy(
        update={
            "method": str(state.get("method") or baseline.method),
            "required_parameters": required,
            "optional_parameters": optional,
            "api_description": description,
        }
    )


def update_from_observations(
    document: ApiDocument,
    observations: Iterable[ExecutionObservation],
    *,
    source: ObservationSource = ObservationSource.CONTROLLED,
) -> OursUpdate:
    selected = list(observations)
    profile = canonicalize(document)
    baseline = snapshot_from_profile(profile)
    observed, delta = detect_contract_drift(profile, selected)
    liveness = assess_liveness(selected, source=source)
    patch = build_contract_patch(delta, auto_applicable_only=True)
    unsupported = [
        f"{change.operation.value}:{change.path}"
        for change in delta.changes
        if not change.auto_applicable and not change.evidence_ids
    ]
    validated, state = validate_patch(patch, baseline)
    if validated.status is PatchStatus.VALIDATED and state is not None:
        updated = _document_from_state(document, state, liveness)
        applied = len(validated.operations)
    else:
        updated = _document_from_state(
            document,
            {
                "method": document.method,
                "request_fields": {
                    item.name: {
                        "type": item.type,
                        "required": True,
                        "enum": [],
                    }
                    for item in document.required_parameters
                }
                | {
                    item.name: {
                        "type": item.type,
                        "required": False,
                        "enum": [],
                    }
                    for item in document.optional_parameters
                },
            },
            liveness,
        )
        applied = 0
    text = canonical_text(canonicalize(updated))
    return OursUpdate(
        document=updated,
        liveness=liveness,
        delta=delta,
        patch=validated,
        applied_operations=applied,
        unsupported_operations=unsupported,
        documentation_text=text,
    )
