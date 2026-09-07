from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any

from pydantic import BaseModel, Field

from .dynamic_models import (
    ChangeKind,
    ChangeOperation,
    ContractChange,
    ContractDelta,
    ContractSnapshot,
    utc_now,
)


class PatchStatus(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"


class JsonPatchOperation(BaseModel):
    op: ChangeOperation
    path: str
    value: Any = None


class DocumentationPatch(BaseModel):
    patch_id: str
    api_key: tuple[str, str]
    base_hash: str
    operations: list[JsonPatchOperation] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    status: PatchStatus = PatchStatus.PROPOSED
    validation_message: str = ""


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def contract_state(snapshot: ContractSnapshot) -> dict[str, Any]:
    def field_value(field: Any) -> dict[str, Any]:
        return {
            "type": field.type,
            "required": field.required,
            "enum": field.enum,
        }

    return {
        "method": snapshot.method,
        "request_fields": {
            field.path: field_value(field) for field in snapshot.request_fields
        },
        "response_fields": {
            field.path: field_value(field) for field in snapshot.response_fields
        },
    }


def _operation_path(change: ContractChange) -> str:
    if change.kind is ChangeKind.METHOD:
        return "/method"
    prefix = (
        "request_fields"
        if change.kind is ChangeKind.REQUEST_FIELD
        else "response_fields"
    )
    field_path = change.path
    suffix = ""
    for candidate in (".type", ".required"):
        if field_path.endswith(candidate):
            field_path = field_path[: -len(candidate)]
            suffix = candidate[1:]
            break
    pointer = f"/{prefix}/{_escape_pointer(field_path)}"
    if suffix:
        pointer += f"/{suffix}"
    return pointer


def _operation_value(change: ContractChange) -> Any:
    if change.operation is ChangeOperation.REMOVE:
        return None
    if (
        change.kind is ChangeKind.RESPONSE_FIELD
        and not change.path.endswith((".type", ".required"))
        and not isinstance(change.new_value, dict)
    ):
        return {
            "type": change.new_value,
            "required": None,
            "enum": [],
        }
    if (
        change.kind is ChangeKind.REQUEST_FIELD
        and change.operation is ChangeOperation.ADD
        and isinstance(change.new_value, dict)
    ):
        return {
            "type": change.new_value.get("type", "unknown"),
            "required": change.new_value.get("required"),
            "enum": change.new_value.get("enum", []),
        }
    return change.new_value


def build_contract_patch(
    delta: ContractDelta,
    *,
    auto_applicable_only: bool = True,
) -> DocumentationPatch:
    selected = [
        change
        for change in delta.changes
        if change.kind is not ChangeKind.LIVENESS
        and (change.auto_applicable or not auto_applicable_only)
    ]
    operations = [
        JsonPatchOperation(
            op=change.operation,
            path=_operation_path(change),
            value=_operation_value(change),
        )
        for change in selected
    ]
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for change in selected
            for evidence_id in change.evidence_ids
        )
    )
    payload = json.dumps(
        {
            "api_key": delta.api_key,
            "base_hash": delta.baseline_hash,
            "operations": [
                operation.model_dump(mode="json") for operation in operations
            ],
            "evidence_ids": evidence_ids,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return DocumentationPatch(
        patch_id=sha256(payload.encode("utf-8")).hexdigest()[:20],
        api_key=delta.api_key,
        base_hash=delta.baseline_hash,
        operations=operations,
        evidence_ids=evidence_ids,
    )


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {pointer!r}")
    return [
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer[1:].split("/")
    ]


def apply_json_patch(
    document: dict[str, Any],
    operations: list[JsonPatchOperation],
) -> dict[str, Any]:
    result = deepcopy(document)
    for operation in operations:
        parts = _pointer_parts(operation.path)
        parent: dict[str, Any] = result
        for part in parts[:-1]:
            nested = parent.get(part)
            if not isinstance(nested, dict):
                if operation.op is ChangeOperation.ADD:
                    nested = {}
                    parent[part] = nested
                else:
                    raise KeyError(operation.path)
            parent = nested
        leaf = parts[-1]
        if operation.op is ChangeOperation.ADD:
            parent[leaf] = deepcopy(operation.value)
        elif operation.op is ChangeOperation.REMOVE:
            if leaf not in parent:
                raise KeyError(operation.path)
            del parent[leaf]
        elif operation.op is ChangeOperation.REPLACE:
            if leaf not in parent:
                raise KeyError(operation.path)
            parent[leaf] = deepcopy(operation.value)
        else:
            raise ValueError(f"Unsupported patch operation: {operation.op}")
    return result


def validate_patch(
    patch: DocumentationPatch,
    baseline: ContractSnapshot,
) -> tuple[DocumentationPatch, dict[str, Any] | None]:
    validated = patch.model_copy(deep=True)
    if patch.base_hash != baseline.source_hash:
        validated.status = PatchStatus.REJECTED
        validated.validation_message = "Base contract hash does not match."
        return validated, None
    try:
        result = apply_json_patch(
            contract_state(baseline),
            patch.operations,
        )
    except (KeyError, ValueError) as exc:
        validated.status = PatchStatus.REJECTED
        validated.validation_message = str(exc)
        return validated, None

    validated.status = PatchStatus.VALIDATED
    validated.validation_message = "Patch applies cleanly to the base contract."
    return validated, result

