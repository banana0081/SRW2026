from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .models import CanonicalProfile


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ObservationSource(StrEnum):
    REAL = "real"
    STABLE_CACHE = "stable_cache"
    STABLE_SIMULATOR = "stable_simulator"
    CONTROLLED = "controlled"


class LivenessState(StrEnum):
    HEALTHY = "healthy"
    REACHABLE = "reachable"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    AUTH_BLOCKED = "auth_blocked"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


class ExecutionObservation(BaseModel):
    observation_id: str = ""
    api_key: tuple[str, str]
    source: ObservationSource
    observed_at: datetime = Field(default_factory=utc_now)
    request_method: str
    request_arguments: dict[str, Any] = Field(default_factory=dict)
    status_code: int | None = None
    latency_ms: float | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_body: Any = None
    error_type: str = ""
    error_message: str = ""

    @model_validator(mode="after")
    def assign_observation_id(self) -> "ExecutionObservation":
        if self.observation_id:
            return self
        payload = json.dumps(
            {
                "api_key": self.api_key,
                "source": self.source,
                "observed_at": self.observed_at.isoformat(),
                "request_method": self.request_method,
                "request_arguments": self.request_arguments,
                "status_code": self.status_code,
                "response_body": self.response_body,
                "error_type": self.error_type,
                "error_message": self.error_message,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        self.observation_id = sha256(payload.encode("utf-8")).hexdigest()[:20]
        return self

    @property
    def establishes_real_liveness(self) -> bool:
        return self.source is ObservationSource.REAL


class LivenessAssessment(BaseModel):
    api_key: tuple[str, str]
    state: LivenessState
    confidence: float = Field(ge=0.0, le=1.0)
    assessed_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    real_world_claim: bool = False


class ContractField(BaseModel):
    path: str
    type: str = "unknown"
    required: bool | None = None
    enum: list[Any] = Field(default_factory=list)
    observed_count: int = 0
    successful_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)


class ContractSnapshot(BaseModel):
    api_key: tuple[str, str]
    method: str = ""
    request_fields: list[ContractField] = Field(default_factory=list)
    response_fields: list[ContractField] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    source_hash: str = ""
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def assign_source_hash(self) -> "ContractSnapshot":
        if self.source_hash:
            return self
        payload = json.dumps(
            {
                "api_key": self.api_key,
                "method": self.method,
                "request_fields": [
                    field.model_dump(mode="json") for field in self.request_fields
                ],
                "response_fields": [
                    field.model_dump(mode="json") for field in self.response_fields
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        self.source_hash = sha256(payload.encode("utf-8")).hexdigest()
        return self


class ChangeKind(StrEnum):
    METHOD = "method"
    REQUEST_FIELD = "request_field"
    RESPONSE_FIELD = "response_field"
    LIVENESS = "liveness"


class ChangeOperation(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


class ContractChange(BaseModel):
    kind: ChangeKind
    operation: ChangeOperation
    path: str
    old_value: Any = None
    new_value: Any = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    auto_applicable: bool = False
    rationale: str = ""


class ContractDelta(BaseModel):
    api_key: tuple[str, str]
    baseline_hash: str
    observed_hash: str
    changes: list[ContractChange] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def confirmed_changes(self) -> list[ContractChange]:
        return [change for change in self.changes if change.auto_applicable]


class DocumentationVersion(BaseModel):
    revision: int
    profile: CanonicalProfile
    liveness: LivenessAssessment | None = None
    observed_contract: ContractSnapshot | None = None
    applied_delta: ContractDelta | None = None
    created_at: datetime = Field(default_factory=utc_now)
    provenance_observation_ids: list[str] = Field(default_factory=list)

