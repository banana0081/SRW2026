from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .dynamic_models import (
    ExecutionObservation,
    LivenessAssessment,
    LivenessState,
    ObservationSource,
)


def classify_observation(observation: ExecutionObservation) -> LivenessState:
    """Classify reachability without confusing validation/auth errors with death."""
    status = observation.status_code
    if status is None:
        return LivenessState.UNAVAILABLE
    if 200 <= status < 400:
        return LivenessState.HEALTHY
    if status in {401, 403}:
        return LivenessState.AUTH_BLOCKED
    if status == 429:
        return LivenessState.RATE_LIMITED
    if status in {404, 410}:
        return LivenessState.UNAVAILABLE
    if status in {405, 409, 415, 422}:
        return LivenessState.REACHABLE
    if status in {408, 425} or 500 <= status < 600:
        return LivenessState.DEGRADED
    if 400 <= status < 500:
        return LivenessState.REACHABLE
    return LivenessState.UNKNOWN


def assess_liveness(
    observations: Iterable[ExecutionObservation],
    *,
    source: ObservationSource,
    now: datetime | None = None,
    ttl: timedelta = timedelta(hours=6),
    min_failures_for_unavailable: int = 3,
    window_size: int = 5,
) -> LivenessAssessment:
    """Aggregate recent probes into a scoped, expiring liveness claim."""
    selected = sorted(
        (item for item in observations if item.source is source),
        key=lambda item: item.observed_at,
    )[-window_size:]
    assessed_at = now or datetime.now(timezone.utc)

    if not selected:
        return LivenessAssessment(
            api_key=("unknown", "unknown"),
            state=LivenessState.UNKNOWN,
            confidence=0.0,
            assessed_at=assessed_at,
            expires_at=assessed_at + ttl,
            reason=f"No observations from source={source.value}.",
            real_world_claim=source is ObservationSource.REAL,
        )

    states = [classify_observation(item) for item in selected]
    counts = Counter(states)
    consecutive_hard_failures = 0
    for item, state in reversed(list(zip(selected, states, strict=True))):
        if state is LivenessState.UNAVAILABLE or (
            state is LivenessState.DEGRADED
            and item.status_code is not None
            and item.status_code >= 500
        ):
            consecutive_hard_failures += 1
        else:
            break

    recent_recovery = (
        len(states) >= 2
        and states[-2:] == [LivenessState.HEALTHY, LivenessState.HEALTHY]
    )
    disruptive_states = {
        LivenessState.DEGRADED,
        LivenessState.UNAVAILABLE,
        LivenessState.AUTH_BLOCKED,
        LivenessState.RATE_LIMITED,
    }

    if recent_recovery:
        state = LivenessState.HEALTHY
        reason = "The API recovered with two consecutive successful probes."
    elif counts[LivenessState.HEALTHY] and any(
        counts[item] for item in disruptive_states
    ):
        state = LivenessState.DEGRADED
        reason = "Recent probes contain both successful and failed outcomes."
    elif counts[LivenessState.HEALTHY]:
        state = LivenessState.HEALTHY
        reason = "Recent probes completed successfully."
    elif counts[LivenessState.REACHABLE]:
        state = LivenessState.REACHABLE
        reason = (
            "The API endpoint responded but rejected the supplied call."
        )
    elif counts[LivenessState.AUTH_BLOCKED] >= 2:
        state = LivenessState.AUTH_BLOCKED
        reason = "Repeated 401/403 responses prevent authenticated validation."
    elif counts[LivenessState.RATE_LIMITED] >= 2:
        state = LivenessState.RATE_LIMITED
        reason = "Repeated HTTP 429 responses prevent availability validation."
    elif consecutive_hard_failures >= min_failures_for_unavailable:
        state = LivenessState.UNAVAILABLE
        reason = (
            f"{consecutive_hard_failures} consecutive transport/5xx failures."
        )
    elif counts[LivenessState.DEGRADED] or counts[LivenessState.UNAVAILABLE]:
        state = LivenessState.DEGRADED
        reason = "Failures observed, but evidence is insufficient for a down claim."
    else:
        state = LivenessState.UNKNOWN
        reason = "Observed outcomes cannot establish availability."

    if state is LivenessState.UNAVAILABLE:
        agreement = consecutive_hard_failures / len(selected)
    else:
        agreement = counts[state] / len(selected)
    evidence_factor = min(1.0, len(selected) / min_failures_for_unavailable)
    confidence = round(0.45 * agreement + 0.55 * evidence_factor, 3)

    return LivenessAssessment(
        api_key=selected[-1].api_key,
        state=state,
        confidence=confidence,
        assessed_at=assessed_at,
        expires_at=assessed_at + ttl,
        evidence_ids=[item.observation_id for item in selected],
        reason=reason,
        real_world_claim=source is ObservationSource.REAL,
    )

