"""OpenRouter price/speed tradeoff used when a provider is not pinned.

A 2x throughput gain may cost at most 1.3x. That indifference curve is
`score = throughput / price ** (log 2 / log 1.3)`. Account-level Throughput
or Price sorts are overridden by sending an explicit `provider` object.
"""

from __future__ import annotations

from math import log
from typing import Any, Iterable, Mapping, Sequence


PRICE_RATIO_FOR_2X_SPEED = 1.3
PRICE_SPEED_ALPHA = log(2.0) / log(PRICE_RATIO_FOR_2X_SPEED)
PROMPT_WEIGHT = 0.45
COMPLETION_WEIGHT = 0.55
THROUGHPUT_PERCENTILE = "p50"
ROUTE_BALANCED = "balanced"
ROUTE_PRICE = "price"
ROUTE_THROUGHPUT = "throughput"
DEFAULT_ROUTE = ROUTE_BALANCED


def blended_unit_price(pricing: Mapping[str, Any] | None) -> float:
    row = pricing or {}
    try:
        prompt = float(row.get("prompt") or 0.0)
    except (TypeError, ValueError):
        prompt = 0.0
    try:
        completion = float(row.get("completion") or 0.0)
    except (TypeError, ValueError):
        completion = 0.0
    return PROMPT_WEIGHT * prompt + COMPLETION_WEIGHT * completion


def throughput_p50(endpoint: Mapping[str, Any]) -> float:
    block = endpoint.get("throughput_last_30m") or {}
    try:
        return float(block.get(THROUGHPUT_PERCENTILE) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def route_score(throughput: float, price: float) -> float:
    if throughput <= 0.0 or price <= 0.0:
        return 0.0
    return throughput / (price**PRICE_SPEED_ALPHA)


def max_price_ratio(speed_ratio: float) -> float:
    """Largest price multiple justified by this speed multiple."""
    if speed_ratio <= 1.0:
        return 1.0
    return PRICE_RATIO_FOR_2X_SPEED ** (log(speed_ratio) / log(2.0))


def worth_paying(price_ratio: float, speed_ratio: float) -> bool:
    return price_ratio <= max_price_ratio(speed_ratio) + 1e-12


def endpoint_tag(endpoint: Mapping[str, Any]) -> str:
    tag = str(endpoint.get("tag") or "").strip()
    if tag:
        return tag
    name = str(endpoint.get("provider_name") or "").strip()
    return "".join(character for character in name.lower() if character.isalnum())


def rank_provider_tags(endpoints: Iterable[Mapping[str, Any]]) -> list[str]:
    """Unique tags, best combined score first. Keep the better endpoint per tag."""
    best: dict[str, tuple[float, float]] = {}
    for endpoint in endpoints:
        tag = endpoint_tag(endpoint)
        if not tag:
            continue
        try:
            status = int(endpoint.get("status") or 0)
        except (TypeError, ValueError):
            status = 0
        if status != 0:
            continue
        price = blended_unit_price(endpoint.get("pricing"))
        speed = throughput_p50(endpoint)
        score = route_score(speed, price)
        if score <= 0.0:
            continue
        current = best.get(tag)
        if current is None or score > current[0]:
            best[tag] = (score, price)
    return [
        tag
        for tag, _metrics in sorted(
            best.items(), key=lambda item: (-item[1][0], item[1][1], item[0])
        )
    ]


def provider_preferences(
    *,
    route: str,
    pin: str | None,
    ranked: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    if pin:
        return {"order": [pin], "allow_fallbacks": False}
    mode = (route or DEFAULT_ROUTE).strip().lower()
    if mode == ROUTE_PRICE:
        return {"sort": "price"}
    if mode == ROUTE_THROUGHPUT:
        return {"sort": "throughput"}
    if ranked:
        return {"order": list(ranked), "allow_fallbacks": True}
    return {"sort": "price"}
