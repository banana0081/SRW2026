from math import isclose

from tooldoc_nir.provider_route import (
    PRICE_RATIO_FOR_2X_SPEED,
    blended_unit_price,
    max_price_ratio,
    provider_preferences,
    rank_provider_tags,
    route_score,
    worth_paying,
)


def test_two_x_speed_pays_at_most_1_3x() -> None:
    assert isclose(max_price_ratio(2.0), PRICE_RATIO_FOR_2X_SPEED)
    assert worth_paying(1.3, 2.0)
    assert not worth_paying(1.31, 2.0)
    assert worth_paying(1.0, 1.0)
    assert not worth_paying(1.01, 1.0)
    assert worth_paying(PRICE_RATIO_FOR_2X_SPEED**2, 4.0)
    assert not worth_paying(PRICE_RATIO_FOR_2X_SPEED**2 + 0.01, 4.0)


def test_score_is_indifferent_on_the_2x_1_3x_curve() -> None:
    cheap = route_score(10.0, 1.0)
    faster = route_score(20.0, 1.3)
    assert isclose(cheap, faster, rel_tol=1e-9)
    assert route_score(20.0, 1.31) < cheap
    assert route_score(20.0, 1.29) > cheap


def test_rank_skips_baidu_style_price_spike() -> None:
    ranked = rank_provider_tags(
        [
            {
                "tag": "novita",
                "status": 0,
                "pricing": {"prompt": "1", "completion": "1"},
                "throughput_last_30m": {"p50": 15},
            },
            {
                "tag": "baidu",
                "status": 0,
                "pricing": {"prompt": "8", "completion": "8"},
                "throughput_last_30m": {"p50": 40},
            },
            {
                "tag": "deepinfra",
                "status": 0,
                "pricing": {"prompt": "1.2", "completion": "1.2"},
                "throughput_last_30m": {"p50": 28},
            },
        ]
    )
    assert ranked[0] == "deepinfra"
    assert ranked[-1] == "baidu"
    assert worth_paying(1.2, 28 / 15)
    assert not worth_paying(8.0, 40 / 15)


def test_blended_price_weights_completion_more() -> None:
    cheap_prompt = blended_unit_price({"prompt": "1", "completion": "10"})
    cheap_completion = blended_unit_price({"prompt": "10", "completion": "1"})
    assert cheap_completion < cheap_prompt


def test_provider_preferences_pin_and_fallbacks() -> None:
    assert provider_preferences(route="balanced", pin="novita") == {
        "order": ["novita"],
        "allow_fallbacks": False,
    }
    assert provider_preferences(route="price", pin=None) == {"sort": "price"}
    assert provider_preferences(
        route="balanced", pin=None, ranked=["novita", "deepinfra"]
    ) == {"order": ["novita", "deepinfra"], "allow_fallbacks": True}
    assert provider_preferences(route="balanced", pin=None, ranked=None) == {
        "sort": "price"
    }
