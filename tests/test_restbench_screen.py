"""Offline tests for the two-stage positive-scope statistics and gates."""

from __future__ import annotations

import math

from tooldoc_nir.restbench_screen import (
    call_fill_rejected,
    execution_valid_hit,
    fill_error_count,
    paired_effect,
    paper_gate,
    per_seed_effects,
    promotion_verdict,
    query_differences,
    scored_rows,
    screen_gate,
    sign_flip_p_value,
    student_t_interval,
)


def _row(arm: str, seed: int, index: int, ok: bool, **extra: object) -> dict:
    return {
        "row": arm,
        "seed": seed,
        "query_index": index,
        "correct_path": ok,
        "error": "",
        "http_errors": 0,
        **extra,
    }


def _cell(
    arm: str,
    seeds: tuple[int, ...],
    hits: dict[int, bool],
) -> list[dict]:
    return [
        _row(arm, seed, index, ok)
        for seed in seeds
        for index, ok in hits.items()
    ]


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


def test_a_query_under_two_seeds_is_two_paired_cells() -> None:
    rows = [
        _row("P", 21, 3, True),
        _row("DRAFT", 21, 3, False),
        _row("P", 22, 3, False),
        _row("DRAFT", 22, 3, True),
    ]
    effect = paired_effect(rows, "P", "DRAFT")
    assert effect["n"] == 2
    assert (effect["wins"], effect["losses"], effect["net"]) == (1, 1, 0)
    assert effect["effect_pp"] == 0.0


def test_a_driver_failure_is_a_miss_and_an_invalidated_query_is_not_scored() -> None:
    rows = scored_rows(
        [
            _row("P", 21, 1, False, error="ReleasedDriverFailure: bad shape",
                 error_kind="driver"),
            _row("DRAFT", 21, 1, True),
            _row("P", 21, 2, True, invalidated=True),
            _row("DRAFT", 21, 2, False, invalidated=True),
        ]
    )
    effect = paired_effect(rows, "P", "DRAFT")
    assert effect["n"] == 1
    assert effect["losses"] == 1


def test_execution_valid_cp_drops_a_path_reached_through_a_broken_call() -> None:
    assert execution_valid_hit(_row("P", 21, 1, True))
    assert not execution_valid_hit(_row("P", 21, 1, True, http_errors=2))
    assert not execution_valid_hit(_row("P", 21, 1, False))


def test_execution_valid_cp_passes_when_http_failed_after_a_correct_fill() -> None:
    assert execution_valid_hit(_row("P", 21, 1, True, http_errors=2, fill_errors=0))
    assert not execution_valid_hit(_row("P", 21, 1, True, http_errors=2, fill_errors=1))
    assert execution_valid_hit(_row("P", 21, 1, True, http_errors=2, id_errors=0))
    assert not execution_valid_hit(_row("P", 21, 1, True, fill_errors=0, id_errors=1))


def test_a_missing_path_id_is_a_fill_reject_and_a_device_error_is_not() -> None:
    missing = {"error": "missing path parameter id", "response": ""}
    device = {
        "error": "HTTP 404",
        "response": '{"error":{"reason":"NO_ACTIVE_DEVICE"}}',
    }
    forbidden = {"error": "HTTP 403", "response": '{"error":{"message":"Forbidden"}}'}
    quota = {"error": "HTTP 429", "response": '{"error":{"reason":"QUOTA_EXCEEDED"}}'}
    empty_player = {"error": "HTTP 404", "response": ""}
    invented = {
        "error": "HTTP 400",
        "response": '{"error":{"message":"Invalid base62 id"}}',
    }
    api = {"api_name": "PUT_me_player_play", "parameters": {"uri": "spotify:track:abc"}}
    assert call_fill_rejected({"parameters": {}}, missing)
    assert not call_fill_rejected(api, device)
    assert not call_fill_rejected(api, forbidden)
    assert not call_fill_rejected(api, quota)
    assert not call_fill_rejected(api, empty_player)
    assert call_fill_rejected({"parameters": {"id": "not-an-id"}}, invented)


def test_an_emitted_query_parameter_is_not_a_fill_reject_when_http_says_missing() -> None:
    volume = {
        "error": "HTTP 400",
        "response": '{"error":{"message":"Required parameter volume_percent missing"}}',
    }
    filled = {"api_name": "PUT_me_player_volume", "parameters": {"volume_percent": 60}}
    omitted = {"api_name": "PUT_me_player_volume", "parameters": {}}
    assert not call_fill_rejected(filled, volume)
    assert call_fill_rejected(omitted, volume)


def test_fill_error_count_only_scores_gold_path_calls() -> None:
    record = {
        "execute_log": {
            "api_result_ls": [
                [{"api_name": "POST_me_player_next", "parameters": {}}],
                [{"api_name": "GET_recommendations", "parameters": {}}],
            ],
            "call_result_ls": [
                "{'error': 'HTTP 404', 'response': '{\"error\":{\"reason\":\"NO_ACTIVE_DEVICE\"}}'}",
                "{'error': 'missing path parameter id', 'response': ''}",
            ],
        }
    }
    assert fill_error_count(record, ["POST_me_player_next"]) == 0
    assert fill_error_count(record, ["POST_me_player_next", "GET_recommendations"]) == 1


def test_per_seed_effects_split_the_replicates() -> None:
    rows = [
        _row("P", 21, 1, True),
        _row("DRAFT", 21, 1, False),
        _row("P", 22, 1, False),
        _row("DRAFT", 22, 1, True),
    ]
    seeds = per_seed_effects(rows, "P", "DRAFT")
    assert seeds[21]["effect_pp"] == 100.0
    assert seeds[22]["effect_pp"] == -100.0


def test_query_differences_average_replicates_before_pairing() -> None:
    """Three seeds of one query are one cluster, not three observations."""
    rows = [
        _row("P", 23, 4, True),
        _row("P", 24, 4, True),
        _row("P", 25, 4, False),
        _row("DRAFT", 23, 4, False),
        _row("DRAFT", 24, 4, False),
        _row("DRAFT", 25, 4, False),
    ]
    differences = query_differences(rows, "P", "DRAFT")
    assert list(differences) == [4]
    assert math.isclose(differences[4], 2.0 / 3.0)


# --------------------------------------------------------------------------
# Intervals and the permutation test
# --------------------------------------------------------------------------


def test_the_seed_interval_uses_the_t_quantile_not_1_96() -> None:
    values = [0.02, 0.04, 0.01, 0.05]
    interval = student_t_interval(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    normal_half = 1.96 * math.sqrt(variance / len(values))
    t_half = mean - interval["ci95"][0]
    assert interval["t_quantile"] > 3.0
    assert t_half > normal_half
    assert interval["replicates"] == 4


def test_one_replicate_reports_no_interval() -> None:
    assert student_t_interval([0.1]) == {
        "replicates": 1,
        "mean": 0.1,
        "values": [0.1],
    }


def test_sign_flip_is_exact_while_it_is_cheap() -> None:
    result = sign_flip_p_value([1.0, 1.0, 1.0])
    assert result["method"] == "exact over 2^3"
    assert result["p_value"] == 0.125


def test_sign_flip_ignores_ties_and_never_reports_zero() -> None:
    degenerate = sign_flip_p_value([0.0, 0.0])
    assert degenerate["p_value"] == 1.0
    assert degenerate["method"] == "degenerate"
    large = sign_flip_p_value([1.0] * 40)
    assert large["method"].startswith("monte carlo")
    assert large["p_value"] > 0.0


def test_a_symmetric_difference_set_is_not_significant() -> None:
    assert sign_flip_p_value([1.0, -1.0, 1.0, -1.0])["p_value"] > 0.05


# --------------------------------------------------------------------------
# Screen A
# --------------------------------------------------------------------------


SEEDS = (21, 22)
PANEL = tuple(range(24))


def _panel_rows(arm: str, wins: int, base: int = 18) -> list[dict]:
    """`base` hits for the baselines, `base + wins` for the candidate."""
    hits = {index: index < base + wins for index in PANEL}
    return _cell(arm, SEEDS, hits)


def _screen_input(candidate_wins: int) -> dict[str, list[dict]]:
    rows = (
        _panel_rows("DRAFT", 0)
        + _panel_rows("Initial", 0)
        + _panel_rows("CurrentNot", 0)
        + _panel_rows("P", candidate_wins)
    )
    return {"ling": list(rows), "flash": list(rows)}


def test_screen_promotes_the_candidate_that_clears_every_contrast() -> None:
    gate = screen_gate(
        _screen_input(2),
        ("P",),
        expected_cells=len(PANEL) * len(SEEDS),
        surfaces={"P": 1},
    )
    verdict = gate["verdicts"]["P"]
    assert verdict["reasons"] == []
    assert verdict["screen_score_pp"] > 2.0
    assert sorted(verdict["contrasts"]) == [
        "flash:DRAFT",
        "flash:Initial",
        "ling:DRAFT",
        "ling:Initial",
    ]
    assert gate["promoted"] == "P"


def test_screen_scores_the_worst_contrast_so_one_model_cannot_pay_for_the_other() -> None:
    strong = _panel_rows("P", 6)
    weak = _panel_rows("P", 0)
    rows = _panel_rows("DRAFT", 0) + _panel_rows("Initial", 0) + _panel_rows(
        "CurrentNot", 0
    )
    gate = screen_gate(
        {"ling": rows + strong, "flash": rows + weak},
        ("P",),
        expected_cells=len(PANEL) * len(SEEDS),
    )
    verdict = gate["verdicts"]["P"]
    assert verdict["screen_score_pp"] == 0.0
    assert gate["promoted"] is None
    assert any("screen score" in reason for reason in verdict["reasons"])


def test_screen_rejects_a_candidate_that_loses_to_the_incumbent() -> None:
    rows = (
        _panel_rows("DRAFT", 0)
        + _panel_rows("Initial", 0)
        + _panel_rows("CurrentNot", 4)
        + _panel_rows("P", 2)
    )
    gate = screen_gate(
        {"ling": list(rows), "flash": list(rows)},
        ("P",),
        expected_cells=len(PANEL) * len(SEEDS),
    )
    verdict = gate["verdicts"]["P"]
    assert gate["promoted"] is None
    assert any("against CurrentNot" in reason for reason in verdict["reasons"])


def test_screen_rejects_a_candidate_one_model_never_ran() -> None:
    rows = _screen_input(4)
    gate = screen_gate(
        {"ling": rows["ling"]},
        ("P",),
        expected_cells=len(PANEL) * len(SEEDS),
    )
    assert gate["promoted"] is None
    assert any(
        "no paired data" in reason for reason in gate["verdicts"]["P"]["reasons"]
    )


def test_screen_rejects_a_single_bad_replicate_even_when_the_pool_is_positive() -> None:
    baselines = _panel_rows("DRAFT", 0) + _panel_rows("Initial", 0) + _panel_rows(
        "CurrentNot", 0
    )
    # Seed 21 wins eight queries, seed 22 loses two: pooled is well positive.
    candidate = _cell("P", (21,), {i: i < 26 for i in PANEL}) + _cell(
        "P", (22,), {i: i < 16 for i in PANEL}
    )
    rows = baselines + candidate
    gate = screen_gate(
        {"ling": list(rows), "flash": list(rows)},
        ("P",),
        expected_cells=len(PANEL) * len(SEEDS),
    )
    verdict = gate["verdicts"]["P"]
    assert verdict["screen_score_pp"] > 2.0
    assert gate["promoted"] is None
    assert any("seed 22" in reason for reason in verdict["reasons"])


def test_screen_promotes_at_most_one_and_breaks_ties_on_fewer_surfaces() -> None:
    rows = (
        _panel_rows("DRAFT", 0)
        + _panel_rows("Initial", 0)
        + _panel_rows("CurrentNot", 0)
        + _panel_rows("P", 2)
        + _panel_rows("PF", 2)
    )
    gate = screen_gate(
        {"ling": list(rows), "flash": list(rows)},
        ("P", "PF"),
        expected_cells=len(PANEL) * len(SEEDS),
        surfaces={"P": 1, "PF": 2},
    )
    assert gate["eligible"] == ["P", "PF"]
    assert gate["promoted"] == "P"


# --------------------------------------------------------------------------
# Promotion B and the paper gate
# --------------------------------------------------------------------------


PROMOTION_SEEDS = (23, 24, 25)
PROMOTION_PANEL = tuple(range(30))


def _promotion_rows(arm: str, hits: int) -> list[dict]:
    return _cell(arm, PROMOTION_SEEDS, {i: i < hits for i in PROMOTION_PANEL})


def test_promotion_confirms_a_consistent_win_on_untouched_queries() -> None:
    rows = (
        _promotion_rows("DRAFT", 20)
        + _promotion_rows("DFSDT", 20)
        + _promotion_rows("Initial", 20)
        + _promotion_rows("P", 26)
    )
    verdict = promotion_verdict(
        {"ling": list(rows), "flash": list(rows)},
        "P",
        expected_queries=len(PROMOTION_PANEL),
    )
    assert verdict["reasons"] == []
    assert verdict["confirmed"] is True
    assert verdict["contrasts"]["ling:DRAFT"]["permutation"]["p_value"] <= 0.05
    assert verdict["contrasts"]["ling:DRAFT"]["positive_seeds"] == 3


def test_promotion_rejects_a_single_query_win_on_the_permutation() -> None:
    """One recovered query out of thirty is 3.3 pp and still p = 0.5."""
    rows = (
        _promotion_rows("DRAFT", 20)
        + _promotion_rows("Initial", 20)
        + _promotion_rows("P", 21)
    )
    verdict = promotion_verdict(
        {"ling": list(rows), "flash": list(rows)},
        "P",
        expected_queries=len(PROMOTION_PANEL),
    )
    assert verdict["contrasts"]["ling:DRAFT"]["effect_pp"] == 3.33
    assert verdict["confirmed"] is False
    assert any("one-sided p=0.5" in reason for reason in verdict["reasons"])


def test_promotion_rejects_a_candidate_that_only_trades_queries() -> None:
    candidate = _cell(
        "P",
        PROMOTION_SEEDS,
        {index: 0 < index <= 20 for index in PROMOTION_PANEL},
    )
    rows = _promotion_rows("DRAFT", 20) + _promotion_rows("Initial", 20) + candidate
    verdict = promotion_verdict(
        {"ling": list(rows), "flash": list(rows)},
        "P",
        expected_queries=len(PROMOTION_PANEL),
    )
    assert verdict["contrasts"]["ling:DRAFT"]["effect_pp"] == 0.0
    assert verdict["confirmed"] is False
    assert any("0.0 pp is below +3.0" in reason for reason in verdict["reasons"])


def test_the_paper_gate_needs_all_four_intervals_above_zero() -> None:
    strong = [0.05, 0.06, 0.04, 0.05]
    weak = [0.05, -0.06, 0.04, -0.05]
    met = paper_gate(
        {
            "ling:DRAFT": strong,
            "ling:Initial": strong,
            "flash:DRAFT": strong,
            "flash:Initial": strong,
        }
    )
    assert met["met"] is True
    assert met["failing"] == []
    missed = paper_gate(
        {
            "ling:DRAFT": strong,
            "ling:Initial": strong,
            "flash:DRAFT": strong,
            "flash:Initial": weak,
        }
    )
    assert missed["met"] is False
    assert missed["failing"] == ["flash:Initial"]


def test_the_paper_gate_is_not_met_by_a_missing_contrast() -> None:
    assert paper_gate({})["met"] is False
