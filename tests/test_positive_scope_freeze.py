"""The frozen Cross-model Positive Scope split and protocol.

Two mistakes these guard against, both of which already happened once:

- screening a candidate and confirming it on the same queries, which cannot
  distinguish a real improvement from having looked twice;
- reporting a table from whichever `Ours` payload happened to appear in the
  most seed roots, instead of the one the protocol committed to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_final_tables import preregistered_digests, table_for
from scripts.freeze_positive_scope import (
    CANDIDATES,
    DISCOVERY_PANEL,
    SCREEN_ARMS,
    holdout_indices,
    stable_view,
)
from tooldoc_nir.provenance import payload_digest

DOCUMENTATION = Path("artifacts/documentation")
SCREEN_PANEL = DOCUMENTATION / "TMDB_screen_a_panel.json"
PROMOTION_PANEL = DOCUMENTATION / "TMDB_promotion_b_panel.json"
PROTOCOL = DOCUMENTATION / "TMDB_positive_scope_protocol.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_frozen_artifact_matches_its_own_digest() -> None:
    for path in (SCREEN_PANEL, PROMOTION_PANEL, PROTOCOL):
        recorded = _load(path)
        assert recorded["selection_digest"] == payload_digest(
            stable_view(recorded)
        ), path


def test_the_two_panels_are_disjoint_and_never_seen_before() -> None:
    screen = set(_load(SCREEN_PANEL)["selection"]["indices"])
    promotion = set(_load(PROMOTION_PANEL)["selection"]["indices"])
    spent = set(DISCOVERY_PANEL) | set(holdout_indices())

    assert len(screen) == 24
    assert len(promotion) == 30
    assert not screen & promotion
    assert not screen & spent
    assert not promotion & spent


def test_the_split_uses_up_the_remaining_eligible_queries() -> None:
    screen = _load(SCREEN_PANEL)["selection"]
    promotion = _load(PROMOTION_PANEL)["selection"]
    assert screen["eligible_count"] == 54
    assert promotion["eligible_count"] == 30
    assert len(screen["indices"]) + len(promotion["indices"]) == 54


def test_the_panels_read_no_outcome() -> None:
    for path in (SCREEN_PANEL, PROMOTION_PANEL):
        policy = _load(path)["selection"]["source_policy"]
        assert "no traces, scores, reports, or model outputs" in policy


def test_the_protocol_pins_a_provider_for_both_models() -> None:
    protocol = _load(PROTOCOL)
    assert set(protocol["providers"]) == set(protocol["models"])
    assert protocol["providers"]["ling"] == "novita"
    assert protocol["providers"]["flash"] == "wafer"
    assert "sailresearch" in protocol["excluded_providers"]
    assert "fails the cell" in protocol["provider_policy"]


def test_the_protocol_pins_the_payload_of_every_arm_it_will_run() -> None:
    protocol = _load(PROTOCOL)
    screen = protocol["stages"]["screen_a"]
    assert screen["arms"] == list(SCREEN_ARMS)
    assert set(screen["arm_payload_digests"]) == set(SCREEN_ARMS)
    for digest in screen["arm_payload_digests"].values():
        assert len(digest) == 64


def test_the_protocol_states_the_error_policy_and_the_stopping_rule() -> None:
    protocol = _load(PROTOCOL)
    assert "never re-rolled" in protocol["error_policy"]["model_result"]
    assert "dropped from every arm" in protocol["error_policy"]["transport"]
    assert "at most one candidate" in protocol["stages"]["screen_a"]["gate"]["promote"]
    assert "negative result" in protocol["stopping_rule"]


def test_the_protocol_points_each_panel_at_the_frozen_selection() -> None:
    protocol = _load(PROTOCOL)
    for stage, path in (
        ("screen_a", SCREEN_PANEL),
        ("promotion_b", PROMOTION_PANEL),
    ):
        recorded = protocol["stages"][stage]["panel"]
        assert recorded["selection_digest"] == _load(path)["selection_digest"]
        assert recorded["n"] == len(_load(path)["selection"]["indices"])


# --------------------------------------------------------------------------
# The reported table can only come from a preregistered payload
# --------------------------------------------------------------------------


def test_the_preregistered_digests_are_the_two_candidates() -> None:
    allowed = preregistered_digests(PROTOCOL)
    assert set(allowed) == set(CANDIDATES)
    protocol = _load(PROTOCOL)
    screen = protocol["stages"]["screen_a"]["arm_payload_digests"]
    assert allowed == {arm: screen[arm] for arm in CANDIDATES}


def _cell(seed: int, digest: str) -> dict:
    return {
        "root": f"root_s{seed:02d}",
        "ours_digest": digest,
        "provider": "wafer",
        "created_at": None,
        "rows": [
            {"row": row, "query_index": index, "correct_path": row == "Ours",
             "error": "", "http_errors": 0, "gold": [], "executed": []}
            for index in range(4)
            for row in ("DFSDT", "DRAFT", "Ours")
        ],
    }


def test_a_root_from_an_unregistered_payload_is_not_reported() -> None:
    allowed = {"P": "a" * 64, "PF": "b" * 64}
    table = table_for({1: _cell(1, "c" * 64)}, allowed_digests=allowed)
    assert table["seeds_included"] == []
    assert "no seed root was produced" in table["error"]


def test_two_candidates_in_one_table_is_an_error_not_a_majority_vote() -> None:
    allowed = {"P": "a" * 64, "PF": "b" * 64}
    table = table_for(
        {1: _cell(1, "a" * 64), 2: _cell(2, "b" * 64), 3: _cell(3, "b" * 64)},
        allowed_digests=allowed,
    )
    assert table["seeds_included"] == []
    assert "mix two preregistered candidates" in table["error"]


def test_the_registered_candidate_is_reported_and_named() -> None:
    allowed = {"P": "a" * 64, "PF": "b" * 64}
    table = table_for(
        {1: _cell(1, "a" * 64), 2: _cell(2, "a" * 64)}, allowed_digests=allowed
    )
    assert table["seeds_included"] == [1, 2]
    assert table["digest_source"] == "preregistered protocol, candidate P"
    assert table["providers"] == ["wafer"]


def test_without_a_protocol_the_majority_rule_still_reports_the_old_roots() -> None:
    table = table_for({1: _cell(1, "a" * 64), 2: _cell(2, "a" * 64)})
    assert table["seeds_included"] == [1, 2]
    assert table["digest_source"].startswith("majority of seed roots")


def test_freezing_a_different_selection_over_a_frozen_one_is_refused(
    tmp_path: Path,
) -> None:
    from scripts.freeze_positive_scope import _write_once

    frozen = _write_once(tmp_path / "panel.json", {"selection": [1, 2]}, "panel")
    assert frozen["selection_digest"]
    _write_once(tmp_path / "panel.json", {"selection": [1, 2]}, "panel")
    with pytest.raises(SystemExit, match="already frozen with a different"):
        _write_once(tmp_path / "panel.json", {"selection": [3, 4]}, "panel")
