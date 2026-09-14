"""Offline tests for the preregistered contract-pilot gate."""

from __future__ import annotations

from scripts.run_contract_pilot import gate, paired_net


def _model() -> dict:
    arms = {
        "DRAFT": {
            "n": 48,
            "cp": 44,
            "errors": 0,
            "missing_final_consumer": 2,
        },
        "CurrentNot": {
            "n": 48,
            "cp": 44,
            "errors": 0,
            "missing_final_consumer": 2,
        },
        "H1": {
            "n": 48,
            "cp": 46,
            "errors": 0,
            "missing_final_consumer": 1,
        },
        "H2": {
            "n": 48,
            "cp": 45,
            "errors": 0,
            "missing_final_consumer": 1,
        },
        "H12": {
            "n": 48,
            "cp": 46,
            "errors": 0,
            "missing_final_consumer": 1,
        },
        "H123": {
            "n": 48,
            "cp": 48,
            "errors": 0,
            "missing_final_consumer": 0,
        },
    }
    return {
        "pooled": arms,
        "paired_vs_currentnot": {
            name: {"baseline": "CurrentNot", "net": 1}
            for name in ("H1", "H2", "H12", "H123")
        },
        "paired_vs_draft": {
            "H1": {"baseline": "DRAFT", "net": 2},
            "H2": {"baseline": "DRAFT", "net": 0},
            "H12": {"baseline": "DRAFT", "net": 2},
            "H123": {"baseline": "DRAFT", "net": 4},
        },
    }


def test_paired_net_keeps_both_seeds_of_the_same_query() -> None:
    rows = [
        {
            "row": "H1",
            "query_index": 1,
            "seed": 11,
            "correct_path": True,
            "error": "",
        },
        {
            "row": "DRAFT",
            "query_index": 1,
            "seed": 11,
            "correct_path": False,
            "error": "",
        },
        {
            "row": "H1",
            "query_index": 1,
            "seed": 12,
            "correct_path": False,
            "error": "",
        },
        {
            "row": "DRAFT",
            "query_index": 1,
            "seed": 12,
            "correct_path": True,
            "error": "",
        },
    ]
    result = paired_net(rows, "H1", "DRAFT")
    assert result["n"] == 2
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["net"] == 0


def test_protocol_gate_excludes_baselines_and_requires_strict_draft_win() -> None:
    report = {"ling": _model(), "flash": _model()}
    verdicts = gate(
        report,
        ("DRAFT", "CurrentNot", "H1", "H2", "H12", "H123"),
        expected_per_model=48,
        eligible_arms=("H1", "H2", "H12"),
        strict_positive_vs_draft=True,
        composition=("H12", ("H1", "H2")),
    )
    assert verdicts["H1"]["promoted"] is True
    assert verdicts["H12"]["promoted"] is True
    assert verdicts["H2"]["promoted"] is False
    assert any("not positive" in reason for reason in verdicts["H2"]["reasons"])
    assert verdicts["DRAFT"]["promoted"] is False
    assert verdicts["CurrentNot"]["promoted"] is False
    assert verdicts["H123"]["promoted"] is False


def test_protocol_gate_rejects_errors_and_failed_composition() -> None:
    report = {"ling": _model(), "flash": _model()}
    report["ling"]["pooled"]["H1"]["errors"] = 1
    report["flash"]["pooled"]["H12"]["cp"] = 43
    verdicts = gate(
        report,
        ("H1", "H2", "H12"),
        expected_per_model=48,
        eligible_arms=("H1", "H2", "H12"),
        strict_positive_vs_draft=True,
        composition=("H12", ("H1", "H2")),
    )
    assert verdicts["H1"]["promoted"] is False
    assert any("indeterminate" in reason for reason in verdicts["H1"]["reasons"])
    assert verdicts["H12"]["promoted"] is False
    assert any("component best" in reason for reason in verdicts["H12"]["reasons"])
