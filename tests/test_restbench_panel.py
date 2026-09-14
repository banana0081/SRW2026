"""Outcome-independence and dataflow checks for the structural holdout."""

from __future__ import annotations

import copy
from pathlib import Path

from tooldoc_nir.restbench_contract_variants import build_graph
from tooldoc_nir.restbench_data import load_instructions, load_queries
from tooldoc_nir.restbench_panel import (
    is_dataflow_chain,
    port_alternatives,
    select_structural_panel,
)


DRAFT_ROOT = Path("external/DRAFT")
DISCOVERY = (0, 5, 13, 17, 20, 33, 51, 57, 62, 66)


def _inputs() -> tuple[list[dict], dict, dict]:
    queries = load_queries(DRAFT_ROOT, "TMDB")
    instructions = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    return queries, instructions, build_graph(instructions)


def test_union_ports_expand_with_the_shared_type() -> None:
    assert port_alternatives("movie_ref|tv_ref|person_ref:int") == {
        "movie_ref:int",
        "tv_ref:int",
        "person_ref:int",
    }
    assert port_alternatives("review_ref:str") == {"review_ref:str"}


def test_chain_gate_keeps_dataflow_and_rejects_benchmark_artifacts() -> None:
    queries, _instructions, graph = _inputs()
    assert is_dataflow_chain(queries[5]["relevant APIs"], graph) == (
        True,
        "eligible",
    )
    # Starts from a detail endpoint without a producer.
    assert is_dataflow_chain(queries[30]["relevant APIs"], graph)[0] is False
    # Repeats a search rather than passing a typed reference.
    assert is_dataflow_chain(queries[78]["relevant APIs"], graph) == (
        False,
        "repeated_api",
    )
    # Two independent list/search calls are not a dataflow chain.
    assert is_dataflow_chain(queries[91]["relevant APIs"], graph)[0] is False
    assert is_dataflow_chain(queries[93]["relevant APIs"], graph) == (
        False,
        "unavailable_reference",
    )


def test_structural_panel_is_deterministic_and_outcome_independent() -> None:
    queries, instructions, graph = _inputs()
    first = select_structural_panel(
        queries,
        instructions,
        graph,
        size=24,
        excluded_indices=DISCOVERY,
    )
    second = select_structural_panel(
        copy.deepcopy(queries),
        instructions,
        graph,
        size=24,
        excluded_indices=DISCOVERY,
    )
    assert first == second
    assert len(first["indices"]) == len(set(first["indices"])) == 24
    assert set(first["indices"]).isdisjoint(DISCOVERY)
    assert len(first["strata_selected"]) >= 6
    assert "traces" in first["source_policy"]
    assert "no traces" in first["source_policy"]

    # Selection never reads the natural-language prompt or any outcome field.
    changed = copy.deepcopy(queries)
    for query in changed:
        query["query"] = "redacted"
        query["model_score"] = 1
    third = select_structural_panel(
        changed,
        instructions,
        graph,
        size=24,
        excluded_indices=DISCOVERY,
    )
    assert third == first
