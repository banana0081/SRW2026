"""Outcome-independent diagnostic panels for RestBench.

The first contract pilot was intentionally assembled from queries that a
historical `Ours` run missed.  It is useful for debugging those trajectories,
but it cannot estimate an improvement: re-running the same stochastic model
regresses toward the mean on an outcome-selected panel.

This module selects a second panel from query structure only.  It reads gold
paths and candidate schemas to retain executable 2/3-hop dataflow chains, then
uses a deterministic, stratum-balanced hash order.  It never reads traces,
scores, reports, or model outputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Mapping, Sequence

from tooldoc_nir.restbench_contract_variants import Endpoint
from tooldoc_nir.restbench_data import candidate_docs, gold_apis


SELECTION_VERSION = "structural-heldout-v1"
DEFAULT_SALT = "cross-model-contracts-2026-09-09"


def port_alternatives(port: str) -> frozenset[str]:
    """Expand ``movie_ref|tv_ref:int`` into its typed alternatives."""
    stem, separator, kind = port.rpartition(":")
    if not separator:
        return frozenset({port})
    return frozenset(f"{name}:{kind}" for name in stem.split("|"))


def output_alternatives(endpoint: Endpoint) -> frozenset[str]:
    ports: set[str] = set()
    for output in endpoint.outputs:
        ports.update(port_alternatives(output.name))
    return frozenset(ports)


def reference_input_groups(endpoint: Endpoint) -> tuple[frozenset[str], ...]:
    """Return only identifiers that must be supplied by an earlier API.

    Text, season number and episode number can come directly from the user.
    Resource references cannot, so those are the ports that make a sequence a
    genuine dataflow chain.
    """
    groups: list[frozenset[str]] = []
    for line in endpoint.inputs:
        if "<-" not in line:
            continue
        alternatives = port_alternatives(line.split("<-", 1)[1].strip())
        references = frozenset(port for port in alternatives if "_ref:" in port)
        if references:
            groups.append(references)
    return tuple(groups)


def is_dataflow_chain(
    gold: Sequence[str],
    graph: Mapping[str, Endpoint],
) -> tuple[bool, str]:
    """Check that every later call consumes a reference produced earlier."""
    if len(gold) not in {2, 3}:
        return False, "not_2_or_3_hop"
    if len(set(gold)) != len(gold):
        return False, "repeated_api"
    if any(name not in graph for name in gold):
        return False, "api_missing_from_graph"

    first = graph[gold[0]]
    if reference_input_groups(first):
        return False, "first_call_needs_reference"
    available = set(output_alternatives(first))
    for name in gold[1:]:
        endpoint = graph[name]
        required = reference_input_groups(endpoint)
        if not required:
            return False, "later_call_has_no_dataflow_input"
        if any(not (set(group) & available) for group in required):
            return False, "unavailable_reference"
        available.update(output_alternatives(endpoint))
    return True, "eligible"


def structural_stratum(gold: Sequence[str], graph: Mapping[str, Endpoint]) -> str:
    """A compact difficulty/family stratum independent of model outcomes."""
    first = graph[gold[0]]
    last = graph[gold[-1]]
    source = (
        "search"
        if first.selector == "text"
        else "listing"
        if first.selector in {"filters", "trending", *(
            # Ranked-list selector names are represented directly.
            "popular",
            "top_rated",
            "now_playing",
            "upcoming",
            "on_the_air",
            "airing_today",
            "latest",
        )}
        else first.entity
    )
    return f"{len(gold)}hop:{source}:{last.entity}"


def _hash_key(salt: str, *parts: object) -> str:
    text = ":".join((salt, *(str(part) for part in parts)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_structural_panels(
    queries: Sequence[dict[str, Any]],
    instructions: Mapping[str, Mapping[str, Any]],
    graph: Mapping[str, Endpoint],
    *,
    sizes: Mapping[str, int],
    excluded_indices: Sequence[int] = (),
    salt: str = DEFAULT_SALT,
) -> dict[str, dict[str, Any]]:
    """Cut the remaining eligible queries into disjoint named panels at once.

    Screening a candidate and confirming it have to happen on different
    queries, and the split has to exist before the candidates are compiled --
    otherwise a promotion is just the screen panel measured twice. Freezing
    both halves in one call is what makes that auditable: the second panel is
    selected from what the first left, in the same deterministic hash order,
    and neither can be re-cut afterwards without changing the digest.
    """
    panels: dict[str, dict[str, Any]] = {}
    taken = list(excluded_indices)
    for name, size in sizes.items():
        selection = select_structural_panel(
            queries,
            instructions,
            graph,
            size=size,
            excluded_indices=taken,
            salt=salt,
        )
        panels[name] = selection
        taken = taken + list(selection["indices"])
    chosen = [index for panel in panels.values() for index in panel["indices"]]
    if len(chosen) != len(set(chosen)):
        raise AssertionError("the panels of one split must be disjoint")
    return panels


def select_structural_panel(
    queries: Sequence[dict[str, Any]],
    instructions: Mapping[str, Mapping[str, Any]],
    graph: Mapping[str, Endpoint],
    *,
    size: int,
    excluded_indices: Sequence[int] = (),
    salt: str = DEFAULT_SALT,
) -> dict[str, Any]:
    """Select a deterministic, stratum-balanced panel without outcomes."""
    if size <= 0:
        raise ValueError("panel size must be positive")
    excluded = set(excluded_indices)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()

    for index, query in enumerate(queries):
        if index in excluded:
            rejected["discovery_panel"] += 1
            continue
        gold = gold_apis(query)
        candidates = {
            str(document.get("tool_name") or "")
            for document in candidate_docs(query, dict(instructions))
        }
        if any(name not in candidates for name in gold):
            rejected["gold_missing_from_candidates"] += 1
            continue
        eligible, reason = is_dataflow_chain(gold, graph)
        if not eligible:
            rejected[reason] += 1
            continue
        stratum = structural_stratum(gold, graph)
        groups[stratum].append(
            {"query_index": index, "gold": gold, "stratum": stratum}
        )

    eligible_count = sum(len(rows) for rows in groups.values())
    if size > eligible_count:
        raise ValueError(
            f"requested {size} queries, but only {eligible_count} are eligible"
        )

    for stratum, rows in groups.items():
        rows.sort(
            key=lambda row: _hash_key(
                salt, stratum, row["query_index"], "|".join(row["gold"])
            )
        )
    stratum_order = sorted(groups, key=lambda name: _hash_key(salt, name))

    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < size:
        added = False
        for stratum in stratum_order:
            rows = groups[stratum]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == size:
                    break
        if not added:
            raise AssertionError("eligible panel exhausted before requested size")
        offset += 1

    # Execution order is separately randomized by the pilot.  A sorted panel
    # makes the immutable selection artifact easy to diff and audit.
    selected.sort(key=lambda row: int(row["query_index"]))
    return {
        "selection_version": SELECTION_VERSION,
        "salt": salt,
        "requested_size": size,
        "eligible_count": eligible_count,
        "excluded_indices": sorted(excluded),
        "indices": [row["query_index"] for row in selected],
        "rows": selected,
        "strata_selected": dict(
            sorted(Counter(row["stratum"] for row in selected).items())
        ),
        "strata_eligible": {
            name: len(rows) for name, rows in sorted(groups.items())
        },
        "rejected": dict(sorted(rejected.items())),
        "source_policy": (
            "query gold paths, candidate membership, and Initial API schemas "
            "only; no traces, scores, reports, or model outputs"
        ),
    }
