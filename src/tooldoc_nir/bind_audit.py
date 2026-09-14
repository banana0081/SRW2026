"""How often the closed TMDB producer map names a gold-path predecessor.

The compiler does not read gold paths. This audit is a post-hoc overlap check:
for each gold hop, does PRODUCERS[path_param] contain an API that already
appears earlier on that gold path? A miss does not mean the compiler cheated;
it means the fill line names a search/list producer, not necessarily the gold
predecessor (collection.parts[i].id is the typical gap).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tooldoc_nir.restbench_ours import PRODUCERS, hop_params, _tool_index


def catalog_by_name(initial: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _tool_index(dict(initial))


def gold_hops(
    gold: Sequence[str],
    catalog: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(gold):
        document = catalog.get(name)
        if document is None:
            continue
        params = hop_params(str(document.get("url") or ""))
        if not params:
            continue
        prior = list(gold[:index])
        param_rows: list[dict[str, Any]] = []
        covered = True
        for param in params:
            producers = [p for p in PRODUCERS.get(param, []) if p in catalog]
            prior_hit = [p for p in producers if p in prior]
            param_rows.append(
                {
                    "param": param,
                    "producers": producers,
                    "prior_hit": prior_hit,
                }
            )
            if not prior_hit:
                covered = False
        rows.append(
            {
                "tool": name,
                "index": index,
                "params": param_rows,
                "covered": covered,
            }
        )
    return rows


def audit_queries(
    queries: Sequence[Mapping[str, Any]],
    initial: Mapping[str, dict[str, Any]],
    gold_of,
) -> dict[str, Any]:
    catalog = catalog_by_name(initial)
    n_hops = 0
    n_covered = 0
    n_first_hops = 0
    later_hops = 0
    later_covered = 0
    misses: list[dict[str, Any]] = []
    for qi, query in enumerate(queries):
        gold = [str(name) for name in gold_of(query) if name]
        hops = gold_hops(gold, catalog)
        for hop in hops:
            n_hops += 1
            if hop["covered"]:
                n_covered += 1
            if hop["index"] == 0:
                n_first_hops += 1
                continue
            later_hops += 1
            if hop["covered"]:
                later_covered += 1
            else:
                misses.append(
                    {
                        "query_index": qi,
                        "gold": gold,
                        "tool": hop["tool"],
                        "params": hop["params"],
                    }
                )
    return {
        "n_queries": len(queries),
        "n_gold_hops": n_hops,
        "n_first_hops": n_first_hops,
        "n_covered": n_covered,
        "covered_share": (n_covered / n_hops) if n_hops else 0.0,
        "later_hops": later_hops,
        "later_covered": later_covered,
        "later_covered_share": (later_covered / later_hops) if later_hops else 0.0,
        "n_later_misses": len(misses),
        "later_miss_examples": misses[:12],
    }
