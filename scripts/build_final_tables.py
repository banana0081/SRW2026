"""Assemble the RestBench tables, intervals and mechanism analysis.

Three things this refuses to do, because each one already went wrong once:

1. Pool seeds that were produced from different `Ours` digests. The seed-0
   roots were written before the documentation was frozen, so they are listed
   as excluded instead of averaged in.
2. Score a harness error as anything other than a miss.
3. Present a canonical number and a contemporaneous number as if they were the
   same measurement. The served endpoint changed between them, and the drift
   section states by how much.

  python scripts/build_final_tables.py
  python scripts/build_final_tables.py --output artifacts/results/final_tables.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from tooldoc_nir.draft_agent_reproduction import _dump_json, _load_json
from tooldoc_nir.restbench_report import (
    load_traces,
    mechanics_profile,
    wilson_interval,
)
from tooldoc_nir.restbench_screen import (
    MODELS,
    execution_valid_hit,
    paper_gate,
    student_t_interval,
)

RESULTS = Path("artifacts/results")
ROWS = ("DFSDT", "DRAFT", "Ours")
BASELINE_ROWS: dict[str, str] = {"DRAFT": "DRAFT", "Initial": "DFSDT"}
SEED_DIR = re.compile(r"^restbench_tmdb_(ling|flash)_s(\d+)$")
PROTOCOL = Path("artifacts/documentation/TMDB_positive_scope_protocol.json")


def _interval(values: Sequence[float]) -> dict[str, Any]:
    """Seed-level interval. Student-t, because there are four to ten seeds."""
    interval = student_t_interval(values)
    seeds = interval.pop("replicates", 0)
    return {"seeds": seeds, **interval}


def preregistered_digests(protocol_path: Path = PROTOCOL) -> dict[str, str]:
    """The `Ours` payloads the frozen protocol allows the table to report.

    `table_for` used to keep whichever digest appeared in the most seed roots,
    which silently promotes a majority of mistakes into the reported numbers.
    The protocol decides instead: it names the candidates by payload digest
    before any of them is run, so a root built from anything else is excluded
    rather than averaged in. The majority rule survives only as the fallback
    for the historical roots, which predate any protocol.
    """
    if not protocol_path.exists():
        return {}
    protocol = _load_json(protocol_path)
    stages = protocol.get("stages") or {}
    allowed = (stages.get("final_validation") or {}).get(
        "candidate_payload_digests"
    ) or {}
    return {str(name): str(digest) for name, digest in allowed.items()}


def _hits(rows: Sequence[dict[str, Any]], row_name: str) -> dict[int, bool]:
    return {
        int(item["query_index"]): bool(item.get("correct_path")) and not item.get("error")
        for item in rows
        if item.get("row") == row_name
    }


def canonical_cells(model_key: str) -> dict[int, dict[str, Any]]:
    """Every canonical seed root for one model, keyed by seed."""
    cells: dict[int, dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("restbench_tmdb_*")):
        match = SEED_DIR.match(path.name)
        if not match or match.group(1) != model_key:
            continue
        traces = path / "traces.jsonl"
        manifest = path / "manifest.json"
        if not traces.exists() or not manifest.exists():
            continue
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        ours = (recorded.get("docs") or {}).get("Ours", {})
        cells[int(match.group(2))] = {
            "root": str(path),
            # The payload digest is the identity of the documentation; the file
            # digest also moves with the line endings of whichever machine
            # wrote the JSON. Older roots only carry the latter.
            "ours_digest": str(ours.get("payload_digest") or ours.get("digest") or ""),
            "provider": str(recorded.get("provider") or ""),
            "created_at": recorded.get("created_at"),
            "rows": load_traces(traces),
        }
    return cells


def table_for(
    cells: dict[int, dict[str, Any]],
    *,
    allowed_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    digests = defaultdict(list)
    for seed, cell in cells.items():
        digests[cell["ours_digest"]].append(seed)
    if not digests:
        return {}
    allowed = dict(allowed_digests or {})
    if allowed:
        present = sorted(set(allowed.values()) & set(digests))
        if len(present) != 1:
            return {
                "digest_source": "preregistered protocol",
                "preregistered": {
                    name: digest[:12] for name, digest in allowed.items()
                },
                "seeds_included": [],
                "seeds_excluded_by_digest": {
                    digest[:12]: sorted(seeds) for digest, seeds in digests.items()
                },
                "error": (
                    "no seed root was produced from a preregistered Ours payload"
                    if not present
                    else "seed roots mix two preregistered candidates"
                ),
            }
        frozen = present[0]
        name = next(key for key, value in allowed.items() if value == frozen)
        digest_source = f"preregistered protocol, candidate {name}"
    else:
        frozen = max(digests, key=lambda key: len(digests[key]))
        digest_source = "majority of seed roots, no protocol frozen yet"
    included = sorted(digests[frozen])
    excluded = {
        digest[:12]: sorted(seeds)
        for digest, seeds in digests.items()
        if digest != frozen
    }
    per_row: dict[str, list[float]] = {name: [] for name in ROWS}
    valid_row: dict[str, list[float]] = {name: [] for name in ROWS}
    deltas: dict[str, list[float]] = {label: [] for label in BASELINE_ROWS}
    errors: dict[int, int] = {}
    providers: set[str] = set()
    pooled_rows: list[dict[str, Any]] = []
    for seed in included:
        rows = cells[seed]["rows"]
        pooled_rows.extend(rows)
        errors[seed] = sum(1 for item in rows if item.get("error"))
        providers.add(str(cells[seed].get("provider") or ""))
        for name in ROWS:
            subset = [item for item in rows if item.get("row") == name]
            if not subset:
                continue
            hits = sum(
                1
                for item in subset
                if item.get("correct_path") and not item.get("error")
            )
            per_row[name].append(hits / len(subset))
            valid_row[name].append(
                sum(1 for item in subset if execution_valid_hit(item)) / len(subset)
            )
        ours = _hits(rows, "Ours")
        for label, baseline_row in BASELINE_ROWS.items():
            baseline = _hits(rows, baseline_row)
            shared = sorted(set(ours) & set(baseline))
            if not shared:
                continue
            wins = sum(1 for index in shared if ours[index] and not baseline[index])
            losses = sum(1 for index in shared if baseline[index] and not ours[index])
            deltas[label].append((wins - losses) / len(shared))
    query_level = {}
    mechanics = {}
    for name in ROWS:
        subset = [item for item in pooled_rows if item.get("row") == name]
        if not subset:
            continue
        hits = sum(
            1 for item in subset if item.get("correct_path") and not item.get("error")
        )
        query_level[name] = wilson_interval(hits, len(subset))
        mechanics[name] = mechanics_profile(subset)
    return {
        "frozen_ours_digest": frozen[:12],
        "digest_source": digest_source,
        "providers": sorted(providers - {""}) or ["unpinned"],
        "seeds_included": included,
        "seeds_excluded_by_digest": excluded,
        "errors_per_seed": errors,
        "max_errors_per_seed": max(errors.values(), default=0),
        "seed_level_cp": {name: _interval(values) for name, values in per_row.items()},
        "seed_level_execution_valid_cp": {
            name: _interval(values) for name, values in valid_row.items()
        },
        "paired_ours_minus_baseline": {
            label: _interval(values) for label, values in deltas.items()
        },
        "paired_ours_minus_draft": _interval(deltas["DRAFT"]),
        "query_level_cp_pooled": query_level,
        "mechanics": mechanics,
    }


def validation_cells(arm: str = "currentnot") -> dict[str, dict[int, dict[str, Any]]]:
    cells: dict[str, dict[int, dict[str, Any]]] = {key: {} for key in MODELS}
    root = RESULTS / "validation"
    if not root.exists():
        return cells
    pattern = re.compile(rf"^{arm}_(ling|flash)_s(\d+)$")
    for path in sorted(root.glob(f"{arm}_*")):
        match = pattern.match(path.name)
        traces = path / "traces.jsonl"
        if not match or not traces.exists():
            continue
        cells[match.group(1)][int(match.group(2))] = {
            "root": str(path),
            "rows": load_traces(traces),
        }
    return cells


def drift_section(
    canonical: dict[str, dict[int, dict[str, Any]]],
    contemporaneous: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Same documentation, same seed, different date: how much moved."""
    section: dict[str, Any] = {}
    for model_key, cells in contemporaneous.items():
        for seed, cell in cells.items():
            before = canonical.get(model_key, {}).get(seed)
            if before is None:
                continue
            entry: dict[str, Any] = {}
            for name in ROWS:
                old = _hits(before["rows"], name)
                new = _hits(cell["rows"], name)
                shared = sorted(set(old) & set(new))
                if not shared:
                    continue
                entry[name] = {
                    "n": len(shared),
                    "canonical": sum(1 for index in shared if old[index]),
                    "now": sum(1 for index in shared if new[index]),
                    "recovered": [
                        index for index in shared if new[index] and not old[index]
                    ],
                    "lost": [
                        index for index in shared if old[index] and not new[index]
                    ],
                }
            section[f"{model_key}_s{seed:02d}"] = {
                "canonical_root": before["root"],
                "canonical_created_at": before["created_at"],
                "contemporaneous_root": cell["root"],
                "rows": entry,
            }
    return section


PANEL = (0, 5, 13, 17, 20, 33, 51, 57, 62, 66)


def selection_bias_check(
    canonical: dict[str, dict[int, dict[str, Any]]],
    contemporaneous: dict[str, dict[int, dict[str, Any]]],
    row_name: str = "Ours",
) -> dict[str, Any]:
    """Separate a moving endpoint from regression to the mean.

    The pilot panel was chosen from the queries this very row failed, so a
    re-run recovers much of it by sampling alone. If the same configuration
    also recovers off-panel queries at the same rate, the endpoint moved; if
    only the panel recovers, the panel was selected on its own outcome and
    cannot measure an improvement.
    """
    checks: dict[str, Any] = {}
    panel = set(PANEL)
    for model_key, cells in contemporaneous.items():
        for seed, cell in cells.items():
            before = canonical.get(model_key, {}).get(seed)
            if before is None:
                continue
            old = _hits(before["rows"], row_name)
            new = _hits(cell["rows"], row_name)
            shared = sorted(set(old) & set(new))
            groups: dict[str, dict[str, Any]] = {}
            for label, indices in (
                ("panel", [index for index in shared if index in panel]),
                ("off_panel", [index for index in shared if index not in panel]),
            ):
                failed = [index for index in indices if not old[index]]
                passed = [index for index in indices if old[index]]
                groups[label] = {
                    "n": len(indices),
                    "was_failing": len(failed),
                    "recovered": sum(1 for index in failed if new[index]),
                    "recovery_rate": (
                        round(sum(1 for index in failed if new[index]) / len(failed), 3)
                        if failed
                        else None
                    ),
                    "was_passing": len(passed),
                    "lost": sum(1 for index in passed if not new[index]),
                    "loss_rate": (
                        round(
                            sum(1 for index in passed if not new[index]) / len(passed), 3
                        )
                        if passed
                        else None
                    ),
                }
            checks[f"{model_key}_s{seed:02d}"] = groups
    return checks


def build(protocol_path: Path = PROTOCOL) -> dict[str, Any]:
    canonical = {key: canonical_cells(key) for key in MODELS}
    contemporaneous = validation_cells()
    allowed = preregistered_digests(protocol_path)
    # The historical canonical roots predate the protocol, so they keep the
    # majority rule; the protocol governs the roots the protocol produced.
    reportable = {
        key: cells
        for key, cells in canonical.items()
        if any(cell["ours_digest"] in set(allowed.values()) for cell in cells.values())
    }
    tables = {
        key: table_for(
            cells, allowed_digests=allowed if key in reportable else None
        )
        for key, cells in canonical.items()
        if cells
    }
    seed_effects: dict[str, list[float]] = {}
    for model_key, table in tables.items():
        for label, interval in (table.get("paired_ours_minus_baseline") or {}).items():
            values = interval.get("values") or []
            if values:
                seed_effects[f"{model_key}:{label}"] = [float(v) for v in values]
    payload: dict[str, Any] = {
        "models": {key: MODELS[key] for key in MODELS},
        "error_policy": (
            "a driver failure or malformed completion counts as a miss; a "
            "query whose transport never recovered is dropped from every arm"
        ),
        "primary_criterion": (
            "seed-level paired Student-t 95% intervals above zero for all four "
            "contrasts (2 models x Ours-DRAFT / Ours-Initial), one frozen "
            "documentation payload on one pinned provider"
        ),
        "secondary_criterion": (
            "execution-valid CP: the gold path was executed with no call "
            "rejected because the agent filled a parameter wrongly"
        ),
        "preregistered_ours_digests": {
            name: digest[:12] for name, digest in allowed.items()
        }
        or "none frozen",
        "protocol": str(protocol_path) if allowed else "",
        "canonical": tables,
        "contemporaneous": {
            key: table_for(
                {
                    seed: {**cell, "ours_digest": "contemporaneous", "created_at": None}
                    for seed, cell in cells.items()
                }
            )
            for key, cells in contemporaneous.items()
            if cells
        },
        "paper_gate": paper_gate(seed_effects),
        "endpoint_drift": drift_section(canonical, contemporaneous),
        "selection_bias_check": selection_bias_check(canonical, contemporaneous),
    }
    return payload


def render(payload: dict[str, Any]) -> None:
    for label in ("canonical", "contemporaneous"):
        block = payload.get(label) or {}
        if not block:
            continue
        print(f"== {label}")
        for model_key, table in block.items():
            seeds = table.get("seeds_included") or []
            print(
                f"  {model_key}: seeds {seeds}, digest {table.get('frozen_ours_digest')}"
                f" ({table.get('digest_source')}), provider "
                f"{table.get('providers')}, max errors/seed "
                f"{table.get('max_errors_per_seed')}"
            )
            if table.get("error"):
                print(f"    {table['error']}")
                continue
            if table.get("seeds_excluded_by_digest"):
                print(f"    excluded by digest: {table['seeds_excluded_by_digest']}")
            for name in ROWS:
                cell = (table.get("seed_level_cp") or {}).get(name) or {}
                if not cell:
                    continue
                interval = cell.get("ci95")
                shown = (
                    f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "n/a"
                )
                print(f"    {name:6s} mean {cell.get('mean')} ci95 {shown}")
            for baseline, paired in (
                table.get("paired_ours_minus_baseline") or {}
            ).items():
                interval = paired.get("ci95")
                shown = (
                    f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "n/a"
                )
                print(
                    f"    Ours-{baseline:7s} paired mean {paired.get('mean')} "
                    f"t-ci95 {shown} over {paired.get('seeds')} seeds"
                )
            for name in ROWS:
                profile = (table.get("mechanics") or {}).get(name) or {}
                if not profile:
                    continue
                print(
                    f"    {name:6s} misses {profile['misses']} "
                    f"(first {profile['missing_first_producer']}, "
                    f"intermediate {profile['missing_intermediate_producer']}, "
                    f"consumer {profile['missing_final_consumer']}, "
                    f"other {profile['other_miss']})"
                )
    gate = payload.get("paper_gate") or {}
    if gate.get("intervals"):
        print("== paper gate: four paired Student-t intervals above zero")
        for label, interval in gate["intervals"].items():
            bounds = interval.get("ci95")
            shown = f"[{bounds[0]:+.3f}, {bounds[1]:+.3f}]" if bounds else "n/a"
            print(
                f"  {label:16s} mean {interval.get('mean')} ci95 {shown} "
                f"over {interval.get('replicates')} seeds"
            )
        print(f"  met: {gate.get('met')} failing: {gate.get('failing')}")
    drift = payload.get("endpoint_drift") or {}
    if drift:
        print("== re-measurement, same documentation and seed")
        for cell, entry in drift.items():
            for name, value in entry["rows"].items():
                print(
                    f"  {cell} {name:6s} {value['canonical']}/{value['n']} -> "
                    f"{value['now']}/{value['n']} "
                    f"(+{len(value['recovered'])} / -{len(value['lost'])})"
                )
    checks = payload.get("selection_bias_check") or {}
    if checks:
        print("== Ours re-measurement split by panel membership")
        for cell, groups in checks.items():
            for label, value in groups.items():
                print(
                    f"  {cell} {label:9s} recovered "
                    f"{value['recovered']}/{value['was_failing']} "
                    f"(rate {value['recovery_rate']}), lost "
                    f"{value['lost']}/{value['was_passing']} "
                    f"(rate {value['loss_rate']})"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "final_tables.json"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROTOCOL,
        help="Frozen protocol whose Ours payload digest decides which seed "
        "roots may be reported.",
    )
    args = parser.parse_args()
    payload = build(args.protocol)
    _dump_json(args.output, payload)
    render(payload)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
