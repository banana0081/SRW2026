"""CP% intervals, Win% vs ReAct, and CP vs n from RestBench traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from tooldoc_nir.draft_agent_reproduction import _dump_json
from tooldoc_nir.selection_decompose import exact_sign_test, paired_bootstrap


def load_traces(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def wilson_interval(
    hits: int, n: int, z: float = 1.96
) -> dict[str, float]:
    if n <= 0:
        return {"p": 0.0, "low": 0.0, "high": 0.0, "n": 0, "hits": 0}
    proportion = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (proportion + z2 / (2 * n)) / denom
    margin = (
        z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4 * n)) / n) / denom
    )
    return {
        "p": round(proportion, 4),
        "low": round(max(0.0, center - margin), 4),
        "high": round(min(1.0, center + margin), 4),
        "n": n,
        "hits": hits,
    }


def by_row(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("row") or "")].append(row)
    return dict(grouped)


def condition_cp(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(1 for row in rows if row.get("correct_path") and not row.get("error"))
    n = len(rows)
    interval = wilson_interval(hits, n)
    return {
        "n": n,
        "hits": hits,
        "errors": sum(1 for row in rows if row.get("error")),
        "cp": interval["p"],
        "cp_ci95": [interval["low"], interval["high"]],
    }


def stage_mechanics(row: dict[str, Any]) -> dict[str, bool]:
    """Where one query broke, in terms of the driver stage that owns it.

    CP alone cannot separate "picked the wrong tool" from "picked the right
    tool and filled the id wrongly", and those live on different documentation
    surfaces. The flags below are computed from the executed call sequence
    against the gold sequence, so a variant that only rewrites
    `tool_description` should move the selection flags and leave the parameter
    flags alone.
    """
    gold = [str(name) for name in (row.get("gold") or [])]
    executed = [str(name) for name in (row.get("executed") or [])]
    first = gold[0] if gold else ""
    last = gold[-1] if gold else ""
    middle = gold[1:-1]
    present = set(executed)
    ordered = correct_subsequence(executed, gold)
    return {
        "driver_error": bool(row.get("error")),
        "no_call": not executed,
        "first_producer_present": bool(first) and first in present,
        "intermediate_producer_present": all(name in present for name in middle),
        "final_consumer_present": bool(last) and last in present,
        "repeat_producer": bool(first) and executed.count(first) > 1,
        "wrong_first": bool(executed) and bool(first) and executed[0] != first,
        "wrong_order": bool(gold)
        and all(name in present for name in gold)
        and not ordered,
        "parameter_or_http_error": int(row.get("http_errors") or 0) > 0,
        "correct_path": bool(ordered) and not row.get("error"),
    }


def correct_subsequence(executed: Sequence[str], gold: Sequence[str]) -> bool:
    remaining = list(gold)
    for name in executed:
        if remaining and name == remaining[0]:
            remaining.pop(0)
    return not remaining


def mechanics_profile(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per flag, plus the misses split by the first stage that failed."""
    flags: dict[str, int] = defaultdict(int)
    misses = 0
    missing_first = 0
    missing_intermediate = 0
    missing_consumer = 0
    for row in rows:
        marks = stage_mechanics(row)
        for name, value in marks.items():
            flags[name] += int(value)
        if marks["correct_path"]:
            continue
        misses += 1
        if not marks["first_producer_present"]:
            missing_first += 1
        elif not marks["intermediate_producer_present"]:
            missing_intermediate += 1
        elif not marks["final_consumer_present"]:
            missing_consumer += 1
    return {
        "n": len(rows),
        "flags": dict(flags),
        "misses": misses,
        "missing_first_producer": missing_first,
        "missing_intermediate_producer": missing_intermediate,
        "missing_final_consumer": missing_consumer,
        "other_miss": misses - missing_first - missing_intermediate - missing_consumer,
    }


def path_index(rows: Sequence[dict[str, Any]]) -> dict[int, bool]:
    index: dict[int, bool] = {}
    for row in rows:
        index[int(row.get("query_index") or 0)] = bool(
            row.get("correct_path")
        ) and not bool(row.get("error"))
    return index


def win_vs_react(
    method: Sequence[dict[str, Any]],
    react: Sequence[dict[str, Any]],
    *,
    method_name: str,
) -> dict[str, Any]:
    left = path_index(method)
    right = path_index(react)
    shared = sorted(set(left) & set(right))
    both = 0
    method_only = 0
    react_only = 0
    neither = 0
    differences: list[int] = []
    for query_index in shared:
        method_hit = left[query_index]
        react_hit = right[query_index]
        differences.append(int(method_hit) - int(react_hit))
        if method_hit and react_hit:
            both += 1
        elif method_hit:
            method_only += 1
        elif react_hit:
            react_only += 1
        else:
            neither += 1
    discordant = method_only + react_only
    return {
        "method": method_name,
        "n": len(shared),
        "both": both,
        "method_only": method_only,
        "react_only": react_only,
        "neither": neither,
        "win_pct_over_n": round(100.0 * method_only / len(shared), 1) if shared else None,
        "pairwise_win_pct": (
            round(100.0 * method_only / discordant, 1) if discordant else None
        ),
        "sign_p": round(exact_sign_test(react_only, method_only), 4),
        "delta_vs_react": paired_bootstrap(differences),
    }


def learning_curve(
    rows: Sequence[dict[str, Any]],
    *,
    step: int = 10,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row.get("query_index") or 0))
    points: list[dict[str, Any]] = []
    hits = 0
    for index, row in enumerate(ordered, start=1):
        if row.get("correct_path") and not row.get("error"):
            hits += 1
        if index % step == 0 or index == len(ordered):
            interval = wilson_interval(hits, index)
            points.append(
                {
                    "n": index,
                    "hits": hits,
                    "cp": interval["p"],
                    "cp_ci95": [interval["low"], interval["high"]],
                }
            )
    return points


def seed_pool(cp_values: Sequence[float]) -> dict[str, Any]:
    count = len(cp_values)
    if count == 0:
        return {"seeds": 0, "mean": None, "std": None, "ci95": None}
    mean = sum(cp_values) / count
    if count == 1:
        return {
            "seeds": 1,
            "mean": round(mean, 4),
            "std": 0.0,
            "ci95": [round(mean, 4), round(mean, 4)],
            "values": [round(value, 4) for value in cp_values],
        }
    variance = sum((value - mean) ** 2 for value in cp_values) / (count - 1)
    std = math.sqrt(variance)
    # t approx for small n; z=1.96 is what Stanislav asked for in the table.
    half = 1.96 * std / math.sqrt(count)
    return {
        "seeds": count,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95": [round(mean - half, 4), round(mean + half, 4)],
        "values": [round(value, 4) for value in cp_values],
    }


def report_from_traces(
    traces: dict[str, list[dict[str, Any]]],
    *,
    react_name: str = "ReAct",
    curve_step: int = 10,
) -> dict[str, Any]:
    seed_cps: dict[str, list[float]] = defaultdict(list)
    first_grouped: dict[str, list[dict[str, Any]]] | None = None
    for rows in traces.values():
        grouped = by_row(rows)
        if first_grouped is None:
            first_grouped = grouped
        for name, subset in grouped.items():
            seed_cps[name].append(float(condition_cp(subset)["cp"]))
    grouped = first_grouped or {}
    react_rows = grouped.get(react_name) or []
    conditions: dict[str, Any] = {}
    for name, rows in grouped.items():
        payload = condition_cp(rows)
        payload["learning_curve"] = learning_curve(rows, step=curve_step)
        payload["mechanics"] = mechanics_profile(rows)
        if len(traces) > 1:
            payload["seeds"] = seed_pool(seed_cps[name])
        if react_rows and name != react_name:
            payload["win_vs_react"] = win_vs_react(
                rows, react_rows, method_name=name
            )
        conditions[name] = payload
    return {"conditions": conditions, "trace_files": list(traces)}


def _iter_trace_paths(items: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            candidate = path / "traces.jsonl"
            if candidate.exists():
                paths.append(candidate)
        else:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RestBench CP% CI, Win% vs ReAct, CP vs n."
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="traces.jsonl files or run directories.",
    )
    parser.add_argument("--react-traces", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--curve-step", type=int, default=10)
    args = parser.parse_args()
    paths = _iter_trace_paths(args.traces)
    loaded = {str(path): load_traces(path) for path in paths}
    if args.react_traces is not None:
        react_path = (
            args.react_traces / "traces.jsonl"
            if args.react_traces.is_dir()
            else args.react_traces
        )
        react_rows = [
            row for row in load_traces(react_path) if row.get("row") == "ReAct"
        ]
        if len(loaded) == 1:
            only = next(iter(loaded.values()))
            only.extend(react_rows)
        else:
            loaded[str(react_path)] = react_rows
    report = report_from_traces(loaded, curve_step=args.curve_step)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        _dump_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
