"""Rescore frozen TMDB traces: execution-valid, gold-id, query-level bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tooldoc_nir.restbench_report import load_traces
from tooldoc_nir.restbench_screen import (
    attach_fill_errors,
    execution_valid_hit,
    hit,
    query_differences,
    student_t_interval,
)
from tooldoc_nir.restbench_tmdb_ids import attach_id_errors, load_gold_ids
from tooldoc_nir.selection_decompose import paired_bootstrap

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results"
OUT = REPO / "artifacts" / "results" / "tmdb_rescored.json"
ROWS = ("DFSDT", "DRAFT", "Ours")
BASELINES = {"DRAFT": "DRAFT", "Initial": "DFSDT"}
MODELS = {"ling": (1, 9), "flash": (1, 9)}


def seed_root(model: str, seed: int) -> Path:
    return RESULTS / f"restbench_tmdb_{model}_s{seed:02d}"


def load_seed(model: str, seed: int, table: dict[str, Any]) -> list[dict[str, Any]]:
    root = seed_root(model, seed)
    traces = root / "traces.jsonl"
    if not traces.exists():
        return []
    rows = attach_fill_errors(load_traces(traces), root)
    return attach_id_errors(rows, root, table)


def finished(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("row") == arm and not row.get("error")]


def rate(rows: list[dict[str, Any]], scorer) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if scorer(row)) / len(rows)


def interval_pp(values: list[float]) -> dict[str, Any]:
    packed = student_t_interval(values)
    if packed.get("replicates", 0) == 0:
        return packed
    out: dict[str, Any] = {
        "seeds": packed["replicates"],
        "mean": round(100.0 * packed["mean"], 1),
        "values": [round(100.0 * value, 1) for value in packed.get("values") or []],
    }
    if "ci95" in packed:
        out["ci95"] = [
            round(100.0 * packed["ci95"][0], 1),
            round(100.0 * packed["ci95"][1], 1),
        ]
    return out


def query_bootstrap(rows: list[dict[str, Any]], arm: str, baseline: str, metric: str) -> dict[str, Any]:
    diffs = query_differences(rows, arm, baseline, metric=metric)
    values = list(diffs.values())
    boot = paired_bootstrap([1 if value > 0 else -1 if value < 0 else 0 for value in values])
    mean_pp = 100.0 * (sum(values) / len(values)) if values else 0.0
    scaled = [100.0 * value for value in values]
    interval = paired_bootstrap(scaled)
    return {
        "n_queries": len(values),
        "mean_pp": round(mean_pp, 2),
        "bootstrap_pp": {
            "mean": interval["mean"],
            "low": interval["low"],
            "high": interval["high"],
        },
        "sign_boot": boot,
        "covers_zero": interval["low"] <= 0.0 <= interval["high"],
    }


def model_table(model: str, seeds: tuple[int, int], table: dict[str, Any]) -> dict[str, Any]:
    start, end = seeds
    per_fill: dict[str, list[float]] = {name: [] for name in ROWS}
    per_id: dict[str, list[float]] = {name: [] for name in ROWS}
    per_path: dict[str, list[float]] = {name: [] for name in ROWS}
    all_rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in range(start, end + 1):
        rows = load_seed(model, seed, table)
        if not rows:
            missing.append(seed)
            continue
        all_rows.extend(rows)
        for arm in ROWS:
            done = finished(rows, arm)
            fill = rate(done, lambda row: execution_valid_hit({k: v for k, v in row.items() if k != "id_errors"}))
            ident = rate(done, execution_valid_hit)
            path = rate(done, hit)
            if fill is not None:
                per_fill[arm].append(fill)
            if ident is not None:
                per_id[arm].append(ident)
            if path is not None:
                per_path[arm].append(path)
    fill_rows = [
        {key: value for key, value in row.items() if key != "id_errors"}
        for row in all_rows
    ]
    paper_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"id_errors", "fill_errors"}
        }
        for row in all_rows
    ]
    contrasts = {}
    for label, baseline in BASELINES.items():
        contrasts[label] = {
            "seed_fill": interval_pp(
                [ours - base for ours, base in zip(per_fill["Ours"], per_fill[baseline])]
            )
            if per_fill["Ours"] and per_fill[baseline]
            else {},
            "query_fill": query_bootstrap(
                fill_rows, "Ours", baseline, "execution_valid_cp"
            ),
            "query_gold_id": query_bootstrap(
                all_rows, "Ours", baseline, "execution_valid_cp"
            ),
            "query_paper_http": query_bootstrap(
                paper_rows, "Ours", baseline, "execution_valid_cp"
            ),
        }
    return {
        "model": model,
        "missing_seeds": missing,
        "fill": {arm: interval_pp(values) for arm, values in per_fill.items()},
        "gold_id": {arm: interval_pp(values) for arm, values in per_id.items()},
        "path": {arm: interval_pp(values) for arm, values in per_path.items()},
        "contrasts": contrasts,
        "n_rows": len(all_rows),
    }


def main() -> None:
    table = load_gold_ids()
    report = {
        model: model_table(model, seeds, table) for model, seeds in MODELS.items()
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
