"""Score Spotify execution-valid CP from frozen traces.

Playback, scope and quota HTTP failures are not fill rejects when the gold
path was reached and identifiers were emitted. Harness leftover errors are
dropped from n.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tooldoc_nir.restbench_report import load_traces
from tooldoc_nir.restbench_screen import (
    execution_valid_hit,
    hit,
    paired_effect,
    student_t_interval,
)
from tooldoc_nir.restbench_spotify_ids import attach_id_errors

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "artifacts" / "results"
ROWS = ("DFSDT", "DRAFT", "Ours")
BASELINES = {"DRAFT": "DRAFT", "Initial": "DFSDT"}


def seed_root(model: str, seed: int) -> Path:
    if model == "flash" and seed == 0:
        return RESULTS / "restbench_spotify_flash"
    if model == "qwen" and seed == 0:
        return RESULTS / "restbench_spotify_qwen"
    return RESULTS / f"restbench_spotify_{model}_s{seed:02d}"


def load_seed(model: str, seed: int) -> list[dict[str, Any]]:
    root = seed_root(model, seed)
    traces = root / "traces.jsonl"
    if not traces.exists():
        return []
    return attach_id_errors(load_traces(traces), root)


def finished(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("row") == arm and not row.get("error")]


def rate(rows: list[dict[str, Any]], scorer) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "hits": 0, "p": None}
    hits = sum(1 for row in rows if scorer(row))
    return {"n": len(rows), "hits": hits, "p": hits / len(rows)}


def interval_pp(values: list[float]) -> dict[str, Any]:
    packed = student_t_interval(values)
    if packed.get("replicates", 0) == 0:
        return packed
    out = {
        "seeds": packed["replicates"],
        "mean": round(100.0 * packed["mean"], 1),
        "values": [round(100.0 * value, 1) for value in packed.get("values") or []],
    }
    if "ci95" in packed:
        out["ci95"] = [round(100.0 * packed["ci95"][0], 1), round(100.0 * packed["ci95"][1], 1)]
    return out


def model_table(model: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    per_row: dict[str, list[float]] = {name: [] for name in ROWS}
    path_row: dict[str, list[float]] = {name: [] for name in ROWS}
    deltas: dict[str, list[float]] = {label: [] for label in BASELINES}
    errors: dict[int, int] = {}
    missing: list[int] = []
    seed_detail: dict[str, Any] = {}
    for seed in seeds:
        rows = load_seed(model, seed)
        if not rows:
            missing.append(seed)
            continue
        errors[seed] = sum(1 for row in rows if row.get("error"))
        detail = {}
        for name in ROWS:
            subset = finished(rows, name)
            exec_rate = rate(subset, execution_valid_hit)
            path_rate = rate(subset, hit)
            if exec_rate["p"] is not None:
                per_row[name].append(exec_rate["p"])
            if path_rate["p"] is not None:
                path_row[name].append(path_rate["p"])
            detail[name] = {"exec": exec_rate, "path": path_rate}
        scored = [row for row in rows if not row.get("error")]
        for label, baseline in BASELINES.items():
            contrast = paired_effect(
                scored, "Ours", baseline, metric="execution_valid_cp"
            )
            if contrast["n"]:
                deltas[label].append(contrast["net"] / contrast["n"])
            detail[f"vs_{label}"] = contrast
        seed_detail[str(seed)] = detail
    return {
        "model": model,
        "seeds_requested": list(seeds),
        "seeds_missing": missing,
        "errors_per_seed": errors,
        "execution_valid_cp": {name: interval_pp(values) for name, values in per_row.items() if values},
        "path_cp": {name: interval_pp(values) for name, values in path_row.items() if values},
        "paired_ours_minus_baseline": {
            label: interval_pp(values) for label, values in deltas.items() if values
        },
        "per_seed": seed_detail,
    }


def main() -> None:
    payload = {
        "ling": model_table("ling", tuple(range(10))),
        "flash": model_table("flash", tuple(range(10))),
        "qwen": model_table("qwen", (0,)),
    }
    out = RESULTS / "spotify_exec.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for model, table in payload.items():
        print(f"== {model}")
        print("  missing", table["seeds_missing"], "errors", table["errors_per_seed"])
        for name, interval in table["execution_valid_cp"].items():
            print(f"  exec {name}: {interval}")
        for name, interval in table["path_cp"].items():
            print(f"  path {name}: {interval}")
        for label, interval in table["paired_ours_minus_baseline"].items():
            print(f"  delta vs {label}: {interval}")


if __name__ == "__main__":
    main()
