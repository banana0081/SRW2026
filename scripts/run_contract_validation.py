"""Full RestBench-TMDB validation of one frozen documentation variant.

The pilot only picks a candidate. This script runs the real comparison the
table needs: n=100, `Initial` / `DRAFT` / the candidate interleaved inside one
process per (model, seed), the released decoding, and a fixed retry policy so
that a rate-limit episode is not scored as a miss.

Stages, deliberately separate so a regression stops the spend:

  # 1. freeze the candidate and run the two hard seeds on both models
  python scripts/run_contract_validation.py --arm H123 --seeds 4 7
  # 2. only if the paired delta is positive on both models
  python scripts/run_contract_validation.py --arm H123 --seeds 0 1 2 3 5 6 8 9
  # 3. the seed-level paired interval that goes into the table
  python scripts/run_contract_validation.py --arm H123 --report --seeds 0 1 2 3 4 5 6 7 8 9

Seed 0 of the historical runs is not comparable with seeds 1-9: those roots
were produced from different digests. This script writes its own roots, so
every seed it reports shares one candidate payload digest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from tooldoc_nir.draft_agent_reproduction import _dump_json, _load_json
from tooldoc_nir.provenance import payload_digest
from tooldoc_nir.restbench_contract_variants import variant_paths
from tooldoc_nir.restbench_report import load_traces, mechanics_profile, wilson_interval
from tooldoc_nir.restbench_screen import (
    MODELS,
    PROVIDERS,
    execution_valid_hit,
    paired_effect,
    paper_gate,
    scored_rows,
    student_t_interval,
)
from tooldoc_nir.restbench_table import is_transport_error

VALIDATION_ROOT = Path("artifacts/results/validation")
MAX_ERRORS_PER_SEED = 1
RETRY_ROUNDS = 3
ROWS = ("DFSDT", "DRAFT", "Ours")

# `DFSDT` is the Initial documentation under the same controller, so it is the
# `Initial` baseline of the plan. Both contrasts have to hold on both models.
BASELINE_ROWS: dict[str, str] = {"DRAFT": "DRAFT", "Initial": "DFSDT"}


def run_root(arm: str, model_key: str, seed: int) -> Path:
    return VALIDATION_ROOT / f"{arm.lower()}_{model_key}_s{seed:02d}"


def _table_command(
    *,
    arm_docs: Path,
    model: str,
    provider: str | None,
    seed: int,
    root: Path,
    workers: int,
    max_cost_usd: float,
    extra: Sequence[str] = (),
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tooldoc_nir.restbench_table",
        "--dataset",
        "TMDB",
        "--rows",
        *ROWS,
        "--ours-docs",
        str(arm_docs),
        "--model",
        model,
        *(("--provider", provider) if provider else ()),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--max-cost-usd",
        str(max_cost_usd),
        "--output-root",
        str(root),
        *extra,
    ]


def transport_faults_in(root: Path) -> int:
    """Count only what a retry can fix; driver failures stay misses."""
    traces = root / "traces.jsonl"
    if not traces.exists():
        return -1
    return sum(1 for row in load_traces(traces) if is_transport_error(row))


def errors_in(root: Path) -> int:
    traces = root / "traces.jsonl"
    if not traces.exists():
        return -1
    return sum(1 for row in load_traces(traces) if row.get("error"))


def run_cell(
    *,
    arm: str,
    model_key: str,
    seed: int,
    workers: int,
    max_cost_usd: float,
    provider: str | None = None,
) -> dict[str, Any]:
    docs, _manifest = variant_paths(arm)
    if not docs.exists():
        raise SystemExit(f"{docs} is missing; build the variants first.")
    root = run_root(arm, model_key, seed)
    command = _table_command(
        arm_docs=docs,
        model=MODELS[model_key],
        provider=provider,
        seed=seed,
        root=root,
        workers=workers,
        max_cost_usd=max_cost_usd,
    )
    print(f"[{arm} {model_key} s{seed:02d}] {' '.join(command[-12:])}", flush=True)
    subprocess.run(command, check=False)
    for attempt in range(1, RETRY_ROUNDS + 1):
        remaining = transport_faults_in(root)
        if remaining <= MAX_ERRORS_PER_SEED:
            break
        print(
            f"[{arm} {model_key} s{seed:02d}] retry {attempt}/{RETRY_ROUNDS}, "
            f"{remaining} transport faults left",
            flush=True,
        )
        subprocess.run(
            _table_command(
                arm_docs=docs,
                model=MODELS[model_key],
                provider=provider,
                seed=seed,
                root=root,
                workers=workers,
                max_cost_usd=max_cost_usd,
                extra=("--retry-errors",),
            ),
            check=False,
        )
    return {
        "root": str(root),
        "errors": errors_in(root),
        "transport_faults": transport_faults_in(root),
        "candidate_digest": payload_digest(_load_json(docs)),
    }


def cell_scores(root: Path) -> dict[str, Any]:
    """CP per row with errors scored as misses, plus the stage mechanics."""
    traces = root / "traces.jsonl"
    if not traces.exists():
        return {}
    rows = scored_rows(load_traces(traces))
    scores: dict[str, Any] = {}
    for name in ROWS:
        subset = [row for row in rows if row.get("row") == name]
        if not subset:
            continue
        profile = mechanics_profile(subset)
        hits = profile["flags"].get("correct_path", 0)
        interval = wilson_interval(hits, len(subset))
        valid = sum(1 for row in subset if execution_valid_hit(row))
        scores[name] = {
            "n": len(subset),
            "hits": hits,
            "cp": interval["p"],
            "cp_ci95": [interval["low"], interval["high"]],
            "execution_valid_hits": valid,
            "execution_valid_cp": round(valid / len(subset), 4),
            "errors": profile["flags"].get("driver_error", 0),
            "transport_faults": sum(1 for row in subset if is_transport_error(row)),
            "missing_first_producer": profile["missing_first_producer"],
            "missing_final_consumer": profile["missing_final_consumer"],
        }
    for label, baseline_row in BASELINE_ROWS.items():
        effect = paired_effect(rows, "Ours", baseline_row)
        scores[f"paired_ours_minus_{label.lower()}"] = {
            **effect,
            "delta": (
                round(effect["net"] / effect["n"], 4) if effect["n"] else None
            ),
        }
    return scores


def report(arm: str, seeds: Sequence[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {"arm": arm, "models": {}}
    digests: set[str] = set()
    providers: set[str] = set()
    seed_effects: dict[str, list[float]] = {}
    for model_key in MODELS:
        cells: dict[str, Any] = {}
        deltas: dict[str, list[float]] = {label: [] for label in BASELINE_ROWS}
        for seed in seeds:
            root = run_root(arm, model_key, seed)
            scores = cell_scores(root)
            if not scores:
                continue
            manifest = root / "manifest.json"
            if manifest.exists():
                recorded = _load_json(manifest)
                digest = (recorded.get("docs") or {}).get("Ours", {}).get(
                    "payload_digest"
                )
                if digest:
                    digests.add(str(digest))
                providers.add(str(recorded.get("provider") or ""))
            cells[f"s{seed:02d}"] = scores
            for label in BASELINE_ROWS:
                delta = scores[f"paired_ours_minus_{label.lower()}"]["delta"]
                if delta is not None:
                    deltas[label].append(float(delta))
        if not cells:
            continue
        payload["models"][model_key] = {
            "cells": cells,
            "paired": {
                label: student_t_interval(values)
                for label, values in deltas.items()
            },
        }
        for label, values in deltas.items():
            if values:
                seed_effects[f"{model_key}:{label}"] = values
    payload["candidate_digests"] = sorted(digests)
    payload["providers"] = sorted(providers)
    payload["one_frozen_candidate"] = len(digests) <= 1
    payload["one_pinned_provider"] = len(providers) <= 1
    payload["both_models_present"] = set(payload["models"]) == set(MODELS)
    payload["paper_gate"] = paper_gate(seed_effects)
    # Four contrasts, both models, one frozen payload on one pinned provider.
    # Anything less is not the comparison the plan asked for.
    payload["primary_criterion_met"] = bool(
        payload["one_frozen_candidate"]
        and payload["one_pinned_provider"]
        and payload["both_models_present"]
        and len(seed_effects) == 2 * len(MODELS)
        and payload["paper_gate"]["met"]
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4, 7])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument(
        "--providers",
        nargs="*",
        default=[],
        metavar="MODEL=PROVIDER",
        help="Pin one OpenRouter provider per model, e.g. ling=novita.",
    )
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-cost-usd", type=float, default=3.00)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    pinned: dict[str, str] = dict(PROVIDERS)
    for item in args.providers:
        model_key, separator, provider = item.partition("=")
        if not separator or model_key not in MODELS:
            raise SystemExit(f"--providers takes MODEL=PROVIDER, got {item!r}")
        pinned[model_key] = provider

    if args.report:
        payload = report(args.arm, args.seeds)
        _dump_json(
            VALIDATION_ROOT / f"{args.arm.lower()}_report.json", payload
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    unknown = [name for name in args.models if name not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; have {list(MODELS)}")
    for seed in args.seeds:
        for model_key in args.models:
            result = run_cell(
                arm=args.arm,
                model_key=model_key,
                seed=seed,
                workers=args.workers,
                max_cost_usd=args.max_cost_usd,
                provider=pinned.get(model_key),
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps(report(args.arm, args.seeds), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
