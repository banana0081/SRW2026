"""Run the frozen Cross-model Positive Scope experiment, one stage at a time.

Everything that decides the outcome was written down before the first model
call: the two disjoint panels, the pinned providers, the arms with their
payload digests, the gate thresholds and the stopping rule. This script only
executes that protocol and refuses to run anything it cannot match against it.

  # Screen A: 24 untouched queries, seeds 21/22, both models, five arms
  python scripts/run_positive_screen.py --stage screen_a
  python scripts/run_positive_screen.py --stage screen_a --report

  # Promotion B: only for the Screen A winner, on 30 further untouched queries
  python scripts/run_positive_screen.py --stage promotion_b
  python scripts/run_positive_screen.py --stage promotion_b --report

Promotion B will not start unless the Screen A report exists and promoted a
candidate, and the full ten-seed validation will not start unless Promotion B
confirmed it. That ordering is the point: a candidate chosen and confirmed on
the same queries has only been looked at twice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # `python scripts/run_positive_screen.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tooldoc_nir.draft_agent_reproduction import _dump_json, _load_json  # noqa: E402
from tooldoc_nir.openrouter import load_env_file  # noqa: E402
from tooldoc_nir.provenance import payload_digest  # noqa: E402
from tooldoc_nir.restbench_contract_variants import (  # noqa: E402
    EXPERIMENT_ROOT,
    variant_paths,
)
from tooldoc_nir.restbench_data import DEFAULT_DRAFT_ROOT  # noqa: E402
from tooldoc_nir.restbench_report import load_traces  # noqa: E402
from tooldoc_nir.restbench_screen import (  # noqa: E402
    MODELS,
    PROVIDERS,
    promotion_verdict,
    scored_rows,
    screen_gate,
)

from scripts.freeze_positive_scope import stable_view  # noqa: E402
from scripts.run_contract_pilot import arm_scores, run_cell  # noqa: E402

DOCUMENTATION = Path("artifacts/documentation")
PROTOCOL = DOCUMENTATION / "TMDB_positive_scope_protocol.json"
RESULTS = Path("artifacts/results/positive_scope")
STAGES = ("screen_a", "promotion_b")


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    """Read the sealed protocol and refuse a payload that was edited."""
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; run scripts/freeze_positive_scope.py first"
        )
    protocol = _load_json(path)
    if protocol.get("selection_digest") != payload_digest(stable_view(protocol)):
        raise SystemExit(f"{path} has an invalid selection digest")
    if dict(protocol["providers"]) != dict(PROVIDERS):
        raise SystemExit(
            f"the protocol pins {protocol['providers']} but the harness would "
            f"use {dict(PROVIDERS)}"
        )
    return protocol


def panel_for(stage: Mapping[str, Any]) -> list[int]:
    """The frozen indices of one stage, checked against the protocol digest."""
    path = Path(stage["panel"]["path"])
    if not path.exists():
        raise SystemExit(f"{path} is missing; the panel must be frozen first")
    panel = _load_json(path)
    if panel.get("selection_digest") != payload_digest(stable_view(panel)):
        raise SystemExit(f"{path} has an invalid selection digest")
    if panel["selection_digest"] != stage["panel"]["selection_digest"]:
        raise SystemExit(f"{path} is not the panel the protocol pinned")
    indices = [int(index) for index in panel["selection"]["indices"]]
    if len(indices) != int(stage["panel"]["n"]):
        raise SystemExit(f"{path} no longer has {stage['panel']['n']} queries")
    return indices


def verify_arm_payloads(arms: Sequence[str], expected: Mapping[str, str]) -> None:
    """The compiled JSON on disk has to be the payload the protocol named."""
    for arm in arms:
        path, _manifest = variant_paths(arm)
        if not path.exists():
            raise SystemExit(
                f"{path} is missing; build the variants with "
                "python -m tooldoc_nir.restbench_contract_variants"
            )
        actual = payload_digest(_load_json(path))
        if actual != expected.get(arm):
            raise SystemExit(
                f"{arm} on disk is {actual[:16]}... but the protocol pinned "
                f"{str(expected.get(arm))[:16]}..."
            )


def stage_root(stage_name: str, root: Path = RESULTS) -> Path:
    return root / stage_name


def stage_rows(
    stage_name: str,
    seeds: Sequence[int],
    *,
    root: Path = RESULTS,
) -> dict[str, list[dict[str, Any]]]:
    """Pool every seed of one stage, per model, dropping invalidated queries."""
    pooled: dict[str, list[dict[str, Any]]] = {}
    for model_key in MODELS:
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            traces = (
                stage_root(stage_name, root) / f"{model_key}_s{seed:02d}"
                / "traces.jsonl"
            )
            if traces.exists():
                rows.extend(scored_rows(load_traces(traces)))
        if rows:
            pooled[model_key] = rows
    return pooled


def selection_surfaces(arms: Sequence[str]) -> dict[str, int]:
    """How many endpoints each candidate rewrote on the selection surface.

    The Screen A tie-break prefers the smaller intervention, so it counts the
    endpoints whose `tool_description` moved rather than the abstract number of
    surfaces, which is one for both candidates.
    """
    counts: dict[str, int] = {}
    for arm in arms:
        _docs, manifest_path = variant_paths(arm)
        if not manifest_path.exists():
            continue
        manifest = _load_json(manifest_path)
        counts[arm] = int(
            (manifest.get("surfaces_touched") or {}).get("tool_description", 0)
        )
    return counts


def run_stage(
    protocol: Mapping[str, Any],
    stage_name: str,
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    models: Sequence[str],
    draft_root: Path,
    max_cost_usd: float,
    root: Path = RESULTS,
) -> list[dict[str, Any]]:
    stage = protocol["stages"][stage_name]
    queries = panel_for(stage)
    provenance = {
        "kind": "preregistered_positive_scope",
        "path": str(PROTOCOL),
        "payload_digest": protocol["selection_digest"],
        "stage": stage_name,
    }
    panel_provenance = {
        "kind": "frozen_manifest",
        "path": stage["panel"]["path"],
        "selection_digest": stage["panel"]["selection_digest"],
    }
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        for model_key in models:
            summary = run_cell(
                model_key=model_key,
                seed=seed,
                arms=arms,
                queries=queries,
                workers=int(protocol["workers"]),
                max_cost_usd=max_cost_usd,
                draft_root=draft_root,
                pilot_root=stage_root(stage_name, root),
                panel_provenance=panel_provenance,
                protocol_provenance=provenance,
                provider=PROVIDERS[model_key],
            )
            summaries.append(
                {
                    "stage": stage_name,
                    "model": model_key,
                    "seed": seed,
                    "cost_usd": summary["cost_usd"],
                    "cp": {arm: value["cp"] for arm, value in summary["arms"].items()},
                    "invalidated_queries": summary["invalidated_queries"],
                    "driver_failures": len(summary["driver_failures"]),
                }
            )
            print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    return summaries


def screen_report(
    protocol: Mapping[str, Any], *, root: Path = RESULTS
) -> dict[str, Any]:
    stage = protocol["stages"]["screen_a"]
    seeds = [int(seed) for seed in stage["seeds"]]
    arms = list(stage["arms"])
    candidates = list(stage["primary_candidates"])
    queries = panel_for(stage)
    pooled = stage_rows("screen_a", seeds, root=root)
    gate = screen_gate(
        pooled,
        candidates,
        expected_cells=len(queries) * len(seeds),
        surfaces=selection_surfaces(candidates),
        baselines=list(stage["baselines"]),
        incumbent=str(stage["incumbent"]),
        models=list(MODELS),
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["selection_digest"],
        "panel": queries,
        "seeds": seeds,
        "providers": dict(PROVIDERS),
        "arm_cp": {
            model_key: arm_scores(rows, arms) for model_key, rows in pooled.items()
        },
        "gate": gate,
    }


def promotion_report(
    protocol: Mapping[str, Any], winner: str, *, root: Path = RESULTS
) -> dict[str, Any]:
    stage = protocol["stages"]["promotion_b"]
    seeds = [int(seed) for seed in stage["seeds"]]
    queries = panel_for(stage)
    pooled = stage_rows("promotion_b", seeds, root=root)
    verdict = promotion_verdict(
        pooled,
        winner,
        baselines=list(protocol["stages"]["screen_a"]["baselines"]),
        models=list(MODELS),
        expected_queries=len(queries),
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["selection_digest"],
        "panel": queries,
        "seeds": seeds,
        "providers": dict(PROVIDERS),
        "arm": winner,
        "arm_payload_digest": stage["candidate_payload_digests"].get(winner, ""),
        "arm_cp": {
            model_key: arm_scores(rows, ["Initial", "DRAFT", winner])
            for model_key, rows in pooled.items()
        },
        "verdict": verdict,
    }


def screen_winner(root: Path = RESULTS) -> str:
    path = stage_root("screen_a", root) / "screen_a_report.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; run --stage screen_a --report before "
            "Promotion B, so the winner is fixed before it is confirmed"
        )
    report = _load_json(path)
    winner = (report.get("gate") or {}).get("promoted")
    if not winner:
        raise SystemExit(
            "Screen A promoted nobody. The stopping rule says to stop tuning "
            "on TMDB and move to Spotify/G3 with the frozen Ours."
        )
    return str(winner)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--output-root", type=Path, default=RESULTS)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to a subset of the protocol seeds, for resuming.",
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--max-cost-usd", type=float, default=1.50)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    stage = protocol["stages"][args.stage]
    root: Path = args.output_root

    if args.stage == "screen_a":
        arms = list(stage["arms"])
        expected = dict(stage["arm_payload_digests"])
    else:
        winner = screen_winner(root)
        arms = [*[arm for arm in stage["arms"] if arm != "screen_a_winner"], winner]
        expected = {
            **{
                arm: protocol["stages"]["screen_a"]["arm_payload_digests"][arm]
                for arm in arms
                if arm != winner
            },
            winner: stage["candidate_payload_digests"][winner],
        }

    if args.report:
        if args.stage == "screen_a":
            payload = screen_report(protocol, root=root)
            gate = payload["gate"]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            print(f"\n{gate['decision']}")
        else:
            payload = promotion_report(protocol, arms[-1], root=root)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            print(
                "\nPromotion B "
                + ("confirmed " if payload["verdict"]["confirmed"] else "rejected ")
                + payload["arm"]
            )
        _dump_json(stage_root(args.stage, root) / f"{args.stage}_report.json", payload)
        return 0

    if not EXPERIMENT_ROOT.exists():
        raise SystemExit(f"{EXPERIMENT_ROOT} is missing; build the variants first.")
    verify_arm_payloads(arms, expected)
    seeds = args.seeds if args.seeds is not None else [
        int(seed) for seed in stage["seeds"]
    ]
    unknown = [seed for seed in seeds if seed not in stage["seeds"]]
    if unknown:
        raise SystemExit(f"--seeds must come from the protocol: {stage['seeds']}")
    unknown_models = [name for name in args.models if name not in MODELS]
    if unknown_models:
        raise SystemExit(f"unknown models {unknown_models}; have {list(MODELS)}")

    load_env_file()
    print(
        f"{args.stage}: {stage['panel']['n']} queries x {len(arms)} arms x "
        f"{len(seeds)} seeds x {len(args.models)} models, arms {arms}, "
        f"providers {dict(PROVIDERS)}",
        flush=True,
    )
    run_stage(
        protocol,
        args.stage,
        arms=arms,
        seeds=seeds,
        models=args.models,
        draft_root=args.draft_root,
        max_cost_usd=args.max_cost_usd,
        root=root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
