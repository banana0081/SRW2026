"""Diagnostic pilot for the stage-localized documentation variants.

Ten RestBench-TMDB queries, both evaluation models, one seed per invocation,
every arm interleaved in a randomized order so that a rate-limit episode or a
warm HTTP cache cannot land on one arm systematically. Decoding stays at the
released setting (T=0.2, top_p=1, max_tokens=2000, retrieval_num=5), the same
as every canonical run, because the pilot has to be comparable to them.

This is diagnosis, not evidence: 10 queries x 2 seeds cannot support a CI, and
none of these numbers belong in the final table. The pilot only decides which
variant is worth a full n=100 two-model run.

  python scripts/run_contract_pilot.py --seed 4
  python scripts/run_contract_pilot.py --seed 7 --arms CurrentNot H123 H2
  python scripts/run_contract_pilot.py --report
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import threading
from typing import Any, Sequence

from tooldoc_nir.draft_agent_reproduction import (
    CostBudget,
    CostCapReached,
    RELEASED_DECODING,
    _dump_json,
    _load_json,
    load_released_draft_module,
)
from tooldoc_nir.openrouter import (
    OpenRouterClient,
    load_env_file,
    normalize_provider,
)
from tooldoc_nir.provenance import payload_digest
from tooldoc_nir.restbench_adapt import url_index, wrap_instructions, wrap_query
from tooldoc_nir.restbench_contract_variants import (
    EXPERIMENT_ROOT,
    VARIANTS,
    variant_paths,
)
from tooldoc_nir.restbench_data import DEFAULT_DRAFT_ROOT, gold_apis, load_queries
from tooldoc_nir.restbench_http import TmdbClient
from tooldoc_nir.restbench_report import load_traces, mechanics_profile
from tooldoc_nir.restbench_screen import (
    EXCLUDED_PROVIDERS,
    MODELS,
    PROVIDERS,
    scored_rows,
)
from tooldoc_nir.restbench_table import (
    _install_driver,
    _job,
    error_kind,
    is_transport_error,
)

# Real 2/3-hop failures. Excluded on purpose: q030 (the producer is not among
# the candidates), q088 (gold demands a redundant detail call), q089 and q091
# (total-order artifact), q096 (a TV entity with movie gold).
PANEL: tuple[int, ...] = (0, 5, 13, 17, 20, 33, 51, 57, 62, 66)

PILOT_ROOT = Path("artifacts/results/pilot_contract")
RETRY_ROUNDS = 3

# Reference points measured on this panel inside the canonical seed runs, where
# `Ours` is the frozen CurrentNot documentation and `DFSDT` is Initial.
HISTORICAL: dict[tuple[str, int, str], int] = {
    ("ling", 4, "Initial"): 8,
    ("ling", 4, "DRAFT"): 8,
    ("ling", 4, "CurrentNot"): 9,
    ("ling", 7, "Initial"): 10,
    ("ling", 7, "DRAFT"): 8,
    ("ling", 7, "CurrentNot"): 9,
    ("flash", 4, "Initial"): 10,
    ("flash", 4, "DRAFT"): 9,
    ("flash", 4, "CurrentNot"): 2,
    ("flash", 7, "Initial"): 10,
    ("flash", 7, "DRAFT"): 9,
    ("flash", 7, "CurrentNot"): 7,
}


def run_root(
    model_key: str,
    seed: int,
    *,
    pilot_root: Path = PILOT_ROOT,
) -> Path:
    return pilot_root / f"{model_key}_s{seed:02d}"


def load_arm_docs(arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for arm in arms:
        doc_path, _manifest_path = variant_paths(arm)
        if not doc_path.exists():
            raise SystemExit(
                f"{doc_path} is missing; build the variants first with "
                "python -m tooldoc_nir.restbench_contract_variants --probe"
            )
        docs[arm] = _load_json(doc_path)
    return docs


def run_cell(
    *,
    model_key: str,
    seed: int,
    arms: Sequence[str],
    queries: Sequence[int],
    workers: int,
    max_cost_usd: float,
    draft_root: Path,
    pilot_root: Path = PILOT_ROOT,
    panel_provenance: dict[str, Any] | None = None,
    protocol_provenance: dict[str, Any] | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    model = MODELS[model_key]
    root = run_root(model_key, seed, pilot_root=pilot_root)
    root.mkdir(parents=True, exist_ok=True)
    traces_path = root / "traces.jsonl"
    traces_path.unlink(missing_ok=True)

    raw_docs = load_arm_docs(arms)
    wrapped_docs = {
        arm: wrap_instructions(value, category="TMDB")
        for arm, value in raw_docs.items()
    }
    urls = {arm: url_index(value) for arm, value in raw_docs.items()}

    all_queries = load_queries(draft_root, "TMDB")
    cache: dict[str, dict[str, Any]] = {}
    tmdb_key = os.environ.get("TMDB_API_KEY", "")
    if not tmdb_key:
        raise SystemExit("TMDB_API_KEY is missing.")
    http = TmdbClient(tmdb_key, cache=cache)

    client = OpenRouterClient.from_env()
    budget = CostBudget(max_cost_usd)
    budget_lock = threading.Lock()
    draft = load_released_draft_module(draft_root)
    _install_driver(draft)

    def payload(arm: str, index: int) -> dict[str, Any]:
        raw = all_queries[index]
        return {
            "row": arm,
            "agent": "dfsdt",
            "query_index": index,
            "gold": gold_apis(raw),
            "query": wrap_query(raw),
            "dataset": wrapped_docs[arm],
            "model": model,
            "provider": provider,
            "retrieval_num": 5,
            "seed": seed,
            "work": str(root / arm.lower() / f"q{index:03d}"),
            "client": client,
            "budget": budget,
            "budget_lock": budget_lock,
            "http": http,
            "urls": urls[arm],
            "draft": draft,
        }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnostic pilot, not a reported result",
        "model": model,
        "model_key": model_key,
        "provider": provider or "",
        "error_policy": (
            "transport and provider-drift faults retry identically and, if "
            "they persist, invalidate the whole arm block of that query; a "
            "malformed completion or driver failure is a miss and is never "
            "re-rolled"
        ),
        "seed": seed,
        "panel": list(queries),
        "panel_provenance": panel_provenance or {"kind": "inline"},
        "protocol_provenance": protocol_provenance or {"kind": "ad_hoc"},
        "arms": list(arms),
        "decoding": RELEASED_DECODING,
        "retrieval_num": 5,
        "workers": workers,
        "retry_rounds": RETRY_ROUNDS,
        "arm_digests": {
            arm: payload_digest(value) for arm, value in raw_docs.items()
        },
    }
    _dump_json(root / "manifest.json", manifest)

    jobs = [(arm, index) for index in queries for arm in arms]
    random.Random(1000 * seed + len(model_key)).shuffle(jobs)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    lock = threading.Lock()

    def execute(pending: list[tuple[str, int]], label: str) -> list[tuple[str, int]]:
        """Run the given cells and return the ones a retry may still fix.

        Only transport and provider faults come back. A malformed completion
        or a driver failure is what this model did with this documentation, so
        re-rolling it would quietly favour whichever arm fails most often.
        """
        retryable: list[tuple[str, int]] = []
        if not pending:
            return retryable
        done = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_job, payload(arm, index)): (arm, index)
                    for arm, index in pending
                }
                for future in as_completed(futures):
                    key = futures[future]
                    row = future.result()
                    with lock:
                        rows[key] = row
                    if is_transport_error(row):
                        retryable.append(key)
                    done += 1
                    flag = "Y" if row.get("correct_path") else "n"
                    print(
                        f"[{model_key} s{seed:02d} {label}] {done}/{len(pending)} "
                        f"{key[0]} q{key[1]:03d} CP={flag} "
                        f"${row.get('cost_usd')} "
                        f"{error_kind(row)} {(row.get('error') or '')[:60]}",
                        flush=True,
                    )
        except CostCapReached as exc:
            print(f"STOP cost-cap: {exc}", flush=True)
        return retryable

    failed = execute(jobs, "pass1")
    for attempt in range(1, RETRY_ROUNDS + 1):
        if not failed:
            break
        print(
            f"[{model_key} s{seed:02d}] retry {attempt}/{RETRY_ROUNDS} on "
            f"{len(failed)} transport faults",
            flush=True,
        )
        failed = execute(failed, f"retry{attempt}")

    ordered = [rows[key] for key in sorted(rows, key=lambda item: (item[0], item[1]))]
    invalidated = sorted({index for _arm, index in failed})
    for row in ordered:
        if int(row["query_index"]) in invalidated:
            row["invalidated"] = True
    traces_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    scored = [row for row in ordered if not row.get("invalidated")]
    summary = {
        "manifest": manifest,
        "cost_usd": round(budget.spent_usd, 6),
        # A query whose transport never cleared is dropped from every arm, not
        # just the arm that happened to hit the failure: a block missing one
        # arm cannot contribute to a paired contrast.
        "invalidated_queries": invalidated,
        "unrecovered_transport": [list(key) for key in sorted(failed)],
        "driver_failures": [
            [row["row"], row["query_index"]]
            for row in scored
            if row.get("error")
        ],
        "arms": arm_scores(scored, arms),
    }
    _dump_json(root / "summary.json", summary)
    return summary


def arm_scores(
    rows: Sequence[dict[str, Any]], arms: Sequence[str]
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    rows = scored_rows(rows)
    for arm in arms:
        subset = [row for row in rows if row.get("row") == arm]
        if not subset:
            continue
        profile = mechanics_profile(subset)
        scores[arm] = {
            "n": len(subset),
            "cp": profile["flags"].get("correct_path", 0),
            "errors": profile["flags"].get("driver_error", 0),
            "missing_first_producer": profile["missing_first_producer"],
            "missing_final_consumer": profile["missing_final_consumer"],
            "wrong_first": profile["flags"].get("wrong_first", 0),
            "repeat_producer": profile["flags"].get("repeat_producer", 0),
            "parameter_or_http_error": profile["flags"].get(
                "parameter_or_http_error", 0
            ),
        }
    return scores


def _hit(row: dict[str, Any]) -> bool:
    return bool(row.get("correct_path")) and not row.get("error")


def _cell_key(row: dict[str, Any]) -> tuple[Any, int]:
    """Seed + query, so pooling two replicates does not overwrite the first."""
    return (row.get("seed"), int(row["query_index"]))


def paired_net(
    rows: Sequence[dict[str, Any]], arm: str, baseline: str
) -> dict[str, Any]:
    left = {_cell_key(row): _hit(row) for row in rows if row.get("row") == arm}
    right = {
        _cell_key(row): _hit(row) for row in rows if row.get("row") == baseline
    }
    shared = sorted(set(left) & set(right))
    wins = sum(1 for key in shared if left[key] and not right[key])
    losses = sum(1 for key in shared if right[key] and not left[key])
    return {
        "baseline": baseline,
        "n": len(shared),
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def reproduction_check(
    model_key: str,
    seed: int,
    panel: Sequence[int],
    *,
    pilot_root: Path = PILOT_ROOT,
) -> dict[str, Any]:
    """Re-measure the frozen documentation against its own canonical run.

    `CurrentNot` in the pilot and `Ours` in the canonical seed run are the same
    payload, the same seed and the same model id, so any gap between them is
    the served endpoint moving under us, not a documentation effect. Recording
    it here keeps the pilot honest about what its baselines mean.
    """
    canonical = Path(f"artifacts/results/restbench_tmdb_{model_key}_s{seed:02d}")
    pilot = run_root(model_key, seed, pilot_root=pilot_root)
    canonical_traces = canonical / "traces.jsonl"
    pilot_traces = pilot / "traces.jsonl"
    if not canonical_traces.exists() or not pilot_traces.exists():
        return {}
    wanted = set(panel)

    def hits(path: Path, row: str) -> dict[int, bool]:
        return {
            int(item["query_index"]): bool(item.get("correct_path"))
            and not item.get("error")
            for item in load_traces(path)
            if item.get("row") == row and int(item["query_index"]) in wanted
        }

    before = hits(canonical_traces, "Ours")
    after = hits(pilot_traces, "CurrentNot")
    shared = sorted(set(before) & set(after))
    frozen = payload_digest(_load_json(Path("artifacts/documentation/TMDB_Ours.json")))
    pilot_manifest = _load_json(pilot / "manifest.json")
    canonical_manifest = _load_json(canonical / "manifest.json")
    return {
        "documentation_identical": frozen
        == pilot_manifest["arm_digests"].get("CurrentNot"),
        "payload_digest": frozen,
        "canonical_run": str(canonical),
        "canonical_created_at": canonical_manifest.get("created_at"),
        "pilot_created_at": pilot_manifest.get("created_at"),
        "canonical_cp": sum(1 for index in shared if before[index]),
        "pilot_cp": sum(1 for index in shared if after[index]),
        "n": len(shared),
        "recovered": [index for index in shared if after[index] and not before[index]],
        "lost": [index for index in shared if before[index] and not after[index]],
    }


def collect(
    seeds: Sequence[int],
    arms: Sequence[str],
    *,
    panel: Sequence[int] = PANEL,
    pilot_root: Path = PILOT_ROOT,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for model_key in MODELS:
        pooled: list[dict[str, Any]] = []
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            path = (
                run_root(model_key, seed, pilot_root=pilot_root) / "traces.jsonl"
            )
            if not path.exists():
                continue
            rows = scored_rows(load_traces(path))
            pooled.extend(rows)
            per_seed[f"s{seed:02d}"] = {
                "arms": arm_scores(rows, arms),
                "historical": (
                    {
                        arm: HISTORICAL[(model_key, seed, arm)]
                        for arm in arms
                        if (model_key, seed, arm) in HISTORICAL
                    }
                    if tuple(panel) == PANEL
                    else {}
                ),
                "reproduction_check": reproduction_check(
                    model_key,
                    seed,
                    panel,
                    pilot_root=pilot_root,
                ),
            }
        if not pooled:
            continue
        report[model_key] = {
            "seeds": per_seed,
            "pooled": arm_scores(pooled, arms),
            "paired_vs_currentnot": {
                arm: paired_net(pooled, arm, "CurrentNot")
                for arm in arms
                if arm != "CurrentNot"
            },
            "paired_vs_draft": {
                arm: paired_net(pooled, arm, "DRAFT")
                for arm in arms
                if arm != "DRAFT"
            },
        }
    return report


def gate(
    report: dict[str, Any],
    arms: Sequence[str],
    *,
    expected_per_model: int = 20,
    eligible_arms: Sequence[str] | None = None,
    strict_positive_vs_draft: bool = False,
    composition: tuple[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Preregistered promotion gate over all panel cells per model.

    The original floors generalize as Ling >=85% and Flash >=90%, with no
    negative paired net against either contemporaneous baseline and no more
    missing consumers than CurrentNot. An arm with a residual top-level error
    after three retries is indeterminate, not bad text.
    """
    thresholds = {
        "ling": math.ceil(0.85 * expected_per_model),
        "flash": math.ceil(0.90 * expected_per_model),
    }
    eligible = set(eligible_arms) if eligible_arms is not None else set(arms)
    verdicts: dict[str, Any] = {}
    for arm in arms:
        reasons: list[str] = []
        promoted = True
        if arm not in eligible:
            verdicts[arm] = {
                "promoted": False,
                "reasons": ["baseline or secondary control, not a primary candidate"],
            }
            continue
        for model_key, floor in thresholds.items():
            model = report.get(model_key)
            if model is None or arm not in model["pooled"]:
                promoted = False
                reasons.append(f"{model_key}: not run")
                continue
            scores = model["pooled"][arm]
            if scores["n"] < expected_per_model:
                promoted = False
                reasons.append(
                    f"{model_key}: {scores['n']}/{expected_per_model} cells only"
                )
            if scores["errors"]:
                promoted = False
                reasons.append(
                    f"{model_key}: {scores['errors']} indeterminate after retries"
                )
            if scores["cp"] < floor:
                promoted = False
                reasons.append(
                    f"{model_key}: CP {scores['cp']}/{scores['n']} below {floor}"
                )
            current_net = (model.get("paired_vs_currentnot") or {}).get(arm)
            if current_net and current_net["net"] < 0:
                promoted = False
                reasons.append(
                    f"{model_key}: net {current_net['net']} "
                    f"vs {current_net['baseline']}"
                )
            draft_net = (model.get("paired_vs_draft") or {}).get(arm)
            if draft_net and (
                draft_net["net"] < 0
                or (strict_positive_vs_draft and draft_net["net"] == 0)
            ):
                promoted = False
                comparator = "not positive" if strict_positive_vs_draft else "negative"
                reasons.append(
                    f"{model_key}: net {draft_net['net']} vs DRAFT is {comparator}"
                )
            current = model["pooled"].get("CurrentNot")
            if (
                current
                and scores["missing_final_consumer"] > current["missing_final_consumer"]
            ):
                promoted = False
                reasons.append(
                    f"{model_key}: missing-consumer {scores['missing_final_consumer']} "
                    f"above CurrentNot {current['missing_final_consumer']}"
                )
        verdicts[arm] = {"promoted": promoted, "reasons": reasons}

    if composition is not None:
        composed, components = composition
        verdict = verdicts.get(composed)
        if verdict is not None and composed in eligible:
            for model_key, model in report.items():
                pooled = model.get("pooled") or {}
                if composed not in pooled or any(name not in pooled for name in components):
                    continue
                component_best = max(pooled[name]["cp"] for name in components)
                if pooled[composed]["cp"] < component_best - 1:
                    verdict["promoted"] = False
                    verdict["reasons"].append(
                        f"{model_key}: {composed} CP {pooled[composed]['cp']} is "
                        f"more than one below component best {component_best}"
                    )
    return verdicts


def interpretation(report: dict[str, Any], arms: Sequence[str]) -> dict[str, Any]:
    """State plainly how much the panel can still discriminate, per model.

    If the frozen documentation no longer reproduces its canonical CP, the
    panel's failure modes have gone away with the endpoint and a high score is
    not evidence that any hypothesis fixed anything.
    """
    notes: dict[str, Any] = {}
    for model_key, block in report.items():
        pooled = block["pooled"]
        cps = {arm: pooled[arm]["cp"] for arm in pooled}
        cells = max((value["n"] for value in pooled.values()), default=0)
        saturated = sum(1 for value in cps.values() if value == cells)
        drift = []
        for seed_key, cell in block["seeds"].items():
            check = cell.get("reproduction_check") or {}
            if not check:
                continue
            if check["pilot_cp"] != check["canonical_cp"]:
                drift.append(
                    {
                        "seed": seed_key,
                        "canonical_cp": check["canonical_cp"],
                        "pilot_cp": check["pilot_cp"],
                        "recovered": check["recovered"],
                        "lost": check["lost"],
                        "documentation_identical": check["documentation_identical"],
                    }
                )
        notes[model_key] = {
            "cells": cells,
            "cp_spread": [min(cps.values(), default=0), max(cps.values(), default=0)],
            "arms_at_ceiling": saturated,
            "baseline_drift": drift,
            "discriminative": saturated <= 2 and not drift,
        }
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument(
        "--arms", nargs="+", default=[spec.name for spec in VARIANTS]
    )
    parser.add_argument("--queries", nargs="+", type=int)
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        help="Frozen structural-panel artifact; its indices replace --queries.",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        help="Preregistered arms, seeds, models, workers and payload digests.",
    )
    parser.add_argument("--output-root", type=Path, default=PILOT_ROOT)
    parser.add_argument(
        "--providers",
        nargs="*",
        default=[],
        metavar="MODEL=PROVIDER",
        help="Pin one OpenRouter provider per model, e.g. ling=novita. "
        "Defaults to the qualified pins in PROVIDERS.",
    )
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-cost-usd", type=float, default=1.50)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report-seeds", nargs="+", type=int, default=[4, 7])
    args = parser.parse_args()

    panel_provenance: dict[str, Any] = {"kind": "inline"}
    if args.panel_manifest is not None:
        panel_manifest = _load_json(args.panel_manifest)
        stable_panel = {
            key: value
            for key, value in panel_manifest.items()
            if key not in {"frozen_at", "git_commit", "selection_digest"}
        }
        recorded_digest = panel_manifest.get("selection_digest")
        if recorded_digest != payload_digest(stable_panel):
            raise SystemExit(f"{args.panel_manifest} has an invalid selection digest")
        frozen_queries = [
            int(index)
            for index in (panel_manifest.get("selection") or {}).get("indices", [])
        ]
        if not frozen_queries:
            raise SystemExit(
                f"{args.panel_manifest} does not contain selection.indices"
            )
        if args.queries is not None and list(args.queries) != frozen_queries:
            raise SystemExit("--queries does not match the frozen panel manifest")
        queries = frozen_queries
        panel_provenance = {
            "kind": "frozen_manifest",
            "path": str(args.panel_manifest),
            "selection_digest": recorded_digest,
        }
    else:
        queries = list(args.queries) if args.queries is not None else list(PANEL)

    protocol_provenance: dict[str, Any] = {"kind": "ad_hoc"}
    if args.protocol_manifest is not None:
        protocol = _load_json(args.protocol_manifest)
        protocol_arms = protocol.get("arms") or {}
        expected_digests: dict[str, str] = {}
        for group in ("baselines", "primary_candidates", "secondary_control"):
            expected_digests.update(protocol_arms.get(group) or {})
        if list(args.arms) != list(expected_digests):
            raise SystemExit(
                f"--arms must match the protocol order: {list(expected_digests)}"
            )
        actual_digests = {
            arm: payload_digest(docs)
            for arm, docs in load_arm_docs(args.arms).items()
        }
        if actual_digests != expected_digests:
            raise SystemExit("variant payload digests do not match the protocol")
        expected_seeds = [int(seed) for seed in protocol.get("seeds") or []]
        if args.report:
            if list(args.report_seeds) != expected_seeds:
                raise SystemExit(
                    f"--report-seeds must match the protocol: {expected_seeds}"
                )
        elif args.seed not in expected_seeds:
            raise SystemExit(f"--seed must be one of the protocol seeds: {expected_seeds}")
        expected_models = set((protocol.get("models") or {}).keys())
        if not set(args.models).issubset(expected_models):
            raise SystemExit(f"--models must be a subset of {sorted(expected_models)}")
        if args.workers != int(protocol.get("workers", -1)):
            raise SystemExit(f"--workers must equal protocol value {protocol.get('workers')}")
        expected_panel_digest = (protocol.get("panel") or {}).get("selection_digest")
        if expected_panel_digest != panel_provenance.get("selection_digest"):
            raise SystemExit("panel selection digest does not match the protocol")
        protocol_provenance = {
            "kind": "preregistered_manifest",
            "path": str(args.protocol_manifest),
            "payload_digest": payload_digest(protocol),
        }

    if args.report:
        report = collect(
            args.report_seeds,
            args.arms,
            panel=queries,
            pilot_root=args.output_root,
        )
        protocol_arms = protocol.get("arms") if args.protocol_manifest else {}
        primary_candidates = list((protocol_arms or {}).get("primary_candidates") or {})
        payload = {
            "report": report,
            "gate": gate(
                report,
                args.arms,
                expected_per_model=len(queries) * len(args.report_seeds),
                eligible_arms=primary_candidates or None,
                strict_positive_vs_draft=bool(primary_candidates),
                composition=("H12", ("H1", "H2"))
                if "H12" in primary_candidates
                else None,
            ),
            "interpretation": interpretation(report, args.arms),
            "panel": queries,
            "panel_provenance": panel_provenance,
            "protocol_provenance": protocol_provenance,
        }
        _dump_json(args.output_root / "pilot_report.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.seed is None:
        raise SystemExit("pass --seed 4 (or 7), or --report")
    load_env_file()
    unknown = [name for name in args.models if name not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; have {list(MODELS)}")
    pinned = dict(PROVIDERS)
    for item in args.providers:
        model_key, separator, provider = item.partition("=")
        if not separator or model_key not in MODELS:
            raise SystemExit(f"--providers takes MODEL=PROVIDER, got {item!r}")
        if normalize_provider(provider) in EXCLUDED_PROVIDERS:
            raise SystemExit(
                f"{provider} is excluded: "
                f"{EXCLUDED_PROVIDERS[normalize_provider(provider)]}"
            )
        pinned[model_key] = provider
    if not EXPERIMENT_ROOT.exists():
        raise SystemExit(f"{EXPERIMENT_ROOT} is missing; build the variants first.")
    for model_key in args.models:
        summary = run_cell(
            model_key=model_key,
            seed=args.seed,
            arms=args.arms,
            queries=queries,
            workers=args.workers,
            max_cost_usd=args.max_cost_usd,
            draft_root=args.draft_root,
            pilot_root=args.output_root,
            panel_provenance=panel_provenance,
            protocol_provenance=protocol_provenance,
            provider=pinned.get(model_key),
        )
        print(
            json.dumps(
                {
                    "model": model_key,
                    "seed": args.seed,
                    "cost_usd": summary["cost_usd"],
                    "arms": {
                        arm: value["cp"] for arm, value in summary["arms"].items()
                    },
                    "indeterminate": summary["indeterminate"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
