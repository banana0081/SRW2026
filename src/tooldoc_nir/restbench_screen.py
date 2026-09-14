"""Paired statistics and preregistered gates for the two-stage positive screen.

Everything here is offline: it reads trace rows and returns numbers, so the
gate that decides whether a documentation variant is promoted can be tested
without spending a token.

Three things this fixes relative to the earlier pilot arithmetic:

1. `DRAFT` is not the only baseline. A variant that beats DRAFT but loses to
   `Initial` has not shown that the documentation helps, so every contrast is
   computed against both, on both models -- four contrasts per candidate.
2. A seed-level interval over three to ten seeds is a t interval, not a normal
   one. `1.96 * SE` at four seeds understates the width by roughly 60%.
3. A query answered under several seeds is one cluster, not several
   independent observations. The promotion test averages replicates inside a
   query and permutes labels at the query level.
"""

from __future__ import annotations

from collections import defaultdict
import ast
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import t as _student_t

MODELS: dict[str, str] = {
    "ling": "inclusionai/ling-3.0-flash",
    "flash": "deepseek/deepseek-v4-flash-0731",
}

# Pinned by scripts/qualify_providers.py on transport evidence only, never on
# benchmark outcomes. The same model id served by two providers is two
# measurements, and the canonical runs drifted by up to eight queries between
# re-runs of identical documentation because of it.
#
#   ling  -> novita: 12/12 valid JSON, 0% truncated, 2.2 s median.
#            DeepInfra also passed but is slower.
#   flash -> wafer:  18/18 valid JSON, 0% truncated, 6.8 s median.
#            DigitalOcean returned parseable JSON on 44% of calls at a 39 s
#            median; DeepInfra failed every call at concurrency 15.
PROVIDERS: dict[str, str] = {
    "ling": "novita",
    "flash": "wafer",
}

# Ruled out before measuring, on the transport evidence already in the usage
# logs of the canonical runs.
EXCLUDED_PROVIDERS: dict[str, str] = {
    "sailresearch": "29.1% of its completions finished on the token limit",
}

# The Screen A / Promotion B thresholds, in percentage points, exactly as
# preregistered. They live here so the protocol artifact and the gate cannot
# drift apart.
SCREEN_MIN_EFFECT_PP = 2.0
SCREEN_MAX_REPLICATE_LOSS_PP = -2.0
PROMOTION_MIN_EFFECT_PP = 3.0
PROMOTION_MIN_POSITIVE_SEEDS = 2
PROMOTION_MAX_P_VALUE = 0.05
PERMUTATION_DRAWS = 20000
PERMUTATION_EXACT_LIMIT = 18


def scored_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop only the cells that carry no model result at all.

    A query invalidated by an unrecovered transport fault has no result in one
    or more arms, so it cannot enter a paired contrast. A driver failure is a
    result and stays in, as a miss.
    """
    return [dict(row) for row in rows if not row.get("invalidated")]


def hit(row: Mapping[str, Any]) -> bool:
    return bool(row.get("correct_path")) and not row.get("error")


# Live HTTP can fail after a well-formed fill: no player, a 403 scope, a 429.
# Those are not a missing or invented identifier. A missing path slot, an
# invalid base62 id, or a required identifier the agent never emitted is.
_FILL_ENV_MARKERS = (
    "no_active_device",
    "no active device",
    "quota_exceeded",
    "too many requests",
    "premium_required",
    "restriction_violated",
)
_REQUIRED_PARAM = re.compile(
    r"(?:required|missing) parameter ([a-z0-9_]+)", re.IGNORECASE
)
ARM_RECORD_DIRS = {
    "DFSDT": "dfsdt",
    "DRAFT": "draft",
    "Ours": "ours",
    "ReAct": "react",
}


def parse_call_result(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_execute_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def iter_executed_calls(
    record: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    log = record.get("execute_log") or {}
    groups = log.get("api_result_ls") or []
    results = log.get("call_result_ls") or []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group, raw in zip(groups, results):
        result = parse_call_result(raw)
        apis = group if isinstance(group, list) else [group]
        apis = [api for api in apis if isinstance(api, dict)]
        if not apis:
            continue
        if len(apis) == 1:
            pairs.append((apis[0], result))
        else:
            for api in apis:
                pairs.append((api, result))
    return pairs


def parameter_was_emitted(api: Mapping[str, Any], name: str) -> bool:
    params = api.get("parameters") or {}
    if not isinstance(params, dict):
        return False
    wanted = name.lower()
    for key, value in params.items():
        if str(key).lower() != wanted:
            continue
        if value is None or value == "":
            return False
        return True
    return False


def call_fill_rejected(
    api: Mapping[str, Any], result: Mapping[str, Any]
) -> bool:
    """True when the live call failed because a parameter was missing or invented."""
    error = str(result.get("error") or "")
    body = str(result.get("response") or "")
    if not error:
        return False
    blob = f"{error}\n{body}".lower()
    if any(marker in blob for marker in _FILL_ENV_MARKERS):
        return False
    status = error.strip().upper()
    if status in {"HTTP 403", "HTTP 429", "HTTP 405"}:
        return False
    if status == "HTTP 404" and not body.strip():
        return False
    if "missing path parameter" in blob:
        return True
    if "invalid base62" in blob:
        return True
    if "resource not found" in blob:
        return True
    required = _REQUIRED_PARAM.search(f"{error}\n{body}")
    if required:
        return not parameter_was_emitted(api, required.group(1))
    if "no type given" in blob:
        return not parameter_was_emitted(api, "type")
    if "no search query" in blob:
        return not (
            parameter_was_emitted(api, "q") or parameter_was_emitted(api, "query")
        )
    if "invalid limit" in blob:
        return True
    if status == "HTTP 400":
        return True
    if status == "HTTP 404":
        return True
    return False


def fill_error_count(
    record: Mapping[str, Any], gold: Sequence[str] | None = None
) -> int:
    gold_names = {str(name) for name in gold or [] if name}
    count = 0
    for api, result in iter_executed_calls(record):
        name = str(api.get("api_name") or api.get("tool_name") or "")
        if gold_names and name not in gold_names:
            continue
        if call_fill_rejected(api, result):
            count += 1
    return count


def attach_fill_errors(
    rows: Sequence[Mapping[str, Any]], run_root: Path
) -> list[dict[str, Any]]:
    """Join traces.jsonl rows to per-query execute logs and count fill rejects."""
    attached: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        folder = ARM_RECORD_DIRS.get(str(row.get("row") or ""))
        if folder is not None:
            record = load_execute_record(
                run_root
                / folder
                / f"q{int(row['query_index']):03d}"
                / "record.jsonl"
            )
            if record is not None:
                item["fill_errors"] = fill_error_count(record, row.get("gold") or [])
        attached.append(item)
    return attached


def execution_valid_hit(row: Mapping[str, Any]) -> bool:
    """CP restricted to paths the agent executed without a fill reject.

    The gold sequence was reached and no call on that path was rejected
    because a parameter was missing or invented. A later HTTP failure on a
    well-formed id (player, scope, quota) is still a hit.
    """
    if not hit(row):
        return False
    if "id_errors" in row:
        return int(row.get("id_errors") or 0) == 0
    if "fill_errors" in row:
        return int(row.get("fill_errors") or 0) == 0
    return int(row.get("http_errors") or 0) == 0


def cell_hits(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    metric: str = "cp",
) -> dict[tuple[Any, int], bool]:
    """One arm's outcomes keyed by (seed, query), so replicates never merge."""
    score = execution_valid_hit if metric == "execution_valid_cp" else hit
    return {
        (row.get("seed"), int(row["query_index"])): score(row)
        for row in rows
        if row.get("row") == arm
    }


def paired_effect(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    baseline: str,
    *,
    metric: str = "cp",
) -> dict[str, Any]:
    """Wins, losses and the effect in percentage points over shared cells."""
    left = cell_hits(rows, arm, metric=metric)
    right = cell_hits(rows, baseline, metric=metric)
    shared = sorted(set(left) & set(right), key=lambda key: (str(key[0]), key[1]))
    wins = sum(1 for key in shared if left[key] and not right[key])
    losses = sum(1 for key in shared if right[key] and not left[key])
    return {
        "arm": arm,
        "baseline": baseline,
        "n": len(shared),
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "effect_pp": round(100.0 * (wins - losses) / len(shared), 2) if shared else None,
    }


def per_seed_effects(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    baseline: str,
    *,
    metric: str = "cp",
) -> dict[Any, dict[str, Any]]:
    by_seed: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row.get("seed")].append(row)
    return {
        seed: paired_effect(subset, arm, baseline, metric=metric)
        for seed, subset in sorted(by_seed.items(), key=lambda item: str(item[0]))
    }


def query_differences(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    baseline: str,
    *,
    metric: str = "cp",
) -> dict[int, float]:
    """Average the replicates of one query, then difference the two arms.

    Three seeds of the same question are three looks at one query, not three
    questions. Collapsing them first is what makes the permutation below a
    query-level test.
    """
    left = cell_hits(rows, arm, metric=metric)
    right = cell_hits(rows, baseline, metric=metric)
    grouped: dict[int, list[float]] = defaultdict(list)
    for (seed, index), value in left.items():
        if (seed, index) in right:
            grouped[index].append(float(value) - float(right[(seed, index)]))
    return {
        index: sum(values) / len(values)
        for index, values in sorted(grouped.items())
        if values
    }


def sign_flip_p_value(
    differences: Sequence[float],
    *,
    draws: int = PERMUTATION_DRAWS,
    seed: int = 20260909,
) -> dict[str, Any]:
    """One-sided label-swap test on per-query differences.

    Swapping the arm labels inside a query flips the sign of that query's
    difference and leaves everything else alone, so the null distribution is
    the set of sign assignments. Exact while it is cheap, Monte Carlo with a
    fixed stream after that.
    """
    values = [value for value in differences if value != 0.0]
    observed = sum(differences)
    count = len(values)
    if count == 0:
        return {
            "queries": len(differences),
            "informative": 0,
            "observed": round(observed, 4),
            "p_value": 1.0,
            "method": "degenerate",
        }
    if count <= PERMUTATION_EXACT_LIMIT:
        total = 0
        extreme = 0
        for signs in itertools.product((1.0, -1.0), repeat=count):
            total += 1
            if sum(sign * value for sign, value in zip(signs, values)) >= observed:
                extreme += 1
        return {
            "queries": len(differences),
            "informative": count,
            "observed": round(observed, 4),
            "p_value": round(extreme / total, 5),
            "method": f"exact over 2^{count}",
        }
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        total = 0.0
        for value in values:
            total += value if rng.random() < 0.5 else -value
        if total >= observed:
            extreme += 1
    return {
        "queries": len(differences),
        "informative": count,
        "observed": round(observed, 4),
        # Add-one, so a test that never saw the observed value still reports a
        # bound rather than an impossible zero.
        "p_value": round((extreme + 1) / (draws + 1), 5),
        "method": f"monte carlo, {draws} draws, seed {seed}",
    }


def student_t_interval(
    values: Sequence[float], *, confidence: float = 0.95
) -> dict[str, Any]:
    """Mean with a Student-t interval over replicates.

    The seed-level interval used to be `mean +- 1.96 * SE`, which is the
    large-sample form. With four or ten seeds the t quantile is the right one
    and the interval is materially wider.
    """
    count = len(values)
    if count == 0:
        return {"replicates": 0}
    mean = sum(values) / count
    if count == 1:
        return {
            "replicates": 1,
            "mean": round(mean, 4),
            "values": [round(mean, 4)],
        }
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    standard_error = math.sqrt(variance / count)
    quantile = float(_student_t.ppf(0.5 + confidence / 2.0, count - 1))
    half = quantile * standard_error
    return {
        "replicates": count,
        "mean": round(mean, 4),
        "sd": round(math.sqrt(variance), 4),
        "se": round(standard_error, 4),
        "t_quantile": round(quantile, 4),
        "ci95": [round(mean - half, 4), round(mean + half, 4)],
        "positive": bool(mean - half > 0.0),
        "values": [round(value, 4) for value in values],
    }


def contrast_grid(
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    arm: str,
    baselines: Sequence[str],
    *,
    metric: str = "cp",
) -> dict[str, dict[str, Any]]:
    """The (model x baseline) contrasts that a candidate has to satisfy."""
    grid: dict[str, dict[str, Any]] = {}
    for model_key, rows in per_model_rows.items():
        for baseline in baselines:
            pooled = paired_effect(rows, arm, baseline, metric=metric)
            grid[f"{model_key}:{baseline}"] = {
                **pooled,
                "model": model_key,
                "per_seed": {
                    str(seed): value
                    for seed, value in per_seed_effects(
                        rows, arm, baseline, metric=metric
                    ).items()
                },
            }
    return grid


def screen_verdict(
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    arm: str,
    *,
    baselines: Sequence[str] = ("DRAFT", "Initial"),
    incumbent: str = "CurrentNot",
    models: Sequence[str] = ("ling", "flash"),
    expected_cells: int,
    surfaces_changed: int = 0,
) -> dict[str, Any]:
    """Screen A: is this candidate worth spending the promotion panel on?

    The screen score is the *worst* of the four contrasts, because the whole
    point of the plan is a variant that helps both models against both
    baselines. Averaging would let a large Ling gain pay for a Flash loss,
    which is the failure mode that produced H2.
    """
    grid = contrast_grid(per_model_rows, arm, baselines)
    reasons: list[str] = []

    missing = [
        f"{model_key}:{baseline}"
        for model_key in models
        for baseline in baselines
        if grid.get(f"{model_key}:{baseline}", {}).get("effect_pp") is None
    ]
    if missing:
        reasons.append(f"no paired data for {sorted(missing)}")

    for label, contrast in sorted(grid.items()):
        if contrast["n"] < expected_cells:
            reasons.append(
                f"{label}: {contrast['n']}/{expected_cells} paired cells"
            )
        for seed, replicate in sorted(contrast["per_seed"].items()):
            effect = replicate.get("effect_pp")
            if effect is not None and effect < SCREEN_MAX_REPLICATE_LOSS_PP:
                reasons.append(
                    f"{label} seed {seed}: {effect:+.1f} pp is below "
                    f"{SCREEN_MAX_REPLICATE_LOSS_PP:+.1f}"
                )

    effects = [
        contrast["effect_pp"]
        for contrast in grid.values()
        if contrast["effect_pp"] is not None
    ]
    score = min(effects) if effects and not missing else None
    if score is not None and score < SCREEN_MIN_EFFECT_PP:
        reasons.append(
            f"screen score {score:+.1f} pp is below {SCREEN_MIN_EFFECT_PP:+.1f}"
        )

    incumbent_nets: dict[str, Any] = {}
    for model_key, rows in per_model_rows.items():
        if arm == incumbent:
            continue
        against = paired_effect(rows, arm, incumbent)
        incumbent_nets[model_key] = against
        if against["n"] and against["net"] < 0:
            reasons.append(
                f"{model_key}: net {against['net']} against {incumbent}"
            )

    consumers: dict[str, Any] = {}
    for model_key, rows in per_model_rows.items():
        mine = _missing_consumers(rows, arm)
        theirs = _missing_consumers(rows, incumbent)
        consumers[model_key] = {"arm": mine, incumbent: theirs}
        if mine is not None and theirs is not None and mine > theirs:
            reasons.append(
                f"{model_key}: {mine} missing consumers against "
                f"{incumbent}'s {theirs}"
            )

    return {
        "arm": arm,
        "screen_score_pp": score,
        "summed_effect_pp": round(sum(effects), 2) if effects else None,
        "surfaces_changed": surfaces_changed,
        "contrasts": grid,
        f"paired_vs_{incumbent}": incumbent_nets,
        "missing_final_consumer": consumers,
        "eligible": not reasons,
        "reasons": reasons,
    }


def _missing_consumers(
    rows: Sequence[Mapping[str, Any]], arm: str
) -> int | None:
    from tooldoc_nir.restbench_report import mechanics_profile

    subset = [row for row in rows if row.get("row") == arm]
    if not subset:
        return None
    return int(mechanics_profile(subset)["missing_final_consumer"])


def screen_gate(
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[str],
    *,
    expected_cells: int,
    surfaces: Mapping[str, int] | None = None,
    baselines: Sequence[str] = ("DRAFT", "Initial"),
    incumbent: str = "CurrentNot",
    models: Sequence[str] = ("ling", "flash"),
) -> dict[str, Any]:
    """Score every candidate and promote at most one.

    Tie-break order is fixed in advance: worst-contrast score, then the summed
    effect, then the smaller number of rewritten surfaces. Deciding this after
    seeing the numbers is how a screen turns into a selection artifact.
    """
    verdicts = {
        arm: screen_verdict(
            per_model_rows,
            arm,
            baselines=baselines,
            incumbent=incumbent,
            models=models,
            expected_cells=expected_cells,
            surfaces_changed=int((surfaces or {}).get(arm, 0)),
        )
        for arm in candidates
    }
    eligible = [arm for arm in candidates if verdicts[arm]["eligible"]]
    eligible.sort(
        key=lambda arm: (
            -float(verdicts[arm]["screen_score_pp"] or 0.0),
            -float(verdicts[arm]["summed_effect_pp"] or 0.0),
            int(verdicts[arm]["surfaces_changed"]),
            arm,
        )
    )
    winner = eligible[0] if eligible else None
    return {
        "stage": "screen_a",
        "thresholds": {
            "min_screen_score_pp": SCREEN_MIN_EFFECT_PP,
            "max_replicate_loss_pp": SCREEN_MAX_REPLICATE_LOSS_PP,
            "baselines": list(baselines),
            "incumbent": incumbent,
            "expected_cells": expected_cells,
        },
        "verdicts": verdicts,
        "eligible": eligible,
        "promoted": winner,
        "decision": (
            f"promote {winner} to Promotion B"
            if winner
            else "stop TMDB tuning: no candidate cleared Screen A"
        ),
    }


def promotion_verdict(
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    arm: str,
    *,
    baselines: Sequence[str] = ("DRAFT", "Initial"),
    models: Sequence[str] = ("ling", "flash"),
    expected_queries: int,
) -> dict[str, Any]:
    """Promotion B: the confirmation on queries no variant has ever seen."""
    contrasts: dict[str, Any] = {}
    reasons: list[str] = []
    for model_key in models:
        rows = per_model_rows.get(model_key)
        if not rows:
            reasons.append(f"{model_key}: not run")
            continue
        for baseline in baselines:
            label = f"{model_key}:{baseline}"
            pooled = paired_effect(rows, arm, baseline)
            differences = query_differences(rows, arm, baseline)
            permutation = sign_flip_p_value(list(differences.values()))
            seeds = per_seed_effects(rows, arm, baseline)
            positive_seeds = sum(
                1 for value in seeds.values() if (value["net"] or 0) > 0
            )
            contrasts[label] = {
                **pooled,
                "queries": len(differences),
                "query_mean_pp": (
                    round(100.0 * sum(differences.values()) / len(differences), 2)
                    if differences
                    else None
                ),
                "permutation": permutation,
                "per_seed": {str(seed): value for seed, value in seeds.items()},
                "positive_seeds": positive_seeds,
                "seeds": len(seeds),
            }
            if len(differences) < expected_queries:
                reasons.append(
                    f"{label}: {len(differences)}/{expected_queries} queries"
                )
            effect = pooled["effect_pp"]
            if effect is None or effect < PROMOTION_MIN_EFFECT_PP:
                reasons.append(
                    f"{label}: {effect} pp is below "
                    f"{PROMOTION_MIN_EFFECT_PP:+.1f}"
                )
            if positive_seeds < PROMOTION_MIN_POSITIVE_SEEDS:
                reasons.append(
                    f"{label}: positive net in {positive_seeds}/{len(seeds)} "
                    f"replicates, needs {PROMOTION_MIN_POSITIVE_SEEDS}"
                )
            if permutation["p_value"] > PROMOTION_MAX_P_VALUE:
                reasons.append(
                    f"{label}: one-sided p={permutation['p_value']} above "
                    f"{PROMOTION_MAX_P_VALUE}"
                )
    return {
        "stage": "promotion_b",
        "arm": arm,
        "thresholds": {
            "min_effect_pp": PROMOTION_MIN_EFFECT_PP,
            "min_positive_replicates": PROMOTION_MIN_POSITIVE_SEEDS,
            "max_one_sided_p": PROMOTION_MAX_P_VALUE,
            "expected_queries": expected_queries,
        },
        "contrasts": contrasts,
        "confirmed": not reasons and bool(contrasts),
        "reasons": reasons,
    }


def paper_gate(
    seed_effects: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """The final table's criterion: four paired t intervals strictly above zero.

    `seed_effects` maps a `model:baseline` label to that contrast's per-seed
    paired deltas over the full 100-query run.
    """
    intervals = {
        label: student_t_interval(list(values))
        for label, values in sorted(seed_effects.items())
    }
    failing = [
        label
        for label, interval in intervals.items()
        if interval.get("positive") is not True
    ]
    return {
        "intervals": intervals,
        "contrasts": sorted(intervals),
        "failing": failing,
        "met": bool(intervals) and not failing,
    }
