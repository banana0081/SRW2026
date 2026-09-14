"""Freeze the Cross-model Positive Scope experiment before the first model call.

Three artifacts, written in one command and never rewritten:

- `TMDB_screen_a_panel.json`    24 queries, for choosing between P and PF
- `TMDB_promotion_b_panel.json` 30 queries, for confirming the choice
- `TMDB_positive_scope_protocol.json` providers, seeds, arms, payload digests,
  the gate thresholds and the stopping rule

The two panels are disjoint and both are disjoint from every query already
used: the ten-query discovery panel and the twenty-four-query structural
holdout. That is the whole reason this exists. The earlier discovery panel was
assembled from queries a historical run had missed, so re-running it recovered
most of them by sampling alone; and the holdout was screened and confirmed on
the same queries, so a promotion there could not be distinguished from having
looked twice.

Re-running this command verifies the frozen artifacts and refuses to overwrite
a different selection.

  python scripts/freeze_positive_scope.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:  # `python scripts/freeze_positive_scope.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tooldoc_nir.draft_agent_reproduction import (  # noqa: E402
    RELEASED_DECODING,
    _dump_json,
    _load_json,
)
from tooldoc_nir.provenance import (  # noqa: E402
    file_digest,
    git_commit,
    payload_digest,
)
from tooldoc_nir.restbench_contract_variants import (  # noqa: E402
    build_graph,
    variant_paths,
)
from tooldoc_nir.restbench_data import (  # noqa: E402
    DEFAULT_DRAFT_ROOT,
    instruction_path,
    load_instructions,
    load_queries,
    test_path,
)
from tooldoc_nir.restbench_panel import split_structural_panels  # noqa: E402
from tooldoc_nir.restbench_screen import (  # noqa: E402
    EXCLUDED_PROVIDERS,
    MODELS,
    PROMOTION_MAX_P_VALUE,
    PROMOTION_MIN_EFFECT_PP,
    PROMOTION_MIN_POSITIVE_SEEDS,
    PROVIDERS,
    SCREEN_MAX_REPLICATE_LOSS_PP,
    SCREEN_MIN_EFFECT_PP,
)

DOCUMENTATION = Path("artifacts/documentation")

# Already spent, and therefore ineligible for either new panel.
DISCOVERY_PANEL = (0, 5, 13, 17, 20, 33, 51, 57, 62, 66)
HOLDOUT_PANEL_ARTIFACT = DOCUMENTATION / "TMDB_contract_holdout_panel.json"

# A fresh randomization over what the previous two panels left. Not the salt of
# the holdout selection, so this split is not a continuation of that order.
SPLIT_SALT = "positive-scope-split-2026-09-09"

STAGES: tuple[tuple[str, int, tuple[int, ...]], ...] = (
    ("screen_a", 24, (21, 22)),
    ("promotion_b", 30, (23, 24, 25)),
)

SCREEN_ARMS: tuple[str, ...] = ("Initial", "DRAFT", "CurrentNot", "P", "PF")
CANDIDATES: tuple[str, ...] = ("P", "PF")
PROMOTION_ARMS: tuple[str, ...] = ("Initial", "DRAFT")

WORKERS = 15
RETRIEVAL_NUM = 5

ERROR_POLICY = {
    "transport": (
        "network, 429 and 5xx retry identically up to three rounds; a query "
        "whose transport never recovers is dropped from every arm of that "
        "seed, because a block missing one arm cannot be paired"
    ),
    "provider_drift": (
        "a completion served by a provider other than the pinned one fails "
        "the cell and is treated like a transport fault"
    ),
    "model_result": (
        "a malformed or truncated completion, and any released-driver "
        "failure, counts as a miss and is never re-rolled"
    ),
}

STOPPING_RULE = (
    "Screen A promotes at most one candidate. If none clears it, stop tuning "
    "on TMDB, report the model crossover as a negative result and move to "
    "Spotify/G3 with the frozen Ours. Promotion B runs only for the promoted "
    "candidate, and the full ten-seed TMDB run only if Promotion B confirms."
)


def holdout_indices(path: Path = HOLDOUT_PANEL_ARTIFACT) -> tuple[int, ...]:
    if not path.exists():
        raise SystemExit(f"{path} is missing; the holdout panel must be frozen first")
    recorded = _load_json(path)
    return tuple(int(index) for index in recorded["selection"]["indices"])


def stable_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"frozen_at", "git_commit", "selection_digest"}
    }


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    stable = stable_view(payload)
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        **stable,
        "selection_digest": payload_digest(stable),
    }


def _write_once(path: Path, candidate: dict[str, Any], label: str) -> dict[str, Any]:
    expected = payload_digest(stable_view(candidate))
    if path.exists():
        recorded = _load_json(path)
        if recorded.get("selection_digest") != payload_digest(stable_view(recorded)):
            raise SystemExit(f"{path} has an invalid selection digest")
        if payload_digest(stable_view(recorded)) != expected:
            raise SystemExit(
                f"{path} is already frozen with a different {label}; use a new "
                "path/version instead of overwriting it"
            )
        print(f"verified {label}: {path} {expected[:16]}")
        return recorded
    sealed = dict(candidate)
    sealed["selection_digest"] = expected
    _dump_json(path, sealed)
    print(f"froze {label}: {path} {expected[:16]}")
    return sealed


def panel_payloads(draft_root: Path) -> dict[str, dict[str, Any]]:
    queries = load_queries(draft_root, "TMDB")
    instructions = load_instructions(draft_root, "TMDB", "Initial")
    graph = build_graph(instructions)
    excluded = tuple(DISCOVERY_PANEL) + holdout_indices()
    selections = split_structural_panels(
        queries,
        instructions,
        graph,
        sizes={name: size for name, size, _seeds in STAGES},
        excluded_indices=excluded,
        salt=SPLIT_SALT,
    )
    payloads: dict[str, dict[str, Any]] = {}
    for name, _size, seeds in STAGES:
        payloads[name] = _seal(
            {
                "dataset": "RestBench-TMDB",
                "stage": name,
                "purpose": (
                    "candidate selection between P and PF"
                    if name == "screen_a"
                    else "confirmation of the Screen A winner on untouched queries"
                ),
                "seeds": list(seeds),
                "previously_used_indices": list(excluded),
                "query_file_digest": file_digest(test_path(draft_root, "TMDB")),
                "initial_docs_file_digest": file_digest(
                    instruction_path(draft_root, "TMDB", "Initial")
                ),
                "initial_docs_payload_digest": payload_digest(instructions),
                "selection": selections[name],
            }
        )
    return payloads


def arm_digests(arms: tuple[str, ...]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for arm in arms:
        path, _manifest = variant_paths(arm)
        if not path.exists():
            raise SystemExit(
                f"{path} is missing; compile the variants before freezing the "
                "protocol, so the protocol pins what will actually be run"
            )
        digests[arm] = payload_digest(_load_json(path))
    return digests


def protocol_payload(
    panels: Mapping[str, Mapping[str, Any]],
    *,
    draft_root: Path,
) -> dict[str, Any]:
    digests = arm_digests(SCREEN_ARMS)
    unpinned = [key for key in MODELS if key not in PROVIDERS]
    if unpinned:
        raise SystemExit(
            f"no provider is pinned for {unpinned}; run "
            "scripts/qualify_providers.py first"
        )
    return _seal(
        {
            "experiment": "cross-model positive scope",
            "dataset": "RestBench-TMDB",
            "models": dict(MODELS),
            "providers": dict(PROVIDERS),
            "provider_policy": (
                "pinned with fallbacks disabled and chosen by transport-only "
                "qualification; a completion served by another provider fails "
                "the cell"
            ),
            "excluded_providers": dict(EXCLUDED_PROVIDERS),
            "provider_qualification": {
                key: f"artifacts/results/provider_qualification/{key}_qualification.json"
                for key in MODELS
            },
            "decoding": dict(RELEASED_DECODING),
            "retrieval_num": RETRIEVAL_NUM,
            "workers": WORKERS,
            "error_policy": ERROR_POLICY,
            "queries_digest": file_digest(test_path(draft_root, "TMDB")),
            "stages": {
                "screen_a": {
                    "panel": {
                        "path": str(DOCUMENTATION / "TMDB_screen_a_panel.json"),
                        "selection_digest": panels["screen_a"]["selection_digest"],
                        "n": len(panels["screen_a"]["selection"]["indices"]),
                    },
                    "seeds": list(STAGES[0][2]),
                    "arms": list(SCREEN_ARMS),
                    "arm_payload_digests": digests,
                    "baselines": ["DRAFT", "Initial"],
                    "incumbent": "CurrentNot",
                    "primary_candidates": list(CANDIDATES),
                    "design": "randomized complete blocks over the arms",
                    "score": (
                        "the minimum paired effect over the four contrasts "
                        "(2 models x DRAFT/Initial)"
                    ),
                    "gate": {
                        "min_screen_score_pp": SCREEN_MIN_EFFECT_PP,
                        "max_replicate_loss_pp": SCREEN_MAX_REPLICATE_LOSS_PP,
                        "net_vs_incumbent": ">= 0 on both models",
                        "missing_final_consumer": "<= the incumbent's, on both models",
                        "promote": "at most one candidate",
                        "tie_break": [
                            "higher minimum contrast",
                            "higher summed effect",
                            "fewer rewritten surfaces",
                        ],
                    },
                },
                "promotion_b": {
                    "panel": {
                        "path": str(DOCUMENTATION / "TMDB_promotion_b_panel.json"),
                        "selection_digest": panels["promotion_b"]["selection_digest"],
                        "n": len(panels["promotion_b"]["selection"]["indices"]),
                    },
                    "seeds": list(STAGES[1][2]),
                    "arms": [*PROMOTION_ARMS, "screen_a_winner"],
                    "candidate_payload_digests": {
                        arm: digests[arm] for arm in CANDIDATES
                    },
                    "analysis": (
                        "average the replicates inside a query, then a "
                        "query-level label-swap permutation test"
                    ),
                    "gate": {
                        "min_effect_pp": PROMOTION_MIN_EFFECT_PP,
                        "min_positive_replicates": PROMOTION_MIN_POSITIVE_SEEDS,
                        "max_one_sided_p": PROMOTION_MAX_P_VALUE,
                        "contrasts_required": 4,
                    },
                },
                "final_validation": {
                    "queries": 100,
                    "seeds": 10,
                    "rows": ["DFSDT", "DRAFT", "Ours"],
                    "candidate_payload_digests": {
                        arm: digests[arm] for arm in CANDIDATES
                    },
                    "primary_metric": "correct path (CP)",
                    "secondary_metric": (
                        "execution-valid CP: the gold path was executed with no "
                        "call rejected because a parameter was filled wrongly"
                    ),
                    "gate": (
                        "the lower bound of the paired Student-t 95% interval "
                        "is above zero for all four contrasts"
                    ),
                },
            },
            "stopping_rule": STOPPING_RULE,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--documentation-root", type=Path, default=DOCUMENTATION)
    args = parser.parse_args()

    root: Path = args.documentation_root
    candidates = panel_payloads(args.draft_root)
    panels = {
        name: _write_once(
            root / f"TMDB_{name}_panel.json", candidates[name], f"{name} panel"
        )
        for name in candidates
    }
    protocol = _write_once(
        root / "TMDB_positive_scope_protocol.json",
        protocol_payload(panels, draft_root=args.draft_root),
        "protocol",
    )

    screen = panels["screen_a"]["selection"]["indices"]
    promotion = panels["promotion_b"]["selection"]["indices"]
    print(json.dumps({
        "screen_a": screen,
        "promotion_b": promotion,
        "disjoint": not set(screen) & set(promotion),
        "providers": protocol["providers"],
        "arm_payload_digests": {
            arm: digest[:12]
            for arm, digest in protocol["stages"]["screen_a"][
                "arm_payload_digests"
            ].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
