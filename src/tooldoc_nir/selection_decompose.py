"""Decompose G3 Raw vs DRAFT agent traces into selection-error classes.

The class counts alone cannot answer whether the conditions differ, because a
tie in the aggregate rate can hide a large exchange of per-query wins. Paired
statistics are therefore reported next to the decomposition.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

from .draft_agent_reproduction import (
    DEFAULT_DRAFT_ROOT,
    _executed_path,
    _load_json,
    correct_path,
    correct_path_released,
    gold_path,
    record_positions,
)

BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 42


def classify_record(record: dict[str, Any], query: dict[str, Any]) -> str:
    gold = gold_path(query)
    gold_set = set(gold)
    gold_tools = {tool for tool, _ in gold}
    executed = _executed_path(record)
    if not executed:
        return "empty_path"
    if correct_path(record, query):
        return "correct_path"
    executed_tools = {tool for tool, _ in executed}
    if executed_tools.isdisjoint(gold_tools):
        return "wrong_tool"
    if not any(item in gold_set for item in executed):
        return "wrong_api"
    return "partial_or_unordered"


def paired_outcome(raw_class: str, draft_class: str) -> str:
    raw_ok = raw_class == "correct_path"
    draft_ok = draft_class == "correct_path"
    if raw_ok and draft_ok:
        return "both_correct"
    if draft_ok:
        return "draft_only"
    if raw_ok:
        return "raw_only"
    return "both_fail"


def exact_sign_test(raw_only: int, draft_only: int) -> float:
    """Two-sided exact binomial test over discordant pairs."""
    discordant = raw_only + draft_only
    if discordant == 0:
        return 1.0
    smaller = min(raw_only, draft_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2 * tail / (2**discordant))


def mcnemar_statistic(raw_only: int, draft_only: int) -> float | None:
    """Chi-square with continuity correction; None when undefined."""
    discordant = raw_only + draft_only
    if discordant == 0:
        return None
    return round((abs(raw_only - draft_only) - 1) ** 2 / discordant, 4)


def paired_bootstrap(
    differences: Sequence[int],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Percentile interval for the mean paired difference, DRAFT minus Raw."""
    count = len(differences)
    if count == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0
        for _ in range(count):
            total += differences[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    return {
        "mean": round(sum(differences) / count, 4),
        "low": round(means[int(0.025 * resamples)], 4),
        "high": round(means[min(resamples - 1, int(0.975 * resamples))], 4),
        "resamples": resamples,
        "seed": seed,
    }


def paired_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_only = sum(row["pair"] == "raw_only" for row in rows)
    draft_only = sum(row["pair"] == "draft_only" for row in rows)
    differences = [
        int(row["draft_correct_path"]) - int(row["raw_correct_path"])
        for row in rows
    ]
    return {
        "discordant_pairs": raw_only + draft_only,
        "raw_only": raw_only,
        "draft_only": draft_only,
        "exact_sign_test_p": round(exact_sign_test(raw_only, draft_only), 4),
        "mcnemar_chi_square_cc": mcnemar_statistic(raw_only, draft_only),
        "paired_delta_draft_minus_raw": paired_bootstrap(differences),
    }


def compare_condition_pair(
    *,
    queries: Sequence[dict[str, Any]],
    left_name: str,
    left_records: dict[int, dict[str, Any]],
    right_name: str,
    right_records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Pairwise CP and sign test for any two conditions, aligned by query id."""
    positions = sorted(set(left_records) & set(right_records))
    if not positions:
        raise ValueError(
            f"No paired records for {left_name} vs {right_name}."
        )
    rows: list[dict[str, Any]] = []
    left_only = 0
    right_only = 0
    differences: list[int] = []
    for index in positions:
        query = queries[index]
        left = left_records[index]
        right = right_records[index]
        left_ok = correct_path(left, query)
        right_ok = correct_path(right, query)
        differences.append(int(right_ok) - int(left_ok))
        if left_ok and not right_ok:
            left_only += 1
        if right_ok and not left_ok:
            right_only += 1
        rows.append(
            {
                "query_index": index,
                "query_id": query.get("query_id"),
                f"{left_name}_class": classify_record(left, query),
                f"{right_name}_class": classify_record(right, query),
                f"{left_name}_correct_path": left_ok,
                f"{right_name}_correct_path": right_ok,
            }
        )
    n = len(positions)
    return {
        "queries": n,
        "left": left_name,
        "right": right_name,
        "paper_cp": {
            left_name: round(
                sum(row[f"{left_name}_correct_path"] for row in rows) / n, 4
            ),
            right_name: round(
                sum(row[f"{right_name}_correct_path"] for row in rows) / n, 4
            ),
        },
        "paired_statistics": {
            "discordant_pairs": left_only + right_only,
            f"{left_name}_only": left_only,
            f"{right_name}_only": right_only,
            "exact_sign_test_p": round(exact_sign_test(left_only, right_only), 4),
            "mcnemar_chi_square_cc": mcnemar_statistic(left_only, right_only),
            f"paired_delta_{right_name}_minus_{left_name}": paired_bootstrap(
                differences
            ),
        },
        "rows": rows,
    }


def _condition_jsonl(run_root: Path, condition: str) -> Path:
    directory = run_root / condition.lower()
    matches = sorted(directory.glob("ToolBench_G3_DFS_*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No released-format output under {directory}.")
    if len(matches) > 1:
        raise ValueError(
            f"{directory} holds several agent outputs: "
            f"{[path.name for path in matches]}. Pass an explicit path."
        )
    return matches[0]


def decompose_g3(
    *,
    draft_root: Path,
    raw_jsonl: Path,
    draft_jsonl: Path,
) -> dict[str, Any]:
    queries = _load_json(
        draft_root / "dataset" / "ToolBench" / "test_data" / "G3.json"
    )
    raw_records = record_positions(raw_jsonl)
    draft_records = record_positions(draft_jsonl)
    # Records are matched by the position the driver stamped on them, so a
    # query dropped by one condition cannot shift the other's alignment.
    positions = sorted(set(raw_records) & set(draft_records))
    n = len(positions)
    rows: list[dict[str, Any]] = []
    for index in positions:
        query = queries[index]
        raw_record = raw_records[index]
        draft_record = draft_records[index]
        raw_class = classify_record(raw_record, query)
        draft_class = classify_record(draft_record, query)
        rows.append(
            {
                "query_index": index,
                "query_id": query.get("query_id"),
                "gold": gold_path(query),
                "raw_class": raw_class,
                "draft_class": draft_class,
                "pair": paired_outcome(raw_class, draft_class),
                "raw_correct_path": correct_path(raw_record, query),
                "draft_correct_path": correct_path(draft_record, query),
                "raw_correct_path_released": correct_path_released(
                    raw_record, query
                ),
                "draft_correct_path_released": correct_path_released(
                    draft_record, query
                ),
                "same_executed_set": set(_executed_path(raw_record))
                == set(_executed_path(draft_record)),
            }
        )

    if not rows:
        raise ValueError("No paired records to decompose.")

    def rate(key: str, value: str) -> float:
        return round(sum(row[key] == value for row in rows) / n, 4)

    return {
        "queries": n,
        "paper_cp": {
            "raw": round(sum(row["raw_correct_path"] for row in rows) / n, 4),
            "draft": round(
                sum(row["draft_correct_path"] for row in rows) / n, 4
            ),
        },
        "released_cp": {
            "raw": round(
                sum(row["raw_correct_path_released"] for row in rows) / n, 4
            ),
            "draft": round(
                sum(row["draft_correct_path_released"] for row in rows) / n, 4
            ),
        },
        "raw_classes": dict(Counter(row["raw_class"] for row in rows)),
        "draft_classes": dict(Counter(row["draft_class"] for row in rows)),
        "paired": dict(Counter(row["pair"] for row in rows)),
        "paired_rates": {
            label: rate("pair", label)
            for label in ("both_correct", "draft_only", "raw_only", "both_fail")
        },
        "paired_statistics": paired_statistics(rows),
        "same_executed_set_rate": round(
            sum(row["same_executed_set"] for row in rows) / n, 4
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose Raw vs DRAFT G3 selection errors."
    )
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/results/g3_calibration_v2"),
        help="Run directory holding initial/ and draft/ agent outputs.",
    )
    parser.add_argument("--raw-jsonl", type=Path, default=None)
    parser.add_argument("--draft-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    raw_jsonl = args.raw_jsonl or _condition_jsonl(args.run_root, "Initial")
    draft_jsonl = args.draft_jsonl or _condition_jsonl(args.run_root, "DRAFT")
    output = args.output or args.run_root / "selection_decompose.json"

    result = decompose_g3(
        draft_root=args.draft_root,
        raw_jsonl=raw_jsonl,
        draft_jsonl=draft_jsonl,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
