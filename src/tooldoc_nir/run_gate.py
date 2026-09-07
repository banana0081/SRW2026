"""Decide whether a finished run may be trusted and extended.

Each stage of the calibration is only allowed to continue if the previous one
is clean on every failure mode that made the first attempt unusable: fabricated
decisions, stub records, misaligned queries, a drifting provider or a simulator
that produced unparseable output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .draft_agent_reproduction import (
    DEFAULT_DRAFT_ROOT,
    _load_json,
    read_jsonl,
    record_positions,
)
from .selection_decompose import _condition_jsonl


def _expected_provider_set(summary: dict[str, Any]) -> set[str]:
    requested = str(summary["manifest"]["agent"]["model"])
    return {requested, requested.split("/")[-1]}


def check_run(
    *,
    run_root: Path,
    draft_root: Path,
    max_exclusion_rate: float = 0.05,
) -> dict[str, Any]:
    summary = _load_json(run_root / "summary.json")
    manifest = summary["manifest"]
    problems: list[str] = []
    notes: list[str] = []

    if summary.get("stop_reason") != "completed":
        problems.append(f"stop_reason is {summary.get('stop_reason')!r}")
    if summary.get("pending_positions"):
        problems.append(
            f"{len(summary['pending_positions'])} queries were never attempted"
        )

    planned = int(summary["planned_queries"])
    scored = int(summary["scored_queries"])
    excluded = summary.get("excluded_positions") or []
    if planned and len(excluded) / planned > max_exclusion_rate:
        problems.append(
            f"{len(excluded)}/{planned} queries were excluded, above the "
            f"tolerated {max_exclusion_rate:.0%}"
        )
    elif excluded:
        notes.append(
            f"{len(excluded)} query/queries excluded pairwise: {excluded}"
        )

    for name, condition in summary["conditions"].items():
        if condition.get("rolled_back_records"):
            problems.append(
                f"{name} rolled back {condition['rolled_back_records']} record(s)"
            )
        if condition.get("unparsed_completions"):
            notes.append(
                f"{name} saw {condition['unparsed_completions']} unparseable "
                "agent completions, passed through as the released None"
            )
        if condition.get("format_resamples"):
            notes.append(
                f"{name} resampled {condition['format_resamples']} completion(s) "
                "that violated the shape the released prompt requires"
            )

    backend = summary["backend_stats"]
    if backend.get("unparseable_simulation"):
        problems.append(
            f"the simulator produced {backend['unparseable_simulation']} "
            "unparseable responses"
        )
    if backend.get("missing_document"):
        notes.append(
            f"{backend['missing_document']} calls hit an API with no ToolEnv "
            "document; the simulator answered from the API name alone"
        )

    # No record may be a harness stub, and every record must sit at the
    # position of the query it answered.
    queries = _load_json(
        draft_root / "dataset" / "ToolBench" / "test_data" / "G3.json"
    )
    plan = _load_json(run_root / "queries.json")
    for name in summary["conditions"]:
        records = record_positions(_condition_jsonl(run_root, name))
        for position, record in records.items():
            if "runner_error" in record or "harness_stub" in record:
                problems.append(f"{name} position {position} is a stub record")
            expected = queries[int(plan[position]["query_index"])]["query"]
            if record.get("question") != expected:
                problems.append(
                    f"{name} position {position} answers the wrong query"
                )

    expected_models = _expected_provider_set(summary)
    simulator_model = str(manifest["simulator"]["model"])
    live_backend = (
        str(manifest.get("protocol", {}).get("backend") or "") == "rapidapi_live"
        or simulator_model == "none"
    )
    for role in ("agent",) if live_backend else ("agent", "simulator"):
        served = set(summary["usage"][role]["served_models"])
        if role == "agent" and served - expected_models:
            problems.append(
                f"the agent was served by {sorted(served)} instead of "
                f"{sorted(expected_models)}"
            )
        providers = summary["usage"][role]["providers"]
        if len(providers) > 1:
            problems.append(f"{role} provider drift across {providers}")

    if not live_backend:
        simulator_calls = [
            row
            for row in read_jsonl(run_root / "usage_simulator.jsonl")
            if row.get("requested_model") != simulator_model
        ]
        if simulator_calls:
            problems.append(
                f"{len(simulator_calls)} simulator calls did not use the pinned "
                f"{simulator_model}"
            )

    return {
        "run_root": str(run_root),
        "passed": not problems,
        "problems": problems,
        "notes": notes,
        "planned_queries": planned,
        "scored_queries": scored,
        "excluded_positions": excluded,
        "correct_path": {
            name: condition["correct_path_rate"]
            for name, condition in summary["conditions"].items()
        },
        "correct_path_released": {
            name: condition["correct_path_rate_released"]
            for name, condition in summary["conditions"].items()
        },
        "cost_usd": summary["usage"]["cost_usd"],
        "agent_model": manifest["agent"]["model"],
        "seed": manifest["agent"]["seed"],
        "revision": manifest["git_commit"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate a finished calibration run before spending more."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--max-exclusion-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = check_run(
        run_root=args.run_root,
        draft_root=args.draft_root,
        max_exclusion_rate=args.max_exclusion_rate,
    )
    output = args.output or args.run_root / "run_gate.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
