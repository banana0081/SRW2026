"""Execution-side G3 metrics: parameter validity and backend success.

Correct Path only scores the sequence of `(tool, api)` names, so a condition can
tie on it while producing calls the documented schema would reject. These
metrics are graded against the pinned StableToolBench ToolEnv documents rather
than against the condition's own documentation, so neither Raw nor DRAFT can
grade its own homework.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from .draft_agent_reproduction import (
    DEFAULT_STABLE_ROOT,
    StableExecutionAdapter,
    _load_json,
    _process_name,
    read_jsonl,
    record_positions,
)
from .selection_decompose import _condition_jsonl

RESERVED_NAMES = frozenset(
    {"from", "class", "return", "false", "true", "id", "and", "", "ID"}
)


def change_name(name: str) -> str:
    """The released parameter-key rewrite, applied to both sides of the check."""
    if name in RESERVED_NAMES:
        return "is_" + name.lower()
    return name


def _schema_index(stable_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Ground-truth request schemas keyed by (category, tool, api)."""
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    tools_root = stable_root / "tools"
    for path in tools_root.glob("*/*.json"):
        try:
            document = _load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(document, dict):
            continue
        category = path.parent.name
        tool = path.stem
        for api in document.get("api_list") or []:
            if not isinstance(api, dict):
                continue
            key = (category, tool, _process_name(str(api.get("name") or "")))
            index[key] = api
    return index


def _flatten_parameters(parameters: Any) -> dict[str, Any]:
    """The released driver accepts either a dict or a list of dicts."""
    flat: dict[str, Any] = {}
    if isinstance(parameters, dict):
        items = parameters.items()
        for key, value in items:
            flat[change_name(str(key))] = value
    elif isinstance(parameters, list):
        for entry in parameters:
            if isinstance(entry, dict):
                for key, value in entry.items():
                    flat[change_name(str(key))] = value
    return flat


def _documented_names(api: Mapping[str, Any], field: str) -> set[str]:
    return {
        change_name(str(item.get("name") or ""))
        for item in api.get(field) or []
        if isinstance(item, dict)
    }


def classify_call(
    call: Mapping[str, Any],
    schemas: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    category = StableExecutionAdapter._category(str(call.get("categoty") or ""))
    tool = _process_name(str(call.get("tool_name") or ""))
    api_name = _process_name(str(call.get("api_name") or ""))
    provided = _flatten_parameters(call.get("parameters"))
    api = schemas.get((category, tool, api_name))
    if api is None:
        return {
            "category": category,
            "tool_name": tool,
            "api_name": api_name,
            "schema_found": False,
            "required_satisfied": None,
            "no_unknown_parameters": None,
            "fully_valid": False,
            "missing_required": [],
            "unknown_parameters": [],
        }
    required = _documented_names(api, "required_parameters")
    optional = _documented_names(api, "optional_parameters")
    missing = sorted(required - set(provided))
    unknown = sorted(set(provided) - required - optional)
    return {
        "category": category,
        "tool_name": tool,
        "api_name": api_name,
        "schema_found": True,
        "required_satisfied": not missing,
        "no_unknown_parameters": not unknown,
        "fully_valid": not missing and not unknown,
        "missing_required": missing,
        "unknown_parameters": unknown,
    }


def parameter_validity(
    records: list[dict[str, Any]],
    schemas: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        query_calls = [
            classify_call(call, schemas)
            for group in record["execute_log"]["api_result_ls"]
            for call in group
        ]
        calls.extend(query_calls)
        resolvable = [row for row in query_calls if row["schema_found"]]
        per_query.append(
            {
                "position": position,
                "calls": len(query_calls),
                "schema_found": len(resolvable),
                "all_valid": bool(resolvable)
                and all(row["fully_valid"] for row in resolvable),
            }
        )
    resolvable = [row for row in calls if row["schema_found"]]

    def share(count: int, total: int) -> float | None:
        return round(count / total, 4) if total else None

    return {
        "calls": len(calls),
        "schema_found": len(resolvable),
        "schema_found_rate": share(len(resolvable), len(calls)),
        "required_satisfied_rate": share(
            sum(row["required_satisfied"] for row in resolvable),
            len(resolvable),
        ),
        "no_unknown_parameter_rate": share(
            sum(row["no_unknown_parameters"] for row in resolvable),
            len(resolvable),
        ),
        "fully_valid_rate": share(
            sum(row["fully_valid"] for row in resolvable), len(resolvable)
        ),
        "queries_with_all_calls_valid": share(
            sum(row["all_valid"] for row in per_query), len(per_query)
        ),
        "top_missing_required": _top_counts(
            name
            for row in resolvable
            for name in row["missing_required"]
        ),
        "top_unknown_parameters": _top_counts(
            name
            for row in resolvable
            for name in row["unknown_parameters"]
        ),
    }


def _top_counts(names: Any, limit: int = 10) -> list[list[Any]]:
    counts: dict[str, int] = defaultdict(int)
    for name in names:
        counts[name] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [[name, count] for name, count in ordered[:limit]]


def backend_success(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Error-free response rate, split by where the response came from."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        grouped[str(call.get("condition") or "-")].append(call)
    report: dict[str, Any] = {}
    for condition, rows in sorted(grouped.items()):
        by_source: dict[str, Any] = {}
        for source in (
            "exact_cache",
            "generated_cache",
            "simulated",
            "schema_gate",
            "live",
            "live_cache",
        ):
            subset = [row for row in rows if row.get("source") == source]
            if not subset:
                continue
            error_free = sum(1 for row in subset if not row.get("error"))
            by_source[source] = {
                "calls": len(subset),
                "error_free": error_free,
                "error_free_rate": round(error_free / len(subset), 4),
                "median_response_chars": median(
                    int(row.get("response_chars") or 0) for row in subset
                ),
            }
        error_free_total = sum(1 for row in rows if not row.get("error"))
        report[condition] = {
            "calls": len(rows),
            "error_free_rate": round(error_free_total / len(rows), 4)
            if rows
            else None,
            "by_source": by_source,
        }
    return report


def evaluate_run(
    *,
    run_root: Path,
    stable_root: Path,
    conditions: tuple[str, ...] = ("Initial", "DRAFT"),
) -> dict[str, Any]:
    schemas = _schema_index(stable_root)
    indexed = {
        condition: record_positions(_condition_jsonl(run_root, condition))
        for condition in conditions
    }
    # Score the same queries in both conditions, matched by driver-stamped ID.
    paired = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    per_condition: dict[str, Any] = {}
    for condition, records in indexed.items():
        per_condition[condition] = {
            "records": len(records),
            "scored_queries": len(paired),
            "parameter_validity": parameter_validity(
                [records[position] for position in paired], schemas
            ),
        }
    return {
        "run_root": str(run_root),
        "stable_root": str(stable_root),
        "documented_apis_indexed": len(schemas),
        "scored_positions": paired,
        "conditions": per_condition,
        "backend_success": backend_success(
            read_jsonl(run_root / "backend_calls.jsonl")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter validity and backend success for a G3 run."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/results/g3_calibration_v2"),
    )
    parser.add_argument("--stable-root", type=Path, default=DEFAULT_STABLE_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_run(
        run_root=args.run_root,
        stable_root=args.stable_root,
    )
    output = args.output or args.run_root / "g3_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
