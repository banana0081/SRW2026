"""Combine a G3 run's summary, selection decomposition and execution metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .draft_agent_reproduction import (
    DEFAULT_DRAFT_ROOT,
    DEFAULT_STABLE_ROOT,
    _dump_json,
    _load_json,
)
from .g3_metrics import evaluate_run
from .selection_decompose import _condition_jsonl, decompose_g3


def summarize_run(
    *,
    run_root: Path,
    draft_root: Path = DEFAULT_DRAFT_ROOT,
    stable_root: Path = DEFAULT_STABLE_ROOT,
) -> dict[str, Any]:
    summary = _load_json(run_root / "summary.json")
    decompose = decompose_g3(
        draft_root=draft_root,
        raw_jsonl=_condition_jsonl(run_root, "Initial"),
        draft_jsonl=_condition_jsonl(run_root, "DRAFT"),
    )
    metrics = evaluate_run(run_root=run_root, stable_root=stable_root)
    printable_decompose = {
        key: value for key, value in decompose.items() if key != "rows"
    }
    return {
        "run_root": str(run_root),
        "stop_reason": summary.get("stop_reason"),
        "planned_queries": summary.get("planned_queries"),
        "scored_queries": summary.get("scored_queries"),
        "excluded_positions": summary.get("excluded_positions"),
        "usage": summary.get("usage"),
        "manifest_fingerprint": (summary.get("manifest") or {}).get("fingerprint"),
        "agent": (summary.get("manifest") or {}).get("agent"),
        "paper_cp": printable_decompose.get("paper_cp"),
        "released_cp": printable_decompose.get("released_cp"),
        "paired": printable_decompose.get("paired"),
        "paired_statistics": printable_decompose.get("paired_statistics"),
        "raw_classes": printable_decompose.get("raw_classes"),
        "draft_classes": printable_decompose.get("draft_classes"),
        "parameter_validity": {
            name: payload.get("parameter_validity")
            for name, payload in (metrics.get("conditions") or {}).items()
        },
        "backend_success": metrics.get("backend_success"),
        "scope_note": summary.get("scope_note"),
        "wording": (
            "Adapted StableToolBench measurement, not a literal RapidAPI "
            "replication of the published table."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--stable-root", type=Path, default=DEFAULT_STABLE_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = summarize_run(
        run_root=args.run_root,
        draft_root=args.draft_root,
        stable_root=args.stable_root,
    )
    output = args.output or args.run_root / "phase_summary.json"
    _dump_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
