"""Freeze an outcome-independent TMDB panel before evaluating variants.

The artifact is immutable: rerunning this command verifies the selection and
refuses to overwrite a different panel.

    python scripts/freeze_structural_panel.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from tooldoc_nir.draft_agent_reproduction import _dump_json, _load_json
from tooldoc_nir.provenance import file_digest, git_commit, payload_digest
from tooldoc_nir.restbench_contract_variants import build_graph
from tooldoc_nir.restbench_data import (
    DEFAULT_DRAFT_ROOT,
    instruction_path,
    load_instructions,
    load_queries,
    test_path,
)
from tooldoc_nir.restbench_panel import select_structural_panel


DISCOVERY_PANEL = (0, 5, 13, 17, 20, 33, 51, 57, 62, 66)
DEFAULT_OUTPUT = Path("artifacts/documentation/TMDB_contract_holdout_panel.json")


def frozen_payload(
    *,
    draft_root: Path,
    size: int,
    excluded_indices: tuple[int, ...] = DISCOVERY_PANEL,
) -> dict[str, Any]:
    queries = load_queries(draft_root, "TMDB")
    instructions = load_instructions(draft_root, "TMDB", "Initial")
    graph = build_graph(instructions)
    selection = select_structural_panel(
        queries,
        instructions,
        graph,
        size=size,
        excluded_indices=excluded_indices,
    )
    stable = {
        "dataset": "RestBench-TMDB",
        "purpose": (
            "outcome-independent diagnostic holdout; not final statistical evidence"
        ),
        "query_file_digest": file_digest(test_path(draft_root, "TMDB")),
        "initial_docs_file_digest": file_digest(
            instruction_path(draft_root, "TMDB", "Initial")
        ),
        "initial_docs_payload_digest": payload_digest(instructions),
        "selection": selection,
    }
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        **stable,
        "selection_digest": payload_digest(stable),
    }


def stable_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"frozen_at", "git_commit", "selection_digest"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    candidate = frozen_payload(draft_root=args.draft_root, size=args.size)
    if args.output.exists():
        recorded = _load_json(args.output)
        expected = payload_digest(stable_view(recorded))
        actual = payload_digest(stable_view(candidate))
        if recorded.get("selection_digest") != expected:
            raise SystemExit(f"{args.output} has an invalid selection digest")
        if actual != expected:
            raise SystemExit(
                f"{args.output} is already frozen with a different selection; "
                "use a new path/version instead of overwriting it"
            )
        print(f"verified {args.output}: {expected}")
        print(json.dumps(recorded["selection"], ensure_ascii=False, indent=2))
        return 0

    candidate["selection_digest"] = payload_digest(stable_view(candidate))
    _dump_json(args.output, candidate)
    print(f"froze {args.output}: {candidate['selection_digest']}")
    print(json.dumps(candidate["selection"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
