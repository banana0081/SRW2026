"""Rebuild every variant and refuse any drift from the frozen payload digest.

The pilot is already running against these documents, so a later refactor of
the compiler must not change a single character of them. This is the check
that says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tooldoc_nir.provenance import payload_digest
from tooldoc_nir.restbench_contract_variants import (
    EXPERIMENT_ROOT,
    VARIANTS,
    build_graph,
    build_variant,
    ID_TYPES,
)
from tooldoc_nir.restbench_data import DEFAULT_DRAFT_ROOT, load_instructions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--dataset", default="TMDB")
    args = parser.parse_args()

    base = load_instructions(args.draft_root, args.dataset, "Initial")
    graph = build_graph(base, id_type=ID_TYPES.get(args.dataset, "str"))
    drift = 0
    for spec in VARIANTS:
        manifest_path = EXPERIMENT_ROOT / f"{args.dataset}_{spec.name}_manifest.json"
        if not manifest_path.exists():
            print(f"{spec.name:16s} not frozen yet")
            continue
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        docs, _manifest = build_variant(
            spec, draft_root=args.draft_root, dataset=args.dataset, graph=graph
        )
        now = payload_digest(docs)
        same = now == recorded["payload_digest"]
        drift += 0 if same else 1
        print(
            f"{spec.name:16s} {'ok' if same else 'DRIFT'} "
            f"frozen={recorded['payload_digest'][:12]} rebuilt={now[:12]}"
        )
    if drift:
        print(f"{drift} variant(s) no longer match their frozen digest.")
        return 1
    print("every variant reproduces its frozen payload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
