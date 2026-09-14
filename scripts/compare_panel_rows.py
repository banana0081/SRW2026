"""Put an old canonical row and a new pilot arm side by side on the panel.

Used to check whether a changed CP comes from the documentation or from the
model endpoint: same payload digest, same seed, same panel, different date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tooldoc_nir.restbench_report import load_traces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-traces", type=Path, required=True)
    parser.add_argument("--old-row", default="Ours")
    parser.add_argument("--new-traces", type=Path, required=True)
    parser.add_argument("--new-row", default="CurrentNot")
    parser.add_argument("--queries", nargs="+", type=int, required=True)
    args = parser.parse_args()

    def index(path: Path, row: str) -> dict[int, dict]:
        return {
            int(item["query_index"]): item
            for item in load_traces(path)
            if item.get("row") == row and int(item["query_index"]) in set(args.queries)
        }

    old = index(args.old_traces, args.old_row)
    new = index(args.new_traces, args.new_row)
    for query in sorted(args.queries):
        left = old.get(query)
        right = new.get(query)
        if left is None or right is None:
            print(f"q{query:03d} missing in one side")
            continue
        print(f"q{query:03d} gold={left['gold']}")
        print(
            f"  {args.old_row:12s} CP={bool(left['correct_path'])} "
            f"err={(left.get('error') or '')[:40]!r} executed={left['executed']}"
        )
        print(
            f"  {args.new_row:12s} CP={bool(right['correct_path'])} "
            f"err={(right.get('error') or '')[:40]!r} executed={right['executed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
