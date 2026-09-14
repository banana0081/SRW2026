"""Write TMDB Bind vs gold-path overlap."""

from __future__ import annotations

import json
from pathlib import Path

from tooldoc_nir.bind_audit import audit_queries
from tooldoc_nir.restbench_data import gold_apis, load_instructions, load_queries

OUT = Path("artifacts/documentation/TMDB_bind_vs_gold.json")


def main() -> None:
    queries = load_queries(Path("external/DRAFT"), "TMDB")
    initial = load_instructions(Path("external/DRAFT"), "TMDB", "Initial")
    report = audit_queries(queries, initial, gold_apis)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_queries": report["n_queries"],
                "n_gold_hops": report["n_gold_hops"],
                "later_hops": report["later_hops"],
                "later_covered": report["later_covered"],
                "later_covered_share": round(report["later_covered_share"], 3),
                "n_later_misses": report["n_later_misses"],
                "path": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
