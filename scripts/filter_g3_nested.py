"""Write the G3 nested-slice report."""

from __future__ import annotations

import json
from pathlib import Path

from tooldoc_nir.draft_agent_reproduction import DEFAULT_DRAFT_ROOT, _load_json
from tooldoc_nir.g3_nested import classify_dataset

OUT = Path("artifacts/documentation/G3_nested.json")


def main() -> None:
    queries = _load_json(DEFAULT_DRAFT_ROOT / "dataset" / "ToolBench" / "test_data" / "G3.json")
    report = classify_dataset(queries)
    slim = {
        "n": report["n"],
        "counts": report["counts"],
        "nested_indices": report["nested_indices"],
        "underspecified_indices": report["underspecified_indices"],
        "independent_n": len(report["independent_indices"]),
        "nested_edges": [
            {"index": i, **report["queries"][i]}
            for i in report["nested_indices"]
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n": slim["n"],
                "counts": slim["counts"],
                "path": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
