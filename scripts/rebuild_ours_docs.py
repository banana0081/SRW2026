"""Rebuild RestBench Ours docs from Initial. Does not touch result traces."""

from __future__ import annotations

import shutil
from pathlib import Path

from tooldoc_nir.draft_agent_reproduction import _dump_json
from tooldoc_nir.restbench_data import DEFAULT_DRAFT_ROOT, load_instructions
from tooldoc_nir.restbench_ours import build_ours_instructions

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    docs = ROOT / "artifacts" / "documentation"
    docs.mkdir(parents=True, exist_ok=True)
    for dataset in ("TMDB", "Spotify"):
        current = docs / f"{dataset}_Ours.json"
        if current.exists():
            backup = docs / f"{dataset}_Ours_with_not.json"
            if not backup.exists():
                shutil.copy2(current, backup)
        base = load_instructions(DEFAULT_DRAFT_ROOT, dataset, "Initial")
        ours, report = build_ours_instructions(base, base_name="Initial")
        _dump_json(current, ours)
        _dump_json(docs / f"{dataset}_Ours_report.json", report)
        print(dataset, report)


if __name__ == "__main__":
    main()
