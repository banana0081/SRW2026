"""Print one line per compiled pilot variant: digest, surfaces, probe evidence."""

from __future__ import annotations

import json
from pathlib import Path

from tooldoc_nir.restbench_contract_variants import EXPERIMENT_ROOT, VARIANTS


def main() -> int:
    for spec in VARIANTS:
        path = EXPERIMENT_ROOT / f"TMDB_{spec.name}_manifest.json"
        if not path.exists():
            print(f"{spec.name:16s} not built")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        surfaces = manifest["surfaces_touched"]
        paths = (manifest.get("probe") or {}).get("paths") or {}
        print(
            f"{spec.name:16s} {manifest['payload_digest'][:12]} "
            f"tool_description={surfaces['tool_description']:2d} "
            f"description={surfaces['description']:2d} "
            f"example={surfaces['example']:2d} "
            f"ports_verified={paths.get('verified', 0)} "
            f"absent={paths.get('absent', 0)} "
            f"nullable={paths.get('nullable', 0)} "
            f"inconclusive={paths.get('inconclusive', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
