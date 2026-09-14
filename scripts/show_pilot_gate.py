"""Print the pooled pilot result, the preregistered gate and its caveats."""

from __future__ import annotations

import json
from pathlib import Path

REPORT = Path("artifacts/results/pilot_contract/pilot_report.json")


def main() -> int:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    report = payload["report"]
    print("pooled CP over 20 cells per model")
    arms = list(next(iter(report.values()))["pooled"])
    header = "arm".ljust(16) + "".join(key.ljust(10) for key in report)
    print(header + "miss_consumer")
    for arm in arms:
        line = arm.ljust(16)
        consumers = []
        for model_key, block in report.items():
            scores = block["pooled"].get(arm)
            line += (f"{scores['cp']}/{scores['n']}" if scores else "-").ljust(10)
            if scores:
                consumers.append(f"{model_key}={scores['missing_final_consumer']}")
        print(line + " ".join(consumers))

    print("\npaired net vs contemporaneous baselines (pooled)")
    for model_key, block in report.items():
        for label in ("paired_vs_currentnot", "paired_vs_draft"):
            nets = block.get(label) or {}
            rendered = ", ".join(
                f"{arm}:{value['net']:+d}" for arm, value in nets.items()
            )
            print(f"  {model_key} {label.replace('paired_vs_', 'vs ')}: {rendered}")

    print("\nreproduction of the frozen documentation against its canonical run")
    for model_key, block in report.items():
        for seed_key, cell in block["seeds"].items():
            check = cell.get("reproduction_check") or {}
            if not check:
                continue
            print(
                f"  {model_key} {seed_key}: canonical {check['canonical_cp']}/"
                f"{check['n']} -> now {check['pilot_cp']}/{check['n']}, "
                f"same documentation={check['documentation_identical']}, "
                f"recovered={check['recovered']}"
            )

    print("\ninterpretation")
    for model_key, note in payload["interpretation"].items():
        print(
            f"  {model_key}: CP spread {note['cp_spread']}, "
            f"{note['arms_at_ceiling']} arms at ceiling, "
            f"discriminative={note['discriminative']}"
        )

    print("\ngate")
    for arm, verdict in payload["gate"].items():
        state = "PROMOTED" if verdict["promoted"] else "held"
        print(f"  {arm:16s} {state:9s} {'; '.join(verdict['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
