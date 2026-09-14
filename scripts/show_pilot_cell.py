"""Print one pilot cell as a table: CP and the stage flags per arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tooldoc_nir.restbench_contract_variants import VARIANTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", help="Pilot cell directories.")
    args = parser.parse_args()
    order = [spec.name for spec in VARIANTS]
    for cell in args.cells:
        path = Path(cell) / "summary.json"
        if not path.exists():
            print(f"{cell}: no summary yet")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"== {Path(cell).name}  ${summary['cost_usd']}  "
            f"indeterminate={summary['indeterminate']}"
        )
        arms = summary["arms"]
        for arm in sorted(arms, key=lambda name: (order.index(name) if name in order else 99)):
            value = arms[arm]
            print(
                f"{arm:16s} CP={value['cp']:2d}/{value['n']:<2d} "
                f"err={value['errors']} "
                f"miss_first={value['missing_first_producer']} "
                f"miss_consumer={value['missing_final_consumer']} "
                f"wrong_first={value['wrong_first']} "
                f"repeat={value['repeat_producer']} "
                f"http_err={value['parameter_or_http_error']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
