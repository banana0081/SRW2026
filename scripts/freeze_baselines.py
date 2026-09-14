"""Freeze the canonical RestBench baselines and repair report provenance.

Two problems this fixes without touching a single measured number:

1. `TMDB_Ours_report.json` claimed `choose_tool_text: ..._no_not` while the
   frozen `TMDB_Ours.json` (file digest `ad3dc760...`) still contains
   `not GET_`. The label is now derived from the documentation text, and the
   report carries the digest of the exact JSON it describes.
2. `file_digest` alone is not portable: `Path.write_text` translates "\n" to
   "\r\n" on Windows, so the same compiler output hashes differently on two
   machines. Every frozen entry therefore also records `payload_digest`, the
   digest of the canonical JSON payload.

The script never rewrites a documentation JSON or a traces file. It refuses
to relabel a report unless the current compiler reproduces the frozen payload
byte for byte after canonicalisation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from tooldoc_nir.draft_agent_reproduction import _dump_json, _load_json
from tooldoc_nir.provenance import file_digest, git_commit, payload_digest
from tooldoc_nir.restbench_data import (
    DEFAULT_DRAFT_ROOT,
    instruction_path,
    load_instructions,
    test_path,
)
from tooldoc_nir.restbench_ours import build_ours_instructions, choose_tool_label

DATASETS = ("TMDB", "Spotify")
DOC_ROOT = Path("artifacts/documentation")
RESULTS_ROOT = Path("artifacts/results")
FREEZE_PATH = DOC_ROOT / "baseline_digests.json"

# Runs that were started and abandoned: the v4 arm replaced `not GET_` with
# `Next: a hop tool`, cost Ling s04 seven queries and was reverted.
NON_CANONICAL_SUFFIXES = ("_v4", "_ours_v3", "_smoke_v2")


def _describe_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "file_digest": "missing"}
    entry: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_digest": file_digest(path),
    }
    if path.suffix == ".json":
        entry["payload_digest"] = payload_digest(_load_json(path))
    return entry


def repair_ours_report(dataset: str, *, draft_root: Path) -> dict[str, Any]:
    """Rewrite the Ours report so its label and digest describe one JSON."""
    frozen_path = DOC_ROOT / f"{dataset}_Ours.json"
    report_path = DOC_ROOT / f"{dataset}_Ours_report.json"
    frozen = _load_json(frozen_path)
    initial = load_instructions(draft_root, dataset, "Initial")
    rebuilt, report = build_ours_instructions(initial, base_name="Initial")
    if payload_digest(rebuilt) != payload_digest(frozen):
        raise SystemExit(
            f"{frozen_path} is not reproducible from the current compiler; "
            "refusing to relabel a documentation this code did not produce."
        )
    label = choose_tool_label(frozen)
    previous = _load_json(report_path) if report_path.exists() else {}
    report["choose_tool_text"] = label
    report["documentation"] = _describe_file(frozen_path)
    report["reproduced_by"] = "scripts/freeze_baselines.py"
    if previous.get("choose_tool_text") not in (None, label):
        report["corrected_label"] = {
            "was": previous["choose_tool_text"],
            "now": label,
            "reason": "label was hand-written and did not match the frozen text",
        }
    _dump_json(report_path, report)
    return report


def _run_entry(run_root: Path) -> dict[str, Any]:
    traces = run_root / "traces.jsonl"
    manifest = run_root / "manifest.json"
    entry: dict[str, Any] = {
        "run": run_root.name,
        "canonical": not any(
            run_root.name.endswith(suffix) for suffix in NON_CANONICAL_SUFFIXES
        ),
        "traces": _describe_file(traces),
    }
    if manifest.exists():
        recorded = _load_json(manifest)
        entry["model"] = recorded.get("model")
        entry["seed"] = recorded.get("seed")
        entry["rows"] = sorted((recorded.get("rows") or {}).keys())
        entry["doc_digests"] = {
            name: value.get("digest")
            for name, value in (recorded.get("docs") or {}).items()
        }
    if traces.exists():
        rows = [
            json.loads(line)
            for line in traces.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        entry["records"] = len(rows)
        entry["errors"] = sum(1 for row in rows if row.get("error"))
    return entry


def build_freeze(draft_root: Path) -> dict[str, Any]:
    documentation: dict[str, Any] = {}
    for dataset in DATASETS:
        for name in ("Initial", "DRAFT"):
            documentation[f"{dataset}/{name}"] = _describe_file(
                instruction_path(draft_root, dataset, name)
            )
        documentation[f"{dataset}/Ours"] = _describe_file(
            DOC_ROOT / f"{dataset}_Ours.json"
        )
        documentation[f"{dataset}/queries"] = _describe_file(
            test_path(draft_root, dataset)
        )
    runs = [
        _run_entry(path)
        for path in sorted(RESULTS_ROOT.glob("restbench_*"))
        if path.is_dir()
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "note": (
            "Canonical inputs for the RestBench tables. Pilot variants live in "
            "artifacts/documentation/experiments/ and must never overwrite "
            "these files."
        ),
        "documentation": documentation,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--check", action="store_true", help="Verify, do not write.")
    args = parser.parse_args()

    freeze = build_freeze(args.draft_root)
    if args.check:
        recorded = _load_json(FREEZE_PATH) if FREEZE_PATH.exists() else {}
        drifted = [
            key
            for key, value in freeze["documentation"].items()
            if (recorded.get("documentation") or {}).get(key, {}).get("payload_digest")
            != value.get("payload_digest")
        ]
        if drifted:
            print("payload drift: " + ", ".join(sorted(drifted)))
            return 1
        print(f"{len(freeze['documentation'])} frozen inputs unchanged.")
        return 0

    for dataset in DATASETS:
        report = repair_ours_report(dataset, draft_root=args.draft_root)
        corrected = report.get("corrected_label")
        print(
            f"{dataset}_Ours_report.json label={report['choose_tool_text']}"
            + (f" (was {corrected['was']})" if corrected else "")
        )
    _dump_json(FREEZE_PATH, freeze)
    canonical = sum(1 for run in freeze["runs"] if run["canonical"])
    print(
        f"froze {len(freeze['documentation'])} inputs and {len(freeze['runs'])} runs "
        f"({canonical} canonical) into {FREEZE_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
