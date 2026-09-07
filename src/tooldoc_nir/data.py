from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable

from huggingface_hub import hf_hub_download
import pandas as pd

from .models import ApiDocument, BenchmarkExample, normalize_identifier


DATASET_ID = "tuandunghcmut/toolbench-v1"
DATASET_REVISION = "36de9b189753ad5de276181974f97df15e8c3202"
TOOL_ENV_DATASET_ID = "stabletoolbench/ToolEnv2404"
TOOL_ENV_REVISION = "b6141cc50e0c72894517786e8c6b95b749270535"
TOOL_ENV_ARCHIVE = "toolenv2404_filtered.tar.gz"

SPLIT_FILES: dict[str, str] = {
    "G1_category": "benchmark/g1_category-00000-of-00001.parquet",
    "G1_instruction": "benchmark/g1_instruction-00000-of-00001.parquet",
    "G1_tool": "benchmark/g1_tool-00000-of-00001.parquet",
    "G2_category": "benchmark/g2_category-00000-of-00001.parquet",
    "G2_instruction": "benchmark/g2_instruction-00000-of-00001.parquet",
    "G3_instruction": "benchmark/g3_instruction-00000-of-00001.parquet",
}


def download_benchmark(data_dir: Path) -> dict[str, Path]:
    """Download the pinned benchmark mirror and return local parquet paths."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in SPLIT_FILES.items():
        downloaded = hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=data_dir,
        )
        paths[split] = Path(downloaded)

    manifest = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "relationship": (
            "Convenience parquet mirror of the ToolBench benchmark splits. "
            "The research protocol remains ToolBench/StableToolBench."
        ),
        "files": SPLIT_FILES,
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def download_tool_environment(data_dir: Path, *, extract: bool = True) -> Path:
    """Download the official StableToolBench April 2024 tool snapshot."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id=TOOL_ENV_DATASET_ID,
            filename=TOOL_ENV_ARCHIVE,
            repo_type="dataset",
            revision=TOOL_ENV_REVISION,
            local_dir=data_dir,
        )
    )
    if not extract:
        return archive

    destination = data_dir / "toolenv2404"
    marker = destination / ".complete"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination, filter="data")
    marker.write_text(
        json.dumps(
            {
                "dataset_id": TOOL_ENV_DATASET_ID,
                "revision": TOOL_ENV_REVISION,
                "archive": TOOL_ENV_ARCHIVE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _decode_json(value: Any, field: str, query_id: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {field} for query {query_id}: {exc}"
            ) from exc
    return value


def load_split(path: Path, split: str) -> list[BenchmarkExample]:
    frame = pd.read_parquet(path)
    required_columns = {"query_id", "query", "api_list", "relevant_apis"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")

    examples: list[BenchmarkExample] = []
    for row in frame.to_dict(orient="records"):
        query_id = str(row["query_id"])
        candidates_raw = _decode_json(row["api_list"], "api_list", query_id)
        relevant_raw = _decode_json(
            row["relevant_apis"], "relevant_apis", query_id
        )
        candidates = [ApiDocument.model_validate(api) for api in candidates_raw]
        relevant = [(str(tool), str(api)) for tool, api in relevant_raw]
        examples.append(
            BenchmarkExample(
                query_id=query_id,
                split=split,
                query=str(row["query"]),
                candidates=candidates,
                relevant_apis=relevant,
            )
        )
    return examples


def load_benchmark(data_dir: Path) -> list[BenchmarkExample]:
    paths = {
        split: data_dir / filename for split, filename in SPLIT_FILES.items()
    }
    missing = [split for split, path in paths.items() if not path.exists()]
    if missing:
        downloaded = download_benchmark(data_dir)
        paths.update(downloaded)

    examples: list[BenchmarkExample] = []
    for split, path in paths.items():
        examples.extend(load_split(path, split))
    return examples


def validate_examples(
    examples: Iterable[BenchmarkExample],
) -> dict[str, int]:
    summary = {
        "examples": 0,
        "candidates": 0,
        "relevant_labels": 0,
        "missing_relevant_labels": 0,
        "duplicate_candidate_keys": 0,
    }
    for example in examples:
        summary["examples"] += 1
        summary["candidates"] += len(example.candidates)
        summary["relevant_labels"] += len(example.relevant_apis)
        keys = [candidate.key for candidate in example.candidates]
        summary["duplicate_candidate_keys"] += len(keys) - len(set(keys))
        present = set(keys)
        summary["missing_relevant_labels"] += sum(
            1 for key in example.relevant_keys if key not in present
        )
    return summary


def _split_summary(examples: list[BenchmarkExample]) -> dict[str, Any]:
    grouped: dict[str, list[BenchmarkExample]] = {}
    for example in examples:
        grouped.setdefault(example.split, []).append(example)
    return {
        split: {
            **validate_examples(rows),
            "min_candidates": min(len(row.candidates) for row in rows),
            "max_candidates": max(len(row.candidates) for row in rows),
            "mean_candidates": round(
                sum(len(row.candidates) for row in rows) / len(rows), 2
            ),
        }
        for split, rows in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate ToolBench benchmark data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/toolbench"),
    )
    parser.add_argument(
        "--with-toolenv",
        action="store_true",
        help="Also download and extract StableToolBench ToolEnv2404.",
    )
    args = parser.parse_args()

    examples = load_benchmark(args.data_dir)
    print(json.dumps(_split_summary(examples), indent=2))
    if args.with_toolenv:
        toolenv_path = download_tool_environment(
            args.data_dir.parent / "stabletoolbench"
        )
        print(f"ToolEnv2404: {toolenv_path}")


if __name__ == "__main__":
    main()

