"""Pairwise CP win% of DFSDT / Ours against DRAFT."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path("artifacts/results")


def hit(row: dict) -> bool | None:
    """Released DFSDT retries a failed sample; a leftover harness error is
    not a scored path. Drop that cell instead of calling it a miss."""
    if row.get("error"):
        return None
    return bool(row.get("correct_path"))


def load_traces(path: Path) -> dict[str, dict[int, bool]]:
    by_arm: dict[str, dict[int, bool]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scored = hit(row)
        if scored is None:
            continue
        arm = str(row.get("row") or "")
        index = int(row.get("query_index") or 0)
        by_arm[arm][index] = scored
    return dict(by_arm)


def contrast(method: dict[int, bool], draft: dict[int, bool]) -> dict:
    shared = sorted(set(method) & set(draft))
    both = method_only = draft_only = neither = 0
    for index in shared:
        m, d = method[index], draft[index]
        if m and d:
            both += 1
        elif m:
            method_only += 1
        elif d:
            draft_only += 1
        else:
            neither += 1
    n = len(shared)
    disc = method_only + draft_only
    return {
        "n": n,
        "both": both,
        "method_only": method_only,
        "draft_only": draft_only,
        "neither": neither,
        "win_pct_over_n": round(100.0 * method_only / n, 1) if n else None,
        "pairwise_win_pct": round(100.0 * method_only / disc, 1) if disc else None,
        "delta_pp": round(100.0 * (method_only - draft_only) / n, 1) if n else None,
    }


def seed_dirs(pattern_prefix: str) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in sorted(ROOT.glob(pattern_prefix + "*")):
        traces = path / "traces.jsonl"
        if not traces.exists():
            continue
        name = path.name
        seed = None
        if name.endswith(tuple(f"_s{i:02d}" for i in range(20))):
            seed = int(name.rsplit("_s", 1)[1])
        elif name in {"restbench_spotify_ling", "restbench_spotify_flash"}:
            seed = 0
        if seed is None:
            continue
        found.append((seed, traces))
    return found


def pack(dirs: list[tuple[int, Path]], methods: tuple[str, ...] = ("DFSDT", "Ours")) -> dict:
    per_seed: dict[str, dict[int, dict]] = {method: {} for method in methods}
    pooled_m: dict[str, dict[int, list[int]]] = {
        method: defaultdict(list) for method in methods
    }
    pooled_d: dict[int, list[int]] = defaultdict(list)
    for seed, traces in dirs:
        arms = load_traces(traces)
        draft = arms.get("DRAFT") or {}
        if not draft:
            continue
        for index, value in draft.items():
            pooled_d[index].append(int(value))
        for method in methods:
            other = arms.get(method) or {}
            if not other:
                continue
            per_seed[method][seed] = contrast(other, draft)
            for index, value in other.items():
                pooled_m[method][index].append(int(value))
    out: dict = {"seeds": {}, "pooled": {}}
    for method in methods:
        out["seeds"][method] = {
            str(seed): per_seed[method][seed]
            for seed in sorted(per_seed[method])
        }
        # query-majority not needed: pool every (seed, query) pair
        method_hits: dict[tuple[int, int], bool] = {}
        draft_hits: dict[tuple[int, int], bool] = {}
        # rebuild from traces for honest pooling
    # redo pooling from traces
    method_map: dict[str, dict[int, bool]] = {method: {} for method in methods}
    draft_map: dict[int, bool] = {}
    # flatten with unique keys via offset
    next_key = 0
    key_draft: dict[int, bool] = {}
    key_method: dict[str, dict[int, bool]] = {method: {} for method in methods}
    for seed, traces in dirs:
        arms = load_traces(traces)
        draft = arms.get("DRAFT") or {}
        for index, value in draft.items():
            key = seed * 1000 + index
            key_draft[key] = value
            for method in methods:
                if method in arms and index in arms[method]:
                    key_method[method][key] = arms[method][index]
            next_key = key
    out["pooled"] = {
        method: contrast(key_method[method], key_draft) for method in methods
    }
    out["n_seeds"] = sorted({seed for seed, _ in dirs})
    return out


def main() -> int:
    tmdb_ling = [(i, ROOT / f"restbench_tmdb_ling_s{i:02d}" / "traces.jsonl") for i in range(1, 10)]
    tmdb_flash = [(i, ROOT / f"restbench_tmdb_flash_s{i:02d}" / "traces.jsonl") for i in range(1, 10)]
    tmdb_ling = [(s, p) for s, p in tmdb_ling if p.exists()]
    tmdb_flash = [(s, p) for s, p in tmdb_flash if p.exists()]
    def spotify_seed_paths(alias: str) -> list[tuple[int, Path]]:
        found: list[tuple[int, Path]] = []
        for seed in range(10):
            numbered = ROOT / f"restbench_spotify_{alias}_s{seed:02d}" / "traces.jsonl"
            legacy = ROOT / f"restbench_spotify_{alias}" / "traces.jsonl"
            path = numbered if numbered.exists() else (legacy if seed == 0 else None)
            if path is None or not path.exists():
                continue
            arms = load_traces(path)
            draft = arms.get("DRAFT") or {}
            if not draft:
                continue
            # A 402/transport wipe writes traces with zero hits on every arm.
            # Those cells are not a seed; pooling them would invent a fake n.
            hits = sum(1 for value in draft.values() if value)
            if hits == 0:
                continue
            found.append((seed, path))
        return found

    spotify = {
        "ling": spotify_seed_paths("ling"),
        "flash": spotify_seed_paths("flash"),
    }
    report = {
        "metric": (
            "CP pairwise vs DRAFT on cells that finished without a harness "
            "error. The released driver resamples a bad completion and then "
            "continues; a leftover error is dropped from n, not scored as a "
            "miss. win_pct_over_n = method_only / n. pairwise_win_pct = "
            "method_only / (method_only + draft_only)."
        ),
        "tmdb": {
            "ling": pack(tmdb_ling),
            "flash": pack(tmdb_flash),
        },
        "spotify": {
            "ling": pack([(s, p) for s, p in spotify["ling"] if p.exists()]),
            "flash": pack([(s, p) for s, p in spotify["flash"] if p.exists()]),
        },
    }
    out = ROOT / "win_vs_draft.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
