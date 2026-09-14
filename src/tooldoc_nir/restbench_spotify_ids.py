"""Frozen RestBench-Spotify identifiers.

User id, the first playlist, and catalog hits for named songs/albums/artists
do not depend on the evaluation seed. A gold-path call is a fill miss when an
identifier it emitted is missing, or is not in that frozen set and did not
appear in a producer response of the same trace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tooldoc_nir.restbench_screen import (
    ARM_RECORD_DIRS,
    iter_executed_calls,
    load_execute_record,
)

REPO = Path(__file__).resolve().parents[2]
GOLD_PATH = REPO / "artifacts" / "documentation" / "Spotify_gold_ids.json"
ID_PARAMS = frozenset(
    {"id", "ids", "user_id", "playlist_id", "uri", "uris", "context_uri"}
)
ID_TOKEN = re.compile(r"[A-Za-z0-9]{16,}")
URI_TOKEN = re.compile(r"spotify:[a-z]+:[A-Za-z0-9]+")
CATALOG_HINTS: tuple[tuple[str, str], ...] = (
    ("mariah", "mariah_carey"),
    ("dark side", "dark_side"),
    ("pink floyd", "dark_side"),
    ("summertime", "summertime_sadness"),
    ("lana del rey", "summertime_sadness"),
    ("taylor swift", "taylor_swift"),
    ("mojito", "jay_chou_mojito"),
    ("jay chou", "jay_chou_mojito"),
    ("beatle", "beatles"),
    ("bigbang", "bigbang"),
    ("big bang", "bigbang"),
    ("when you believe", "when_you_believe"),
    ("maroon 5", "maroon_5"),
    ("quiet songs", "quiet_songs"),
)
_NAMED_HINTS = frozenset(hint for hint, key in CATALOG_HINTS if key != "quiet_songs")


def load_gold_ids(path: Path = GOLD_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens_from_value(value: Any) -> set[str]:
    found: set[str] = set()
    if value is None:
        return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found.update(_tokens_from_value(item))
        return found
    text = str(value)
    found.update(URI_TOKEN.findall(text))
    for token in ID_TOKEN.findall(text):
        found.add(token)
        found.add(f"spotify:track:{token}")
        found.add(f"spotify:album:{token}")
        found.add(f"spotify:artist:{token}")
        found.add(f"spotify:playlist:{token}")
        found.add(f"spotify:user:{token}")
    return found


def _tokens_from_blob(text: str) -> set[str]:
    return _tokens_from_value(text)


def _named_catalog_query(question: str) -> bool:
    text = (question or "").lower()
    return any(hint in text for hint in _NAMED_HINTS)


def harvest_producer_ids(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    gold_names: Sequence[str],
    skip_index: int,
    include_search: bool,
) -> set[str]:
    """Ids from other gold-path producers in this trace, not from the call under test.

    GET_search is skipped on named-entity queries so a docs-example URI cannot
    hide behind unrelated search hits. Account list endpoints stay in so
    historical playlist / following / library ids survive later account mutation.
    """
    allowed_names = set(gold_names)
    found: set[str] = set()
    for index, (api, result) in enumerate(pairs):
        if index == skip_index:
            continue
        name = str(api.get("api_name") or api.get("tool_name") or "")
        if name not in allowed_names:
            continue
        if name == "GET_search" and not include_search:
            continue
        found.update(_tokens_from_blob(str(result.get("response") or "")))
    return found


def harvest_trace_ids(record: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for api, result in iter_executed_calls(record):
        name = str(api.get("api_name") or api.get("tool_name") or "")
        if name == "GET_search":
            continue
        found.update(_tokens_from_blob(str(result.get("response") or "")))
    return found


def frozen_ids_for_query(
    question: str, gold: Sequence[str], table: Mapping[str, Any]
) -> set[str]:
    allowed: set[str] = set()
    user_id = str(table.get("user_id") or "")
    if user_id:
        allowed.update(_tokens_from_value(user_id))
    playlists = table.get("playlists") or {}
    allowed.update(_tokens_from_value(playlists.get("first")))
    allowed.update(_tokens_from_value(playlists.get("ordered_ids")))
    allowed.update(_tokens_from_value((playlists.get("by_name") or {}).values()))
    text = (question or "").lower()
    catalog = table.get("catalog") or {}
    for hint, key in CATALOG_HINTS:
        if hint in text:
            row = catalog.get(key) or {}
            allowed.update(_tokens_from_value(row.get("ids")))
            allowed.update(_tokens_from_value(row.get("uris")))
    return {token for token in allowed if token}


def emitted_identifiers(api: Mapping[str, Any]) -> dict[str, str]:
    params = api.get("parameters") or {}
    if not isinstance(params, dict):
        return {}
    emitted: dict[str, str] = {}
    for key, value in params.items():
        name = str(key).lower()
        if name not in ID_PARAMS:
            continue
        if value is None or value == "":
            emitted[name] = ""
        else:
            emitted[name] = str(value)
    return emitted


def gold_id_misses(
    record: Mapping[str, Any],
    *,
    question: str,
    gold: Sequence[str],
    table: Mapping[str, Any],
) -> int:
    gold_names = [str(name) for name in gold if name]
    if not gold_names:
        return 0
    freeze = frozen_ids_for_query(question, gold_names, table)
    pairs = iter_executed_calls(record)
    include_search = not _named_catalog_query(question)
    misses = 0
    for index, (api, result) in enumerate(pairs):
        name = str(api.get("api_name") or api.get("tool_name") or "")
        if name not in gold_names:
            continue
        error = str(result.get("error") or "")
        emitted = emitted_identifiers(api)
        if "missing path parameter" in error.lower():
            misses += 1
            continue
        allowed = freeze | harvest_producer_ids(
            pairs,
            gold_names=gold_names,
            skip_index=index,
            include_search=include_search,
        )
        for value in emitted.values():
            tokens = _tokens_from_value(value)
            if not tokens or not tokens.issubset(allowed):
                misses += 1
                break
    return misses


def attach_id_errors(
    rows: Sequence[Mapping[str, Any]],
    run_root: Path,
    table: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    gold = table if table is not None else load_gold_ids()
    attached: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        folder = ARM_RECORD_DIRS.get(str(row.get("row") or ""))
        if folder is not None:
            record = load_execute_record(
                run_root
                / folder
                / f"q{int(row['query_index']):03d}"
                / "record.jsonl"
            )
            if record is not None:
                item["id_errors"] = gold_id_misses(
                    record,
                    question=str(row.get("question") or ""),
                    gold=row.get("gold") or [],
                    table=gold,
                )
        attached.append(item)
    return attached
