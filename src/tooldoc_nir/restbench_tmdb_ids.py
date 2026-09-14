"""Frozen RestBench-TMDB identifiers.

Named-entity ids (Sofia Coppola, Titanic, Star Wars collection) do not depend
on the evaluation seed. List endpoints (popular, trending, top-rated) have no
frozen hit: a gold-path id is allowed only if it appeared in a producer
response of the same trace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tooldoc_nir.restbench_ours import hop_params
from tooldoc_nir.restbench_screen import (
    ARM_RECORD_DIRS,
    iter_executed_calls,
    load_execute_record,
)

REPO = Path(__file__).resolve().parents[2]
GOLD_PATH = REPO / "artifacts" / "documentation" / "TMDB_gold_ids.json"

ID_PARAMS = frozenset(
    {
        "movie_id",
        "tv_id",
        "person_id",
        "collection_id",
        "company_id",
        "review_id",
        "credit_id",
        "network_id",
        "season_number",
        "episode_number",
    }
)
SEARCH_TYPES = {
    "GET_search_movie": "movie",
    "GET_search_person": "person",
    "GET_search_tv": "tv",
    "GET_search_collection": "collection",
    "GET_search_company": "company",
}
LIST_PRODUCERS = frozenset(
    {
        "GET_movie_popular",
        "GET_movie_now_playing",
        "GET_movie_top_rated",
        "GET_movie_upcoming",
        "GET_movie_latest",
        "GET_tv_popular",
        "GET_tv_top_rated",
        "GET_tv_on_the_air",
        "GET_tv_airing_today",
        "GET_tv_latest",
        "GET_person_popular",
        "GET_trending_media_type_time_window",
        "GET_discover_movie",
        "GET_discover_tv",
    }
)
PARAM_BUCKET = {
    "movie_id": "movie_ids",
    "tv_id": "tv_ids",
    "person_id": "person_ids",
    "collection_id": "collection_ids",
    "company_id": "company_ids",
    "review_id": "review_ids",
    "credit_id": "credit_ids",
    "network_id": "network_ids",
    "season_number": "season_numbers",
    "episode_number": "episode_numbers",
}
QUOTED_RE = re.compile(r'"([^"]{2,80})"|“([^”]{2,80})”')
COLLECTION_RE = re.compile(
    r"collection (?:called |named )?(?:the )?(.+?)(?:\.|$)", re.I
)
DIRECTED_RE = re.compile(
    r"(?:directed by|actor|actress)\s+([A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+)+)"
)
POSSESSIVE_RE = re.compile(
    r"([A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+)+)'s"
)
COMPANY_RE = re.compile(
    r"(?:company |logo of (?:the )?|for )([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)"
)
TITLE_HINTS: tuple[tuple[str, str, str], ...] = (
    ("sofia coppola", "person", "Sofia Coppola"),
    ("dark knight", "movie", "The Dark Knight"),
    ("star wars", "collection", "Star Wars"),
    ("walt disney", "company", "Walt Disney"),
    ("titanic", "movie", "Titanic"),
    ("christopher nolan", "person", "Christopher Nolan"),
    ("martin scorsese", "person", "Martin Scorsese"),
    ("leonardo dicaprio", "person", "Leonardo DiCaprio"),
    ("catherine hardwicke", "person", "Catherine Hardwicke"),
    ("breaking bad", "tv", "Breaking Bad"),
    ("twilight", "movie", "Twilight"),
    ("the matrix", "movie", "The Matrix"),
    ("clint eastwood", "person", "Clint Eastwood"),
    ("francis ford coppola", "person", "Francis Ford Coppola"),
    ("paramount pictures", "company", "Paramount Pictures"),
    ("universal pictures", "company", "Universal Pictures"),
    ("harry potter", "collection", "Harry Potter"),
    ("hunger games", "collection", "The Hunger Games"),
    ("the hobbit", "collection", "The Hobbit"),
    ("fast and the furious", "collection", "The Fast and the Furious"),
    ("lord of the rings", "collection", "The Lord of the Rings"),
    ("lord of the ring", "movie", "The Lord of the Rings"),
    ("house of cards", "tv", "House of Cards"),
    ("django unchained", "movie", "Django Unchained"),
    ("the last of us", "tv", "The Last of Us"),
    ("friends", "tv", "Friends"),
    ("2 broke girls", "tv", "2 Broke Girls"),
    ("big bang theory", "tv", "The Big Bang Theory"),
    ("westworld", "tv", "Westworld"),
    ("game of thrones", "tv", "Game of Thrones"),
    ("band of brothers", "tv", "Band of Brothers"),
    ("the mandalorian", "tv", "The Mandalorian"),
    ("mandalorian", "tv", "The Mandalorian"),
    ("jeremy clarkson", "person", "Jeremy Clarkson"),
    ("black mirror", "tv", "Black Mirror"),
    ("cate blanchett", "person", "Cate Blanchett"),
    ("david schwimmer", "person", "David Schwimmer"),
    ("avatar: the way of water", "movie", "Avatar: The Way of Water"),
    ("avatar", "movie", "Avatar"),
    ("shawshank", "movie", "The Shawshank Redemption"),
    ("double life of veronique", "movie", "The Double Life of Veronique"),
    ("mulholland drive", "movie", "Mulholland Drive"),
    ("twin peaks", "tv", "Twin Peaks"),
    ("akira kurosawa", "person", "Akira Kurosawa"),
    ("spielberg", "person", "Steven Spielberg"),
    ("scarlett johansson", "person", "Scarlett Johansson"),
    ("sword art online", "tv", "Sword Art Online"),
    ("yui aragaki", "person", "Yui Aragaki"),
    ("gen hoshino", "person", "Gen Hoshino"),
    ("we married as job", "movie", "We Married as a Job"),
    ("barbie", "movie", "Barbie"),
    ("death note", "tv", "Death Note"),
    ("katherine lanasa", "person", "Katherine LaNasa"),
    ("oppenheimer", "movie", "Oppenheimer"),
    ("the witcher", "tv", "The Witcher"),
)


def load_gold_ids(path: Path = GOLD_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_named_terms(question: str, gold: Sequence[str]) -> list[tuple[str, str]]:
    """(search_type, query) pairs implied by the question and gold names."""
    text = question or ""
    lower = text.lower()
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, query: str) -> None:
        query = query.strip(" .,'\"")
        if len(query) < 2:
            return
        key = (kind, query.lower())
        if key in seen:
            return
        seen.add(key)
        found.append((kind, query))

    for needle, kind, query in TITLE_HINTS:
        if needle in lower:
            add(kind, query)
    for left, right in QUOTED_RE.findall(text):
        span = left or right
        kind = next((SEARCH_TYPES[name] for name in gold if name in SEARCH_TYPES), "movie")
        add(kind, span)
    match = COLLECTION_RE.search(text)
    if match and any(name == "GET_search_collection" for name in gold):
        add("collection", match.group(1))
    match = DIRECTED_RE.search(text)
    if match:
        add("person", match.group(1))
    match = POSSESSIVE_RE.search(text)
    if match:
        add("person", match.group(1))
    if any(name == "GET_search_company" or name == "GET_company_company_id" for name in gold):
        match = COMPANY_RE.search(text)
        if match:
            add("company", match.group(1))
    return found


def empty_buckets() -> dict[str, set[str]]:
    return {name: set() for name in PARAM_BUCKET.values()}


def _add(buckets: dict[str, set[str]], name: str, value: Any) -> None:
    if value is None or value == "":
        return
    buckets[name].add(str(value))


def harvest_payload(payload: Any, buckets: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    found = buckets if buckets is not None else empty_buckets()
    if isinstance(payload, list):
        for item in payload:
            harvest_payload(item, found)
        return found
    if not isinstance(payload, dict):
        return found
    results = payload.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            media = str(item.get("media_type") or "")
            ident = item.get("id")
            if media == "person" or "known_for" in item:
                _add(found, "person_ids", ident)
            elif media == "tv" or "first_air_date" in item or "name" in item and "title" not in item:
                _add(found, "tv_ids", ident)
            else:
                _add(found, "movie_ids", ident)
            if isinstance(item.get("known_for"), list):
                harvest_payload(item["known_for"], found)
    parts = payload.get("parts")
    if isinstance(parts, list):
        for item in parts:
            if isinstance(item, dict):
                _add(found, "movie_ids", item.get("id"))
    for field, bucket in (
        ("cast", "person_ids"),
        ("crew", "person_ids"),
        ("production_companies", "company_ids"),
        ("networks", "network_ids"),
        ("seasons", "season_numbers"),
        ("episodes", "episode_numbers"),
        ("guest_stars", "person_ids"),
    ):
        for item in payload.get(field) or []:
            if not isinstance(item, dict):
                continue
            if bucket == "season_numbers":
                _add(found, bucket, item.get("season_number"))
            elif bucket == "episode_numbers":
                _add(found, bucket, item.get("episode_number"))
            elif bucket == "person_ids":
                _add(found, bucket, item.get("id"))
                _add(found, "credit_ids", item.get("credit_id"))
            else:
                _add(found, bucket, item.get("id"))
    if "title" in payload and payload.get("id") is not None and "parts" not in payload:
        _add(found, "movie_ids", payload.get("id"))
    if "first_air_date" in payload or (
        "name" in payload and "seasons" in payload and payload.get("id") is not None
    ):
        _add(found, "tv_ids", payload.get("id"))
    if payload.get("credit_id"):
        _add(found, "credit_ids", payload.get("credit_id"))
    return found


ID_IN_TEXT = re.compile(r'"id"\s*:\s*(\d+)')


def harvest_text(text: str) -> dict[str, set[str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if payload is not None:
        return harvest_payload(payload)
    found = empty_buckets()
    for ident in ID_IN_TEXT.findall(text or ""):
        for bucket in (
            "movie_ids",
            "tv_ids",
            "person_ids",
            "collection_ids",
            "company_ids",
            "review_ids",
        ):
            found[bucket].add(ident)
    return found


def harvest_from_call(name: str, body: str) -> dict[str, set[str]]:
    found = harvest_text(body)
    if name == "GET_search_collection":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return found
        for item in (payload.get("results") or []) if isinstance(payload, dict) else []:
            if isinstance(item, dict):
                _add(found, "collection_ids", item.get("id"))
                found["tv_ids"].discard(str(item.get("id") or ""))
                found["movie_ids"].discard(str(item.get("id") or ""))
    if name == "GET_search_company":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return found
        for item in (payload.get("results") or []) if isinstance(payload, dict) else []:
            if isinstance(item, dict):
                _add(found, "company_ids", item.get("id"))
                found["tv_ids"].discard(str(item.get("id") or ""))
    return found


def harvest_producer_ids(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    gold_names: Sequence[str],
    skip_index: int,
) -> dict[str, set[str]]:
    found = empty_buckets()
    allowed = set(gold_names)
    for index, (api, result) in enumerate(pairs):
        if index == skip_index:
            continue
        name = str(api.get("api_name") or api.get("tool_name") or "")
        if name not in allowed:
            continue
        harvested = harvest_from_call(name, str(result.get("response") or ""))
        for bucket, values in harvested.items():
            found[bucket].update(values)
    return found


def frozen_ids_for_query(
    question: str, gold: Sequence[str], table: Mapping[str, Any]
) -> dict[str, set[str]]:
    found = empty_buckets()
    named = table.get("named") or {}
    lower = (question or "").lower()
    for needle, key in table.get("hints") or []:
        if needle in lower:
            row = named.get(key) or {}
            for bucket in found:
                found[bucket].update(str(item) for item in (row.get(bucket) or []) if item)
    by_query = table.get("by_query") or {}
    # Prefer exact question match when present.
    row = by_query.get(question) or {}
    for bucket in found:
        found[bucket].update(str(item) for item in (row.get(bucket) or []) if item)
    for token in re.findall(r"\b(\d{1,2})\b", question or ""):
        found["season_numbers"].add(token)
        found["episode_numbers"].add(token)
    return found


def emitted_identifiers(api: Mapping[str, Any], url: str = "") -> dict[str, str]:
    params = api.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    emitted: dict[str, str] = {}
    names = set(ID_PARAMS)
    if url:
        names.update(hop_params(url))
    for key, value in params.items():
        name = str(key).lower()
        if name not in names:
            continue
        emitted[name] = "" if value is None else str(value)
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
        if not emitted:
            continue
        harvest = harvest_producer_ids(
            pairs, gold_names=gold_names, skip_index=index
        )
        for param, value in emitted.items():
            if value == "":
                misses += 1
                break
            bucket = PARAM_BUCKET.get(param)
            if bucket is None:
                continue
            allowed = freeze[bucket] | harvest[bucket]
            if value not in allowed:
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
