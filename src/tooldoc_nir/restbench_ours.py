"""Build RestBench-TMDB Ours docs on top of a released base (DRAFT or Initial).

The first Ours write lifted every URL placeholder into required_parameters and
appended the same "do not invent identifiers" line to all 54 APIs. That lost
to DFSDT: the agent repeated search/list and skipped the ID hop.

This write leaves the schema and the base purpose alone. Hop APIs get a
fill contract (which prior call yields the path id). Catalog APIs, and hops
that themselves emit further ids, get a response contract (which JSON
fields the payload contains, which downstream hops those ids feed).

choose_tool sees the Initial purpose on hop APIs. Catalog APIs append one
scope sentence to `tool_description` so selection knows a list stub is not
credits/images/HQ. Fill and the full response line live in the guideline
(`description`) for choose_parameter. No "call this after" / hop-gate.
No queries, gold paths or eval traces are read.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from urllib.parse import urlparse

from tooldoc_nir.restbench_http import TmdbClient, https_url, path_param_names

ENUM_PARAMS = frozenset({"media_type", "time_window", "type"})
ID_FIELD_RE = re.compile(r'"id"\s*:\s*(\d+)')

# Catalog APIs that return an entity id. Names come from TMDB_Initial URLs,
# not from RestBench gold paths.
PRODUCERS: dict[str, list[str]] = {
    "movie_id": [
        "GET_search_movie",
        "GET_discover_movie",
        "GET_movie_popular",
        "GET_movie_now_playing",
        "GET_movie_top_rated",
        "GET_movie_upcoming",
        "GET_movie_latest",
        "GET_trending_media_type_time_window",
    ],
    "tv_id": [
        "GET_search_tv",
        "GET_discover_tv",
        "GET_tv_popular",
        "GET_tv_top_rated",
        "GET_tv_on_the_air",
        "GET_tv_airing_today",
        "GET_tv_latest",
        "GET_trending_media_type_time_window",
    ],
    "person_id": [
        "GET_search_person",
        "GET_person_popular",
        "GET_trending_media_type_time_window",
    ],
    "company_id": [
        "GET_search_company",
        "GET_movie_movie_id",
        "GET_tv_tv_id",
    ],
    "collection_id": ["GET_search_collection"],
    "network_id": ["GET_tv_tv_id"],
    "review_id": ["GET_movie_movie_id_reviews", "GET_tv_tv_id_reviews"],
    "credit_id": [
        "GET_movie_movie_id_credits",
        "GET_tv_tv_id_credits",
        "GET_tv_tv_id_season_season_number_credits",
        "GET_tv_tv_id_season_season_number_episode_episode_number_credits",
    ],
    "season_number": ["GET_tv_tv_id"],
    "episode_number": ["GET_tv_tv_id_season_season_number"],
}

ID_FIELD: dict[str, str] = {
    "movie_id": "results[i].id",
    "tv_id": "results[i].id",
    "person_id": "results[i].id",
    "company_id": "results[i].id or production_companies[i].id",
    "collection_id": "results[i].id",
    "network_id": "networks[i].id",
    "review_id": "results[i].id",
    "credit_id": "cast[i].credit_id or crew[i].credit_id",
    "season_number": "seasons[i].season_number or the question",
    "episode_number": "episodes[i].episode_number or the question",
}

# Observed list-payload fields. Names come from TMDB/Spotify catalogs and
# live probes, not from gold paths.
CATALOG_FIELDS: dict[str, str] = {
    "GET_search_movie": "results[i].{id,title,release_date,overview}",
    "GET_search_tv": "results[i].{id,name,first_air_date,overview}",
    "GET_search_person": "results[i].{id,name,known_for} (known_for is a stub)",
    "GET_search_company": "results[i].{id,name}",
    "GET_search_collection": "results[i].{id,name}",
    "GET_search": "albums.items / artists.items / tracks.items (id, name)",
    "GET_movie_popular": "results[i].{id,title,release_date,overview}",
    "GET_movie_now_playing": "results[i].{id,title,release_date,overview}",
    "GET_movie_top_rated": "results[i].{id,title,release_date,overview}",
    "GET_movie_upcoming": "results[i].{id,title,release_date,overview}",
    "GET_tv_popular": "results[i].{id,name,first_air_date,overview}",
    "GET_tv_top_rated": "results[i].{id,name,first_air_date,overview}",
    "GET_tv_on_the_air": "results[i].{id,name,first_air_date,overview}",
    "GET_tv_airing_today": "results[i].{id,name,first_air_date,overview}",
    "GET_person_popular": "results[i].{id,name,known_for} (known_for is a stub)",
    "GET_trending_media_type_time_window": "results[i].{id,title|name}",
    "GET_movie_movie_id": "id, title, release_date, production_companies[i].id",
    "GET_tv_tv_id": "id, name, networks[i].{id,name}, production_companies[i].id",
    "GET_collection_collection_id": "id, name, parts[i].id, production_companies",
    "GET_person_person_id": "id, name, biography",
}

SEED_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("GET_search_movie", {"query": "Inception"}),
    ("GET_search_tv", {"query": "Game of Thrones"}),
    ("GET_search_person", {"query": "Tom Hanks"}),
    ("GET_search_company", {"query": "Disney"}),
    ("GET_search_collection", {"query": "Star Wars"}),
)

FOLLOWUP_CALLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GET_movie_movie_id_credits", ("movie_id",)),
    ("GET_movie_movie_id_reviews", ("movie_id",)),
    ("GET_tv_tv_id", ("tv_id",)),
    ("GET_tv_tv_id_season_season_number", ("tv_id", "season_number")),
)


def _tool_index(initial: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(doc.get("tool_name") or ""): doc for doc in initial.values()}


def hop_params(url: str) -> list[str]:
    return [name for name in path_param_names(url) if name not in ENUM_PARAMS]


def _present(catalog: dict[str, dict[str, Any]], names: list[str]) -> list[str]:
    return [name for name in names if name in catalog]


def entity_from_url(url: str) -> str | None:
    path = urlparse(https_url(url)).path
    parts = [part for part in path.split("/") if part]
    for index, part in enumerate(parts):
        if part.startswith("{") and index:
            return parts[index - 1]
    return None


def inferred_producers(
    param: str, url: str, catalog: dict[str, dict[str, Any]]
) -> list[str]:
    known = _present(catalog, PRODUCERS.get(param, []))
    if known:
        return known
    entity = (entity_from_url(url) or "").lower()
    search_hits: list[str] = []
    other: list[str] = []
    for name, document in catalog.items():
        if hop_params(str(document.get("url") or "")):
            continue
        source_url = str(document.get("url") or "").lower()
        if "/me" in source_url:
            continue
        lower = name.lower()
        if "search" in lower:
            search_hits.append(name)
        elif entity and entity.rstrip("s") in lower:
            other.append(name)
    return search_hits + other


def _parse_body(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _first_id(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("id"), int):
        return int(payload["id"])
    results = payload.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        value = results[0].get("id")
        if isinstance(value, int):
            return value
    return None


def _nested_id(payload: Any, *path: str) -> int | None:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if isinstance(node, list) and node and isinstance(node[0], dict):
        value = node[0].get("id") or node[0].get("credit_id")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value:
            return None
    if isinstance(node, int):
        return node
    return None


def _nested_text(payload: Any, *path: str) -> str | None:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if isinstance(node, list) and node and isinstance(node[0], dict):
        value = node[0].get("credit_id") or node[0].get("id")
        if value is not None:
            return str(value)
    if node is not None:
        return str(node)
    return None


def _id_from_text(text: str) -> int | None:
    payload = _parse_body(text)
    found = _first_id(payload)
    if found is not None:
        return found
    match = ID_FIELD_RE.search(text)
    return int(match.group(1)) if match else None


def probe_ids(
    initial: dict[str, dict[str, Any]],
    client: TmdbClient,
) -> tuple[dict[str, str], dict[str, Any]]:
    catalog = _tool_index(initial)
    ids: dict[str, str] = {}
    stats = {"ok": 0, "fail": 0, "calls": 0}

    def call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        document = catalog.get(tool_name)
        if document is None:
            return None
        stats["calls"] += 1
        try:
            row = client.call(
                url_template=str(document.get("url") or ""),
                arguments=arguments,
            )
        except (ValueError, OSError, TypeError):
            stats["fail"] += 1
            return None
        if row.get("error"):
            stats["fail"] += 1
            return None
        stats["ok"] += 1
        return row

    for tool_name, arguments in SEED_CALLS:
        row = call(tool_name, arguments)
        if row is None:
            continue
        found = _id_from_text(str(row.get("response") or ""))
        entity = {
            "GET_search_movie": "movie_id",
            "GET_search_tv": "tv_id",
            "GET_search_person": "person_id",
            "GET_search_company": "company_id",
            "GET_search_collection": "collection_id",
        }[tool_name]
        if found is not None:
            ids[entity] = str(found)

    if "tv_id" in ids and "season_number" not in ids:
        ids["season_number"] = "1"
    if "episode_number" not in ids:
        ids["episode_number"] = "1"

    for tool_name, needed in FOLLOWUP_CALLS:
        if any(name not in ids for name in needed):
            continue
        row = call(tool_name, {name: ids[name] for name in needed})
        if row is None:
            continue
        payload = _parse_body(str(row.get("response") or ""))
        if tool_name == "GET_movie_movie_id_credits":
            credit = _nested_text(payload, "cast")
            if credit:
                ids["credit_id"] = credit
        elif tool_name == "GET_movie_movie_id_reviews":
            review = _first_id(payload)
            if review is not None:
                ids["review_id"] = str(review)
        elif tool_name == "GET_tv_tv_id":
            network = _nested_id(payload, "networks")
            if network is not None:
                ids["network_id"] = str(network)
            seasons = payload.get("seasons") if isinstance(payload, dict) else None
            if isinstance(seasons, list) and seasons:
                for season in seasons:
                    if isinstance(season, dict) and season.get("season_number") not in (
                        None,
                        0,
                    ):
                        ids["season_number"] = str(season["season_number"])
                        break

    return ids, stats


def hop_contract(
    document: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    ids: dict[str, str],
) -> str | None:
    params = hop_params(str(document.get("url") or ""))
    if not params:
        return None
    sources: list[str] = []
    for param in params:
        producers = inferred_producers(param, str(document.get("url") or ""), catalog)
        entity = entity_from_url(str(document.get("url") or ""))
        field = ID_FIELD.get(
            param,
            f"{entity}.items[i].id" if entity else f"{param} from a previous response",
        )
        if producers:
            shown = ", ".join(producers[:3])
            if len(producers) > 3:
                shown += ", ..."
            sources.append(f"{param} from {shown} ({field})")
        else:
            sources.append(f"{param} from a previous response ({field})")
    example_bits = [f"{name}={ids[name]}" for name in params if name in ids]
    # Fill-instruction only. "Call this after" / "list is not this endpoint"
    # leaked into choose_tool and made the agent skip the hop (stuck_first).
    lines = [" ".join(sources)]
    if example_bits:
        lines.append("Example: " + ", ".join(example_bits) + ".")
    return " ".join(lines)


def produced_params(tool_name: str) -> list[str]:
    return [param for param, producers in PRODUCERS.items() if tool_name in producers]


def consumer_apis(
    param: str, catalog: dict[str, dict[str, Any]], limit: int = 4
) -> list[str]:
    names = [
        name
        for name, document in catalog.items()
        if param in hop_params(str(document.get("url") or ""))
    ]

    def rank(name: str) -> tuple[int, str]:
        lower = name.lower()
        if "credit" in lower:
            return (0, name)
        if "image" in lower:
            return (1, name)
        return (2, name)

    names.sort(key=rank)
    return names[:limit]


def response_contract(
    document: dict[str, Any], catalog: dict[str, dict[str, Any]]
) -> str | None:
    name = str(document.get("tool_name") or "")
    produced = produced_params(name)
    hops: list[str] = []
    for param in produced:
        hops.extend(consumer_apis(param, catalog))
    unique: list[str] = []
    for hop in hops:
        if hop not in unique:
            unique.append(hop)
    field = CATALOG_FIELDS.get(name)
    if field is None and produced:
        field = ID_FIELD.get(produced[0], "results[i].id")
    if field is None and not unique:
        return None
    parts: list[str] = []
    if field:
        parts.append(f"Response: {field}.")
    if unique:
        shown = ", ".join(unique[:4])
        if len(unique) > 4:
            shown += ", ..."
        parts.append(f"Downstream: {shown}.")
    return " ".join(parts) if parts else None


def catalog_scope_line(contract: str) -> str:
    """One sentence for choose_tool on catalog APIs. No hop-gate."""
    compact = contract.replace("Response: ", "Returns ").replace("Downstream: ", "not ")
    if len(compact) > 220:
        compact = compact[:217].rstrip(",; ") + "..."
    return compact


def build_ours_instructions(
    base: dict[str, dict[str, Any]],
    *,
    client: TmdbClient | None = None,
    base_name: str = "base",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    catalog = _tool_index(base)
    ids: dict[str, str] = {}
    probe = {"ok": 0, "fail": 0, "calls": 0}
    if client is not None and "GET_search_movie" in catalog:
        ids, probe = probe_ids(base, client)

    ours: dict[str, dict[str, Any]] = {}
    hops = 0
    responses = 0
    scopes = 0
    examples = 0
    untouched = 0
    for key, document in base.items():
        updated = copy.deepcopy(document)
        original = str(updated.get("description") or "").rstrip()
        updated["tool_description"] = original
        extra: list[str] = []
        fill = hop_contract(updated, catalog, ids)
        if fill:
            extra.append(fill)
            hops += 1
            if "Example:" in fill:
                examples += 1
        response = response_contract(updated, catalog)
        if response:
            extra.append(response)
            responses += 1
            if not hop_params(str(updated.get("url") or "")):
                updated["tool_description"] = f"{original} {catalog_scope_line(response)}".strip()
                scopes += 1
        if extra:
            updated["description"] = f"{original}\n\n" + "\n".join(extra)
        else:
            untouched += 1
        ours[key] = updated

    report = {
        "n": len(ours),
        "base": base_name,
        "hops_annotated": hops,
        "responses_annotated": responses,
        "catalog_scopes": scopes,
        "catalogs_untouched": untouched,
        "examples_from_probe": examples,
        "schema_changed": False,
        "prose_rewritten": False,
        "hop_style": "fill",
        "choose_tool_text": "base_purpose_on_hops_scope_on_catalogs",
        "gold_used": False,
        "probe": probe,
        "seed_ids": ids,
    }
    return ours, report
