"""One-shot live lookup of RestBench-TMDB named-entity ids."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tooldoc_nir.openrouter import load_env_file
from tooldoc_nir.restbench_data import gold_apis, load_queries
from tooldoc_nir.restbench_http import TmdbClient
from tooldoc_nir.restbench_tmdb_ids import TITLE_HINTS, extract_named_terms

OUT = Path("artifacts/documentation/TMDB_gold_ids.json")
SEARCH = {
    "movie": "https://api.themoviedb.org/3/search/movie",
    "person": "https://api.themoviedb.org/3/search/person",
    "tv": "https://api.themoviedb.org/3/search/tv",
    "collection": "https://api.themoviedb.org/3/search/collection",
    "company": "https://api.themoviedb.org/3/search/company",
}
DETAIL = {
    "collection": "https://api.themoviedb.org/3/collection/{collection_id}",
}


def _ids(items: list[Any]) -> list[str]:
    return [str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")]


def _search(client: TmdbClient, kind: str, query: str) -> dict[str, Any]:
    row = client.call(url_template=SEARCH[kind], arguments={"query": query})
    if row.get("error"):
        return {"query": query, "type": kind, "error": row["error"], "ids": []}
    try:
        payload = json.loads(row.get("response") or "{}")
    except json.JSONDecodeError:
        return {"query": query, "type": kind, "error": "bad_json", "ids": []}
    items = [item for item in (payload.get("results") or []) if isinstance(item, dict)]
    out: dict[str, Any] = {
        "query": query,
        "type": kind,
        "ids": _ids(items[:5]),
        "names": [str(item.get("title") or item.get("name") or "") for item in items[:5]],
    }
    if kind == "collection" and out["ids"]:
        detail = client.call(
            url_template=DETAIL["collection"],
            arguments={"collection_id": out["ids"][0]},
        )
        try:
            body = json.loads(detail.get("response") or "{}")
        except json.JSONDecodeError:
            body = {}
        parts = [part for part in (body.get("parts") or []) if isinstance(part, dict)]
        out["movie_ids"] = _ids(parts)
        out["collection_ids"] = out["ids"][:1]
    return out


def _row_from_hits(hits: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "movie_ids": [],
        "tv_ids": [],
        "person_ids": [],
        "collection_ids": [],
        "company_ids": [],
    }
    for hit in hits:
        kind = hit.get("type")
        ids = [str(item) for item in (hit.get("ids") or []) if item]
        if kind == "movie":
            buckets["movie_ids"].extend(ids)
        elif kind == "tv":
            buckets["tv_ids"].extend(ids)
        elif kind == "person":
            buckets["person_ids"].extend(ids)
        elif kind == "company":
            buckets["company_ids"].extend(ids)
        elif kind == "collection":
            buckets["collection_ids"].extend(hit.get("collection_ids") or ids[:1])
            buckets["movie_ids"].extend(str(item) for item in (hit.get("movie_ids") or []) if item)
    for key, values in buckets.items():
        seen: list[str] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        buckets[key] = seen
    return buckets


def main() -> None:
    load_env_file()
    client = TmdbClient(os.environ.get("TMDB_API_KEY", ""))
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    named: dict[str, dict[str, Any]] = {}
    hints: list[list[str]] = []
    for needle, kind, query in TITLE_HINTS:
        key = needle.replace(" ", "_")
        hints.append([needle, key])
        hit = cache.get((kind, query.lower()))
        if hit is None:
            hit = _search(client, kind, query)
            cache[(kind, query.lower())] = hit
        named[key] = {**_row_from_hits([hit]), "search": hit}

    queries = load_queries(Path("external/DRAFT"), "TMDB")
    by_query: dict[str, dict[str, list[str]]] = {}
    for query in queries:
        question = str(query.get("query") or "")
        terms = extract_named_terms(question, gold_apis(query))
        hits: list[dict[str, Any]] = []
        for kind, term in terms:
            hit = cache.get((kind, term.lower()))
            if hit is None:
                hit = _search(client, kind, term)
                cache[(kind, term.lower())] = hit
            hits.append(hit)
        if hits:
            by_query[question] = _row_from_hits(hits)

    payload = {
        "named": named,
        "hints": hints,
        "by_query": by_query,
        "n_named": len(named),
        "n_queries_frozen": len(by_query),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "n_named": payload["n_named"], "n_queries_frozen": payload["n_queries_frozen"]}))


if __name__ == "__main__":
    main()
