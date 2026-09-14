"""Slice ToolBench G3 into queries that need an output-to-input transition.

Most released G3 instructions are parallel: every gold API's identifiers are
already in the user text (a YouTube id, a quoted search term, a category name).
Stas asked for the complementary slice: a later required identifier is not in
the query, so the agent has to take it from an earlier response.

Classification reads only the query text, the gold API names, and the
per-query api_list schemas. It does not read traces or gold argument values.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

SKIP_PARAMS = frozenset(
    {
        "format",
        "page",
        "per_page",
        "callback",
        "lang",
        "language",
        "locale",
        "sort",
        "summary_response",
        "full_response",
        "cgeo",
        "api_key",
        "apikey",
        "key",
        "rapidapi_key",
        "host",
        "type",
        "method",
        "max_id",
        "since_id",
        "count",
        "limit",
        "offset",
    }
)

IDENT_RE = re.compile(
    r"(^id$|_id$|Id$|ID$|uri$|url$|username$|user$|channel$|video$|sha$)",
    re.I,
)
QUOTED_RE = re.compile(r"'([^']{1,80})'|\"([^\"]{1,80})\"")


def _api_key(tool: str, api: str) -> tuple[str, str]:
    return (str(tool), str(api))


def gold_pairs(query: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in query.get("relevant APIs") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def _api_index(query: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for api in query.get("api_list") or []:
        if not isinstance(api, dict):
            continue
        key = _api_key(str(api.get("tool_name") or ""), str(api.get("api_name") or ""))
        index[key] = api
    return index


def required_params(api: Mapping[str, Any]) -> list[dict[str, Any]]:
    names: list[dict[str, Any]] = []
    for item in api.get("required_parameters") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name or name.lower() in SKIP_PARAMS:
            continue
        names.append(item)
    return names


def quoted_spans(text: str) -> list[str]:
    found: list[str] = []
    for left, right in QUOTED_RE.findall(text):
        token = left or right
        if token:
            found.append(token)
    return found


def is_identifier(name: str, description: str = "") -> bool:
    if IDENT_RE.search(name):
        return True
    blob = f"{name} {description}".lower()
    return "identifier" in blob or " id " in f" {blob} " or blob.endswith(" id")


def value_in_query(item: Mapping[str, Any], query_text: str) -> bool:
    """Whether the user text already names a fill for this required field."""
    name = str(item.get("name") or "").lower()
    description = str(item.get("description") or "")
    default = str(item.get("default") or "")
    spans = quoted_spans(query_text)
    if default and len(default) >= 3 and default not in {"json", "xml", "php"}:
        if default in query_text:
            return True
    lexical = name in {"query", "q", "search", "keyword", "text", "category", "term"}
    if lexical and spans:
        return True
    if is_identifier(str(item.get("name") or ""), description):
        for span in spans:
            if re.fullmatch(r"[A-Za-z0-9_-]{4,}", span):
                return True
    return False


def classify_query(query: Mapping[str, Any]) -> dict[str, Any]:
    text = str(query.get("query") or "")
    gold = gold_pairs(query)
    index = _api_index(query)
    edges: list[dict[str, str]] = []
    missing_without_producer: list[dict[str, str]] = []

    for later_i, later_key in enumerate(gold):
        later = index.get(later_key)
        if later is None:
            continue
        for item in required_params(later):
            if not is_identifier(
                str(item.get("name") or ""), str(item.get("description") or "")
            ):
                continue
            if value_in_query(item, text):
                continue
            param = str(item.get("name") or "")
            prior = gold[:later_i]
            if prior:
                producer = prior[-1]
                edges.append(
                    {
                        "from_tool": producer[0],
                        "from_api": producer[1],
                        "to_tool": later_key[0],
                        "to_api": later_key[1],
                        "param": param,
                    }
                )
            else:
                missing_without_producer.append(
                    {
                        "to_tool": later_key[0],
                        "to_api": later_key[1],
                        "param": param,
                    }
                )

    if edges:
        label = "nested"
    elif missing_without_producer:
        label = "underspecified"
    else:
        label = "independent"

    return {
        "query_id": query.get("query_id"),
        "n_gold": len(gold),
        "label": label,
        "edges": edges,
        "underspecified": missing_without_producer,
    }


def classify_dataset(queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [classify_query(query) for query in queries]
    buckets: dict[str, list[int]] = {
        "nested": [],
        "independent": [],
        "underspecified": [],
    }
    for index, row in enumerate(rows):
        buckets[str(row["label"])].append(index)
    return {
        "n": len(queries),
        "nested_indices": buckets["nested"],
        "independent_indices": buckets["independent"],
        "underspecified_indices": buckets["underspecified"],
        "counts": {key: len(value) for key, value in buckets.items()},
        "queries": rows,
    }
