"""Stage-localized documentation variants for the cross-model contract pilot.

The frozen `Ours` documentation puts routing facts on every surface at once:
`tool_description` (read by `choose_tool`), the guideline `description` (read
by `task_decompose` and `choose_parameter`) and, on DRAFT, `example`. That is
why one phrase could help Ling and hurt Flash: `not GET_...` gave Ling a
contrast and Flash a prohibition, while `Next: a hop tool` cost Ling seven
queries on s04.

Each variant here writes to exactly one surface, so a pilot can attribute an
effect to a stage instead of to a sentence:

- H1 `witness`     -> `example` only        (choose_parameter, choose_API)
- H2 `fingerprint`  -> `tool_description`    (choose_tool)
- H3 `ports`        -> `description` only    (task_decompose, choose_parameter)
- `fill_desc`       -> `description` only, the deterministic fill line without
  `Response:`, `Downstream:`, negations or probed identifiers

Sources: the released `Initial.json` / `DRAFT.json` URLs, resource types and
parameter schemas, plus optional generic live probes that confirm a response
path exists. No RestBench query, gold path or eval trace is read; the build
asserts that in `assert_source_only`.
"""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from tooldoc_nir.draft_agent_reproduction import _dump_json, _load_json
from tooldoc_nir.provenance import file_digest, git_commit, payload_digest
from tooldoc_nir.restbench_data import (
    DEFAULT_DRAFT_ROOT,
    instruction_path,
    load_instructions,
)
from tooldoc_nir.restbench_http import (
    SpotifyClient,
    TmdbClient,
    https_url,
    path_param_names,
)
from tooldoc_nir.restbench_ours import (
    ENUM_PARAMS,
    POSITIVE_SCOPE_LIMIT,
    hop_contract,
    hop_params,
    positive_scope_line,
    probe_ids,
    response_contract,
)

EXPERIMENT_ROOT = Path("artifacts/documentation/experiments")

SCHEMA_FIELDS = ("ID", "tool_name", "url", "required_parameters", "optional_parameters")

# Values that cannot be real TMDB identifiers, so a literal copy is visible in
# the executed URL instead of hiding behind a plausible id.
SENTINELS: dict[str, Any] = {
    "movie_ref:int": 9999991,
    "tv_ref:int": 9999992,
    "person_ref:int": 9999993,
    "company_ref:int": 9999994,
    "collection_ref:int": 9999995,
    "network_ref:int": 9999996,
    "keyword_ref:int": 9999997,
    "review_ref:str": "ffffffffffffffffffffff01",
    "credit_ref:str": "ffffffffffffffffffffff02",
    "season_number:int": 91,
    "episode_number:int": 92,
}

PARAM_PORT: dict[str, str] = {
    "movie_id": "movie_ref:int",
    "tv_id": "tv_ref:int",
    "person_id": "person_ref:int",
    "company_id": "company_ref:int",
    "collection_id": "collection_ref:int",
    "network_id": "network_ref:int",
    "keyword_id": "keyword_ref:int",
    "review_id": "review_ref:str",
    "credit_id": "credit_ref:str",
    "season_number": "season_number:int",
    "episode_number": "episode_number:int",
    "guest_star_id": "person_ref:int",
}

ENTITY_WORD: dict[str, str] = {
    "movie": "movie",
    "tv": "TV show",
    "person": "person",
    "company": "company",
    "collection": "collection",
    "network": "network",
    "review": "review",
    "credit": "credit",
    "genre": "genre",
    "trending": "title",
}

RANKED_FACETS: dict[str, str] = {
    "popular": "ranked by current popularity",
    "top_rated": "ranked by user rating",
    "now_playing": "currently playing in theaters",
    "upcoming": "scheduled for an upcoming release",
    "on_the_air": "airing during the current week",
    "airing_today": "airing today",
    "latest": "",
}

PLURAL: dict[str, str] = {
    "person": "people",
    "company": "companies",
    "TV show": "TV shows",
    "movie": "movies",
    "collection": "collections",
    "network": "networks",
    "review": "reviews",
    "credit": "credits",
    "title": "titles",
    "genre": "genres",
}

NON_ID_OUTPUT: dict[str, str] = {
    "images": "image file paths",
    "release_dates": "release dates per country",
}

PORT_NOUN: dict[str, str] = {
    "movie_ref": "movie",
    "tv_ref": "TV show",
    "person_ref": "person",
    "company_ref": "company",
    "collection_ref": "collection",
    "network_ref": "network",
    "keyword_ref": "keyword",
    "review_ref": "review",
    "credit_ref": "credit",
}

LIST_SELECTORS = frozenset({"text", "filters", "trending"})

# TMDB identifiers are integers, Spotify's are base62 strings.
ID_TYPES = {"TMDB": "int", "Spotify": "str"}


@dataclass(frozen=True)
class Port:
    """One typed edge of the dataflow graph."""

    name: str
    path: str = ""
    verified: str = "unprobed"

    def render_output(self) -> str:
        return f"OUTPUT {self.name} <- {self.path}"


@dataclass(frozen=True)
class Endpoint:
    tool_name: str
    url: str
    entity: str
    facet: str
    selector: str
    inputs: tuple[str, ...]
    outputs: tuple[Port, ...]


@dataclass(frozen=True)
class VariantSpec:
    """A named, reproducible point in the surface x semantics space."""

    name: str
    base: str = "Initial"
    fingerprint: bool = False
    ports: bool = False
    witness: bool = False
    fill_desc: bool = False
    shuffle_ports: bool = False
    copy_frozen: str = ""
    positive_scope: bool = False
    role_clause: bool = False
    derive_from: str = ""

    def flags(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "fingerprint_in_tool_description": self.fingerprint,
            "ports_in_description": self.ports,
            "witness_in_example": self.witness,
            "fill_line_in_description": self.fill_desc,
            "shuffled_port_control": self.shuffle_ports,
            "copied_frozen_documentation": self.copy_frozen or None,
            "negation_free_catalog_scope": self.positive_scope,
            "role_clause_on_reference_endpoints": self.role_clause,
            "guideline_inherited_from": self.derive_from or None,
        }


FROZEN_OURS = "artifacts/documentation/{dataset}_Ours.json"

VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("Initial"),
    VariantSpec("DRAFT", base="DRAFT"),
    VariantSpec("CurrentNot", copy_frozen=FROZEN_OURS),
    VariantSpec("FillDesc", fill_desc=True),
    VariantSpec("H1", witness=True),
    VariantSpec("H2", fingerprint=True),
    VariantSpec("H12", fingerprint=True, witness=True),
    VariantSpec("H3", ports=True),
    VariantSpec("H123", fingerprint=True, ports=True, witness=True),
    VariantSpec("DRAFT_FillDesc", base="DRAFT", fill_desc=True),
    VariantSpec("H3Shuffled", ports=True, shuffle_ports=True),
    # The two Cross-model Positive Scope candidates. Both keep the frozen
    # guideline `description` -- it is what lets Ling decompose the task and it
    # never reaches `choose_tool` -- and differ from `CurrentNot` only on the
    # selection surface.
    VariantSpec("P", positive_scope=True, derive_from=FROZEN_OURS),
    VariantSpec(
        "PF", positive_scope=True, role_clause=True, derive_from=FROZEN_OURS
    ),
)


def variant_by_name(name: str) -> VariantSpec:
    for spec in VARIANTS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown variant {name!r}; have {[v.name for v in VARIANTS]}")


def tool_index(docs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(doc.get("tool_name") or ""): dict(doc) for doc in docs.values()}


VERSION_SEGMENT = re.compile(r"^v?\d+$")


def path_parts(url: str) -> list[str]:
    """Path segments after the API version, for TMDB `/3/...` and Spotify `/v1/...`."""
    parts = [part for part in urlparse(https_url(url)).path.split("/") if part]
    if parts and VERSION_SEGMENT.match(parts[0]):
        return parts[1:]
    return parts


def _singular(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def port_for_param(param: str, url: str, *, id_type: str = "int") -> str:
    """Named parameters win; anything else takes the type of its own resource.

    TMDB spells the owner into the parameter (`movie_id`), Spotify does not
    (`/albums/{id}`), so the fallback reads the segment that precedes the
    placeholder. Without this, a transfer run would have to be retuned by hand,
    which is exactly what the transfer is supposed to avoid.
    """
    if param in PARAM_PORT:
        return PARAM_PORT[param]
    parts = path_parts(url)
    owner = ""
    for index, part in enumerate(parts):
        if part == "{" + param + "}" and index:
            owner = parts[index - 1]
            break
    if not owner:
        return f"{param}:{id_type}"
    return f"{_singular(owner)}_ref:{id_type}"


def discovered_outputs(payload: Any, *, id_type: str = "str") -> tuple[Port, ...]:
    """Ports read off a probed payload: every `id` field and where it sits.

    Used only for datasets the rule table does not describe, so the TMDB graph
    stays exactly what the frozen pilot variants were compiled from.
    """
    found: dict[str, str] = {}

    def walk(node: Any, path: str, depth: int) -> None:
        if depth > 4 or len(found) > 8:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "id" and isinstance(value, (str, int)):
                    owner = path.split(".")[-1].removesuffix("[*]") or "resource"
                    name = f"{_singular(owner)}_ref:{id_type}"
                    found.setdefault(name, f"{path}.id" if path else "$.id")
                elif isinstance(value, (dict, list)):
                    walk(value, f"{path}.{key}" if path else f"$.{key}", depth + 1)
        elif isinstance(node, list) and node:
            walk(node[0], f"{path}[*]", depth)

    walk(payload, "", 0)
    return tuple(Port(name=name, path=path) for name, path in sorted(found.items()))


def _placeholder(part: str) -> bool:
    return part.startswith("{") and part.endswith("}")


def classify(url: str) -> tuple[str, str, str]:
    """Return (entity, facet, selector) from the URL shape alone."""
    parts = path_parts(url)
    if not parts:
        return "", "", "reference"
    head = parts[0]
    if head in {"search", "discover"} and len(parts) > 1:
        return parts[1], "", "text" if head == "search" else "filters"
    if head == "genre":
        return (parts[1] if len(parts) > 1 else "genre"), "genre_list", "none"
    if head == "trending":
        return "trending", "", "trending"
    tail = parts[-1]
    if len(parts) == 2 and not _placeholder(tail) and tail in RANKED_FACETS:
        return head, "", tail
    concrete = [part for part in parts[1:] if not _placeholder(part)]
    if not concrete:
        nested = [part for part in parts[1:] if _placeholder(part)]
        if len(nested) >= 3:
            return head, "episode", "reference"
        if len(nested) == 2:
            return head, "season", "reference"
        return head, "", "reference"
    facet = concrete[-1]
    if facet in {"season", "episode"}:
        return head, facet, "reference"
    return head, facet, "reference"


def output_ports(endpoint_url: str) -> tuple[Port, ...]:
    """Typed outputs from resource type plus the recorded response shape."""
    entity, facet, selector = classify(endpoint_url)
    ref = PARAM_PORT.get(f"{entity}_id", f"{entity}_ref:int")
    ports: list[Port] = []

    def add(name: str, path: str) -> None:
        if name and all(port.name != name or port.path != path for port in ports):
            ports.append(Port(name=name, path=path))

    if entity == "trending":
        add("movie_ref|tv_ref|person_ref:int", "$.results[*].id")
        return tuple(ports)
    if facet == "genre_list":
        return (Port("genre_id:int", "$.genres[*].id"),)
    if selector in {"text", "filters"} or (facet == "" and selector in RANKED_FACETS):
        add(ref, "$.id" if selector == "latest" else "$.results[*].id")
        return tuple(ports)
    if facet in {"similar", "recommendations"}:
        add(ref, "$.results[*].id")
        return tuple(ports)
    if facet in {"credits", "aggregate_credits"}:
        add("person_ref:int", "$.cast[*].id")
        add("person_ref:int", "$.crew[*].id")
        add("credit_ref:str", "$.cast[*].credit_id")
        return tuple(ports)
    if facet == "movie_credits":
        add("movie_ref:int", "$.cast[*].id")
        add("movie_ref:int", "$.crew[*].id")
        return tuple(ports)
    if facet == "tv_credits":
        add("tv_ref:int", "$.cast[*].id")
        add("tv_ref:int", "$.crew[*].id")
        return tuple(ports)
    if facet == "reviews":
        add("review_ref:str", "$.results[*].id")
        return tuple(ports)
    if facet == "keywords":
        add("keyword_ref:int", "$.keywords[*].id" if entity == "movie" else "$.results[*].id")
        return tuple(ports)
    if facet in NON_ID_OUTPUT:
        return tuple(ports)
    if facet == "season":
        add("episode_number:int", "$.episodes[*].episode_number")
        return tuple(ports)
    if facet == "episode":
        add("person_ref:int", "$.guest_stars[*].id")
        add("credit_ref:str", "$.guest_stars[*].credit_id")
        return tuple(ports)
    if facet:
        return tuple(ports)
    # Detail endpoint: /<resource>/{<resource>_id}
    if entity == "review":
        add("review_ref:str", "$.id")
        add("movie_ref|tv_ref:int", "$.media_id")
        return tuple(ports)
    if entity == "credit":
        add("person_ref:int", "$.person.id")
        add("movie_ref|tv_ref:int", "$.media.id")
        return tuple(ports)
    add(ref, "$.id")
    if entity == "movie":
        add("company_ref:int", "$.production_companies[*].id")
        add("collection_ref:int", "$.belongs_to_collection.id")
    if entity == "tv":
        add("company_ref:int", "$.production_companies[*].id")
        add("network_ref:int", "$.networks[*].id")
        add("season_number:int", "$.seasons[*].season_number")
    if entity == "collection":
        add("movie_ref:int", "$.parts[*].id")
    return tuple(ports)


def input_ports(
    document: Mapping[str, Any], *, id_type: str = "int"
) -> tuple[str, ...]:
    url = str(document.get("url") or "")
    inputs: list[str] = []
    for param in path_param_names(url):
        if param in ENUM_PARAMS:
            continue
        port = port_for_param(param, url, id_type=id_type)
        if port:
            inputs.append(f"INPUT path.{param} <- {port}")
    required = [
        str(item.get("name") or "")
        for item in document.get("required_parameters") or []
    ]
    if "query" in required:
        inputs.append("INPUT query.query <- text")
    return tuple(inputs)


def build_graph(
    base: Mapping[str, Mapping[str, Any]], *, id_type: str = "int"
) -> dict[str, Endpoint]:
    graph: dict[str, Endpoint] = {}
    for document in base.values():
        name = str(document.get("tool_name") or "")
        url = str(document.get("url") or "")
        entity, facet, selector = classify(url)
        graph[name] = Endpoint(
            tool_name=name,
            url=url,
            entity=entity,
            facet=facet,
            selector=selector,
            inputs=input_ports(document, id_type=id_type),
            outputs=output_ports(url),
        )
    return graph


def _recover_json(text: str) -> tuple[Any, bool]:
    """Parse a body the HTTP client truncated, reporting whether it was cut.

    The harness caps every response, so a plain `json.loads` fails on the long
    payloads and the probe would report almost everything as unprobed. Cutting
    back to the last element boundary and closing the open containers keeps the
    prefix, which is where `results` and `cast` live. A missing path in a cut
    payload is inconclusive, not evidence of absence: `crew` and `seasons` sit
    past the budget on exactly the endpoints that matter.
    """
    try:
        return json.loads(text), False
    except (json.JSONDecodeError, TypeError):
        pass
    stack: list[str] = []
    in_string = False
    escaped = False
    boundaries: list[tuple[int, tuple[str, ...]]] = []
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if stack:
                stack.pop()
        elif char == "," and stack:
            boundaries.append((index, tuple(stack)))
    for cut, open_stack in reversed(boundaries[-200:]):
        closers = "".join("]" if item == "[" else "}" for item in reversed(open_stack))
        try:
            return json.loads(text[:cut] + closers), True
        except json.JSONDecodeError:
            continue
    return None, True


def _path_probe(payload: Any, path: str) -> tuple[str, Any]:
    """Classify a `$.a[*].b` path as present, nullable or missing.

    A field the API declares but leaves empty for the probed record — Inception
    has no `belongs_to_collection` — is nullable, not missing. Only a key the
    payload does not have at all is evidence that the edge is wrong.
    """
    node: Any = payload
    for step in path.removeprefix("$.").split("."):
        if step.endswith("[*]"):
            key = step[: -len("[*]")]
            if not isinstance(node, dict) or key not in node:
                return "missing", None
            node = node[key]
            if not isinstance(node, list):
                return "missing", None
            if not node:
                return "nullable", None
            node = node[0]
            continue
        if node is None:
            return "nullable", None
        if not isinstance(node, dict) or step not in node:
            return "missing", None
        node = node[step]
    if node is None:
        return "nullable", None
    return "present", node


def _probe_arguments(
    document: Mapping[str, Any],
    entity: str,
    port_values: Mapping[str, Any],
    *,
    id_type: str = "int",
) -> dict[str, Any] | None:
    """Generic call arguments, or None while some port value is still unknown."""
    arguments: dict[str, Any] = {}
    url = str(document.get("url") or "")
    for param in path_param_names(url):
        if param in ENUM_PARAMS:
            arguments[param] = "movie" if param == "media_type" else "week"
            continue
        port = port_for_param(param, url, id_type=id_type)
        if port not in port_values:
            return None
        arguments[param] = port_values[port]
    if any(
        str(item.get("name")) == "query"
        for item in document.get("required_parameters") or []
    ):
        arguments["query"] = {
            "person": "Tom Hanks",
            "tv": "Game of Thrones",
            "company": "Disney",
            "collection": "Star Wars",
        }.get(entity, "Inception")
    return arguments


def probe_graph(
    base: Mapping[str, Mapping[str, Any]],
    graph: Mapping[str, Endpoint],
    client: TmdbClient | SpotifyClient,
    *,
    id_type: str = "int",
    discover: bool = False,
) -> tuple[dict[str, Endpoint], dict[str, Any]]:
    """Confirm each declared response path with generic calls, never a query.

    The probe starts from ids found by generic searches ("Inception", "Tom
    Hanks"), then harvests every port value it verifies and uses those to reach
    endpoints that were unreachable in the first pass, which is how review,
    network and credit endpoints get evidence. An endpoint the probe cannot
    reach keeps its ports and is recorded as `unprobed`, so the manifest says
    which edges are evidence and which are schema inference.
    """
    catalog = tool_index(base)
    ids, stats = probe_ids(dict(base), client)
    port_values: dict[str, Any] = {
        port_for_param(param, "", id_type=id_type): value
        for param, value in ids.items()
    }
    tally = {
        "verified": 0,
        "nullable": 0,
        "absent": 0,
        "inconclusive": 0,
        "unprobed": 0,
        "endpoints_called": 0,
    }
    absent: dict[str, list[str]] = {}
    states: dict[str, tuple[Port, ...]] = {}
    unreached: set[str] = set(graph)
    for _round in range(3):
        reached: list[str] = []
        for name in sorted(unreached):
            endpoint = graph[name]
            document = catalog.get(name) or {}
            arguments = _probe_arguments(
                document, endpoint.entity, port_values, id_type=id_type
            )
            if arguments is None:
                continue
            reached.append(name)
            payload: Any = None
            truncated = False
            try:
                row = client.call(
                    url_template=str(document.get("url") or ""), arguments=arguments
                )
                tally["endpoints_called"] += 1
                if not row.get("error"):
                    payload, truncated = _recover_json(str(row.get("response") or ""))
            except (ValueError, OSError, TypeError):
                payload = None
            if payload is None:
                continue
            if discover and not endpoint.outputs:
                found = discovered_outputs(payload, id_type=id_type)
                if found:
                    graph = {**graph, name: replace(endpoint, outputs=found)}
                    endpoint = graph[name]
                    tally["discovered"] = tally.get("discovered", 0) + len(found)
            ports: list[Port] = []
            for port in endpoint.outputs:
                found, value = _path_probe(payload, port.path)
                if found == "present":
                    state = "verified"
                    port_values.setdefault(port.name, value)
                elif found == "nullable":
                    state = "nullable"
                elif truncated:
                    state = "inconclusive"
                else:
                    state = "absent"
                    absent.setdefault(name, []).append(port.render_output())
                tally[state] += 1
                ports.append(replace(port, verified=state))
            states[name] = tuple(ports)
        if not reached:
            break
        unreached.difference_update(reached)
    checked: dict[str, Endpoint] = {}
    for name, endpoint in graph.items():
        ports = states.get(name)
        if ports is None:
            tally["unprobed"] += len(endpoint.outputs)
            checked[name] = endpoint
            continue
        checked[name] = replace(endpoint, outputs=ports)
    return checked, {
        "seed_ids": ids,
        "seed_probe": stats,
        "harvested_ports": sorted(port_values),
        "paths": tally,
        "endpoints_unreached": sorted(unreached),
        "dropped_absent_ports": absent,
    }


def port_block(endpoint: Endpoint, *, shuffled: tuple[Port, ...] | None = None) -> str:
    """Render the ports. A path the probe reached and did not find is dropped."""
    outputs = shuffled if shuffled is not None else endpoint.outputs
    outputs = tuple(port for port in outputs if port.verified != "absent")
    lines = ["Dataflow ports:"]
    lines.extend(endpoint.inputs)
    lines.extend(port.render_output() for port in outputs)
    lines.append(
        "A port value is either stated in the question or taken from any "
        "earlier response that emitted the same port."
    )
    return "\n".join(lines)


def _port_prose(port: str, *, plural: bool) -> str:
    """Turn a typed port into plain prose for the selection surface."""
    body = port.split(":")[0]
    parts = body.split("|")
    if all(part.endswith("_ref") for part in parts):
        nouns = [PORT_NOUN.get(part, part[: -len("_ref")]) for part in parts]
        joined = nouns[0] if len(nouns) == 1 else (
            ", ".join(nouns[:-1]) + " or " + nouns[-1]
        )
        if plural:
            return f"{joined} references (id)"
        return f"a {joined} reference (id)"
    words = " or ".join(part.replace("_", " ") for part in parts)
    if plural:
        return f"{words}s" if not words.endswith("s") else words
    article = "an" if words[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {words}"


def _subject(endpoint: Endpoint) -> str:
    word = ENTITY_WORD.get(endpoint.entity, endpoint.entity or "record")
    facet = endpoint.facet
    plural = PLURAL.get(word, f"{word}s")
    if facet == "":
        if endpoint.selector == "latest":
            return f"the most recently added {word}"
        if endpoint.selector in LIST_SELECTORS or endpoint.selector in RANKED_FACETS:
            return plural
        return f"the stored fields of one {word}"
    if facet == "credits":
        return f"the cast and crew of one {word}"
    if facet == "movie_credits":
        return f"the movie filmography of one {word}"
    if facet == "tv_credits":
        return f"the television filmography of one {word}"
    if facet == "similar":
        return f"{plural} similar to one {word}"
    if facet == "recommendations":
        return f"{plural} recommended from one {word}"
    if facet == "images":
        return f"the image files of one {word}"
    if facet == "season":
        return "the stored fields of one television season"
    if facet == "episode":
        return "the stored fields of one television episode"
    if facet == "genre_list":
        return f"the {word} genre list"
    return f"the {facet.replace('_', ' ')} of one {word}"


def _selector_clause(endpoint: Endpoint) -> str:
    selector = endpoint.selector
    if selector == "text":
        field = "name" if endpoint.entity == "person" else "title"
        return f"matching a {field} text query"
    if selector == "filters":
        return "matching filter criteria"
    if selector == "trending":
        return "trending in a chosen media type and time window"
    if selector in RANKED_FACETS:
        return RANKED_FACETS[selector]
    if selector == "none":
        return ""
    ports = [
        item.split(" <- ")[-1]
        for item in endpoint.inputs
        if item.startswith("INPUT path.")
    ]
    if not ports:
        return ""
    prose = [_port_prose(port, plural=False) for port in dict.fromkeys(ports)]
    return "identified by " + " plus ".join(prose)


def _output_clause(endpoint: Endpoint) -> str:
    if endpoint.outputs:
        names = dict.fromkeys(port.name for port in endpoint.outputs)
        prose = [_port_prose(name, plural=True) for name in names]
        return "outputs " + ", ".join(dict.fromkeys(prose))
    described = NON_ID_OUTPUT.get(endpoint.facet)
    return f"outputs {described}" if described else "outputs descriptive fields"


def _atom_key(endpoint: Endpoint, atoms: Sequence[str]) -> tuple[Any, ...]:
    values = {
        "entity": endpoint.entity,
        "facet": endpoint.facet,
        "selector": endpoint.selector,
        "output": tuple(port.name for port in endpoint.outputs),
    }
    return tuple(values[atom] for atom in atoms)


ATOM_ORDER = ("entity", "facet", "selector", "output")


def minimal_atoms(
    endpoint: Endpoint, graph: Mapping[str, Endpoint]
) -> tuple[str, ...]:
    """Smallest atom set that still separates this endpoint from all others.

    Uniqueness is decided over the whole 54-endpoint catalog, not over the
    candidate list of one benchmark query, so the same fingerprint holds for
    every question. Entity, facet and output type always stay: they are the
    producer/consumer fact the hypothesis is about. The `identified by ...`
    clause is dropped whenever the rest already separates the endpoint, and
    a list selector (`popular`, `top_rated`, a text query) is never dropped
    because for those endpoints it is the whole identity.
    """
    atoms = list(ATOM_ORDER)
    if endpoint.selector != "reference":
        return tuple(atoms)
    trial = [atom for atom in atoms if atom != "selector"]
    key = _atom_key(endpoint, trial)
    collides = any(
        other.tool_name != endpoint.tool_name and _atom_key(other, trial) == key
        for other in graph.values()
    )
    return tuple(atoms) if collides else tuple(trial)


def fingerprint_line(endpoint: Endpoint, graph: Mapping[str, Endpoint]) -> str:
    """Positive distinguishing sentence: no negation, no sibling API name."""
    atoms = minimal_atoms(endpoint, graph)
    clause = f"Retrieves {_subject(endpoint)}"
    if "selector" in atoms:
        selector = _selector_clause(endpoint)
        if selector:
            clause += f" {selector}"
    if "output" in atoms:
        clause += f"; {_output_clause(endpoint)}"
    return clause.strip().rstrip(".") + "."


# What a detail endpoint answers with when its only id output is the id it was
# given. Naming that echo a useful output is what made H2 advertise an input
# as a product and cost Ling the TV bridges.
DESCRIPTIVE_ANSWER: dict[str, str] = {
    "person": "the biography, birthday and place of birth",
    "company": "the company description, headquarters and origin country",
    "network": "the network name and origin country",
    "keyword": "the keyword name",
    "review": "the review text and its author",
    "credit": "the credited job and department",
    "genre": "the genre names and ids",
}

ROLE_CLAUSE_LIMIT = 160


def consumed_reference_ports(endpoint: Endpoint) -> tuple[str, ...]:
    """The typed identifiers this endpoint can only get from an earlier call."""
    ports: list[str] = []
    for line in endpoint.inputs:
        if not line.startswith("INPUT path.") or "<-" not in line:
            continue
        port = line.split("<-", 1)[1].strip()
        if "_ref:" in port and port not in ports:
            ports.append(port)
    return tuple(ports)


def novel_output_ports(endpoint: Endpoint) -> tuple[str, ...]:
    """Outputs that are not the endpoint echoing back its own input id.

    `$.id` on a detail endpoint is the identifier the caller already had. Every
    other path -- `$.results[*].id`, `$.networks[*].id`, `$.cast[*].id` -- is a
    reference the caller did not have before this call.
    """
    consumed = set(consumed_reference_ports(endpoint))
    ports: list[str] = []
    for port in endpoint.outputs:
        if port.verified == "absent":
            continue
        if port.path == "$.id" and port.name in consumed:
            continue
        if port.name not in ports:
            ports.append(port.name)
    return tuple(ports)


def role_clause(endpoint: Endpoint) -> str:
    """`Consumes <typed ref>; returns <what the caller did not already have>.`

    Added to the original purpose, never replacing it: H2 replaced the purpose
    with a uniform fingerprint and Ling lost the TV->company/network/person
    bridges on q053, q060, q071 and q073.
    """
    consumed = consumed_reference_ports(endpoint)
    if not consumed:
        return ""
    inputs = " plus ".join(
        _port_prose(port, plural=False) for port in consumed
    )
    novel = novel_output_ports(endpoint)
    if novel:
        prose = list(dict.fromkeys(_port_prose(port, plural=True) for port in novel))
        listed = (
            prose[0]
            if len(prose) == 1
            else ", ".join(prose[:-1]) + " and " + prose[-1]
        )
        result = listed
    else:
        result = (
            NON_ID_OUTPUT.get(endpoint.facet)
            or DESCRIPTIVE_ANSWER.get(endpoint.entity)
            or "the stored descriptive fields"
        )
    clause = f"Consumes {inputs}; returns {result}."
    if len(clause) > ROLE_CLAUSE_LIMIT:
        raise ValueError(
            f"role clause for {endpoint.tool_name} is {len(clause)} characters, "
            f"over the {ROLE_CLAUSE_LIMIT} the plan allows: {clause!r}"
        )
    return clause


def _nest(path: str, value: Any) -> Any:
    """`$.results[*].id` + 7 -> {"results": [{"id": 7}]}."""
    node: Any = value
    for step in reversed(path.removeprefix("$.").split(".")):
        node = {step[:-3]: [node]} if step.endswith("[*]") else {step: node}
    return node


def _illustrative_response(port: str, graph: Mapping[str, Endpoint]) -> Any:
    """Shape one prior response that really does carry this port."""
    sentinel = SENTINELS.get(port)
    if sentinel is None:
        return None
    candidates = [
        item.path
        for endpoint in graph.values()
        for item in endpoint.outputs
        if item.name == port
    ]
    path = next(
        (item for item in candidates if item.startswith("$.results")),
        candidates[0] if candidates else "$.id",
    )
    return _nest(path, sentinel)


def witness_example(
    document: Mapping[str, Any], graph: Mapping[str, Endpoint]
) -> dict[str, Any] | None:
    """Provenance witness for `example`: sentinel in, sentinel out, copy rule."""
    params = hop_params(str(document.get("url") or ""))
    if not params:
        return None
    prior: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    for param in params:
        port = PARAM_PORT.get(param, "")
        sentinel = SENTINELS.get(port)
        if sentinel is None:
            continue
        parameters[param] = sentinel
        shape = _illustrative_response(port, graph)
        if isinstance(shape, dict):
            prior = _merge(prior, shape)
    if not parameters:
        return None
    names = list(parameters)
    listed = names[0] if len(names) == 1 else (
        ", ".join(names[:-1]) + " and " + names[-1]
    )
    verb = "repeats" if len(names) == 1 else "repeat"
    return {
        "Prior response (illustration)": prior,
        "Parameters": parameters,
        "Provenance": (
            f"{listed} {verb} a value the prior response already contains, at "
            "the position shown above. The values here are placeholders: copy "
            "what the actual previous response returned, never the literal "
            "above."
        ),
    }


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge(merged[key], value)
        elif (
            key in merged
            and isinstance(merged[key], list)
            and isinstance(value, list)
            and merged[key]
            and value
            and isinstance(merged[key][0], dict)
            and isinstance(value[0], dict)
        ):
            merged[key] = [_merge(merged[key][0], value[0])]
        else:
            merged[key] = value
    return merged


def _shuffled_outputs(
    graph: Mapping[str, Endpoint], seed: int
) -> dict[str, tuple[Port, ...]]:
    """Control arm: keep every port line, permute which endpoint carries it."""
    names = sorted(graph)
    payloads = [graph[name].outputs for name in names]
    rng = random.Random(seed)
    order = list(range(len(names)))
    for _ in range(64):
        rng.shuffle(order)
        if all(
            payloads[index] != payloads[position]
            for position, index in enumerate(order)
        ):
            break
    return {names[position]: payloads[index] for position, index in enumerate(order)}


def build_positive_variant(
    spec: VariantSpec,
    base: Mapping[str, Mapping[str, Any]],
    graph: Mapping[str, Endpoint],
    *,
    dataset: str,
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compile `P` / `PF`: the frozen guideline, a rewritten selection surface.

    The guideline `description` and `example` are taken verbatim from the
    frozen documentation, so the only thing that moves relative to
    `CurrentNot` is what `choose_tool` reads. `P` drops the negation from the
    catalog scope; `PF` additionally appends a role clause to the endpoints
    that consume a reference.
    """
    frozen = _load_json(Path(spec.derive_from.format(dataset=dataset)))
    missing = sorted(set(base) - set(frozen))
    if missing:
        raise ValueError(
            f"{spec.name}: the frozen documentation is missing {missing}; it "
            "must cover the same catalog as the base."
        )
    docs: dict[str, dict[str, Any]] = {}
    for key, document in base.items():
        updated = copy.deepcopy(dict(frozen[key]))
        name = str(document.get("tool_name") or "")
        purpose = str(document.get("description") or "").rstrip()
        clauses: list[str] = []
        if not hop_params(str(document.get("url") or "")):
            contract = response_contract(dict(document), dict(catalog))
            if contract:
                scope = positive_scope_line(contract)
                if scope:
                    clauses.append(scope)
        if spec.role_clause and name in graph:
            clause = role_clause(graph[name])
            if clause:
                clauses.append(clause)
        updated["tool_description"] = " ".join([purpose, *clauses]).rstrip()
        docs[key] = updated
    assert_inherited_guideline(spec, frozen, docs)
    return docs


def build_variant(
    spec: VariantSpec,
    *,
    draft_root: Path = DEFAULT_DRAFT_ROOT,
    dataset: str = "TMDB",
    graph: Mapping[str, Endpoint] | None = None,
    probe: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    base = load_instructions(draft_root, dataset, spec.base)
    if spec.copy_frozen:
        docs = _load_json(Path(spec.copy_frozen.format(dataset=dataset)))
        return docs, _manifest(
            spec, base, docs, dataset=dataset, draft_root=draft_root, probe=probe
        )
    resolved = (
        dict(graph)
        if graph is not None
        else build_graph(base, id_type=ID_TYPES.get(dataset, "str"))
    )
    catalog = tool_index(base)
    if spec.positive_scope:
        docs = build_positive_variant(
            spec, base, resolved, dataset=dataset, catalog=catalog
        )
        manifest = _manifest(
            spec, base, docs, dataset=dataset, draft_root=draft_root, probe=probe
        )
        assert_variant_invariants(spec, base, docs)
        return docs, manifest
    shuffled = _shuffled_outputs(resolved, seed=20260909) if spec.shuffle_ports else {}
    docs: dict[str, dict[str, Any]] = {}
    for key, document in base.items():
        updated = copy.deepcopy(dict(document))
        name = str(updated.get("tool_name") or "")
        purpose = str(updated.get("description") or "").rstrip()
        updated["tool_description"] = purpose
        endpoint = resolved.get(name)
        extra: list[str] = []
        if spec.fill_desc:
            fill = hop_contract(updated, catalog, {})
            if fill:
                extra.append(fill)
        if spec.ports and endpoint is not None:
            extra.append(
                port_block(
                    endpoint,
                    shuffled=shuffled.get(name) if spec.shuffle_ports else None,
                )
            )
        if extra:
            updated["description"] = f"{purpose}\n\n" + "\n\n".join(extra)
        if spec.fingerprint and endpoint is not None:
            updated["tool_description"] = fingerprint_line(endpoint, resolved)
        if spec.witness:
            example = witness_example(updated, resolved)
            if example is not None:
                updated["example"] = example
        docs[key] = updated
    manifest = _manifest(
        spec, base, docs, dataset=dataset, draft_root=draft_root, probe=probe
    )
    assert_variant_invariants(spec, base, docs)
    return docs, manifest


def schema_diff(
    base: Mapping[str, Mapping[str, Any]], docs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    changed: dict[str, list[str]] = {}
    for key, document in docs.items():
        original = base.get(key) or {}
        drift = [
            field
            for field in SCHEMA_FIELDS
            if document.get(field) != original.get(field)
        ]
        if drift:
            changed[str(document.get("tool_name") or key)] = drift
    surfaces = {
        "tool_description": 0,
        "description": 0,
        "example": 0,
    }
    for key, document in docs.items():
        original = base.get(key) or {}
        purpose = str(original.get("description") or "").rstrip()
        if str(document.get("tool_description") or "") != purpose:
            surfaces["tool_description"] += 1
        if str(document.get("description") or "") != str(original.get("description") or ""):
            surfaces["description"] += 1
        if document.get("example") != original.get("example"):
            surfaces["example"] += 1
    return {"schema_changed": changed, "surfaces_touched": surfaces}


def assert_variant_invariants(
    spec: VariantSpec,
    base: Mapping[str, Mapping[str, Any]],
    docs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail the build when the label and the text disagree."""
    diff = schema_diff(base, docs)
    if diff["schema_changed"]:
        raise ValueError(
            f"{spec.name} changed the request schema of "
            f"{sorted(diff['schema_changed'])}; variants must be prose-only."
        )
    touched = diff["surfaces_touched"]
    expected_surfaces = {
        "tool_description": spec.fingerprint or spec.positive_scope,
        # `P` / `PF` inherit the frozen guideline verbatim, so `description`
        # differs from `Initial` by construction. `assert_inherited_guideline`
        # is what proves it was inherited rather than rewritten.
        "description": spec.ports or spec.fill_desc or bool(spec.derive_from),
        "example": spec.witness,
    }
    for surface, expected in expected_surfaces.items():
        if expected and not touched[surface]:
            raise ValueError(
                f"{spec.name} claims to write {surface} but no document changed."
            )
        if not expected and touched[surface]:
            raise ValueError(
                f"{spec.name} wrote {touched[surface]} {surface} fields it does "
                "not declare; a stage-localized variant must stay on its surface."
            )
    if spec.fingerprint:
        lines = [str(doc.get("tool_description") or "") for doc in docs.values()]
        for line in lines:
            if re.search(r"\bnot\b|\bdo not\b|\bavoid\b|GET_", line):
                raise ValueError(
                    f"{spec.name} fingerprint is not purely positive: {line!r}"
                )
        if len(set(lines)) != len(lines):
            raise ValueError(f"{spec.name} fingerprints are not unique per endpoint.")
    if spec.ports:
        for doc in docs.values():
            body = str(doc.get("description") or "")
            if "Dataflow ports:" in body and "call after" in body.lower():
                raise ValueError(f"{spec.name} leaked an ordering imperative.")
    if spec.witness:
        for doc in docs.values():
            example = doc.get("example")
            if not hop_params(str(doc.get("url") or "")):
                continue
            if not isinstance(example, dict) or "Provenance" not in example:
                continue
            if "placeholder" not in str(example["Provenance"]):
                raise ValueError(
                    f"{spec.name} witness omits the copy invariant on "
                    f"{doc.get('tool_name')}."
                )
    if spec.fill_desc:
        for doc in docs.values():
            body = str(doc.get("description") or "")
            if "Response:" in body or "Downstream:" in body:
                raise ValueError(
                    f"{spec.name} must not carry response or downstream contracts."
                )
    if spec.positive_scope:
        assert_positive_selection_surface(spec, base, docs)


# More specific first, so the reported reason names the actual construct
# rather than the bare `not` inside it.
FORBIDDEN_SELECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdo not\b", "a prohibition"),
    (r"\bavoid\b", "a prohibition"),
    (r"\bnever\b", "a prohibition"),
    (r"\bnot\b", "a negation"),
    (r"\b(?:GET|PUT|POST|DELETE)_", "a sibling API name"),
    (r"\bcall (?:this|after|first)\b", "an ordering imperative"),
    (r"\b(?:then|next|afterwards) call\b", "an ordering imperative"),
    (r"\buse (?:this )?(?:after|before|first)\b", "an ordering imperative"),
)


def assert_positive_selection_surface(
    spec: VariantSpec,
    base: Mapping[str, Mapping[str, Any]],
    docs: Mapping[str, Mapping[str, Any]],
) -> None:
    """The Cross-model Positive Scope contract, checked on the built text.

    Every clause here corresponds to a diagnosed failure: the negation that
    contradicted the task on Flash, the replaced purpose that cost Ling its
    bridges, the ordering hint that made the agent skip a hop, and an append
    long enough to push the purpose out of the model's attention.
    """
    for key, document in docs.items():
        name = str(document.get("tool_name") or key)
        purpose = str((base.get(key) or {}).get("description") or "").rstrip()
        selection = str(document.get("tool_description") or "")
        if not selection.startswith(purpose):
            raise ValueError(
                f"{spec.name} does not keep the original purpose of {name} as "
                "an exact prefix of the selection surface."
            )
        appended = selection[len(purpose) :].strip()
        if len(appended) > POSITIVE_SCOPE_LIMIT + ROLE_CLAUSE_LIMIT:
            raise ValueError(
                f"{spec.name} appends {len(appended)} characters to {name}."
            )
        for pattern, description in FORBIDDEN_SELECTION_PATTERNS:
            if re.search(pattern, appended, flags=re.IGNORECASE):
                raise ValueError(
                    f"{spec.name} appends {description} to {name}: {appended!r}"
                )


def assert_inherited_guideline(
    spec: VariantSpec,
    frozen: Mapping[str, Mapping[str, Any]],
    docs: Mapping[str, Mapping[str, Any]],
) -> None:
    """`P` / `PF` must move the selection surface and nothing else.

    If the guideline drifted too, a Screen A result could not be attributed to
    the selection stage, which is the whole point of compiling these two.
    """
    for key, document in docs.items():
        name = str(document.get("tool_name") or key)
        reference = frozen.get(key) or {}
        for field in ("description", "example"):
            if document.get(field) != reference.get(field):
                raise ValueError(
                    f"{spec.name} changed the inherited {field} of {name}; the "
                    "guideline surface has to stay identical to the frozen "
                    "documentation."
                )


BANNED_CALLS = ("load_queries", "gold_apis", "correct_path", "test_path", "load_traces")


def assert_source_only(module_path: Path | None = None) -> dict[str, Any]:
    """Prove the compiler reads documentation, not the benchmark it is scored on.

    Checking call names in the parsed module, rather than grepping the text,
    keeps this honest: the token list below is data, so it cannot trip its own
    check, and a future edit that starts reading queries or gold paths fails
    the build instead of quietly making the variants unpublishable.
    """
    path = module_path or Path(__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            called.add(target.id)
        elif isinstance(target, ast.Attribute):
            called.add(target.attr)
    leaked = sorted(called & set(BANNED_CALLS))
    if leaked:
        raise ValueError(f"variant compiler reads benchmark inputs: {leaked}")
    return {
        "inputs": ["tool_instruction JSON", "endpoint URLs", "generic live probes"],
        "benchmark_readers_absent": list(BANNED_CALLS),
        "compiler_digest": file_digest(path),
        "ours_compiler_digest": file_digest(path.with_name("restbench_ours.py")),
    }


def _manifest(
    spec: VariantSpec,
    base: Mapping[str, Mapping[str, Any]],
    docs: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
    draft_root: Path,
    probe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base_path = instruction_path(draft_root, dataset, spec.base)
    manifest: dict[str, Any] = {
        "variant": spec.name,
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "flags": spec.flags(),
        "base_documentation": {
            "name": spec.base,
            "path": str(base_path),
            "file_digest": file_digest(base_path),
            "payload_digest": payload_digest(base),
        },
        "payload_digest": payload_digest(docs),
        "n": len(docs),
        "provenance": assert_source_only(),
        "probe": dict(probe or {}),
        "gold_used": False,
    }
    manifest.update(schema_diff(base, docs))
    return manifest


def variant_paths(name: str, *, dataset: str = "TMDB") -> tuple[Path, Path]:
    stem = f"{dataset}_{name}"
    return (
        EXPERIMENT_ROOT / f"{stem}.json",
        EXPERIMENT_ROOT / f"{stem}_manifest.json",
    )


def write_variant(
    spec: VariantSpec,
    docs: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    dataset: str = "TMDB",
    overwrite: bool = False,
) -> Path:
    """Write once. A frozen variant is never silently replaced mid-pilot."""
    doc_path, manifest_path = variant_paths(spec.name, dataset=dataset)
    if doc_path.exists() and not overwrite:
        existing = payload_digest(_load_json(doc_path))
        if existing != manifest["payload_digest"]:
            raise SystemExit(
                f"{doc_path} already exists with payload {existing[:16]}... and "
                f"the new build is {str(manifest['payload_digest'])[:16]}...; "
                "pass --overwrite only before the pilot starts."
            )
        _dump_json(manifest_path, manifest)
        return doc_path
    _dump_json(doc_path, docs)
    _dump_json(manifest_path, manifest)
    return doc_path


def build_all(
    *,
    draft_root: Path = DEFAULT_DRAFT_ROOT,
    dataset: str = "TMDB",
    names: Iterable[str] | None = None,
    client: TmdbClient | SpotifyClient | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    base = load_instructions(draft_root, dataset, "Initial")
    id_type = ID_TYPES.get(dataset, "str")
    graph = build_graph(base, id_type=id_type)
    probe: dict[str, Any] = {"probed": False}
    if client is not None:
        graph, probe = probe_graph(
            base,
            graph,
            client,
            id_type=id_type,
            # The response-shape rules were written from the TMDB catalog; on
            # another API the ports come from the probed payload instead.
            discover=dataset != "TMDB",
        )
        probe = {"probed": True, **probe}
    wanted = set(names) if names else {spec.name for spec in VARIANTS}
    written: dict[str, Any] = {}
    for spec in VARIANTS:
        if spec.name not in wanted:
            continue
        docs, manifest = build_variant(
            spec,
            draft_root=draft_root,
            dataset=dataset,
            graph=graph,
            probe=probe,
        )
        path = write_variant(
            spec, docs, manifest, dataset=dataset, overwrite=overwrite
        )
        written[spec.name] = {
            "path": str(path),
            "payload_digest": manifest["payload_digest"],
            "surfaces_touched": manifest["surfaces_touched"],
        }
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--dataset", default="TMDB", choices=["TMDB", "Spotify"])
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Confirm response paths with generic live TMDB calls.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--show",
        default="",
        help="Print the compiled surfaces of one endpoint and exit.",
    )
    args = parser.parse_args()

    if args.show:
        base = load_instructions(args.draft_root, args.dataset, "Initial")
        graph = build_graph(base, id_type=ID_TYPES.get(args.dataset, "str"))
        endpoint = graph[args.show]
        print(fingerprint_line(endpoint, graph))
        print(port_block(endpoint))
        document = tool_index(base)[args.show]
        print(json.dumps(witness_example(document, graph), ensure_ascii=False, indent=2))
        return 0

    client: TmdbClient | SpotifyClient | None = None
    if args.probe:
        import os

        from tooldoc_nir.openrouter import load_env_file

        load_env_file()
        if args.dataset == "TMDB":
            key = os.environ.get("TMDB_API_KEY", "")
            if not key:
                raise SystemExit("TMDB_API_KEY is missing; drop --probe or set it.")
            client = TmdbClient(key, cache={})
        else:
            client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
            secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
            refresh = os.environ.get("SPOTIFY_REFRESH_TOKEN", "")
            if not client_id or not secret:
                raise SystemExit("SPOTIFY_CLIENT_ID / _SECRET are missing.")
            client = SpotifyClient(
                client_id, secret, refresh_token=refresh, cache={}
            )
    written = build_all(
        draft_root=args.draft_root,
        dataset=args.dataset,
        names=args.variants,
        client=client,
        overwrite=args.overwrite,
    )
    print(json.dumps(written, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
