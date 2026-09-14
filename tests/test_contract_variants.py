"""Invariants for the stage-localized pilot variants.

The point of these gates is that a variant cannot lie about which stage of the
DRAFT driver it talks to. `choose_tool` reads `tool_description`,
`task_decompose` reads the guideline `description`, `choose_parameter` reads the
guideline including `example`; if a variant writes a surface it does not
declare, the pilot can no longer attribute an effect to a stage and the build
must fail instead.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tooldoc_nir.restbench_contract_variants import (
    SENTINELS,
    VARIANTS,
    Port,
    assert_inherited_guideline,
    assert_source_only,
    assert_variant_invariants,
    build_graph,
    build_variant,
    fingerprint_line,
    role_clause,
    schema_diff,
    variant_by_name,
    _path_probe,
    _recover_json,
)
from tooldoc_nir.restbench_data import load_instructions
from tooldoc_nir.restbench_ours import hop_params

DRAFT_ROOT = Path("external/DRAFT")


def _docs(name: str) -> dict[str, dict]:
    docs, _manifest = build_variant(variant_by_name(name), draft_root=DRAFT_ROOT)
    return docs


def _named(docs: dict[str, dict], tool_name: str) -> dict:
    return next(doc for doc in docs.values() if doc["tool_name"] == tool_name)


def test_every_variant_writes_only_the_surface_it_declares() -> None:
    for spec in VARIANTS:
        if spec.copy_frozen:
            continue
        base = load_instructions(DRAFT_ROOT, "TMDB", spec.base)
        docs, manifest = build_variant(spec, draft_root=DRAFT_ROOT)
        touched = schema_diff(base, docs)["surfaces_touched"]
        assert bool(touched["tool_description"]) == (
            spec.fingerprint or spec.positive_scope
        ), spec.name
        assert bool(touched["description"]) == (
            spec.ports or spec.fill_desc or bool(spec.derive_from)
        ), spec.name
        assert bool(touched["example"]) == spec.witness, spec.name
        assert manifest["schema_changed"] == {}, spec.name
        assert manifest["gold_used"] is False


def test_a_variant_labelled_as_another_surface_does_not_build() -> None:
    base = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    witness_docs = _docs("H1")
    with pytest.raises(ValueError, match="tool_description"):
        assert_variant_invariants(variant_by_name("H2"), base, witness_docs)


def test_h2_fingerprints_are_unique_and_purely_positive() -> None:
    docs = _docs("H2")
    lines = [str(doc["tool_description"]) for doc in docs.values()]
    assert len(set(lines)) == len(lines) == 54
    for line in lines:
        assert "not " not in line
        assert "GET_" not in line
        assert len(line) < 240
    search = _named(docs, "GET_search_movie")["tool_description"]
    credits = _named(docs, "GET_movie_movie_id_credits")["tool_description"]
    assert "movie references (id)" in search
    assert "person references (id)" in credits
    assert docs["2"]["description"] == load_instructions(DRAFT_ROOT, "TMDB", "Initial")["2"]["description"]


def test_h2_fingerprint_is_global_not_per_query() -> None:
    base = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    graph = build_graph(base)
    full = fingerprint_line(graph["GET_person_person_id_movie_credits"], graph)
    subset = {
        name: graph[name]
        for name in ("GET_person_person_id_movie_credits", "GET_search_person")
    }
    assert fingerprint_line(graph["GET_person_person_id_movie_credits"], subset) == full


def test_h3_ports_cover_the_edges_the_manual_map_was_missing() -> None:
    docs = _docs("H3")
    collection = _named(docs, "GET_collection_collection_id")["description"]
    movie_credits = _named(docs, "GET_movie_movie_id_credits")["description"]
    tv_credits = _named(docs, "GET_person_person_id_tv_credits")["description"]
    tv_detail = _named(docs, "GET_tv_tv_id")["description"]
    assert "OUTPUT movie_ref:int <- $.parts[*].id" in collection
    assert "OUTPUT person_ref:int <- $.cast[*].id" in movie_credits
    assert "OUTPUT tv_ref:int <- $.cast[*].id" in tv_credits
    assert "OUTPUT company_ref:int <- $.production_companies[*].id" in tv_detail
    assert "OUTPUT network_ref:int <- $.networks[*].id" in tv_detail
    assert "OUTPUT season_number:int <- $.seasons[*].season_number" in tv_detail
    hop = _named(docs, "GET_person_person_id_movie_credits")["description"]
    assert "INPUT path.person_id <- person_ref:int" in hop
    assert "call after" not in hop.lower()
    assert "stated in the question" in hop


def test_h3_shuffled_control_keeps_the_lines_and_moves_them() -> None:
    ports = _docs("H3")
    shuffled = _docs("H3Shuffled")
    moved = 0
    for key, document in ports.items():
        if document["description"] != shuffled[key]["description"]:
            moved += 1
    assert moved > 40
    def outputs(docs: dict[str, dict]) -> list[str]:
        return sorted(
            line
            for doc in docs.values()
            for line in str(doc["description"]).splitlines()
            if line.startswith("OUTPUT ")
        )

    assert outputs(ports) == outputs(shuffled)


def test_h1_witness_lives_in_example_with_a_copy_invariant() -> None:
    docs = _docs("H1")
    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    credits = _named(docs, "GET_person_person_id_movie_credits")
    example = credits["example"]
    sentinel = SENTINELS["person_ref:int"]
    assert example["Parameters"] == {"person_id": sentinel}
    assert example["Prior response (illustration)"] == {"results": [{"id": sentinel}]}
    assert "placeholder" in example["Provenance"]
    assert "GET_" not in str(example)
    assert credits["description"] == initial[str(credits["ID"])]["description"]
    assert credits["tool_description"] == initial[str(credits["ID"])]["description"]
    for key, document in docs.items():
        if hop_params(str(document["url"])):
            continue
        assert "example" not in document, document["tool_name"]
        assert document == {**initial[key], "tool_description": initial[key]["description"]}


def test_filldesc_drops_response_downstream_and_negation() -> None:
    docs = _docs("FillDesc")
    for document in docs.values():
        body = str(document["description"])
        assert "Response:" not in body
        assert "Downstream:" not in body
        assert "not GET_" not in body
        assert "Example:" not in body
    person = _named(docs, "GET_person_person_id")["description"]
    assert "GET_search_person" in person
    assert "person_id from" in person


def test_h123_puts_each_semantics_on_its_own_surface() -> None:
    docs = _docs("H123")
    fingerprints = _docs("H2")
    ports = _docs("H3")
    witness = _docs("H1")
    for key, document in docs.items():
        assert document["tool_description"] == fingerprints[key]["tool_description"]
        assert document["description"] == ports[key]["description"]
        assert document.get("example") == witness[key].get("example")


def test_h12_composes_only_the_two_surviving_hypotheses() -> None:
    docs = _docs("H12")
    fingerprints = _docs("H2")
    witness = _docs("H1")
    initial = _docs("Initial")
    for key, document in docs.items():
        assert document["tool_description"] == fingerprints[key]["tool_description"]
        assert document["description"] == initial[key]["description"]
        assert document.get("example") == witness[key].get("example")


# --------------------------------------------------------------------------
# Cross-model Positive Scope: `P` and `PF`
# --------------------------------------------------------------------------


FROZEN = Path("artifacts/documentation/TMDB_Ours.json")


def _frozen() -> dict[str, dict]:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_p_drops_the_negation_and_keeps_the_positive_scope() -> None:
    docs = _docs("P")
    search = _named(docs, "GET_search_movie")["tool_description"]
    assert search == (
        "Search for movies. Returns results[i].{id,title,release_date,overview}."
    )
    assert "not GET_" not in search
    for document in docs.values():
        appended = str(document["tool_description"])[
            len(_named(_docs("Initial"), document["tool_name"])["tool_description"]) :
        ]
        assert "GET_" not in appended, document["tool_name"]


def test_p_and_pf_inherit_the_frozen_guideline_verbatim() -> None:
    """The guideline helps Ling decompose and never reaches choose_tool."""
    frozen = _frozen()
    for name in ("P", "PF"):
        docs = _docs(name)
        for key, document in docs.items():
            assert document["description"] == frozen[key]["description"], key
            assert document.get("example") == frozen[key].get("example"), key


def test_p_and_pf_keep_the_original_purpose_as_an_exact_prefix() -> None:
    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    for name in ("P", "PF"):
        for key, document in _docs(name).items():
            purpose = str(initial[key]["description"]).rstrip()
            assert str(document["tool_description"]).startswith(purpose), key


def test_pf_adds_a_role_clause_on_top_of_p_rather_than_replacing_it() -> None:
    plain = _docs("P")
    roles = _docs("PF")
    extended = 0
    for key, document in roles.items():
        assert str(document["tool_description"]).startswith(
            str(plain[key]["tool_description"])
        ), key
        if document["tool_description"] != plain[key]["tool_description"]:
            extended += 1
    # 19 catalogs carry only the positive scope, 33 reference endpoints gain a
    # role clause, and the two genre lists have neither a scope nor an input.
    assert extended == 33
    assert len(plain) == 54


def test_pf_role_clauses_name_new_results_not_the_echoed_input() -> None:
    docs = _docs("PF")
    detail = _named(docs, "GET_person_person_id")["tool_description"]
    bridge = _named(docs, "GET_tv_tv_id")["tool_description"]
    credits = _named(docs, "GET_movie_movie_id_credits")["tool_description"]
    images = _named(docs, "GET_movie_movie_id_images")["tool_description"]

    assert detail.endswith(
        "Consumes a person reference (id); returns the biography, birthday "
        "and place of birth."
    )
    assert "returns person references" not in detail
    assert bridge.endswith(
        "Consumes a TV show reference (id); returns company references (id), "
        "network references (id) and season numbers."
    )
    assert "returns TV show references" not in bridge
    assert credits.endswith(
        "Consumes a movie reference (id); returns person references (id) and "
        "credit references (id)."
    )
    assert images.endswith(
        "Consumes a movie reference (id); returns image file paths."
    )


def test_pf_leaves_the_catalogs_without_a_consumes_clause() -> None:
    """A search API consumes text, so a `Consumes <ref>` clause would be false."""
    docs = _docs("PF")
    for document in docs.values():
        if hop_params(str(document["url"])):
            continue
        assert "Consumes" not in str(document["tool_description"]), (
            document["tool_name"]
        )


def test_the_appended_clause_stays_inside_the_length_budget() -> None:
    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    for name in ("P", "PF"):
        for key, document in _docs(name).items():
            purpose = str(initial[key]["description"]).rstrip()
            appended = str(document["tool_description"])[len(purpose) :].strip()
            assert len(appended) <= 160, (name, key, len(appended))


def test_a_negation_or_an_ordering_hint_in_the_scope_fails_the_build() -> None:
    base = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    spec = variant_by_name("P")
    for injected, message in (
        (" Do not use GET_movie_movie_id_credits.", "prohibition"),
        (" Call this after the search API.", "ordering imperative"),
        (" Avoid the detail endpoint.", "prohibition"),
    ):
        docs = _docs("P")
        key = next(iter(docs))
        docs[key] = {
            **docs[key],
            "tool_description": docs[key]["tool_description"] + injected,
        }
        with pytest.raises(ValueError, match=message):
            assert_variant_invariants(spec, base, docs)


def test_a_rewritten_guideline_fails_the_inheritance_check() -> None:
    frozen = _frozen()
    docs = _docs("P")
    key = next(iter(docs))
    docs[key] = {**docs[key], "description": "rewritten"}
    with pytest.raises(ValueError, match="inherited description"):
        assert_inherited_guideline(variant_by_name("P"), frozen, docs)


def test_the_role_clause_refuses_to_exceed_its_budget() -> None:
    base = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    graph = build_graph(base)
    endpoint = graph["GET_tv_tv_id"]
    inflated = replace(
        endpoint,
        outputs=endpoint.outputs
        + tuple(
            Port(name=f"padding_field_number_{index}:int", path="$.x[*].id")
            for index in range(12)
        ),
    )
    with pytest.raises(ValueError, match="over the 160"):
        role_clause(inflated)


def test_source_only_check_catches_a_compiler_that_reads_the_benchmark(
    tmp_path: Path,
) -> None:
    assert assert_source_only()["compiler_digest"]
    leaky = tmp_path / "leaky.py"
    leaky.write_text("def build():\n    return load_queries('TMDB')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="load_queries"):
        assert_source_only(leaky)


def test_truncated_probe_bodies_are_recovered_and_flagged() -> None:
    complete = '{"results": [{"id": 27205}]}'
    payload, truncated = _recover_json(complete)
    assert payload == {"results": [{"id": 27205}]} and truncated is False
    cut = '{"results": [{"id": 27205, "title": "Inception"}, {"id": 155, "ti'
    payload, truncated = _recover_json(cut)
    assert truncated is True
    assert _path_probe(payload, "$.results[*].id")[1] == 27205
    assert _path_probe(payload, "$.crew[*].id")[0] == "missing"
    assert _path_probe({"belongs_to_collection": None}, "$.belongs_to_collection.id")[
        0
    ] == "nullable"
