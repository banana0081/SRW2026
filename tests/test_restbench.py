"""Offline gates for RestBench HTTP, tool schemas and Correct Path."""

from __future__ import annotations

from pathlib import Path

from tooldoc_nir.restbench_adapt import wrap_instructions, wrap_query
from tooldoc_nir.restbench_data import (
    candidate_docs,
    correct_path,
    gold_apis,
    load_instructions,
    load_queries,
    openai_tools,
    parameter_entries,
)
from tooldoc_nir.restbench_http import fill_url, path_param_names


DRAFT_ROOT = Path("external/DRAFT")


def test_path_params_are_pulled_from_the_url() -> None:
    url = "http://api.themoviedb.org/3/person/{person_id}/movie_credits"
    assert path_param_names(url) == ["person_id"]
    filled, leftover = fill_url(url, {"person_id": 123, "language": "en"})
    assert filled.endswith("/person/123/movie_credits")
    assert filled.startswith("https://")
    assert leftover == {"language": "en"}


def test_correct_path_is_an_ordered_subsequence() -> None:
    gold = ["GET_search_person", "GET_person_person_id_movie_credits"]
    assert correct_path(
        ["GET_tv_popular", "GET_search_person", "GET_person_person_id_movie_credits"],
        gold,
    )
    assert not correct_path(["GET_person_person_id_movie_credits", "GET_search_person"], gold)
    assert not correct_path(["GET_search_person"], gold)


def test_tmdb_candidates_and_gold_match_released_q0() -> None:
    queries = load_queries(DRAFT_ROOT, "TMDB")
    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    draft = load_instructions(DRAFT_ROOT, "TMDB", "DRAFT")
    query = queries[0]
    assert gold_apis(query) == [
        "GET_search_person",
        "GET_person_person_id_movie_credits",
    ]
    initial_docs = candidate_docs(query, initial)
    draft_docs = candidate_docs(query, draft)
    assert [doc["tool_name"] for doc in initial_docs] == [
        doc["tool_name"] for doc in draft_docs
    ]
    assert "GET_search_person" in [doc["tool_name"] for doc in initial_docs]
    assert initial_docs[0]["description"] != draft_docs[0]["description"]


def test_openai_tool_names_fit_the_64_char_limit() -> None:
    draft = load_instructions(DRAFT_ROOT, "TMDB", "DRAFT")
    tools, mapping = openai_tools(list(draft.values())[:20])
    for tool in tools:
        name = tool["function"]["name"]
        assert len(name) <= 64
        assert name.startswith("api_")
        assert mapping[name]["tool_name"]


def test_person_credits_exposes_the_path_id() -> None:
    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    document = next(
        doc
        for doc in initial.values()
        if doc["tool_name"] == "GET_person_person_id_movie_credits"
    )
    names = [str(item["name"]) for item in parameter_entries(document)]
    assert "person_id" in names


def test_restbench_wrap_fits_inference_dfsdt() -> None:
    queries = load_queries(DRAFT_ROOT, "TMDB")
    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    wrapped_docs = wrap_instructions(initial)
    wrapped = wrap_query(queries[0])
    gold = gold_apis(queries[0])
    assert wrapped["relevant APIs"] == [[name, name] for name in gold]
    assert wrapped["api_list"][0]["api_name"] == wrapped["api_list"][0]["tool_name"]
    person = wrapped_docs["34"]
    assert person["category"] == "TMDB"
    assert "GET_search_person" in person["tool_guidelines"]
    assert person["url"].endswith("/search/person")
    guideline = person["tool_guidelines"]["GET_search_person"]
    assert "query" in str(guideline["required_parameters"])


def test_ours_annotates_hops_without_changing_schema_or_search_prose() -> None:
    from tooldoc_nir.restbench_ours import build_ours_instructions

    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    ours, report = build_ours_instructions(initial)
    person = ours["2"]
    popular = ours["1"]
    assert person["tool_name"] == "GET_person_person_id"
    assert popular["tool_name"] == "GET_tv_popular"
    assert person["required_parameters"] == initial["2"]["required_parameters"]
    assert "Response:" in popular["description"]
    assert popular["tool_description"].startswith(initial["1"]["description"])
    assert "not " in popular["tool_description"]
    assert person["tool_description"] == initial["2"]["description"]
    assert "GET_search_person" in person["description"]
    assert "person_id" in person["description"]
    assert "call this after" not in person["description"].lower()
    assert "not this endpoint" not in person["description"].lower()
    search = next(doc for doc in ours.values() if doc["tool_name"] == "GET_search_person")
    assert "known_for" in search["description"]
    assert "GET_person_person_id_movie_credits" in search["description"]
    assert search["tool_description"] != initial[str(search["ID"])]["description"]
    wrapped = wrap_instructions(ours)["2"]
    assert wrapped["tool_description"] == initial["2"]["description"]
    assert "GET_search_person" in wrapped["tool_guidelines"]["GET_person_person_id"]["description"]
    wrapped_search = wrap_instructions(ours)[str(search["ID"])]
    assert "Returns" in wrapped_search["tool_description"]
    assert report["gold_used"] is False
    assert report["schema_changed"] is False
    assert report["hops_annotated"] > 0
    assert report["responses_annotated"] > 0


def test_http_method_follows_rest_prefix() -> None:
    from tooldoc_nir.restbench_http import http_method_from_tool

    assert http_method_from_tool("GET_search") == "GET"
    assert http_method_from_tool("PUT_me_player_play") == "PUT"
    assert http_method_from_tool("POST_playlists_playlist_id_tracks") == "POST"


def test_ours_fill_contract_on_spotify_album_points_at_search() -> None:
    from tooldoc_nir.restbench_ours import build_ours_instructions

    initial = load_instructions(DRAFT_ROOT, "Spotify", "Initial")
    ours, report = build_ours_instructions(initial, base_name="Initial")
    album = ours["0"]
    assert album["tool_name"] == "GET_albums_id"
    assert album["tool_description"] == initial["0"]["description"]
    assert "GET_search" in album["description"]
    assert report["gold_used"] is False
    assert report["schema_changed"] is False
    me = next(doc for doc in ours.values() if doc["tool_name"] == "GET_me")
    initial_me = next(doc for doc in initial.values() if doc["tool_name"] == "GET_me")
    assert me["description"] == initial_me["description"]
    search = next(doc for doc in ours.values() if doc["tool_name"] == "GET_search")
    assert "Response:" in search["description"]


def test_ours_stacked_on_draft_keeps_draft_search_and_examples() -> None:
    from tooldoc_nir.restbench_ours import build_ours_instructions

    draft = load_instructions(DRAFT_ROOT, "TMDB", "DRAFT")
    ours, report = build_ours_instructions(draft, base_name="DRAFT")
    popular = ours["1"]
    images = next(
        doc for doc in ours.values() if doc["tool_name"] == "GET_company_company_id_images"
    )
    draft_images = next(
        doc for doc in draft.values() if doc["tool_name"] == "GET_company_company_id_images"
    )
    assert report["base"] == "DRAFT"
    assert "Response:" in popular["description"]
    assert images["tool_description"] == draft_images["description"]
    assert draft_images["description"] in images["description"]
    assert images.get("example") == draft_images.get("example")
    assert "GET_search_company" in images["description"]
    assert images["required_parameters"] == draft_images["required_parameters"]
