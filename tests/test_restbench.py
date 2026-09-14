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


def test_choose_tool_label_reads_the_text_instead_of_being_declared() -> None:
    from tooldoc_nir.restbench_ours import (
        assert_label_matches,
        build_ours_instructions,
        choose_tool_label,
    )

    initial = load_instructions(DRAFT_ROOT, "TMDB", "Initial")
    ours, report = build_ours_instructions(initial)
    assert report["choose_tool_text"].endswith("_with_not")
    assert choose_tool_label(ours) == report["choose_tool_text"]
    assert_label_matches(ours, report["choose_tool_text"])
    stripped = {
        key: {
            **doc,
            "tool_description": str(doc["tool_description"]).replace("not GET_", "GET_"),
        }
        for key, doc in ours.items()
    }
    assert choose_tool_label(stripped).endswith("_no_not")
    try:
        assert_label_matches(stripped, report["choose_tool_text"])
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a mislabelled documentation must not build")


def test_frozen_tmdb_ours_matches_its_recorded_digest() -> None:
    import json

    from tooldoc_nir.provenance import payload_digest

    freeze = Path("artifacts/documentation/baseline_digests.json")
    if not freeze.exists():
        return
    recorded = json.loads(freeze.read_text(encoding="utf-8"))
    entry = recorded["documentation"]["TMDB/Ours"]
    current = json.loads(
        Path("artifacts/documentation/TMDB_Ours.json").read_text(encoding="utf-8")
    )
    assert payload_digest(current) == entry["payload_digest"]


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


def test_stage_mechanics_separate_selection_from_parameter_failures() -> None:
    from tooldoc_nir.restbench_report import mechanics_profile, stage_mechanics

    gold = ["GET_search_person", "GET_person_person_id_movie_credits"]
    hit = stage_mechanics(
        {"gold": gold, "executed": gold, "error": "", "http_errors": 0}
    )
    assert hit["correct_path"] and hit["first_producer_present"]
    assert not hit["wrong_first"] and not hit["wrong_order"]

    stuck = stage_mechanics(
        {
            "gold": gold,
            "executed": ["GET_search_person", "GET_search_person"],
            "error": "",
            "http_errors": 0,
        }
    )
    assert stuck["repeat_producer"] and not stuck["final_consumer_present"]
    assert not stuck["correct_path"]

    swapped = stage_mechanics(
        {"gold": gold, "executed": list(reversed(gold)), "error": "", "http_errors": 0}
    )
    assert swapped["wrong_first"] and swapped["wrong_order"]

    silent = stage_mechanics({"gold": gold, "executed": [], "error": "", "http_errors": 0})
    assert silent["no_call"] and not silent["wrong_first"]

    profile = mechanics_profile(
        [
            {"gold": gold, "executed": gold, "error": "", "http_errors": 0},
            {
                "gold": gold,
                "executed": ["GET_search_person"],
                "error": "",
                "http_errors": 3,
            },
            {"gold": gold, "executed": ["GET_tv_popular"], "error": "", "http_errors": 0},
        ]
    )
    assert profile["misses"] == 2
    assert profile["missing_final_consumer"] == 1
    assert profile["missing_first_producer"] == 1
    assert profile["flags"]["parameter_or_http_error"] == 1


def test_query_indices_pin_the_pilot_panel() -> None:
    import pytest

    from tooldoc_nir.restbench_table import select_queries

    queries = [{"query_id": index} for index in range(100)]
    rows, indices = select_queries(queries, indices=[0, 5, 33, 5])
    assert indices == [0, 5, 33]
    assert [row["query_id"] for row in rows] == [0, 5, 33]
    rows, indices = select_queries(queries, start=10, limit=3)
    assert indices == [10, 11, 12]
    with pytest.raises(SystemExit, match=r"out of range: \[999\]"):
        select_queries(queries, indices=[999])


def test_wilson_and_win_vs_react() -> None:
    from tooldoc_nir.restbench_report import (
        learning_curve,
        report_from_traces,
        wilson_interval,
        win_vs_react,
    )

    interval = wilson_interval(93, 100)
    assert interval["p"] == 0.93
    assert interval["low"] < 0.93 < interval["high"]
    method = [
        {"row": "Ours", "query_index": i, "correct_path": i < 3, "error": ""}
        for i in range(5)
    ]
    react = [
        {"row": "ReAct", "query_index": i, "correct_path": i == 4, "error": ""}
        for i in range(5)
    ]
    wins = win_vs_react(method, react, method_name="Ours")
    assert wins["method_only"] == 3
    assert wins["react_only"] == 1
    assert wins["n"] == 5
    curve = learning_curve(method, step=2)
    assert curve[-1]["n"] == 5
    assert curve[-1]["hits"] == 3
    report = report_from_traces(
        {
            "seed0": method
            + [{"row": "ReAct", "query_index": i, "correct_path": False, "error": ""} for i in range(5)]
        }
    )
    assert report["conditions"]["Ours"]["hits"] == 3
    assert "win_vs_react" in report["conditions"]["Ours"]

