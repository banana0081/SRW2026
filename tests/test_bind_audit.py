"""Closed producer map vs RestBench gold paths. Offline."""

from tooldoc_nir.bind_audit import gold_hops
from tooldoc_nir.restbench_ours import PRODUCERS


def test_person_credits_is_covered_by_search_person() -> None:
    catalog = {
        "GET_search_person": {"tool_name": "GET_search_person", "url": "http://api.themoviedb.org/3/search/person"},
        "GET_person_person_id_movie_credits": {
            "tool_name": "GET_person_person_id_movie_credits",
            "url": "http://api.themoviedb.org/3/person/{person_id}/movie_credits",
        },
    }
    hops = gold_hops(
        ["GET_search_person", "GET_person_person_id_movie_credits"],
        catalog,
    )
    assert len(hops) == 1
    assert hops[0]["covered"] is True
    assert "GET_search_person" in hops[0]["params"][0]["prior_hit"]


def test_collection_then_reviews_is_a_map_miss() -> None:
    catalog = {
        "GET_search_collection": {
            "tool_name": "GET_search_collection",
            "url": "http://api.themoviedb.org/3/search/collection",
        },
        "GET_collection_collection_id": {
            "tool_name": "GET_collection_collection_id",
            "url": "http://api.themoviedb.org/3/collection/{collection_id}",
        },
        "GET_movie_movie_id_reviews": {
            "tool_name": "GET_movie_movie_id_reviews",
            "url": "http://api.themoviedb.org/3/movie/{movie_id}/reviews",
        },
        "GET_search_movie": {
            "tool_name": "GET_search_movie",
            "url": "http://api.themoviedb.org/3/search/movie",
        },
    }
    hops = gold_hops(
        [
            "GET_search_collection",
            "GET_collection_collection_id",
            "GET_movie_movie_id_reviews",
        ],
        catalog,
    )
    reviews = [row for row in hops if row["tool"] == "GET_movie_movie_id_reviews"][0]
    assert reviews["covered"] is False
    assert "GET_search_movie" in PRODUCERS["movie_id"]
    assert "GET_collection_collection_id" not in PRODUCERS["movie_id"]
