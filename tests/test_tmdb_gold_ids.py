"""Named-entity freeze rejects invented TMDB ids; list hops harvest the producer."""

from tooldoc_nir.restbench_tmdb_ids import extract_named_terms, gold_id_misses


def _record(calls: list[tuple[str, dict, str, str]]) -> dict:
    apis = []
    results = []
    for name, params, error, body in calls:
        apis.append([{"api_name": name, "parameters": params}])
        results.append({"error": error, "response": body})
    return {"execute_log": {"api_result_ls": apis, "call_result_ls": results}}


TABLE = {
    "named": {
        "titanic": {"movie_ids": ["597"], "tv_ids": [], "person_ids": []},
        "fast_and_the_furious": {
            "collection_ids": ["9485"],
            "movie_ids": ["584", "9615", "9799"],
        },
    },
    "hints": [
        ["titanic", "titanic"],
        ["fast and the furious", "fast_and_the_furious"],
    ],
    "by_query": {},
}


def test_extracts_named_people_and_collections() -> None:
    assert ("person", "Sofia Coppola") in extract_named_terms(
        "give me the number of movies directed by Sofia Coppola",
        ["GET_search_person", "GET_person_person_id_movie_credits"],
    )
    assert ("collection", "The Fast and the Furious") in extract_named_terms(
        "Give me a review of a movie from the collection The Fast and the Furious.",
        ["GET_search_collection", "GET_collection_collection_id", "GET_movie_movie_id_reviews"],
    )


def test_invented_movie_id_is_a_miss() -> None:
    question = "give me some reviews of Titanic"
    gold = ["GET_search_movie", "GET_movie_movie_id_reviews"]
    search = '{"results":[{"id":597,"title":"Titanic"}]}'
    invented = _record(
        [
            ("GET_search_movie", {"query": "Titanic"}, "", search),
            ("GET_movie_movie_id_reviews", {"movie_id": "11111"}, "HTTP 404", ""),
        ]
    )
    correct = _record(
        [
            ("GET_search_movie", {"query": "Titanic"}, "", search),
            ("GET_movie_movie_id_reviews", {"movie_id": "597"}, "", '{"results":[]}'),
        ]
    )
    assert gold_id_misses(invented, question=question, gold=gold, table=TABLE) >= 1
    assert gold_id_misses(correct, question=question, gold=gold, table=TABLE) == 0


def test_popular_list_harvests_the_producer() -> None:
    question = "What is the most popular movie right now and what is its keywords?"
    gold = ["GET_movie_popular", "GET_movie_movie_id_keywords"]
    popular = '{"results":[{"id":42,"title":"Now"}]}'
    ok = _record(
        [
            ("GET_movie_popular", {}, "", popular),
            ("GET_movie_movie_id_keywords", {"movie_id": "42"}, "", '{"keywords":[]}'),
        ]
    )
    bad = _record(
        [
            ("GET_movie_popular", {}, "", popular),
            ("GET_movie_movie_id_keywords", {"movie_id": "11111"}, "HTTP 404", ""),
        ]
    )
    assert gold_id_misses(ok, question=question, gold=gold, table=TABLE) == 0
    assert gold_id_misses(bad, question=question, gold=gold, table=TABLE) >= 1


def test_collection_parts_are_allowed_movie_ids() -> None:
    question = "Give me a review of a movie from the collection The Fast and the Furious."
    gold = [
        "GET_search_collection",
        "GET_collection_collection_id",
        "GET_movie_movie_id_reviews",
    ]
    search = '{"results":[{"id":9485,"name":"The Fast and the Furious Collection"}]}'
    collection = '{"id":9485,"parts":[{"id":584,"title":"2 Fast 2 Furious"}]}'
    ok = _record(
        [
            ("GET_search_collection", {"query": "Fast"}, "", search),
            ("GET_collection_collection_id", {"collection_id": "9485"}, "", collection),
            ("GET_movie_movie_id_reviews", {"movie_id": "584"}, "", '{"results":[]}'),
        ]
    )
    stub = _record(
        [
            ("GET_search_collection", {"query": "Fast"}, "", search),
            ("GET_collection_collection_id", {"collection_id": "9485"}, "", collection),
            ("GET_movie_movie_id_reviews", {"movie_id": "9485"}, "HTTP 404", ""),
        ]
    )
    assert gold_id_misses(ok, question=question, gold=gold, table=TABLE) == 0
    assert gold_id_misses(stub, question=question, gold=gold, table=TABLE) >= 1
