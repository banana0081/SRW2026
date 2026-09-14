"""Frozen Spotify identifiers are seed-independent and reject invented docs examples."""

from __future__ import annotations

from pathlib import Path

from tooldoc_nir.restbench_screen import load_execute_record
from tooldoc_nir.restbench_spotify_ids import gold_id_misses, load_gold_ids

REPO = Path(__file__).resolve().parents[1]
Q002 = (
    REPO
    / "artifacts"
    / "results"
    / "restbench_spotify_ling_s00"
    / "ours"
    / "q002"
    / "record.jsonl"
)


def _record(playlist: str, uri: str, search_body: str) -> dict:
    return {
        "execute_log": {
            "api_result_ls": [
                [{"api_name": "GET_search", "parameters": {"q": "Summertime Sadness", "type": "track"}}],
                [{"api_name": "GET_me_playlists", "parameters": {}}],
                [
                    {
                        "api_name": "POST_playlists_playlist_id_tracks",
                        "parameters": {"playlist_id": playlist, "uris": uri},
                    }
                ],
            ],
            "call_result_ls": [
                "{'error': '', 'response': %r}" % search_body,
                "{'error': '', 'response': '{\"items\":[{\"id\":\"0LMjOXEvbgonT0jK3iAqsZ\"}]}'}",
                "{'error': 'HTTP 403', 'response': '{\"error\":{\"message\":\"Forbidden\"}}'}",
            ],
        }
    }


def test_the_docs_example_track_is_not_summertime_sadness() -> None:
    table = load_gold_ids()
    gold = [
        "GET_search",
        "GET_me_playlists",
        "POST_playlists_playlist_id_tracks",
    ]
    question = "Add Summertime Sadness by Lana Del Rey in my first playlist"
    search = '{"tracks":{"items":[{"id":"3BJe4B8zGnqEdQPMvfVjuS","uri":"spotify:track:3BJe4B8zGnqEdQPMvfVjuS"}]}}'
    invented = _record(
        "0LMjOXEvbgonT0jK3iAqsZ",
        "spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
        search,
    )
    correct = _record(
        "0LMjOXEvbgonT0jK3iAqsZ",
        "spotify:track:3BJe4B8zGnqEdQPMvfVjuS",
        search,
    )
    assert gold_id_misses(invented, question=question, gold=gold, table=table) >= 1
    assert gold_id_misses(correct, question=question, gold=gold, table=table) == 0


def test_ling_s00_ours_q002_used_the_docs_example_uri() -> None:
    record = load_execute_record(Q002)
    assert record is not None
    gold = [
        "GET_search",
        "GET_me_playlists",
        "POST_playlists_playlist_id_tracks",
    ]
    question = "Add Summertime Sadness by Lana Del Rey in my first playlist"
    assert gold_id_misses(record, question=question, gold=gold, table=load_gold_ids()) >= 1
