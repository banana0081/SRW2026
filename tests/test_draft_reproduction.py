from math import log2
from pathlib import Path

from tooldoc_nir.draft_reproduction import (
    BM25Okapi,
    _mean_reciprocal_rank,
    _ndcg_at_k,
    _recall_at_k,
    _relevant_document_ids,
    evaluate_bm25,
    load_restbench,
)

DRAFT_ROOT = Path("external/DRAFT")


def test_ranking_metrics_use_all_relevant_tools() -> None:
    ranking = ["irrelevant", "relevant_a", "relevant_b"]
    relevant = {"relevant_a", "relevant_b"}

    assert _ndcg_at_k(ranking, relevant, 1) == 0.0
    assert 0.0 < _ndcg_at_k(ranking, relevant, 3) < 1.0
    assert _recall_at_k(ranking, relevant, 2) == 0.5
    assert _mean_reciprocal_rank(ranking, relevant) == 0.5


def test_bm25_ranks_matching_document_first() -> None:
    retriever = BM25Okapi(
        [
            ["movie", "reviews"],
            ["weather", "forecast"],
            ["music", "albums"],
        ]
    )

    scores = retriever.scores(["weather"])
    assert scores.argmax() == 1


def test_binary_ndcg_matches_standard_closed_form() -> None:
    ranking = ["irrelevant", "relevant_a"] + [
        f"irrelevant_{index}" for index in range(20)
    ]
    expected = (1.0 / log2(3)) / (1.0 + 1.0 / log2(3))

    assert abs(
        _ndcg_at_k(ranking, {"relevant_a", "relevant_b"}, 10) - expected
    ) < 1e-12
    assert _ndcg_at_k(["gold"], {"gold", "other"}, 1) == 1.0


def test_released_spotify_qrel_alias_is_resolved() -> None:
    documents, queries = load_restbench(
        DRAFT_ROOT,
        dataset="Spotify",
        condition="Initial",
    )
    query = queries[39]

    assert "GET_track_id" in query["relevant APIs"]
    relevant, warnings = _relevant_document_ids(documents, query)
    tracks_id = next(
        document["_corpus_id"]
        for document in documents
        if document["tool_name"] == "GET_tracks_id"
    )
    assert tracks_id in relevant
    assert warnings == []


def test_canonical_bm25_recovers_published_tmdb_hit_at_one() -> None:
    documents, queries = load_restbench(
        DRAFT_ROOT,
        dataset="TMDB",
        condition="Initial",
    )
    result = evaluate_bm25(
        documents,
        queries,
        serialization="description",
        tokenizer="words",
    )

    assert result["metrics"]["ndcg@1"] == 0.24
    assert result["qrel_warnings"] == []
