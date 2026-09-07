from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from math import log, log2
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np
import requests


DEFAULT_DRAFT_ROOT = Path("external/DRAFT")
DATASETS = ("TMDB", "Spotify")
CONDITIONS = ("Initial", "DRAFT")
EXPECTED_BM25 = {
    ("TMDB", "Initial"): {"ndcg@1": 0.240, "ndcg@10": 0.350},
    ("TMDB", "DRAFT"): {"ndcg@1": 0.290, "ndcg@10": 0.394},
    ("Spotify", "Initial"): {"ndcg@1": 0.439, "ndcg@10": 0.539},
    ("Spotify", "DRAFT"): {"ndcg@1": 0.439, "ndcg@10": 0.542},
}
EXPECTED_CONTRIEVER = {
    ("TMDB", "Initial"): {"ndcg@1": 0.290, "ndcg@10": 0.404},
    ("TMDB", "DRAFT"): {"ndcg@1": 0.310, "ndcg@10": 0.441},
    ("Spotify", "Initial"): {"ndcg@1": 0.456, "ndcg@10": 0.496},
    ("Spotify", "DRAFT"): {"ndcg@1": 0.474, "ndcg@10": 0.492},
}
QREL_ALIASES = {
    "GET_track_id": "GET_tracks_id",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parameter_text(parameters: Iterable[dict[str, Any]]) -> str:
    values: list[str] = []
    for parameter in parameters:
        schema = parameter.get("schema") or {}
        values.extend(
            str(value)
            for value in (
                parameter.get("name", ""),
                schema.get("type", parameter.get("type", "")),
                parameter.get("description", schema.get("description", "")),
            )
            if value not in (None, "")
        )
    return " ".join(values)


def serialize_document(document: dict[str, Any], mode: str) -> str:
    description = str(document.get("description") or "")
    name = str(document.get("tool_name") or "")
    required = document.get("required_parameters") or []
    optional = document.get("optional_parameters") or []
    parameters = _parameter_text([*required, *optional])
    if mode == "description":
        return description
    if mode == "name_description":
        return f"{name} {description}"
    if mode == "description_parameters":
        return f"{description} {parameters}"
    if mode == "name_description_parameters":
        return f"{name} {description} {parameters}"
    if mode == "colt":
        required_json = json.dumps(
            required,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        optional_json = json.dumps(
            optional,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            f"tool_name: {name}, api_description: {description}, "
            f"required_params: {required_json}, "
            f"optional_params: {optional_json}"
        )
    if mode == "json":
        public_document = {
            key: value
            for key, value in document.items()
            if not key.startswith("_")
        }
        return json.dumps(public_document, ensure_ascii=False, sort_keys=True)
    raise ValueError(f"Unknown serialization mode: {mode}")


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_SKLEARN_WORD_RE = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str, mode: str) -> list[str]:
    lowered = text.lower()
    if mode == "words":
        return _WORD_RE.findall(lowered)
    if mode == "sklearn":
        return _SKLEARN_WORD_RE.findall(lowered)
    if mode == "whitespace":
        return lowered.split()
    raise ValueError(f"Unknown tokenizer mode: {mode}")


class BM25Okapi:
    """Small deterministic implementation matching rank_bm25's BM25Okapi."""

    def __init__(
        self,
        corpus: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = np.asarray([len(document) for document in corpus])
        self.avgdl = float(self.doc_len.mean()) if corpus else 0.0
        self.frequencies = [Counter(document) for document in corpus]

        document_frequency: Counter[str] = Counter()
        for document in corpus:
            document_frequency.update(set(document))
        corpus_size = len(corpus)
        self.idf = {
            term: log(corpus_size - frequency + 0.5) - log(frequency + 0.5)
            for term, frequency in document_frequency.items()
        }
        average_idf = (
            sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        )
        floor = epsilon * average_idf
        for term, value in self.idf.items():
            if value < 0:
                self.idf[term] = floor

    def scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        if not query or not self.corpus or self.avgdl == 0:
            return scores
        denominator_length = self.k1 * (
            1 - self.b + self.b * self.doc_len / self.avgdl
        )
        for term in query:
            frequencies = np.asarray(
                [document.get(term, 0) for document in self.frequencies],
                dtype=float,
            )
            scores += (self.idf.get(term, 0.0) * frequencies * (self.k1 + 1)) / (
                frequencies + denominator_length
            )
        return scores


def _ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    gains = [
        1.0 / log2(rank + 2)
        for rank, document_id in enumerate(ranking[:k])
        if document_id in relevant
    ]
    ideal = sum(
        1.0 / log2(rank + 2) for rank in range(min(k, len(relevant)))
    )
    return sum(gains) / ideal


def _recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def _mean_reciprocal_rank(
    ranking: list[str],
    relevant: set[str],
) -> float:
    for rank, document_id in enumerate(ranking, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def load_restbench(
    root: Path,
    *,
    dataset: str,
    condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents_path = (
        root
        / "dataset"
        / "RestBench"
        / "tool_instruction"
        / f"{dataset}_{condition}.json"
    )
    queries_path = (
        root / "dataset" / "RestBench" / "test_data" / f"{dataset}.json"
    )
    document_map = _read_json(documents_path)
    documents = [
        {**document_map[key], "_corpus_id": f"doc:{key}"}
        for key in sorted(document_map, key=int)
    ]
    queries = _read_json(queries_path)
    return documents, queries


def _document_id(document: dict[str, Any]) -> str:
    return str(document.get("_corpus_id", document["tool_name"]))


def _relevant_document_ids(
    documents: list[dict[str, Any]],
    query: dict[str, Any],
) -> tuple[set[str], list[str]]:
    name_to_ids: dict[str, list[str]] = {}
    for document in documents:
        name = str(document["tool_name"])
        name_to_ids.setdefault(name, []).append(_document_id(document))
    ambiguous = {
        name: ids for name, ids in name_to_ids.items() if len(ids) > 1
    }
    if ambiguous:
        raise ValueError(
            "Tool-name qrels are ambiguous for duplicate corpus documents: "
            f"{ambiguous}"
        )

    relevant: set[str] = set()
    warnings: list[str] = []
    for raw_name in dict.fromkeys(
        str(name) for name in query["relevant APIs"]
    ):
        name = QREL_ALIASES.get(raw_name, raw_name)
        ids = name_to_ids.get(name, [])
        if not ids:
            warnings.append(
                f"query {query['query_id']}: qrel {raw_name!r} "
                "is absent from the corpus"
            )
            continue
        relevant.add(ids[0])
    return relevant, warnings


def evaluate_bm25(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    serialization: str,
    tokenizer: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, Any]:
    document_ids = [_document_id(document) for document in documents]
    corpus = [
        tokenize(serialize_document(document, serialization), tokenizer)
        for document in documents
    ]
    retriever = BM25Okapi(corpus, k1=k1, b=b)
    per_query: list[dict[str, Any]] = []
    qrel_warnings: list[str] = []
    for query in queries:
        scores = retriever.scores(tokenize(str(query["query"]), tokenizer))
        order = np.argsort(-scores, kind="stable")
        ranking = [document_ids[index] for index in order]
        relevant, warnings = _relevant_document_ids(documents, query)
        qrel_warnings.extend(warnings)
        per_query.append(
            {
                "query_id": query["query_id"],
                "ndcg@1": _ndcg_at_k(ranking, relevant, 1),
                "ndcg@10": _ndcg_at_k(ranking, relevant, 10),
                "recall@1": _recall_at_k(ranking, relevant, 1),
                "recall@10": _recall_at_k(ranking, relevant, 10),
                "mrr": _mean_reciprocal_rank(ranking, relevant),
                "ranking": ranking[:10],
            }
        )
    metric_names = ("ndcg@1", "ndcg@10", "recall@1", "recall@10", "mrr")
    return {
        "queries": len(per_query),
        "documents": len(documents),
        "serialization": serialization,
        "tokenizer": tokenizer,
        "metrics": {
            metric: round(
                sum(float(row[metric]) for row in per_query) / len(per_query),
                6,
            )
            for metric in metric_names
        },
        "qrel_warnings": qrel_warnings,
        "per_query": per_query,
    }


def _ranking_metrics(
    *,
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    score_matrix: np.ndarray,
) -> dict[str, Any]:
    document_ids = [_document_id(document) for document in documents]
    per_query: list[dict[str, Any]] = []
    qrel_warnings: list[str] = []
    for query, scores in zip(queries, score_matrix, strict=True):
        order = np.argsort(-scores, kind="stable")
        ranking = [document_ids[index] for index in order]
        relevant, warnings = _relevant_document_ids(documents, query)
        qrel_warnings.extend(warnings)
        per_query.append(
            {
                "query_id": query["query_id"],
                "ndcg@1": _ndcg_at_k(ranking, relevant, 1),
                "ndcg@10": _ndcg_at_k(ranking, relevant, 10),
                "recall@1": _recall_at_k(ranking, relevant, 1),
                "recall@10": _recall_at_k(ranking, relevant, 10),
                "mrr": _mean_reciprocal_rank(ranking, relevant),
                "ranking": ranking[:10],
            }
        )
    metric_names = ("ndcg@1", "ndcg@10", "recall@1", "recall@10", "mrr")
    return {
        "queries": len(per_query),
        "documents": len(document_ids),
        "metrics": {
            metric: round(
                sum(float(row[metric]) for row in per_query) / len(per_query),
                6,
            )
            for metric in metric_names
        },
        "qrel_warnings": qrel_warnings,
        "per_query": per_query,
    }


@lru_cache(maxsize=4)
def _sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def evaluate_contriever(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    serialization: str,
    model_name: str,
    batch_size: int,
    similarity: str,
) -> dict[str, Any]:
    if similarity not in {"cosine", "dot"}:
        raise ValueError(f"Unknown similarity: {similarity}")
    model = _sentence_transformer(model_name)
    document_texts = [
        serialize_document(document, serialization) for document in documents
    ]
    query_texts = [str(query["query"]) for query in queries]
    document_embeddings = model.encode(
        document_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=similarity == "cosine",
        show_progress_bar=False,
    )
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=similarity == "cosine",
        show_progress_bar=False,
    )
    score_matrix = query_embeddings @ document_embeddings.T
    result = _ranking_metrics(
        documents=documents,
        queries=queries,
        score_matrix=score_matrix,
    )
    result.update(
        {
            "serialization": serialization,
            "model": model_name,
            "similarity": similarity,
        }
    )
    return result


def evaluate_elasticsearch_bm25(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    serialization: str,
    elasticsearch_url: str,
    index_name: str,
) -> dict[str, Any]:
    endpoint = elasticsearch_url.rstrip("/")
    requests.delete(f"{endpoint}/{index_name}", timeout=30)
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "title": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"},
            }
        },
    }
    response = requests.put(
        f"{endpoint}/{index_name}",
        json=mapping,
        timeout=30,
    )
    response.raise_for_status()

    bulk_lines: list[str] = []
    for document in documents:
        document_id = _document_id(document)
        bulk_lines.append(
            json.dumps(
                {"index": {"_index": index_name, "_id": document_id}},
                ensure_ascii=False,
            )
        )
        bulk_lines.append(
            json.dumps(
                {
                    "title": "",
                    "text": serialize_document(document, serialization),
                },
                ensure_ascii=False,
            )
        )
    response = requests.post(
        f"{endpoint}/_bulk?refresh=wait_for",
        data="\n".join(bulk_lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
        timeout=60,
    )
    response.raise_for_status()
    bulk_result = response.json()
    if bulk_result.get("errors"):
        raise RuntimeError(f"Elasticsearch bulk indexing failed: {bulk_result}")

    request_lines: list[str] = []
    for query in queries:
        request_lines.append(
            json.dumps(
                {"index": index_name, "search_type": "dfs_query_then_fetch"}
            )
        )
        request_lines.append(
            json.dumps(
                {
                    "_source": False,
                    "query": {
                        "multi_match": {
                            "query": str(query["query"]),
                            "type": "best_fields",
                            "fields": ["title", "text"],
                            "tie_breaker": 0.5,
                        }
                    },
                    "size": min(11, len(documents)),
                }
            )
        )
    response = requests.post(
        f"{endpoint}/_msearch",
        data="\n".join(request_lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
        timeout=120,
    )
    response.raise_for_status()
    search_results = response.json()["responses"]

    per_query: list[dict[str, Any]] = []
    qrel_warnings: list[str] = []
    for query, search_result in zip(queries, search_results, strict=True):
        if "error" in search_result:
            raise RuntimeError(f"Elasticsearch query failed: {search_result}")
        ranking = [
            str(hit["_id"])
            for hit in search_result.get("hits", {}).get("hits", [])
            if str(hit["_id"]) != str(query["query_id"])
        ][:10]
        relevant, warnings = _relevant_document_ids(documents, query)
        qrel_warnings.extend(warnings)
        per_query.append(
            {
                "query_id": query["query_id"],
                "ndcg@1": _ndcg_at_k(ranking, relevant, 1),
                "ndcg@10": _ndcg_at_k(ranking, relevant, 10),
                "recall@1": _recall_at_k(ranking, relevant, 1),
                "recall@10": _recall_at_k(ranking, relevant, 10),
                "mrr": _mean_reciprocal_rank(ranking, relevant),
                "ranking": ranking,
            }
        )
    metric_names = ("ndcg@1", "ndcg@10", "recall@1", "recall@10", "mrr")
    requests.delete(f"{endpoint}/{index_name}", timeout=30)
    return {
        "queries": len(per_query),
        "documents": len(documents),
        "serialization": serialization,
        "engine": "Elasticsearch BM25",
        "elasticsearch_version_target": "7.17.9",
        "analyzer": "english",
        "search_type": "dfs_query_then_fetch",
        "metrics": {
            metric: round(
                sum(float(row[metric]) for row in per_query) / len(per_query),
                6,
            )
            for metric in metric_names
        },
        "qrel_warnings": qrel_warnings,
        "per_query": per_query,
    }


def run_bm25_matrix(
    *,
    root: Path,
    serializations: list[str],
    tokenizers: list[str],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for serialization in serializations:
        for tokenizer_mode in tokenizers:
            for dataset in DATASETS:
                for condition in CONDITIONS:
                    documents, queries = load_restbench(
                        root,
                        dataset=dataset,
                        condition=condition,
                    )
                    result = evaluate_bm25(
                        documents,
                        queries,
                        serialization=serialization,
                        tokenizer=tokenizer_mode,
                    )
                    expected = EXPECTED_BM25[(dataset, condition)]
                    result.update(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "expected": expected,
                            "absolute_error": round(
                                sum(
                                    abs(result["metrics"][metric] - value)
                                    for metric, value in expected.items()
                                ),
                                6,
                            ),
                        }
                    )
                    result.pop("per_query")
                    runs.append(result)
    ranked_settings: list[dict[str, Any]] = []
    for serialization in serializations:
        for tokenizer_mode in tokenizers:
            selected = [
                run
                for run in runs
                if run["serialization"] == serialization
                and run["tokenizer"] == tokenizer_mode
            ]
            ranked_settings.append(
                {
                    "serialization": serialization,
                    "tokenizer": tokenizer_mode,
                    "total_absolute_error": round(
                        sum(run["absolute_error"] for run in selected),
                        6,
                    ),
                }
            )
    ranked_settings.sort(
        key=lambda item: (item["serialization"], item["tokenizer"])
    )
    return {
        "retriever": "BM25Okapi",
        "draft_commit": "324685d1b65622fe4d6ccdc11473f1e0f62af5d0",
        "runs": runs,
        "serialization_tokenizer_sensitivity": ranked_settings,
        "note": (
            "DRAFT does not publish its retrieval implementation or document "
            "serialization. This matrix reports standard BM25 variants without "
            "using the expected scores to alter rankings."
        ),
    }


def run_elasticsearch_bm25_matrix(
    *,
    root: Path,
    serializations: list[str],
    elasticsearch_url: str,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for serialization_index, serialization in enumerate(serializations):
        for dataset in DATASETS:
            for condition in CONDITIONS:
                documents, queries = load_restbench(
                    root,
                    dataset=dataset,
                    condition=condition,
                )
                index_name = (
                    f"draft-{serialization_index}-{dataset}-{condition}"
                ).lower()
                result = evaluate_elasticsearch_bm25(
                    documents,
                    queries,
                    serialization=serialization,
                    elasticsearch_url=elasticsearch_url,
                    index_name=index_name,
                )
                expected = EXPECTED_BM25[(dataset, condition)]
                result.update(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "expected": expected,
                        "absolute_error": round(
                            sum(
                                abs(result["metrics"][metric] - value)
                                for metric, value in expected.items()
                            ),
                            6,
                        ),
                    }
                )
                result.pop("per_query")
                runs.append(result)
    sensitivity = [
        {
            "serialization": serialization,
            "total_absolute_error": round(
                sum(
                    run["absolute_error"]
                    for run in runs
                    if run["serialization"] == serialization
                ),
                6,
            ),
        }
        for serialization in sorted(serializations)
    ]
    return {
        "retriever": "Elasticsearch BM25",
        "draft_commit": "324685d1b65622fe4d6ccdc11473f1e0f62af5d0",
        "runs": runs,
        "serialization_sensitivity": sensitivity,
        "note": (
            "This follows the author-adjacent COLT/BEIR stack: Elasticsearch "
            "English analyzer, default BM25 k1=1.2/b=0.75, one shard and "
            "dfs_query_then_fetch. DRAFT itself does not publish Table 3 code."
        ),
    }


def run_contriever_matrix(
    *,
    root: Path,
    serializations: list[str],
    model_name: str,
    batch_size: int,
    similarity: str,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for serialization in serializations:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                documents, queries = load_restbench(
                    root,
                    dataset=dataset,
                    condition=condition,
                )
                result = evaluate_contriever(
                    documents,
                    queries,
                    serialization=serialization,
                    model_name=model_name,
                    batch_size=batch_size,
                    similarity=similarity,
                )
                expected = EXPECTED_CONTRIEVER[(dataset, condition)]
                result.update(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "expected": expected,
                        "absolute_error": round(
                            sum(
                                abs(result["metrics"][metric] - value)
                                for metric, value in expected.items()
                            ),
                            6,
                        ),
                    }
                )
                result.pop("per_query")
                runs.append(result)
    ranked_settings = []
    for serialization in serializations:
        selected = [
            run for run in runs if run["serialization"] == serialization
        ]
        ranked_settings.append(
            {
                "serialization": serialization,
                "total_absolute_error": round(
                    sum(run["absolute_error"] for run in selected),
                    6,
                ),
            }
        )
    ranked_settings.sort(key=lambda item: item["serialization"])
    return {
        "retriever": "Contriever",
        "model": model_name,
        "draft_commit": "324685d1b65622fe4d6ccdc11473f1e0f62af5d0",
        "runs": runs,
        "serialization_sensitivity": ranked_settings,
        "note": (
            "DRAFT does not publish its Contriever checkpoint, serialization, "
            "pooling or normalization. The settings here are explicit sensitivity "
            "runs; reported-score proximity must not be used to select a protocol."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce DRAFT's RestBench retrieval comparison."
    )
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument(
        "--retriever",
        choices=["bm25", "bm25_es", "contriever"],
        default="bm25",
    )
    parser.add_argument(
        "--serializations",
        nargs="+",
        default=["description"],
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["words"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/draft_bm25_reproduction.json"),
    )
    parser.add_argument(
        "--contriever-model",
        default="nthakur/contriever-base-msmarco",
    )
    parser.add_argument(
        "--similarity",
        choices=["cosine", "dot"],
        default="cosine",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--elasticsearch-url",
        default="http://localhost:9200",
    )
    args = parser.parse_args()

    if args.retriever == "contriever":
        result = run_contriever_matrix(
            root=args.draft_root,
            serializations=args.serializations,
            model_name=args.contriever_model,
            batch_size=args.batch_size,
            similarity=args.similarity,
        )
    elif args.retriever == "bm25_es":
        result = run_elasticsearch_bm25_matrix(
            root=args.draft_root,
            serializations=args.serializations,
            elasticsearch_url=args.elasticsearch_url,
        )
    else:
        result = run_bm25_matrix(
            root=args.draft_root,
            serializations=args.serializations,
            tokenizers=args.tokenizers,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_key = (
        "serialization_sensitivity"
        if args.retriever in {"contriever", "bm25_es"}
        else "serialization_tokenizer_sensitivity"
    )
    print(json.dumps(result[summary_key], indent=2))


if __name__ == "__main__":
    main()
