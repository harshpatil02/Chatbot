from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from app.database import get_connection

try:
    from flashrank import Ranker
except Exception:  # pragma: no cover - optional dependency
    Ranker = None

try:
    import bm25s
except Exception:  # pragma: no cover - optional dependency
    bm25s = None


ranker: Any | None = None


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if token]


def lexical_similarity(query: str, text: str) -> float:
    query_terms = set(_tokenize(query))
    text_terms = set(_tokenize(text))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _build_embedding(text: str, dimension: int = 1536) -> List[float]:
    tokens = _tokenize(text)
    vector = [0.0] * dimension
    for token in tokens:
        index = sum(ord(char) for char in token) % dimension
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * dimension
    return [value / norm for value in vector]


def _get_ranker() -> Any | None:
    global ranker
    if ranker is not None:
        return ranker
    if Ranker is None:
        return None
    try:
        ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2")
    except Exception:  # pragma: no cover - dependency/version differences
        ranker = None
    return ranker


def _bm25_score(query: str, text: str, document_frequency: Dict[str, int], average_document_length: float, total_documents: int) -> float:
    query_terms = _tokenize(query)
    document_terms = _tokenize(text)
    if not query_terms or not document_terms:
        return 0.0

    k1 = 1.5
    b = 0.75
    document_length = len(document_terms)
    score = 0.0

    for token in set(query_terms):
        term_frequency = document_terms.count(token)
        if term_frequency == 0:
            continue
        df = document_frequency.get(token, 0)
        idf = math.log((total_documents - df + 0.5) / (df + 0.5) + 1.0)
        numerator = term_frequency * (k1 + 1.0)
        denominator = term_frequency + k1 * (1.0 - b + b * (document_length / average_document_length if average_document_length else 1.0))
        score += idf * (numerator / denominator)

    return score


def bm25_search(query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    if not documents:
        return []

    tokenized_documents = [_tokenize(document["content"]) for document in documents]
    if bm25s is not None and tokenized_documents:
        try:
            import numpy as np

            corpus = [document["content"] for document in documents]
            tokenized = [terms for terms in tokenized_documents if terms]
            if not tokenized:
                return []
            scores = bm25s.BM25Okapi(tokenized).get_scores([query])
            scored = []
            for document, score in zip(documents, scores):
                scored.append({**document, "score": float(score), "bm25_score": float(score)})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[:top_k]
        except Exception:
            pass

    document_frequency: Dict[str, int] = {}
    for terms in tokenized_documents:
        for token in set(terms):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    average_document_length = sum(len(terms) for terms in tokenized_documents) / max(len(tokenized_documents), 1)
    scored = []
    for document in documents:
        score = _bm25_score(query, document["content"], document_frequency, average_document_length, len(documents))
        scored.append({**document, "score": score, "bm25_score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rrf_rerank(*ranked_lists: List[Dict[str, Any]], top_k: int = 5, rank_constant: int = 60) -> List[Dict[str, Any]]:
    scores: Dict[int, Dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):
            doc_id = document.get("id")
            if doc_id is None:
                continue
            entry = scores.setdefault(doc_id, {"id": doc_id, "content": document.get("content", ""), "metadata": document.get("metadata", {}), "score": 0.0})
            entry["score"] += 1.0 / (rank_constant + rank)
            if "bm25_score" in document and "bm25_score" not in entry:
                entry["bm25_score"] = document.get("bm25_score", 0.0)
            if "vector_score" in document and "vector_score" not in entry:
                entry["vector_score"] = document.get("vector_score", 0.0)
            if "lexical_score" in document and "lexical_score" not in entry:
                entry["lexical_score"] = document.get("lexical_score", 0.0)

    reranked = [dict(item) for item in scores.values()]
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        conn = get_connection()
    except Exception:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, chunk_metadata, embedding FROM document_chunks WHERE content IS NOT NULL"
            )
            rows = cur.fetchall()
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not rows:
        return []

    query_embedding = _build_embedding(query)
    lexical_results: List[Dict[str, Any]] = []
    vector_results: List[Dict[str, Any]] = []
    documents = []

    for row in rows:
        content = row["content"]
        lexical_score = lexical_similarity(query, content)
        documents.append({"id": row["id"], "content": content, "metadata": row["chunk_metadata"]})
        lexical_results.append({"id": row["id"], "content": content, "metadata": row["chunk_metadata"], "lexical_score": lexical_score})

        embedding = row.get("embedding")
        if embedding is None:
            vector_score = 0.0
        else:
            vector_score = _cosine_similarity(query_embedding, list(embedding))
        vector_results.append({"id": row["id"], "content": content, "metadata": row["chunk_metadata"], "vector_score": vector_score})

    lexical_results.sort(key=lambda item: item["lexical_score"], reverse=True)
    vector_results.sort(key=lambda item: item["vector_score"], reverse=True)
    bm25_results = bm25_search(query, documents, top_k=10)

    combined = rrf_rerank(
        lexical_results[:10],
        vector_results[:10],
        bm25_results,
        top_k=top_k,
    )

    scored = []
    for item in combined:
        scored.append(
            {
                "id": item["id"],
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "bm25_score": item.get("bm25_score", 0.0),
                "vector_score": item.get("vector_score", 0.0),
                "lexical_score": item.get("lexical_score", 0.0),
                "score": item.get("score", 0.0),
            }
        )

    scored.sort(key=lambda entry: entry["score"], reverse=True)
    return scored[:top_k]


def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not documents:
        return []

    ranker = _get_ranker()
    if ranker is None:
        scored = []
        for document in documents:
            lexical_score = lexical_similarity(query, document["content"])
            scored.append({**document, "reranked_score": lexical_score + (document.get("score", 0.0) / 10.0)})
        scored.sort(key=lambda item: item["reranked_score"], reverse=True)
        return scored

    try:
        pairs = [(query, document["content"]) for document in documents]
        ranked = ranker.rank(pairs) if hasattr(ranker, "rank") else []
        reranked = []
        for document in documents:
            reranked.append({**document, "reranked_score": document.get("score", 0.0)})
        return reranked
    except Exception:  # pragma: no cover
        return documents
