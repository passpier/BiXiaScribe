"""Reusable retrieval helper on top of the Chroma collection built by
`indexer.build_index`.

This is the interface Stage 2 (CrewAI) agents plug into. It's factored out
of what used to be inline logic in scripts/test_retrieval.py so both the CLI
script and the dialogue agent's tool call the same code path.

Two retrieval modes, selected by config.RETRIEVAL_MODE:
- "vector": Chroma's embedding-similarity search only (the original behavior).
- "hybrid" (default): fuses vector search with a hand-rolled BM25 keyword
  index (lexical.py) via Reciprocal Rank Fusion. Vector search alone tends to
  under-match exact wuxia proper nouns (獨孤九劍, 六脈神劍) since embeddings
  favor semantic similarity over exact tokens; BM25 fixes that, RRF fixes the
  fact that cosine distance and BM25 scores aren't on comparable scales (RRF
  only needs each method's *rank*, not its raw score, to combine them).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import config
from .embedding import get_embedding_function
from .indexer import get_chroma_client
from .lexical import BM25Index

# Reciprocal Rank Fusion constant. 60 is the value used in the original RRF
# paper (Cormack et al., 2009) and is not sensitive to corpus size -- it just
# damps the influence of very low ranks relative to rank 1.
_RRF_K = 60


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float | None = None
    rrf_score: float | None = None


class CollectionNotFoundError(RuntimeError):
    """Raised when the Chroma collection hasn't been built yet."""


def get_query_collection():
    """Open the indexed collection in query mode (RETRIEVAL_QUERY task_type).

    Raises CollectionNotFoundError with a clear message if scripts/build_index.py
    hasn't been run yet -- mirrors the guidance in test_retrieval.py.
    """
    client = get_chroma_client()
    try:
        return client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=get_embedding_function(config.EMBED_TASK_TYPE_QUERY),
        )
    except Exception as exc:  # noqa: BLE001
        raise CollectionNotFoundError(
            f"Could not open collection '{config.COLLECTION_NAME}': {exc}\n"
            "Did you run scripts/build_index.py first?"
        ) from exc


# Lazy per-process cache of (collection identity, BM25Index, ids, documents,
# metadatas) -- built once from Chroma's stored documents on first hybrid
# query, then reused for every subsequent query in the process (mirrors
# embedding.py's `_local_model` singleton). Keyed on (name, count) so a
# `--reset` + rebuild mid-process (as in tests) invalidates the cache instead
# of silently searching stale documents.
_bm25_cache: tuple[tuple[str, int], BM25Index, list[str], list[str], list[dict]] | None = None


def _get_bm25_index(collection) -> tuple[BM25Index, list[str], list[str], list[dict]]:
    global _bm25_cache
    cache_key = (collection.name, collection.count())
    if _bm25_cache is not None and _bm25_cache[0] == cache_key:
        return _bm25_cache[1:]

    stored = collection.get(include=["documents", "metadatas"])
    ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]
    index = BM25Index(documents)

    _bm25_cache = (cache_key, index, ids, documents, metadatas)
    return index, ids, documents, metadatas


def _vector_search(query: str, top_n: int, collection) -> list[RetrievedChunk]:
    results = collection.query(query_texts=[query], n_results=top_n)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        RetrievedChunk(
            text=doc,
            source=meta.get("source", ""),
            chunk_index=meta.get("chunk_index", -1),
            distance=dist,
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def _hybrid_search(query: str, top_k: int, collection) -> list[RetrievedChunk]:
    # Pull a wider candidate pool from each method than top_k so RRF has
    # enough overlap/ranking signal to work with before truncating.
    pool_n = max(top_k * 4, 20)

    vector_hits = _vector_search(query, pool_n, collection)
    index, _ids, documents, metadatas = _get_bm25_index(collection)
    bm25_hits = index.search(query, top_n=pool_n)

    # RRF: score(doc) = sum over methods ranking it of 1 / (k + rank), rank
    # starting at 1. Using rank rather than each method's raw score sidesteps
    # cosine-distance vs. BM25-score being on incomparable numeric scales.
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_by_key: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_hits, start=1):
        key = f"{chunk.source}::{chunk.chunk_index}"
        rrf_scores[key] += 1.0 / (_RRF_K + rank)
        chunk_by_key[key] = chunk

    for rank, (doc_idx, _bm25_score) in enumerate(bm25_hits, start=1):
        meta = metadatas[doc_idx]
        key = f"{meta.get('source', '')}::{meta.get('chunk_index', -1)}"
        rrf_scores[key] += 1.0 / (_RRF_K + rank)
        if key not in chunk_by_key:
            chunk_by_key[key] = RetrievedChunk(
                text=documents[doc_idx],
                source=meta.get("source", ""),
                chunk_index=meta.get("chunk_index", -1),
            )

    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    results = []
    for key in ranked_keys:
        chunk = chunk_by_key[key]
        chunk.rrf_score = rrf_scores[key]
        results.append(chunk)
    return results


def retrieve(
    query: str, top_k: int = 5, collection=None, mode: str | None = None
) -> list[RetrievedChunk]:
    """Query the index for the top-k chunks most relevant to `query`.

    Pass an already-open `collection` (e.g. from get_query_collection()) to
    avoid re-opening it on every call -- useful when a caller issues many
    queries in a loop (like the dialogue agent's retrieval tool).

    `mode` overrides config.RETRIEVAL_MODE ("vector" or "hybrid") for this
    call only -- e.g. scripts/eval_retrieval.py runs both modes side by side.
    """
    if collection is None:
        collection = get_query_collection()

    if collection.count() == 0:
        return []

    mode = (mode or config.RETRIEVAL_MODE).strip().lower()
    if mode == "hybrid":
        return _hybrid_search(query, top_k, collection)
    return _vector_search(query, top_k, collection)
