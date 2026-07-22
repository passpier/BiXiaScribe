"""Reusable retrieval helper on top of the Chroma collection built by
`indexer.build_index`.

This is the interface Stage 2 (CrewAI) agents plug into. It's factored out
of what used to be inline logic in scripts/test_retrieval.py so both the CLI
script and the dialogue agent's tool call the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .embedding import get_embedding_function
from .indexer import get_chroma_client


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float


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


def retrieve(query: str, top_k: int = 5, collection=None) -> list[RetrievedChunk]:
    """Query the index for the top-k chunks most similar to `query`.

    Pass an already-open `collection` (e.g. from get_query_collection()) to
    avoid re-opening it on every call -- useful when a caller issues many
    queries in a loop (like the dialogue agent's retrieval tool).
    """
    if collection is None:
        collection = get_query_collection()

    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=top_k)
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
