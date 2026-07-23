"""Embedding functions compatible with Chroma's EmbeddingFunction interface,
plus standalone helpers for document/query embedding.

Two backends are supported, selected via config.EMBED_BACKEND:
- "bge-m3" (default): BAAI/bge-m3 run locally via FlagEmbedding. Offline, no
  API quota, 1024-dim dense vectors, strong Chinese retrieval quality.
- "gemini": Google's gemini-embedding-001 API. Subject to free-tier rate
  limits and a hard daily quota (see RATE_LIMIT_BACKOFF_SECONDS below) —
  opt in only if you have paid quota or a small corpus.

Notes:
- gemini-embedding-001 natively outputs 3072-dim vectors; we truncate to
  config.GEMINI_EMBED_DIM via output_dimensionality to save storage/query cost.
  Truncated (Matryoshka) embeddings are NOT unit-length by construction, so
  we L2-normalize them ourselves before handing them to Chroma. This matters
  because the collection uses cosine distance.
- BGE-M3 dense vectors are ~unit-length already, but we re-normalize them
  too for consistency with the above.
- task_type differs between indexing (RETRIEVAL_DOCUMENT) and querying
  (RETRIEVAL_QUERY) for Gemini — using the right one measurably improves
  retrieval quality for asymmetric search. BGE-M3 is instruction-free for
  retrieval, so task_type is ignored on that backend.
"""
from __future__ import annotations

import time

from chromadb.api.types import EmbeddingFunction
from google import genai
from google.genai import errors, types

from . import config

RATE_LIMIT_STATUS_CODE = 429
RATE_LIMIT_BACKOFF_SECONDS = 20.0  # free-tier quota windows are typically per-minute

_MIN_TORCH_VERSION = (2, 6)  # transformers refuses torch.load below this (CVE-2025-32434)

_local_model = None  # lazily-loaded BGEM3FlagModel singleton


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def embed_texts(
    texts: list[str],
    task_type: str,
    batch_size: int | None = None,
    max_retries: int = 5,
) -> list[list[float]]:
    """Embed a list of texts, batched, returning L2-normalized vectors.
    Dispatches to the active backend (config.EMBED_BACKEND)."""
    if not texts:
        return []

    if config.EMBED_BACKEND == "gemini":
        return _gemini_embed_texts(
            texts, task_type, batch_size or config.GEMINI_EMBED_BATCH_SIZE, max_retries
        )
    return _local_embed_texts(texts, batch_size or config.LOCAL_EMBED_BATCH_SIZE)


# --- Local (BGE-M3) backend ---


def _check_local_backend_env() -> None:
    """Fail fast with an actionable message if torch is too old to load
    bge-m3's weights, instead of letting transformers raise its CVE-2025-32434
    ValueError deep inside FlagEmbedding/AutoModel.from_pretrained. This is
    almost always caused by running under the system Python instead of the
    project .venv (see docs/MILESTONES.md's Quickstart)."""
    import sys

    import torch

    try:
        major_minor = tuple(int(p) for p in torch.__version__.split(".")[:2])
    except ValueError:
        return  # unparseable version string -- don't block, let the real load speak
    if major_minor < _MIN_TORCH_VERSION:
        raise RuntimeError(
            f"EMBED_BACKEND=bge-m3 needs torch >= 2.6, but found torch "
            f"{torch.__version__} under {sys.executable}. This usually means "
            f"the script is running under the system Python instead of the "
            f"project venv. Re-run with the venv:\n"
            f"    .venv/bin/python scripts/build_index.py\n"
            f"(or `source .venv/bin/activate` first). To use this interpreter "
            f"anyway, upgrade torch: pip install 'torch>=2.6'."
        )


def _get_local_model():
    global _local_model
    if _local_model is None:
        _check_local_backend_env()
        from FlagEmbedding import BGEM3FlagModel

        kwargs = {"use_fp16": config.LOCAL_EMBED_USE_FP16}
        if config.LOCAL_EMBED_DEVICE:
            kwargs["device"] = config.LOCAL_EMBED_DEVICE
        _local_model = BGEM3FlagModel(config.LOCAL_EMBED_MODEL, **kwargs)
    return _local_model


def _local_embed_texts(texts: list[str], batch_size: int) -> list[list[float]]:
    model = _get_local_model()
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=config.LOCAL_EMBED_MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    # dense_vecs rows are numpy arrays of np.float32; Chroma's upsert
    # validator requires native Python floats, not numpy scalars.
    return [_normalize([float(v) for v in vec]) for vec in output["dense_vecs"]]


# --- Gemini backend ---


def _get_client() -> genai.Client:
    return genai.Client(api_key=config.require_api_key())


def _gemini_embed_texts(
    texts: list[str], task_type: str, batch_size: int, max_retries: int
) -> list[list[float]]:
    client = _get_client()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(_embed_batch_with_retry(client, batch, task_type, max_retries))

    return embeddings


def _retry_after_seconds(exc: errors.APIError) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _embed_batch_with_retry(
    client: genai.Client, batch: list[str], task_type: str, max_retries: int
) -> list[list[float]]:
    delay = 1.0
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=config.GEMINI_EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.GEMINI_EMBED_DIM,
                ),
            )
            return [_normalize(list(e.values)) for e in result.embeddings]
        except errors.APIError as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            if exc.code == RATE_LIMIT_STATUS_CODE:
                # Free-tier quota: back off for a fixed, longer window rather
                # than a short exponential ramp, and honor Retry-After if the
                # API sends one.
                wait = _retry_after_seconds(exc) or RATE_LIMIT_BACKOFF_SECONDS
                print(f"    rate limited (429), waiting {wait:.0f}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                time.sleep(delay)
                delay *= 2
        except Exception as exc:  # noqa: BLE001 - retry on other transient errors
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(
        f"Gemini embedding failed after {max_retries} attempts: {last_error}"
    ) from last_error


class CorpusEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function, backend-aware.

    Subclasses chromadb's EmbeddingFunction (rather than just duck-typing
    __call__) so we inherit its default embed_query() -> __call__ behavior;
    Chroma's query() path calls embed_query specifically, which a bare
    duck-typed object doesn't provide.

    Chroma calls this with a list of documents both when writing (add/upsert)
    and when querying (query_texts), so `task_type` is fixed per-instance:
    use a document-mode instance for the collection's default embedding
    function, and a query-mode instance when issuing queries. (task_type is
    only meaningful for the Gemini backend; BGE-M3 ignores it.)
    """

    def __init__(self, task_type: str = config.EMBED_TASK_TYPE_DOCUMENT):
        self.task_type = task_type

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma's required param name
        return embed_texts(list(input), task_type=self.task_type)

    def name(self) -> str:
        # Required by Chroma's EmbeddingFunction protocol so it can validate
        # a collection's embedding function hasn't silently changed. Encodes
        # backend + model + dim so switching backends is caught rather than
        # silently mixing incompatible vector spaces in one collection.
        return (
            f"{config.EMBED_BACKEND}-{config.EMBED_MODEL}-{config.EMBED_DIM}d-"
            f"{self.task_type.lower()}"
        )


def get_embedding_function(task_type: str = config.EMBED_TASK_TYPE_DOCUMENT) -> CorpusEmbeddingFunction:
    return CorpusEmbeddingFunction(task_type)
