"""Embedding functions compatible with Chroma's EmbeddingFunction interface,
plus standalone helpers for document/query embedding.

BAAI/bge-m3 run locally via FlagEmbedding: offline, no API key, no quota,
1024-dim dense vectors, strong Chinese retrieval quality.

Notes:
- BGE-M3 dense vectors are ~unit-length already, but we L2-normalize them
  ourselves for consistency, since the collection uses cosine distance.
- task_type (RETRIEVAL_DOCUMENT vs. RETRIEVAL_QUERY) is accepted by
  embed_texts()/CorpusEmbeddingFunction but ignored by BGE-M3 itself, which
  is instruction-free for retrieval. It's kept because it's baked into
  CorpusEmbeddingFunction.name(), which Chroma uses to detect a changed
  embedding function on an existing collection.
"""
from __future__ import annotations

import threading

from chromadb.api.types import EmbeddingFunction

from . import config

_MIN_TORCH_VERSION = (2, 6)  # transformers refuses torch.load below this (CVE-2025-32434)

_local_model = None  # lazily-loaded BGEM3FlagModel singleton
# Guards _local_model's init. CrewAI's native tool-call executor runs multiple
# tool calls from one LLM turn concurrently in a ThreadPoolExecutor (see
# crew/tools.py's WuxiaRetrievalTool, which calls into this module) -- without
# this lock, several threads can all see _local_model is None at once and each
# instantiate their own multi-GB BGEM3FlagModel simultaneously, which is what
# silently OOM-killed a real run. Double-checked locking keeps the fast path
# (already loaded) lock-free.
_local_model_lock = threading.Lock()


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def embed_texts(
    texts: list[str],
    task_type: str,  # noqa: ARG001 - accepted for CorpusEmbeddingFunction.name() compat; BGE-M3 ignores it
    batch_size: int | None = None,
) -> list[list[float]]:
    """Embed a list of texts, batched, returning L2-normalized vectors."""
    if not texts:
        return []
    return _local_embed_texts(texts, batch_size or config.LOCAL_EMBED_BATCH_SIZE)


# --- Local (BGE-M3) backend ---


def _check_local_backend_env() -> None:
    """Fail fast with an actionable message if torch is too old to load
    bge-m3's weights, instead of letting transformers raise its CVE-2025-32434
    ValueError deep inside FlagEmbedding/AutoModel.from_pretrained. This is
    almost always caused by running under the system Python instead of the
    project .venv (see README.md's Quickstart / CLAUDE.md's Setup)."""
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
        with _local_model_lock:
            if _local_model is None:  # re-check: another thread may have won the race
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


class CorpusEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function.

    Subclasses chromadb's EmbeddingFunction (rather than just duck-typing
    __call__) so we inherit its default embed_query() -> __call__ behavior;
    Chroma's query() path calls embed_query specifically, which a bare
    duck-typed object doesn't provide.

    Chroma calls this with a list of documents both when writing (add/upsert)
    and when querying (query_texts), so `task_type` is fixed per-instance:
    use a document-mode instance for the collection's default embedding
    function, and a query-mode instance when issuing queries. task_type has
    no effect on BGE-M3 itself -- it only feeds name()'s collection-naming
    string below, kept for backward compatibility with existing collections.
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


def get_embedding_function(
    task_type: str = config.EMBED_TASK_TYPE_DOCUMENT,
) -> CorpusEmbeddingFunction:
    return CorpusEmbeddingFunction(task_type)
