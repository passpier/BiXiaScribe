"""txt -> chunk -> embed -> Chroma indexing."""
from __future__ import annotations

import time
from pathlib import Path

import chromadb

from . import config
from .chunking import chunk_text
from .embedding import embed_texts, get_embedding_function

# Chinese novel txt dumps (e.g. the GitHub 金庸/古龍精校版 sources) are
# frequently saved in GB18030 or Big5 rather than UTF-8. Try UTF-8 first
# (the common case) and fall back to these before giving up.
FALLBACK_ENCODINGS = ["utf-8", "gb18030", "big5"]


def _read_text_any_encoding(file_path: Path) -> str:
    raw = file_path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        last_error.encoding if last_error else "utf-8",
        last_error.object if last_error else raw,
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        f"{file_path.name}: could not decode with any of {FALLBACK_ENCODINGS}",
    )


def get_chroma_client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI, reset: bool = False):
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass

    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(config.EMBED_TASK_TYPE_DOCUMENT),
        metadata={"hnsw:space": "cosine"},
    )


def _iter_corpus_files(corpus_path: Path) -> list[Path]:
    if corpus_path.is_file():
        return [corpus_path]
    if corpus_path.is_dir():
        return sorted(corpus_path.glob("*.txt"))
    raise FileNotFoundError(f"Corpus path not found: {corpus_path}")


def _existing_ids(collection, ids: list[str]) -> set[str]:
    """Which of these ids are already in the collection, checked in batches
    to avoid one huge get() call on large corpora."""
    existing: set[str] = set()
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        result = collection.get(ids=batch, include=[])
        existing.update(result["ids"])
    return existing


def build_index(corpus_path: Path | str, reset: bool = False) -> int:
    """Chunk and index every .txt file under `corpus_path` (or the single
    file itself) into the Chroma collection. Returns the number of chunks
    newly written this run.

    Resumable: each embedding-API-batch is upserted to Chroma immediately
    after it succeeds (rather than waiting for the whole file), and chunks
    already present in the collection are skipped before calling the API.
    So if a run dies partway (e.g. hitting a free-tier rate limit), simply
    rerunning the same command picks up where it left off without
    re-embedding — and without re-spending API quota on — chunks already
    indexed.
    """
    corpus_path = Path(corpus_path)
    files = _iter_corpus_files(corpus_path)
    if not files:
        print(f"No .txt files found under {corpus_path}")
        return 0

    client = get_chroma_client()
    collection = get_collection(client, reset=reset)

    newly_indexed = 0
    start_time = time.time()

    for file_path in files:
        text = _read_text_any_encoding(file_path)

        if (
            file_path.name.startswith("webnovel-")
            and config.WEBNOVEL_MAX_CHARS > 0
            and len(text) > config.WEBNOVEL_MAX_CHARS
        ):
            print(
                f"  capping {file_path.name}: {len(text):,} -> "
                f"{config.WEBNOVEL_MAX_CHARS:,} chars (WEBNOVEL_MAX_CHARS)"
            )
            text = text[: config.WEBNOVEL_MAX_CHARS]

        chunks = chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        if not chunks:
            print(f"  skip (empty after chunking): {file_path.name}")
            continue

        ids = [f"{file_path.name}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]

        already_done = _existing_ids(collection, ids)
        pending = [
            (id_, doc, meta)
            for id_, doc, meta in zip(ids, chunks, metadatas)
            if id_ not in already_done
        ]

        if not pending:
            print(f"  already indexed, skipping: {file_path.name} ({len(chunks)} chunks)")
            continue

        if already_done:
            print(
                f"  resuming {file_path.name}: {len(already_done)} chunks already indexed, "
                f"{len(pending)} remaining"
            )

        file_done = 0
        for batch_start in range(0, len(pending), config.EMBED_BATCH_SIZE):
            batch = pending[batch_start : batch_start + config.EMBED_BATCH_SIZE]
            batch_ids = [b[0] for b in batch]
            batch_docs = [b[1] for b in batch]
            batch_metas = [b[2] for b in batch]

            try:
                batch_embeddings = embed_texts(
                    batch_docs, task_type=config.EMBED_TASK_TYPE_DOCUMENT
                )
            except Exception as exc:  # noqa: BLE001 - surface, but preserve progress made so far
                print(
                    f"  stopped after {newly_indexed} newly-indexed chunk(s) this run "
                    f"({file_done}/{len(pending)} into {file_path.name}): {exc}\n"
                    f"  Already-embedded chunks were saved. Rerun the same command to resume."
                )
                raise

            # Precomputed embeddings are passed directly, so Chroma will not
            # re-invoke the (rate-limited) embedding function for this write.
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embeddings,
            )
            file_done += len(batch)
            newly_indexed += len(batch)
            print(f"    {file_done}/{len(pending)} new chunks indexed <- {file_path.name}")

    elapsed = time.time() - start_time
    print(
        f"Done: {newly_indexed} new chunk(s) written in {elapsed:.1f}s. "
        f"Collection now has {collection.count()} vector(s)."
    )
    return newly_indexed
