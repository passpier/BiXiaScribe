#!/usr/bin/env python3
"""CLI: run a query against the Chroma index and print top-k retrieved chunks,
to sanity-check that retrieval quality is reasonable.

Usage:
    python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.retrieval import (  # noqa: E402
    CollectionNotFoundError,
    get_query_collection,
    retrieve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Query text (Chinese OK).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--preview-len", type=int, default=120, help="Chars of chunk text to print."
    )
    args = parser.parse_args()

    try:
        collection = get_query_collection()
    except CollectionNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    if collection.count() == 0:
        print("Collection is empty. Run scripts/build_index.py first.")
        raise SystemExit(1)

    chunks = retrieve(args.query, top_k=args.top_k, collection=collection)

    print(f"Query: {args.query!r}  (top {len(chunks)} of {collection.count()} indexed chunks)\n")
    for rank, chunk in enumerate(chunks, start=1):
        preview = chunk.text[: args.preview_len].replace("\n", " ")
        print(
            f"[{rank}] distance={chunk.distance:.4f}  "
            f"source={chunk.source}  chunk={chunk.chunk_index}"
        )
        print(f"    {preview}{'...' if len(chunk.text) > args.preview_len else ''}\n")


if __name__ == "__main__":
    main()
