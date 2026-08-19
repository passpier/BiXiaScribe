#!/usr/bin/env python3
"""One-off analysis CLI: for every historical generation run's logged
wuxia_corpus_search queries (out/generation_runs*.jsonl's retrieval_queries
field), replay each query against the real Chroma index and measure how
much of what came back is a chunk that already came back earlier in the
same run.

Motivation (see the "降低長篇劇本生成成本" plan / CLAUDE.md's cost analysis):
prompt_tokens ~ retrieval_calls almost perfectly (R^2=0.99 on the layered
pipeline), and a spot-check of one real run's 13 queries ("慧空沉穩內斂的對話",
"李捕頭說話語氣", "急性子捕頭與僧人對話", ...) showed them clustered around a
handful of underlying questions ("what does this NPC sound like") despite
being lexically distinct. If those near-duplicate queries are mostly
retrieving the *same* underlying chunks, a run-scoped dedup cache
(crew/tools.py's proposed Stage 2.3) is a real, close-to-free win; if the
chunks are mostly distinct, dedup wouldn't help and shouldn't be built.

This script answers that with real numbers instead of a guess -- it costs
nothing (local embedding model + local Chroma, no OpenRouter call) but does
need an already-built index (python scripts/build_index.py).

Usage:
    python scripts/analyze_retrieval_dedup.py
    python scripts/analyze_retrieval_dedup.py --top-k 3 --glob "out/generation_runs*.jsonl"
    python scripts/analyze_retrieval_dedup.py --min-queries 4  # skip short/no-retrieval runs
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.retrieval import (  # noqa: E402
    CollectionNotFoundError,
    get_query_collection,
    retrieve,
)


def _iter_rows(pattern: str):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["_source_file"] = path
                yield row


def _chunk_key(chunk) -> tuple[str, int]:
    """Identity for "is this the same chunk as one we already saw this
    run" -- (source file, chunk_index) is the corpus's own stable identity
    for a chunk, unlike its retrieved text (which is identical for a
    duplicate hit anyway, but comparing by key is cheaper and matches how a
    real dedup cache in crew/tools.py would key its "already shown" set)."""
    return (chunk.source, chunk.chunk_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--glob", default="out/generation_runs*.jsonl")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Must match crew/tools.py's default."
    )
    parser.add_argument(
        "--min-queries", type=int, default=2, help="Skip runs with fewer logged queries than this."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-run detail, not just the aggregate."
    )
    args = parser.parse_args()

    try:
        collection = get_query_collection()
    except CollectionNotFoundError as exc:
        print(f"No Chroma index found: {exc}", file=sys.stderr)
        print("Run `python scripts/build_index.py` first.", file=sys.stderr)
        raise SystemExit(1) from exc

    if collection.count() == 0:
        print("Chroma collection is empty -- run scripts/build_index.py first.", file=sys.stderr)
        raise SystemExit(1)

    dup_ratios: list[float] = []
    chars_saveable: list[int] = []
    rows_seen = 0
    rows_skipped_short = 0
    rows_skipped_no_queries = 0

    for row in _iter_rows(args.glob):
        queries: list[str] = row.get("retrieval_queries") or []
        if not queries:
            rows_skipped_no_queries += 1
            continue
        if len(queries) < args.min_queries:
            rows_skipped_short += 1
            continue

        rows_seen += 1
        seen_keys: set[tuple[str, int]] = set()
        total_chunks = 0
        dup_chunks = 0
        dup_chars = 0

        for query in queries:
            chunks = retrieve(query, top_k=args.top_k, collection=collection)
            for chunk in chunks:
                total_chunks += 1
                key = _chunk_key(chunk)
                if key in seen_keys:
                    dup_chunks += 1
                    dup_chars += len(chunk.text)
                else:
                    seen_keys.add(key)

        ratio = dup_chunks / total_chunks if total_chunks else 0.0
        dup_ratios.append(ratio)
        chars_saveable.append(dup_chars)

        if args.verbose:
            print(
                f"{row.get('variant', '?'):16} {row.get('mode', '?'):8} "
                f"queries={len(queries):3} chunks={total_chunks:3} "
                f"dup={dup_chunks:3} ({ratio:5.1%})  chars_saveable={dup_chars:5}  "
                f"[{row.get('_source_file')}]"
            )

    print()
    print(f"Runs analyzed: {rows_seen}")
    print(f"Runs skipped (no retrieval_queries logged): {rows_skipped_no_queries}")
    print(f"Runs skipped (< --min-queries={args.min_queries}): {rows_skipped_short}")
    if not dup_ratios:
        print("Nothing to summarize -- no run had enough logged queries.")
        return

    print()
    print("Duplicate-chunk ratio across runs (chunk already returned earlier in the same run):")
    print(f"  mean   = {statistics.mean(dup_ratios):.1%}")
    print(f"  median = {statistics.median(dup_ratios):.1%}")
    print(f"  min/max = {min(dup_ratios):.1%} / {max(dup_ratios):.1%}")
    print()
    print("Chars that a run-scoped dedup cache would have saved re-sending:")
    print(f"  mean   = {statistics.mean(chars_saveable):.0f} chars/run")
    print(f"  total  = {sum(chars_saveable):.0f} chars across {rows_seen} runs")
    print()
    print(
        "Interpretation: a high duplicate ratio means Stage 2.3's run-scoped chunk\n"
        "dedup (crew/tools.py) is worth building -- semantically-similar queries are\n"
        "mostly re-fetching the same corpus text. A low ratio means the queries are\n"
        "actually diversifying what gets retrieved, and dedup wouldn't recover much."
    )


if __name__ == "__main__":
    main()
