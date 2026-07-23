#!/usr/bin/env python3
"""CLI: run every query in eval/retrieval_eval.jsonl against both "vector" and
"hybrid" retrieval modes and print a side-by-side comparison, so "hybrid
should help wuxia proper nouns" is a repeatable number instead of eyeballing
scripts/test_retrieval.py output on one query at a time.

Metrics per query (only computed against queries that specify ground truth --
the eval set's last few "thematic" queries deliberately have none, since
they're semantic/atmosphere queries with no single correct source):
  - source-hit@k : did >=1 retrieved chunk come from a `relevant_sources` file?
  - term-hit@k   : did >=1 retrieved chunk's text contain a `must_contain` term?
  - MRR          : reciprocal rank of the first chunk from a relevant source
                   (0 if none of the top-k are relevant).

Usage:
    python scripts/eval_retrieval.py [--top-k 5] [--eval-file eval/retrieval_eval.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.retrieval import (  # noqa: E402
    CollectionNotFoundError,
    RetrievedChunk,
    get_query_collection,
    retrieve,
)

DEFAULT_EVAL_FILE = Path(__file__).resolve().parents[1] / "eval" / "retrieval_eval.jsonl"
MODES = ["vector", "hybrid"]


def _load_queries(path: Path) -> list[dict]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        queries.append(json.loads(line))
    return queries


def _score(
    chunks: list[RetrievedChunk], relevant_sources: list[str], must_contain: list[str]
) -> dict:
    has_ground_truth = bool(relevant_sources) or bool(must_contain)
    if not has_ground_truth:
        return {"has_ground_truth": False}

    source_hit = False
    term_hit = False
    mrr = 0.0

    for rank, chunk in enumerate(chunks, start=1):
        is_relevant_source = relevant_sources and chunk.source in relevant_sources
        if is_relevant_source:
            source_hit = True
            if mrr == 0.0:
                mrr = 1.0 / rank
        if must_contain and any(term in chunk.text for term in must_contain):
            term_hit = True

    return {
        "has_ground_truth": True,
        "source_hit": source_hit if relevant_sources else None,
        "term_hit": term_hit if must_contain else None,
        "mrr": mrr if relevant_sources else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results per query per mode.")
    parser.add_argument(
        "--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Path to the eval JSONL file."
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

    queries = _load_queries(args.eval_file)
    print(f"Evaluating {len(queries)} queries from {args.eval_file} at top-{args.top_k}\n")

    # results[mode] -> list of per-query score dicts (only ones with ground truth)
    results: dict[str, list[dict]] = {mode: [] for mode in MODES}

    for q in queries:
        query = q["query"]
        relevant_sources = q.get("relevant_sources", [])
        must_contain = q.get("must_contain", [])

        print(f"Query: {query!r}")
        for mode in MODES:
            chunks = retrieve(query, top_k=args.top_k, collection=collection, mode=mode)
            score = _score(chunks, relevant_sources, must_contain)
            sources_seen = ", ".join(dict.fromkeys(c.source for c in chunks)) or "(none)"

            if score["has_ground_truth"]:
                bits = []
                if score["source_hit"] is not None:
                    bits.append(f"source_hit={score['source_hit']}")
                if score["term_hit"] is not None:
                    bits.append(f"term_hit={score['term_hit']}")
                if score["mrr"] is not None:
                    bits.append(f"mrr={score['mrr']:.3f}")
                print(f"  [{mode:6s}] {'  '.join(bits)}  sources={sources_seen}")
                results[mode].append(score)
            else:
                print(f"  [{mode:6s}] (no ground truth -- thematic query)  sources={sources_seen}")
        print()

    print("=" * 60)
    print(f"Aggregate over {len(results['vector'])} ground-truth quer(y/ies):\n")
    for mode in MODES:
        scored = results[mode]
        if not scored:
            print(f"  [{mode:6s}] no ground-truth queries scored")
            continue
        source_hits = [s["source_hit"] for s in scored if s["source_hit"] is not None]
        term_hits = [s["term_hit"] for s in scored if s["term_hit"] is not None]
        mrrs = [s["mrr"] for s in scored if s["mrr"] is not None]

        def _rate(vals: list[bool]) -> str:
            return f"{sum(vals) / len(vals):.1%}" if vals else "n/a"

        avg_mrr = f"{sum(mrrs) / len(mrrs):.3f}" if mrrs else "n/a"
        print(
            f"  [{mode:6s}] source-hit@{args.top_k}={_rate(source_hits)}  "
            f"term-hit@{args.top_k}={_rate(term_hits)}  MRR={avg_mrr}"
        )


if __name__ == "__main__":
    main()
