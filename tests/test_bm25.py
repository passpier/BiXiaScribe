"""Pure unit tests for lexical.char_ngram_tokens / BM25Index and the RRF
fusion logic in retrieval.py — no API key, no network, no Chroma."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.lexical import BM25Index, char_ngram_tokens  # noqa: E402


def test_cjk_run_tokenizes_to_overlapping_bigrams():
    tokens = char_ngram_tokens("獨孤九劍")
    assert tokens == ["獨孤", "孤九", "九劍"]


def test_single_cjk_char_becomes_unigram():
    tokens = char_ngram_tokens("劍")
    assert tokens == ["劍"]


def test_latin_digit_runs_tokenize_as_lowercased_words():
    tokens = char_ngram_tokens("Hello 獨孤 world123")
    assert "hello" in tokens
    assert "world123" in tokens
    assert "獨孤" in tokens


def test_bm25_ranks_exact_proper_noun_match_first():
    docs = [
        "風清揚傳授劍法之時，反覆強調破劍式、破刀式，臨敵時見招拆招。",
        "獨孤九劍的精要在於「無招」，見招拆招，後發制人，乃劍魔獨孤求敗生平絕學。",
        "丐幫降龍十八掌威猛剛陽，乃武林中屈指可數的絕頂武功之一。",
    ]
    index = BM25Index(docs)
    results = index.search("獨孤九劍的劍法精要", top_n=3)

    assert results, "expected at least one match"
    top_doc_idx, _score = results[0]
    assert top_doc_idx == 1, "the chunk containing the exact proper noun should rank first"


def test_bm25_no_match_returns_empty():
    index = BM25Index(["江湖夜雨十年燈", "西風獨自涼"])
    assert index.search("完全不相干的查詢字串", top_n=5) == []


def test_bm25_empty_corpus_is_safe():
    index = BM25Index([])
    assert index.search("任何查詢", top_n=5) == []


def test_rrf_fusion_favors_docs_ranked_highly_by_both_methods():
    # Hand-built example mirroring retrieval.py's _hybrid_search fusion:
    # score(doc) = sum over methods ranking it of 1/(k + rank).
    k = 60
    vector_ranked = ["a", "b", "c"]  # a=1st, b=2nd, c=3rd by vector distance
    bm25_ranked = ["c", "a", "b"]  # c=1st, a=2nd, b=3rd by BM25 score

    scores = {}
    for rank, doc in enumerate(vector_ranked, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    for rank, doc in enumerate(bm25_ranked, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores, key=lambda d: scores[d], reverse=True)
    # "a" is 1st+2nd (avg rank 1.5), "c" is 3rd+1st (avg rank 2), "b" is
    # 2nd+3rd (avg rank 2.5) -- "a" should win, "b" should come last.
    assert ranked[0] == "a"
    assert ranked[-1] == "b"


if __name__ == "__main__":
    # Allow running as a plain script without pytest.
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All BM25 tests passed.")
