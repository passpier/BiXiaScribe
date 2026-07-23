"""Hand-rolled, zero-dependency BM25 keyword index, used to complement vector
search for wuxia proper nouns (獨孤九劍, 六脈神劍, 降龍十八掌...) where pure vector
search under-performs exact/keyword matches -- see the 武俠 RPG 劇本 RAG 架構方案
design doc's "hybrid retrieval (BM25 + 向量)" recommendation.

Deliberately pure Python, no jieba/rank_bm25, in the same spirit as
chunking.py: Chinese prose has no whitespace tokenization, and word-segmenting
libraries like jieba would split a proper noun like 獨孤九劍 into 獨孤/九劍 (or
worse) without a custom dictionary, which defeats the point of adding keyword
search for exact proper-noun matches in the first place.

Instead we tokenize CJK text into overlapping **character bigrams**
(獨孤九劍 -> 獨孤/孤九/九劍) -- a query for the same phrase produces the same
bigrams, so it matches the full proper noun without needing to know it's a
proper noun at all. Non-CJK runs (latin/digits) are tokenized as ordinary
lowercased words, since bigram-ing e.g. an English NPC name would be lossy.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# One contiguous run of "CJK-ish" characters (CJK Unified Ideographs + the
# common full-width punctuation-adjacent ranges) vs. one contiguous run of
# everything else (we only care about carving out latin/digit words from the
# rest; punctuation/whitespace tokens are discarded either way).
_CJK_RUN = re.compile(r"[㐀-鿿豈-﫿]+")
_WORD_RUN = re.compile(r"[A-Za-z0-9]+")

# Standard BM25(Okapi) constants.
_K1 = 1.5
_B = 0.75


def char_ngram_tokens(text: str) -> list[str]:
    """Tokenize `text` into overlapping character bigrams for CJK runs, plus
    lowercased alphanumeric words for any latin/digit runs. A single-character
    CJK run (too short to bigram) is kept as a unigram token so short queries
    still match something."""
    tokens: list[str] = []

    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    tokens.extend(w.lower() for w in _WORD_RUN.findall(text))

    return tokens


class BM25Index:
    """In-memory BM25(Okapi) inverted index over a fixed list of documents.

    Built once from the full set of indexed chunks (see retrieval.py's lazy
    singleton) and queried many times -- construction tokenizes every
    document, so it's the expensive part; `search()` itself only touches the
    postings lists of the query's own tokens.
    """

    def __init__(self, docs: list[str]):
        self.n_docs = len(docs)
        self.doc_lengths: list[int] = []
        # token -> list of (doc_idx, term_frequency), i.e. the postings list.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        total_len = 0
        for doc_idx, doc in enumerate(docs):
            tokens = char_ngram_tokens(doc)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for token, tf in Counter(tokens).items():
                self.postings[token].append((doc_idx, tf))

        self.avgdl = (total_len / self.n_docs) if self.n_docs else 0.0
        self.postings = dict(self.postings)

    def _idf(self, token: str) -> float:
        n_containing = len(self.postings.get(token, ()))
        if n_containing == 0:
            return 0.0
        # BM25's standard idf variant, floored at 0 isn't applied here on
        # purpose (Okapi's classic form can go slightly negative for terms in
        # the majority of docs) -- with bigram tokens over a mixed 25-book
        # corpus this is a non-issue in practice, and clamping would just
        # obscure a genuinely uninformative term rather than help ranking.
        return math.log(
            (self.n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1
        )

    def search(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Return up to `top_n` (doc_idx, bm25_score) pairs, highest score
        first, for documents sharing at least one token with the query."""
        if self.n_docs == 0:
            return []

        query_tokens = set(char_ngram_tokens(query))
        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self._idf(token)
            if idf <= 0:
                continue
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + _K1 * (1 - _B + _B * doc_len / self.avgdl)
                scores[doc_idx] += idf * tf * (_K1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_n]
