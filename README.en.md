<div align="center">

# BiXiaScribe

**Retrieve wuxia source text with RAG, hand it to a multi-agent pipeline that writes a structured wuxia RPG script.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/passpier/BiXiaScribe?style=social)](https://github.com/passpier/BiXiaScribe/stargazers)

[繁體中文](./README.md) | [English](./README.en.md)

</div>

---

## What is this?

BiXiaScribe is a wuxia (武俠) RPG script generator. Give it one line of a story requirement
(e.g. "a Shaolin disciple leaves the temple to investigate a massacre"), and it retrieves
relevant passages from your own corpus of wuxia novels, then hands them to three specialized
LLM agents (writer → dialogue → proofreader) that produce a structured script JSON — NPCs,
events, branching choices, triggers — usable as source material for downstream game production
(e.g. RPG Maker).

Compared to just prompting ChatGPT directly for a script, BiXiaScribe differs in:

- **RAG retrieval over real source text, not just the model's imagination of "wuxia voice"** —
  it indexes wuxia novels you collect yourself and feeds retrieved passages into the dialogue
  agent's prompt, so wording and move names stay closer to the source material.
- **A Chinese-aware chunker** — a hand-written recursive chunker that measures length in
  characters and prefers splitting at paragraph/punctuation boundaries, instead of reusing
  token-splitting logic built for English NLP.
- **Structured output with automated cross-reference validation** — the three agents' output
  is checked in Python (not by asking the LLM again) for consistency of fields like `npc_id`
  and `next_event_id` — it's not just trusted because the proofreader agent says it's fine.
- **Local-first, runnable end-to-end at zero cost** — the default embedding backend is the
  local `bge-m3` model (offline, free, no API key); the LLM also has a `fake` mode so tests
  never make a real API call.

### Table of Contents

- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Design Notes](#design-notes)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Installation

### Prerequisites

- Python ≥ 3.12 (`crewai`/Stage 2 requires ≥ 3.10; this repo standardizes on 3.12)

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Only needed if you'll use the Gemini embedding backend or Stage 2 (CrewAI)
cp .env.example .env
```

### Verify installation

No API key, no network required — confirms your environment in under 10 seconds:

```bash
python tests/test_chunking.py
# Expected: a series of PASS lines for the chunking test cases, ending in OK
```

<details>
<summary>Common installation issues</summary>

- **A `chromadb`-related Rust panic** (e.g. `range start index ... out of range`):
  `crewai` hard-requires `chromadb~=1.1.0`, which `requirements.txt` already pins to match.
  If your `data/chroma/` was built under a newer chromadb, opening it will crash — delete
  `data/chroma/` and rebuild with `python scripts/build_index.py --reset`.
- **Chroma errors after switching `EMBED_BACKEND`**: `CorpusEmbeddingFunction.name()` encodes
  backend/model/dimension/task_type into the collection name, so you can't switch backends
  against an existing `data/chroma/` in place — rebuild with `--reset`.

</details>

---

## Usage Examples

### 1. Build an index (Stage 1)

Smoke test against the bundled sample corpus:

```bash
python scripts/build_index.py --corpus tests/sample_corpus.txt
```

For your own corpus, drop `.txt` files into `data/corpus/` (not assumed to be UTF-8 — it tries
utf-8 → gb18030 → big5 in order), then:

```bash
python scripts/build_index.py
# add --reset to wipe and rebuild the collection
```

Indexing is resumable: already-indexed chunk IDs are skipped and upserts happen per-batch, so
re-running after a crash or rate limit is safe.

### 2. Query the index

```bash
python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3
```

Example output:

```text
[1] distance=0.1823  source=笑傲江湖.txt
    ...獨孤九劍的精要在於「無招」，見招拆招，後發制人...

[2] distance=0.2456  source=笑傲江湖.txt
    ...風清揚傳授劍法之時，反覆強調破劍式、破刀式...
```

Prints each result's distance score, source filename, and a text preview, so you can eyeball
whether retrieval is semantically relevant. Defaults to `hybrid` mode (vector search fused
with BM25 keyword search via Reciprocal Rank Fusion) — more accurate than vector-only for
wuxia proper nouns like 獨孤九劍 / 六脈神劍. Add `--mode vector` to compare against the
vector-only path.

To compare retrieval quality across many queries instead of eyeballing one at a time:

```bash
python scripts/eval_retrieval.py
```

Runs the curated wuxia query set in `eval/retrieval_eval.jsonl` through both modes and prints
a source-hit@k / term-hit@k / MRR comparison table.

### 3. Generate a script (Stage 2)

Requires an existing index, plus `LLM_BACKEND=openrouter` + `OPENROUTER_API_KEY` (set in `.env`):

```bash
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

The resulting `script.json` looks roughly like this (full field definitions in
[`src/bixiascribe/schema.py`](./src/bixiascribe/schema.py)):

```json
{
  "title": "...",
  "premise": "...",
  "variables": [{ "id": "...", "name": "...", "initial": "..." }],
  "npcs": [{ "id": "...", "name": "...", "identity": "...", "personality": "...", "speech_style": "..." }],
  "events": [
    {
      "id": "...",
      "title": "...",
      "location": "...",
      "triggers": [...],
      "dialogue": [{ "npc_id": "...", "line": "...", "emotion": "..." }],
      "branches": [{ "id": "...", "choice_text": "...", "next_event_id": "..." }]
    }
  ]
}
```

Omit `--out` to print the JSON to stdout. After generation, cross-references like `npc_id` and
`next_event_id` are re-verified in Python via `schema.validate_references()` — not just trusted
to the proofreader agent's say-so. If problems are found, the proofreader agent gets a targeted
retry with the specific problems listed (up to twice) before the run is reported as failed,
instead of discarding the whole generation.

---

## Features

- ✅ **Stage 1: Chinese-aware RAG indexing** — txt → Chinese-aware recursive chunking →
  embedding (local `bge-m3` or Gemini API) → Chroma vector index, with resumable indexing.
- ✅ **Hybrid retrieval (vector + BM25)** — a hand-rolled, zero-dependency Chinese
  character-bigram BM25 index fused with vector search via Reciprocal Rank Fusion, improving
  retrieval accuracy for wuxia proper nouns; `scripts/eval_retrieval.py` gives a repeatable
  quality comparison between the two modes.
- ✅ **Stage 2: 3-agent script generation** — writer (event/branch skeleton) → dialogue
  (RAG-fed for wuxia voice) → proofreader (schema + cross-reference validation, with a targeted
  repair retry when problems are found), producing a structured script JSON.
- ✅ **Dual-backend switching for zero-cost development** — both embedding and LLM have an
  offline/free mode (`bge-m3`, `fake` LLM); unit tests never hit a real API.
- 📋 **Streamlit UI** (planned)

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Vector store | [Chroma](https://www.trychroma.com/) (embedded `PersistentClient`, local folder) |
| Embedding | `bge-m3` ([FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), local) / `gemini-embedding-001` (Google API) |
| Multi-agent framework | [CrewAI](https://www.crewai.com/) |
| LLM routing | [OpenRouter](https://openrouter.ai/) (via CrewAI's `LLM` + litellm's `openrouter/` prefix) |
| Validation | [pydantic](https://docs.pydantic.dev/) |

> **Why `bge-m3` by default instead of Gemini?** Local, offline, no API key, no rate limits —
> well suited to repeatedly rebuilding indexes during development. The Gemini backend is kept
> around for when cloud-grade embedding quality is needed.

> **Why route through OpenRouter instead of a provider SDK?** Switching models is just an env
> var change (`LLM_MODEL` / `LLM_MODEL_WRITER` etc.), not a code change or a new SDK integration.

Supported environments: Python ≥ 3.12 (`crewai` requires ≥ 3.10; this repo standardizes on 3.12).

---

## Design Notes

For readers also learning RAG/embeddings as they go:

- **Chroma in embedded mode** — `PersistentClient` writes straight to a local folder, no
  separate server or paid cloud service, zero cost and zero ops during development.
- **Gemini embedding dimension and distance metric** — `gemini-embedding-001` output vectors
  are truncated to 1536 dimensions and L2-normalized so Chroma can compare them with cosine
  distance — after normalization, Euclidean distance is mathematically equivalent to cosine
  distance, which is the standard recommended approach, not an arbitrary choice. Indexing uses
  the `RETRIEVAL_DOCUMENT` task_type and querying uses `RETRIEVAL_QUERY` — Gemini's embedding
  model encodes "this text is meant to be found" differently from "this text is a search
  query," so specifying them separately improves retrieval quality.
- **Why a hand-written chunker** — Chinese prose has no whitespace word boundaries, so
  token-splitting logic built for English NLP tools doesn't work well. `src/bixiascribe/chunking.py`
  measures length in characters instead, and prefers splitting at paragraph/punctuation
  boundaries; it's pure Python with no external dependencies, which makes it easier to reason
  about and debug.
- **Resumable indexing** — already-indexed chunk IDs are skipped, and writes happen as
  per-batch upserts, so if the process dies mid-run or hits an API rate limit, re-running the
  same command picks up where it left off instead of starting over.
- **Why a hand-rolled BM25 instead of `rank_bm25` + `jieba`** — Chinese word-segmentation
  libraries like jieba, without a custom dictionary, tend to split a proper noun like 獨孤九劍
  into 獨孤／九劍 or worse, defeating the point of adding keyword search in the first place.
  Tokenizing as Chinese **character bigrams** instead (獨孤九劍 → 獨孤／孤九／九劍) means a
  query tokenizes the same way and naturally matches the full proper noun, no dictionary
  needed. The two methods' scores are fused with **Reciprocal Rank Fusion** (rank-based, not a
  weighted average of raw scores) because cosine distance and BM25 scores live on incomparable
  numeric scales — RRF sidesteps that entirely.

---

## Contributing

Contributions of any kind are welcome — bug reports, suggestions, or code:

- 🐛 **Found a bug?** Open an issue via the [bug report template](https://github.com/passpier/BiXiaScribe/issues/new?template=bug_report.md).
- 💡 **Have a feature idea?** Use the [feature request template](https://github.com/passpier/BiXiaScribe/issues/new?template=feature_request.md), or start a [discussion](https://github.com/passpier/BiXiaScribe/discussions).
- 🔧 **Want to contribute code?** See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for local dev setup,
  test, and lint commands.

---

## License

This project's code is licensed under the [MIT License](./LICENSE).

> ⚠️ This license covers only the source code in this repo. The wuxia novel corpus under
> `data/corpus/`, and third-party model weights such as `bge-m3` / `gemini-embedding-001`,
> are not distributed with this repo — use them under their own respective licenses.

---

## Contact

- 💬 Questions & discussion: [GitHub Discussions](https://github.com/passpier/BiXiaScribe/discussions)
- 🐛 Bugs/features: [Issues](https://github.com/passpier/BiXiaScribe/issues)
- 👤 Maintainer: [@passpier](https://github.com/passpier)

> This is a solo side project — response times may be irregular, thanks for your patience 🙏

<div align="center">

⭐ If this project helped you, consider giving it a star!

</div>
