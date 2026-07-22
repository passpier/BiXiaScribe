# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BiXiaScribe is a 武俠 RPG script generator. This repo currently covers **stage 1 and stage 2**:
- Stage 1: a RAG indexing pipeline (`txt → Chinese-aware chunking → embedding → Chroma index`).
- Stage 2: a 3-agent CrewAI pipeline (編劇 → 對話 → 校對) that consumes Stage 1's retrieval to
  generate a structured 劇本 JSON (see `src/bixiascribe/schema.py`, `src/bixiascribe/crew/`).

Streamlit and any RPG Maker export/conversion are future stages — do not assume they exist yet.

## Setup

```bash
python3.12 -m venv .venv   # crewai (Stage 2) requires Python >=3.10; the repo targets 3.12
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # only needed if using the Gemini embedding backend or Stage 2 (OpenRouter)
```

There is no `pyproject.toml`/`setup.py` — the package is never `pip install`ed. `scripts/*.py` and
`tests/test_chunking.py` each manually prepend `src/` to `sys.path`. Any new script or test that imports
`bixiascribe` needs the same `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))` pattern.

## Commands

```bash
# Smoke test the full pipeline against the sample corpus
python scripts/build_index.py --corpus tests/sample_corpus.txt

# Real corpus: drop txt files into data/corpus/, then
python scripts/build_index.py
# --reset wipes and rebuilds the collection

# Query the index
python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3

# Unit tests (no API key needed, no external deps)
python tests/test_chunking.py
# or: pytest tests/

# Stage 2: generate a script (needs an existing index + OPENROUTER_API_KEY)
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

## Embedding backends

Controlled by `EMBED_BACKEND` in `.env`:
- `bge-m3` (default) — local, offline, no API key, no quota. Uses `FlagEmbedding`/torch.
- `gemini` — Google's `gemini-embedding-001` API, requires `GEMINI_API_KEY`, subject to free-tier
  rate limits. `src/bixiascribe/embedding.py` has built-in 429 backoff/retry.

`CorpusEmbeddingFunction.name()` encodes backend+model+dim+task_type. **Don't switch `EMBED_BACKEND`
against an existing `data/chroma/` collection without `--reset`** — Chroma will error on a mismatch.

## Stage 2: CrewAI script generation

Three agents run as a sequential `Crew` (`src/bixiascribe/crew/pipeline.py`): 編劇 (writer, produces an
event/branch skeleton per `schema.Script` with `dialogue=[]`) → 對話 (dialogue, fills in NPC lines using
the `WuxiaRetrievalTool`, which wraps `retrieval.retrieve()` from Stage 1) → 校對 (proofreader, checks
schema + npc_id/next_event_id cross-references — re-verified in Python via `schema.validate_references()`
after the crew finishes, not just trusted to the LLM).

Model calls go through OpenRouter (via crewai's `LLM` + litellm's `openrouter/` model prefix), never a
provider SDK directly, so switching models is an env var change. Controlled by `LLM_BACKEND` in `.env`,
mirroring the `EMBED_BACKEND` fake-vs-real split:
- `openrouter` (default) — real generation. Needs `OPENROUTER_API_KEY`. `LLM_MODEL` sets the default
  model for all three agents; `LLM_MODEL_WRITER` / `LLM_MODEL_DIALOGUE` / `LLM_MODEL_PROOF` override
  per-agent.
- `fake` — offline, deterministic canned responses (`src/bixiascribe/llm.py::FakeLLM`), no key/network/
  cost. This is what `tests/test_crew_pipeline.py` uses.

**`crewai` hard-pins `chromadb~=1.1.0`** (not optional/behind an extra) — `requirements.txt` pins
`chromadb` to match. If `data/chroma/` was built under a newer chromadb, opening it will crash with a
Rust panic (`range start index ... out of range`); rebuild with `python scripts/build_index.py --reset`
after fixing the chroma directory (delete `data/chroma/` first if `--reset` itself can't open the client).

## Linting

```bash
pip install -r requirements-dev.txt
ruff check .
```

## Gotchas

- Corpus text files are not assumed UTF-8: `indexer._read_text_any_encoding` tries utf-8 → gb18030 → big5
  in order (common for Chinese novel txt dumps).
- Indexing is resumable: already-indexed chunk IDs are skipped and upserts happen per-batch, so re-running
  after a crash/rate-limit is safe.
- Chunking measures length in **characters**, not tokens or words (Chinese prose is unspaced).
- `data/corpus/` and `data/chroma/` are gitignored and already populated locally — don't assume a clean
  checkout has them.
