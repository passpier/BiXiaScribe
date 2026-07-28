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

# Optional: expand the corpus with a wuxia-flavored subset of the
# wdndev/webnovel-chinese HuggingFace dataset (streamed, no 39GB download).
# The dataset has no genre label, so this is a two-step, deliberately manual
# filter -- see scripts/prepare_webnovel.py's module docstring for the full
# workflow.
python scripts/prepare_webnovel.py --list-titles --max-scan 50000   # eyeball titles first
python scripts/prepare_webnovel.py --titles-file my_wuxia_titles.txt
python scripts/build_index.py   # do NOT pass --reset -- only new webnovel- chunks get embedded

# Query the index (--mode vector|hybrid, default from RETRIEVAL_MODE)
python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3

# Compare vector vs. hybrid retrieval quality across a curated query set
python scripts/eval_retrieval.py

# Unit tests (no API key needed, no external deps)
python tests/test_chunking.py
# or: pytest tests/

# Stage 2: generate a script (needs an existing index + OPENROUTER_API_KEY)
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

## Retrieval

`retrieval.retrieve()` supports two modes via `RETRIEVAL_MODE` in `.env`:
- `hybrid` (default) — fuses Chroma vector search with a hand-rolled, zero-dependency
  BM25 keyword index (`src/bixiascribe/lexical.py`) via Reciprocal Rank Fusion. Helps
  wuxia proper nouns (獨孤九劍, 六脈神劍) that pure vector search under-matches. The BM25
  index tokenizes CJK text into character bigrams (no jieba/dictionary needed) and is
  built lazily once per process from the documents already stored in Chroma, then cached.
- `vector` — the original vector-only behavior; pass `mode="vector"` to `retrieve()` or
  `--mode vector` to `scripts/test_retrieval.py` to compare against hybrid on the same query.

Compare retrieval quality across queries with `python scripts/eval_retrieval.py`, which runs
`eval/retrieval_eval.jsonl`'s curated wuxia queries through both modes and reports
source-hit@k / term-hit@k / MRR side by side.

`data/corpus/webnovel-*.txt` files (from `scripts/prepare_webnovel.py`) are capped per-file
at `WEBNOVEL_MAX_CHARS` (default 1,000,000 chars) when indexed, since webnovel is ~6x the
金庸 corpus by size and mostly 玄幻/仙俠 rather than 武俠 — uncapped it would dilute 武俠語感
in retrieval. `金庸-*.txt` files are always indexed in full. Set `WEBNOVEL_MAX_CHARS=0` to
disable the cap.

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

`run_pipeline()` doesn't just trust `crew_output.pydantic`: `pipeline.py::_coerce_script` falls back to
`crew_output.json_dict`, then to `schema.parse_script_json()` scanning `crew_output.raw` for the last
JSON object that actually validates as a `Script` — real models often wrap JSON in explanatory prose,
which trips up CrewAI's own coercion even though the JSON itself is fine. If `validate_references()`
still finds dangling `npc_id`/`next_event_id` references after that, the proofreader agent gets up to
`MAX_REPAIR_ATTEMPTS` (2) targeted repair passes — re-running just the proofread task via
`Task.execute_sync()`, not the whole crew — before `run_pipeline()` raises `PipelineError`.
`crew.kickoff()` itself is also wrapped, so provider errors (401/429/timeouts) surface as `PipelineError`
instead of a raw traceback.

Model calls go through OpenRouter (via crewai's `LLM` + litellm's `openrouter/` model prefix), never a
provider SDK directly, so switching models is an env var change. Controlled by `LLM_BACKEND` in `.env`,
mirroring the `EMBED_BACKEND` fake-vs-real split:
- `openrouter` (default) — real generation. Needs `OPENROUTER_API_KEY`. `LLM_MODEL` sets the default
  model for all three agents; `LLM_MODEL_WRITER` / `LLM_MODEL_DIALOGUE` / `LLM_MODEL_PROOF` override
  per-agent (see `.env.example` for per-role tuning guidance). **Whatever model backs the 對話 (dialogue)
  agent must support function calling/tool use** — otherwise it never calls `wuxia_corpus_search` and the
  RAG retrieval this pipeline is built around silently never fires; the pipeline still produces a valid
  script either way, just without corpus-grounded wording, so this is easy to miss.
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
