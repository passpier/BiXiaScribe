# BiXiaScribe — Milestones & Stage Checklist

This is a living document. It compares the target architecture described in
the *武俠 RPG 劇本 RAG 架構方案 (2026)* design doc (Obsidian vault) against
what's actually implemented in this repo, so it's easy to answer "how far
along are we, really?" at a glance. Update it whenever a checklist item
changes state — don't let it drift from reality.

**Status legend:** ✅ done and verified · ⚠️ partially done / done differently
than planned · ❌ not started.

## Headline

Stage 1 (indexing) and Stage 2 (3-agent script generation) are **code-complete
and unit-tested** (`pytest tests/` → 7 passed using the offline `fake` LLM
backend). But the pipeline has **never been run against a real LLM** — the
`.env` had no `LLM_BACKEND` / `OPENROUTER_API_KEY` set, so
`scripts/generate_script.py` failed before making a single real model call.
That's the real reason output quality feels unknown: there isn't yet a
generated script to judge. The corpus on disk now has all 14 金庸 novels in
`data/corpus/`, though it's unverified whether all 14 have actually been
upserted into the Chroma collection (vs. just the original 《俠客行》) —
run `scripts/build_index.py` (resumable, safe to re-run) to make sure. A
`scripts/prepare_webnovel.py` script has also been added to optionally pull
a wuxia-flavored subset of the `wdndev/webnovel-chinese` HuggingFace dataset
for extra 語感, but it hasn't been run for real yet either (see the Corpus
row below). In short: the pipeline plumbing works: what it produces with a
real model, against real corpus breadth, hasn't been observed yet.

## Architecture vs. reality, by area

| Area | Design doc target | Current state | Status |
|---|---|---|---|
| Vector store | Chroma (prototype) → Qdrant Cloud (remote) | Chroma embedded (`data/chroma/`), local only | ✅ (for prototype stage) |
| Retrieval framework | LlamaIndex + hybrid retrieval (BM25 keyword + vector) | Hand-written chunker (`chunking.py`) + Chroma vector-only query, no LlamaIndex, no BM25 | ⚠️ |
| Embedding | Gemini free tier or BGE-M3 | Both implemented, `bge-m3` default (local, offline) | ✅ |
| Multi-agent orchestration | CrewAI: writer → dialogue → proofreader | Implemented in `src/bixiascribe/crew/` (`agents.py`, `tasks.py`, `pipeline.py`), sequential `Crew`, tests pass | ✅ |
| Model routing | OpenRouter, swap model via env var | Wired via `llm.py::build_llm` (litellm `openrouter/` prefix) — **never exercised against a real API call** | ❌ (untested) |
| Structured output + validation | Custom JSON schema + cross-reference check | `schema.py` (pydantic) + `validate_references()`, re-checked in Python after crew finishes, not just LLM self-report | ✅ |
| Corpus | 14 novels (~金庸 full set) + wuxia-flavored subset of `wdndev/webnovel-chinese` (HF dataset, for 語感) | 14 金庸 novels on disk in `data/corpus/`; `scripts/prepare_webnovel.py` added to pull a filtered webnovel subset (no genre label on the dataset, so filtering is title-whitelist/keyword-based) — not yet run against a real index | ⚠️ |
| Frontend | Streamlit prototype UI | Not started | ❌ |
| Compute host | Oracle Cloud Always Free ARM VM | Local dev machine only | ❌ (not needed yet) |
| RPG Maker export | JSON → RPG Maker event converter | Not started (explicitly a later stage per CLAUDE.md) | ❌ |

## Stage-by-stage checklist

### Stage 1 — RAG indexing pipeline
- [x] Chinese-aware chunking (character-length based, paragraph/punctuation-preferring) — `chunking.py`
- [x] Dual embedding backend (`bge-m3` local / `gemini` API) — `embedding.py`
- [x] Resumable indexing (skip already-indexed chunk IDs, per-batch upsert) — `indexer.py`
- [x] Chroma persistent index + query helper — `indexer.py`, `retrieval.py`
- [ ] Confirm all 14 金庸 novels (already on disk in `data/corpus/`) are actually upserted into the Chroma collection, not just 《俠客行》
- [ ] Run `scripts/prepare_webnovel.py` for real against `wdndev/webnovel-chinese` to pull a wuxia-flavored subset (no genre label on the dataset — filtering is title-whitelist/keyword based, see the script's `--list-titles` step) and index it alongside 金庸
- [ ] Hybrid retrieval (BM25 + vector) for wuxia proper nouns (招式/門派 names)
- [ ] Retrieval quality evaluation beyond eyeballing `scripts/test_retrieval.py` output

### Stage 2 — CrewAI script generation
- [x] `Script` schema (pydantic) — `schema.py`
- [x] Three agents (編劇/writer, 對話/dialogue, 校對/proofreader) — `crew/agents.py`
- [x] RAG retrieval tool for the dialogue agent — `crew/tools.py`
- [x] Python-side cross-reference validation (`npc_id`, `next_event_id`) after crew output — `pipeline.py`
- [x] Dual LLM backend (`fake` offline / `openrouter` real) — `llm.py`
- [x] Unit tests exercising the full 3-agent wiring offline — `tests/test_crew_pipeline.py`
- [ ] **Run the pipeline against a real model at least once and read the output** (this is the actual next unblock — see Quickstart below)
- [ ] Per-agent model tuning (`LLM_MODEL_WRITER` / `_DIALOGUE` / `_PROOF`) once a baseline model's output has been seen
- [ ] Any quality feedback loop (e.g. re-running proofreader, human edit pass) beyond a single crew pass

### Stage 3 — Streamlit frontend
- [ ] Not started. Explicitly future scope per `CLAUDE.md` — do not assume it exists yet.

### Stage 4 — Deployment & RPG Maker export
- [ ] Not started. Both explicitly future scope per `CLAUDE.md`.

## Gap to "product-grade", in priority order

This is the concrete answer to "what's actually blocking quality":

1. **Run the pipeline against a real model and read the output.** Nothing
   below matters until there's an actual generated script to critique — right
   now the "low quality" concern has no artifact behind it.
2. **Expand the corpus** (the remaining 13 novels) — retrieval variety is
   capped by how much source text is indexed.
3. **Hybrid retrieval (BM25 + vector)** — the design doc calls this out
   specifically for wuxia proper nouns (e.g. 獨孤九劍, 六脈神劍), where pure
   vector search under-performs exact/keyword matches.
4. **A retrieval/generation quality evaluation method** — currently the only
   check is manually reading `scripts/test_retrieval.py` output; there's no
   repeatable way to compare before/after a change.
5. **Streamlit preview UI** — once script quality is trusted, this makes
   iteration much faster than reading raw JSON.

## Quickstart — the two things that were actually broken

1. **Use the venv's Python, not the system one.** `crewai` *is* installed —
   just in `.venv/`. Run scripts as `.venv/bin/python scripts/generate_script.py …`,
   or `source .venv/bin/activate` first. Running plain `python …` will hit
   `ModuleNotFoundError: No module named 'crewai'` even though it's installed.
2. **`.env` needs a Stage 2 (LLM) section.** It previously only had the
   Stage 1 embedding lines. It now defaults to `LLM_BACKEND=fake`, so
   `generate_script.py` runs fully offline (no API key, no cost) and produces
   a real `script.json` using the same deterministic logic the test suite
   exercises. To get a real model's output, edit `.env`: set
   `LLM_BACKEND=openrouter` and fill in `OPENROUTER_API_KEY` (get one at
   https://openrouter.ai/keys) — that's the only change needed, no code
   changes, per `CLAUDE.md`'s backend-switching pattern.

```bash
.venv/bin/python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

*Note: the checked-in `.env` also contains a live `GEMINI_API_KEY`. `.env` is
gitignored so this isn't currently exposed, but if it was ever committed by
mistake in the past, rotate that key.*
