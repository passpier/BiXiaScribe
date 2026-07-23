# BiXiaScribe — Milestones & Stage Checklist

This is a living document. It compares the target architecture described in
the *武俠 RPG 劇本 RAG 架構方案 (2026)* design doc (Obsidian vault) against
what's actually implemented in this repo, so it's easy to answer "how far
along are we, really?" at a glance. Update it whenever a checklist item
changes state — don't let it drift from reality.

**Status legend:** ✅ done and verified · ⚠️ partially done / done differently
than planned · ❌ not started.

## Headline

**Stage 1 (indexing) is now code-complete, unit-tested, and actually run
end-to-end against its intended corpus breadth** — all 14 金庸 novels plus 11
capped webnovel books (from `scripts/prepare_webnovel.py`) are indexed into
`data/chroma/` via `scripts/build_index.py --reset` (the stray
`sample_corpus.txt` chunk left over from an earlier smoke test is gone too).
Retrieval is now hybrid by default (`RETRIEVAL_MODE=hybrid`): a hand-rolled,
zero-dependency BM25 keyword index (`lexical.py`) fused with Chroma vector
search via Reciprocal Rank Fusion (`retrieval.py`), which measurably helps
wuxia proper nouns (獨孤九劍, 六脈神劍) that pure vector search under-matches —
see `scripts/eval_retrieval.py` for a repeatable side-by-side comparison
against vector-only, replacing what used to be pure eyeballing of
`scripts/test_retrieval.py` output.

Stage 2 (3-agent script generation) is still **code-complete and
unit-tested** (`pytest tests/` → all passing using the offline `fake` LLM
backend) but has **never been run against a real LLM** — the `.env` had no
`LLM_BACKEND` / `OPENROUTER_API_KEY` set, so `scripts/generate_script.py`
failed before making a single real model call. That's the real reason
Stage 2 output quality feels unknown: there isn't yet a generated script to
judge. With Stage 1's corpus and retrieval now solid, **running Stage 2
against a real model is the next actual unblock** — see Quickstart below.

## Architecture vs. reality, by area

| Area | Design doc target | Current state | Status |
|---|---|---|---|
| Vector store | Chroma (prototype) → Qdrant Cloud (remote) | Chroma embedded (`data/chroma/`), local only | ✅ (for prototype stage) |
| Retrieval framework | LlamaIndex + hybrid retrieval (BM25 keyword + vector) | Hand-written chunker (`chunking.py`) + hand-rolled BM25 (`lexical.py`) fused with Chroma vector search via RRF (`retrieval.py`), no LlamaIndex | ⚠️ (hybrid retrieval done; still no LlamaIndex — deliberate, see `retrieval.py`'s module docstring) |
| Embedding | Gemini free tier or BGE-M3 | Both implemented, `bge-m3` default (local, offline) | ✅ |
| Multi-agent orchestration | CrewAI: writer → dialogue → proofreader | Implemented in `src/bixiascribe/crew/` (`agents.py`, `tasks.py`, `pipeline.py`), sequential `Crew`, tests pass | ✅ |
| Model routing | OpenRouter, swap model via env var | Wired via `llm.py::build_llm` (litellm `openrouter/` prefix) — **never exercised against a real API call** | ❌ (untested) |
| Structured output + validation | Custom JSON schema + cross-reference check | `schema.py` (pydantic) + `validate_references()`, re-checked in Python after crew finishes, not just LLM self-report | ✅ |
| Corpus | 14 novels (~金庸 full set) + wuxia-flavored subset of `wdndev/webnovel-chinese` (HF dataset, for 語感) | All 14 金庸 novels + 11 webnovel books (from `scripts/prepare_webnovel.py`) indexed into `data/chroma/` — webnovel capped per-file via `WEBNOVEL_MAX_CHARS` to keep 武俠語感 dominant | ✅ |
| Frontend | Streamlit prototype UI | Not started | ❌ |
| Compute host | Oracle Cloud Always Free ARM VM | Local dev machine only | ❌ (not needed yet) |
| RPG Maker export | JSON → RPG Maker event converter | Not started (explicitly a later stage per CLAUDE.md) | ❌ |

## Stage-by-stage checklist

### Stage 1 — RAG indexing pipeline
- [x] Chinese-aware chunking (character-length based, paragraph/punctuation-preferring) — `chunking.py`
- [x] Dual embedding backend (`bge-m3` local / `gemini` API) — `embedding.py`
- [x] Resumable indexing (skip already-indexed chunk IDs, per-batch upsert) — `indexer.py`
- [x] Chroma persistent index + query helper — `indexer.py`, `retrieval.py`
- [x] Confirm all 14 金庸 novels (already on disk in `data/corpus/`) are actually upserted into the Chroma collection, not just 《俠客行》 — ran `scripts/build_index.py --reset`; the stray `sample_corpus.txt` chunk that had leaked into the collection from an earlier smoke test is also gone now that `--reset` wiped and rebuilt it
- [x] Run `scripts/prepare_webnovel.py` for real against `wdndev/webnovel-chinese` to pull a wuxia-flavored subset (no genre label on the dataset — filtering is title-whitelist/keyword based, see the script's `--list-titles` step) and index it alongside 金庸 — 11 webnovel books already on disk; `indexer.py` now caps each `webnovel-*.txt` file at `WEBNOVEL_MAX_CHARS` (default 1,000,000 chars, env-overridable) before chunking, since the webnovel set is ~6x the 金庸 corpus by size and mostly 玄幻/仙俠 — uncapped it would dilute 武俠語感 rather than add to it. 金庸 files are always indexed in full.
- [x] Hybrid retrieval (BM25 + vector) for wuxia proper nouns (招式/門派 names) — `lexical.py` (hand-rolled, zero-dependency character-bigram BM25, chosen over `rank_bm25`+`jieba` so proper nouns like 獨孤九劍 aren't mis-segmented by a dictionary-less tokenizer) fused with Chroma vector search via Reciprocal Rank Fusion in `retrieval.py`. Switch via `RETRIEVAL_MODE` (`hybrid` default / `vector`); `scripts/test_retrieval.py --mode` compares the two on one query.
- [x] Retrieval quality evaluation beyond eyeballing `scripts/test_retrieval.py` output — `scripts/eval_retrieval.py` + `eval/retrieval_eval.jsonl` (14 curated wuxia queries, ground truth verified against the actual decoded corpus text) reports source-hit@k / term-hit@k / MRR for `vector` vs `hybrid` side by side.

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
   now the "low quality" concern has no artifact behind it. Stage 1's corpus,
   hybrid retrieval, and eval harness are done, so this is now the actual
   next unblock — see Quickstart below.
2. **Per-agent model tuning** once a baseline model's output has been seen
   (`LLM_MODEL_WRITER` / `_DIALOGUE` / `_PROOF`).
3. **A generation quality feedback loop** — e.g. re-running the proofreader,
   a human edit pass — beyond a single crew pass. (Stage 1 already has its
   own repeatable retrieval eval: `scripts/eval_retrieval.py`.)
4. **Streamlit preview UI** — once script quality is trusted, this makes
   iteration much faster than reading raw JSON.

Stage 1 is no longer on this list — corpus breadth (14 金庸 + capped webnovel),
hybrid retrieval, and a repeatable retrieval eval are all done (see the Stage 1
checklist above).

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
