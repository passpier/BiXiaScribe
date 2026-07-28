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

Stage 2 (3-agent script generation) is **code-complete, hardened,
unit-tested, and has now been run against a real model** —
`.venv/bin/python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案"
--out out/script-first-run.json` against `deepseek/deepseek-chat` (all three
roles) produced a schema-valid, reference-clean 3-event/2-NPC script in
~184s for 15,960 tokens (4 requests) — a small fraction of a cent. The
proofreader's repair loop fired once and fixed a dangling reference on the
first pass, exactly the safety net it was built for. The 對話 agent *did*
call `wuxia_corpus_search` (once, for the tea-house NPC's opening line) —
RAG grounding is confirmed working end-to-end, though a single tool call
across three events is modest; whether that's "worked as intended" or
"under-used" is worth watching across more runs before drawing a
conclusion. `pipeline.py::run_pipeline` no longer trusts only
`crew_output.pydantic` — `_coerce_script` also falls back to `json_dict`
and a schema-validating scan of the raw output (`schema.parse_script_json`),
and now reports which of the three it used via `RunReport.coerced_from`
(this run: `pydantic`, the highest-trust path — no salvage needed).
`crew.kickoff()` errors are caught and re-raised as `PipelineError`
(now carrying a partial `RunReport` via `.report`) instead of a raw
traceback. `WuxiaRetrievalTool` now tracks call/failure counts
(`crew/tools.py::get_stats()`) so the dialogue agent's tool usage is
visible in a `RunReport` instead of only inferable from verbose log
scrollback, and degrades to a message instead of raising on any retrieval
failure. `scripts/generate_script.py --preflight-only` checks
`LLM_BACKEND`/API key/index presence before spending a token, and every
real run now prints a report (models, elapsed time, token usage, repair
attempts, retrieval call count) to stderr. Per-agent model guidance is
documented in `.env.example`, though all three roles still share
`LLM_MODEL` (`deepseek/deepseek-chat`) — no per-role split has been chosen
yet.

## Architecture vs. reality, by area

| Area | Design doc target | Current state | Status |
|---|---|---|---|
| Vector store | Chroma (prototype) → Qdrant Cloud (remote) | Chroma embedded (`data/chroma/`), local only | ✅ (for prototype stage) |
| Retrieval framework | LlamaIndex + hybrid retrieval (BM25 keyword + vector) | Hand-written chunker (`chunking.py`) + hand-rolled BM25 (`lexical.py`) fused with Chroma vector search via RRF (`retrieval.py`), no LlamaIndex | ⚠️ (hybrid retrieval done; still no LlamaIndex — deliberate, see `retrieval.py`'s module docstring) |
| Embedding | Gemini free tier or BGE-M3 | Both implemented, `bge-m3` default (local, offline) | ✅ |
| Multi-agent orchestration | CrewAI: writer → dialogue → proofreader | Implemented in `src/bixiascribe/crew/` (`agents.py`, `tasks.py`, `pipeline.py`), sequential `Crew`, tests pass | ✅ |
| Model routing | OpenRouter, swap model via env var | Wired via `llm.py::build_llm` (litellm `openrouter/` prefix) — exercised end-to-end against `deepseek/deepseek-chat` on OpenRouter | ✅ |
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
- [x] Unit tests exercising the full 3-agent wiring offline — `tests/test_crew_pipeline.py`,
      `tests/test_crew_tools.py`
- [x] **Run the pipeline against a real model at least once and read the output** — ran against
      `deepseek/deepseek-chat` (all three roles) via `scripts/generate_script.py`; produced a
      schema-valid, reference-clean script (see Headline above and `out/script-first-run.json`,
      gitignored). Added a `RunReport` (`pipeline.py::run_pipeline_with_report`) and a
      `--preflight-only` flag (`scripts/generate_script.py`) so future runs surface token spend,
      repair attempts, and — critically — whether `wuxia_corpus_search` was actually called,
      without reading raw CrewAI verbose log.
- [x] Per-agent model tuning guidance (`LLM_MODEL_WRITER` / `_DIALOGUE` / `_PROOF`) — documented in
      `.env.example` with reasoning per role (writer: cheap/structured, dialogue: worth spending more
      on + **must support function calling** or `wuxia_corpus_search` silently never fires, proofreader:
      cheap/mechanical). `llm.py::ModelChoice` + `scripts/eval_generation.py` now let several splits be
      A/B'd in one process (see below) instead of one env-var edit + restart per model tried.
- [x] **A/B harness for per-agent model splits** — `llm.py::ModelChoice` is threaded explicitly through
      `build_llm()` → `crew/agents.py`'s agent factories → `run_pipeline_with_report(..., models=...)`,
      so model ids no longer have to be frozen at import time via env vars. `scripts/eval_generation.py`
      runs a {variant} x {requirement} matrix (`eval/model_variants.json` x
      `eval/script_requirements.txt`), logs one row per run (`RunReport.to_dict()` +
      `crew/metrics.py::script_metrics()`) to `out/generation_runs.jsonl`, and prints a per-variant
      aggregate table — the Stage 2 counterpart to `scripts/eval_retrieval.py`. One real run
      (2026-07-28, `baseline` = all `deepseek/deepseek-chat` vs. `prose-split` = dialogue on
      `qwen/qwen3-235b-a22b`, 2 requirements each):
      - `prose-split`'s dialogue read noticeably more 武俠 (action beats in parentheses, richer
        vocabulary like 穿花拂柳步) than `baseline`'s.
      - But `prose-split` had **zero retrieval calls in both runs** (vs. 1/2 for `baseline`), despite
        `qwen3-235b-a22b` supporting tool-calling per OpenRouter's `/models` metadata — confirming
        "supports function calling" and "reliably calls the tool in a CrewAI ReAct loop" are different
        guarantees, and `retrieval_calls` needs checking per model rather than assumed from the
        capability flag.
      - `cheap-ends` (qwen3-30b-a3b / deepseek-chat-v3.1 / qwen3-30b-a3b) is defined in
        `eval/model_variants.json` but not yet run.
      Net: one data point isn't enough to *pick* a final split yet — see the updated gap-list item
      below — but the harness itself is done, and the tradeoff it surfaces (prose quality vs. RAG
      grounding) is now a concrete number instead of a guess.
- [x] Quality feedback loop beyond a single crew pass — `pipeline.py::run_pipeline` now: (1) falls back
      through `crew_output.pydantic` → `json_dict` → a schema-validating scan of `raw`
      (`schema.parse_script_json`) instead of discarding the whole run when CrewAI's own coercion fails
      on prose-wrapped JSON; (2) re-runs just the proofread task (not the whole crew) up to twice when
      `validate_references()` finds dangling `npc_id`/`next_event_id` references, keeping whichever
      version has fewer problems; (3) wraps `crew.kickoff()` so provider errors raise `PipelineError`
      with context instead of a raw traceback. `WuxiaRetrievalTool` (previously untested — the `fake`
      backend never calls tools) now degrades to a message on any retrieval failure instead of raising.

### Stage 3 — Streamlit frontend
- [ ] Not started. Explicitly future scope per `CLAUDE.md` — do not assume it exists yet.

### Stage 4 — Deployment & RPG Maker export
- [ ] Not started. Both explicitly future scope per `CLAUDE.md`.

## Gap to "product-grade", in priority order

This is the concrete answer to "what's actually blocking quality", now that
a baseline real-model run exists (see Headline above):

1. **Pick a concrete per-agent model split.** The A/B harness (`scripts/eval_generation.py`,
   see Headline above) now exists and one real 2-variant x 2-requirement matrix has run, but
   that's still not enough runs to settle on a final split -- `prose-split`'s stronger prose
   came with zero RAG grounding in both its runs, which needs to be understood (is it the
   model, or does its tool-calling need a different prompt/task wording?) before picking a
   default. Next: run `cheap-ends` too, add a couple more requirements to
   `eval/script_requirements.txt`, and re-run `--repeat 2-3` per variant so the
   `retrieval_calls` numbers aren't each based on just 2 samples.
2. **Understand *why* `prose-split` never calls `wuxia_corpus_search`.** Not just "watch more
   runs" anymore -- this is now a specific, reproduced behavior (2/2 runs) for one model despite
   it supporting tool-calling per OpenRouter. Worth checking whether it's specific to
   `qwen/qwen3-235b-a22b`, or whether other tool-capable non-`deepseek` models show the same
   pattern, before concluding "dialogue models need RAG-specific prompting" vs. "pick a
   different model."
3. **Streamlit preview UI** — once script quality is trusted across more than
   one run, this makes iteration much faster than reading raw JSON. `out/eval/*.json` from the
   A/B harness is exactly the kind of output this would make easier to read.

Stage 1 is no longer on this list — corpus breadth (14 金庸 + capped webnovel),
hybrid retrieval, and a repeatable retrieval eval are all done (see the Stage 1
checklist above).

## Quickstart

1. **Use the venv's Python, not the system one.** `crewai` *is* installed —
   just in `.venv/`. Run scripts as `.venv/bin/python scripts/generate_script.py …`,
   or `source .venv/bin/activate` first. Running plain `python …` will hit
   `ModuleNotFoundError: No module named 'crewai'` even though it's installed.
2. **`.env` needs `LLM_BACKEND=openrouter` and `OPENROUTER_API_KEY` for real
   generation.** Set `LLM_BACKEND=fake` instead to run fully offline (no API
   key, no cost) using the same deterministic logic the test suite exercises
   — useful for wiring checks, not for judging output quality. Switching is
   an env var only, no code change, per `CLAUDE.md`'s backend-switching
   pattern.
3. **Run `--preflight-only` first** to confirm the backend/key/index are all
   in place before spending a token:
   ```bash
   .venv/bin/python scripts/generate_script.py --requirement "測試" --preflight-only
   ```
4. **Then run for real** — a run report (models, elapsed time, token usage,
   repair attempts, retrieval call count) prints to stderr afterward:
   ```bash
   .venv/bin/python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
   ```

*Note: the checked-in `.env` also contains a live `GEMINI_API_KEY`. `.env` is
gitignored so this isn't currently exposed, but if it was ever committed by
mistake in the past, rotate that key.*
