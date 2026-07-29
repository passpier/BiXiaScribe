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
documented in `.env.example`. As of the Phase C A/B matrix (2026-07-29, 10 samples/variant, see
Stage 2 checklist below), all three roles staying on shared `LLM_MODEL`
(`deepseek/deepseek-chat`) is a deliberate conclusion, not an unmade decision — it's the
fastest, cheapest, and structurally richest of the five splits tested, with no other variant
beating it on every axis at once.

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
| Frontend | Streamlit prototype UI | Read-only review UI shipped (`ui/app.py` + `src/bixiascribe/review.py`) — browse + side-by-side compare `out/eval/*.json`; generation-from-UI not started | ⚠️ |
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
      **Phase B (2026-07-28, `out/generation_runs_2.jsonl`, same requirement, 1 run each)** added two
      control variants specifically to pin down the zero-retrieval finding:
      - `dialogue-control-openai` (`openai/gpt-4o-mini`, OpenAI-format-native so OpenRouter's tools
        payload passes through with minimal translation): **4** retrieval calls
        (`江湖信任度`, `滅門案`, `鄰居`, `神秘人物`), 41,845 tokens, 188s. This establishes the
        tool-call-propensity ceiling under `tool_choice="auto"` is high when nothing about model
        format is fighting the pipeline — i.e. `prose-split`'s zeros aren't a wiring bug.
      - `dialogue-control-qwen` (`qwen/qwen3-30b-a3b`, the smaller cousin of `prose-split`'s
        `qwen3-235b-a22b`): **0** retrieval calls, 11,182 tokens, 176s — same as its much larger
        relative. This is the second qwen-family model to show zero tool calls, which shifts the
        working hypothesis from "that one specific large MoE doesn't call the tool" toward "the qwen
        family doesn't reliably choose this tool in a CrewAI ReAct loop" — though each variant is
        still only n=1, which Phase C (below) is what actually settles it.
      **Phase C (2026-07-29, `out/generation_runs_phase_c.jsonl`, all 5 requirements in
      `eval/script_requirements.txt`, `--repeat 2` = 10 samples/variant)** settled the split
      question with real sample sizes instead of n=1/n=2 anecdotes:

      | variant | success | elapsed_s | retrieval_calls avg | zero-call runs | repair_attempts | tokens avg | events avg | avg_line_chars |
      |---|---|---|---|---|---|---|---|---|
      | `baseline` (all deepseek-chat) | 10/10 | 126.4 | 2.10 | 4/10 | 0.80 | 16,492 | 4.0 | 26.6 |
      | `prose-split` (qwen3-235b-a22b dialogue) | 10/10 | 188.5 | 0.40 | 6/10 | 1.10 | 13,166 | 2.9 | 45.7 |
      | `dialogue-control-openai` (gpt-4o-mini dialogue) | 10/10 | 189.9 | 3.30 | 0/10 | 0.90 | 28,002 | 2.4 | 33.3 |
      | `dialogue-control-qwen` (qwen3-30b-a3b dialogue) | 10/10 | 239.5 | 0.00 | **10/10** | 1.20 | 10,851 | 2.5 | 23.7 |
      | `cheap-ends` (qwen3-30b-a3b writer+proof) | **0/10** | — | — | — | — | — | — | — |

      **The qwen-family hypothesis is confirmed but more nuanced than "never calls the tool":**
      `dialogue-control-qwen` (the smaller 30b model) is a clean **10/10 zero** — deterministic,
      not a fluke. But `prose-split` (the larger 235b model) came in at **6/10 zero, 4/10
      nonzero** — worse than `baseline`'s 4/10 zero, but not the "2-for-2 always-zero" pattern
      Phase B's n=1 sample suggested. Ranked by retrieval reliability:
      `dialogue-control-openai` (0% zero) > `baseline` (40% zero) > `prose-split` (60% zero) >
      `dialogue-control-qwen` (100% zero, never). The revised takeaway: propensity to call
      `wuxia_corpus_search` under `tool_choice="auto"` in a CrewAI ReAct loop degrades with
      "how OpenAI-native the model's tool-calling format is," and within the qwen family
      specifically, the smaller model is *worse* at it than the larger one, not equally bad —
      so "qwen family" is directionally real but the smaller `qwen3-30b-a3b` is a stronger,
      cleaner example of the failure mode than `qwen3-235b-a22b`.

      **`cheap-ends` is currently unusable, not just under-tested.** All 10/10 runs failed
      identically: `qwen/qwen3-30b-a3b` (used for both the writer and proofreader roles here)
      prepends explanatory prose before the JSON object in its structured-output response, e.g.
      `了。\n{"title":"残卷...` or `s.\n{"title":"Secret Cou...`. This breaks *inside* CrewAI's own
      `beta.chat.completions.parse()` call, before `pipeline.py::_coerce_script`'s
      pydantic→json_dict→raw_scan fallback chain ever gets a chance to salvage it — the raw-scan
      salvage only helps when CrewAI hands back a `CrewOutput` at all, not when the provider
      round-trip itself throws. This is a different, more fundamental incompatibility than the
      "dialogue agent doesn't call the RAG tool" issue above: `qwen3-30b-a3b` doesn't reliably
      honor `output_pydantic`'s strict JSON-only contract when used for the writer/proofreader
      roles, even though the pipeline's writer/proof prompts don't require tool use at all.

      **Reading 6 scripts by hand (`baseline`/`prose-split`/`dialogue-control-openai`, 2 reps
      each, same requirement) confirms the earlier prose read**, now with more evidence:
      `prose-split`'s dialogue consistently uses period vocabulary and imagery a plain LLM
      wouldn't reach for on its own — 「少林施主遠道而來，可是衝著這壇『風雪山神廟』老酒？」,
      「阿彌陀佛，青雲血光乃業風所召，施主此去需以戒為師，切忌以劍證道」, parenthetical action
      beats like （喉間滾出森冷笑聲）. `baseline` is competent but plainer/more expository
      ("據我所查，案發當晚有人見到幾名黑衣人從酒樓後門出入，行跡可疑"). `dialogue-control-openai`
      is the plainest of the three and occasionally reads slightly modern for the genre ("嘿，小子，
      江湖事非比尋常"). The prose-quality vs. RAG-grounding trade-off from Phase A/B holds up at
      n=10, and `prose-split`'s structural counts (events=2.9, lowest of the four working variants)
      are a second, smaller cost on top of the retrieval gap.

      **No single variant dominates on every axis, so the default (`LLM_MODEL` = all
      `deepseek/deepseek-chat`, i.e. what `baseline` already is) stays as-is** rather than
      forcing a pick: `baseline` is the fastest, cheapest, structurally richest, and has decent
      (if imperfect) retrieval grounding — a genuinely solid all-arounder, not just "the one we
      happened to test first." `dialogue-control-openai` is the retrieval-reliability ceiling but
      costs ~70% more tokens for prose that reads *plainer* than `prose-split`, not better than
      `baseline`. `prose-split` is worth keeping documented as an **opt-in for users who value
      武俠語感 over reliable corpus grounding** — e.g. for a final polish pass on dialogue after
      the structural skeleton is locked in — rather than the default. `dialogue-control-qwen` is
      strictly dominated (worse prose than `prose-split`, worse retrieval than everything) and
      `cheap-ends` needs a different writer/proof model before it's usable at all.
- [x] Quality feedback loop beyond a single crew pass — `pipeline.py::run_pipeline` now: (1) falls back
      through `crew_output.pydantic` → `json_dict` → a schema-validating scan of `raw`
      (`schema.parse_script_json`) instead of discarding the whole run when CrewAI's own coercion fails
      on prose-wrapped JSON; (2) re-runs just the proofread task (not the whole crew) up to twice when
      `validate_references()` finds dangling `npc_id`/`next_event_id` references, keeping whichever
      version has fewer problems; (3) wraps `crew.kickoff()` so provider errors raise `PipelineError`
      with context instead of a raw traceback. `WuxiaRetrievalTool` (previously untested — the `fake`
      backend never calls tools) now degrades to a message on any retrieval failure instead of raising.
- [x] **Concurrency bug found and fixed during eval matrix runs** — a real `eval_generation.py` run died
      mid-way; the log showed five interleaved "Fetching 30 files"/"Loading weights" blocks instead of
      one. Root cause: CrewAI's native tool-call executor runs several `wuxia_corpus_search` calls from
      one LLM turn concurrently in a thread pool, but every module-level lazy singleton behind
      `WuxiaRetrievalTool` was written assuming single-threaded first-call init. N threads all saw the
      singleton as unset and each built their own copy — for `embedding.py::_get_local_model` that meant
      N simultaneous multi-GB `BGEM3FlagModel` loads, which is what OOM-killed the run; for
      `retrieval.py::_get_bm25_index` it meant N redundant full-corpus BM25 rebuilds. Fixed with
      double-checked locking on both of those, plus three locks in `crew/tools.py`: `_collection_lock`
      (Chroma collection init), `_stats_lock` (`RetrievalStats` counters — a plain `+=` from multiple
      threads silently loses increments), and `_retrieval_lock` serializing the actual lookup work. The
      serialization is deliberately coarse — a retrieval query is one short sentence, so queuing costs
      next to nothing, versus the untested risk of N concurrent `encode()` calls sharing one loaded
      BGE-M3 model. Lock ordering is documented in `crew/tools.py`: `_retrieval_lock` is always acquired
      first, and `_collection_lock`/`_stats_lock` only ever nest inside it, never the reverse, so there's
      no cycle. Covered by a new regression test in `tests/test_crew_tools.py`; full suite is 34/34,
      ruff clean, and the re-run showed a single weights-loading block instead of five.
- [x] **Second crash found and fixed during the Phase C matrix** — a separate bug from the one
      above: `pipeline.py::run_pipeline_with_report`'s repair loop calls `_repair()`, which calls
      `task.execute_sync()` directly, *outside* the `try/except` that wraps `crew.kickoff()` and
      converts provider errors to `PipelineError`. When OpenRouter's `deepseek/deepseek-chat`
      routing went unstable mid-matrix (StreamLake rejecting `response_format=json_schema`,
      falling back to a rate-limited DeepInfra), a persistent failure during a repair pass
      propagated as a raw, uncaught `openai.BadRequestError` and killed the whole
      `eval_generation.py` process 11 rows into a 50-row run instead of just failing that one row
      — exactly the failure mode `PipelineError` exists to prevent, just on a code path that
      wasn't covered by it. Fixed by wrapping the `_repair()` call in the same `try/except
      Exception: continue` pattern already used when a repair attempt produces no valid script —
      a failed repair attempt is treated as "this attempt didn't help," letting the loop retry or
      fall through to the existing final `PipelineError` if `best_problems` is still non-empty.
      Full suite still 34/34 after the fix; the re-launched Phase C matrix ran all 50 rows to
      completion with no further crashes.

### Stage 3 — Streamlit frontend
- [x] Read-only review UI over `out/eval/*.json` — `ui/app.py` + `src/bixiascribe/review.py`
  (2026-07-29). Three modes: 單篇閱讀 (single-script), 並排比較 (side-by-side variant comparison,
  ordinal event alignment since event ids aren't stable across variants), 總覽表 (overview table).
- [x] Side-by-side variant comparison for the same requirement — the exact manual workflow Phase C's
  "hand-reading 6 scripts" section below used to do by opening JSON files one at a time.
- [x] `tests/test_review.py` (18 tests, no streamlit import — mechanically enforced) covering the
  filename/JSONL join, the run-only fallback for failed runs with no script file, and that metrics
  are recomputed from disk rather than trusted from a possibly-stale JSONL row.
- [ ] Triggering generation from the UI (requirement input → retrieve → generate → preview) — the
  full four-step flow from the 武俠 RPG 劇本 RAG 架構方案 doc. Not started; needs API key handling,
  long-running-request UX, and error states that the read-only viewer doesn't have to deal with.
- [ ] Editing/save-back from the UI.

### Stage 4 — Deployment & RPG Maker export
- [ ] Not started. Both explicitly future scope per `CLAUDE.md`.

## Gap to "product-grade", in priority order

This is the concrete answer to "what's actually blocking quality", now that
a baseline real-model run exists (see Headline above):

1. ~~Pick a concrete per-agent model split.~~ **Done as of Phase C (2026-07-29, see Headline
   above), at n=10/variant.** Verdict: keep the default (`LLM_MODEL` = all
   `deepseek/deepseek-chat`, i.e. `baseline`) — it's the fastest, cheapest, structurally
   richest, and has decent retrieval grounding (60% of runs call `wuxia_corpus_search`), and no
   other variant beats it on every axis at once. `prose-split` is worth documenting as an
   opt-in for a dialogue-only prose-polish pass (clearly the richest 武俠語感, confirmed by
   hand-reading 6 scripts) for users willing to trade away reliable RAG grounding (60% zero-call)
   and slightly thinner event structure for it. `cheap-ends` needs a different writer/proof
   model — `qwen3-30b-a3b` fails 10/10 on structured output (see below) — before it's usable.
2. ~~Test the qwen-family zero-retrieval hypothesis at scale.~~ **Done as of Phase C.** Refined,
   not simply confirmed: `dialogue-control-qwen` (`qwen3-30b-a3b`) is a clean, deterministic
   **10/10 zero** retrieval calls, but `prose-split` (`qwen3-235b-a22b`) came in at 6/10 zero —
   worse than `baseline`'s 4/10, but not "always zero" the way Phase B's n=1 sample suggested.
   Reliability ranks `dialogue-control-openai` (0% zero) > `baseline` (40%) > `prose-split` (60%)
   > `dialogue-control-qwen` (100%, never) — directionally a qwen-family weakness, but the
   *smaller* qwen model is the more deterministic offender, not the larger one.
3. **`cheap-ends` needs a different writer/proofreader model.** New finding from Phase C, not
   previously on this list: `qwen/qwen3-30b-a3b` fails **10/10** structured-output parses when
   used for the writer/proofreader roles (it prepends explanatory prose before the JSON object,
   which breaks CrewAI's own `beta.chat.completions.parse()` before this pipeline's
   pydantic→json_dict→raw_scan salvage chain ever runs) — a different failure mode from the
   dialogue-agent tool-calling gap above, and currently a hard blocker for that variant rather
   than a quality tradeoff.
4. ~~Streamlit preview UI.~~ **Done (first slice), 2026-07-29** — `ui/app.py` (read-only, three
   modes including side-by-side variant comparison) replaces hand-opening `out/eval/*.json`.
   Remaining gap: generation-from-UI (item 3 above's `cheap-ends` fix is unrelated and still open
   too — separate blocker, not solved by the UI).

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
