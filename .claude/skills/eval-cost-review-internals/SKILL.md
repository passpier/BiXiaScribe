---
name: eval-cost-review-internals
description: Deep internals of BiXiaScribe's cost/estimation modules (pricing.py, estimate.py), the eval_generation.py A/B harness, and the review.py/ui/app.py data layer behind the Streamlit review UI (including the 生成-from-UI generation flow). Use when editing src/bixiascribe/pricing.py, estimate.py, scripts/eval_generation.py, src/bixiascribe/review.py, src/bixiascribe/generation.py, or ui/app.py.
---

## Evaluating model splits and cost

`src/bixiascribe/pricing.py` converts a run's `token_usage`/`token_usage_by_role` into
`cost_usd`/`cost_basis` (`"by_role"` when per-role usage covers every role exactly, `"uniform"` when
every role shares one model, `"uniform_lower_bound"` for a mixed-model run with only a run-wide total,
`"unknown_price"` — never a fabricated `$0.00` — when nothing can be priced), against
`eval/model_prices.json` (an OpenRouter price snapshot, regenerate with `scripts/refresh_prices.py`).
`generation.build_run_row()` computes this for every row automatically; `eval_generation.py`'s
`print_aggregate()`/`--from-jsonl` also compute it **retroactively** for older JSONL rows that predate
this field, so historical logs can be priced without re-running anything. `pricing.quality_unit_costs()`
turns a cost into `usd_per_event`/`usd_per_dialogue_line`/`usd_per_1k_dialogue_chars` — the "is the
pricier model actually worth it" numbers, not just a total.

`scripts/eval_generation.py` is the harness built on top of that: it runs a {variant} x
{requirement} matrix (variants from `eval/model_variants.json`, requirements from
`eval/script_requirements.txt`), appends one `RunReport.to_dict()` row (plus
`crew/metrics.py::script_metrics()` structural counts — event/NPC/dialogue counts, NPC speaking
coverage, avg line length) per run to `out/generation_runs.jsonl`, saves each generated script under
`out/eval/`, and prints a per-variant aggregate (success rate, mean tokens/elapsed,
`retrieval_calls` including how many runs had **zero** — the concrete number for whether the
dialogue-agent tool-calling failure mode is typical or rare — see script-generation-internals).
`--dry-run` reuses `generate_script.py::preflight()` plus a check that every variant has all three
model ids filled in. `--from-jsonl` re-prints the aggregate from a past log without spending anything.
`crew/metrics.py` is deliberately structural-metrics-only, not an LLM-as-judge prose score — see
its module docstring for why; reading `out/eval/*.json` by hand is still how 武俠語感 gets judged.

### Pre-run / in-run estimation (`src/bixiascribe/estimate.py`)

`pricing.py` answers "what did a *finished* run cost"; `estimate.py` answers "what will/does this
run cost and how long will it take", before or while tokens are being spent — the review UI's 生成
form, a running `GenerationJob`, and both CLI scripts (`generate_script.py`'s
`--preflight-only`/pre-run printout, `eval_generation.py --dry-run`) all call into this one module
rather than each guessing independently. Motivated by a real cancelled UI run
(`.bixia_state/1787309292-req-d232acf2d8`, `layered`/`long`): cancelled after 19 minutes and $0.0039
spent with 1/30 beats done, extrapolating to ~$0.065/~2 hours to finish — money was never the
blocker, but nothing surfaced that number before 19 minutes were already spent finding out. That
same run's 30 beats also turned out to be a single linear causal chain (`plan_batches()` returns 30
batches of width 1), so `SCENE_CONCURRENCY` had zero effect on it — see
`openspec/changes/archive/profile-layered-pipeline-cost/design.md`'s 證據七.

Every `RunEstimate` carries a `basis` string, most to least trustworthy: `"measured_run"` (this
run's own already-completed scenes, via `crew/scene_metrics.py`/`orchestrator.load_scene_metrics()`
— the direct payoff of the per-scene attribution in gmud-schema-internals) → `"history_mode_length"`
(`out/generation_runs*.jsonl` rows matching the same pipeline_mode **and** script_length) →
`"history_mode"` (same pipeline_mode only, token/time priors scaled by
`length.LengthSpec.events_scale` to the requested length) → `"prior"` (no matching history at all —
a fixed, documented-provenance default sourced from the two real `scene_meta_*.json` sidecars this
project had as of 2026-08-21, replacing `eval_generation.py`'s old `_BASE_TOKENS` constant, whose
docstring claimed historical scaling it never actually did) → `"unknown_price"` (token counts exist
but no `eval/model_prices.json` entry covers the models involved — `cost_usd`/`cost_low`/`cost_high`
are `None`, never a guessed `$0`, same convention as `pricing.estimate_cost()`).

A layered run's causal-dependency batch structure (`orchestrator.py::plan_batches()`) is always
supplied by the *caller*, never re-derived inside `estimate.py` itself — `estimate.py` is
deliberately dependency-free (no crewai/checkpoint I/O) so both the UI and CLI scripts can share it,
and reimplementing the Kahn-level-ordering logic a second time there would risk the two copies
silently disagreeing about which beats can run concurrently. Without a real `beats.json` to read
(e.g. the UI form before any run has started), `estimate_run()` falls back to "every beat is its own
batch" — parallelism 1.0 — which isn't an arbitrarily conservative guess; it's what both design.md's
證據一 and 證據七 actually observed on real beat sheets.

`generation.py`'s `estimate_for_form()` (pre-run, no checkpoint yet) and `GenerationJob.estimate()`
(mid-run, switches to `"measured_run"` once `orchestrator.load_beat_sheet()`/`load_scene_metrics()`
have something to read) are the two entry points `ui/app.py` calls — `GenerationJob.estimate()` is
cached for 5 seconds internally since it's polled from a `@st.fragment(run_every=1.0)` progress
panel and would otherwise re-read `beats.json` plus every `scene_meta_*.json` sidecar off disk once
a second for the run's whole duration.

A still-valid, non-obvious finding: `retrieval_calls` shows that "the model supports function
calling" per OpenRouter's `/models` metadata is not the same guarantee as "it reliably chooses to
call the tool in a CrewAI ReAct loop" — this needs checking per model via `retrieval_calls`, not
assumed from the provider's capability flag. See `docs/BENCHMARKS.md` for the current per-agent
model-split A/B numbers, and `docs/DESIGN_NOTES.md` for the full methodology.

## Review UI + generation-from-UI

`src/bixiascribe/review.py` is the read-only data layer behind `ui/app.py`'s three review modes: it
discovers `out/eval/*.json` scripts, joins each to its `out/generation_runs*.jsonl` run metadata (the
only join key is `script_path`, deduped by keeping the row with the largest `ts` when more than one run
wrote to the same path), and exposes `discover_scripts()` / `overview_rows()` / `group_by_requirement()`
etc. It is deliberately kept **streamlit-free** — the UI is meant to be swappable for a different
frontend later (e.g. a desktop app), and the point of the split is that swap shouldn't require
rewriting this module. `tests/test_review.py` asserts this mechanically (`"import streamlit" not in
review.py`'s source), not just by convention.

Two data-quality details this module handles, learned from real `out/` data rather than assumed:
a `rep > 0` script file can be the target of more than one JSONL row across separate
`eval_generation.py` invocations (crash-resilient append-only logs), so `latest_run_by_path()` picks
the newest; and a JSONL row's embedded `script_metrics()` counts can go stale if a later rep overwrote
the file after that row was written, so `overview_rows()` never trusts them — it always recomputes
`script_metrics(load_script(path))` from whatever is currently on disk. Runs that failed before ever
writing a script file still show up as browsable records with `path=None`, `source="run-only"`, and
the run's `error` text, instead of silently disappearing because there's no JSON to open.

**`discover_scripts()` has a second discovery source: `.bixia_state/<run_id>/` checkpoints.**
`crew/orchestrator.py::run_layered()` only ever checkpoints its final `Script` to
`config.BIXIA_STATE_DIR/<run_id>/script.json` — publishing to `out/eval/*.json` + a JSONL run row is a
separate step that only `generation.generate()` does (see below), which `scripts/generate_script.py
--pipeline-mode layered` never calls (it only honors `--out`). Without this second source, a layered
run kicked off from that CLI would be tokens-spent but permanently invisible to the review UI.
`discover_checkpoint_runs()` scans `state_dir/*/state.json`, keeps only `stage == "done"` dirs that
also have a `script.json`, and returns them newest-`last_updated`-first, capped at
`config.CHECKPOINT_REVIEW_LIMIT` (default 20, `0` = unlimited) — a dev machine accumulates many
short/test run dirs (including offline-test `FakeLLM` runs) alongside the few real long runs actually
worth reviewing; uncapped, those would bury the real ones in every dropdown and in 總覽表. Each becomes
a `ScriptRecord` with `variant=f"checkpoint:{run_id}"` (unique by construction, and it's the same
string `--run-id` expects to resume that run) and a synthesized `RunRecord` (`mode="layered"`,
`scenes_generated=len(completed_scene_ids)`). `discover_scripts(include_checkpoints=...)` (default
`True`) merges these in; `ui/app.py`'s sidebar has a checkbox for it. A run interrupted before
`stage="done"` has no assembled `script.json` yet and still won't appear here — resume it with
`--run-id` first.

Every checkpoint file (not just `script.json`) is wrapped in `orchestrator.save_checkpoint()`'s
`{"schema_version", "data"}` envelope, so `load_script()` unwraps it structurally (exactly the two keys
`schema_version`/`data`, the latter a dict) before validating as `Script` — **not** by importing
`orchestrator.py`'s `_SCHEMA_VERSION`, since `review.py` is deliberately crewai-free (importing
`orchestrator` would also be circular: it already imports `review`).
`tests/test_orchestrator.py::test_saved_script_checkpoint_round_trips_through_review_load_script` pins
this contract from the producer side, alongside `tests/test_review.py`'s own envelope-unwrap test.

`src/bixiascribe/generation.py` is the second, equally streamlit-free module behind `ui/app.py`'s 生成
mode, which triggers a real generation run from the browser. It owns `preflight()` (shared by both
CLIs and the UI) and `generate()`/`GenerationJob` — see their docstrings for the row/filename
conventions and the busy-lock that keeps two concurrent runs from corrupting `crew/tools.py`'s
retrieval-stats global. A run's script lands under `out/eval/{ui-variant}__{slug}.json` (the `ui-`
prefix keeps it from colliding with an eval-harness artifact at the same path) and its row in
`out/generation_runs_ui.jsonl` — a separate file from the eval harness's, but still matched by
`config.RUN_LOG_GLOB`'s glob, so `review.py`'s discovery (and therefore all three review modes) picks
up UI runs automatically with zero changes to that module.

**A real run takes 126–240s and CrewAI's `step_callback` never fires for this crew** (verified against
crewai 1.15.5: the toolless 編劇/校對 agents take `_invoke_loop_native_no_tools`, which skips
`_invoke_step_callback` entirely — only `_invoke_loop_react`/`_invoke_loop_native_tools` call it, and
the 對話 agent frequently never uses its tool at all — see script-generation-internals). So a blocking
UI call could only repaint ~3 times (crewai's `task_callback`, which does fire once per task) over
those minutes with no ticking clock. `generate()` accepts an `on_step` callback (translated into
`crew/pipeline.py`'s crewai-independent `StepEvent`) for exactly that reason, and `GenerationJob` runs
one generation on a background thread so `ui/app.py` can poll `job.snapshot()` via
`st.fragment(run_every=1.0)` for a live elapsed clock, a task-boundary progress bar, and a working 取消
button — the worker thread never touches `st.*` or `session_state` itself, only the job's own
lock-guarded fields, so there's no `ScriptRunContext` issue. Persisting the script/row happens on the
worker thread too, so an accidental browser refresh mid-run loses the UI's handle on the job but not
the already-spent tokens — the result still shows up in 單篇閱讀 on the next load.

`GenerationJob.pending_scenes()` / `.scene_context()` (thin wrappers over `crew/orchestrator.py`'s
`load_pending_scenes()` / `load_scene_context()`, both pure checkpoint reads) let the UI's batch-
confirmation panel show a staged layered-mode batch's actual 標題/地點/台詞/分支 (via the same
`_render_event()` the read-only modes already use), not just the beat ids `JobSnapshot.pending_scene_ids`
carries — a bare id list is a blind-confirm UX otherwise. That panel is rendered *outside*
`_render_generation_progress()`'s `@st.fragment(run_every=1.0)` — a fragment repainting every second
can swallow a button click mid-press — so `_render_generation_progress()` leaves the fragment via a
full-app `st.rerun()` as soon as it sees `snap.awaiting_confirmation`, and the page's non-fragment
body renders the static confirmation panel instead until it's resolved.

A "layered" run cancelled while parked in `gate` (e.g. the UI's 取消 button during batch
confirmation) used to leave **no** JSONL row at all — `run_layered()`'s `gate(pending_ids)` call sits
outside `dispatch_batch()`'s own try/except, so `generation.GenerationCancelled` unwound past
`generate()`'s only two row-writing exits (success, `except PipelineError`) uncaught, and every token
already spent on the batch went unrecorded. `run_layered()` now takes an `on_report` callback, invoked
from `_finalize_report()` on every one of its exit paths (so a caller always has the latest partial
`RunReport`, exception or not), and the `gate()` call site itself is wrapped so an exception there
still calls `_finalize_report()`/`on_report` before re-raising **unchanged** — the exception type must
survive intact, since `GenerationJob._run()` branches on `except GenerationCancelled` specifically to
report `status="cancelled"` rather than `"failed"`. `generate()` uses this hook to write a row (via
the same `build_run_row()`/`pricing.estimate_cost()` path every other row goes through) from inside its
own `except GenerationCancelled`/`except Exception` handlers before re-raising, attaching the row to
`GenerationCancelled.row` so `GenerationJob._run()` can hand it to the UI as a `GenerationResult`. Cost
is still computed only in `out/generation_runs*.jsonl` rows, never in `.bixia_state/` checkpoints —
`discover_checkpoint_runs()`'s synthesized `RunRecord` (see above) has no token/model data to price at
all, so a `variant="checkpoint:*"` record's cost is unavoidably `cost_usd=None`/`cost_basis=""`.
`ui/app.py::_render_run_meta()` shows this cost (a "成本 (USD)" metric alongside the existing
耗時/retrieval_calls/total_tokens ones, `None` rendered as `—` with an explicit "無法定價（不代表免費）"
caption, never a bare `$0`) for every path that already reaches it — 單篇閱讀, 生成 mode's失敗-run
panel, and (new) a cancelled-run panel, stashed in `st.session_state["gen_cancelled_result"]` since
`_render_generation_progress()`'s `st.rerun()` right after detecting `status="cancelled"` would
otherwise wipe an inline `st.warning()` before the user ever saw it.

`generation.Variant.ui_visible` (default `true`) filters `ui/app.py`'s 模型變體 dropdown without
touching what `eval_generation.py --variants ...` can run — `eval/model_variants.json` sets it `false`
on `long-cheap`/`long-mimo` (unreliable/expensive, see their notes) and the retrieval-toggle
`baseline-norag` variant, keeping them reproducible for the CLI harness (and `docs/BENCHMARKS.md`'s
tables) without cluttering the browser picker.

`ui/app.py` is only Streamlit widgets on top of both modules: four modes (單篇閱讀 / 並排比較 / 總覽表 /
生成). Run with `pip install -r requirements-ui.txt && .venv/bin/streamlit run ui/app.py` (running
under the wrong interpreter, e.g. a system/anaconda `streamlit` on `PATH`, silently breaks
`wuxia_corpus_search` — see CLAUDE.md's Gotchas) — the three review
modes stay read-only (no API key, no Chroma, no tokens spent); 生成 needs both, like the CLI scripts.
Side-by-side comparison aligns events **ordinally** (event *i* of each variant, side by side) rather
than by id/title, since event ids are model-generated and have no stable identity across variants.
