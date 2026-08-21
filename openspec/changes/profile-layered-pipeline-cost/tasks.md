## 1. `scene_metrics.py` core module

- [x] 1.1 Create `src/bixiascribe/crew/scene_metrics.py`: `SceneMetric` (pydantic `BaseModel`,
      fields `beat_id`/`elapsed_s`/`call_elapsed_s`/`repair_elapsed_s`/`llm_calls`/
      `reasoning_tokens`/`total_tokens`/`guardrail_retries`/`retrieval_calls`/
      `structured_fallbacks`, all defaulted) and `SceneStats` (`scenes: dict[str, SceneMetric]`,
      `as_rows() -> list[dict]` in first-seen order), following `crew/execute.py`'s
      module-level-`_stats` + `threading.Lock` + `get_stats()`/`reset_stats()` pattern. No
      crewai import.
- [x] 1.2 Implement `scene_scope(beat_id)` as a `threading.local()`-backed `@contextmanager`:
      sets the current-scene id for this thread, times the block (including on exception), folds
      `elapsed_s` into that beat's `SceneMetric` on exit.
- [x] 1.3 Implement `record_call_elapsed(sec)`, `record_guardrail_retry(beat_id=None)`,
      `record_retrieval_call()`, `record_structured_fallback()`,
      `record_usage(beat_id, usage_delta)`, `record_repair_elapsed(beat_id, sec)` — each a no-op
      when called with no active scope on the current thread (and, for `record_usage`/
      `record_repair_elapsed`, when `beat_id` doesn't match the current scope), so legacy-mode
      calls into `execute.run_task()`/`WuxiaRetrievalTool._run()` are unaffected.
- [x] 1.4 Write `tests/test_scene_metrics.py`: no-op-outside-scope for every recorder;
      `scene_scope` records elapsed time even when the block raises; concurrent scopes on
      separate threads attribute independently; `as_rows()` shape/order; `reset_stats()` zeroes
      state.

## 2. Wire the recorders into existing call sites

- [x] 2.1 `crew/orchestrator.py::_default_write_scene`: wrap the body in
      `scene_metrics.scene_scope(beat.id)`; after computing the usage delta, call
      `scene_metrics.record_usage(beat.id, usage)` (maps `successful_requests`→`llm_calls`,
      `reasoning_tokens`, `total_tokens`).
- [x] 2.2 `crew/orchestrator.py::_default_repair_scene`: time the call and call
      `scene_metrics.record_repair_elapsed(event.id, ...)`; verify usage from this call folds
      into the same beat (relies on `event.id` already being forced to `beat.id` before
      `_validate_scene()` calls this).
- [x] 2.3 `crew/execute.py::run_task`: time whichever `execute_sync()` call succeeds and call
      `scene_metrics.record_call_elapsed(...)`; call `scene_metrics.record_structured_fallback()`
      alongside the existing `_record(note)` fallback bookkeeping.
- [x] 2.4 `crew/tools.py::WuxiaRetrievalTool._run`: call `scene_metrics.record_retrieval_call()`
      next to the existing `_stats.calls += 1`.
- [x] 2.5 `crew/tasks.py::_scene_guardrail`: on a non-empty `problems` list (rejection), call
      `scene_metrics.record_guardrail_retry(beat.id)` before returning `(False, feedback)`.
- [x] 2.6 Confirm (via `tests/test_orchestrator_parallel.py`'s existing serial-vs-concurrent
      coverage) that both `dispatch_next()` (serial, `SCENE_CONCURRENCY=1`) and `dispatch_batch()`
      (concurrent) paths produce attributed metrics — `dispatch_batch()` delegates to
      `dispatch_next()` when ungated, so both must go through the same instrumented runner.

## 3. Sidecar persistence for resumed runs

- [x] 3.1 Add `_scene_meta_path(run_id, beat_id)` in `crew/orchestrator.py`, alongside
      `_scene_path`/`_pending_scene_path`, pointing at
      `.bixia_state/<run_id>/scene_meta_<beat_id>.json`.
- [x] 3.2 In `dispatch_next()` and `dispatch_batch()`, write the beat's `SceneMetric` via the
      existing `save_checkpoint()` immediately after the scene's own `Event` checkpoint is
      written, for both the committed and staged-pending branches.
- [x] 3.3 Implement `load_scene_metrics(run_id)`: read every sidecar back, filtered to beat ids
      whose committed `scene_<id>.json` exists (never surface pending/rejected batch metrics).
- [x] 3.4 Confirm no `_SCHEMA_VERSION` bump is needed — `detect_stage()` doesn't consult this
      sidecar, so an in-flight pre-change checkpoint resumes normally with metrics starting from
      the resume point.
- [x] 3.5 `tests/test_orchestrator.py`: add a resume test that pre-seeds `scene_meta_*.json`
      sidecars (plus their `scene_*.json`), resumes the run, and asserts the final report covers
      *all* scenes, not just newly-generated ones. Confirm
      `test_two_tuple_runners_still_supported` still passes unmodified.

## 4. Thread through RunReport / JSONL / review.RunRecord / UI

- [x] 4.1 `crew/pipeline.py::RunReport`: add `scene_metrics: list[dict[str, Any]] =
      field(default_factory=list)`, with the standard "empty for rows logged before this field
      existed" comment; add the matching key to `to_dict()`.
- [x] 4.2 `crew/orchestrator.py::run_layered`: call `scene_metrics.reset_stats()` next to the
      existing `reset_stats()`/`execute.reset_stats()` pair at run start; in `_finalize_report()`,
      set `report.scene_metrics = load_scene_metrics(run_id)` (runs on all four existing exit
      paths, including the crash converter, so a failed run's partial attribution is included for
      free).
- [x] 4.3 Confirm `generation.py::build_run_row()` needs no change (it does
      `row.update(report.to_dict())`).
- [x] 4.4 `review.py::RunRecord`: add `scene_metrics: tuple[dict, ...] = ()` and
      `tuple(row.get("scene_metrics") or ())` in `from_row`, matching the existing old-row
      default convention.
- [x] 4.5 `ui/app.py::_render_run_meta`: add a fourth expander (after the existing
      `quality_problems` one) with `st.dataframe` of `run.scene_metrics`, sorted by `elapsed_s`
      descending; columns beat_id / 總秒數 / 最後一次呼叫秒數 / 修補秒數 / LLM 呼叫 /
      reasoning tokens / guardrail 重試 / 檢索次數.
- [x] 4.6 `scripts/generate_script.py`'s stderr report: add one line naming the 3 slowest scenes
      by `elapsed_s` from `report.scene_metrics` (skip the line when `scene_metrics` is empty,
      e.g. legacy mode).

## 5. Remaining tests

- [x] 5.1 `tests/test_orchestrator_parallel.py`: extend alongside
      `test_token_usage_accumulation_is_thread_safe` and
      `test_retrieval_stats_correct_under_concurrent_tool_calls` — each beat in a concurrent
      batch gets exactly one `SceneMetric` row, with retrieval calls attributed to the correct
      beat.
- [x] 5.2 `tests/test_review.py`: extend `test_run_record_from_partial_row_uses_empty_defaults`
      and `test_run_record_from_row_reads_layered_fields` to cover `scene_metrics`.
- [x] 5.3 `tests/test_structured_fallback.py`: add a case asserting per-scene
      `structured_fallbacks` attribution via `scene_metrics`.
- [x] 5.4 `tests/test_guardrail_wiring.py`: add a case where a rejecting scene guardrail bumps
      `guardrail_retries` on the relevant `SceneMetric` (build the task with guardrails explicitly
      active, since `LLM_BACKEND=fake` forces guardrails off).

## 6. Docs and full verification

- [x] 6.1 `CLAUDE.md`: add a "Per-scene 執行歸因" subsection under "Script generation" covering
      the thread-local scope (and why the runner, not `dispatch_batch`, is the hook point), the
      sidecar + why no `_SCHEMA_VERSION` bump, and that `llm_calls` comes from
      `successful_requests`.
- [x] 6.2 `docs/DESIGN_NOTES.md`: add a dated section recording the r²=0.27 finding and this
      instrumentation as the answer to it.
- [x] 6.3 Run the full offline verification sequence (fake-backend layered run, resume, serial
      `SCENE_CONCURRENCY=1` run, legacy-mode run with `scene_metrics == []`, UI smoke check) from
      design.md/the approved plan's Verification section.
- [x] 6.4 Run `pytest tests/` and `ruff check .`, confirm both clean.
- [x] 6.5 Run `openspec validate --strict` for this change, confirm it passes.

## 7. `run-cost-estimation` (motivated by design.md's 證據七)

- [x] 7.1 Fix `crew/scene_metrics.py`'s per-scene retrieval-call undercount: switch `_current`
      from `threading.local()` to `contextvars.ContextVar`, since crewai's native tool-calling
      loop dispatches concurrent tool calls via
      `ThreadPoolExecutor.submit(contextvars.copy_context().run, ...)`, which only propagates
      ContextVars (not thread-locals) across that thread hop. `tests/test_scene_metrics.py`:
      add a regression test reproducing the ThreadPoolExecutor hop.
- [x] 7.2 Create `src/bixiascribe/estimate.py` (pure, no crewai/streamlit import):
      `RunEstimate`, `load_history()`, `estimate_run()`, `estimate_remaining()` -- basis priority
      `measured_run` > `history_mode_length` > `history_mode` > `prior` > `unknown_price`
      (decision 五), batch widths supplied by the caller via `plan_batches()` rather than
      re-implemented here (decision 六). `tests/test_estimate.py`: one case per basis tier, an
      unpriced-model case asserting `None` not `0`, a batch-widths/concurrency case, an
      `estimate_remaining()` fallback-when-nothing-completed case.
- [x] 7.3 `crew/orchestrator.py`: add public `load_beat_sheet(run_id)` so callers outside this
      module can get a real beat count / feed `plan_batches()` without reaching into
      `_beats_path()`/`load_checkpoint()` internals directly.
- [x] 7.4 `scripts/eval_generation.py`: `_estimate_matrix_cost()` rebuilt on top of
      `estimate.estimate_run()`; delete the stale `_BASE_TOKENS` constant (its docstring claimed
      historical scaling it never actually did); `dry_run()`'s printout gains an estimated-time
      line alongside the existing cost line.
- [x] 7.5 `scripts/generate_script.py`: `_print_estimate()` prints the pre-run estimate under
      both `--preflight-only` and right before a real run starts; when `--run-id` resumes an
      existing layered checkpoint with a `beats.json` already on disk, use its real beat count
      and `plan_batches()` output instead of the script_length-only prior.
- [x] 7.6 `src/bixiascribe/generation.py`: `estimate_for_form()` (pre-run, for the UI form) and
      `GenerationJob.estimate()` (mid-run, `measured_run` basis once scenes have committed,
      5-second cache since it's polled from a 1s-interval UI fragment).
- [x] 7.7 `ui/app.py`: a pre-run estimate panel above the "開始生成" button (updates live as
      variant/script_length/pipeline_mode/retrieval selections change); an ETA/running-cost line
      in the generation-progress fragment; a "confirming this batch still needs ~X more" line in
      the batch-confirmation panel.
- [x] 7.8 `CLAUDE.md`: add a "Pre-run / in-run estimation" subsection under "Evaluating model
      splits and cost" covering the basis priority, why `batch_widths` is caller-supplied, and
      the contextvars root cause from 7.1. Update the existing "Per-scene 執行歸因" subsection's
      "thread-local" wording to "context-local" per the 7.1 fix.
- [x] 7.9 `docs/DESIGN_NOTES.md`: a dated section recording the $0.07-but-2-hours /
      30-beats-fully-linear / 71%-reasoning-tokens findings (design.md's 證據七) as the baseline
      for a future `reasoning_effort` change.
- [x] 7.10 Run `pytest tests/` and `ruff check .` again after 7.7-7.9, confirm both clean;
      `openspec validate --strict` for this change.
