## Why

Five concrete operational problems surfaced while using the review UI day-to-day, sharpened by one
measured fact: **all 20 script files currently on disk fail `Script.model_validate()`** (12
`out/eval/*.json` + 8 `.bixia_state/*/script.json`, 0 loadable) — the in-flight
`2026-08-22-slim-script-schema-mvp` change rewrote `Script` to a flat/ID-referenced shape with no
migration (its own design.md decision four). Every one of `discover_scripts()`'s 24 records
currently renders "無法讀取此劇本檔案" in 單篇閱讀. There is no way to clear this backlog and no
way to get a readable script back into the UI without re-spending tokens on a fresh run — `ui/app.py`
has never had delete, export, or import.

Separately: `.bixia_state/` currently holds 3 layered runs stuck at `stage="scenes"` that look
resumable but are schema v1/v3 against the current `_SCHEMA_VERSION = 4` — `load_checkpoint()`
silently returns `None` on a version mismatch, so a naive "resume" would restart from scratch at
full cost with no warning. Any resume feature must gate on this, not just display it.

The other three problems are display/config accuracy gaps: `_render_run_meta()`'s layered branch
shows only 3 model roles even though `generation._cost_models()` already knows layered runs use a
4th (`proof`, via causal repair); the 生成 form's model fields are free-text with zero validation
against what's actually been tested; and the 自訂篇幅 fields carry no explanation of what they do,
including a real trap — `events` only affects the legacy pipeline, `chapters`/`beats_per_chapter`
only affect layered, and filling the wrong one for the active pipeline mode silently does nothing to
the output (though `events` still affects layered's cost/time *estimate* via `LengthSpec.events_scale`).

## What Changes

- **Script library management** (script-library): delete a script record's file (and, for a
  `source=="checkpoint"` record, its whole `.bixia_state/<run_id>/` directory) without touching
  `out/generation_runs*.jsonl` — the run row survives and the record reappears as `source="run-only"`
  with cost/error intact. Export a script as a JSON download. Import a script two ways: upload a
  JSON file that gets copied into `out/eval/` under the standard naming convention (so it's
  discoverable with zero new discovery code), or load an arbitrary path ad hoc for one-off viewing
  without copying. A bulk "delete all unreadable" action, given the 20/20-broken state above.
- **Model catalog** (model-catalog): a new curated `eval/model_catalog.json` + `src/bixiascribe/catalog.py`
  — the single source of truth for which models are tested/usable/deprecated, joined against
  `eval/model_prices.json` at load time (prices are never duplicated). Backs both the richer
  single-script model display and the constrained model dropdowns below.
- **Richer, accurate model display in 單篇閱讀**: `_render_run_meta()` shows every role a run's
  pipeline mode actually used (fixing the layered branch's missing 4th role, `proof`), each with a
  price/role-purpose/tested-status popover sourced from the catalog, plus the reasoning-token share
  of completion tokens.
- **Reasoning-effort control**: one new global (not per-role) `REASONING_EFFORT` setting
  (`default`/`none`/`low`/`medium`/`high`), threaded through `ModelChoice` into crewai's native
  `LLM.reasoning_effort` field, selectable in the 生成 form, and recorded on every run row so runs
  with different effort levels are comparable. Defaults to `default` (send nothing — byte-identical
  to today's behavior) rather than `none`, so this change is a no-op until a user opts in.
- **Constrained model selection**: the 生成 form's three free-text model inputs become dropdowns
  sourced from the catalog, restricted to tested DeepSeek V4 models (`deepseek-chat` stays
  selectable as a labeled baseline/V3 reference since `eval/model_variants.json`'s `baseline`
  variant still uses it; `glm-5.2` is excluded as confirmed non-viable).
- **Run resume** (run-resume): a new `discover_resumable_runs()` lists `.bixia_state/<run_id>/`
  directories with a readable `state.json` but no `script.json` (interrupted layered runs),
  disjoint from the existing `discover_checkpoint_runs()` (finished runs only).
  `GenerationJob` gains an optional `run_id` to resume one. **Hard-gated on schema version**: a
  checkpoint whose envelope `schema_version` doesn't match the current pipeline's is refused with an
  error, not just a warning, because resuming it would silently regenerate from scratch at full
  cost. All 3 currently-interrupted runs are schema v3 against v4 and are refused today.
- **Custom-length field documentation** (script-length): each of the four `自訂篇幅` fields gets
  real help text stating which pipeline mode(s) actually read it, sourced from verified
  `crew/tasks.py` call sites, plus a visible warning suffix when a field is inert for the
  currently-selected pipeline mode.

## Capabilities

### New Capabilities
- `script-library`: delete/export/import for scripts discovered by the review UI.
- `model-catalog`: curated model metadata (tested status, role fit, human-readable description)
  joined against `eval/model_prices.json`, backing both display and input constraints.
- `run-resume`: discovery and schema-gated resumption of interrupted layered runs.

### Modified Capabilities
- `script-length`: each custom-length field now discloses which pipeline mode(s) it affects.

## Impact

- **Code**: new `src/bixiascribe/catalog.py`, new `src/bixiascribe/library.py`, new
  `eval/model_catalog.json`; modified `src/bixiascribe/review.py` (role-key helpers,
  `discover_resumable_runs()`, versioned envelope read), `src/bixiascribe/generation.py`
  (`Variant.reasoning_effort`, `GenerationJob(run_id=...)`, `_cost_models()` deduplicated against
  `review.role_keys_for_mode()`), `src/bixiascribe/llm.py` (`ModelChoice.reasoning_effort`,
  `build_llm()`), `src/bixiascribe/config.py` (`REASONING_EFFORT`), `src/bixiascribe/length.py`
  (`FIELD_HELP`), `src/bixiascribe/crew/pipeline.py`/`crew/orchestrator.py` (`RunReport.reasoning_effort`),
  `ui/app.py` (delete/export/import UI, model dropdowns, reasoning-effort selector, resume UI,
  length-field help text).
- **Data**: none of this reads or writes existing `out/eval/*.json`/`.bixia_state/` content
  differently — delete/export/import operate on whatever files already validate (or don't) under the
  current `Script` schema; broken files can now be deleted or replaced via import instead of being
  permanently stuck.
- **Tests**: new `tests/test_catalog.py`, `tests/test_library.py`, `tests/test_llm_reasoning.py`;
  modified `tests/test_review.py`, `tests/test_generation.py`, `tests/test_length_spec.py`.
  `tests/test_pricing.py` deliberately unmodified (reasoning tokens are already priced inside
  `completion_tokens`; this change adds display, not a new pricing model).
- **Docs**: `CLAUDE.md` (Review UI section, `REASONING_EFFORT` line), `.env.example`
  (`REASONING_EFFORT`), `README.md`/`README.en.md` (update the "📋 規劃中" line — import/export/delete
  land, editing-in-place and RPG Maker export remain future work), `CONTRIBUTING.md` (name
  `library.py`/`catalog.py` in the streamlit-free rule).
