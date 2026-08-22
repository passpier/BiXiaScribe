## 1. `src/bixiascribe/length.py` — custom-length field help text

- [x] 1.1 Add `FIELD_HELP: dict[str, dict[str, str]]` (keys matching `_CUSTOM_FIELDS`) with
      `label`/`affects`/`help` per field, grounded in verified `crew/tasks.py` call sites:
      `events` → legacy writer task only (`tasks.py:184`, also feeds `LengthSpec.events_scale`'s
      cost/time estimate for either mode); `chapters`/`beats_per_chapter` → layered beat_expand
      task only (`tasks.py:337-338`); `min_dialogue` → both dialogue tasks
      (`tasks.py:210,213,388,390`).
- [x] 1.2 Add `_AFFECTS_LABEL` mapping `{"legacy": ..., "layered": ..., "both": ...}` for the UI
      warning suffix.

## 2. `src/bixiascribe/catalog.py` + `eval/model_catalog.json` — model catalog foundation

- [x] 2.1 Create `eval/model_catalog.json`: `models` (deepseek-v4-flash-0731 tested,
      deepseek-v4-pro tested, deepseek-chat baseline, glm-5.2 unusable), `roles` (six role
      key → label/modes/note), `reasoning_efforts` (default/none/low/medium/high → label/note).
      No price/context/tool-support fields — those join from `eval/model_prices.json`.
- [x] 2.2 `catalog.py`: `ModelInfo`/`RoleInfo`/`Catalog` frozen dataclasses; `load_catalog(path,
      prices=None)` (imports only `config`+`pricing`, never `llm.py`); `Catalog.describe(model_id)`
      (never raises, degrades unknown ids to `status="untested"`); `Catalog.selectable(role=None)`
      (excludes `unusable`); `Catalog.roles_for_mode(mode)`; `model_label(info)` (format_func,
      omits price segment when unpriced rather than showing `$0`); `normalize_reasoning_effort(value)`
      (validate-and-fallback to `"default"`, imitating `config.py:202-208`'s `CAUSAL_VALIDATION`
      pattern).

## 3. `src/bixiascribe/review.py` — role-key helpers + resumable-run discovery

- [x] 3.1 Add `role_keys_for_mode(mode: str) -> tuple[str, ...]` and `run_role_models(run:
      RunRecord) -> list[tuple[str, str]]`, with the layered set including `"proof"` (see design.md
      實測四).
- [x] 3.2 Split `_read_envelope()` into `_read_envelope_versioned(path) -> tuple[int | None, dict |
      None]` plus a one-line `_read_envelope()` wrapper preserving current behavior exactly (no
      change to `discover_checkpoint_runs()`'s existing tests).
- [x] 3.3 Add `ResumableRun` frozen dataclass (`run_id, path, requirement, stage,
      completed_scene_ids, pending_scene_ids, last_updated, schema_version`) and
      `discover_resumable_runs(state_dir=config.BIXIA_STATE_DIR, limit=config.CHECKPOINT_REVIEW_LIMIT)
      -> list[ResumableRun]`: readable `state.json`, no `script.json`, newest-first, corrupt dirs
      skipped silently — same conventions as `discover_checkpoint_runs()`.
- [x] 3.4 Add `RunRecord.reasoning_effort: str = ""` field, read via `row.get("reasoning_effort")
      or ""`.
- [x] 3.5 Update `ScriptRecord`'s docstring/comment (currently missing `"checkpoint"`) to enumerate
      all four `source` values including the new `"adhoc"`.
- [x] 3.6 Add `reasoning_effort` to `overview_rows()`'s output dict.

## 4. `src/bixiascribe/library.py` — new module: delete/export/import

- [x] 4.1 `DeletePlan` frozen dataclass; `plan_delete(rec, *, state_dir=...) -> DeletePlan`;
      `delete_record(rec, *, state_dir=...) -> DeletePlan` — file-only delete for
      jsonl/filename-sourced records, whole-directory delete for checkpoint-sourced records, never
      touches JSONL logs; raises `ValueError` for any path outside `config.EVAL_SCRIPTS_DIR`/
      `state_dir`.
- [x] 4.2 `export_bytes(script) -> bytes` (byte-identical to `generation.py:598`'s
      `model_dump_json(indent=2, exclude_none=False)`), `export_filename(rec) -> str`.
- [x] 4.3 `ImportRejected(ValueError)`; `validate_import(payload: bytes) -> Script` (parses JSON,
      unwraps a `{"schema_version","data"}` envelope the same structural way `review.load_script()`
      does, wraps all failures as `ImportRejected`).
- [x] 4.4 `import_script(payload, *, variant, requirement, rep=None, scripts_dir=...) -> Path` —
      writes via `generation.script_path_for()` + `generation.next_rep()`.
- [x] 4.5 `load_ad_hoc(path: Path) -> tuple[Script, ScriptRecord]` — returns `source="adhoc"`,
      `run=None`.

## 5. `src/bixiascribe/generation.py` — dedup role list, reasoning effort, resume plumbing

- [x] 5.1 Rewrite `_cost_models()` (`:326-340`) to build its role dict from
      `review.role_keys_for_mode(report.mode)` instead of a hardcoded literal, for both legacy and
      layered branches.
- [x] 5.2 `Variant.reasoning_effort: str | None = None` (appended last, per the existing
      `session_doc_max_tokens`/`script_length`/`use_retrieval` convention); `from_dict` reads it;
      `to_model_choice()` passes `reasoning_effort=self.reasoning_effort or config.REASONING_EFFORT`.
- [x] 5.3 `generate(..., reasoning_effort: str | None = None)` — three-level resolution (explicit >
      variant > config), canonicalized via `catalog.normalize_reasoning_effort()`, mirroring
      `script_length`'s existing resolution at `:479-511`.
- [x] 5.4 `GenerationJob.__init__(..., run_id: str | None = None, reasoning_effort: str | None =
      None)` — `run_id` used to resume an existing `.bixia_state/<run_id>/` instead of always
      minting a fresh one (`:697-699`); `reasoning_effort` stored and forwarded in `_run()`.
- [x] 5.5 Re-export `CHECKPOINT_SCHEMA_VERSION = crew.orchestrator._SCHEMA_VERSION` (already
      imported at `:44`) so the UI can compare versions without `review.py` importing
      `orchestrator.py`.

## 6. `src/bixiascribe/llm.py` — reasoning effort wired to crewai's native field

- [x] 6.1 `ModelChoice.reasoning_effort: str = config.REASONING_EFFORT`.
- [x] 6.2 `build_llm()`: after normalizing via `catalog.normalize_reasoning_effort(models.reasoning_effort)`,
      set `llm_kwargs["reasoning_effort"] = effort` only when `effort != "default"` — uses crewai's
      native `LLM.reasoning_effort` field (confirmed present, `Literal["none","low","medium","high"]
      | None`), not `extra_body`, so no change to the existing `additional_params` assignment at
      `:140` and no merge-collision risk with provider routing.

## 7. `src/bixiascribe/config.py` — `REASONING_EFFORT`

- [x] 7.1 Add `REASONING_EFFORT`, imitating `CAUSAL_VALIDATION`'s validate-and-fallback pattern
      (`:202-208`): allowed set `{"default","none","low","medium","high"}`, default `"default"`.

## 8. `src/bixiascribe/crew/pipeline.py` / `crew/orchestrator.py` — record reasoning effort on runs

- [x] 8.1 `RunReport.reasoning_effort: str = ""` (next to `script_length`, `pipeline.py:161`).
- [x] 8.2 Set it at both `RunReport` construction sites — `pipeline.py:438` (legacy) and
      `orchestrator.py:1340` (layered) — from `models.reasoning_effort`.

## 9. `ui/app.py` — wire everything into widgets

- [x] 9.1 Module-level `_CATALOG = catalog.load_catalog()` alongside existing cached data access.
- [x] 9.2 `_render_run_meta()`: replace the two hardcoded 3-column model blocks with one loop over
      `review.run_role_models(run)`, each column showing role label + model display name plus a
      `st.popover` with full model id, role note, price, tool-support, tested status; add a
      reasoning-token-share caption.
- [x] 9.3 `_render_script()`: add `st.download_button` for `library.export_bytes(script)` after the
      metrics row.
- [x] 9.4 Add `@st.dialog("確認刪除")` confirm flow wired into the 單篇閱讀 branch and the
      unreadable-file branch (`:789-790`); clear `st.cache_data` **before** `st.rerun()`; add a
      bulk "刪除所有無法讀取的劇本" action.
- [x] 9.5 Add sidebar import UI: `st.file_uploader` + variant/requirement inputs + explicit import
      button (not import-on-upload, since the uploader re-delivers its buffer every rerun); plus an
      ad-hoc path input feeding `library.load_ad_hoc()` into `st.session_state["adhoc_record"]`.
      Do **not** add `key=` to the existing sidebar variant/rep selectboxes (`:763-777`) — would
      leave a deleted variant string stuck in session state.
- [x] 9.6 生成 form: reorder so `use_layered` checkbox precedes the variant/自訂 block (自訂模型's
      role list depends on it); replace the three free-text model inputs with `st.selectbox`es from
      `_CATALOG.selectable(role_key)` per `review.role_keys_for_mode(mode)`; add the reasoning-effort
      `st.selectbox` below the model block; forward both into `ui_variant`/`GenerationJob`.
- [x] 9.7 生成 form: add a "續跑未完成的執行" expander from `review.discover_resumable_runs()`; on
      selection, compare `schema_version` against `generation.CHECKPOINT_SCHEMA_VERSION` and hard
      block (reset selection to `None`, `st.error`) on mismatch; lock the requirement field when a
      resume is selected; warn that variant/篇幅/檢索/reasoning settings aren't persisted in the
      checkpoint.
- [x] 9.8 `gen_last_result` branch: add the same export button as 9.3.
- [x] 9.9 自訂篇幅 block (`:654-671`): loop over `length.FIELD_HELP`, add `help=` text, append
      `⚠ 僅 legacy` / `⚠ 僅 layered` suffix when the field is inert for the current pipeline mode;
      add a caption noting all four are unenforced prompt targets; show `length.PRESETS` via
      `st.dataframe` as reference.

## 10. Tests

- [x] 10.1 `tests/test_catalog.py` (new): bidirectional id-set consistency between
      `model_catalog.json` and `model_prices.json`; every `model_variants.json` model id has a
      catalog entry; `recommended_roles` values are valid role keys; `describe()` degrades
      gracefully on unknown ids; missing/corrupt catalog file → empty `Catalog`; `selectable()`
      excludes glm-5.2, includes deepseek-chat; `normalize_reasoning_effort` falls back to
      `"default"` on empty/garbage input.
- [x] 10.2 `tests/test_library.py` (new), tmp-dir only: `export_bytes` byte-identical to
      `generation.py:598`'s serialization; delete leaves JSONL untouched and
      `discover_scripts()` picks the run back up as `source="run-only"` with cost/error intact;
      checkpoint delete removes the whole run directory; out-of-bounds delete raises `ValueError`;
      `import_script` output is parseable by `parse_script_filename()` and discoverable; repeated
      import of the same requirement gets `__rep1`; malformed/schema-invalid payloads raise
      `ImportRejected` and write nothing; a checkpoint envelope payload is accepted; `load_ad_hoc`
      returns `source="adhoc"`.
- [x] 10.3 `tests/test_review.py` (modified): `discover_resumable_runs` cases mirroring
      `discover_checkpoint_runs`'s existing ~six cases (stage filter, schema_version reported
      verbatim including mismatched versions, ordering/limit, corrupt/missing dirs);
      `role_keys_for_mode("layered")` includes `"proof"`; `RunRecord.from_row` reads
      `reasoning_effort` defaulting to `""`; `_read_envelope()` behavior-unchanged regression after
      the versioned split.
- [x] 10.4 `tests/test_generation.py` (modified): `GenerationJob(run_id=...)` actually resumes an
      existing checkpoint dir under `_isolated_state_dir()` (no second dir created, existing
      `scene_*.json` files untouched); legacy mode still yields `run_id == ""`; `_cost_models()`'s
      key set equals `review.role_keys_for_mode(report.mode)`; `Variant` round-trips
      `reasoning_effort`; `generate()`'s three-level resolution; run row carries
      `reasoning_effort`.
- [x] 10.5 `tests/test_llm_reasoning.py` (new): mock `crewai.LLM`, assert `reasoning_effort="default"`
      passes no such kwarg (today's behavior preserved), `"none"/"low"/"medium"/"high"` pass it
      through unchanged, and it coexists with provider-routing `additional_params` without either
      overwriting the other.
- [x] 10.6 `tests/test_length_spec.py` (modified): `set(FIELD_HELP) == set(_CUSTOM_FIELDS)`; every
      `affects` value is one of `{"legacy","layered","both"}`.
- [x] 10.7 `tests/test_pricing.py`: deliberately unmodified (see design.md 決策六).
- [x] 10.8 Run `pytest tests/` and `ruff check .` clean.

## 11. Docs

- [x] 11.1 `CLAUDE.md`: update the Review UI description, add a `REASONING_EFFORT` line to the env
      var documentation.
- [x] 11.2 `.env.example`: add `REASONING_EFFORT` (commented, default `default`).
- [x] 11.3 `README.md`/`README.en.md`: update the "📋 規劃中" line — import/export/delete land;
      editing-in-place and RPG Maker export remain future work.
- [x] 11.4 `CONTRIBUTING.md` (or wherever the streamlit-free rule is documented, if it exists as a
      separate file — otherwise `CLAUDE.md`'s Gotchas): name `library.py`/`catalog.py` alongside
      `review.py`/`generation.py` in the streamlit-free rule.

## 12. Verification

- [x] 12.1 Offline: `ruff check .`; `pytest tests/`.
- [x] 12.2 `LLM_BACKEND=fake .venv/bin/streamlit run ui/app.py` manual walkthrough: delete a broken
      record, bulk-delete, import a fake-generated script, ad hoc load, layered run shows 4 roles
      with popovers, model dropdowns constrained, reasoning-effort selection recorded on the run
      row, resume listing shown but hard-blocked on the 3 existing v3-vs-v4 checkpoints, custom
      length fields show mode-inert warnings correctly.
