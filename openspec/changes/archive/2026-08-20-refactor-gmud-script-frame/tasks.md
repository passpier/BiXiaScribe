## 1. Schema

- [x] 1.1 Add `FactionRelation`/`Faction` models to `schema.py`
- [x] 1.2 Add `StatThreshold` model
- [x] 1.3 Add `SubLocation`/`Region` models
- [x] 1.4 Add `ProgressiveReveal`/`TruthLayer` models
- [x] 1.5 Add `Clue` model
- [x] 1.6 Rename `ChapterOutline` to `Chapter`, add `hook`/`event_ids`/`converge_event_id`/
      `clue_ids`, and update `Outline.chapters: list[Chapter]`
- [x] 1.7 Add `SkillCheck` model
- [x] 1.8 Add `StatCondition`/`Ending` models
- [x] 1.9 Extend `Branch` with `cost`, `immediate_feedback`, `payoff_chapter_id`,
      `payoff_description`, `converges_to_event_id`
- [x] 1.10 Extend `Event` with `chapter_id`, `scene_kind`, `region_id`, `sub_location_id`,
      `checks: list[SkillCheck]`, `clue_ids`
- [x] 1.11 Extend `NPC` with `faction_id`, `surface_motive`, `true_motive`,
      `attitude_by_threshold`
- [x] 1.12 Extend `PlayerCharacter` with `origin`, `weakness`, `token_item_id`,
      `relation_to_core_event`
- [x] 1.13 Extend `Script` with `theme`, `goal`, `tone`, `factions`, `regions`,
      `truth: TruthLayer | None`, `stat_thresholds`, `chapters`, `clues`, `endings`
- [x] 1.14 Extend `ExtractionResult` with `theme`/`goal`/`tone`, `factions`, `regions`, `truth`,
      `stat_thresholds`, `clues`, `endings`
- [x] 1.15 Extend `validate_references()` to cross-check every new id class listed in design.md
      (faction relations, stat-threshold targets, event region/sub-location/chapter/clue refs,
      skill-check targets, clue refs, chapter refs, ending refs, branch payoff/convergence refs,
      player token item, progressive-reveal refs)
- [x] 1.16 Add `validate_stat_thresholds(script)` (coverage + non-overlap + unlocks-something)
- [x] 1.17 Add `validate_truth_layering(script)` (reveals resolve, non-decreasing chapter order)
- [x] 1.18 Extend `tests/test_rpg_schema.py` and `tests/test_schema_layered.py` for every model
      above and both new validators; add a round-trip test loading an existing `out/eval/*.json`
      fixture to prove the additive-defaults claim
- [x] 1.19 Grep-fix every `ChapterOutline` reference (`tests/test_schema_layered.py`,
      `tests/test_orchestrator.py`, `tests/test_orchestrator_parallel.py`) to `Chapter`

## 2. Guardrails (pure/offline, zero-token testable)

- [x] 2.1 Add `check_choice_quality(event)` — missing `cost`, and the 假選擇 text-overlap +
      shared-effect-target heuristic
- [x] 2.2 Add `check_delayed_payoff(script)` — deferred effects without `payoff_chapter_id`/
      `payoff_description`, or a payoff chapter earlier than the branch's own
- [x] 2.3 Add `check_stat_narrative(script)` wrapping `validate_stat_thresholds`
- [x] 2.4 Add `check_truth_pacing(script, up_to_chapter)` — hidden-fact substring leak detection
- [x] 2.5 Add `check_convergence(script)` — missing/unreachable `converge_event_id`, >3-branch
      choice points
- [x] 2.6 Add `check_check_fallback(event)` — `SkillCheck` with neither failure branch nor item
      bypass, or empty `failure_cost`
- [x] 2.7 Add `check_scene_information(event)` — no clue/item/reveal unlocked
- [x] 2.8 Add `check_scene_mix(script)` — flavor scenes substantially outnumbering main scenes
- [x] 2.9 Add `check_regions(script)` — regions with <2 sub-locations, dangling sub-location refs
- [x] 2.10 Extend `tests/test_guardrails.py` with pass/fail cases for all nine checks

## 3. Context builder + SessionDocument

- [x] 3.1 Add `faction_cards`, `threshold_card`, `chapter_card`, `region_card`, `truth_public`,
      `truth_unlocked` fields to `SessionDocument`, declared before `current_beat`, all
      `str`/`list[str]`
- [x] 3.2 Add `_faction_card`/`_threshold_card`/`_chapter_card`/`_region_card` builders in
      `context_builder.py`
- [x] 3.3 Filter `truth_unlocked` to progressive reveals whose chapter is ≤ the current beat's
      chapter; never construct a field carrying `truth.hidden`
- [x] 3.4 Add the new cards to the never-trimmed priority tier alongside `character_cards`
- [x] 3.5 Extend `tests/test_context_builder.py`: new cards present, hidden truth never appears
      in any built `SessionDocument`, `current_beat` still recoverable via `parse_model_json`
      (guards the field-order constraint)

## 4. Prompts, agents, guardrail wiring

- [x] 4.1 Add `_GMUD_WORLD_CLAUSE` (factions + 陣營值門檻表 + regions + truth layers + endings)
      to `tasks.py`, shared by `make_writer_task` and `make_extract_task`
- [x] 4.2 Add `_CHOICE_DESIGN_CLAUSE` (三原則 + the guide's 錯誤示範/正確示範 pair), shared by
      `make_writer_task` and `make_scene_write_task`
- [x] 4.3 Fold `make_extract_task`'s inlined RPG-entities re-wording back onto the shared
      `_RPG_ENTITIES_CLAUSE`
- [x] 4.4 Extend `make_beat_expand_task` to request `hook`/`converge_event_id` per chapter and
      `scene_kind` per beat, and wire its first guardrail (chapter hook + convergence + scene mix)
- [x] 4.5 Extend `make_scene_write_task` to request `scene_kind`, `checks`, `clue_ids`, and the
      new `Branch` cost/payoff/convergence fields
- [x] 4.6 Update agent backstories (`agents.py`) for the writer/extractor/beat-expander/
      scene-writer roles to reflect the GMUD frame
- [x] 4.7 Extend `tests/test_guardrail_wiring.py` for the new beat-expand guardrail
- [x] 4.8 Update `length.py`: add `scene_mix` to `PRESETS`, `_derive_from_events`,
      `LengthSpec.targets`/`canonical`, and `_CUSTOM_FIELDS`
- [x] 4.9 Deliberately update `tests/test_script_length.py`'s byte-for-byte prompt regression
      guards for the `short` preset's new scene-mix clause

## 5. Orchestrator, pipeline, causal, metrics, FakeLLM

- [x] 5.1 Update `_assemble_script` to copy the new `ExtractionResult` fields into `Script`, and
      copy `outline.chapters` into `script.chapters` with `event_ids` backfilled from the
      beat→event map
- [x] 5.2 Bump `_SCHEMA_VERSION` 1 → 2; make `load_checkpoint` return `None` on a version mismatch
      instead of validating a stale payload
- [x] 5.3 Extend `load_scene_context` to also return a quest map (fixes the UI gap noted in
      design.md/proposal.md)
- [x] 5.4 Extend `causal.py::event_to_node` to fold `Branch.effect_ops` into postconditions
      (additive union with the existing `effects`-text source) and `SkillCheck` outcomes into
      edges
- [x] 5.5 Extend `_repair`'s prompt in `pipeline.py` to name the newly-checkable reference classes,
      not just `dialogue.npc_id`/`next_event_id`
- [x] 5.6 Update `FakeLLM` to emit the new frame (factions/regions/truth/chapters/clues/checks/
      endings) so offline tests exercise real shapes, and fix the pre-existing gap where fake
      `ExtractionResult` omits `player`/`items`/`quests`
- [x] 5.7 Add GMUD-shape structural metrics to `metrics.py`: `branches_with_cost_pct`,
      `branches_with_payoff_pct`, `checks_with_fallback_pct`, `main_scene_ratio`,
      `events_with_clue_pct`, `chapters_with_convergence_pct`, `stat_threshold_coverage_pct`,
      `faction_count`, `ending_count`
- [x] 5.8 Extend `tests/test_orchestrator.py`, `tests/test_orchestrator_parallel.py`,
      `tests/test_crew_pipeline.py`, `tests/test_crew_layered_pipeline.py`,
      `tests/test_causal_consistency.py`, `tests/test_metrics.py` for all of the above

## 6. review.py + ui/app.py

- [x] 6.1 Add `chapter_names()`, `clue_names()`, `faction_names()`, `region_names()` resolvers to
      `review.py`; add the missing `quest_names` to `__all__`
- [x] 6.2 Add UI tabs to `_render_script`: 章節／勢力／地圖／線索／真相／結局／門檻表
- [x] 6.3 Extend `_render_event` to show `scene_kind`, `checks` (with failure route), `clue_ids`,
      and the branch 代價／延遲回收／收斂節點 fields
- [x] 6.4 Fix `_render_batch_confirmation`'s missing `quests` argument to `_render_event`
- [x] 6.5 Resolve `starting_items`/`acquired_in_event_id`/`giver_npc_id`/etc. to names using the
      resolver maps already in scope, instead of rendering raw ids
- [x] 6.6 Extend `tests/test_review.py` for the new resolvers

## 7. Docs and verification

- [x] 7.1 Add a "GMUD script frame" section to `CLAUDE.md` documenting the new schema and the
      checkpoint-compat note; update `docs/DESIGN_NOTES.md`'s methodology section
- [x] 7.2 Run `pytest tests/ -q` and `ruff check .`
- [x] 7.3 Run `python tests/test_chunking.py` to confirm the RAG pipeline is unaffected
- [x] 7.4 Run a `LLM_BACKEND=fake --pipeline-mode layered` smoke generation and assert the new
      frame fields are populated in the output JSON
- [x] 7.5 Run one real generation (`scripts/eval_generation.py --variants flash-glm-prose
      --max-requirements 1 --pipeline-mode layered`) and check the new metrics in
      `out/generation_runs.jsonl` to confirm the shape actually lands with a live model, watching
      for `LLM_MAX_TOKENS` truncation given the larger schema
- [x] 7.6 Browse the generated script in `.venv/bin/streamlit run ui/app.py` to confirm the new
      tabs render correctly
