## Why

`Script` (`src/bixiascribe/schema.py`) is the product of two additive passes — the original RPG
gameplay entities (`variables`/`effect_ops`/`triggers`) plus the 2026-08-20 GMUD frame
(`stat_thresholds`/`FactionRelation`/`ProgressiveReveal`/`SkillCheck`/`scene_kind`/
`converge_event_id`). The 2026-08-21 slimming pass only removed concepts the 武俠單人劇本生成範例
guide has no equivalent for (Region/Quest); it never revisited the general-purpose abstractions
(`variables`, per-branch `effect_ops`, a threshold table for a "唯一數值" system) that MVP doesn't
need. `crewai.utilities.converter.ensure_all_properties_required()` makes every schema field
wire-required regardless of `schema.py`'s own defaults, so every one of these mostly-empty fields
is restated by the model on every `Event`/`Branch`/`NPC` — directly on the critical path measured in
`openspec/changes/archive/2026-08-21-profile-layered-pipeline-cost/design.md`'s 證據七
(`call_elapsed_s / elapsed_s ≈ 99.99%`, ~71% of completion tokens are reasoning tokens).

Measured against `武俠劇本資料庫Schema設計.md` (a flat, ID-referenced schema designed for an actual
game engine, not a nested outline document), adopting its shape cuts the wire schema
`Script` 13.6KB → 8.0KB (−41%), `Event` 4.0KB → 2.4KB (−39%), `ExtractionResult` 8.5KB → 4.8KB
(−44%).

## What Changes

- **BREAKING**: Rewrite `Script` and every nested model to the flat/ID-referenced shape: `Meta`
  (title/theme/goal/tone, replacing 5 top-level fields), `Stat` (single value, replacing
  `PlayerCharacter.stats: list[Variable]` and `StatThreshold`), `Player` (origin/flaw/token, no
  id), `Faction` (single `motive`, no `relations`), `NPC` (adds `role`, drops
  `first_appearance_event_id`/`identity`/`surface_motive`; keeps `personality`/`speech_style` for
  dialogue-agent register), `Truth` (`revealed: list[str]` replacing `ProgressiveReveal`), `Item`/
  `Clue` (`from_event`), `Chapter` (`start_event` replacing `event_ids`/`converge_event_id`, adds
  `loc`), `DialogueLine` (drops `emotion`), `Check` (single object replacing `list[SkillCheck]`),
  `Choice` (replaces `Branch`: `delta: int` replaces `effect_ops`, `payoff_at` replaces
  `payoff_description`/`converges_to_event_id`, drops `immediate_feedback`), `Ending`
  (`min`/`max` replacing `stat_conditions`/`required_branch_ids`).
- **Deleted outright**: `Variable`, `EffectOp`, `Trigger`, `FactionRelation`, `StatThreshold`,
  `StatCondition`, `ProgressiveReveal`.
- **Kept beyond the guide's literal shape** (pipeline-load-bearing, not new scope): `Event.summary`/
  `.title` (scene-to-scene memory via `context_builder.py`), `Choice.effects: str` (the sole
  postcondition source for `causal.py`'s causal-consistency graph — dropping it alongside
  `effect_ops` would silently disable `check_scene_consistency()`), `Chapter.summary`.
  `Event.triggers` becomes `Event.preconditions: list[str]` rather than being deleted, for the
  same causal-consistency reason (precondition side).
- Extend `validate_references()` for the new id-reference shape; delete
  `validate_stat_thresholds()`, `validate_npc_introductions()`, `validate_truth_layering()` (their
  backing fields no longer exist).
- Delete five guardrail checks whose backing fields are gone (`check_delayed_payoff`,
  `check_stat_narrative`, `check_single_stat`, `check_scene_mix`, `check_convergence`); rewrite
  four to the new fields (`check_choice_quality`, `check_check_fallback`,
  `check_scene_information`, `check_beat_expand_rpg`); fix the pre-existing contradiction where
  `check_script_rpg`/`check_extraction_rpg` demanded `stats >= 2` while `check_single_stat` (report-
  only) demanded exactly 1 — moot once `stat` is a single object, not a list.
- Rewrite every prompt clause in `crew/tasks.py`/`crew/agents.py` that names a removed/renamed
  field, including `pipeline.py::_repair`'s cross-reference checklist (the densest field-name
  listing in the repo); prune three already-dead prompt clauses with no schema backing at all
  (`branch_candidates`, 任務/quests, 地區/regions — leftover from before the 2026-08-21 pass).
- Update every downstream consumer: `causal.py` (precondition/postcondition sources, edge
  construction), `context_builder.py`/`SessionDocument` (drop `threshold_card`, `truth_unlocked`
  keyed by array index instead of `reveal_chapter_id`), `orchestrator.py` (`_assemble_script`,
  `_SCHEMA_VERSION` 3→4), `normalize.py` (shrunk `next_event_id` backfill chain), `metrics.py`
  (drop GMUD-shape metrics tied to removed fields), `llm.py` (`FakeLLM` fixtures), `review.py`,
  `ui/app.py` (drop 變數/門檻表 tabs, rewrite event/branch rendering), `README.md`/`README.en.md`'s
  `script.json` examples.
- Remove the `flash-glm-prose` eval variant (`eval/model_variants.json`) — confirmed non-viable
  (`z-ai/glm-5.2` doesn't support structured JSON schema output, wraps JSON in a ```json fence that
  hangs crewai's `output_pydantic` parser) — and its corresponding note in CLAUDE.md's Known
  limitations.

## Capabilities

### Modified Capabilities
- `script-frame`: replaces the GMUD structural frame (factions/relations, stat-threshold table,
  three-layer progressive truth, chapter convergence, skill-check list, branch cost/feedback/
  payoff/convergence) with the flat/ID-referenced shape — single faction motive, single player
  stat, ordered truth-reveal list, chapter start-event chaining, single per-event check with
  pass/fail routing, per-choice numeric delta with payoff-chapter annotation.
- `script-length`: drops the `scene_mix` (main:flavor ratio) target — `Event.scene_kind` no
  longer exists to measure it against.

## Impact

- **Code**: `src/bixiascribe/schema.py` (full rewrite), `src/bixiascribe/crew/causal.py`,
  `src/bixiascribe/crew/guardrails.py`, `src/bixiascribe/crew/tasks.py`,
  `src/bixiascribe/crew/agents.py`, `src/bixiascribe/crew/context_builder.py`,
  `src/bixiascribe/crew/orchestrator.py`, `src/bixiascribe/crew/normalize.py`,
  `src/bixiascribe/crew/metrics.py`, `src/bixiascribe/crew/pipeline.py` (repair prompt),
  `src/bixiascribe/llm.py` (`FakeLLM`), `src/bixiascribe/length.py` (drop `scene_mix`),
  `src/bixiascribe/review.py`, `ui/app.py`, `eval/model_variants.json`.
- **Data**: `_SCHEMA_VERSION` 3→4 invalidates in-flight `.bixia_state/` checkpoints (restart, no
  migration, same convention as the 1→2 and 2→3 bumps). This is a breaking rename, not just field
  deletion — `meta` is a new required field and `DialogueLine.npc` replaces the required
  `npc_id` — so existing `out/eval/*.json` (12 files) and completed `.bixia_state/*/script.json`
  checkpoints fail `Script.model_validate()` outright rather than degrading field-by-field.
  `ui/app.py::_load()` and `review.py::overview_rows()` already wrap every `load_script()` call in
  try/except for exactly this class of failure, so this surfaces as "無法讀取此劇本檔案" in the
  UI rather than a crash; no migration/backfill is performed.
- **Tests**: `tests/test_rpg_schema.py`, `tests/test_guardrails.py`, `tests/test_metrics.py`,
  `tests/test_normalize.py`, `tests/test_context_builder.py`, `tests/test_causal_consistency.py`,
  `tests/test_schema_layered.py`, `tests/test_orchestrator.py`,
  `tests/test_orchestrator_parallel.py`, `tests/test_crew_pipeline.py`,
  `tests/test_crew_layered_pipeline.py`, `tests/test_guardrail_wiring.py`,
  `tests/test_script_length.py` (deliberate byte-for-byte prompt-text update),
  `tests/test_review.py` (raw-dict fixtures + hardcoded `schema_version: 2` → 4),
  `tests/test_eval_matrix_docs.py` (variant removal).
- **Docs**: `README.md`/`README.en.md`'s `script.json` example blocks (must stay in lockstep),
  `openspec/specs/script-frame/spec.md` and `openspec/specs/script-length/spec.md` (also fixes
  pre-existing doc drift: the current spec still describes Region/sub-location and item-bypass
  skill checks that the 2026-08-21 pass already removed from code without updating this spec),
  `.claude/skills/gmud-schema-internals/SKILL.md` (new Phase 4 section), `docs/DESIGN_NOTES.md`
  (new Phase 4 section, same write-up convention as Phase 2/3), `CLAUDE.md` (drop the
  `flash-glm-prose`/glm-5.2 known-limitation line).
- **Cost**: this is the change whose payoff Phase 2 left unverified — see this change's own
  `design.md` for the paid A/B plan against the `flash-only` variant (the only variant that
  currently passes and the one production actually runs), not `flash-glm-prose` (confirmed
  non-viable, removed by this change) and not `deepseek-v4-pro`/`baseline` (different model tier,
  not a valid before/after comparison for this change).
