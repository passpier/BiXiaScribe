## Why

Measured against a GMUD (通用劇本框架) authoring guide the user supplied, `Script`
(`src/bixiascribe/schema.py`) still can't express most of what that guide treats as mandatory for a
playable 武俠 RPG script: faction/陣營值 threshold rules, region/地圖 structure, three-layer 真相
disclosure, chapter-level hooks and convergence points, clues, skill checks with failure fallbacks,
and endings. It also can't express the guide's most-often-wrong authoring mistake — 假選擇
(choices that differ only in degree, not in real tradeoffs) — because `Branch` has no `cost` or
`payoff` fields to check against. Nothing downstream (prompts, guardrails, metrics) can enforce or
even measure these because the schema has no field for them to hang off of.

## What Changes

- Add a full GMUD frame to `Script`: `Faction`/`FactionRelation`, `StatThreshold` (陣營值/數值門檻
  表), `Region`/`SubLocation`, `TruthLayer`/`ProgressiveReveal` (公開／逐步得知／私藏), `Clue`,
  `Chapter` (hook + 收斂節點), `SkillCheck` (含失敗替代路線), `Ending`.
- Extend `Branch` with 抉擇點 fields (`cost`, `immediate_feedback`, `payoff_chapter_id`,
  `payoff_description`, `converges_to_event_id`); extend `Event` with `chapter_id`, `scene_kind`
  (主要/調味), `region_id`, `sub_location_id`, `checks`, `clue_ids`; extend `NPC` with
  `faction_id`, `surface_motive`/`true_motive`; extend `PlayerCharacter` with `origin`,
  `weakness`, `token_item_id`, `relation_to_core_event`.
- **BREAKING**: rename `ChapterOutline` to `Chapter` and unify it with the new `Script.chapters`
  field (`Outline.chapters: list[Chapter]`) — today's layered pipeline builds chapters and
  discards them at assembly time; this makes them survive into the final `Script`.
- Extend `validate_references()` to cross-check every new id class; add
  `validate_stat_thresholds()` and `validate_truth_layering()` as new, separate validators (kept
  out of the LLM repair loop, same pattern as `validate_npc_introductions()`).
- Add nine new offline guardrail checks (`crew/guardrails.py`) mechanically enforcing the guide's
  五、品質檢查清單 (假選擇 detection, delayed-payoff annotation, stat narrative-meaning coverage,
  truth-layer pacing, chapter convergence, skill-check failure fallback, scene information density,
  main/flavor scene mix, region structure), wired as CrewAI task guardrails including a first
  guardrail on the beat-expand task.
- Extend prompts (`crew/tasks.py`) to ask for the new frame and to teach 抉擇點設計三原則 with the
  guide's own 錯誤示範/正確示範 pair; extend `SessionDocument`/`context_builder.py` so a scene
  prompt only ever sees 真相 already unlocked by its own chapter — 提前爆料私藏真相 becomes
  structurally impossible, not just checked.
- Update every downstream consumer: `orchestrator.py` (`_assemble_script`, checkpoint schema
  version bump), `causal.py` (fold `effect_ops`/`SkillCheck` into the causal graph), `metrics.py`
  (new structural GMUD-shape metrics), `pipeline.py` (repair prompt coverage), `llm.py`
  (`FakeLLM` shape), `review.py` (new id resolvers), `ui/app.py` (new tabs, plus two pre-existing
  rendering bugs found during this audit — a missing `quests` arg in the batch-confirmation
  panel, and several id fields rendered unresolved when the resolver map is already in scope).

## Capabilities

### New Capabilities
- `script-frame`: the GMUD-style structural frame a generated 武俠 RPG script must carry —
  factions, stat thresholds, regions, truth layers, chapters, clues, skill checks, endings, and
  the 抉擇點 (cost/payoff/convergence) shape on every branch — plus the guardrails and validators
  that enforce it.

### Modified Capabilities
- `script-length`: the `main`:`flavor` scene-mix ratio (guide: 主要場景略多於調味場景, ~3:2)
  becomes a fourth resolved target alongside `events`/`chapters`/`beats_per_chapter`/
  `min_dialogue`, threaded through `LengthSpec.targets` the same way the existing four are.

## Impact

- **Code**: `src/bixiascribe/schema.py` (new/extended models, new validators),
  `src/bixiascribe/crew/guardrails.py` (nine new checks), `src/bixiascribe/crew/tasks.py` (prompt
  clauses, new guardrail wiring), `src/bixiascribe/crew/context_builder.py` (new session cards,
  truth-layer filtering), `src/bixiascribe/crew/orchestrator.py` (`_assemble_script`, schema
  version), `src/bixiascribe/crew/causal.py`, `src/bixiascribe/crew/metrics.py`,
  `src/bixiascribe/crew/pipeline.py` (repair prompt), `src/bixiascribe/llm.py` (`FakeLLM`),
  `src/bixiascribe/length.py` (`scene_mix` target), `src/bixiascribe/review.py`, `ui/app.py`.
- **Data**: every new field is additive/defaulted, so existing `out/eval/*.json` scripts keep
  parsing. `.bixia_state/` checkpoints are deliberately invalidated by a schema-version bump — an
  in-flight layered run restarts rather than resuming into a half-old shape.
- **Tests**: `tests/test_rpg_schema.py`, `tests/test_schema_layered.py`, `tests/test_guardrails.py`,
  `tests/test_guardrail_wiring.py`, `tests/test_context_builder.py`, `tests/test_orchestrator*.py`,
  `tests/test_crew_*.py`, `tests/test_metrics.py`, `tests/test_causal_consistency.py`,
  `tests/test_review.py`, `tests/test_script_length.py` (deliberate byte-for-byte prompt-text
  update).
- **Docs**: `CLAUDE.md` gains a "GMUD script frame" section documenting the new schema and the
  checkpoint-compat note; `docs/DESIGN_NOTES.md` gets the methodology update.
- **Cost**: a larger schema means larger per-response JSON (watch `LLM_MAX_TOKENS` for silent
  truncation) and new guardrail retries cost real tokens — worth an eval A/B against a `baseline`
  variant.
