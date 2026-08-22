## 1. `src/bixiascribe/schema.py` — rewrite core models and validators

- [x] 1.1 Replace `Variable`/`Faction`/`FactionRelation`/`NPC`/`TruthLayer`/`ProgressiveReveal`/
      `Clue`/`SkillCheck`/`StatCondition`/`Ending`/`Branch`/`Event`/`PlayerCharacter`/`Item`/
      `Script` with `Meta`/`Stat`/`Player`/`Faction`/`NPC`/`Truth`/`Item`/`Clue`/`Chapter`/
      `DialogueLine`/`Check`/`Choice`/`Event`/`Ending`/`Script` per the design.md target shape.
      Delete `Variable`, `EffectOp`, `Trigger`, `FactionRelation`, `StatThreshold`,
      `StatCondition`, `ProgressiveReveal` outright.
- [x] 1.2 Update `Chapter`/`Outline`/`Beat`/`BeatSheet`/`ExtractionResult` (layered models) to the
      new shape: `Beat` drops `scene_kind`; `ExtractionResult` gains `meta`/`stat`/`player`, drops
      `variables`/`stat_thresholds`.
- [x] 1.3 Rewrite `validate_references()` for the new id-reference graph: `dialogue.npc`,
      `choices[].next`, `check.on_pass`/`on_fail`, `event.chapter_id`/`clue_ids`/`npc_ids`,
      `chapter.start_event`, `item.from_event`, `clue.from_event`, `choice.payoff_at` (chapter
      id), `npc.faction_id`.
- [x] 1.4 Delete `validate_stat_thresholds()`, `validate_npc_introductions()`,
      `validate_truth_layering()` (backing fields gone).
- [x] 1.5 Update `SessionDocument` to drop `threshold_card`; keep the "all str/list[str] fields
      before `current_beat`" ordering constraint.
- [x] 1.6 `PlotNode`/`PlotEdge`/`CausalPlotGraph`/`validate_causal_graph`/`validate_outline_beats`
      unaffected — leave as-is, confirm no reference to deleted fields.

## 2. `src/bixiascribe/crew/causal.py` — precondition/postcondition sources

- [x] 2.1 `event_to_node()`: preconditions from `event.preconditions` (was
      `event.triggers[*].condition`); postconditions from `choice.effects` only.
- [x] 2.2 Delete `_effect_op_texts()`.
- [x] 2.3 `build_graph()`: branch-flow edges from `choice.next`; replace the `event.checks` loop
      with a single `event.check.on_pass`/`.on_fail` pair of edges.
- [x] 2.4 `check_scene_consistency()`: read `candidate.preconditions` directly (no more
      `parse_fact(_field(t, "condition"))` over `triggers`).
- [x] 2.5 `repair_scene_task()` prompt text: "triggers" → "preconditions" throughout; keep the
      "don't add/remove branch/trigger count" constraint, reworded to choices/preconditions.

## 3. `src/bixiascribe/crew/guardrails.py` — delete, rewrite, fix the stats contradiction

- [x] 3.1 Delete `check_delayed_payoff`, `check_stat_narrative`, `check_single_stat`,
      `check_scene_mix`, `check_convergence`.
- [x] 3.2 Rewrite `check_choice_quality`: 假選擇 detection via `choice.text` similarity + same-sign
      `delta` overlap (was `effect_ops` target-id set overlap).
- [x] 3.3 Rewrite `check_check_fallback`: read the single `event.check.on_fail`/`.fail_cost`
      instead of iterating `event.checks`.
- [x] 3.4 Rewrite `check_scene_information`: an event with no `clue_ids` and no choice with
      non-empty `effects` is content-free.
- [x] 3.5 `check_extraction_rpg`/`check_script_rpg`: replace the `len(player.stats) >= 2` check
      with `script.stat is not None` — this also resolves the pre-existing contradiction against
      the now-deleted `check_single_stat`.
- [x] 3.6 `check_beat_expand_rpg`: drop the hook/converge_event_id/scene_kind checks; keep
      chapter-coverage checking only.
- [x] 3.7 Add one new report-only check for ending range overlap/coverage (design.md's mitigation
      for the lost `stat_thresholds` guarantee) — place near `check_choice_quality` or as its own
      function, fold into `collect_quality_problems()`.
- [x] 3.8 Add a `preconditions` non-empty check to `check_scene_rpg` (design.md's mitigation for
      `list[str]` being easier to leave empty than `list[Trigger]` was) — report-only unless
      offline testing shows it needs to be a hard guardrail.
- [x] 3.9 Update `crew/guardrails.py`'s own module docstring / `collect_quality_problems()`
      comment enumerating "nine checks" to match the new count.

## 4. `src/bixiascribe/crew/tasks.py` / `crew/agents.py` — prompt text

- [x] 4.1 `_RPG_ENTITIES_CLAUSE`: `stats 至少 2 個` → single `stat`; drop
      `first_appearance_event_id` clause; `acquired_in_event_id` → `from_event`.
- [x] 4.2 `_GMUD_WORLD_CLAUSE`: drop the `stat_thresholds` paragraph and `relations`/`stance`
      wording; `truth.progressive`+`reveal_chapter_id` → `truth.revealed` (ordered list);
      `endings` wording → `min`/`max`.
- [x] 4.3 `_CHOICE_DESIGN_CLAUSE`: `effect_ops` → `effects`+`delta`; drop `immediate_feedback`;
      `payoff_description` → `payoff_at` (chapter id); drop the `converges_to_event_id` sentence.
- [x] 4.4 Writer/extract/beat_expand/scene_write task `description`s: drop `variables`, rename
      `triggers` → `preconditions`, drop `scene_kind`, drop `converge_event_id`, `checks` →
      `check`.
- [x] 4.5 Delete dead prompt clauses with no schema backing: `tasks.py`'s `branch_candidates`
      mention, `agents.py`'s 任務(quests) and 地區(regions) mentions.
- [x] 4.6 `crew/pipeline.py::_repair` prompt: rewrite the entire cross-reference checklist to the
      new id-reference graph (mirrors `validate_references()`'s new checks from task 1.3).

## 5. `src/bixiascribe/crew/context_builder.py`

- [x] 5.1 Drop `_threshold_card`/`threshold_card`.
- [x] 5.2 `_faction_card`: read `faction.motive`.
- [x] 5.3 `_player_card`: read `player.name`/`.origin`/`.flaw`/`.token` and `script.stat`
      (id/name/init) instead of `player.stats`.
- [x] 5.4 `_truth_unlocked`: index into `truth.revealed` by array position relative to the
      current beat's chapter position (design's "array order expresses reveal pacing"), not
      `reveal.reveal_chapter_id`. `truth.hidden` (now a plain `str`) still never read by this
      module.
- [x] 5.5 `_chapter_card`: drop `converge_event_id`, keep `loc`/`summary`/`hook`.
- [x] 5.6 Verify `SessionDocument` field-order constraint still holds after 5.1-5.5's field
      changes (all str/list[str] fields before `current_beat`).

## 6. `src/bixiascribe/crew/orchestrator.py` / `normalize.py` / `metrics.py`

- [x] 6.1 `_assemble_script()`: rebuild as `Script(meta=..., stat=..., player=..., ...)`; backfill
      `Chapter.start_event` from the chapter's first beat id (was backfilling `event_ids`).
- [x] 6.2 Bump `_SCHEMA_VERSION` 3 → 4.
- [x] 6.3 `normalize.py::_fix_next_event_ids`: shrink the fallback chain to "next event in
      sequence" only (both `converges_to_event_id` and `chapter.converge_event_id` sources are
      gone); update the note string to accurately describe the single remaining fallback tier —
      do not leave stale wording implying a converge-based backfill was attempted.
- [x] 6.4 `normalize.py::_backfill_missing_chapters`: build `Chapter(id=..., title="")` (no
      `summary` field mismatch).
- [x] 6.5 `metrics.py::script_metrics`: drop the `"variables"` key.
- [x] 6.6 `metrics.py::gmud_metrics`: drop `stat_threshold_coverage_pct`, `main_scene_pct`,
      `converge_declared_pct`; rewrite `branches_with_effects`/`checks_with_fallback_pct` for the
      new `choice.effects`/single `event.check` shape.
- [x] 6.7 `review.py`/JSONL consumers of `script_metrics()` keys: confirm existing
      missing-key-defaults-to-0 convention (`review.py:560,580`-style) covers the dropped keys for
      old logged rows — no code change expected here, just verification.

## 7. `src/bixiascribe/llm.py` — FakeLLM fixtures

- [x] 7.1 Rebuild `_fake_writer_script`, `_fake_extraction`, `_fake_beat_sheet`, `_fake_scene`,
      `_fake_fill_dialogue`, `_fake_proofread` to the new shape, mutually cross-referencing
      consistently (offline tests rely on `validate_references() == []` against these fixtures).
- [x] 7.2 Delete `_fake_factions`/`_fake_truth`/`_fake_stat_thresholds`/`_fake_endings` helper
      functions that build now-deleted sub-models; rebuild the ones still needed (`Faction`,
      `Truth`, `Ending`) inline or as slimmer helpers.
- [x] 7.3 Verify `_extract_script_json()`'s literal `"events"`/`"npcs"` key check (used to decide
      "is this Script-shaped?") still discriminates correctly now that `Script` also has a nested
      `meta` object — confirm `parse_model_json`'s largest-span rule doesn't pick a wrong nested
      match.

## 8. `src/bixiascribe/review.py` / `ui/app.py` / `generation.py` / `scripts/generate_script.py`

- [x] 8.1 `review.py::npc_names()`: drop the `script.player.id` branch (`Player` has no `id`;
      dialogue speaker for the player uses the fixed string `"player"`).
- [x] 8.2 `ui/app.py`: delete `_variable_rows()` and the 變數 tab; delete the 門檻表 tab; rewrite
      `_render_event()` (drop scene_kind/triggers/emotion/effect_ops/immediate_feedback
      rendering, collapse the `checks` loop to a single `check` block); 勢力 tab reads
      `faction.motive`; 真相 tab's progressive section reads `truth.revealed` as a plain string
      list; `script.title` → `script.meta.title` everywhere it's read.
- [x] 8.3 `scripts/generate_script.py`: update any `script.title`/`.premise` access to
      `script.meta.title`/`.theme`.
- [x] 8.4 `generation.py`/`wire.py`: confirm no field-name-specific code needs changes (`wire.py`
      is fully reflective — only its stale docstring field-count numbers need updating).

## 9. Tests

- [x] 9.1 `tests/test_rpg_schema.py`: rewrite all ~30 `validate_references()` fixture/assertion
      pairs for the new id-reference graph.
- [x] 9.2 `tests/test_guardrails.py`: rewrite ~90 constructions for the new models; delete test
      cases for the five removed guardrails; add cases for the two new checks (task 3.7, 3.8).
- [x] 9.3 `tests/test_metrics.py`, `tests/test_normalize.py`, `tests/test_context_builder.py`,
      `tests/test_causal_consistency.py`, `tests/test_schema_layered.py`,
      `tests/test_orchestrator.py`, `tests/test_orchestrator_parallel.py`,
      `tests/test_crew_pipeline.py`, `tests/test_crew_layered_pipeline.py`,
      `tests/test_guardrail_wiring.py`: update fixtures to the new schema.
- [x] 9.4 `tests/test_script_length.py`: deliberate byte-for-byte prompt-text regression update
      to match the rewritten prompt clauses (tasks 4.1-4.5).
- [x] 9.5 `tests/test_review.py`: update the raw-dict `_script_json()` fixture to the new field
      names; bump the hardcoded `schema_version: 2` literals (lines ~447/470/481) to 4.
- [x] 9.6 `tests/test_eval_matrix_docs.py`: confirm it passes after `flash-glm-prose` removal
      (task 10.1).
- [x] 9.7 Run `pytest tests/` and `ruff check .` clean.

## 10. Eval variant cleanup

- [x] 10.1 Remove the `flash-glm-prose` entry from `eval/model_variants.json` (confirmed
      non-viable: `z-ai/glm-5.2` doesn't support structured JSON schema output).
- [x] 10.2 Remove the corresponding glm-5.2/markdown-fence note from CLAUDE.md's "Known
      limitations" section.

## 11. Docs

- [x] 11.1 `README.md` (`## 輸出格式`, ~lines 122-163) and `README.en.md` (`## Output format`,
      ~lines 140-181): rewrite the `script.json` example blocks to the new shape, keeping both
      files' blocks in lockstep.
- [x] 11.2 `openspec/specs/script-frame/spec.md`: replace with `## MODIFIED Requirements` /
      `## REMOVED Requirements` reflecting the new flat shape; also fix pre-existing drift the
      2026-08-21 pass left behind (Region/sub-location, item-bypass skill checks, payoff-chapter
      wording never got removed from this spec despite the code already dropping them).
- [x] 11.3 `openspec/specs/script-length/spec.md`: remove the `scene_mix` target.
- [x] 11.4 `.claude/skills/gmud-schema-internals/SKILL.md`: add a "Phase 4" section describing
      what changed, following the existing Phase 2/3 write-up convention; correct the guardrail
      list to match the post-deletion count.
- [x] 11.5 `docs/DESIGN_NOTES.md`: add a "Phase 4" section (deleted-fields list, measured wire
      schema numbers, what was deliberately kept and why — same format as Phase 2/3).

## 12. Verification (see design.md's Open Questions for anything deferred)

- [x] 12.1 Offline: `ruff check .`; `pytest tests/`; `LLM_BACKEND=fake` smoke test of both
      `--pipeline-mode legacy` and `--pipeline-mode layered` via `scripts/generate_script.py`,
      confirming `validate_references() == []` on both outputs.
- [x] 12.2 Wire-schema size regression: measure `wire.lenient_mirror(M).model_json_schema()`
      length for `Script`/`Event`/`ExtractionResult`/`BeatSheet` — confirmed: Script 8236,
      Event 2469, ExtractionResult 4990, BeatSheet 1785 (design.md table target was
      Script≈8016/Event≈2445/ExtractionResult≈4794/BeatSheet≈1785).
- [x] 12.3 UI compatibility: verified via `review.load_script()` against all 12 existing
      `out/eval/*.json` scripts. This is a breaking rename (new required `meta`, renamed required
      `DialogueLine.npc`), not just field deletion, so every one of the 12 raises
      `ValidationError` rather than degrading field-by-field — confirmed `ui/app.py::_load()` and
      `review.py::overview_rows()` both already wrap `load_script()` in try/except, so this
      surfaces as "無法讀取此劇本檔案" per record, not a crash. design.md's decision four /
      proposal.md's Impact section were corrected to describe this accurately (originally claimed
      the gentler `extra="ignore"` field-drop behavior, which was wrong for a rename this size).
- [ ] 12.4 Paid A/B (final step, real tokens): `python scripts/eval_generation.py --variants
      flash-only --pipeline-mode layered --script-length long --dry-run`, then a real run with
      `--max-requirements 1`. Compare `out/generation_runs.jsonl`'s `elapsed_s`/`token_usage`/
      `scene_metrics[].call_elapsed_s`/`reasoning_tokens`/`structured_fallbacks`/
      `guardrail_retries` against the `.bixia_state/1787309292-req-d232acf2d8` baseline and
      `out/generation_runs_ui.jsonl`'s `ui-flash-only` row (same model/mode/length). Also check
      `crew/metrics.py::gmud_metrics()` structural coverage didn't collapse — elapsed improving
      alone doesn't validate this change.
