## Context

See `proposal.md` - Why. This document covers how the GMUD frame is implemented across
`src/bixiascribe/schema.py` and every module that reads/writes `Script`/`ExtractionResult`/
`SessionDocument`. Two structural constraints already in the codebase shape every decision below:

- `SessionDocument.current_beat` must remain the **last** declared field, and every other field
  must be `str`/`list[str]` (never a nested model) — `llm.py::FakeLLM`'s scene_writer branch
  recovers the target beat via `schema.parse_model_json`, which keeps the *last* JSON object in
  the prompt text that validates as `Beat`. A nested model field earlier in the class could
  accidentally match instead. `tests/test_context_builder.py:240-252` guards this.
- Both repair loops (`pipeline.py::_repair` for legacy, `orchestrator.py`'s proofread tail for
  layered) call `schema.validate_references()` and trust whatever it returns. Extending that one
  function extends both loops for free — the pattern the earlier RPG-shape work already relies on,
  and this change reuses it rather than inventing a second validation entry point.

## Goals / Non-Goals

**Goals:**
- Every structural concept the GMUD guide calls mandatory (factions, stat thresholds, regions,
  truth layers, chapters, clues, checks, endings, choice cost/payoff/convergence) has a schema
  field, is asked for in the generation prompts, and is checkable offline.
- 提前爆料私藏真相 is prevented structurally (hidden facts never enter a scene prompt before their
  reveal point), not just flagged after the fact.
- Every existing `out/eval/*.json` script and every existing test keeps working unless this design
  explicitly calls out a breaking change.

**Non-Goals:**
- Not building an RPG Maker exporter or an editing/save-back UI — both are already out of scope
  per `CLAUDE.md`, and the GMUD frame doesn't change that.
- Not adding a fourth pipeline mode or changing `PIPELINE_MODE`'s legacy/layered split.
- Not making the new fields `Literal`-typed — they follow the existing `Variable.kind`/
  `EffectOp.target_kind` degrade-not-crash convention (a model filling in an unexpected string
  should not crash the run).
- Not attempting semantic truth-leak detection beyond substring/keyword matching in
  `check_truth_pacing` — a full NLI-style check is out of scope; the structural prevention (hidden
  facts never entering the prompt) is the primary defense, the checker is a backstop.

## Decisions

**Extend `Script`/`ExtractionResult` in place rather than introducing a new top-level model.**
Alternative considered: a separate `GmudFrame` model nested under `Script`. Rejected because it
would require every consumer (`orchestrator._assemble_script`, `review.py`, `ui/app.py`,
`metrics.py`) to learn a second traversal path for what is conceptually still "the script," and it
would give `validate_references()` two objects to walk instead of one. Flat fields on `Script`
keep the single-source-of-truth property the module docstring already claims.

**Rename `ChapterOutline` → `Chapter` and unify with `Script.chapters` (the one breaking change).**
Today `_assemble_script` builds chapters during beat expansion and discards everything but
`outline.title`/`.premise` at assembly time — the chapters simply vanish from the final `Script`.
Keeping two chapter-shaped models (`ChapterOutline` for the checkpoint, a new one for the final
output) would let them drift, the same drift already observed between `_RPG_ENTITIES_CLAUSE` and
the extract task's inlined re-wording. Unifying means `_assemble_script` copies `outline.chapters`
straight across (backfilling `event_ids` from the beat→event map it already computes), and there
is exactly one chapter shape to keep in sync with prompts/guardrails/UI. This is the one place the
user's "allow breaking renames" answer is spent — every other new field is additive.

**Keep `validate_stat_thresholds()` and `validate_truth_layering()` separate from
`validate_references()`**, following the existing `validate_npc_introductions()` precedent.
`validate_references()` feeds both LLM repair loops, which assume everything they see is worth a
repair pass. Threshold coverage and truth-pacing are narrative-quality judgments, not structural
dangling-id bugs — sending them into a repair loop that retries against an LLM would burn tokens
on problems a repair pass may not reliably fix. They're consumed by the guardrails module instead,
same routing as NPC introduction checks.

**Prevent premature truth disclosure structurally, not just by a checker.** `SessionDocument`
gains `truth_unlocked` (progressive reveals whose chapter is ≤ the current beat's chapter) and
never gains a field carrying `truth.hidden` at all — a scene-writer call has no way to see a
hidden fact regardless of prompt-following. `check_truth_pacing` in guardrails remains as a
backstop against a model restating a hidden fact's substance in its own words from public/
progressive context, which structural exclusion can't catch.

**`Branch.effects`/`.effect_ops` stay untouched; `causal.py::event_to_node` gains a second,
additive postcondition source.** `causal.py` currently derives postconditions only from
`Branch.effects` (free text) even though `validate_references()` checks the structured
`effect_ops`. Rather than switching `event_to_node` over to `effect_ops` (which could silently
drop postconditions on any script written before this change, since old scripts may only have
`effects` populated), it reads both and unions them. Backward compatible, and it closes the gap
noted during the schema audit where structured effect data was invisible to causal consistency
checking.

**Checkpoint schema version bump (1 → 2) invalidates in-flight `.bixia_state/` runs rather than
attempting a migration.** `orchestrator.load_checkpoint` already swallows `ValidationError` into
"stage not done" and `_SCHEMA_VERSION` is already written to every checkpoint but never compared
on load. Writing a real migration for mid-generation checkpoint state (extraction/beats/scenes
possibly missing the new fields entirely) is materially more complex than restarting the run, and
`orchestrator.py`'s own `detect_stage()` docstring already treats "always re-derive from disk" as
the invariant to preserve. A restarted run is the honest behavior; the alternative (validating a
v1 payload as if it were v2) would silently accept checkpoints missing required new structure.

**`scene_mix` becomes a fifth `length.py` target, threaded exactly like the existing four**, rather
than a separate config knob. The guide expresses scene balance as a ratio *of the same script* the
other four targets already describe (event/chapter/beat/dialogue counts), so it belongs in the
same `LengthSpec`/`_length_target()` machinery `SCRIPT_LENGTH`/`--script-length`/`Variant
.script_length` already resolve through, rather than inventing a second per-run setting with its
own resolution order.

## Risks / Trade-offs

- **[Risk] A much larger `Script`/`Event`/`ExtractionResult` JSON per LLM response increases the
  chance of provider-side truncation, which today silently produces an invalid partial JSON that
  `_coerce_script`'s raw-scan salvage may or may not recover.** → Mitigation: call out
  `LLM_MAX_TOKENS` explicitly in the verification steps below and in the `CLAUDE.md` update;
  measure actual JSON size against a real model before considering this done.
- **[Risk] Nine new guardrail checks × `GUARDRAIL_MAX_RETRIES` retries is a real token-cost
  increase**, on top of the size increase above. → Mitigation: `metrics.py` gains the GMUD-shape
  metrics specifically so `eval_generation.py` can A/B the cost delta against a `baseline` variant
  before this becomes the default expectation for every run.
- **[Risk] The `choice_quality` 假選擇 heuristic (text-overlap + shared effect targets + no
  distinct cost) is a heuristic, not a semantic judgment** — it can both under- and over-flag.
  → Mitigation: guardrail feedback is advisory (retries the task with feedback, doesn't hard-fail
  the run), and `check_script_rpg`'s existing pattern of returning human-readable Chinese problem
  strings means a false positive is at worst a wasted retry, not a blocked run.
- **[Risk] `ChapterOutline` → `Chapter` rename breaks any code or checkpoint referencing the old
  name.** → Mitigation: it's an additive-shape rename (same fields plus more), so old
  `beats.json` checkpoints fail to validate under the new required-field set anyway once
  `_SCHEMA_VERSION` is bumped — they're already being invalidated for the reasons above, so the
  rename doesn't add a second compatibility break on top of that one.
- **[Trade-off] Keeping `stat_thresholds`/`truth_layering` validators separate from
  `validate_references()` means the layered orchestrator's proofread tail and the legacy repair
  loop never automatically see these problems** — they only surface via guardrails during
  generation, not as a final safety net after assembly. → Accepted: this mirrors
  `validate_npc_introductions()`'s existing scope, and adding a second post-assembly check pass
  is a larger change than this proposal's scope; worth a follow-up if guardrail coverage proves
  insufficient in practice.

## Migration Plan

1. Schema + validators land first (additive, no prompt changes yet) — every existing test and
   on-disk script keeps passing/parsing at this point, proving the additive claim before anything
   downstream depends on it.
2. Guardrails land next, pure/offline, testable with zero tokens.
3. Context builder + prompts + agent wiring land together, since the truth-layer filtering and the
   prompts that reference the new frame are two halves of one behavior.
4. Orchestrator/pipeline/causal/metrics/FakeLLM land together — this is where the checkpoint
   version bump takes effect, so it's staged after the schema shape it's protecting is final.
5. review.py/ui/app.py land last, once the shape they're rendering is stable.
6. No feature flag: unlike `PIPELINE_MODE`'s legacy/layered split (a deliberate zero-cost rollback
   lever), this schema change has no equivalent "old frame" mode to fall back to — rollback is a
   git revert, matching how the earlier RPG-shape addition (`player`/`items`/`quests`) shipped
   without its own toggle.

Full staged task breakdown is in `tasks.md`.
