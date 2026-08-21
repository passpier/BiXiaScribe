---
name: gmud-schema-internals
description: Deep internals of BiXiaScribe's GMUD (通用劇本框架) script frame — factions, truth-layer disclosure, chapters/clues/skill-checks/endings, the schema-slimming pass, GMUD guardrails/prompts, wire.py/normalize.py's output-schema-strictness fixes, structured-output truncation handling, and per-scene execution metrics. Use when editing schema.py, wire.py, normalize.py, crew/guardrails.py, crew/execute.py, crew/scene_metrics.py, crew/tasks.py's prompt clauses, or crew/causal.py's postcondition handling.
---

Measured against a GMUD (通用劇本框架) authoring guide, `Script`/`ExtractionResult` (`schema.py`)
carry a second layer of structure on top of the RPG-shape entities (see the
script-generation-internals skill): factions and a 陣營值/數值門檻表, three-layer 真相 disclosure
(公開／逐步得知／私藏), chapters with hooks and convergence points, clues, skill checks with failure
fallbacks, endings, and a 抉擇點 (cost/payoff/convergence) shape on every branch. Every field this adds
is optional/defaulted, so a script produced before this frame existed still loads and validates — only
one rename is breaking: `ChapterOutline` became `Chapter` and unified with `Script.chapters`
(previously, `_assemble_script` built chapters during beat expansion and discarded everything but
`outline.title`/`.premise` at assembly time; `Chapter` now survives into the final `Script`, with
`event_ids` backfilled by grouping `BeatSheet.beats` by `chapter_id` since `Beat.id == Event.id` by
convention).

New models: `Faction`/`FactionRelation` (結盟/敵對/中立/附庸), `StatThreshold` (a stat value range plus
what it unlocks — `unlocks_kind` one of branch/event/npc_attitude/ending, `unlocks_id` the target),
`TruthLayer`/`ProgressiveReveal` (`public`/`progressive`/`hidden`), `Clue`, `Chapter`
(`hook`/`event_ids`/`converge_event_id`), `SkillCheck` (含失敗替代路線 — `failure_branch_id` +
`failure_cost`), `StatCondition`/`Ending`. Extended: `Branch` gains `cost`/`immediate_feedback`/
`payoff_description`/`converges_to_event_id`; `Event` gains `chapter_id`/`scene_kind`
(main/flavor)/`checks`/`clue_ids`; `NPC` gains `faction_id`/`surface_motive`/`true_motive`;
`PlayerCharacter` gains `origin`/`weakness`/`token_item_id`/`relation_to_core_event`; `Beat` gains
`scene_kind` (the beat_expander's own guess at the eventual Event.scene_kind, checkable before any
Event exists — see the beat-expand guardrail below).

`validate_references()` cross-checks every new id class (faction relations, stat-threshold
stat_id/unlocks_id, event chapter_id/clue_ids, skill-check targets, branch converges_to_event_id,
chapter converge_event_id/event_ids, clue found_in_event_id, ending
stat_conditions/required_branch_ids, progressive-reveal reveal_chapter_id/reveal_event_id, player
token_item_id) — both existing repair loops (`pipeline.py::_repair`, the layered orchestrator's
proofread tail) see these for free. `validate_stat_thresholds()` (coverage of every branch-targeted
stat + non-overlapping ranges + every threshold unlocking something) and `validate_truth_layering()`
(progressive reveals in non-decreasing chapter order) are kept **separate** from
`validate_references()`, following the existing `validate_npc_introductions()` precedent —
narrative-quality judgments a repair loop shouldn't be trusted to fix, consumed by the guardrails
module instead, not the LLM repair loops.

## Phase 2: schema slimming against the 武俠單人劇本生成範例 guide

`Region`/`SubLocation`/`Quest` and several purely-annotative fields (`Branch.condition`/
`.payoff_chapter_id`, `SkillCheck.kind`/`.difficulty`/`.item_bypass_id`, `Chapter.beat_ids`/
`.clue_ids`, `NPC.attitude_by_threshold`, `Clue.serves`, `ExtractionResult.props`/
`.branch_candidates`) were deleted outright (not deprecated) — measured against a simpler 單人劇本
authoring guide with no region/quest concept, this shrinks the wire schema `crewai.utilities.
converter.generate_model_description()` sends the provider by ~17-20% (`Script` 16.8KB → 13.6KB,
`Event` 4.9KB → 4.0KB, `ExtractionResult` 10.8KB → 8.5KB) and cuts required fields per object
(`Event` 14→11, `Branch` 11→9, `SkillCheck` 8→5) — see "Output-schema strictness" below for why every
one of those fields is wire-required regardless of `schema.py`'s own defaults. **`Branch.effects` was
deliberately kept**, not dropped alongside `.condition` — `crew/causal.py::event_to_node()` reads it
as the primary `PlotNode.postconditions` source for causal conflict detection; `effect_ops` alone
(rendered as `"target_id：op=value"`) never collides with a trigger-derived precondition, so dropping
`effects` would have silently gutted causal validation. `SkillCheck`'s fallback is now
`failure_branch_id`+`failure_cost` only (no `item_bypass_id` alternative). The guide's 唯一數值
(exactly one player stat, sliced into three non-overlapping `stat_threshold` ranges) is enforced by
prompt wording (`_GMUD_WORLD_CLAUSE` in `crew/tasks.py`) plus a new offline guardrail,
`guardrails.check_single_stat()`, folded into `collect_quality_problems()` (report-only, same as the
other nine narrative-quality checks). `_SCHEMA_VERSION` bumped 2 → 3 in `crew/orchestrator.py` — an
in-flight checkpoint from before this change simply restarts, same "no migration, restart" convention
as the 1 → 2 bump. Existing `out/eval/*.json` and `.bixia_state/` script checkpoints still load
unmodified: `schema.py` sets no `model_config`, so pydantic's default `extra="ignore"` just drops the
now-unknown fields on read.

## Guardrails and prompts

`crew/guardrails.py` gained nine pure/offline checks mechanically enforcing the guide's
五、品質檢查清單: `check_choice_quality` (missing `cost` + 假選擇 text-overlap/shared-effect-target
heuristic), `check_delayed_payoff` (a deferred effect — one with no `immediate_feedback` — must
declare a payoff), `check_stat_narrative` (wraps `validate_stat_thresholds`), `check_truth_pacing`
(hidden-fact substring leak detection up to a given chapter — a backstop, not the primary defense; see
below), `check_convergence` (missing/unreachable chapter `converge_event_id`, >3-branch choice
points), `check_check_fallback` (a `SkillCheck` with no failure branch, or a failure branch with no
`failure_cost`), `check_scene_information` (an event unlocking no clue/item/story-state change),
`check_scene_mix` (flavor scenes outnumbering main scenes). A tenth, `check_beat_expand_rpg`, checks
the beat-expand stage specifically (chapter hook/converge_event_id, main/flavor beat mix) and is
wired onto `make_beat_expand_task` — the first guardrail on that task. All ten follow the same
`GUARDRAILS_ENABLED`/`LLM_BACKEND=fake` wiring convention as the RPG-shape guardrails (see
script-generation-internals). Phase 2 (schema slimming, above) removed `check_regions` (its
region/sub-location structure no longer exists) and added an eleventh, `check_single_stat` (唯一數值:
exactly one player stat, ≥3 non-overlapping thresholds), to this same not-wired-as-a-`Task.guardrail`,
report-only set.

`crew/tasks.py` gained two shared prompt clauses: `_GMUD_WORLD_CLAUSE` (factions + 陣營值門檻表 +
regions + truth layers + endings, injected into both `make_writer_task` and `make_extract_task` — the
extract task's inlined RPG-entities wording was also folded back onto the shared
`_RPG_ENTITIES_CLAUSE` here) and `_CHOICE_DESIGN_CLAUSE` (抉擇點設計三原則 — cost/immediate feedback/
delayed payoff+convergence — plus the guide's own 錯誤示範/正確示範 pair, injected into both
`make_writer_task` and `make_scene_write_task`). `make_beat_expand_task` additionally asks for each
chapter's `hook`/`converge_event_id` (the latter as a placeholder event id, since no Event exists yet
at that stage) and each beat's `scene_kind`, using `length.py`'s new `scene_mix` target for the
main:flavor ratio wording.

## Truth is withheld structurally, not just checked

`SessionDocument` gained `faction_cards`/`threshold_card`/`chapter_card`/`region_card`/`truth_public`/
`truth_unlocked` (all `str`/`list[str]`, declared before `current_beat` — same field-order constraint
`character_cards` etc. already had to respect). `truth_unlocked` is filtered in
`context_builder.py::build_session_document()` to progressive reveals whose chapter is ≤ the current
beat's chapter (via `beat_sheet.outline.chapters`' array order); `TruthLayer.hidden` is never read by
this module at all, so no field on `SessionDocument` can ever carry a hidden fact regardless of
prompt-following — this is what makes 提前爆料私藏真相 structurally impossible, not merely flagged
after the fact. `check_truth_pacing` remains as a backstop against a model restating a hidden fact's
substance in its own words from public/progressive context, which structural exclusion can't catch.

## Downstream consumers

`orchestrator.py::_assemble_script` copies the new `ExtractionResult` fields
(theme/goal/tone/factions/regions/truth/stat_thresholds/clues/endings) into the final `Script`, and
copies `outline.chapters` across with `event_ids` backfilled. `_SCHEMA_VERSION` bumped 1 → 2 — an
in-flight `.bixia_state/<run_id>/` checkpoint from before this change fails to validate against the
new required-field set and simply restarts rather than being migrated (`load_checkpoint` already
treats a version mismatch as "no checkpoint", so this needed no new code, just the bump). This is a
deliberate choice, not an oversight — see `openspec/changes/archive/`'s design doc for the "Checkpoint
schema version bump" decision. `load_scene_context()` now also returns a `quest_id -> name` map (a
pre-existing UI gap: the batch-confirmation panel had been missing quest names for pending scenes).

`causal.py::event_to_node` now folds `Branch.effect_ops` into postconditions as an additive union with
the existing `effects`-text source (rendered as `"target_id：op=value"` fact-shaped strings), and
`build_graph` adds an edge for every `SkillCheck.success_next_event_id` (a `failure_branch_id` already
gets its own edge via the normal branches loop, since it names a `Branch.id` whose own
`next_event_id` is what matters). `pipeline.py::_repair`'s prompt names every newly-checkable
reference class, not just `dialogue.npc_id`/`next_event_id`. `llm.py::FakeLLM`'s fixtures
(`_fake_writer_script`/`_fake_extraction`/`_fake_beat_sheet`/`_fake_scene`) emit a full,
mutually-consistent GMUD frame (factions/regions/truth/stat_thresholds/chapters/clues/checks/endings)
so offline tests exercise the real shape end-to-end — this also fixed a pre-existing gap where the
fake `ExtractionResult` omitted `player`/`items`/`quests` entirely. `crew/metrics.py::gmud_metrics()`
adds nine structural coverage metrics (`branches_with_cost_pct`, `branches_with_payoff_pct`,
`checks_with_fallback_pct`, `main_scene_ratio`, `events_with_clue_pct`,
`chapters_with_convergence_pct`, `stat_threshold_coverage_pct`, `faction_count`, `ending_count`) folded
into `script_metrics()`, so `eval_generation.py` can A/B a variant's GMUD-shape coverage the same way
it already A/Bs RPG-shape/continuity coverage.

`review.py` gained `chapter_names()`/`clue_names()`/`faction_names()`/`region_names()` resolvers
(`region_names()` covers both region ids and sub-location ids in one map, since both resolve to "some
place name" in the UI); `ui/app.py`'s `_render_script` gained 章節／勢力／地圖／線索／真相／結局／
門檻表 tabs, `_render_event` now shows `scene_kind`/`checks` (with resolved failure route)/`clue_ids`
and each branch's 代價／立即回饋／延遲回收／收斂節點, and two pre-existing rendering bugs found
during this audit were fixed in passing: `_render_batch_confirmation`'s missing `quests` argument to
`_render_event`, and several id fields (starting_items, item acquired_in_event_id, quest
giver_npc_id/start_event_id/complete_event_id) that were rendered as raw ids instead of being resolved
to names via the resolver maps already in scope.

## Output-schema strictness and the wire.py/normalize.py pass-rate fixes

**Non-obvious**: `schema.py`'s `default=""` / `default_factory=list` on most fields does not
mean those fields are optional as far as a real model call is concerned. `crewai.utilities.
converter.generate_model_description()` (what actually builds the JSON schema sent to the
provider for `output_pydantic`) runs `ensure_all_properties_required()` on it and sets
`strict: true`/`additionalProperties: false` -- every field becomes required on the wire
regardless of what schema.py says. Measured against `Event`: 4 pydantic-required fields become
14/14 wire-required (Branch 11/11, SkillCheck 8/8); `Script`'s wire schema is ~14.8KB with
18/18 top-level fields required. If the provider's structured-output enforcement is imperfect
(not guaranteed for every OpenRouter route/model) and a field is dropped -- observed against
`deepseek-v4-flash-0731`: a `Branch` missing only `next_event_id`, otherwise complete and
internally consistent (`cost`/`immediate_feedback`/`payoff_description`/
`converges_to_event_id` all filled in) -- `openai.lib._parsing._completions.
parse_chat_completion`'s `model_validate_json` raises `ValidationError` **inside**
`Task.execute_sync()`, before `pipeline.py::_coerce_model`'s three-tier (pydantic -> json_dict
-> raw_scan) rescue ever runs. Against this specific failure mode that rescue logic was dead
code. This is also a meaningful chunk of "why is this slow" -- a long script's Branch/SkillCheck
objects have to restate several always-empty-string fields per branch regardless of content.

`src/bixiascribe/wire.py` (pure, no crewai/config import) is the fix: `lenient_mirror(Model)`
builds a same-shaped model, recursively, where every field (including nested models) has a
default, and every six task factories in `crew/tasks.py` now pass
`output_pydantic=wire.lenient_mirror(X)` instead of `X` itself. `ensure_all_properties_required`
still marks every field required on the wire schema sent to the provider -- this does not shrink
it -- but every one of those wire-required fields now has a schema-legal fallback
(`""`/`0`/`False`/`[]`/a recursively-empty nested model), so a dropped field produces
inspectable empty data instead of a hard `ValidationError`. `wire.to_strict()` converts a filled
mirror instance back to the real strict model. `pipeline.py::_coerce_model` and
`tasks.py::_coerce_for_guardrail` both gained a `wire.lenient_mirror(model_cls)` isinstance
check (source tier `"lenient_mirror"` in `RunReport.coerced_from`) between the existing
`pydantic`/`json_dict` tiers.

`src/bixiascribe/crew/normalize.py` (`normalize_script(script) -> (Script, list[str])`, pure,
same style as `causal.py`/`guardrails.py`) runs mechanical, offline reference repairs on the
now-inspectable-but-incomplete output, before `validate_references()` -- called from both
`pipeline.py::run_pipeline_with_report` and `orchestrator.py::run_layered`, right after a Script
is coerced/assembled. It only fixes what's mechanically inferrable, never fabricates narrative
content: a dangling/empty `Branch.next_event_id` is backfilled from
`converges_to_event_id` -> the branch's chapter's `converge_event_id` -> the next event in
sequence, in that priority order; if `Script.chapters` is empty but multiple events agree on an
undeclared `chapter_id`, a blank `Chapter` skeleton is backfilled for it (title/summary left for
a real repair pass); purely-annotative dangling ids (`region_id`/`sub_location_id`/`clue_ids`/
`branch.payoff_chapter_id`) are just cleared. A dangling `npc_id` in dialogue, or anything else
`validate_references()` checks, is deliberately left untouched -- same "narrative-quality
judgment a mechanical pass shouldn't be trusted to fix" boundary as
`validate_stat_thresholds()`/`validate_truth_layering()`/`validate_npc_introductions()`.
`RunReport.normalize_notes` records what was fixed (empty list = nothing needed fixing).

`SessionDocument` gained `allowed_ids` (closed menu of every legal chapter/region/
sub_location/clue/item/quest id, built by `context_builder.py::_allowed_ids()` from the
beat_sheet's outline + the extraction -- must stay before `current_beat`, same field-order
constraint the other new fields already had to respect). `make_scene_write_task`'s prompt now
says chapter_id/region_id/sub_location_id/clue_ids/payoff_chapter_id/converges_to_event_id must
come from this list or be left blank, not invented -- this addresses `validate_references()`'s
"unknown chapter_id"/"unknown region_id" class of problem at the prompt level, with
`normalize.py`'s chapter-backfill as the safety net for whatever still gets through.

Separately, while diagnosing the above: `make_scene_write_task`'s `check_scene_rpg` guardrail
computed `known_npc_ids` from `session.character_cards` only, never `session.player_card` --
any scene where the player has a dialogue line (`npc_id == player.id`, the normal case) was
guardrail-rejected as "unknown NPC speaking", burning all `GUARDRAIL_MAX_RETRIES` retries before
ever reaching JSON coercion. `validate_references()`'s own `dialogue_target_ids = npc_ids |
player_ids` already treated the player as a valid speaker; only this one guardrail was out of
sync. Fixed by unioning `session.player_card` into `known_npc_ids`.

Nine of the ten GMUD guardrail checks in `crew/guardrails.py`
(`check_choice_quality`/`check_delayed_payoff`/`check_stat_narrative`/`check_truth_pacing`/
`check_convergence`/`check_check_fallback`/`check_scene_information`/`check_scene_mix`/
`check_regions` -- only `check_beat_expand_rpg` is actually wired as a `Task(guardrail=...)`,
alongside `check_script_rpg`/`check_extraction_rpg`/`check_scene_rpg`) were never wired to any
task despite this file's docstring previously implying otherwise. Measured against real
generated scripts, retrying against them in-loop would add 10-28 extra findings per
already-generated script -- narrower and more failure-prone than `validate_references()`'s
purely mechanical checks. `guardrails.collect_quality_problems(script)` aggregates all nine
over a finished script and is called once, report-only, at the end of both pipelines;
`RunReport.quality_problems` (and the matching `review.RunRecord` field) carry the result,
surfaced in the review UI's 執行紀錄 tab without gating the run.

## Structured-output parse failures: truncation, not missing fields

A distinct failure mode from the wire.py/normalize.py one above, easy to conflate with it because
the symptom looks similar (a `ValidationError` inside `Task.execute_sync()`, before
`_coerce_model`'s own fallback chain ever runs): the provider's response is **truncated mid-object**
(observed against `deepseek-v4-flash-0731` from the UI's 生成 mode: `content == '{\n '`), not merely
missing one field. `wire.py`'s lenient mirror (every field defaulted) does nothing for this --
there's no JSON to even partially parse. `crewai`'s own `Agent.max_retry_limit` (now `1`, down from
its default `2`, on all six `make_*_agent()` factories in `crew/agents.py`) retries the exact same
call shape a couple of times first, absorbing a one-off transient truncation, but burns a full-price
call each time and does nothing if a model/provider truncates structured output for this prompt
shape systematically.

`src/bixiascribe/crew/execute.py` (`run_task()`) adds one more level: on
`execute.is_structured_parse_error()` (a `pydantic.ValidationError`/`json.JSONDecodeError`, or a
message containing `Invalid JSON`/`json_invalid`/`validation error for Lenient`), retry the same task
**once** rebuilt in free-text mode -- no `output_pydantic`, the JSON Schema spelled out in
`expected_output` instead (`crew/tasks.py::_freeform_expected_output`) -- so `pipeline.py::
_coerce_model`'s raw-text salvage actually gets a shot at the response. Any other exception
(401/429/timeout) is re-raised immediately -- a second call can't fix those and would just double the
cost. All six `make_*_task()` factories (plus `causal.repair_scene_task`) now accept a `structured:
bool = True` parameter; `structured=True`'s prompt text is byte-identical to before this existed.
`orchestrator.py`'s four `_default_*` stage runners call `execute.run_task()` instead of
`task.execute_sync()` directly; the legacy pipeline's `crew.kickoff()` runs all three tasks as one
opaque unit, so on a structured-parse failure there's no single task to retarget -- `pipeline.py::
run_pipeline_with_report()` rebuilds all three tasks in free-text mode and reruns the whole crew once
instead. `STRUCTURED_OUTPUT` (`.env`, default `auto`) is the escape hatch: `off` skips the structured
attempt entirely for every task from the first call, for a model/provider known to truncate
persistently, or for exercising the free-text path in offline tests. `RunReport.structured_fallbacks`/
`.llm_notes` (and the matching `review.RunRecord` fields, JSONL `structured_fallbacks`/`llm_notes`,
0/`[]` for rows logged before this existed) record how many times this fired and why; the review UI's
執行紀錄 tab shows them as a "模型輸出降級紀錄" expander next to the existing normalize-notes one.

**A real, pre-existing bug this exposed, not introduced by it**: `schema.parse_model_json()`'s
raw-text scan used to keep the *last* dict in the text that validated against the target model,
intended to skip past a real model's explanatory preamble and land on the final JSON answer. For a
model where every field has a default -- `ExtractionResult` is the clearest example -- pydantic's
default `extra="ignore"` means *any* dict validates, including a nested sub-object inside the real
answer itself (e.g. one entry of its own `npcs` list, scanned after the top-level object). "Last
match wins" would then silently pick that nested fragment over the complete answer. This was already
reachable before this fix (raw_scan is the pre-existing final fallback tier), but rare in practice
since a well-formed structured response usually lands in `output.pydantic` and never reaches raw
scanning at all; the free-text fallback above makes raw scanning the *primary* path for six task
types, which made the bug immediately visible in this fix's own offline verification
(`STRUCTURED_OUTPUT=off` against `LLM_BACKEND=fake`: `_default_extract` came back with `npcs=[]`
despite the canned fixture having two). Fixed by preferring the **largest-span** validating match
instead of simply the last one -- a parent object's span always strictly contains, and is therefore
never smaller than, any of its own nested matches, so the real top-level answer always wins over a
nested fragment; ties still break toward the later match, preserving the original prose-skipping
behavior for schemas where this ambiguity doesn't arise.

## Per-scene 執行歸因 (`crew/scene_metrics.py`)

A real `script_length=long` layered run took 1h46m for 19 scenes, and per-scene output size only
explained 27% of the elapsed-time variance (r²=0.27 against real checkpoint data -- see
`openspec/changes/archive/profile-layered-pipeline-cost/design.md`'s 證據二). `RunReport` used to
record only run-wide aggregates (`elapsed_s`/`token_usage`/`retrieval_calls`/`structured_fallbacks`),
so the ~73% of unexplained time couldn't be attributed to a specific scene or mechanism (reasoning
tokens, guardrail retries, structured-output fallback retries). `crew/scene_metrics.py` fixes
this: `RunReport.scene_metrics` (and the matching JSONL/`review.RunRecord` fields, `[]`/`()` for
rows logged before this existed) now carries, per beat id, `elapsed_s`/`call_elapsed_s`/
`repair_elapsed_s`/`llm_calls`/`reasoning_tokens`/`total_tokens`/`guardrail_retries`/
`retrieval_calls`/`structured_fallbacks` -- surfaced in the review UI's 執行紀錄 tab as a
sortable table, and as a "3 slowest scenes" line in `generate_script.py`'s stderr report.

Follows the same module-level, `threading.Lock`-guarded accumulator convention as
`crew/tools.py::RetrievalStats`/`crew/execute.py::FallbackStats` (`reset_stats()` at the start of
one `run_layered()` call, `get_stats()` read back into `RunReport` at the end). What's different
here is **attribution**: `execute.run_task()` (used by all four `_default_*` stage runners) and
`WuxiaRetrievalTool._run()` have no beat id in their own call signature, and changing that would
ripple through every existing `StageRunners` test stand-in's tuple shape. Instead, `scene_scope
(beat_id)` (a `@contextmanager`) sets a **context-local** "current scene" marker for the duration
of `_default_write_scene()`'s call; every `record_*()` helper reads that context-local and is a
no-op when none is active. **Non-obvious**: this was originally a plain `threading.local()`, which
undercounted `retrieval_calls` -- verified against a real run
(`.bixia_state/1787309292-req-d232acf2d8`): the run-level `RunReport.retrieval_calls` was 3 for the
one scene generated, but that scene's `scene_meta_*.json` sidecar recorded 0. crewai's own native
tool-calling loop (`crewai/agents/crew_agent_executor.py`, ~line 746) dispatches concurrent tool
calls from one LLM turn via `ThreadPoolExecutor.submit(contextvars.copy_context().run, ...)` --
`copy_context().run()` only propagates the calling thread's `contextvars.ContextVar` state into the
new worker thread, never a `threading.local()`'s (verified empirically: a `threading.local()`
attribute set before `submit()` reads back `None` inside the pool worker; a `ContextVar` set the
same way reads back correctly). `WuxiaRetrievalTool._run()` runs on exactly that pool worker, so it
could never see the beat id `_default_write_scene()` set on the ReAct loop's own thread. `_current`
is now a `contextvars.ContextVar` instead -- this still isolates `dispatch_batch()`'s own concurrent
per-scene worker threads from each other (each such thread's context starts fresh via its own
`scene_scope()` call), while also surviving crewai's inner thread-pool hop. This is sound because
crewai's ReAct loop, the guardrail callback, and the tool's `_run()` otherwise execute synchronously
relative to whichever worker `dispatch_batch()`'s `ThreadPoolExecutor` submitted the scene to. The
instrumentation lives in
the **runner** (`_default_write_scene`/`_default_repair_scene`/`execute.run_task`/
`WuxiaRetrievalTool._run`/the scene guardrail closure), not in `dispatch_batch()` itself, because
`dispatch_batch()` delegates to the serial `dispatch_next()` whenever `concurrency <= 1` and
un-gated (`orchestrator.py:875-887`) -- a hook installed only in the thread-pool path would
silently miss every serial run. `llm_calls` is read off crewai's own per-LLM-instance
`successful_requests` counter (already folded into the existing `_usage_delta()` machinery), not a
separate ReAct-round counter -- `agent.agent_executor.iterations` was considered and rejected
because it resets to 0 at the top of every `_invoke_loop`, so it would only ever report the last
retry's round count, not the scene's total.

Because a layered run can resume from `.bixia_state/<run_id>/` across process restarts, and an
already-completed scene is never regenerated, its metrics don't exist in a resumed process's
in-memory accumulator. `SceneMetric` is therefore also persisted as a sidecar checkpoint --
`.bixia_state/<run_id>/scene_meta_<beat_id>.json`, written via the existing `save_checkpoint()`
right after each scene's own `Event` checkpoint (both the committed and staged-pending branches of
`dispatch_next()`/`dispatch_batch()`) -- and `RunReport.scene_metrics` is always built by
`load_scene_metrics()` reading these back off disk, not by trusting the in-process accumulator.
This needed **no `_SCHEMA_VERSION` bump**: `detect_stage()` never consults this sidecar (same
status as the derived `causal_graph.json`), so an in-flight pre-change checkpoint resumes normally
and simply starts reporting metrics from the resume point onward. `load_scene_metrics()` filters
to beats whose committed `scene_<id>.json` checkpoint actually exists, so a stray sidecar from a
rejected/still-pending batch never leaks into a finished run's report.
