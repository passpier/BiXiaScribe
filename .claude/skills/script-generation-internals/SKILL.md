---
name: script-generation-internals
description: Deep internals of BiXiaScribe's script-generation pipelines — legacy vs layered mode, RPG-shaped output entities (player/items/quests/NPC introductions), guardrails, real-time causal consistency validation, per-agent model splits, and the script-length knob. Use when editing src/bixiascribe/crew/ (pipeline.py, orchestrator.py, agents.py, tasks.py, causal.py, guardrails.py, context_builder.py), schema.py's core RPG entities, llm.py, or generate_script.py/eval_generation.py's generation-side flags.
---

Two pipeline modes coexist, selected by `PIPELINE_MODE` in `.env` (default `legacy`) or
`--pipeline-mode` on `generate_script.py`/`eval_generation.py`/the UI's checkbox.

## `legacy`

Three agents run as a sequential `Crew` (`src/bixiascribe/crew/pipeline.py`): 編劇 (writer, produces an
event/branch skeleton per `schema.Script` with `dialogue=[]`) → 對話 (dialogue, fills in NPC lines using
the `WuxiaRetrievalTool`, which wraps `retrieval.retrieve()`) → 校對 (proofreader, checks
schema + npc_id/next_event_id cross-references — re-verified in Python via `schema.validate_references()`
after the crew finishes, not just trusted to the LLM).

`run_pipeline()` doesn't just trust `crew_output.pydantic`: `pipeline.py::_coerce_script` falls back to
`crew_output.json_dict`, then to `schema.parse_script_json()` scanning `crew_output.raw` for the last
JSON object that actually validates as a `Script` — real models often wrap JSON in explanatory prose,
which trips up CrewAI's own coercion even though the JSON itself is fine. If `validate_references()`
still finds dangling `npc_id`/`next_event_id` references (plus the RPG cross-references below) after
that, the proofreader agent gets up to `MAX_REPAIR_ATTEMPTS` (2) targeted repair passes — re-running
just the proofread task via `Task.execute_sync()`, not the whole crew — before `run_pipeline()` raises
`PipelineError`. `crew.kickoff()` itself is also wrapped, so provider errors (401/429/timeouts) surface
as `PipelineError` instead of a raw traceback.

## RPG-shaped output: player/items/quests/NPC introductions

Early script output read like a novel outline, not something playable: no first-class 玩家 concept (a
model would invent a fake `npc_player`/`npc_narrator` NPC as a workaround), variables that were always
plain booleans (no numeric 屬性), a `props` list the extractor filled in but that was never threaded
into the final `Script`, and NPCs all speaking in the very first event with no sense of who's being
introduced when. `schema.py` now has first-class `PlayerCharacter` (with `stats: list[Variable]`,
`kind="stat"` distinguishing numeric attributes from plain story `Variable`s), `Item`
(`acquired_in_event_id`), `Quest` (`event_ids`), and `EffectOp` (a structured
`target_kind`/`target_id`/`op`/`value` replacing free-text `Branch.effects`, which is now a
human-readable summary only). `NPC.first_appearance_event_id`/`introduction` record how/when a
character is introduced. `Script.player`/`.items`/`.quests` are the final-output home for all of this;
`ExtractionResult` (layered path) carries the same three fields, and `orchestrator.py::_assemble_script()`
copies them straight into the final `Script` — this is what makes them survive past the extraction
stage (`props: list[str]` is kept only for backward-compat with old checkpoints, deprecated in favor of
`items`, never read downstream). `schema.validate_references()` was extended to cross-check every new
id (item/quest/effect_op targets, `player.starting_items`, `NPC.first_appearance_event_id`,
`Quest.giver_npc_id`/`event_ids`) alongside its original `npc_id`/`next_event_id` checks — both the
legacy repair loop and the layered orchestrator's already re-run this same function, so this extension
required no changes to either repair loop. A separate `validate_npc_introductions()` checks that no NPC
speaks in an event earlier than its own `first_appearance_event_id` — kept out of
`validate_references()` deliberately so it doesn't feed the repair loops (which assume every problem
they see is worth an LLM repair pass); it's consumed by the guardrails module instead.

`src/bixiascribe/crew/guardrails.py` is a pure, offline module (no crewai/LLM import, same style as
`crew/causal.py`) providing `check_script_rpg()`/`check_extraction_rpg()`/`check_scene_rpg()` — each
returns a list of Chinese problem strings (including a heuristic that flags an NPC whose id/name looks
like a fake player/narrator role, the concrete failure mode observed before this existed) — plus
`as_feedback()` to render them as one repair instruction. These are wired onto
`make_writer_task()`/`make_extract_task()`/`make_scene_write_task()` in `crew/tasks.py` as CrewAI
`Task(guardrail=..., guardrail_max_retries=config.GUARDRAIL_MAX_RETRIES)` callbacks — a purely-Python
check that returns `(False, feedback)` and forces CrewAI to retry that task in-loop with the feedback
attached, before the whole crew/pipeline finishes (unlike the proofread repair loop, which only runs
after). `GUARDRAILS_ENABLED` (`.env`, default `true`) and `GUARDRAIL_MAX_RETRIES` (`.env`, default `2`)
control this; **guardrails are always off under `LLM_BACKEND=fake`** regardless of `GUARDRAILS_ENABLED`
(`tasks.py::_guardrails_active()`) — `FakeLLM`'s canned responses can never satisfy an RPG-shape check,
so a guardrail retry loop against it would just spin `GUARDRAIL_MAX_RETRIES` times on every offline
test. `RunReport.guardrails_enabled`/`.guardrail_max_retries` (and the matching JSONL/`review.RunRecord`
fields, defaulting to `False`/`0` for rows logged before this existed) record whether a given run had
guardrails on.

The layered pipeline's `SessionDocument` (`crew/context_builder.py::build_session_document()`) gained
`player_card`/`item_cards`/`quest_cards` (never trimmed, same priority tier as `character_cards`) and
`introduced_npc_ids` (every NPC who has already spoken in an earlier committed scene) — this is what
lets a `scene_write` task's guardrail tell "an NPC this scene was never told about" apart from "an NPC
who's already been properly introduced".

## `layered` (recommended path; legacy kept as fallback)

`PIPELINE_MODE` defaults to `legacy` not because `layered` is unproven, but because keeping the default
at `legacy` is a documented zero-cost rollback lever if `layered` ever misbehaves in the wild — both
pipelines remain fully reachable regardless of this default. `SCENE_CONCURRENCY` (default 3) controls
parallel scene generation.

The layered path decomposes generation into extractor → beat_expander → scene_writer agents
(`crew/agents.py`) over causal-graph schema models (`schema.py`: `Beat`, `BeatSheet`,
`CausalPlotGraph`, `validate_causal_graph()`), driven by a stateful, checkpointed
`crew/orchestrator.py` — checkpoints land under `.bixia_state/<run_id>/`, `detect_stage()` resumes
after a crash, and `plan_batches()`/`dispatch_batch()` parallelize scene generation across
causal-dependency batches behind a batch-confirmation gate. `orchestrator.py::run_layered()` is
the one-call entry point (`(Script, RunReport)`, same shape as `run_pipeline_with_report()`) that
every real caller (`generate_script.py`, `eval_generation.py` via `generation.generate()`, `ui/app.py`)
uses directly; `crew/pipeline.py::run_layered_pipeline()` is an older, uncheckpointed thin wrapper over
it kept only for `tests/test_crew_layered_pipeline.py` — it has no production callers and doesn't
forward `run_id`/`gate`/`concurrency`, so prefer `orchestrator.run_layered()` in new code.

Each `scene_writer` call gets more than just its own `Beat` plus an NPC subset in isolation —
`crew/context_builder.py::build_session_document()` hands it a `schema.SessionDocument` that also
includes summaries of already-completed scenes, so a scene can stay consistent with what happened
before it instead of contradicting it. This is bounded by `SESSION_DOC_MAX_TOKENS` (default 4000,
dependency-free char-based estimate, no real tokenizer): once the serialized document would exceed
that, the lowest-priority scene summary is dropped first — causal ancestors of the current beat
outrank unrelated older scenes (via `Beat.causal_deps`, unioned with the real `CausalPlotGraph`'s
transitive edge closure when one is available), and character cards/the current beat are never
dropped. `SessionDocument.current_beat` is a typed `Beat`, not a string — `llm.py::FakeLLM`'s
scene_writer branch recovers the target beat by scanning the prompt for a JSON object that validates
as `Beat` (`schema.parse_model_json`), which only works if it's a real nested object rather than a
doubly-JSON-encoded string.

## Real-time causal consistency validation

Every generated scene is checked against the causal graph **as it's produced**, not only at the
final proofread step. `crew/causal.py` (pure, offline, no LLM/crewai import) mechanically derives
a `CausalPlotGraph` from fields the `Event` schema already has — `PlotNode.preconditions` from
`event.triggers[*].condition`, `PlotNode.postconditions` from `event.branches[*].effects`,
`PlotEdge`s from `Beat.causal_deps` and `Event.branches[*].next_event_id` — no schema change, no
extra prompt/token cost. `check_scene_consistency()` compares a candidate scene's precondition
against its nearest causal ancestor's postcondition via a conservative Chinese fact normalizer
plus a curated antonym table (unparseable text never false-positives); `orchestrator.py` rebuilds
the graph from scratch from every committed scene rather than mutating it incrementally (mirrors
`detect_stage()`'s "always re-derive from disk" invariant), persisting it to
`.bixia_state/<run_id>/causal_graph.json`.

`CAUSAL_VALIDATION` (`.env`) controls what happens on a conflict: `off` skips the check entirely;
`warn` records `RunReport.causal_problems` without blocking; `repair` (default) sends the scene
back for a targeted repair pass (like `_repair()`'s proofread loop) before falling back to `warn`
behavior; `strict` refuses to checkpoint a scene that's still inconsistent after repair, raising
`PipelineError`. **Non-obvious**: under `LLM_BACKEND=fake`, `repair` degrades to `warn` — the fake
LLM has no way to actually resolve a semantic conflict, so there's nothing for a repair pass to fix.

## Model calls and per-agent model splits

Model calls go through OpenRouter (via crewai's `LLM` + litellm's `openrouter/` model prefix), never a
provider SDK directly, so switching models is an env var change. Controlled by `LLM_BACKEND` in `.env`:
- `openrouter` (default) — real generation. Needs `OPENROUTER_API_KEY`. `LLM_MODEL` sets the default
  model for all three legacy agents; `LLM_MODEL_WRITER` / `LLM_MODEL_DIALOGUE` / `LLM_MODEL_PROOF`
  override per-agent (see `.env.example` for per-role tuning guidance). **Whatever model backs the
  對話 (dialogue) agent must support function calling/tool use** — otherwise it never calls
  `wuxia_corpus_search` and the RAG retrieval this pipeline is built around silently never fires; the
  pipeline still produces a valid script either way, just without corpus-grounded wording, so this is
  easy to miss.
- `fake` — offline, deterministic canned responses (`src/bixiascribe/llm.py::FakeLLM`), no key/network/
  cost. This is what `tests/test_crew_pipeline.py` uses.

`scripts/generate_script.py --preflight-only` checks `LLM_BACKEND`/`OPENROUTER_API_KEY`/index presence
before spending a token. `run_pipeline_with_report()` (what `generate_script.py` actually calls;
`run_pipeline()` is a thin Script-only wrapper kept for existing callers/tests) returns a `RunReport`
alongside the `Script` — model ids, elapsed time, `crew_output.token_usage`, which `_coerce_script`
fallback level produced the result (`coerced_from`: `pydantic`/`json_dict`/`raw_scan`), repair attempt
count, and retrieval call/failure counts from `crew/tools.py`'s `RetrievalStats` (module-level counter,
reset per run via `tools.reset_stats()`). `generate_script.py` prints this report to stderr after every
real run — `retrieval_calls == 0` is the concrete signal that the dialogue-agent tool-calling failure
mode above actually happened, instead of having to infer it from verbose log scrollback. On failure,
`PipelineError.report` carries whatever partial `RunReport` was gathered.

`llm.py::ModelChoice` (a frozen dataclass with `writer`/`dialogue`/`proof` fields, defaulting to
`config.LLM_MODEL_*`, plus `extractor`/`beat_expander`/`scene_writer` for the layered pipeline) is
threaded explicitly through `build_llm()` → `crew/agents.py`'s six `make_*_agent()` factories →
`run_pipeline_with_report(..., models=...)`/`orchestrator.run_layered(..., models=...)`, instead of
relying on env vars read once at import time. This is what lets one process run several model splits
back to back without editing `.env` and restarting. `generation.Variant.to_model_choice()` fills all
six roles from a variant's `writer`/`dialogue`/`proof` (falling back extractor/beat_expander to
`writer` and scene_writer to `dialogue`) unless the variant sets `extractor`/`beat_expander`/
`scene_writer` explicitly.

`generation.Variant.script_length` (`"short"`/`"medium"`/`"long"`, `None` falls back to
`config.SCRIPT_LENGTH`) is the per-variant override for how long a generated script's prompt asks the
model to aim for — see "Script length" below for what this actually controls (and doesn't).

`LLM_PROVIDER_ONLY`/`LLM_PROVIDER_SORT` (`.env`) pin OpenRouter provider routing (forwarded via
litellm's `extra_body`, not verified against a live response as of this writing — see `llm.py::
build_llm()`'s comment) — needed because a model's cheapest/default route isn't always tool-capable
(e.g. a model's cheapest endpoint can report `tools: false`, which would silently break
`wuxia_corpus_search` the same way an unsupported dialogue model does). `LLM_PROVIDER_ONLY` is a
process-wide env var (`config.py` reads it once at import time), so one `eval_generation.py`
invocation can't pin different providers per variant in the same matrix — pin per invocation instead
(see `docs/DESIGN_NOTES.md`'s "比較不同 agent 的模型組合" section for the multi-provider command
pattern).

## Script length

Nothing in `schema.py` bounds how long a generated script is, and `build_llm()` passes no
`max_tokens` by default — the only thing that has ever controlled length is `crew/tasks.py`'s prompt
wording. `SCRIPT_LENGTH` (`.env`; also `--script-length` on `generate_script.py`/
`eval_generation.py`, `Variant.script_length` per-variant, and the review UI's 生成 mode's "劇本篇幅"
selector) is an opt-in prompt-level target — `"short"` (default) asks for a floor of "至少 1-2 章 x 1
beat" (layered) / "至少 2 個 events" (legacy), `"medium"`/`"long"` ask for progressively more
chapters/beats/dialogue depth. This is still just a target, not an enforced cap — a model can
under/over-shoot it, and `LLM_MAX_TOKENS` (`.env`) is the one real ceiling worth setting alongside a
longer target, since a low provider-side default can silently truncate a longer Event's JSON.

Besides the three presets, `SCRIPT_LENGTH` (and every other entry point above) also accepts a custom
`"custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT,scene_mix=TEXT"` spec, parsed by
the dependency-free `src/bixiascribe/length.py` (no imports from `config`/`crew`/`crewai`, so both
`config.py` and `crew/tasks.py` can depend on it without a cycle). Any subset of the five fields may
be given — `"custom:events=20"` alone is valid; missing fields are derived from `events`
(`chapters`/`beats_per_chapter` scaled proportionally, `min_dialogue`/`scene_mix` tiered by `events`).
An unparseable `custom:` string, or any value that's neither a preset nor `custom:`-prefixed, falls
back to `"short"`, the same degrade-not-crash convention as `CAUSAL_VALIDATION`/`PIPELINE_MODE`.
`scene_mix` (added alongside the GMUD script frame — see the gmud-schema-internals skill) expresses
the guide's 主要場景略多於調味場景 guidance as a fifth prompt target, threaded through `LengthSpec`/
`_length_target()` exactly like the other four — it feeds `make_beat_expand_task`'s prompt, which is
where a beat first gets a `scene_kind` ("main"/"flavor") assigned.
`config.SCRIPT_LENGTH` and `RunReport.script_length`/JSONL rows always hold the canonicalized,
fully-resolved form (`LengthSpec.canonical` — either the bare preset name, or a `custom:...` string
with every field filled in), so a logged run's target length is self-describing without
cross-referencing the configuration active at run time. `crew/tasks.py`'s `_length_target()` is a
thin wrapper over `length.parse_length_spec(script_length).targets` — prompt text itself is
unchanged from before this knob existed for the `short` default (see
`tests/test_script_length.py`'s byte-for-byte regression guards).
