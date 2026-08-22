"""JSON schema for a generated 武俠 RPG script, shared by all three CrewAI
agents (see crew/tasks.py) and the final pipeline output.

This schema isn't specified anywhere else in the repo/docs (the architecture
doc only lists field *categories* -- NPC/事件/分支/變數/觸發條件 -- in its
pipeline diagram), so it's defined here as the single source of truth:

- 編劇 agent (writer) fills everything except Event.dialogue (left empty).
- 對話 agent (dialogue) fills Event.dialogue for every event, using RAG
  retrieval for 武俠語感.
- 校對 agent (proofreader) validates the whole Script against this schema
  plus the cross-reference rules in validate_references().

Not assumed to be the final RPG Maker export format -- that conversion is a
later stage per CLAUDE.md.

Phase 4 (2026-08-22, see openspec/changes/2026-08-22-slim-script-schema-mvp):
rewritten to the flat/ID-referenced shape from `武俠劇本資料庫Schema設計.md`
(a schema designed for an actual game engine, not a nested outline document)
-- every entity is a flat top-level array, cross-referenced by id, instead of
the earlier GMUD frame's nested objects (FactionRelation, StatThreshold,
ProgressiveReveal, multi-check lists, structured EffectOp). Three fields are
deliberately kept beyond the guide's literal shape because the pipeline
depends on them: `Event.summary`/`.title` (cross-scene memory for the
layered pipeline's context_builder), `Choice.effects` (the sole
postcondition source for causal.py's consistency graph), `Chapter.summary`.
`Event.triggers` became `Event.preconditions: list[str]` rather than being
dropped, for the matching reason on the precondition side -- see this
change's design.md for the full rationale.
"""
from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError


class Meta(BaseModel):
    """Script-level identity, read once at load time. Replaces the earlier
    five parallel top-level fields (title/premise/theme/goal/tone) -- premise
    is folded into theme per the design guide."""

    title: str
    theme: str = ""
    goal: str = ""
    tone: str = ""


class Stat(BaseModel):
    """The single numeric value this engine tracks (心境值/正邪值 etc.).
    Replaces PlayerCharacter.stats: list[Variable] + the deleted
    StatThreshold table -- if a script ever needs more than one stat, this
    goes back to being a list, but the guide's 唯一數值 principle says it
    shouldn't for this project's scope."""

    id: str = "mood"
    name: str = ""
    init: int = 50


class Player(BaseModel):
    id: str = "player"
    name: str = ""
    origin: str = ""
    flaw: str = ""
    token: str = ""  # free-text token/keepsake description, not an Item id


class Faction(BaseModel):
    id: str
    name: str
    motive: str = ""  # replaces alignment + relations[].stance matrix


class NPC(BaseModel):
    id: str
    name: str
    faction_id: str = ""
    role: str = ""
    # Kept beyond the guide's literal 4-field NPC (id/name/faction_id/role):
    # speech_style is a direct input to the dialogue agent's RAG prompt (it
    # is what determines 武俠語感), and personality feeds line-to-line
    # character consistency -- these are quality-load-bearing, not
    # decorative, unlike the GMUD-era first_appearance_event_id/identity/
    # surface_motive/true_motive fields this rewrite drops.
    personality: str = ""
    speech_style: str = ""


class Truth(BaseModel):
    """三層真相: public (known from the start) / revealed (progressively
    unlocked, in reveal order -- an ordered list replaces the earlier
    ProgressiveReveal objects each bound to a reveal_chapter_id) / hidden
    (withheld until the ending -- context_builder.py never constructs a
    field carrying this at all)."""

    public: str = ""
    revealed: list[str] = Field(default_factory=list)
    hidden: str = ""


class Item(BaseModel):
    id: str
    name: str
    from_event: str = ""  # "" = held from the start


class Clue(BaseModel):
    id: str
    name: str
    from_event: str = ""


class Chapter(BaseModel):
    id: str
    title: str
    summary: str = ""  # kept beyond the guide's literal shape: context_builder._chapter_card()
    loc: str = ""
    start_event: str = ""


class DialogueLine(BaseModel):
    npc: str  # npcs[*].id, or "player" for the protagonist's own line
    line: str


class Check(BaseModel):
    """單一判定機制: at most one per event (a single object, not a list,
    unlike the earlier SkillCheck list) -- on_pass/on_fail are the only two
    branch targets, and a failed check always advances the story via
    on_fail + fail_cost rather than dead-ending it."""

    on_pass: str = ""
    on_fail: str = ""
    fail_cost: str = ""


class Choice(BaseModel):
    """Replaces Branch. `delta` is a plain int against the script's single
    Stat, replacing the earlier structured multi-target EffectOp list.
    `effects` (free-text) is deliberately kept -- causal.py::event_to_node()
    reads it as the primary PlotNode.postconditions source; `delta` alone
    can never be compared against a precondition string. `payoff_at` (a
    chapter id) replaces payoff_description (free text) +
    converges_to_event_id (a convergence-graph guarantee this rewrite drops
    entirely, per the design guide's simpler "delayed payoff lands in some
    chapter" model)."""

    id: str
    text: str
    next: str = ""
    cost: str = ""
    effects: str = ""
    delta: int = 0
    payoff_at: str = ""


class Event(BaseModel):
    id: str
    # Kept beyond the guide's literal shape: title/summary are the layered
    # pipeline's only cross-scene memory (context_builder.py::
    # _scene_summary()/review.event_titles()/metrics.continuity_metrics).
    # location is dropped -- location now lives on Chapter.loc (線性地點鏈).
    title: str = ""
    summary: str = ""
    chapter_id: str = ""
    npc_ids: list[str] = Field(default_factory=list)
    # Renamed from Trigger (type+condition) to a plain string list --
    # causal.py only ever read the condition text, so the type field carried
    # no independent meaning. Still the sole PlotNode.preconditions source.
    preconditions: list[str] = Field(default_factory=list)
    clue_ids: list[str] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    check: Check | None = None
    choices: list[Choice] = Field(default_factory=list)


class Ending(BaseModel):
    """Selected by where the script's single Stat value falls -- replaces
    StatCondition/required_branch_ids with a plain value range, since there
    is now only one stat to condition on."""

    id: str
    name: str
    description: str = ""
    min: int = 0
    max: int = 100


class Script(BaseModel):
    meta: Meta
    stat: Stat | None = None
    player: Player | None = None
    factions: list[Faction] = Field(default_factory=list)
    npcs: list[NPC] = Field(default_factory=list)
    truth: Truth | None = None
    items: list[Item] = Field(default_factory=list)
    clues: list[Clue] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    endings: list[Ending] = Field(default_factory=list)


def validate_references(script: Script) -> list[str]:
    """Cross-reference checks pydantic's field-level validation can't do:
    every id one entity names must point at something that actually exists
    elsewhere in the script. Returns a list of human-readable problem
    descriptions (empty list = fully consistent).

    This is what the 校對 agent (proofreader) runs to check the writer/
    dialogue agents' output before it's accepted as final -- both the
    legacy repair loop (pipeline.py::_repair) and the layered orchestrator
    re-run this same function, so extending it here automatically extends
    both repair loops without touching their code.
    """
    problems: list[str] = []

    npc_ids = {npc.id for npc in script.npcs}
    event_ids = {event.id for event in script.events}
    faction_ids = {faction.id for faction in script.factions}
    chapter_ids = {chapter.id for chapter in script.chapters}
    clue_ids = {clue.id for clue in script.clues}

    dialogue_target_ids = npc_ids | {"player"}

    for event in script.events:
        for line in event.dialogue:
            if line.npc not in dialogue_target_ids:
                problems.append(
                    f"event {event.id!r}: dialogue references unknown npc {line.npc!r}"
                )
        if event.chapter_id and event.chapter_id not in chapter_ids:
            problems.append(
                f"event {event.id!r}: unknown chapter_id {event.chapter_id!r}"
            )
        for npc_id in event.npc_ids:
            if npc_id not in npc_ids:
                problems.append(
                    f"event {event.id!r}: npc_ids references unknown npc {npc_id!r}"
                )
        for clue_id in event.clue_ids:
            if clue_id not in clue_ids:
                problems.append(
                    f"event {event.id!r}: clue_ids references unknown clue {clue_id!r}"
                )
        if event.check:
            if event.check.on_pass and event.check.on_pass not in event_ids:
                problems.append(
                    f"event {event.id!r}: check.on_pass references unknown "
                    f"event {event.check.on_pass!r}"
                )
            if event.check.on_fail and event.check.on_fail not in event_ids:
                problems.append(
                    f"event {event.id!r}: check.on_fail references unknown "
                    f"event {event.check.on_fail!r}"
                )
        for choice in event.choices:
            if choice.next and choice.next not in event_ids:
                problems.append(
                    f"event {event.id!r}: choice {choice.id!r} points to unknown "
                    f"next {choice.next!r}"
                )
            if choice.payoff_at and choice.payoff_at not in chapter_ids:
                problems.append(
                    f"event {event.id!r}: choice {choice.id!r} references unknown "
                    f"payoff_at {choice.payoff_at!r}"
                )

    for chapter in script.chapters:
        if chapter.start_event and chapter.start_event not in event_ids:
            problems.append(
                f"chapter {chapter.id!r}: unknown start_event {chapter.start_event!r}"
            )

    for clue in script.clues:
        if clue.from_event and clue.from_event not in event_ids:
            problems.append(
                f"clue {clue.id!r}: unknown from_event {clue.from_event!r}"
            )

    for item in script.items:
        if item.from_event and item.from_event not in event_ids:
            problems.append(
                f"item {item.id!r}: unknown from_event {item.from_event!r}"
            )

    for npc in script.npcs:
        if npc.faction_id and npc.faction_id not in faction_ids:
            problems.append(
                f"npc {npc.id!r}: unknown faction_id {npc.faction_id!r}"
            )

    return problems


M = TypeVar("M", bound=BaseModel)


def parse_model_json(text: str, model_cls: type[M]) -> M | None:
    """Salvage a `model_cls` instance from a blob of text that may wrap the
    JSON in prose or markdown (real LLMs do this a lot more than the `fake`
    backend does).

    Scans every '{' in `text` via JSONDecoder.raw_decode (tolerant of
    surrounding non-JSON content) and keeps the *largest-span* dict that
    actually validates as `model_cls` (ties broken toward the later match)
    -- validating against the schema, not just checking for the presence of
    a couple of field names, is what keeps this from mistaking a JSON
    *Schema* CrewAI injects into prompts (whose "properties" object can
    happen to contain the same field names) for an actual instance.

    Largest-span, not simply "last one wins": for a model where every field
    has a default (e.g. schema.ExtractionResult) -- a fairly common shape
    in this codebase -- pydantic's default `extra="ignore"` means literally
    *any* dict validates, including a nested sub-object inside the real
    top-level answer (e.g. one entry of its own "npcs" list). Since that
    nested object's '{' is scanned after the top-level one's, a pure
    "last match wins" rule would silently pick the nested fragment over the
    real, complete answer -- observed in practice via the free-text
    structured-output fallback (crew/execute.py), which makes this tier the
    *primary* path for several all-optional-field schemas instead of a rare
    corner case. A parent object's span always strictly contains (and is
    therefore never smaller than) any of its own nested matches, so
    preferring the largest span fixes that while still preferring a later
    match over an earlier, same-sized one -- the original behavior this
    docstring described, which exists to skip past unrelated JSON-looking
    text a real model sometimes emits before its actual final answer.

    Returns None if nothing in `text` validates as `model_cls`.
    """
    decoder = json.JSONDecoder()
    best: M | None = None
    best_span = -1
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        try:
            candidate = model_cls.model_validate(obj)
        except ValidationError:
            continue
        span = end - i
        if span >= best_span:  # >= keeps "a later, equal-size match wins"
            best = candidate
            best_span = span
    return best


def parse_script_json(text: str) -> Script | None:
    """Salvage a Script from a blob of text that may wrap the JSON in prose
    or markdown. Thin wrapper around `parse_model_json` kept for backward
    compatibility with existing callers/tests -- see that function's
    docstring for the scanning algorithm.
    """
    return parse_model_json(text, Script)


# --- Layered generation models -------------------------------------------
#
# These support the "分層生成" pipeline (extractor -> beat_expander ->
# scene_writer, see crew/pipeline.py::run_layered_pipeline) that decomposes
# the single writer agent's "outline + branches + variables + NPCs in one
# shot" responsibility into stages with their own persistable, independently
# retryable output. They are additive: the existing `Script`/`Event`/etc.
# models above are the pipeline's final output format either way, and
# nothing here changes their behavior.


class Outline(BaseModel):
    title: str
    premise: str = ""  # internal-only; _assemble_script maps this into Script.meta.theme
    chapters: list[Chapter] = Field(default_factory=list)


class Beat(BaseModel):
    id: str
    chapter_id: str
    summary: str  # 3-8 景/章 per beat, expands into one Event when written
    npc_ids: list[str] = Field(default_factory=list)
    causal_deps: list[str] = Field(default_factory=list)  # ids of prerequisite beats/events


class BeatSheet(BaseModel):
    """beat_expander's output: an Outline plus the Beats that flesh it out.
    Wrapped in one model (rather than returning a bare list) because
    `Task.output_pydantic` takes a single model class, and this is also the
    natural on-disk checkpoint unit for a `beats.json` (see orchestrator.py)."""

    outline: Outline
    beats: list[Beat] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """extractor's output: the cast/world frame pulled out of the raw user
    requirement, before any beat/scene structure exists.

    orchestrator.py::_assemble_script() copies these straight into the
    final Script -- this is what makes them survive past the extraction
    stage."""

    meta: Meta | None = None
    stat: Stat | None = None
    player: Player | None = None
    npcs: list[NPC] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    factions: list[Faction] = Field(default_factory=list)
    truth: Truth | None = None
    clues: list[Clue] = Field(default_factory=list)
    endings: list[Ending] = Field(default_factory=list)


class PlotNode(BaseModel):
    id: str  # corresponds to an Event.id or Beat.id
    preconditions: list[str] = Field(default_factory=list)  # variable conditions
    postconditions: list[str] = Field(default_factory=list)  # variable changes caused


class PlotEdge(BaseModel):
    from_id: str
    to_id: str
    # Narrative condition text, currently always "" -- Choice has no condition field
    condition: str = ""


class CausalPlotGraph(BaseModel):
    nodes: list[PlotNode] = Field(default_factory=list)
    edges: list[PlotEdge] = Field(default_factory=list)


def validate_causal_graph(graph: CausalPlotGraph) -> list[str]:
    """Reference-integrity checks for a CausalPlotGraph: every edge's
    from_id/to_id must point at a node that actually exists. Returns a list
    of human-readable problem descriptions (empty list = fully consistent),
    in the same style as validate_references() above.

    This only checks structural referential integrity. Checking whether a
    scene's postconditions actually contradict an existing, uncovered
    precondition elsewhere in the graph is a semantic check left to
    crew/causal.py::check_scene_consistency(), run per-scene during
    generation rather than only here.
    """
    problems: list[str] = []
    node_ids = {node.id for node in graph.nodes}

    for edge in graph.edges:
        if edge.from_id not in node_ids:
            problems.append(
                f"edge {edge.from_id!r} -> {edge.to_id!r}: unknown from_id {edge.from_id!r}"
            )
        if edge.to_id not in node_ids:
            problems.append(
                f"edge {edge.from_id!r} -> {edge.to_id!r}: unknown to_id {edge.to_id!r}"
            )

    return problems


def validate_outline_beats(outline: Outline, beats: list[Beat]) -> list[str]:
    """Cross-reference check between an Outline and its Beats: every
    Beat.chapter_id must correspond to some outline.chapters[*].id. Returns
    a list of human-readable problem descriptions (empty list = consistent).
    """
    problems: list[str] = []
    chapter_ids = {chapter.id for chapter in outline.chapters}

    for beat in beats:
        if beat.chapter_id not in chapter_ids:
            problems.append(
                f"beat {beat.id!r}: unknown chapter_id {beat.chapter_id!r}"
            )

    return problems


# --- Context compression --------------------------------------------------


class SessionDocument(BaseModel):
    """Compressed, bounded context handed to one scene_writer call in place
    of a raw beat + NPC-subset dump -- see
    crew/context_builder.py::build_session_document() for how
    it's assembled and kept under config.SESSION_DOC_MAX_TOKENS.

    `current_beat` is a typed Beat, not a str holding beat.model_dump_json():
    llm.py::FakeLLM's scene_writer branch recovers the target beat by
    scanning the prompt text for a JSON object that validates as Beat
    (schema.parse_model_json). A string field would double-JSON-encode the
    beat (escaped quotes), which parse_model_json's raw_decode can never
    recover -- every offline/fake test would silently regress to the
    "beat is None" fallback. Declaring it last matters too:
    parse_model_json keeps the *last* validating match it finds, and no
    other field here (list[str]/int) can accidentally validate as a Beat,
    since Beat requires id/chapter_id/summary that NPC-derived strings and
    plain summaries don't structurally have.
    """

    character_cards: list[str] = Field(default_factory=list)
    scene_summaries: list[str] = Field(default_factory=list)
    omitted_scene_count: int = 0
    # Player/item context for RPG-shaped scene writing, and which NPCs
    # have already been introduced by an earlier committed scene -- all
    # list[str]/str so none can accidentally validate as a Beat (see
    # docstring above). Must stay before current_beat.
    player_card: list[str] = Field(default_factory=list)
    item_cards: list[str] = Field(default_factory=list)
    introduced_npc_ids: list[str] = Field(default_factory=list)
    # GMUD world context, same str/list[str]-only constraint as above.
    # truth_unlocked carries only progressive reveals already unlocked by
    # the current beat's chapter -- this field (and this class) never
    # carries Truth.hidden at all, so a scene prompt has no way to see
    # a hidden fact regardless of prompt-following (see context_builder.py).
    faction_cards: list[str] = Field(default_factory=list)
    chapter_card: list[str] = Field(default_factory=list)
    truth_public: list[str] = Field(default_factory=list)
    truth_unlocked: list[str] = Field(default_factory=list)
    # Closed menu of every id a scene_writer call is allowed to reference
    # for chapter_id/clue_ids/item ids -- see
    # crew/context_builder.py::_allowed_ids(). Framing the prompt as "pick
    # from this list, leave blank if nothing fits" instead of "make one up"
    # is what prevents validate_references()'s "unknown chapter_id" class
    # of problem at the source, rather than catching it after the fact in
    # crew/normalize.py. list[str]/str-only, same constraint as the fields
    # above -- must stay before current_beat.
    allowed_ids: list[str] = Field(default_factory=list)
    current_beat: Beat
