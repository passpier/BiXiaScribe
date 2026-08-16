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
"""
from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError


class Variable(BaseModel):
    id: str
    name: str
    initial: str | int | bool
    description: str = ""


class NPC(BaseModel):
    id: str
    name: str
    identity: str  # e.g. 門派/身份, "少林寺俗家弟子"
    personality: str
    speech_style: str  # 語氣/用詞習慣, feeds the dialogue agent's RAG prompt


class Trigger(BaseModel):
    type: str  # e.g. "on_enter", "on_variable", "on_item"
    condition: str


class DialogueLine(BaseModel):
    npc_id: str
    line: str
    emotion: str = ""


class Branch(BaseModel):
    id: str
    choice_text: str
    condition: str = ""
    effects: str = ""
    next_event_id: str


class Event(BaseModel):
    id: str
    title: str
    location: str
    summary: str
    triggers: list[Trigger] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    branches: list[Branch] = Field(default_factory=list)


class Script(BaseModel):
    title: str
    premise: str
    variables: list[Variable] = Field(default_factory=list)
    npcs: list[NPC] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)


def validate_references(script: Script) -> list[str]:
    """Cross-reference checks pydantic's field-level validation can't do:
    every dialogue.npc_id and branch.next_event_id must point at something
    that actually exists in the script. Returns a list of human-readable
    problem descriptions (empty list = fully consistent).

    This is what the 校對 agent (proofreader) runs to check the writer/
    dialogue agents' output before it's accepted as final.
    """
    problems: list[str] = []

    npc_ids = {npc.id for npc in script.npcs}
    event_ids = {event.id for event in script.events}

    for event in script.events:
        for line in event.dialogue:
            if line.npc_id not in npc_ids:
                problems.append(
                    f"event {event.id!r}: dialogue references unknown npc_id {line.npc_id!r}"
                )
        for branch in event.branches:
            if branch.next_event_id not in event_ids:
                problems.append(
                    f"event {event.id!r}: branch {branch.id!r} points to unknown "
                    f"next_event_id {branch.next_event_id!r}"
                )

    return problems


M = TypeVar("M", bound=BaseModel)


def parse_model_json(text: str, model_cls: type[M]) -> M | None:
    """Salvage a `model_cls` instance from a blob of text that may wrap the
    JSON in prose or markdown (real LLMs do this a lot more than the `fake`
    backend does).

    Scans every '{' in `text` via JSONDecoder.raw_decode (tolerant of
    surrounding non-JSON content) and keeps the *last* dict that actually
    validates as `model_cls` -- validating against the schema, not just
    checking for the presence of a couple of field names, is what keeps
    this from mistaking a JSON *Schema* CrewAI injects into prompts (whose
    "properties" object can happen to contain the same field names) for an
    actual instance.

    Returns None if nothing in `text` validates as `model_cls`.
    """
    decoder = json.JSONDecoder()
    best: M | None = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        try:
            candidate = model_cls.model_validate(obj)
        except ValidationError:
            continue
        best = candidate  # keep scanning -- a later match is more recent
    return best


def parse_script_json(text: str) -> Script | None:
    """Salvage a Script from a blob of text that may wrap the JSON in prose
    or markdown. Thin wrapper around `parse_model_json` kept for backward
    compatibility with existing callers/tests -- see that function's
    docstring for the scanning algorithm.
    """
    return parse_model_json(text, Script)


# --- Layered generation models (BiXiaScribe 重構 Phase 1) ---------------
#
# These support a future "分層生成" pipeline (extractor -> beat_expander ->
# scene_writer, see crew/pipeline.py::run_layered_pipeline) that decomposes
# the single writer agent's "outline + branches + variables + NPCs in one
# shot" responsibility into stages with their own persistable, independently
# retryable output. They are additive: the existing `Script`/`Event`/etc.
# models above are the pipeline's final output format either way, and
# nothing here changes their behavior.


class ChapterOutline(BaseModel):
    id: str
    title: str
    summary: str
    beat_ids: list[str] = Field(default_factory=list)


class Outline(BaseModel):
    title: str
    premise: str
    chapters: list[ChapterOutline] = Field(default_factory=list)


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
    """extractor's output: the cast/variables/props pulled out of the raw
    user requirement, before any beat/scene structure exists."""

    npcs: list[NPC] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    branch_candidates: list[str] = Field(default_factory=list)


class PlotNode(BaseModel):
    id: str  # corresponds to an Event.id or Beat.id
    preconditions: list[str] = Field(default_factory=list)  # variable conditions
    postconditions: list[str] = Field(default_factory=list)  # variable changes caused


class PlotEdge(BaseModel):
    from_id: str
    to_id: str
    condition: str = ""  # corresponds to Branch.condition


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
    precondition elsewhere in the graph is a semantic check left to a later
    stage (see the 因果一致性即時校驗 phase in the refactor plan).
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


# --- Context compression (BiXiaScribe 重構 Phase 5) ---------------------


class SessionDocument(BaseModel):
    """Compressed, bounded context handed to one scene_writer call in place
    of the raw beat + NPC-subset dump make_scene_write_task used before
    Phase 5 -- see crew/context_builder.py::build_session_document() for how
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
    current_beat: Beat
