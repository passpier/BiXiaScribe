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

from pydantic import BaseModel, Field


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
