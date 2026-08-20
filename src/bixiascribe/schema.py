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
    # flag | stat | item | quest -- free string (not Literal), same
    # degrade-not-crash convention as CAUSAL_VALIDATION/PIPELINE_MODE: a
    # model filling in something else shouldn't fail the whole script.
    # "stat" is what a PlayerCharacter.stats entry should use (內力/聲望/
    # 銀兩 etc.); plain boolean story flags stay "flag", the pre-existing
    # default so old callers/scripts are unaffected.
    kind: str = "flag"


class NPC(BaseModel):
    id: str
    name: str
    identity: str  # e.g. 門派/身份, "少林寺俗家弟子"
    personality: str
    speech_style: str  # 語氣/用詞習慣, feeds the dialogue agent's RAG prompt
    # Which Event first introduces this NPC, and how -- lets
    # validate_npc_introductions() catch an NPC speaking before they've
    # been introduced (the "5 NPCs all talk in event 0, nobody's been
    # introduced" symptom). "" (default) = not tracked, no check applies.
    first_appearance_event_id: str = ""
    introduction: str = ""
    faction_id: str = ""
    surface_motive: str = ""
    true_motive: str = ""
    # Free-text entries like "聲望≥50: 友好" describing how attitude shifts
    # across StatThreshold ranges -- narrative content, not itself a
    # cross-checked reference (StatThreshold.unlocks_id is what
    # validate_references() checks against npc ids for unlocks_kind=
    # "npc_attitude").
    attitude_by_threshold: list[str] = Field(default_factory=list)


class Trigger(BaseModel):
    type: str  # e.g. "on_enter", "on_variable", "on_item"
    condition: str


class DialogueLine(BaseModel):
    npc_id: str
    line: str
    emotion: str = ""


class EffectOp(BaseModel):
    """One structured branch effect, e.g. {give item}/{advance quest},
    replacing the free-text Branch.effects string with something
    validate_references() can actually check target ids against."""

    target_kind: str  # variable | stat | item | quest
    target_id: str
    op: str  # set | add | give | take | start | complete
    value: str = ""


class FactionRelation(BaseModel):
    faction_id: str
    stance: str = ""  # 結盟/敵對/中立/附庸 -- free string, same degrade-not-crash convention


class Faction(BaseModel):
    id: str
    name: str
    alignment: str = ""  # ideology/stance description, not a Literal
    relations: list[FactionRelation] = Field(default_factory=list)


class StatThreshold(BaseModel):
    """One 陣營值/數值門檻表 row: a value range for a player stat, and what
    that range unlocks. `unlocks_kind` mirrors EffectOp.target_kind's free-
    string convention -- branch | event | npc_attitude | ending."""

    id: str
    stat_id: str  # PlayerCharacter.stats[*].id
    min_value: int | None = None
    max_value: int | None = None
    unlocks_kind: str = ""
    unlocks_id: str = ""
    description: str = ""


class SubLocation(BaseModel):
    id: str
    name: str
    function: str = ""  # 打聽消息/交易/療傷/學習技能 etc.


class Region(BaseModel):
    id: str
    name: str
    unlock_condition: str = ""
    sub_locations: list[SubLocation] = Field(default_factory=list)


class ProgressiveReveal(BaseModel):
    id: str
    fact: str
    reveal_chapter_id: str = ""
    reveal_event_id: str = ""


class TruthLayer(BaseModel):
    """三層真相: 公開 (known from the start) / 逐步得知 (progressive) / 私藏
    (hidden until its reveal point -- see context_builder.py, which never
    constructs a field carrying `hidden` at all)."""

    public: list[str] = Field(default_factory=list)
    progressive: list[ProgressiveReveal] = Field(default_factory=list)
    hidden: list[str] = Field(default_factory=list)


class Clue(BaseModel):
    id: str
    name: str
    found_in_event_id: str = ""
    serves: str = ""  # which mystery/ability-gated path this clue serves


class SkillCheck(BaseModel):
    """含失敗替代路線: attribute contest/dice/probability/item bypass, always
    with either a failure_branch_id or an item_bypass_id so a failed check
    advances the story instead of dead-ending it."""

    id: str
    kind: str = ""  # attribute_contest | dice | probability | item_bypass
    stat_id: str = ""
    difficulty: str = ""
    success_next_event_id: str = ""
    failure_branch_id: str = ""
    failure_cost: str = ""
    item_bypass_id: str = ""  # Item.id that bypasses the check entirely


class StatCondition(BaseModel):
    stat_id: str
    min_value: int | None = None
    max_value: int | None = None


class Ending(BaseModel):
    id: str
    name: str
    description: str = ""
    stat_conditions: list[StatCondition] = Field(default_factory=list)
    required_branch_ids: list[str] = Field(default_factory=list)


class Branch(BaseModel):
    id: str
    choice_text: str
    condition: str = ""
    effects: str = ""  # human-readable summary; effect_ops is the structured form
    next_event_id: str
    effect_ops: list[EffectOp] = Field(default_factory=list)
    # 抉擇點設計三原則 fields: what the player gives up (never merely a stat
    # delta), immediate feedback, and -- when the effect isn't resolved in
    # this same event -- the delayed payoff plus the point every path
    # eventually converges back to.
    cost: str = ""
    immediate_feedback: str = ""
    payoff_chapter_id: str = ""
    payoff_description: str = ""
    converges_to_event_id: str = ""


class Event(BaseModel):
    id: str
    title: str
    location: str
    summary: str
    triggers: list[Trigger] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    branches: list[Branch] = Field(default_factory=list)
    quest_id: str = ""
    chapter_id: str = ""
    scene_kind: str = ""  # main (主要/推進真相) | flavor (調味) -- free string
    region_id: str = ""
    sub_location_id: str = ""
    checks: list[SkillCheck] = Field(default_factory=list)
    clue_ids: list[str] = Field(default_factory=list)


class PlayerCharacter(BaseModel):
    id: str = "player"
    name: str = ""
    identity: str = ""  # 出身/門派
    stats: list[Variable] = Field(default_factory=list)  # kind="stat" entries
    starting_items: list[str] = Field(default_factory=list)  # Item.id
    origin: str = ""
    weakness: str = ""
    token_item_id: str = ""  # Item.id -- the player's defining token/keepsake
    relation_to_core_event: str = ""


class Item(BaseModel):
    id: str
    name: str
    description: str = ""
    acquired_in_event_id: str = ""  # "" = held from the start


class Quest(BaseModel):
    id: str
    name: str
    objective: str = ""
    giver_npc_id: str = ""
    start_event_id: str = ""
    complete_event_id: str = ""
    event_ids: list[str] = Field(default_factory=list)


class Script(BaseModel):
    title: str
    premise: str
    variables: list[Variable] = Field(default_factory=list)
    npcs: list[NPC] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    player: PlayerCharacter | None = None
    items: list[Item] = Field(default_factory=list)
    quests: list[Quest] = Field(default_factory=list)
    theme: str = ""
    goal: str = ""
    tone: str = ""
    factions: list[Faction] = Field(default_factory=list)
    regions: list[Region] = Field(default_factory=list)
    truth: TruthLayer | None = None
    stat_thresholds: list[StatThreshold] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    clues: list[Clue] = Field(default_factory=list)
    endings: list[Ending] = Field(default_factory=list)


def validate_references(script: Script) -> list[str]:
    """Cross-reference checks pydantic's field-level validation can't do:
    every dialogue.npc_id and branch.next_event_id must point at something
    that actually exists in the script (plus the RPG entities below).
    Returns a list of human-readable problem descriptions (empty list =
    fully consistent).

    This is what the 校對 agent (proofreader) runs to check the writer/
    dialogue agents' output before it's accepted as final -- both the
    legacy repair loop (pipeline.py::_repair) and the layered orchestrator
    re-run this same function, so extending it here automatically extends
    both repair loops without touching their code.
    """
    problems: list[str] = []

    npc_ids = {npc.id for npc in script.npcs}
    event_ids = {event.id for event in script.events}
    item_ids = {item.id for item in script.items}
    quest_ids = {quest.id for quest in script.quests}
    variable_ids = {var.id for var in script.variables}
    player_ids = {script.player.id} if script.player else set()
    stat_ids = {stat.id for stat in script.player.stats} if script.player else set()
    faction_ids = {faction.id for faction in script.factions}
    region_ids = {region.id for region in script.regions}
    sub_location_ids = {
        sub.id for region in script.regions for sub in region.sub_locations
    }
    chapter_ids = {chapter.id for chapter in script.chapters}
    clue_ids = {clue.id for clue in script.clues}
    ending_ids = {ending.id for ending in script.endings}
    branch_ids = {
        branch.id for event in script.events for branch in event.branches
    }

    dialogue_target_ids = npc_ids | player_ids

    for event in script.events:
        for line in event.dialogue:
            if line.npc_id not in dialogue_target_ids:
                problems.append(
                    f"event {event.id!r}: dialogue references unknown npc_id {line.npc_id!r}"
                )
        if event.quest_id and event.quest_id not in quest_ids:
            problems.append(
                f"event {event.id!r}: unknown quest_id {event.quest_id!r}"
            )
        if event.chapter_id and event.chapter_id not in chapter_ids:
            problems.append(
                f"event {event.id!r}: unknown chapter_id {event.chapter_id!r}"
            )
        if event.region_id and event.region_id not in region_ids:
            problems.append(
                f"event {event.id!r}: unknown region_id {event.region_id!r}"
            )
        if event.sub_location_id and event.sub_location_id not in sub_location_ids:
            problems.append(
                f"event {event.id!r}: unknown sub_location_id {event.sub_location_id!r}"
            )
        for clue_id in event.clue_ids:
            if clue_id not in clue_ids:
                problems.append(
                    f"event {event.id!r}: clue_ids references unknown clue {clue_id!r}"
                )
        for check in event.checks:
            if check.stat_id and check.stat_id not in stat_ids:
                problems.append(
                    f"event {event.id!r}: check {check.id!r} references unknown "
                    f"stat_id {check.stat_id!r}"
                )
            if check.success_next_event_id and check.success_next_event_id not in event_ids:
                problems.append(
                    f"event {event.id!r}: check {check.id!r} references unknown "
                    f"success_next_event_id {check.success_next_event_id!r}"
                )
            if check.failure_branch_id and check.failure_branch_id not in branch_ids:
                problems.append(
                    f"event {event.id!r}: check {check.id!r} references unknown "
                    f"failure_branch_id {check.failure_branch_id!r}"
                )
            if check.item_bypass_id and check.item_bypass_id not in item_ids:
                problems.append(
                    f"event {event.id!r}: check {check.id!r} references unknown "
                    f"item_bypass_id {check.item_bypass_id!r}"
                )
        for branch in event.branches:
            if branch.next_event_id not in event_ids:
                problems.append(
                    f"event {event.id!r}: branch {branch.id!r} points to unknown "
                    f"next_event_id {branch.next_event_id!r}"
                )
            if branch.payoff_chapter_id and branch.payoff_chapter_id not in chapter_ids:
                problems.append(
                    f"event {event.id!r}: branch {branch.id!r} references unknown "
                    f"payoff_chapter_id {branch.payoff_chapter_id!r}"
                )
            if branch.converges_to_event_id and branch.converges_to_event_id not in event_ids:
                problems.append(
                    f"event {event.id!r}: branch {branch.id!r} references unknown "
                    f"converges_to_event_id {branch.converges_to_event_id!r}"
                )
            for op in branch.effect_ops:
                target_ids_by_kind = {
                    "variable": variable_ids,
                    "stat": stat_ids,
                    "item": item_ids,
                    "quest": quest_ids,
                }
                valid_ids = target_ids_by_kind.get(op.target_kind)
                if valid_ids is None:
                    problems.append(
                        f"event {event.id!r}: branch {branch.id!r} effect_op has "
                        f"unknown target_kind {op.target_kind!r}"
                    )
                elif op.target_id not in valid_ids:
                    problems.append(
                        f"event {event.id!r}: branch {branch.id!r} effect_op "
                        f"references unknown {op.target_kind} {op.target_id!r}"
                    )

    for faction in script.factions:
        for relation in faction.relations:
            if relation.faction_id not in faction_ids:
                problems.append(
                    f"faction {faction.id!r}: relation references unknown "
                    f"faction {relation.faction_id!r}"
                )

    for threshold in script.stat_thresholds:
        if threshold.stat_id not in stat_ids:
            problems.append(
                f"stat_threshold {threshold.id!r}: unknown stat_id {threshold.stat_id!r}"
            )
        unlocks_ids_by_kind = {
            "branch": branch_ids,
            "event": event_ids,
            "npc_attitude": npc_ids,
            "ending": ending_ids,
        }
        valid_unlock_ids = unlocks_ids_by_kind.get(threshold.unlocks_kind)
        if valid_unlock_ids is None:
            problems.append(
                f"stat_threshold {threshold.id!r}: unknown unlocks_kind "
                f"{threshold.unlocks_kind!r}"
            )
        elif threshold.unlocks_id not in valid_unlock_ids:
            problems.append(
                f"stat_threshold {threshold.id!r}: references unknown "
                f"{threshold.unlocks_kind} {threshold.unlocks_id!r}"
            )

    for chapter in script.chapters:
        if chapter.converge_event_id and chapter.converge_event_id not in event_ids:
            problems.append(
                f"chapter {chapter.id!r}: unknown converge_event_id "
                f"{chapter.converge_event_id!r}"
            )
        for eid in chapter.event_ids:
            if eid not in event_ids:
                problems.append(
                    f"chapter {chapter.id!r}: event_ids references unknown event {eid!r}"
                )
        for clue_id in chapter.clue_ids:
            if clue_id not in clue_ids:
                problems.append(
                    f"chapter {chapter.id!r}: clue_ids references unknown clue {clue_id!r}"
                )

    for clue in script.clues:
        if clue.found_in_event_id and clue.found_in_event_id not in event_ids:
            problems.append(
                f"clue {clue.id!r}: unknown found_in_event_id {clue.found_in_event_id!r}"
            )

    for ending in script.endings:
        for condition in ending.stat_conditions:
            if condition.stat_id not in stat_ids:
                problems.append(
                    f"ending {ending.id!r}: stat_conditions references unknown "
                    f"stat_id {condition.stat_id!r}"
                )
        for branch_id in ending.required_branch_ids:
            if branch_id not in branch_ids:
                problems.append(
                    f"ending {ending.id!r}: required_branch_ids references unknown "
                    f"branch {branch_id!r}"
                )

    if script.truth:
        for reveal in script.truth.progressive:
            if reveal.reveal_chapter_id and reveal.reveal_chapter_id not in chapter_ids:
                problems.append(
                    f"progressive_reveal {reveal.id!r}: unknown reveal_chapter_id "
                    f"{reveal.reveal_chapter_id!r}"
                )
            if reveal.reveal_event_id and reveal.reveal_event_id not in event_ids:
                problems.append(
                    f"progressive_reveal {reveal.id!r}: unknown reveal_event_id "
                    f"{reveal.reveal_event_id!r}"
                )

    for npc in script.npcs:
        if npc.faction_id and npc.faction_id not in faction_ids:
            problems.append(
                f"npc {npc.id!r}: unknown faction_id {npc.faction_id!r}"
            )
        if npc.first_appearance_event_id and npc.first_appearance_event_id not in event_ids:
            problems.append(
                f"npc {npc.id!r}: unknown first_appearance_event_id "
                f"{npc.first_appearance_event_id!r}"
            )

    for item in script.items:
        if item.acquired_in_event_id and item.acquired_in_event_id not in event_ids:
            problems.append(
                f"item {item.id!r}: unknown acquired_in_event_id "
                f"{item.acquired_in_event_id!r}"
            )

    for quest in script.quests:
        if quest.giver_npc_id and quest.giver_npc_id not in npc_ids:
            problems.append(
                f"quest {quest.id!r}: unknown giver_npc_id {quest.giver_npc_id!r}"
            )
        for field_name in ("start_event_id", "complete_event_id"):
            value = getattr(quest, field_name)
            if value and value not in event_ids:
                problems.append(
                    f"quest {quest.id!r}: unknown {field_name} {value!r}"
                )
        for eid in quest.event_ids:
            if eid not in event_ids:
                problems.append(
                    f"quest {quest.id!r}: event_ids references unknown event {eid!r}"
                )

    if script.player:
        for item_id in script.player.starting_items:
            if item_id not in item_ids:
                problems.append(
                    f"player: starting_items references unknown item {item_id!r}"
                )
        if script.player.token_item_id and script.player.token_item_id not in item_ids:
            problems.append(
                f"player: unknown token_item_id {script.player.token_item_id!r}"
            )

    return problems


def validate_npc_introductions(script: Script) -> list[str]:
    """NPC-introduction consistency, kept separate from validate_references()
    so it doesn't feed the existing repair loops (which assume any problem
    they see is worth an LLM repair pass) -- this is intended for the
    guardrails module instead, run at task-completion time, not after the
    whole script is assembled.

    Two checks, using Script.events' array order as the event sequence
    (the schema has no other ordering signal):
    - an NPC's first line of dialogue must occur no earlier than the event
      named by its first_appearance_event_id, if that field is set (an NPC
      introduced *after* they've already spoken is the bug this catches --
      being introduced earlier than their first line, or in that same
      event, is normal and not flagged);
    - an NPC with dialogue but no first_appearance_event_id set at all, and
      no introduction text, is flagged as an unintroduced NPC.

    NPCs that never speak are not checked -- nothing to introduce.
    """
    problems: list[str] = []

    event_index = {event.id: i for i, event in enumerate(script.events)}
    first_speaking_event: dict[str, str] = {}
    for event in script.events:
        for line in event.dialogue:
            first_speaking_event.setdefault(line.npc_id, event.id)

    npcs_by_id = {npc.id: npc for npc in script.npcs}

    for npc_id, event_id in first_speaking_event.items():
        npc = npcs_by_id.get(npc_id)
        if npc is None:
            continue  # unknown npc_id is validate_references()'s job
        if not npc.first_appearance_event_id and not npc.introduction:
            problems.append(
                f"npc {npc.id!r}: speaks in event {event_id!r} but has no "
                "first_appearance_event_id/introduction"
            )
            continue
        intro_id = npc.first_appearance_event_id
        if not intro_id or intro_id not in event_index or event_id not in event_index:
            continue  # dangling id is validate_references()'s job
        if event_index[intro_id] > event_index[event_id]:
            problems.append(
                f"npc {npc.id!r}: first speaks in event {event_id!r} (index "
                f"{event_index[event_id]}) but first_appearance_event_id "
                f"{intro_id!r} comes later (index {event_index[intro_id]})"
            )

    return problems


def validate_stat_thresholds(script: Script) -> list[str]:
    """Narrative-quality check for 陣營值/數值門檻表, kept separate from
    validate_references() for the same reason as validate_npc_introductions()
    -- coverage/overlap/unlocks-something are judgment calls a repair loop
    shouldn't be trusted to fix, not dangling-id bugs. Consumed by the
    guardrails module (check_stat_narrative) instead.

    Three checks:
    - every stat targeted by a branch's structured effect has at least one
      declared StatThreshold covering it (a stat with no threshold is purely
      decorative -- no narrative meaning attached);
    - no two thresholds for the same stat declare overlapping ranges;
    - every threshold actually unlocks something (non-empty unlocks_kind/id).
    """
    problems: list[str] = []

    targeted_stats: set[str] = {
        op.target_id
        for event in script.events
        for branch in event.branches
        for op in branch.effect_ops
        if op.target_kind == "stat"
    }

    thresholds_by_stat: dict[str, list[StatThreshold]] = {}
    for threshold in script.stat_thresholds:
        thresholds_by_stat.setdefault(threshold.stat_id, []).append(threshold)
        if not threshold.unlocks_kind or not threshold.unlocks_id:
            problems.append(
                f"stat_threshold {threshold.id!r}: does not unlock anything "
                "(missing unlocks_kind/unlocks_id)"
            )

    for stat_id in sorted(targeted_stats):
        if stat_id not in thresholds_by_stat:
            problems.append(
                f"stat {stat_id!r}: modified by a branch effect but has no "
                "declared stat_threshold -- no narrative meaning attached"
            )

    for stat_id, thresholds in thresholds_by_stat.items():
        ranged = sorted(
            thresholds,
            key=lambda t: (t.min_value if t.min_value is not None else float("-inf")),
        )
        for a, b in zip(ranged, ranged[1:]):
            a_max = a.max_value if a.max_value is not None else float("inf")
            b_min = b.min_value if b.min_value is not None else float("-inf")
            if a_max >= b_min:
                problems.append(
                    f"stat {stat_id!r}: thresholds {a.id!r} and {b.id!r} have "
                    "overlapping ranges"
                )

    return problems


def validate_truth_layering(script: Script) -> list[str]:
    """Narrative-pacing check for TruthLayer.progressive: reveals must occur
    in non-decreasing chapter order, using script.chapters' array order as
    the chapter sequence (the schema has no other ordering signal, same
    convention as validate_npc_introductions()'s use of Script.events'
    order). Kept separate from validate_references() for the same reason --
    a repair loop can't reliably fix a pacing judgment, so this feeds the
    guardrails module instead.

    Reveals with a dangling/unresolvable reveal_chapter_id are skipped here
    (validate_references()'s job); only reveals that resolve to a real
    chapter participate in the ordering check.
    """
    problems: list[str] = []

    if script.truth is None:
        return problems

    chapter_index = {chapter.id: i for i, chapter in enumerate(script.chapters)}

    ordered_reveals = [
        reveal
        for reveal in script.truth.progressive
        if reveal.reveal_chapter_id and reveal.reveal_chapter_id in chapter_index
    ]

    for prev, nxt in zip(ordered_reveals, ordered_reveals[1:]):
        if chapter_index[prev.reveal_chapter_id] > chapter_index[nxt.reveal_chapter_id]:
            problems.append(
                f"progressive_reveal {nxt.id!r}: reveal_chapter_id "
                f"{nxt.reveal_chapter_id!r} comes before preceding reveal "
                f"{prev.id!r}'s chapter {prev.reveal_chapter_id!r} -- reveals must "
                "be in non-decreasing chapter order"
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


# --- Layered generation models -------------------------------------------
#
# These support the "分層生成" pipeline (extractor -> beat_expander ->
# scene_writer, see crew/pipeline.py::run_layered_pipeline) that decomposes
# the single writer agent's "outline + branches + variables + NPCs in one
# shot" responsibility into stages with their own persistable, independently
# retryable output. They are additive: the existing `Script`/`Event`/etc.
# models above are the pipeline's final output format either way, and
# nothing here changes their behavior.


class Chapter(BaseModel):
    id: str
    title: str
    summary: str
    beat_ids: list[str] = Field(default_factory=list)
    hook: str = ""
    event_ids: list[str] = Field(default_factory=list)
    converge_event_id: str = ""
    clue_ids: list[str] = Field(default_factory=list)


class Outline(BaseModel):
    title: str
    premise: str
    chapters: list[Chapter] = Field(default_factory=list)


class Beat(BaseModel):
    id: str
    chapter_id: str
    summary: str  # 3-8 景/章 per beat, expands into one Event when written
    npc_ids: list[str] = Field(default_factory=list)
    causal_deps: list[str] = Field(default_factory=list)  # ids of prerequisite beats/events
    # main (主要/推進真相) | flavor (調味) -- the beat_expander's own guess at
    # what the eventual Event.scene_kind should be, checkable at beat-expand
    # time (before any Event exists) via crew/guardrails.py's beat-expand
    # guardrail. Free string, same degrade-not-crash convention as
    # Event.scene_kind.
    scene_kind: str = ""


class BeatSheet(BaseModel):
    """beat_expander's output: an Outline plus the Beats that flesh it out.
    Wrapped in one model (rather than returning a bare list) because
    `Task.output_pydantic` takes a single model class, and this is also the
    natural on-disk checkpoint unit for a `beats.json` (see orchestrator.py)."""

    outline: Outline
    beats: list[Beat] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """extractor's output: the cast/variables/items/quests pulled out of the
    raw user requirement, before any beat/scene structure exists.

    orchestrator.py::_assemble_script() copies player/items/quests straight
    into the final Script -- this is what makes them survive past the
    extraction stage (props used to be extracted here and then silently
    dropped; see CLAUDE.md's script-generation section for that history)."""

    npcs: list[NPC] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)
    player: PlayerCharacter | None = None
    items: list[Item] = Field(default_factory=list)
    quests: list[Quest] = Field(default_factory=list)
    # Deprecated: superseded by `items` above, which extractor prompts
    # should populate instead. Kept only so old checkpoints/tests that
    # still set it don't fail to parse; never read downstream.
    props: list[str] = Field(default_factory=list)
    branch_candidates: list[str] = Field(default_factory=list)
    theme: str = ""
    goal: str = ""
    tone: str = ""
    factions: list[Faction] = Field(default_factory=list)
    regions: list[Region] = Field(default_factory=list)
    truth: TruthLayer | None = None
    stat_thresholds: list[StatThreshold] = Field(default_factory=list)
    clues: list[Clue] = Field(default_factory=list)
    endings: list[Ending] = Field(default_factory=list)


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
    # Player/item/quest context for RPG-shaped scene writing, and which
    # NPCs have already been introduced by an earlier committed scene --
    # all list[str]/str so none can accidentally validate as a Beat (see
    # docstring above). Must stay before current_beat.
    player_card: list[str] = Field(default_factory=list)
    item_cards: list[str] = Field(default_factory=list)
    quest_cards: list[str] = Field(default_factory=list)
    introduced_npc_ids: list[str] = Field(default_factory=list)
    # GMUD world context, same str/list[str]-only constraint as above.
    # truth_unlocked carries only progressive reveals already unlocked by
    # the current beat's chapter -- this field (and this class) never
    # carries TruthLayer.hidden at all, so a scene prompt has no way to see
    # a hidden fact regardless of prompt-following (see context_builder.py).
    faction_cards: list[str] = Field(default_factory=list)
    threshold_card: list[str] = Field(default_factory=list)
    chapter_card: list[str] = Field(default_factory=list)
    region_card: list[str] = Field(default_factory=list)
    truth_public: list[str] = Field(default_factory=list)
    truth_unlocked: list[str] = Field(default_factory=list)
    current_beat: Beat
