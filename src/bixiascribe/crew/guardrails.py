"""RPG-shape guardrails for CrewAI Task(guardrail=...) checks.

Prior to this module, "should this script feel like an RPG script and not a
novel outline" was purely a prompt-wording hope -- nothing checked that a
model actually filled in a player/items/quests, and nothing stopped it from
inventing a fake "npc_player"/"npc_narrator" NPC as a workaround for the
schema having no real player concept (a symptom actually observed in
out/eval/*.json runs before this module existed). schema.validate_references()
only checks that ids that *are* present point somewhere real -- it can't
tell you a required field was skipped entirely, or that a model routed
around a missing concept by faking an NPC for it.

These functions are pure and offline (no crewai/LLM import), matching the
style of crew/causal.py -- cheap to unit test, and importable from
tasks.py/orchestrator.py without pulling in a crewai dependency chain.
Every check function returns a list[str] of human-readable problems (empty
= passes), the same convention as schema.py's validate_* functions, so
`as_feedback()` can uniformly turn any of them into one Chinese repair
instruction for a CrewAI guardrail's (False, feedback) contract.

Guardrails must be disabled entirely under LLM_BACKEND=fake -- see
crew/tasks.py's wiring -- because llm.py's FakeLLM canned responses don't
happen to satisfy these checks, and a guardrail retry loop against a
backend that can never fix the thing it's told to fix would just spin
guardrail_max_retries times on every offline/fake test run.
"""
from __future__ import annotations

import re

from ..schema import (
    Event,
    ExtractionResult,
    Script,
    validate_npc_introductions,
    validate_references,
)

# Heuristic name/id fragments that indicate a model routed around the
# missing player/narrator concept by faking an NPC for it -- the concrete
# failure mode this module exists to catch (see module docstring).
# Deliberately doesn't include "說書人" (storyteller) -- unlike "旁白"
# (narrator), that's a legitimate wuxia teahouse NPC profession in its own
# right, not just a narrator stand-in, so including it would false-positive
# on a perfectly valid script.
_FAKE_ROLE_PATTERNS = re.compile(
    r"player|玩家|narrator|旁白|讀者", re.IGNORECASE
)


def _looks_like_fake_role(npc_id: str, npc_name: str) -> bool:
    return bool(_FAKE_ROLE_PATTERNS.search(npc_id) or _FAKE_ROLE_PATTERNS.search(npc_name))


def check_extraction_rpg(extraction: ExtractionResult) -> list[str]:
    """Guardrail for make_extract_task: the extractor is where player/
    items/quests first get a chance to exist -- catching a missing one
    here is cheaper than catching it after beat expansion has already run
    on top of an incomplete extraction."""
    problems: list[str] = []

    if extraction.player is None:
        problems.append("缺少 player（玩家角色）欄位")
    elif len(extraction.player.stats) < 2:
        problems.append("player.stats（玩家屬性）少於 2 個，至少要有數值型屬性如內力/聲望/銀兩")

    if not extraction.items:
        problems.append("items（道具）為空，至少要有 1 件關鍵道具")

    if not extraction.quests:
        problems.append("quests（任務）為空，至少要有 1 條主線任務")

    for npc in extraction.npcs:
        if _looks_like_fake_role(npc.id, npc.name):
            problems.append(
                f"npc {npc.id!r}（{npc.name!r}）疑似是假冒的玩家/旁白角色——"
                "玩家請放進 player 欄位，不要放進 npcs"
            )

    return problems


def check_script_rpg(script: Script) -> list[str]:
    """Guardrail for make_writer_task (legacy pipeline): the writer
    produces the whole Script in one shot, so this checks the full set of
    RPG entities plus everything schema.validate_references()/
    validate_npc_introductions() already know how to check."""
    problems: list[str] = []

    if script.player is None:
        problems.append("缺少 player（玩家角色）欄位")
    elif len(script.player.stats) < 2:
        problems.append("player.stats（玩家屬性）少於 2 個，至少要有數值型屬性如內力/聲望/銀兩")

    if not script.items:
        problems.append("items（道具）為空，至少要有 1 件關鍵道具")
    else:
        event_ids = {event.id for event in script.events}
        item_ids = {item.id for item in script.items}
        referenced_item_ids = {
            op.target_id
            for event in script.events
            for branch in event.branches
            for op in branch.effect_ops
            if op.target_kind == "item"
        }
        # acquired_in_event_id == "" means "held from the start" per
        # schema.Item's own docstring, not "unreachable"; a non-empty value
        # only counts if it actually names a real event (a dangling id is
        # separately flagged by validate_references() below).
        referenced_item_ids |= {
            item.id
            for item in script.items
            if not item.acquired_in_event_id or item.acquired_in_event_id in event_ids
        }
        if script.player:
            referenced_item_ids |= set(script.player.starting_items)
        unreachable = item_ids - referenced_item_ids
        if unreachable:
            problems.append(
                f"以下道具沒有任何事件能取得，玩家永遠拿不到：{sorted(unreachable)}"
            )

    if not script.quests:
        problems.append("quests（任務）為空，至少要有 1 條主線任務")
    else:
        event_ids = {event.id for event in script.events}
        for quest in script.quests:
            if not quest.event_ids or not any(eid in event_ids for eid in quest.event_ids):
                problems.append(
                    f"quest {quest.id!r} 的 event_ids 沒有對應到任何實際存在的 event"
                )

    for npc in script.npcs:
        if _looks_like_fake_role(npc.id, npc.name):
            problems.append(
                f"npc {npc.id!r}（{npc.name!r}）疑似是假冒的玩家/旁白角色——"
                "玩家請放進 player 欄位，不要放進 npcs，敘述請放進 event.summary"
            )
        if not npc.first_appearance_event_id:
            problems.append(f"npc {npc.id!r} 缺少 first_appearance_event_id（首次登場事件）")

    problems.extend(validate_references(script))
    problems.extend(validate_npc_introductions(script))

    return problems


def check_scene_rpg(
    event: Event,
    *,
    known_npc_ids: set[str],
    introduced_npc_ids: set[str],
) -> list[str]:
    """Guardrail for make_scene_write_task (layered pipeline): one Event at
    a time, so this can only check what's local to this scene plus
    introduction ordering against scenes already committed before it
    (introduced_npc_ids, from already-committed Events -- not this one).

    known_npc_ids scopes the check to NPCs the scene_writer was actually
    handed (SessionDocument.character_cards's subset), so a scene isn't
    penalized for not introducing an NPC it was never told about.
    """
    problems: list[str] = []

    if not event.dialogue:
        problems.append(f"event {event.id!r} 沒有任何台詞")

    # An NPC that neither appears in known_npc_ids (this scene's cast, per
    # SessionDocument.character_cards) nor was already introduced by an
    # earlier committed scene is speaking without ever having been
    # introduced to this scene_writer call at all.
    for line in event.dialogue:
        if line.npc_id not in known_npc_ids and line.npc_id not in introduced_npc_ids:
            problems.append(
                f"event {event.id!r}: npc_id {line.npc_id!r} 說話，但不在本場戲的登場"
                "名單也未曾在先前場次登場過"
            )

    return problems


def as_feedback(problems: list[str]) -> str:
    """Render a list of problems (from any check_* function above) as one
    Chinese repair instruction for a CrewAI Task guardrail's feedback
    string -- what the agent sees on its next retry attempt."""
    bullet_list = "\n".join(f"- {p}" for p in problems)
    return (
        "以下 RPG 遊戲性要求未滿足，請修正後重新產出完整結果（不要只回覆說明文字）：\n"
        f"{bullet_list}"
    )
