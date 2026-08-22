"""RPG-shape guardrails for CrewAI Task(guardrail=...) checks.

Prior to this module, "should this script feel like an RPG script and not a
novel outline" was purely a prompt-wording hope -- nothing checked that a
model actually filled in a player/items, and nothing stopped it from
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

Phase 4 (2026-08-22, see openspec/changes/2026-08-22-slim-script-schema-mvp):
five checks were deleted outright because their backing fields no longer
exist (check_delayed_payoff, check_stat_narrative, check_single_stat,
check_scene_mix, check_convergence -- immediate_feedback/payoff_description/
stat_thresholds/scene_kind/converge_event_id are all gone from schema.py).
This also resolved a pre-existing contradiction: check_script_rpg/
check_extraction_rpg used to demand player.stats >= 2 while (the now-deleted)
check_single_stat demanded exactly 1 -- moot now that there's a single
Stat object, not a list.
"""
from __future__ import annotations

import difflib
import re

from ..schema import BeatSheet, Event, ExtractionResult, Script, validate_references

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
    items first get a chance to exist -- catching a missing one here is
    cheaper than catching it after beat expansion has already run on top
    of an incomplete extraction."""
    problems: list[str] = []

    if extraction.player is None:
        problems.append("缺少 player（玩家角色）欄位")
    if extraction.stat is None:
        problems.append("缺少 stat（唯一數值）欄位")

    if not extraction.items:
        problems.append("items（道具）為空，至少要有 1 件關鍵道具")

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
    RPG entities plus everything schema.validate_references() already
    knows how to check."""
    problems: list[str] = []

    if script.player is None:
        problems.append("缺少 player（玩家角色）欄位")
    if script.stat is None:
        problems.append("缺少 stat（唯一數值）欄位")

    if not script.items:
        problems.append("items（道具）為空，至少要有 1 件關鍵道具")
    else:
        event_ids = {event.id for event in script.events}
        item_ids = {item.id for item in script.items}
        # from_event == "" means "held from the start" per schema.Item's
        # own docstring, not "unreachable"; a non-empty value only counts
        # if it actually names a real event (a dangling id is separately
        # flagged by validate_references() below).
        referenced_item_ids = {
            item.id for item in script.items if not item.from_event or item.from_event in event_ids
        }
        unreachable = item_ids - referenced_item_ids
        if unreachable:
            problems.append(
                f"以下道具沒有任何事件能取得，玩家永遠拿不到：{sorted(unreachable)}"
            )

    for npc in script.npcs:
        if _looks_like_fake_role(npc.id, npc.name):
            problems.append(
                f"npc {npc.id!r}（{npc.name!r}）疑似是假冒的玩家/旁白角色——"
                "玩家請放進 player 欄位，不要放進 npcs，敘述請放進 event.summary"
            )

    problems.extend(validate_references(script))

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

    if not event.preconditions:
        problems.append(
            f"event {event.id!r} 沒有任何 preconditions——至少寫出這場戲成立的前提一句話"
        )

    # An NPC that neither appears in known_npc_ids (this scene's cast, per
    # SessionDocument.character_cards) nor was already introduced by an
    # earlier committed scene is speaking without ever having been
    # introduced to this scene_writer call at all.
    for line in event.dialogue:
        if line.npc not in known_npc_ids and line.npc not in introduced_npc_ids:
            problems.append(
                f"event {event.id!r}: npc {line.npc!r} 說話，但不在本場戲的登場"
                "名單也未曾在先前場次登場過"
            )

    return problems


def _similar_choice_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def check_choice_quality(event: Event) -> list[str]:
    """Guardrail for 抉擇點設計三原則: every choice with a non-empty effects
    description must declare a `cost` (what the player gives up, not merely
    a stat delta), and two choices within the same event must not be a
    假選擇 -- highly similar choice text, the same-signed delta, and no
    distinct cost from one another."""
    problems: list[str] = []

    for choice in event.choices:
        if choice.effects and not choice.cost:
            problems.append(
                f"event {event.id!r} choice {choice.id!r}: 有效果但缺少 cost（代價）"
            )

    for i, a in enumerate(event.choices):
        for b in event.choices[i + 1:]:
            same_sign_delta = a.delta != 0 and (a.delta > 0) == (b.delta > 0)
            same_cost = (a.cost or "") == (b.cost or "")
            if same_sign_delta and same_cost and _similar_choice_text(a.text, b.text):
                problems.append(
                    f"event {event.id!r}: 選項 {a.id!r} 與 {b.id!r} 疑似假選擇（文字相似、"
                    "數值增減方向相同、代價未區分）"
                )

    return problems


def check_truth_pacing(script: Script, up_to_chapter: str) -> list[str]:
    """Backstop against a model restating a hidden/not-yet-revealed fact's
    substance in its own words -- structural exclusion (context_builder.py
    never handing truth.hidden to a scene prompt) is the primary defense,
    this is a substring-match safety net over already-committed content.

    truth.revealed's *array order* is the only pacing signal now
    (ProgressiveReveal.reveal_chapter_id no longer exists) -- this checks
    against script.truth.hidden only, since there is no per-reveal chapter
    binding left to resolve "unrevealed up to this chapter" against.
    Not attempting semantic leak detection -- unparseable/unresolvable
    chapter ids never false-positive, they're simply skipped."""
    problems: list[str] = []
    if script.truth is None or not script.truth.hidden:
        return problems

    chapter_index = {chapter.id: i for i, chapter in enumerate(script.chapters)}
    limit = chapter_index.get(up_to_chapter)
    if limit is None:
        return problems

    fact = script.truth.hidden
    for event in script.events:
        idx = chapter_index.get(event.chapter_id)
        if idx is None or idx > limit:
            continue
        text = event.summary + " " + " ".join(line.line for line in event.dialogue)
        if fact in text:
            problems.append(f"event {event.id!r}: 疑似提前洩漏私藏真相「{fact}」")

    return problems


def check_check_fallback(event: Event) -> list[str]:
    """Guardrail for the event's single skill check: it must declare
    on_fail (so a failed check advances the story instead of dead-ending
    it) with a fail_cost."""
    if event.check is None:
        return []
    problems: list[str] = []
    if not event.check.on_fail:
        problems.append(f"event {event.id!r} check: 沒有失敗路線（on_fail），可能是死路")
    elif not event.check.fail_cost:
        problems.append(f"event {event.id!r} check: 有失敗路線但未宣告 fail_cost")
    return problems


def check_scene_information(event: Event) -> list[str]:
    """Guardrail against content-free scenes: an event should unlock at
    least a clue or some other story-state change (a choice with a
    non-empty effects description), so investigative/main scenes carry
    information."""
    has_clue = bool(event.clue_ids)
    has_effect = any(choice.effects for choice in event.choices)
    if not (has_clue or has_effect):
        return [f"event {event.id!r}: 沒有解鎖任何線索/效果，可能是純填充場景"]
    return []


def check_ending_ranges(script: Script) -> list[str]:
    """Guardrail replacing the deleted stat_thresholds coverage check: with
    a single Stat and endings selected purely by value range, the ranges
    themselves must not overlap and should cover the stat's declared value
    space, or ending selection becomes ambiguous or has gaps."""
    problems: list[str] = []
    if not script.endings:
        return problems

    ranged = sorted(script.endings, key=lambda e: e.min)
    for a, b in zip(ranged, ranged[1:]):
        if a.max >= b.min:
            problems.append(
                f"ending {a.id!r} 與 {b.id!r} 的區間重疊（{a.min}-{a.max} 與 {b.min}-{b.max}）"
            )

    return problems


def check_beat_expand_rpg(beat_sheet: BeatSheet) -> list[str]:
    """Guardrail for make_beat_expand_task: catch a chapter with no beats
    assigned to it before scene_writer runs scenes on top of an
    under-planned outline."""
    problems: list[str] = []

    beat_chapter_ids = {beat.chapter_id for beat in beat_sheet.beats}
    for chapter in beat_sheet.outline.chapters:
        if chapter.id not in beat_chapter_ids:
            problems.append(f"chapter {chapter.id!r}: 沒有任何 beat 涵蓋這個章節")

    return problems


def collect_quality_problems(script: Script) -> list[str]:
    """Aggregate the check_* functions above that are NOT wired as a Task
    guardrail (check_choice_quality/check_truth_pacing/check_check_fallback/
    check_scene_information/check_ending_ranges) over a finished Script, for
    report-only visibility.

    Deliberately not retried in-loop: measured against real generated
    scripts, wiring these as guardrails would have added 10-28 extra
    findings per script on top of what already made a run fail -- narrower
    than validate_references()'s purely-mechanical checks, a
    "narrative-quality judgment, not something a repair loop should be
    trusted to fix blind" category. Called once at the end of
    run_pipeline_with_report()/run_layered(), result stored on
    RunReport.quality_problems -- visible in the review UI without gating
    the run."""
    problems: list[str] = []
    problems.extend(check_ending_ranges(script))
    if script.chapters:
        problems.extend(check_truth_pacing(script, script.chapters[-1].id))
    for event in script.events:
        problems.extend(check_choice_quality(event))
        problems.extend(check_check_fallback(event))
        problems.extend(check_scene_information(event))
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
