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

import difflib
import re

from ..schema import (
    BeatSheet,
    Event,
    ExtractionResult,
    Script,
    validate_npc_introductions,
    validate_references,
    validate_stat_thresholds,
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


def _similar_choice_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def check_choice_quality(event: Event) -> list[str]:
    """Guardrail for 抉擇點設計三原則: every branch with a structured effect
    must declare a `cost` (what the player gives up, not merely a stat
    delta), and two branches within the same event must not be a 假選擇 --
    highly similar choice text, the same structured-effect targets, and no
    distinct cost from one another."""
    problems: list[str] = []

    for branch in event.branches:
        if branch.effect_ops and not branch.cost:
            problems.append(
                f"event {event.id!r} branch {branch.id!r}: 有結構化效果但缺少 cost（代價）"
            )

    for i, a in enumerate(event.branches):
        for b in event.branches[i + 1:]:
            a_targets = {(op.target_kind, op.target_id) for op in a.effect_ops}
            b_targets = {(op.target_kind, op.target_id) for op in b.effect_ops}
            shared_targets = bool(a_targets & b_targets)
            same_cost = (a.cost or "") == (b.cost or "")
            if shared_targets and same_cost and _similar_choice_text(a.choice_text, b.choice_text):
                problems.append(
                    f"event {event.id!r}: 選項 {a.id!r} 與 {b.id!r} 疑似假選擇（文字相似、"
                    "效果目標重疊、代價未區分）"
                )

    return problems


def check_delayed_payoff(script: Script) -> list[str]:
    """Guardrail for delayed effects: a branch's structured effect that
    isn't resolved with immediate_feedback in its own event must declare a
    payoff_chapter_id/payoff_description, and if it names a payoff chapter,
    that chapter must not come before the branch's own chapter."""
    problems: list[str] = []
    chapter_index = {chapter.id: i for i, chapter in enumerate(script.chapters)}

    for event in script.events:
        for branch in event.branches:
            if not branch.effect_ops or branch.immediate_feedback:
                continue
            if not branch.payoff_chapter_id and not branch.payoff_description:
                problems.append(
                    f"event {event.id!r} branch {branch.id!r}: 效果延遲生效但未宣告 "
                    "payoff_chapter_id/payoff_description"
                )
                continue
            if (
                branch.payoff_chapter_id in chapter_index
                and event.chapter_id in chapter_index
                and chapter_index[branch.payoff_chapter_id] < chapter_index[event.chapter_id]
            ):
                problems.append(
                    f"event {event.id!r} branch {branch.id!r}: payoff_chapter_id "
                    f"{branch.payoff_chapter_id!r} 早於本身所在章節 {event.chapter_id!r}"
                )

    return problems


def check_stat_narrative(script: Script) -> list[str]:
    """Guardrail wrapping schema.validate_stat_thresholds() -- every stat a
    branch effect can change must have narrative meaning attached via a
    declared 陣營值/數值門檻 rule."""
    return validate_stat_thresholds(script)


def check_truth_pacing(script: Script, up_to_chapter: str) -> list[str]:
    """Backstop against a model restating a hidden/not-yet-revealed fact's
    substance in its own words -- structural exclusion (context_builder.py
    never handing truth.hidden to a scene prompt) is the primary defense,
    this is a substring-match safety net over already-committed content up
    to and including `up_to_chapter`.

    Not attempting semantic leak detection (see design.md's non-goals) --
    unparseable/unresolvable chapter ids never false-positive, they're
    simply skipped."""
    problems: list[str] = []
    if script.truth is None:
        return problems

    chapter_index = {chapter.id: i for i, chapter in enumerate(script.chapters)}
    limit = chapter_index.get(up_to_chapter)
    if limit is None:
        return problems

    unrevealed_facts = list(script.truth.hidden)
    for reveal in script.truth.progressive:
        reveal_idx = chapter_index.get(reveal.reveal_chapter_id)
        if reveal_idx is None or reveal_idx > limit:
            unrevealed_facts.append(reveal.fact)
    unrevealed_facts = [fact for fact in unrevealed_facts if fact]
    if not unrevealed_facts:
        return problems

    for event in script.events:
        idx = chapter_index.get(event.chapter_id)
        if idx is None or idx > limit:
            continue
        text = event.summary + " " + " ".join(line.line for line in event.dialogue)
        for fact in unrevealed_facts:
            if fact in text:
                problems.append(f"event {event.id!r}: 疑似提前洩漏私藏真相「{fact}」")

    return problems


def _reachable_events(start_ids: set[str], events_by_id: dict[str, Event]) -> set[str]:
    seen = set(start_ids)
    frontier = list(start_ids)
    while frontier:
        current = frontier.pop()
        event = events_by_id.get(current)
        if event is None:
            continue
        for branch in event.branches:
            if branch.next_event_id not in seen:
                seen.add(branch.next_event_id)
                frontier.append(branch.next_event_id)
    return seen


def check_convergence(script: Script) -> list[str]:
    """Guardrail for 收斂節點: a chapter with branching events must declare
    a converge_event_id that every branch path within the chapter can
    actually reach, and no single event should offer more than 3 choices."""
    problems: list[str] = []
    events_by_id = {event.id: event for event in script.events}

    for chapter in script.chapters:
        chapter_events = [e for e in script.events if e.chapter_id == chapter.id]
        has_branching = any(len(e.branches) > 1 for e in chapter_events)
        if not has_branching:
            continue
        if not chapter.converge_event_id:
            problems.append(f"chapter {chapter.id!r}: 有分支但未宣告 converge_event_id")
            continue
        if chapter.converge_event_id not in events_by_id:
            continue  # dangling id is validate_references()'s job
        reachable = _reachable_events({e.id for e in chapter_events}, events_by_id)
        if chapter.converge_event_id not in reachable:
            problems.append(
                f"chapter {chapter.id!r}: converge_event_id "
                f"{chapter.converge_event_id!r} 無法從章節內的分支到達"
            )

    for event in script.events:
        if len(event.branches) > 3:
            problems.append(f"event {event.id!r}: 選項數量超過 3 個（{len(event.branches)}）")

    return problems


def check_check_fallback(event: Event) -> list[str]:
    """Guardrail for skill checks: every SkillCheck must declare either a
    failure_branch_id or an item_bypass_id (so a failed check advances the
    story instead of dead-ending it), and a declared failure branch must
    come with a failure_cost."""
    problems: list[str] = []
    for check in event.checks:
        if not check.failure_branch_id and not check.item_bypass_id:
            problems.append(
                f"event {event.id!r} check {check.id!r}: 沒有失敗分支也沒有道具替代路線，"
                "可能是死路"
            )
        elif check.failure_branch_id and not check.failure_cost:
            problems.append(
                f"event {event.id!r} check {check.id!r}: 有失敗分支但未宣告 failure_cost"
            )
    return problems


def check_scene_information(event: Event) -> list[str]:
    """Guardrail against content-free scenes: an event should unlock at
    least a clue, an item, or some other story-state change (variable/quest
    effect), so investigative/main scenes carry information."""
    has_clue = bool(event.clue_ids)
    has_item_or_reveal = any(
        op.target_kind in ("item", "variable", "quest")
        for branch in event.branches
        for op in branch.effect_ops
    )
    if not (has_clue or has_item_or_reveal):
        return [f"event {event.id!r}: 沒有解鎖任何線索/道具/真相，可能是純填充場景"]
    return []


def check_scene_mix(script: Script) -> list[str]:
    """Guardrail for the guide's 主要場景略多於調味場景 guidance: flavor
    scenes should not outnumber main scenes."""
    main_count = sum(1 for event in script.events if event.scene_kind == "main")
    flavor_count = sum(1 for event in script.events if event.scene_kind == "flavor")
    if flavor_count > main_count:
        return [f"調味場景（{flavor_count}）多於主要場景（{main_count}），主要場景應略多"]
    return []


def check_regions(script: Script) -> list[str]:
    """Guardrail for region structure: every region needs at least 2
    functional sub-locations, and an event's sub_location_id must actually
    belong to the region it's declared under."""
    problems: list[str] = []

    for region in script.regions:
        if len(region.sub_locations) < 2:
            problems.append(f"region {region.id!r}: 少於 2 個子地點")

    sub_to_region = {
        sub.id: region.id for region in script.regions for sub in region.sub_locations
    }
    for event in script.events:
        if not event.sub_location_id:
            continue
        owning_region = sub_to_region.get(event.sub_location_id)
        if owning_region is None:
            continue  # dangling id is validate_references()'s job
        if event.region_id and owning_region != event.region_id:
            problems.append(
                f"event {event.id!r}: sub_location_id {event.sub_location_id!r} 不屬於 "
                f"宣告的 region_id {event.region_id!r}"
            )

    return problems


def check_beat_expand_rpg(beat_sheet: BeatSheet) -> list[str]:
    """Guardrail for make_beat_expand_task: the beat_expander is where
    chapter hook/convergence and each beat's intended main/flavor kind
    first get a chance to exist, before any Event does -- catching a
    missing one here is cheaper than catching it only after scene_writer
    has already run scenes on top of an under-planned outline.

    Convergence reachability itself can't be checked yet (no Events exist
    at this stage) -- that's check_convergence(script)'s job, once scenes
    exist. This only checks that a convergence point was *declared* for any
    chapter with more than one beat (a proxy for "this chapter will
    branch")."""
    problems: list[str] = []

    for chapter in beat_sheet.outline.chapters:
        if not chapter.hook:
            problems.append(f"chapter {chapter.id!r}: 缺少 hook（開場鉤子）")
        chapter_beats = [b for b in beat_sheet.beats if b.chapter_id == chapter.id]
        if len(chapter_beats) > 1 and not chapter.converge_event_id:
            problems.append(
                f"chapter {chapter.id!r}: 有多個 beat（可能分支）但未宣告 converge_event_id"
            )

    main_count = sum(1 for beat in beat_sheet.beats if beat.scene_kind == "main")
    flavor_count = sum(1 for beat in beat_sheet.beats if beat.scene_kind == "flavor")
    if flavor_count > main_count:
        problems.append(f"調味 beat（{flavor_count}）多於主要 beat（{main_count}），主要場景應略多")

    return problems


def collect_quality_problems(script: Script) -> list[str]:
    """Aggregate the nine check_* functions above that are NOT wired as a
    Task guardrail (check_choice_quality/check_delayed_payoff/
    check_stat_narrative/check_truth_pacing/check_convergence/
    check_check_fallback/check_scene_information/check_scene_mix/
    check_regions) over a finished Script, for report-only visibility.

    Deliberately not retried in-loop: measured against real generated
    scripts, wiring these as guardrails would have added 10-28 extra
    findings per script on top of what already made a run fail -- narrower
    than validate_references()'s purely-mechanical checks, closer to
    validate_stat_thresholds()/validate_truth_layering()/
    validate_npc_introductions()'s "narrative-quality judgment, not
    something a repair loop should be trusted to fix blind" category.
    Called once at the end of run_pipeline_with_report()/run_layered(),
    result stored on RunReport.quality_problems -- visible in the review
    UI without gating the run."""
    problems: list[str] = []
    problems.extend(check_delayed_payoff(script))
    problems.extend(check_stat_narrative(script))
    problems.extend(check_convergence(script))
    problems.extend(check_scene_mix(script))
    problems.extend(check_regions(script))
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
