"""Mechanical, offline repair of dangling cross-references in a generated
Script, run before schema.validate_references() so the existing LLM repair
loops (pipeline.py::_repair, the layered orchestrator's proofread tail) only
ever see problems that actually need a model's judgment.

Two problems this targets, both observed in real generation runs against
deepseek-v4-flash-0731 (see docs/DESIGN_NOTES.md's phase-zero probe):

1. `Branch.next_event_id` missing even though the branch is otherwise
   complete and coherent (cost/immediate_feedback/payoff_description/
   converges_to_event_id all filled in) -- the model clearly had somewhere
   for this branch to go, it just didn't repeat that id in the one field
   `validate_references()` actually checks. `converges_to_event_id` (or the
   branch's own chapter's `converge_event_id`, or simply "the next event in
   sequence") is a solid mechanical guess, cheaper than a full repair pass.
2. `Event.chapter_id` naming a chapter that was never declared in
   `Script.chapters` at all -- observed with every event in a run agreeing
   on the same undeclared chapter id, i.e. the model had a real chapter in
   mind and simply never emitted the Chapter object for it.

Everything else this module touches is purely annotative (clue_ids) --
clearing a dangling id there loses a cosmetic cross-reference, never
invents new narrative content, and is far cheaper than spending a
repair-pass retry on something that isn't a narrative problem.

Deliberately NOT handled here (left for validate_references() + the
existing repair loops, same "narrative-quality judgments a repair loop
shouldn't be trusted to fix mechanically" boundary as
validate_stat_thresholds()/validate_truth_layering()/
validate_npc_introductions()): dangling npc_id, item/effect_op target ids,
faction relations, stat thresholds, endings, truth layering. Those need
either real content (a modeled NPC or item that doesn't exist yet) or a
semantic judgment call this module has no basis to make.
"""
from __future__ import annotations

from ..schema import Chapter, Script


def normalize_script(script: Script) -> tuple[Script, list[str]]:
    """Return (normalized_script, notes) -- notes describes what was fixed
    for RunReport bookkeeping. `script` itself is not mutated; the returned
    Script is a deep copy with fixes applied. An empty `notes` list means
    nothing needed fixing."""
    script = script.model_copy(deep=True)
    notes: list[str] = []

    _backfill_missing_chapters(script, notes)
    _fix_next_event_ids(script, notes)
    _clear_dangling_annotations(script, notes)

    return script, notes


def _backfill_missing_chapters(script: Script, notes: list[str]) -> None:
    """If no chapter was declared at all but events agree on one or more
    chapter ids, build placeholder Chapter skeletons for them rather than
    leaving every one of those events with a dangling chapter_id. Title/
    summary/hook are left blank -- narrative content that a mechanical
    pass has no basis to invent; validate_references() no longer flags the
    id itself, and check_beat_expand_rpg-style guardrails can still flag
    the blank hook if that matters downstream."""
    if script.chapters:
        return
    referenced = sorted({event.chapter_id for event in script.events if event.chapter_id})
    for chapter_id in referenced:
        script.chapters.append(Chapter(id=chapter_id, title="", summary=""))
        notes.append(
            f"補建缺失的 chapter {chapter_id!r}"
            "（Script.chapters 為空，但多個 event 引用了這個 chapter_id）"
        )


def _fix_next_event_ids(script: Script, notes: list[str]) -> None:
    event_ids = {event.id for event in script.events}
    chapter_by_id = {chapter.id: chapter for chapter in script.chapters}
    event_order = [event.id for event in script.events]

    for index, event in enumerate(script.events):
        chapter = chapter_by_id.get(event.chapter_id) if event.chapter_id else None
        for branch in event.branches:
            if branch.next_event_id and branch.next_event_id in event_ids:
                continue

            candidate = ""
            reason = ""
            if branch.converges_to_event_id and branch.converges_to_event_id in event_ids:
                candidate = branch.converges_to_event_id
                reason = "沿用 branch.converges_to_event_id"
            elif chapter is not None and chapter.converge_event_id in event_ids:
                candidate = chapter.converge_event_id
                reason = "沿用所屬 chapter 的 converge_event_id"
            elif index + 1 < len(event_order):
                candidate = event_order[index + 1]
                reason = "回填為事件序列中的下一個 event"

            if not candidate:
                continue  # leave dangling -- validate_references() + repair loop take over

            old = branch.next_event_id
            branch.next_event_id = candidate
            notes.append(
                f"event {event.id!r} branch {branch.id!r}: next_event_id "
                f"{old!r} -> {candidate!r}（{reason}）"
            )


def _clear_dangling_annotations(script: Script, notes: list[str]) -> None:
    """Purely-annotative ids: clearing a dangling reference here costs
    nothing narratively (nothing reads clue_ids to decide plot content) and
    is strictly cheaper than spending a repair-pass retry on a cosmetic
    cross-reference."""
    clue_ids = {clue.id for clue in script.clues}

    for event in script.events:
        kept_clue_ids = [cid for cid in event.clue_ids if cid in clue_ids]
        if len(kept_clue_ids) != len(event.clue_ids):
            dropped = sorted(set(event.clue_ids) - set(kept_clue_ids))
            notes.append(f"event {event.id!r}: 清空未知的 clue_ids {dropped}")
            event.clue_ids = kept_clue_ids
