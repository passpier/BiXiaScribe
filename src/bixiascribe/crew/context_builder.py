"""Builds the compressed SessionDocument handed to each scene_writer call
(BiXiaScribe 重構 Phase 5).

Before this module, make_scene_write_task() serialized only the current
Beat plus its NPC subset -- each scene was written with zero visibility
into scenes it causally depends on, so scene N could contradict scene N-1
simply because it never saw it. build_session_document() adds that
continuity back, but under a hard token budget, so it doesn't degrade into
the unbounded-growth problem the legacy dialogue/proofread tasks' whole-
Script `context=[prior_task]` chaining has.

No CausalPlotGraph is produced anywhere yet (Phase 6's job) -- causal
ancestry here is derived from Beat.causal_deps via the BeatSheet, when one
is supplied.
"""
from __future__ import annotations

from .. import config
from ..schema import Beat, BeatSheet, Event, ExtractionResult, SessionDocument

# Per-scene summary is truncated to this many characters before it's even
# considered for the budget loop, so one runaway summary can't dominate the
# document on its own.
_MAX_SUMMARY_CHARS = 120

# CJK codepoints roughly cost 1 token each in most tokenizers; ASCII/JSON
# punctuation is closer to 1 token per ~4 characters. No tokenizer
# dependency (mirrors chunking.py's char-based measurement), biased high so
# the estimate stays conservative rather than silently overrunning.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3000, 0x303F),  # CJK punctuation
    (0xFF00, 0xFFEF),  # fullwidth forms
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """Dependency-free, CJK-aware token estimate: CJK codepoints count ~1
    token each, everything else ~1/4 token (rounded up)."""
    cjk_chars = sum(1 for ch in text if _is_cjk(ch))
    other_chars = len(text) - cjk_chars
    return cjk_chars + -(-other_chars // 4)  # ceil division for "other"


def _character_card(npc) -> str:
    return (
        f"{npc.id}｜{npc.name}（{npc.identity}）"
        f"性格：{npc.personality}；語氣：{npc.speech_style}"
    )


def _relevant_npcs(beat: Beat, extraction: ExtractionResult) -> list:
    """Same NPC-subset filter make_scene_write_task used pre-Phase-5:
    npcs named by the beat, falling back to every extracted NPC when the
    beat names none that exist."""
    matched = [npc for npc in extraction.npcs if npc.id in beat.npc_ids]
    return matched or list(extraction.npcs)


def _scene_summary(event: Event) -> str:
    summary = event.summary or ""
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "…"
    return f"{event.id}｜{event.title}：{summary}"


def _causal_ancestors(beat_id: str, beats_by_id: dict[str, Beat]) -> set[str]:
    """Transitive closure of causal_deps for `beat_id` -- the scenes this
    one could actually contradict, walked through the BeatSheet. Guards
    against cycles/self-references with a `seen` set."""
    ancestors: set[str] = set()
    frontier = [beat_id]
    seen = {beat_id}
    while frontier:
        current = frontier.pop()
        beat = beats_by_id.get(current)
        if beat is None:
            continue
        for dep in beat.causal_deps:
            if dep in seen:
                continue
            seen.add(dep)
            ancestors.add(dep)
            frontier.append(dep)
    return ancestors


def build_session_document(
    current_beat: Beat,
    extraction: ExtractionResult,
    completed_scenes: list[Event],
    *,
    beat_sheet: BeatSheet | None = None,
    graph=None,  # CausalPlotGraph, unused until Phase 6 populates one
    max_tokens: int | None = None,
) -> SessionDocument:
    """Assemble a token-bounded SessionDocument for `current_beat`.

    Priority order when trimming to fit `max_tokens`:
    1. character_cards and current_beat are never dropped.
    2. scene_summaries: causal ancestors of current_beat (via
       beat_sheet.beats[*].causal_deps) rank above unrelated scenes, and
       within each tier, more recently completed scenes rank higher.
    3. The lowest-priority summary is dropped first, incrementing
       omitted_scene_count, until the estimated size fits or nothing more
       can be dropped.

    `max_tokens=0` (or any non-positive value) disables trimming entirely --
    every completed-scene summary is kept regardless of estimated size. This
    is the "untrimmed" arm of the Phase 5 quality regression (see
    docs/BiXiaScribe_REFACTORING_PLAN.md): config.SESSION_DOC_MAX_TOKENS is
    clamped to >= 1, so the env var alone can never express "never trim" --
    only an explicit caller can. `None` (the default) still means "fall back
    to config.SESSION_DOC_MAX_TOKENS", unchanged from before.
    """
    del graph  # reserved for Phase 6 (因果一致性即時校驗)
    max_tokens = max_tokens if max_tokens is not None else config.SESSION_DOC_MAX_TOKENS
    unlimited = max_tokens <= 0

    character_cards = [_character_card(npc) for npc in _relevant_npcs(current_beat, extraction)]

    if beat_sheet is not None:
        beats_by_id = {b.id: b for b in beat_sheet.beats}
        ancestor_ids = _causal_ancestors(current_beat.id, beats_by_id)
    else:
        ancestor_ids = set()

    # completed_scenes is assumed in completion order; reverse for
    # "most recent first" within a tier.
    ancestor_scenes = [e for e in completed_scenes if e.id in ancestor_ids]
    other_scenes = [e for e in completed_scenes if e.id not in ancestor_ids]
    ordered = list(reversed(ancestor_scenes)) + list(reversed(other_scenes))
    ancestor_count = len(ancestor_scenes)

    summaries = [_scene_summary(e) for e in ordered]

    doc = SessionDocument(
        character_cards=character_cards,
        scene_summaries=list(summaries),
        omitted_scene_count=0,
        current_beat=current_beat,
    )
    del ancestor_count  # only used to build `ordered`'s priority split above

    # Drop the lowest-priority summary (the tail of `summaries`: oldest
    # non-ancestor scenes first, then oldest ancestor scenes) until the
    # document fits `max_tokens` or nothing is left to drop. Character
    # cards and current_beat are never touched by this loop.
    while (
        not unlimited
        and doc.scene_summaries
        and estimate_tokens(doc.model_dump_json()) > max_tokens
    ):
        doc.scene_summaries.pop()
        doc.omitted_scene_count += 1

    return doc
