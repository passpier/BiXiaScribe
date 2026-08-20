"""Builds the compressed SessionDocument handed to each scene_writer call.

Without this module, a scene_writer call would only see the current Beat
plus its NPC subset -- each scene would be written with zero visibility
into scenes it causally depends on, so scene N could contradict scene N-1
simply because it never saw it. build_session_document() adds that
continuity back, but under a hard token budget, so it doesn't degrade into
the unbounded-growth problem the legacy dialogue/proofread tasks' whole-
Script `context=[prior_task]` chaining has.

Causal ancestry is derived from Beat.causal_deps via the BeatSheet, when
one is supplied, and unioned with whatever ancestry a real CausalPlotGraph
(crew/causal.py::build_graph()) adds via its edges.
The union is deliberate: both sources are best-effort (a beat_sheet can
omit a dep the graph's edges captured via branch flow, or vice versa), so
taking the union is more conservative than picking either alone.
"""
from __future__ import annotations

from .. import config
from ..schema import Beat, BeatSheet, CausalPlotGraph, Event, ExtractionResult, SessionDocument

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


def _player_card(extraction: ExtractionResult) -> list[str]:
    player = extraction.player
    if player is None:
        return []
    stats = "、".join(f"{s.name}={s.initial}" for s in player.stats)
    return [f"{player.id}｜{player.name}（{player.identity}）屬性：{stats}"]


def _item_cards(extraction: ExtractionResult) -> list[str]:
    return [f"{item.id}｜{item.name}：{item.description}" for item in extraction.items]


def _quest_cards(extraction: ExtractionResult) -> list[str]:
    return [f"{quest.id}｜{quest.name}：{quest.objective}" for quest in extraction.quests]


def _faction_card(extraction: ExtractionResult) -> list[str]:
    cards = []
    for faction in extraction.factions:
        relations = "、".join(f"{r.faction_id}:{r.stance}" for r in faction.relations)
        cards.append(f"{faction.id}｜{faction.name}（{faction.alignment}）關係：{relations}")
    return cards


def _threshold_card(extraction: ExtractionResult) -> list[str]:
    cards = []
    for threshold in extraction.stat_thresholds:
        lo = threshold.min_value if threshold.min_value is not None else "-"
        hi = threshold.max_value if threshold.max_value is not None else "-"
        cards.append(
            f"{threshold.id}｜{threshold.stat_id} [{lo},{hi}]：解鎖 "
            f"{threshold.unlocks_kind}={threshold.unlocks_id}（{threshold.description}）"
        )
    return cards


def _region_card(extraction: ExtractionResult) -> list[str]:
    cards = []
    for region in extraction.regions:
        subs = "、".join(f"{sub.name}（{sub.function}）" for sub in region.sub_locations)
        cards.append(f"{region.id}｜{region.name}：{subs}")
    return cards


def _chapter_index(beat_sheet: BeatSheet | None) -> dict[str, int]:
    if beat_sheet is None:
        return {}
    return {chapter.id: i for i, chapter in enumerate(beat_sheet.outline.chapters)}


def _chapter_card(current_beat: Beat, beat_sheet: BeatSheet | None) -> list[str]:
    if beat_sheet is None:
        return []
    chapter = next(
        (c for c in beat_sheet.outline.chapters if c.id == current_beat.chapter_id), None
    )
    if chapter is None:
        return []
    return [
        f"{chapter.id}｜{chapter.title}｜hook：{chapter.hook}｜"
        f"收斂點：{chapter.converge_event_id}"
    ]


def _truth_public(extraction: ExtractionResult) -> list[str]:
    if extraction.truth is None:
        return []
    return list(extraction.truth.public)


def _truth_unlocked(
    current_beat: Beat, extraction: ExtractionResult, beat_sheet: BeatSheet | None
) -> list[str]:
    """Progressive reveals whose chapter is <= the current beat's chapter --
    this is what makes 提前爆料私藏真相 structurally impossible, not just
    checked: TruthLayer.hidden is never read here, so no field on
    SessionDocument can ever carry it, regardless of prompt-following."""
    if extraction.truth is None or beat_sheet is None:
        return []
    chapter_index = _chapter_index(beat_sheet)
    current_idx = chapter_index.get(current_beat.chapter_id)
    if current_idx is None:
        return []
    unlocked = []
    for reveal in extraction.truth.progressive:
        reveal_idx = chapter_index.get(reveal.reveal_chapter_id)
        if reveal_idx is not None and reveal_idx <= current_idx:
            unlocked.append(f"{reveal.id}｜{reveal.fact}")
    return unlocked


def _introduced_npc_ids(completed_scenes: list[Event]) -> list[str]:
    """NPCs who have already spoken in some earlier committed scene -- used
    by check_scene_rpg (guardrails.py) to allow a scene to reference them
    without re-introducing them. Order-preserving, first-seen order."""
    seen: dict[str, None] = {}
    for event in completed_scenes:
        for line in event.dialogue:
            seen.setdefault(line.npc_id, None)
    return list(seen)


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


def _graph_ancestors(node_id: str, graph: CausalPlotGraph) -> set[str]:
    """Transitive closure of `node_id`'s ancestors via `graph`'s edges
    (from_id -> to_id, i.e. "from_id happens before to_id"). Same
    seen-set cycle guard as _causal_ancestors()."""
    incoming: dict[str, list[str]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.to_id, []).append(edge.from_id)

    ancestors: set[str] = set()
    frontier = [node_id]
    seen = {node_id}
    while frontier:
        current = frontier.pop()
        for dep in incoming.get(current, []):
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
    graph: CausalPlotGraph | None = None,
    max_tokens: int | None = None,
) -> SessionDocument:
    """Assemble a token-bounded SessionDocument for `current_beat`.

    Priority order when trimming to fit `max_tokens`:
    1. character_cards, player_card/item_cards/quest_cards,
       faction_cards/threshold_card/chapter_card/region_card/truth_public/
       truth_unlocked, and current_beat are never dropped -- only
       scene_summaries shrink.
    2. scene_summaries: causal ancestors of current_beat (via
       beat_sheet.beats[*].causal_deps, unioned with `graph`'s edges when a
       CausalPlotGraph is supplied) rank above unrelated scenes, and within
       each tier, more recently completed scenes rank higher.
    3. The lowest-priority summary is dropped first, incrementing
       omitted_scene_count, until the estimated size fits or nothing more
       can be dropped.

    `max_tokens=0` (or any non-positive value) disables trimming entirely --
    every completed-scene summary is kept regardless of estimated size. This
    is the "untrimmed" arm of the context-compression quality regression
    (see `eval/context_compression_variants.json`): config.SESSION_DOC_MAX_TOKENS
    is clamped to >= 1, so the env var alone can never express "never trim" --
    only an explicit caller can. `None` (the default) still means "fall back
    to config.SESSION_DOC_MAX_TOKENS", unchanged from before.
    """
    max_tokens = max_tokens if max_tokens is not None else config.SESSION_DOC_MAX_TOKENS
    unlimited = max_tokens <= 0

    character_cards = [_character_card(npc) for npc in _relevant_npcs(current_beat, extraction)]

    ancestor_ids: set[str] = set()
    if beat_sheet is not None:
        beats_by_id = {b.id: b for b in beat_sheet.beats}
        ancestor_ids |= _causal_ancestors(current_beat.id, beats_by_id)
    if graph is not None:
        ancestor_ids |= _graph_ancestors(current_beat.id, graph)

    # completed_scenes is assumed in completion order; reverse for
    # "most recent first" within a tier.
    ancestor_scenes = [e for e in completed_scenes if e.id in ancestor_ids]
    other_scenes = [e for e in completed_scenes if e.id not in ancestor_ids]
    ordered = list(reversed(ancestor_scenes)) + list(reversed(other_scenes))

    summaries = [_scene_summary(e) for e in ordered]

    doc = SessionDocument(
        character_cards=character_cards,
        scene_summaries=summaries,
        omitted_scene_count=0,
        player_card=_player_card(extraction),
        item_cards=_item_cards(extraction),
        quest_cards=_quest_cards(extraction),
        introduced_npc_ids=_introduced_npc_ids(completed_scenes),
        faction_cards=_faction_card(extraction),
        threshold_card=_threshold_card(extraction),
        chapter_card=_chapter_card(current_beat, beat_sheet),
        region_card=_region_card(extraction),
        truth_public=_truth_public(extraction),
        truth_unlocked=_truth_unlocked(current_beat, extraction, beat_sheet),
        current_beat=current_beat,
    )

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
