"""Structural quality proxies for a generated Script -- pure functions over
schema.py, no crew/agent involved, so they're unit-testable on hand-built
Scripts. Deliberately cheap and deterministic (counts/coverage/ratios), not
an LLM-as-judge prose score -- judging whether dialogue actually *sounds*
武俠 is left to a human reading scripts/eval_generation.py's saved output --
see that script's module docstring for why.

`continuity_metrics()` extends this with five more deterministic,
structural proxies -- still not LLM-as-judge -- aimed specifically at
whether the scene writer had visibility into scenes it causally depends
on. They exist for the layered pipeline's compressed-vs-untrimmed-context
quality regression (see `eval/context_compression_variants.json`): the
original nine
metrics are all driven by the beat sheet (event/NPC counts) or by per-event
choices that don't depend on prior scenes, so none of them can tell whether
context compression hurt continuity. The five below are computed only over
what a scene writer actually authors -- Event.title/.summary/.location,
dialogue lines, and branches -- and are annotated below with which
direction ("higher is better" / "lower is better") indicates less memory
loss across scenes.

`gmud_metrics()` adds nine more, aimed at the GMUD script frame this
repository's schema.py gained (factions/stat thresholds/regions/truth
layers/chapters/skill checks/endings, and the cost/payoff/convergence shape
on branches) -- whether a generated script actually carries that structure
rather than leaving it at its (additive, backward-compatible) empty
default. See gmud_metrics()'s own docstring for each metric.
"""
from __future__ import annotations

import re

from ..schema import Script

# Strip whitespace and common CJK/ASCII punctuation before comparing text
# for near-duplication -- two lines that only differ by a trailing "。" vs
# "！" shouldn't count as distinct.
_PUNCT_RE = re.compile(
    r"[\s，。！？、；：「」『』（）()\[\]{}\"'.,!?;:\-—～~…]+"
)


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text.strip().casefold())


def script_metrics(script: Script) -> dict[str, float | int]:
    """Structural counts and coverage ratios for one generated Script.
    Ratios are 0.0 (not None) when their denominator is empty, so callers
    can average this dict across many runs without filtering out Nones."""
    events = script.events
    npcs = script.npcs

    n_events = len(events)
    n_npcs = len(npcs)
    n_variables = len(script.variables)
    n_branches = sum(len(e.branches) for e in events)
    n_dialogue_lines = sum(len(e.dialogue) for e in events)

    events_with_dialogue = sum(1 for e in events if e.dialogue)
    speaking_npc_ids = {line.npc_id for e in events for line in e.dialogue}

    all_lines = [line.line for e in events for line in e.dialogue]
    avg_line_chars = sum(len(line) for line in all_lines) / len(all_lines) if all_lines else 0.0

    return {
        "events": n_events,
        "npcs": n_npcs,
        "variables": n_variables,
        "branches": n_branches,
        "dialogue_lines": n_dialogue_lines,
        "dialogue_lines_per_event": n_dialogue_lines / n_events if n_events else 0.0,
        "events_with_dialogue_pct": events_with_dialogue / n_events if n_events else 0.0,
        "npc_speaking_pct": len(speaking_npc_ids & {n.id for n in npcs}) / n_npcs
        if n_npcs
        else 0.0,
        "avg_line_chars": avg_line_chars,
        **continuity_metrics(script),
        **gmud_metrics(script),
    }


def continuity_metrics(script: Script) -> dict[str, float]:
    """Five structural proxies for cross-scene continuity -- whether later
    events read as if the writer remembered earlier ones. Ratios are 0.0
    when their denominator is empty, same convention as script_metrics().

    - distinct_event_title_pct (higher is better): fraction of events with
      a normalized-unique title. A writer with no memory of prior scenes
      tends to reuse generic titles (初遇/對峙/決戰).
    - repeated_dialogue_pct (lower is better): fraction of dialogue lines
      that are a normalized duplicate of another line anywhere in the
      script. Verbatim reuse across events is the classic memoryless
      symptom -- the writer reinvents the same greeting/threat each time.
    - prior_entity_reference_pct (higher is better, the headline metric):
      of events at index >= 1, the fraction whose summary/dialogue/branch
      text names an NPC who spoke in some earlier event but not in this
      one, or a location used in some earlier, different event. Excluding
      NPCs present in the scoring event is what makes this a *callback*
      measure, not a restatement of the character cards (which are never
      trimmed by build_session_document() and so can't discriminate here).
    - connected_event_pct (higher is better): fraction of events targeted
      by some *other* event's branch. validate_references() already
      guarantees branch targets aren't dangling, so this measures
      connectivity density, not validity -- orphan events are what a
      writer produces when it can't see the surrounding graph.
    - self_loop_branch_pct (lower is better): fraction of branches whose
      next_event_id points back at their own event -- a degenerate
      fallback a memoryless writer reaches for when it has no other valid
      id to name, and one validate_references() happily accepts.
    """
    events = script.events
    n_events = len(events)

    # distinct_event_title_pct
    titles = [_normalize(e.title) for e in events]
    distinct_event_title_pct = len(set(titles)) / n_events if n_events else 0.0

    # repeated_dialogue_pct
    all_lines = [_normalize(line.line) for e in events for line in e.dialogue]
    n_lines = len(all_lines)
    repeated_dialogue_pct = (n_lines - len(set(all_lines))) / n_lines if n_lines else 0.0

    # prior_entity_reference_pct
    npc_names_by_id = {npc.id: npc.name for npc in script.npcs}
    speakers_so_far: set[str] = set()
    locations_so_far: set[str] = set()
    referencing_events = 0
    for idx, event in enumerate(events):
        this_event_speakers = {line.npc_id for line in event.dialogue}
        prior_npc_names = {
            npc_names_by_id[npc_id]
            for npc_id in speakers_so_far - this_event_speakers
            if npc_id in npc_names_by_id and len(npc_names_by_id[npc_id]) >= 2
        }
        prior_locations = {
            loc for loc in locations_so_far if loc and loc != event.location and len(loc) >= 2
        }
        if idx >= 1 and (prior_npc_names or prior_locations):
            haystack = _normalize(
                event.summary
                + "".join(line.line for line in event.dialogue)
                + "".join(f"{b.choice_text}{b.condition}{b.effects}" for b in event.branches)
            )
            if any(_normalize(name) in haystack for name in prior_npc_names) or any(
                _normalize(loc) in haystack for loc in prior_locations
            ):
                referencing_events += 1
        speakers_so_far |= this_event_speakers
        if event.location:
            locations_so_far.add(event.location)
    prior_entity_reference_pct = referencing_events / (n_events - 1) if n_events > 1 else 0.0

    # connected_event_pct
    event_ids = {e.id for e in events}
    targeted_ids: set[str] = set()
    for e in events:
        for b in e.branches:
            if b.next_event_id != e.id and b.next_event_id in event_ids:
                targeted_ids.add(b.next_event_id)
    connected_event_pct = len(targeted_ids) / n_events if n_events else 0.0

    # self_loop_branch_pct
    all_branches = [b for e in events for b in e.branches]
    n_branches = len(all_branches)
    self_loop_branches = sum(1 for e in events for b in e.branches if b.next_event_id == e.id)
    self_loop_branch_pct = self_loop_branches / n_branches if n_branches else 0.0

    return {
        "distinct_event_title_pct": distinct_event_title_pct,
        "repeated_dialogue_pct": repeated_dialogue_pct,
        "prior_entity_reference_pct": prior_entity_reference_pct,
        "connected_event_pct": connected_event_pct,
        "self_loop_branch_pct": self_loop_branch_pct,
    }


def gmud_metrics(script: Script) -> dict[str, float | int]:
    """Nine structural proxies for whether a generated script actually
    carries the GMUD frame this change added (schema.py's Faction/
    StatThreshold/Region/TruthLayer/Chapter/SkillCheck/Ending, and
    Branch's cost/payoff/convergence fields) rather than leaving those
    fields at their empty defaults -- lets eval_generation.py A/B a
    variant's GMUD-shape coverage the same way the other metrics already
    A/B RPG-shape/continuity coverage. Same "ratios are 0.0 when their
    denominator is empty" convention as script_metrics()/continuity_metrics().

    - branches_with_cost_pct / branches_with_payoff_pct (higher is
      better): among branches with a structured effect (effect_ops),
      the fraction declaring a cost / declaring some payoff (immediate_
      feedback, or a deferred payoff_chapter_id/payoff_description).
    - checks_with_fallback_pct (higher is better): fraction of SkillChecks
      with a failure_branch_id or item_bypass_id -- see
      guardrails.check_check_fallback.
    - main_scene_ratio (target: somewhat above 0.5, per the guide's
      主要場景略多於調味場景): main_count / (main_count + flavor_count)
      among events whose scene_kind is set.
    - events_with_clue_pct (higher is better): fraction of events
      unlocking at least one clue.
    - chapters_with_convergence_pct (higher is better): fraction of
      chapters declaring a converge_event_id.
    - stat_threshold_coverage_pct (higher is better): fraction of stats
      targeted by a branch's structured effect that have at least one
      covering stat_threshold -- see schema.validate_stat_thresholds.
    - faction_count / ending_count: raw counts, not ratios.
    """
    events = script.events
    n_events = len(events)

    branches_with_effect = [
        b for e in events for b in e.branches if b.effect_ops
    ]
    n_branches_with_effect = len(branches_with_effect)
    branches_with_cost_pct = (
        sum(1 for b in branches_with_effect if b.cost) / n_branches_with_effect
        if n_branches_with_effect
        else 0.0
    )
    branches_with_payoff_pct = (
        sum(
            1
            for b in branches_with_effect
            if b.immediate_feedback or b.payoff_chapter_id or b.payoff_description
        )
        / n_branches_with_effect
        if n_branches_with_effect
        else 0.0
    )

    all_checks = [c for e in events for c in e.checks]
    n_checks = len(all_checks)
    checks_with_fallback_pct = (
        sum(1 for c in all_checks if c.failure_branch_id or c.item_bypass_id) / n_checks
        if n_checks
        else 0.0
    )

    main_count = sum(1 for e in events if e.scene_kind == "main")
    flavor_count = sum(1 for e in events if e.scene_kind == "flavor")
    main_scene_ratio = (
        main_count / (main_count + flavor_count) if (main_count + flavor_count) else 0.0
    )

    events_with_clue_pct = (
        sum(1 for e in events if e.clue_ids) / n_events if n_events else 0.0
    )

    n_chapters = len(script.chapters)
    chapters_with_convergence_pct = (
        sum(1 for c in script.chapters if c.converge_event_id) / n_chapters
        if n_chapters
        else 0.0
    )

    targeted_stats = {
        op.target_id
        for e in events
        for b in e.branches
        for op in b.effect_ops
        if op.target_kind == "stat"
    }
    covered_stats = {t.stat_id for t in script.stat_thresholds}
    stat_threshold_coverage_pct = (
        len(targeted_stats & covered_stats) / len(targeted_stats) if targeted_stats else 0.0
    )

    return {
        "branches_with_cost_pct": branches_with_cost_pct,
        "branches_with_payoff_pct": branches_with_payoff_pct,
        "checks_with_fallback_pct": checks_with_fallback_pct,
        "main_scene_ratio": main_scene_ratio,
        "events_with_clue_pct": events_with_clue_pct,
        "chapters_with_convergence_pct": chapters_with_convergence_pct,
        "stat_threshold_coverage_pct": stat_threshold_coverage_pct,
        "faction_count": len(script.factions),
        "ending_count": len(script.endings),
    }
