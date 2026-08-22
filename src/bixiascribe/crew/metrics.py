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

`gmud_metrics()` adds structural proxies for the GMUD-ish frame this
repository's schema.py carries (factions/truth/chapters/checks/endings, and
the cost/payoff shape on choices) -- whether a generated script actually
carries that structure rather than leaving it at its (additive,
backward-compatible) empty default. See gmud_metrics()'s own docstring for
each metric.

Phase 4 (2026-08-22): main_scene_ratio/chapters_with_convergence_pct/
stat_threshold_coverage_pct were dropped -- their backing fields
(Event.scene_kind, Chapter.converge_event_id, Script.stat_thresholds) no
longer exist.
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
    n_branches = sum(len(e.choices) for e in events)
    n_dialogue_lines = sum(len(e.dialogue) for e in events)

    events_with_dialogue = sum(1 for e in events if e.dialogue)
    speaking_npc_ids = {line.npc for e in events for line in e.dialogue}

    all_lines = [line.line for e in events for line in e.dialogue]
    avg_line_chars = sum(len(line) for line in all_lines) / len(all_lines) if all_lines else 0.0

    return {
        "events": n_events,
        "npcs": n_npcs,
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
      of events at index >= 1, the fraction whose summary/dialogue/choice
      text names an NPC who spoke in some earlier event but not in this
      one. Excluding NPCs present in the scoring event is what makes this
      a *callback* measure, not a restatement of the character cards
      (which are never trimmed by build_session_document() and so can't
      discriminate here).
    - connected_event_pct (higher is better): fraction of events targeted
      by some *other* event's choice. validate_references() already
      guarantees choice targets aren't dangling, so this measures
      connectivity density, not validity -- orphan events are what a
      writer produces when it can't see the surrounding graph.
    - self_loop_branch_pct (lower is better): fraction of choices whose
      next points back at their own event -- a degenerate fallback a
      memoryless writer reaches for when it has no other valid id to name,
      and one validate_references() happily accepts.
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
    referencing_events = 0
    for idx, event in enumerate(events):
        this_event_speakers = {line.npc for line in event.dialogue}
        prior_npc_names = {
            npc_names_by_id[npc_id]
            for npc_id in speakers_so_far - this_event_speakers
            if npc_id in npc_names_by_id and len(npc_names_by_id[npc_id]) >= 2
        }
        if idx >= 1 and prior_npc_names:
            haystack = _normalize(
                event.summary
                + "".join(line.line for line in event.dialogue)
                + "".join(f"{c.text}{c.effects}" for c in event.choices)
            )
            if any(_normalize(name) in haystack for name in prior_npc_names):
                referencing_events += 1
        speakers_so_far |= this_event_speakers
    prior_entity_reference_pct = referencing_events / (n_events - 1) if n_events > 1 else 0.0

    # connected_event_pct
    event_ids = {e.id for e in events}
    targeted_ids: set[str] = set()
    for e in events:
        for c in e.choices:
            if c.next != e.id and c.next in event_ids:
                targeted_ids.add(c.next)
    connected_event_pct = len(targeted_ids) / n_events if n_events else 0.0

    # self_loop_branch_pct
    all_choices = [c for e in events for c in e.choices]
    n_branches = len(all_choices)
    self_loop_branches = sum(1 for e in events for c in e.choices if c.next == e.id)
    self_loop_branch_pct = self_loop_branches / n_branches if n_branches else 0.0

    return {
        "distinct_event_title_pct": distinct_event_title_pct,
        "repeated_dialogue_pct": repeated_dialogue_pct,
        "prior_entity_reference_pct": prior_entity_reference_pct,
        "connected_event_pct": connected_event_pct,
        "self_loop_branch_pct": self_loop_branch_pct,
    }


def gmud_metrics(script: Script) -> dict[str, float | int]:
    """Structural proxies for whether a generated script actually carries
    the GMUD-ish frame schema.py has (Faction/Truth/Chapter/Check/Ending,
    and the cost/payoff shape on choices) rather than leaving those fields
    at their empty defaults -- lets eval_generation.py A/B a variant's
    GMUD-shape coverage the same way the other metrics already A/B
    RPG-shape/continuity coverage. Same "ratios are 0.0 when their
    denominator is empty" convention as script_metrics()/continuity_metrics().

    - branches_with_cost_pct / branches_with_payoff_pct (higher is
      better): among choices with a non-empty effects description,
      the fraction declaring a cost / declaring a payoff_at.
    - checks_with_fallback_pct (higher is better): fraction of events with
      a check that declare on_fail -- see guardrails.check_check_fallback.
    - events_with_clue_pct (higher is better): fraction of events
      unlocking at least one clue.
    - faction_count / ending_count: raw counts, not ratios.
    """
    events = script.events
    n_events = len(events)

    choices_with_effect = [c for e in events for c in e.choices if c.effects]
    n_choices_with_effect = len(choices_with_effect)
    branches_with_cost_pct = (
        sum(1 for c in choices_with_effect if c.cost) / n_choices_with_effect
        if n_choices_with_effect
        else 0.0
    )
    branches_with_payoff_pct = (
        sum(1 for c in choices_with_effect if c.payoff_at) / n_choices_with_effect
        if n_choices_with_effect
        else 0.0
    )

    events_with_check = [e for e in events if e.check is not None]
    n_events_with_check = len(events_with_check)
    checks_with_fallback_pct = (
        sum(1 for e in events_with_check if e.check.on_fail) / n_events_with_check
        if n_events_with_check
        else 0.0
    )

    events_with_clue_pct = (
        sum(1 for e in events if e.clue_ids) / n_events if n_events else 0.0
    )

    return {
        "branches_with_cost_pct": branches_with_cost_pct,
        "branches_with_payoff_pct": branches_with_payoff_pct,
        "checks_with_fallback_pct": checks_with_fallback_pct,
        "events_with_clue_pct": events_with_clue_pct,
        "faction_count": len(script.factions),
        "ending_count": len(script.endings),
    }
