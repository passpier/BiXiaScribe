"""Structural quality proxies for a generated Script -- pure functions over
schema.py, no crew/agent involved, so they're unit-testable on hand-built
Scripts. Deliberately cheap and deterministic (counts/coverage/ratios), not
an LLM-as-judge prose score -- judging whether dialogue actually *sounds*
武俠 is left to a human reading scripts/eval_generation.py's saved output --
see that script's module docstring for why.

`continuity_metrics()` (BiXiaScribe 重構 Phase 5) extends this with five
more deterministic, structural proxies -- still not LLM-as-judge -- aimed
specifically at whether the scene writer had visibility into scenes it
causally depends on. They exist for the layered pipeline's
compressed-vs-untrimmed-context quality regression (see
docs/BiXiaScribe_REFACTORING_PLAN.md's Phase 5 section): the original nine
metrics are all driven by the beat sheet (event/NPC counts) or by per-event
choices that don't depend on prior scenes, so none of them can tell whether
context compression hurt continuity. The five below are computed only over
what a scene writer actually authors -- Event.title/.summary/.location,
dialogue lines, and branches -- and are annotated below with which
direction ("higher is better" / "lower is better") indicates less memory
loss across scenes.
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
