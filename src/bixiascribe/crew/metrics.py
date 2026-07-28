"""Structural quality proxies for a generated Script -- pure functions over
schema.py, no crew/agent involved, so they're unit-testable on hand-built
Scripts. Deliberately cheap and deterministic (counts/coverage/ratios), not
an LLM-as-judge prose score: judging whether dialogue actually *sounds*
武俠 is left to a human reading scripts/eval_generation.py's saved output --
see that script's module docstring for why.
"""
from __future__ import annotations

from ..schema import Script


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
    }
