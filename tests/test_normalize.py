"""Unit tests for crew/normalize.py's mechanical reference repair. No LLM/
network involved, mirroring tests/test_guardrails.py's style. Each case
covers one mechanical fix, plus a reverse assertion that a genuinely
semantic problem (not one of the mechanical cases) is left alone for
schema.validate_references() + the existing repair loops to catch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.crew.normalize import (  # noqa: E402
    normalize_scene_npc_ids,
    normalize_script,
)
from bixiascribe.schema import (  # noqa: E402
    Chapter,
    Choice,
    Clue,
    DialogueLine,
    Event,
    Meta,
    Script,
    validate_references,
)


def test_next_backfilled_from_sequence():
    """The only remaining fallback tier (Phase 4 dropped the
    converges_to_event_id/chapter.converge_event_id tiers, see
    normalize.py's module docstring): "the next event in sequence"."""
    script = Script(
        meta=Meta(title="t"),
        chapters=[Chapter(id="ch1", title="c", summary="s")],
        events=[
            Event(
                id="ev1", title="a", summary="s", chapter_id="ch1",
                choices=[Choice(id="b1", text="x", next="")],
            ),
            Event(id="ev2", title="b", summary="s", chapter_id="ch1"),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].choices[0].next == "ev2"
    assert notes


def test_next_left_dangling_when_no_fallback_available():
    """Single-event script, choice.next names a genuinely unknown event --
    nothing mechanical to backfill from (there's no "next event in
    sequence" either). Must be left alone (not silently fabricated) so
    validate_references() still flags it for a real repair pass."""
    script = Script(
        meta=Meta(title="t"),
        events=[
            Event(
                id="ev1", title="a", summary="s",
                choices=[Choice(id="b1", text="x", next="no-such-event")],
            ),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].choices[0].next == "no-such-event"
    assert not notes
    assert validate_references(out)  # still flagged -- not silently accepted


def test_missing_chapters_backfilled_from_event_chapter_ids():
    script = Script(
        meta=Meta(title="t"),
        events=[
            Event(id="ev1", title="a", summary="s", chapter_id="ch1"),
            Event(id="ev2", title="b", summary="s", chapter_id="ch1"),
        ],
    )
    out, notes = normalize_script(script)
    assert [c.id for c in out.chapters] == ["ch1"]
    assert notes


def test_dangling_clue_ids_are_cleared():
    script = Script(
        meta=Meta(title="t"),
        clues=[Clue(id="c1", name="C")],
        events=[
            Event(
                id="ev1", title="a", summary="s",
                clue_ids=["c1", "c-unknown"],
            ),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].clue_ids == ["c1"]
    assert notes


def test_dangling_npc_in_dialogue_is_not_touched():
    """A dangling NPC reference in dialogue is a real content problem
    (invent a matching NPC, or drop the line) -- normalize_script has no
    mechanical fix for it and must leave it for validate_references() +
    the existing LLM repair loop."""
    script = Script(
        meta=Meta(title="t"),
        events=[
            Event(
                id="ev1", title="a", summary="s",
                dialogue=[DialogueLine(npc="npc-ghost", line="x")],
            ),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].dialogue[0].npc == "npc-ghost"
    assert not notes
    problems = validate_references(out)
    assert any("npc-ghost" in p for p in problems)


def _scene(npc: str) -> Event:
    return Event(
        id="ev1", title="a", summary="s",
        dialogue=[DialogueLine(npc=npc, line="x")],
    )


def test_scene_npc_name_rewritten_to_id():
    """The exact production case (out/generation_runs_ui.jsonl,
    run 1787381935-req-ca28a2312e): scene_writer filled dialogue[].npc with
    the NPC's display name '陳掌柜' instead of its id 'npc_innkeeper'."""
    event, notes = normalize_scene_npc_ids(
        _scene("陳掌柜"),
        known_npc_ids={"npc_innkeeper"},
        name_to_id={"陳掌柜": "npc_innkeeper"},
    )
    assert event.dialogue[0].npc == "npc_innkeeper"
    assert len(notes) == 1


def test_scene_npc_already_an_id_is_untouched():
    original = _scene("npc_innkeeper")
    event, notes = normalize_scene_npc_ids(
        original,
        known_npc_ids={"npc_innkeeper"},
        name_to_id={"陳掌柜": "npc_innkeeper"},
    )
    assert event is original
    assert notes == []


def test_scene_npc_matched_after_punctuation_strip():
    event, notes = normalize_scene_npc_ids(
        _scene("陳・掌柜"),
        known_npc_ids={"npc_innkeeper"},
        name_to_id={"陳掌柜": "npc_innkeeper"},
    )
    assert event.dialogue[0].npc == "npc_innkeeper"
    assert notes


def test_scene_npc_fuzzy_match_above_threshold():
    """A long-enough name with a small edit distance clears the 0.8 ratio
    threshold. Short (<=3 char) CJK names are NOT reliably caught by this
    tier -- see test_scene_npc_short_name_variant_not_fuzzy_matched below,
    which is exactly why the exact/stripped tiers (not fuzzy matching) are
    the primary defense for the real production case."""
    event, notes = normalize_scene_npc_ids(
        _scene("城西鐵匠鋪王大鎚"),
        known_npc_ids={"npc_smith"},
        name_to_id={"城西鐵匠鋪王大錘": "npc_smith"},
    )
    assert event.dialogue[0].npc == "npc_smith"
    assert notes


def test_scene_npc_short_name_variant_not_fuzzy_matched():
    """A 1-of-3-character difference in a short CJK name (陳掌櫃 vs 陳掌柜)
    scores well under the 0.8 ratio threshold -- documents the known limit
    of the fuzzy tier rather than asserting a false guarantee."""
    event, notes = normalize_scene_npc_ids(
        _scene("陳掌櫃"),
        known_npc_ids={"npc_innkeeper"},
        name_to_id={"陳掌柜": "npc_innkeeper"},
    )
    assert event.dialogue[0].npc == "陳掌櫃"
    assert notes == []


def test_scene_npc_ambiguous_tie_is_a_noop():
    event, notes = normalize_scene_npc_ids(
        _scene("陳掌柜"),
        known_npc_ids=set(),
        name_to_id={"陳掌柜甲": "npc_a", "陳掌柜乙": "npc_b"},
    )
    assert event.dialogue[0].npc == "陳掌柜"
    assert notes == []


def test_scene_npc_unknown_name_left_alone():
    event, notes = normalize_scene_npc_ids(
        _scene("路人甲"),
        known_npc_ids={"npc_innkeeper"},
        name_to_id={"陳掌柜": "npc_innkeeper"},
    )
    assert event.dialogue[0].npc == "路人甲"
    assert notes == []


if __name__ == "__main__":
    test_next_backfilled_from_sequence()
    test_next_left_dangling_when_no_fallback_available()
    test_missing_chapters_backfilled_from_event_chapter_ids()
    test_dangling_clue_ids_are_cleared()
    test_dangling_npc_in_dialogue_is_not_touched()
    print("All tests passed.")
