"""Unit tests for crew/normalize.py's mechanical reference repair. No LLM/
network involved, mirroring tests/test_guardrails.py's style. Each case
covers one mechanical fix, plus a reverse assertion that a genuinely
semantic problem (not one of the three mechanical cases) is left alone for
schema.validate_references() + the existing repair loops to catch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.crew.normalize import normalize_script  # noqa: E402
from bixiascribe.schema import (  # noqa: E402
    Branch,
    Chapter,
    Clue,
    Event,
    Script,
    validate_references,
)


def test_next_event_id_backfilled_from_converges_to_event_id():
    script = Script(
        title="t",
        premise="p",
        chapters=[Chapter(id="ch1", title="c", summary="s", converge_event_id="ev2")],
        events=[
            Event(
                id="ev1", title="a", location="l", summary="s", chapter_id="ch1",
                branches=[
                    Branch(
                        id="b1", choice_text="x", next_event_id="",
                        converges_to_event_id="ev2",
                    )
                ],
            ),
            Event(id="ev2", title="b", location="l", summary="s", chapter_id="ch1"),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].branches[0].next_event_id == "ev2"
    assert notes


def test_next_event_id_backfilled_from_chapter_converge_point():
    script = Script(
        title="t",
        premise="p",
        chapters=[Chapter(id="ch1", title="c", summary="s", converge_event_id="ev2")],
        events=[
            Event(
                id="ev1", title="a", location="l", summary="s", chapter_id="ch1",
                branches=[Branch(id="b1", choice_text="x", next_event_id="")],
            ),
            Event(id="ev2", title="b", location="l", summary="s", chapter_id="ch1"),
        ],
    )
    out, _ = normalize_script(script)
    assert out.events[0].branches[0].next_event_id == "ev2"


def test_next_event_id_backfilled_from_sequence_as_last_resort():
    script = Script(
        title="t",
        premise="p",
        events=[
            Event(
                id="ev1", title="a", location="l", summary="s",
                branches=[Branch(id="b1", choice_text="x", next_event_id="")],
            ),
            Event(id="ev2", title="b", location="l", summary="s"),
        ],
    )
    out, _ = normalize_script(script)
    assert out.events[0].branches[0].next_event_id == "ev2"


def test_next_event_id_left_dangling_when_no_fallback_available():
    """Single-event script, no chapter, no converges_to_event_id -- nothing
    mechanical to backfill from. Must be left alone (not silently
    fabricated) so validate_references() still flags it for a real repair
    pass."""
    script = Script(
        title="t",
        premise="p",
        events=[
            Event(
                id="ev1", title="a", location="l", summary="s",
                branches=[Branch(id="b1", choice_text="x", next_event_id="")],
            ),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].branches[0].next_event_id == ""
    assert not notes
    assert validate_references(out)  # still flagged -- not silently accepted


def test_missing_chapters_backfilled_from_event_chapter_ids():
    script = Script(
        title="t",
        premise="p",
        events=[
            Event(id="ev1", title="a", location="l", summary="s", chapter_id="ch1"),
            Event(id="ev2", title="b", location="l", summary="s", chapter_id="ch1"),
        ],
    )
    out, notes = normalize_script(script)
    assert [c.id for c in out.chapters] == ["ch1"]
    assert notes


def test_dangling_clue_ids_are_cleared():
    script = Script(
        title="t",
        premise="p",
        clues=[Clue(id="c1", name="C")],
        events=[
            Event(
                id="ev1", title="a", location="l", summary="s",
                clue_ids=["c1", "c-unknown"],
            ),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].clue_ids == ["c1"]
    assert notes


def test_dangling_npc_id_in_dialogue_is_not_touched():
    """A dangling NPC reference in dialogue is a real content problem
    (invent a matching NPC, or drop the line) -- normalize_script has no
    mechanical fix for it and must leave it for validate_references() +
    the existing LLM repair loop."""
    script = Script(
        title="t",
        premise="p",
        events=[
            Event(
                id="ev1", title="a", location="l", summary="s",
                dialogue=[{"npc_id": "npc-ghost", "line": "x"}],
            ),
        ],
    )
    out, notes = normalize_script(script)
    assert out.events[0].dialogue[0].npc_id == "npc-ghost"
    assert not notes
    problems = validate_references(out)
    assert any("npc-ghost" in p for p in problems)


if __name__ == "__main__":
    test_next_event_id_backfilled_from_converges_to_event_id()
    test_next_event_id_backfilled_from_chapter_converge_point()
    test_next_event_id_backfilled_from_sequence_as_last_resort()
    test_next_event_id_left_dangling_when_no_fallback_available()
    test_missing_chapters_backfilled_from_event_chapter_ids()
    test_dangling_clue_ids_are_cleared()
    test_dangling_npc_id_in_dialogue_is_not_touched()
    print("All tests passed.")
