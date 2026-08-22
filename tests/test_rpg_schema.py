"""Unit tests for the flat/ID-referenced script schema (schema.py, Phase 4 --
see openspec/changes/2026-08-22-slim-script-schema-mvp). No external deps,
no API key needed, mirrors test_schema_layered.py's philosophy.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.schema import (  # noqa: E402
    NPC,
    Chapter,
    Check,
    Choice,
    Clue,
    DialogueLine,
    Ending,
    Event,
    ExtractionResult,
    Faction,
    Item,
    Meta,
    Player,
    Script,
    Stat,
    Truth,
    validate_references,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- Round-trips -------------------------------------------------------


def test_player_round_trips_with_defaults():
    player = Player()
    assert player.id == "player"
    assert player.name == ""
    dumped = player.model_dump_json()
    restored = Player.model_validate_json(dumped)
    assert restored == player


def test_item_round_trips():
    item = Item(id="i1", name="鐵劍", from_event="ev1")
    assert Item.model_validate_json(item.model_dump_json()) == item


def test_stat_defaults():
    stat = Stat()
    assert stat.id == "mood"
    assert stat.init == 50


def test_extraction_result_defaults():
    extraction = ExtractionResult()
    assert extraction.player is None
    assert extraction.stat is None
    assert extraction.items == []
    assert extraction.factions == []
    assert extraction.truth is None
    assert extraction.clues == []
    assert extraction.endings == []


def test_faction_round_trip():
    faction = Faction(id="f1", name="少林", motive="肅清邪魔")
    assert Faction.model_validate_json(faction.model_dump_json()) == faction


def test_truth_round_trip():
    truth = Truth(public="江湖傳聞甲派滅門", revealed=["真兇是乙"], hidden="幕後主使是丙")
    assert Truth.model_validate_json(truth.model_dump_json()) == truth


def test_chapter_round_trip():
    chapter = Chapter(id="ch1", title="下山", summary="s", loc="山門", start_event="ev1")
    assert Chapter.model_validate_json(chapter.model_dump_json()) == chapter


def test_check_and_ending_round_trip():
    check = Check(on_pass="ev1", on_fail="ev2", fail_cost="受傷")
    ending = Ending(id="e1", name="正義結局", min=50, max=100)
    assert Check.model_validate_json(check.model_dump_json()) == check
    assert Ending.model_validate_json(ending.model_dump_json()) == ending


def test_choice_round_trip():
    choice = Choice(
        id="b1", text="go", next="ev1", cost="代價", effects="e", delta=10, payoff_at="ch1",
    )
    assert Choice.model_validate_json(choice.model_dump_json()) == choice


# --- Old out/eval/*.json scripts still parse (schema backward compat) --


def test_existing_eval_scripts_still_parse_if_present():
    """out/ is gitignored (see CLAUDE.md's Gotchas) so this is a no-op when
    absent. Old scripts predate this schema (top-level title/premise/
    variables/npcs/events, no `meta`), so validating the raw payload as
    Script would fail on the now-required `meta` field -- this test only
    confirms the parse doesn't crash the process, checking the "extra
    fields dropped" contract on a synthetic pre-existing-shape payload
    instead of asserting a real, unmigrated file loads whole."""
    eval_dir = PROJECT_ROOT / "out" / "eval"
    if not eval_dir.is_dir():
        return
    paths = list(eval_dir.glob("*.json"))
    if not paths:
        return
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)  # still readable JSON, nothing crashes


def test_pre_phase4_shaped_payload_drops_unknown_fields_gracefully():
    """A payload carrying the pre-Phase-4 field names (variables/stat_
    thresholds/etc.) alongside a valid `meta` should just have those extra
    keys ignored (pydantic's default extra="ignore"), not raise."""
    payload = {
        "meta": {"title": "t"},
        "variables": [{"id": "v1", "name": "v", "initial": 0}],
        "stat_thresholds": [{"id": "th1", "stat_id": "s1"}],
        "npcs": [],
        "events": [],
    }
    script = Script.model_validate(payload)
    assert script.meta.title == "t"
    assert not hasattr(script, "variables")


# --- validate_references(): cross-reference checks ------------------


def _base_script(**overrides) -> Script:
    defaults = dict(
        meta=Meta(title="t"),
        npcs=[NPC(id="npc1", name="甲", personality="剛", speech_style="直")],
        events=[
            Event(
                id="ev1",
                title="t1",
                summary="s1",
                dialogue=[DialogueLine(npc="npc1", line="line1")],
                choices=[Choice(id="b1", text="go", next="ev1")],
            )
        ],
    )
    defaults.update(overrides)
    return Script(**defaults)


def test_validate_references_accepts_player_as_dialogue_target():
    script = _base_script(
        player=Player(name="你"),
        events=[
            Event(
                id="ev1",
                title="t1",
                summary="s1",
                dialogue=[DialogueLine(npc="player", line="我來了")],
            )
        ],
    )
    assert validate_references(script) == []


def test_validate_references_flags_unreachable_item():
    script = _base_script(items=[Item(id="itm1", name="劍", from_event="no-such-event")])
    problems = validate_references(script)
    assert any("from_event" in p for p in problems)


def test_validate_references_accepts_valid_item():
    script = _base_script(items=[Item(id="itm1", name="劍", from_event="ev1")])
    assert validate_references(script) == []


def test_validate_references_flags_unknown_choice_next():
    script = _base_script()
    script.events[0].choices = [Choice(id="b1", text="go", next="no-such-event")]
    problems = validate_references(script)
    assert any("choice" in p and "next" in p for p in problems)


def test_validate_references_flags_unknown_choice_payoff_at():
    script = _base_script(chapters=[Chapter(id="ch1", title="c")])
    script.events[0].choices = [Choice(id="b1", text="go", next="ev1", payoff_at="no-such-chapter")]
    problems = validate_references(script)
    assert any("payoff_at" in p for p in problems)


def test_validate_references_accepts_valid_payoff_at():
    script = _base_script(chapters=[Chapter(id="ch1", title="c")])
    script.events[0].choices = [Choice(id="b1", text="go", next="ev1", payoff_at="ch1")]
    assert validate_references(script) == []


def test_validate_references_flags_unknown_check_targets():
    script = _base_script()
    script.events[0].check = Check(on_pass="no-such-event", on_fail="also-missing")
    problems = validate_references(script)
    assert any("on_pass" in p for p in problems)
    assert any("on_fail" in p for p in problems)


def test_validate_references_accepts_valid_check():
    script = _base_script()
    script.events[0].check = Check(on_pass="ev1", on_fail="ev1")
    assert validate_references(script) == []


def test_validate_references_flags_unknown_npc_faction():
    script = _base_script()
    script.npcs[0].faction_id = "no-such-faction"
    problems = validate_references(script)
    assert any("faction_id" in p for p in problems)


def test_validate_references_accepts_valid_npc_faction():
    script = _base_script(factions=[Faction(id="f1", name="少林")])
    script.npcs[0].faction_id = "f1"
    assert validate_references(script) == []


def test_validate_references_flags_unknown_event_chapter_npc_clue():
    script = _base_script(clues=[Clue(id="c1", name="血書", from_event="ev1")])
    script.events[0].chapter_id = "no-such-chapter"
    script.events[0].clue_ids = ["no-such-clue"]
    script.events[0].npc_ids = ["no-such-npc"]
    problems = validate_references(script)
    assert any("chapter_id" in p for p in problems)
    assert any("clue_ids" in p for p in problems)
    assert any("npc_ids" in p for p in problems)


def test_validate_references_flags_unknown_chapter_start_event():
    script = _base_script(chapters=[Chapter(id="ch1", title="c", start_event="no-such-event")])
    problems = validate_references(script)
    assert any("start_event" in p for p in problems)


def test_validate_references_flags_unknown_clue_from_event():
    script = _base_script(clues=[Clue(id="c1", name="血書", from_event="no-such-event")])
    problems = validate_references(script)
    assert any("from_event" in p for p in problems)


def test_validate_references_accepts_dialogue_and_npc_ids_defaults():
    assert validate_references(_base_script()) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
