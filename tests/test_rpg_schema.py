"""Unit tests for the RPG-shape entities added to schema.py (player/items/
quests/effect_ops/NPC introductions) -- see CLAUDE.md's script generation
section. No external deps, no API key needed, mirrors
test_schema_layered.py's philosophy.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.schema import (  # noqa: E402
    NPC,
    Branch,
    DialogueLine,
    EffectOp,
    Event,
    ExtractionResult,
    Item,
    PlayerCharacter,
    Quest,
    Script,
    Variable,
    validate_npc_introductions,
    validate_references,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- Round-trips -------------------------------------------------------


def test_player_character_round_trips_with_defaults():
    player = PlayerCharacter()
    assert player.id == "player"
    assert player.stats == []
    dumped = player.model_dump_json()
    restored = PlayerCharacter.model_validate_json(dumped)
    assert restored == player


def test_item_and_quest_round_trip():
    item = Item(id="i1", name="鐵劍", description="一把鐵劍", acquired_in_event_id="ev1")
    quest = Quest(id="q1", name="尋劍", objective="找到鐵劍", event_ids=["ev1", "ev2"])
    assert Item.model_validate_json(item.model_dump_json()) == item
    assert Quest.model_validate_json(quest.model_dump_json()) == quest


def test_variable_kind_defaults_to_flag():
    assert Variable(id="v1", name="v", initial=False).kind == "flag"
    assert Variable(id="v2", name="內力", initial=100, kind="stat").kind == "stat"


def test_extraction_result_defaults_include_new_rpg_fields():
    extraction = ExtractionResult()
    assert extraction.player is None
    assert extraction.items == []
    assert extraction.quests == []
    assert extraction.props == []  # deprecated but still present for old data


# --- Old out/eval/*.json scripts still parse (schema backward compat) --


def test_existing_eval_scripts_still_parse_if_present():
    """out/ is gitignored (see CLAUDE.md's Gotchas) so this is a no-op when
    absent -- but if scripts from before this schema change are present
    locally, they must still validate as Script (every new field has a
    default)."""
    eval_dir = PROJECT_ROOT / "out" / "eval"
    if not eval_dir.is_dir():
        return
    paths = list(eval_dir.glob("*.json"))
    if not paths:
        return
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        script = Script.model_validate(data)
        assert script.player is None or isinstance(script.player, PlayerCharacter)
        assert isinstance(script.items, list)
        assert isinstance(script.quests, list)


# --- validate_references(): new cross-reference checks ------------------


def _base_script(**overrides) -> Script:
    defaults = dict(
        title="t",
        premise="p",
        npcs=[NPC(id="npc1", name="甲", identity="俠客", personality="剛", speech_style="直")],
        events=[
            Event(
                id="ev1",
                title="t1",
                location="l1",
                summary="s1",
                dialogue=[DialogueLine(npc_id="npc1", line="line1")],
                branches=[Branch(id="b1", choice_text="go", next_event_id="ev1")],
            )
        ],
    )
    defaults.update(overrides)
    return Script(**defaults)


def test_validate_references_accepts_player_id_as_dialogue_target():
    script = _base_script(
        player=PlayerCharacter(id="player", name="你"),
        events=[
            Event(
                id="ev1",
                title="t1",
                location="l1",
                summary="s1",
                dialogue=[DialogueLine(npc_id="player", line="我來了")],
            )
        ],
    )
    assert validate_references(script) == []


def test_validate_references_flags_unknown_quest_id_on_event():
    script = _base_script()
    script.events[0].quest_id = "no-such-quest"
    problems = validate_references(script)
    assert any("quest_id" in p for p in problems)


def test_validate_references_flags_unreachable_item():
    script = _base_script(items=[Item(id="itm1", name="劍", acquired_in_event_id="no-such-event")])
    problems = validate_references(script)
    assert any("acquired_in_event_id" in p for p in problems)


def test_validate_references_flags_dangling_quest_event_ids():
    script = _base_script(quests=[Quest(id="q1", name="Q", event_ids=["no-such-event"])])
    problems = validate_references(script)
    assert any("event_ids references unknown event" in p for p in problems)


def test_validate_references_flags_dangling_effect_op_target():
    script = _base_script()
    script.events[0].branches[0].effect_ops = [
        EffectOp(target_kind="item", target_id="no-such-item", op="give")
    ]
    problems = validate_references(script)
    assert any("effect_op" in p for p in problems)


def test_validate_references_accepts_valid_effect_op():
    script = _base_script(
        items=[Item(id="itm1", name="劍")],
        events=[
            Event(
                id="ev1",
                title="t1",
                location="l1",
                summary="s1",
                dialogue=[DialogueLine(npc_id="npc1", line="line1")],
                branches=[
                    Branch(
                        id="b1",
                        choice_text="go",
                        next_event_id="ev1",
                        effect_ops=[EffectOp(target_kind="item", target_id="itm1", op="give")],
                    )
                ],
            )
        ],
    )
    assert validate_references(script) == []


def test_validate_references_flags_unknown_player_starting_item():
    script = _base_script(player=PlayerCharacter(starting_items=["no-such-item"]))
    problems = validate_references(script)
    assert any("starting_items" in p for p in problems)


def test_validate_references_flags_unknown_npc_first_appearance_event():
    script = _base_script()
    script.npcs[0].first_appearance_event_id = "no-such-event"
    problems = validate_references(script)
    assert any("first_appearance_event_id" in p for p in problems)


def test_validate_references_flags_unknown_quest_giver_npc():
    script = _base_script(quests=[Quest(id="q1", name="Q", giver_npc_id="no-such-npc")])
    problems = validate_references(script)
    assert any("giver_npc_id" in p for p in problems)


# --- validate_npc_introductions() ---------------------------------------


def test_validate_npc_introductions_clean_when_intro_is_at_or_before_first_line():
    npc = NPC(
        id="npc1", name="甲", identity="俠客", personality="剛", speech_style="直",
        first_appearance_event_id="ev1",
    )
    script = Script(
        title="t", premise="p", npcs=[npc],
        events=[
            Event(id="ev1", title="t1", location="l", summary="s",
                  dialogue=[DialogueLine(npc_id="npc1", line="hi")]),
        ],
    )
    assert validate_npc_introductions(script) == []


def test_validate_npc_introductions_flags_npc_speaking_before_intro():
    npc = NPC(
        id="npc1", name="甲", identity="俠客", personality="剛", speech_style="直",
        first_appearance_event_id="ev2",
    )
    script = Script(
        title="t", premise="p", npcs=[npc],
        events=[
            Event(id="ev1", title="t1", location="l", summary="s",
                  dialogue=[DialogueLine(npc_id="npc1", line="hi")]),
            Event(id="ev2", title="t2", location="l", summary="s"),
        ],
    )
    problems = validate_npc_introductions(script)
    assert any("npc1" in p for p in problems)


def test_validate_npc_introductions_flags_missing_introduction_entirely():
    npc = NPC(id="npc1", name="甲", identity="俠客", personality="剛", speech_style="直")
    script = Script(
        title="t", premise="p", npcs=[npc],
        events=[
            Event(id="ev1", title="t1", location="l", summary="s",
                  dialogue=[DialogueLine(npc_id="npc1", line="hi")]),
        ],
    )
    problems = validate_npc_introductions(script)
    assert any("first_appearance_event_id/introduction" in p for p in problems)


def test_validate_npc_introductions_silent_when_never_speaks():
    npc = NPC(id="npc1", name="甲", identity="俠客", personality="剛", speech_style="直")
    script = Script(title="t", premise="p", npcs=[npc], events=[])
    assert validate_npc_introductions(script) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
