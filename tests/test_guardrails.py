"""Unit tests for crew/guardrails.py's pure RPG-shape checks. No LLM/network
involved -- these functions never call crewai -- mirroring
tests/test_causal_consistency.py's split between pure-function tests and
integration tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.crew.guardrails import (  # noqa: E402
    as_feedback,
    check_extraction_rpg,
    check_scene_rpg,
    check_script_rpg,
)
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
)


def _npc(id_="npc1", name="甲") -> NPC:
    return NPC(
        id=id_, name=name, identity="俠客", personality="剛烈", speech_style="直來直去",
        first_appearance_event_id="ev1",
    )


def _full_script() -> Script:
    return Script(
        title="t",
        premise="p",
        npcs=[_npc()],
        player=PlayerCharacter(
            id="player", name="你",
            stats=[Variable(id="hp", name="內力", initial=100, kind="stat"),
                   Variable(id="rep", name="聲望", initial=0, kind="stat")],
        ),
        items=[Item(id="itm1", name="劍", acquired_in_event_id="ev1")],
        quests=[Quest(id="q1", name="Q", objective="找劍", event_ids=["ev1"])],
        events=[
            Event(
                id="ev1", title="t1", location="l", summary="s",
                dialogue=[DialogueLine(npc_id="npc1", line="hi")],
                branches=[
                    Branch(
                        id="b1", choice_text="go", next_event_id="ev1",
                        effect_ops=[EffectOp(target_kind="item", target_id="itm1", op="give")],
                    )
                ],
            )
        ],
    )


# --- check_script_rpg ----------------------------------------------------


def test_check_script_rpg_passes_a_fully_populated_script():
    assert check_script_rpg(_full_script()) == []


def test_check_script_rpg_flags_missing_player():
    script = _full_script()
    script.player = None
    problems = check_script_rpg(script)
    assert any("player" in p for p in problems)


def test_check_script_rpg_flags_too_few_stats():
    script = _full_script()
    script.player.stats = [Variable(id="hp", name="內力", initial=100, kind="stat")]
    problems = check_script_rpg(script)
    assert any("stats" in p for p in problems)


def test_check_script_rpg_flags_empty_items():
    script = _full_script()
    script.items = []
    problems = check_script_rpg(script)
    assert any("items" in p for p in problems)


def test_check_script_rpg_flags_unreachable_item():
    script = _full_script()
    # acquired_in_event_id names an event that doesn't exist -- not the
    # same as "" (held from the start, always reachable).
    script.items = [Item(id="itm1", name="劍", acquired_in_event_id="no-such-event")]
    script.events[0].branches[0].effect_ops = []  # no effect_op grants it either
    problems = check_script_rpg(script)
    assert any("道具" in p for p in problems)


def test_check_script_rpg_accepts_item_held_from_start():
    script = _full_script()
    script.items = [Item(id="itm1", name="劍")]  # acquired_in_event_id="" = held from start
    script.events[0].branches[0].effect_ops = []
    problems = check_script_rpg(script)
    assert not any("道具" in p for p in problems)


def test_check_script_rpg_flags_empty_quests():
    script = _full_script()
    script.quests = []
    problems = check_script_rpg(script)
    assert any("quests" in p for p in problems)


def test_check_script_rpg_flags_quest_with_no_matching_event():
    script = _full_script()
    script.quests = [Quest(id="q1", name="Q", objective="x", event_ids=["no-such-event"])]
    problems = check_script_rpg(script)
    assert any("q1" in p for p in problems)


def test_check_script_rpg_flags_fake_player_npc():
    script = _full_script()
    script.npcs.append(_npc(id_="npc_player", name="玩家"))
    problems = check_script_rpg(script)
    assert any("npc_player" in p for p in problems)


def test_check_script_rpg_flags_fake_narrator_npc():
    script = _full_script()
    script.npcs.append(_npc(id_="npc_narrator", name="旁白"))
    problems = check_script_rpg(script)
    assert any("npc_narrator" in p for p in problems)


def test_check_script_rpg_flags_missing_first_appearance():
    script = _full_script()
    script.npcs[0].first_appearance_event_id = ""
    problems = check_script_rpg(script)
    assert any("first_appearance_event_id" in p for p in problems)


def test_check_script_rpg_includes_validate_references_problems():
    script = _full_script()
    script.events[0].dialogue.append(DialogueLine(npc_id="no-such-npc", line="?"))
    problems = check_script_rpg(script)
    assert any("no-such-npc" in p for p in problems)


# --- check_extraction_rpg -------------------------------------------------


def _full_extraction() -> ExtractionResult:
    return ExtractionResult(
        npcs=[_npc()],
        player=PlayerCharacter(
            stats=[Variable(id="hp", name="內力", initial=100, kind="stat"),
                   Variable(id="rep", name="聲望", initial=0, kind="stat")],
        ),
        items=[Item(id="itm1", name="劍")],
        quests=[Quest(id="q1", name="Q", objective="找劍")],
    )


def test_check_extraction_rpg_passes_a_fully_populated_extraction():
    assert check_extraction_rpg(_full_extraction()) == []


def test_check_extraction_rpg_flags_missing_player():
    extraction = _full_extraction()
    extraction.player = None
    assert any("player" in p for p in check_extraction_rpg(extraction))


def test_check_extraction_rpg_flags_empty_items_and_quests():
    extraction = _full_extraction()
    extraction.items = []
    extraction.quests = []
    problems = check_extraction_rpg(extraction)
    assert any("items" in p for p in problems)
    assert any("quests" in p for p in problems)


def test_check_extraction_rpg_flags_fake_role_npc():
    extraction = _full_extraction()
    extraction.npcs.append(_npc(id_="npc_narrator", name="旁白"))
    assert any("npc_narrator" in p for p in check_extraction_rpg(extraction))


# --- check_scene_rpg -------------------------------------------------------


def test_check_scene_rpg_passes_dialogue_from_known_npc():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        dialogue=[DialogueLine(npc_id="npc1", line="hi")],
    )
    assert check_scene_rpg(event, known_npc_ids={"npc1"}, introduced_npc_ids=set()) == []


def test_check_scene_rpg_passes_dialogue_from_already_introduced_npc():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        dialogue=[DialogueLine(npc_id="npc2", line="hi")],
    )
    assert check_scene_rpg(event, known_npc_ids=set(), introduced_npc_ids={"npc2"}) == []


def test_check_scene_rpg_flags_dialogue_from_unknown_uintroduced_npc():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        dialogue=[DialogueLine(npc_id="ghost", line="hi")],
    )
    problems = check_scene_rpg(event, known_npc_ids=set(), introduced_npc_ids=set())
    assert any("ghost" in p for p in problems)


def test_check_scene_rpg_flags_empty_dialogue():
    event = Event(id="ev1", title="t", location="l", summary="s")
    problems = check_scene_rpg(event, known_npc_ids=set(), introduced_npc_ids=set())
    assert any("沒有任何台詞" in p for p in problems)


# --- as_feedback -----------------------------------------------------------


def test_as_feedback_renders_bullet_list():
    feedback = as_feedback(["問題一", "問題二"])
    assert "問題一" in feedback
    assert "問題二" in feedback
    assert feedback.count("- ") == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
