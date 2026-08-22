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
    check_check_fallback,
    check_choice_quality,
    check_ending_ranges,
    check_extraction_rpg,
    check_scene_information,
    check_scene_rpg,
    check_script_rpg,
    check_truth_pacing,
)
from bixiascribe.schema import (  # noqa: E402
    NPC,
    Chapter,
    Check,
    Choice,
    DialogueLine,
    Ending,
    Event,
    ExtractionResult,
    Item,
    Meta,
    Player,
    Script,
    Stat,
    Truth,
)


def _npc(id_="npc1", name="甲") -> NPC:
    return NPC(id=id_, name=name, personality="剛烈", speech_style="直來直去")


def _full_script() -> Script:
    return Script(
        meta=Meta(title="t"),
        npcs=[_npc()],
        player=Player(name="你"),
        stat=Stat(id="rep", name="聲望"),
        items=[Item(id="itm1", name="劍", from_event="ev1")],
        events=[
            Event(
                id="ev1", title="t1", summary="s",
                dialogue=[{"npc": "npc1", "line": "hi"}],
                choices=[Choice(id="b1", text="go", next="ev1", effects="拿劍")],
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


def test_check_script_rpg_flags_missing_stat():
    script = _full_script()
    script.stat = None
    problems = check_script_rpg(script)
    assert any("stat" in p for p in problems)


def test_check_script_rpg_flags_empty_items():
    script = _full_script()
    script.items = []
    problems = check_script_rpg(script)
    assert any("items" in p for p in problems)


def test_check_script_rpg_flags_unreachable_item():
    script = _full_script()
    # from_event names an event that doesn't exist -- not the same as ""
    # (held from the start, always reachable).
    script.items = [Item(id="itm1", name="劍", from_event="no-such-event")]
    problems = check_script_rpg(script)
    assert any("道具" in p for p in problems)


def test_check_script_rpg_accepts_item_held_from_start():
    script = _full_script()
    script.items = [Item(id="itm1", name="劍")]  # from_event="" = held from start
    problems = check_script_rpg(script)
    assert not any("道具" in p for p in problems)


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


def test_check_script_rpg_includes_validate_references_problems():
    script = _full_script()
    script.events[0].dialogue.append(DialogueLine(npc="no-such-npc", line="?"))
    problems = check_script_rpg(script)
    assert any("no-such-npc" in p for p in problems)


# --- check_extraction_rpg -------------------------------------------------


def _full_extraction() -> ExtractionResult:
    return ExtractionResult(
        npcs=[_npc()],
        player=Player(),
        stat=Stat(),
        items=[Item(id="itm1", name="劍")],
    )


def test_check_extraction_rpg_passes_a_fully_populated_extraction():
    assert check_extraction_rpg(_full_extraction()) == []


def test_check_extraction_rpg_flags_missing_player():
    extraction = _full_extraction()
    extraction.player = None
    assert any("player" in p for p in check_extraction_rpg(extraction))


def test_check_extraction_rpg_flags_missing_stat():
    extraction = _full_extraction()
    extraction.stat = None
    assert any("stat" in p for p in check_extraction_rpg(extraction))


def test_check_extraction_rpg_flags_empty_items():
    extraction = _full_extraction()
    extraction.items = []
    problems = check_extraction_rpg(extraction)
    assert any("items" in p for p in problems)


def test_check_extraction_rpg_flags_fake_role_npc():
    extraction = _full_extraction()
    extraction.npcs.append(_npc(id_="npc_narrator", name="旁白"))
    assert any("npc_narrator" in p for p in check_extraction_rpg(extraction))


# --- check_scene_rpg -------------------------------------------------------


def test_check_scene_rpg_passes_dialogue_from_known_npc():
    event = Event(
        id="ev1", title="t", summary="s",
        preconditions=["已抵達現場"],
        dialogue=[{"npc": "npc1", "line": "hi"}],
    )
    assert check_scene_rpg(event, known_npc_ids={"npc1"}, introduced_npc_ids=set()) == []


def test_check_scene_rpg_passes_dialogue_from_already_introduced_npc():
    event = Event(
        id="ev1", title="t", summary="s",
        preconditions=["已抵達現場"],
        dialogue=[{"npc": "npc2", "line": "hi"}],
    )
    assert check_scene_rpg(event, known_npc_ids=set(), introduced_npc_ids={"npc2"}) == []


def test_check_scene_rpg_flags_dialogue_from_unknown_uintroduced_npc():
    event = Event(
        id="ev1", title="t", summary="s",
        preconditions=["已抵達現場"],
        dialogue=[{"npc": "ghost", "line": "hi"}],
    )
    problems = check_scene_rpg(event, known_npc_ids=set(), introduced_npc_ids=set())
    assert any("ghost" in p for p in problems)


def test_check_scene_rpg_flags_empty_dialogue():
    event = Event(id="ev1", title="t", summary="s", preconditions=["已抵達現場"])
    problems = check_scene_rpg(event, known_npc_ids=set(), introduced_npc_ids=set())
    assert any("沒有任何台詞" in p for p in problems)


def test_check_scene_rpg_flags_empty_preconditions():
    event = Event(
        id="ev1", title="t", summary="s",
        dialogue=[{"npc": "npc1", "line": "hi"}],
    )
    problems = check_scene_rpg(event, known_npc_ids={"npc1"}, introduced_npc_ids=set())
    assert any("preconditions" in p for p in problems)


# --- as_feedback -----------------------------------------------------------


def test_as_feedback_renders_bullet_list():
    feedback = as_feedback(["問題一", "問題二"])
    assert "問題一" in feedback
    assert "問題二" in feedback
    assert feedback.count("- ") == 2


# --- check_choice_quality --------------------------------------------------


def test_check_choice_quality_flags_missing_cost():
    event = Event(
        id="ev1", title="t", summary="s",
        choices=[Choice(id="b1", text="拿劍", next="ev1", effects="拿到劍")],
    )
    problems = check_choice_quality(event)
    assert any("缺少 cost" in p for p in problems)


def test_check_choice_quality_passes_choice_with_cost():
    event = Event(
        id="ev1", title="t", summary="s",
        choices=[Choice(id="b1", text="拿劍", next="ev1", cost="失去信任", effects="拿到劍")],
    )
    assert check_choice_quality(event) == []


def test_check_choice_quality_flags_false_choice_pair():
    event = Event(
        id="ev1", title="t", summary="s",
        choices=[
            Choice(id="b1", text="立刻上前救援受傷少女", next="ev1", delta=1),
            Choice(id="b2", text="立刻上前救援受傷少年", next="ev1", delta=1),
        ],
    )
    problems = check_choice_quality(event)
    assert any("假選擇" in p for p in problems)


def test_check_choice_quality_passes_distinct_choices():
    event = Event(
        id="ev1", title="t", summary="s",
        choices=[
            Choice(id="b1", text="正面迎戰", next="ev1", cost="消耗內力", delta=5),
            Choice(id="b2", text="悄悄潛行離開", next="ev1", cost="錯過線索", delta=-5),
        ],
    )
    assert check_choice_quality(event) == []


# --- check_truth_pacing ------------------------------------------------------


def _chapter_script(**overrides) -> Script:
    defaults = dict(
        meta=Meta(title="t"),
        chapters=[
            Chapter(id="ch1", title="c1", summary="s"),
            Chapter(id="ch2", title="c2", summary="s"),
        ],
        events=[
            Event(id="ev1", title="t1", summary="s", chapter_id="ch1"),
        ],
    )
    defaults.update(overrides)
    return Script(**defaults)


def test_check_truth_pacing_flags_hidden_fact_leak():
    script = _chapter_script(
        truth=Truth(hidden="幕後主使是丙"),
        events=[
            Event(id="ev1", title="t1", summary="真相是幕後主使是丙", chapter_id="ch1"),
        ],
    )
    problems = check_truth_pacing(script, "ch1")
    assert any("提前洩漏" in p for p in problems)


def test_check_truth_pacing_ignores_facts_not_leaked():
    script = _chapter_script(
        truth=Truth(hidden="幕後主使是丙"),
        events=[
            Event(id="ev1", title="t1", summary="眾人議論紛紛", chapter_id="ch1"),
        ],
    )
    assert check_truth_pacing(script, "ch1") == []


def test_check_truth_pacing_silent_when_no_hidden_fact():
    script = _chapter_script(truth=Truth(public="公開事實"))
    assert check_truth_pacing(script, "ch1") == []


# --- check_check_fallback ----------------------------------------------------


def test_check_check_fallback_flags_dead_end_check():
    event = Event(id="ev1", title="t", summary="s", check=Check())
    problems = check_check_fallback(event)
    assert any("可能是死路" in p for p in problems)


def test_check_check_fallback_flags_missing_failure_cost():
    event = Event(id="ev1", title="t", summary="s", check=Check(on_fail="ev1"))
    problems = check_check_fallback(event)
    assert any("fail_cost" in p for p in problems)


def test_check_check_fallback_passes_with_failure_route_and_cost():
    event = Event(
        id="ev1", title="t", summary="s",
        check=Check(on_fail="ev1", fail_cost="受傷"),
    )
    assert check_check_fallback(event) == []


def test_check_check_fallback_silent_when_no_check():
    event = Event(id="ev1", title="t", summary="s")
    assert check_check_fallback(event) == []


# --- check_scene_information --------------------------------------------


def test_check_scene_information_flags_content_free_scene():
    event = Event(id="ev1", title="t", summary="s")
    problems = check_scene_information(event)
    assert any("純填充場景" in p for p in problems)


def test_check_scene_information_passes_scene_with_clue():
    event = Event(id="ev1", title="t", summary="s", clue_ids=["c1"])
    assert check_scene_information(event) == []


def test_check_scene_information_passes_scene_with_choice_effect():
    event = Event(
        id="ev1", title="t", summary="s",
        choices=[Choice(id="b1", text="go", next="ev1", effects="拿到劍")],
    )
    assert check_scene_information(event) == []


# --- check_ending_ranges ------------------------------------------------


def test_check_ending_ranges_passes_non_overlapping():
    script = _chapter_script(endings=[
        Ending(id="e1", name="a", min=0, max=49),
        Ending(id="e2", name="b", min=50, max=100),
    ])
    assert check_ending_ranges(script) == []


def test_check_ending_ranges_flags_overlap():
    script = _chapter_script(endings=[
        Ending(id="e1", name="a", min=0, max=60),
        Ending(id="e2", name="b", min=50, max=100),
    ])
    problems = check_ending_ranges(script)
    assert any("重疊" in p for p in problems)


def test_check_ending_ranges_silent_when_no_endings():
    script = _chapter_script()
    assert check_ending_ranges(script) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
