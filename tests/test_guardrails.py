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
    check_convergence,
    check_delayed_payoff,
    check_extraction_rpg,
    check_scene_information,
    check_scene_mix,
    check_scene_rpg,
    check_script_rpg,
    check_single_stat,
    check_stat_narrative,
    check_truth_pacing,
)
from bixiascribe.schema import (  # noqa: E402
    NPC,
    Branch,
    Chapter,
    DialogueLine,
    EffectOp,
    Event,
    ExtractionResult,
    Item,
    PlayerCharacter,
    ProgressiveReveal,
    Script,
    SkillCheck,
    StatThreshold,
    TruthLayer,
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
    )


def test_check_extraction_rpg_passes_a_fully_populated_extraction():
    assert check_extraction_rpg(_full_extraction()) == []


def test_check_extraction_rpg_flags_missing_player():
    extraction = _full_extraction()
    extraction.player = None
    assert any("player" in p for p in check_extraction_rpg(extraction))


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


# --- check_choice_quality --------------------------------------------------


def test_check_choice_quality_flags_missing_cost():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        branches=[
            Branch(id="b1", choice_text="拿劍", next_event_id="ev1",
                   effect_ops=[EffectOp(target_kind="item", target_id="itm1", op="give")]),
        ],
    )
    problems = check_choice_quality(event)
    assert any("缺少 cost" in p for p in problems)


def test_check_choice_quality_passes_branch_with_cost():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        branches=[
            Branch(id="b1", choice_text="拿劍", next_event_id="ev1", cost="失去信任",
                   effect_ops=[EffectOp(target_kind="item", target_id="itm1", op="give")]),
        ],
    )
    assert check_choice_quality(event) == []


def test_check_choice_quality_flags_false_choice_pair():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        branches=[
            Branch(id="b1", choice_text="立刻上前救援受傷少女", next_event_id="ev1",
                   effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add", value="1")]),
            Branch(id="b2", choice_text="立刻上前救援受傷少年", next_event_id="ev1",
                   effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add", value="1")]),
        ],
    )
    problems = check_choice_quality(event)
    assert any("假選擇" in p for p in problems)


def test_check_choice_quality_passes_distinct_branches():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        branches=[
            Branch(id="b1", choice_text="正面迎戰", next_event_id="ev1", cost="消耗內力",
                   effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add")]),
            Branch(id="b2", choice_text="悄悄潛行離開", next_event_id="ev1", cost="錯過線索",
                   effect_ops=[EffectOp(target_kind="item", target_id="itm1", op="take")]),
        ],
    )
    assert check_choice_quality(event) == []


# --- check_delayed_payoff ---------------------------------------------------


def _chapter_script(**overrides) -> Script:
    defaults = dict(
        title="t", premise="p",
        chapters=[
            Chapter(id="ch1", title="c1", summary="s"),
            Chapter(id="ch2", title="c2", summary="s"),
        ],
        events=[
            Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1"),
        ],
    )
    defaults.update(overrides)
    return Script(**defaults)


def test_check_delayed_payoff_flags_undeclared_deferred_effect():
    script = _chapter_script(events=[
        Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1",
              branches=[Branch(
                  id="b1", choice_text="go", next_event_id="ev1",
                  effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add")],
              )]),
    ])
    problems = check_delayed_payoff(script)
    assert any("payoff_description" in p for p in problems)


def test_check_delayed_payoff_passes_immediate_feedback():
    script = _chapter_script(events=[
        Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1",
              branches=[Branch(
                  id="b1", choice_text="go", next_event_id="ev1",
                  immediate_feedback="當場獲得聲望",
                  effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add")],
              )]),
    ])
    assert check_delayed_payoff(script) == []


# --- check_stat_narrative ---------------------------------------------------


def test_check_stat_narrative_flags_uncovered_stat():
    script = _chapter_script(events=[
        Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1",
              branches=[Branch(
                  id="b1", choice_text="go", next_event_id="ev1",
                  effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add")],
              )]),
    ])
    assert any("narrative meaning" in p for p in check_stat_narrative(script))


def test_check_stat_narrative_passes_covered_stat():
    script = _chapter_script(
        stat_thresholds=[
            StatThreshold(id="th1", stat_id="rep", unlocks_kind="event", unlocks_id="ev1"),
        ],
        events=[
            Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1",
                  branches=[Branch(
                      id="b1", choice_text="go", next_event_id="ev1",
                      effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add")],
                  )]),
        ],
    )
    assert check_stat_narrative(script) == []


# --- check_single_stat --------------------------------------------------


def test_check_single_stat_passes_one_stat_with_three_thresholds():
    script = _chapter_script(
        player=PlayerCharacter(
            id="player", stats=[Variable(id="rep", name="心境值", initial=0, kind="stat")],
        ),
        stat_thresholds=[
            StatThreshold(id="th1", stat_id="rep", min_value=0, max_value=30,
                          unlocks_kind="ending", unlocks_id="e1"),
            StatThreshold(id="th2", stat_id="rep", min_value=31, max_value=70,
                          unlocks_kind="ending", unlocks_id="e1"),
            StatThreshold(id="th3", stat_id="rep", min_value=71, max_value=100,
                          unlocks_kind="ending", unlocks_id="e1"),
        ],
    )
    assert check_single_stat(script) == []


def test_check_single_stat_flags_more_than_one_stat():
    script = _chapter_script(
        player=PlayerCharacter(
            id="player",
            stats=[
                Variable(id="rep", name="心境值", initial=0, kind="stat"),
                Variable(id="hp", name="內力", initial=100, kind="stat"),
            ],
        ),
    )
    problems = check_single_stat(script)
    assert any("唯一數值" in p for p in problems)


def test_check_single_stat_flags_fewer_than_three_thresholds():
    script = _chapter_script(
        player=PlayerCharacter(
            id="player", stats=[Variable(id="rep", name="心境值", initial=0, kind="stat")],
        ),
        stat_thresholds=[
            StatThreshold(id="th1", stat_id="rep", min_value=0, max_value=50,
                          unlocks_kind="ending", unlocks_id="e1"),
        ],
    )
    problems = check_single_stat(script)
    assert any("至少要切成 3 個區間" in p for p in problems)


def test_check_single_stat_silent_when_no_player():
    script = _chapter_script()
    assert check_single_stat(script) == []


# --- check_truth_pacing ------------------------------------------------------


def test_check_truth_pacing_flags_hidden_fact_leak():
    script = _chapter_script(
        truth=TruthLayer(hidden=["幕後主使是丙"]),
        events=[
            Event(
                id="ev1", title="t1", location="l", summary="真相是幕後主使是丙",
                chapter_id="ch1",
            ),
        ],
    )
    problems = check_truth_pacing(script, "ch1")
    assert any("提前洩漏" in p for p in problems)


def test_check_truth_pacing_ignores_facts_not_leaked():
    script = _chapter_script(
        truth=TruthLayer(hidden=["幕後主使是丙"]),
        events=[
            Event(id="ev1", title="t1", location="l", summary="眾人議論紛紛", chapter_id="ch1"),
        ],
    )
    assert check_truth_pacing(script, "ch1") == []


def test_check_truth_pacing_allows_reveal_after_its_own_chapter():
    script = _chapter_script(
        truth=TruthLayer(progressive=[
            ProgressiveReveal(id="pr1", fact="真兇是乙", reveal_chapter_id="ch2"),
        ]),
        events=[
            Event(id="ev1", title="t1", location="l", summary="真兇是乙", chapter_id="ch2"),
        ],
    )
    assert check_truth_pacing(script, "ch2") == []


# --- check_convergence -------------------------------------------------------


def test_check_convergence_flags_missing_converge_point():
    script = _chapter_script(events=[
        Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1",
              branches=[Branch(id="b1", choice_text="a", next_event_id="ev2"),
                        Branch(id="b2", choice_text="b", next_event_id="ev3")]),
        Event(id="ev2", title="t2", location="l", summary="s", chapter_id="ch1"),
        Event(id="ev3", title="t3", location="l", summary="s", chapter_id="ch1"),
    ])
    problems = check_convergence(script)
    assert any("converge_event_id" in p for p in problems)


def test_check_convergence_passes_reachable_convergence():
    script = _chapter_script(
        chapters=[Chapter(id="ch1", title="c1", summary="s", converge_event_id="ev3")],
        events=[
            Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1",
                  branches=[Branch(id="b1", choice_text="a", next_event_id="ev2"),
                            Branch(id="b2", choice_text="b", next_event_id="ev3")]),
            Event(id="ev2", title="t2", location="l", summary="s", chapter_id="ch1",
                  branches=[Branch(id="b3", choice_text="c", next_event_id="ev3")]),
            Event(id="ev3", title="t3", location="l", summary="s", chapter_id="ch1"),
        ],
    )
    assert check_convergence(script) == []


def test_check_convergence_flags_too_many_branches():
    script = _chapter_script(events=[
        Event(id="ev1", title="t1", location="l", summary="s", chapter_id="ch1",
              branches=[
                  Branch(id=f"b{i}", choice_text=f"c{i}", next_event_id="ev1") for i in range(4)
              ]),
    ])
    problems = check_convergence(script)
    assert any("選項數量超過 3 個" in p for p in problems)


# --- check_check_fallback ----------------------------------------------------


def test_check_check_fallback_flags_dead_end_check():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        checks=[SkillCheck(id="sk1", stat_id="rep")],
    )
    problems = check_check_fallback(event)
    assert any("可能是死路" in p for p in problems)


def test_check_check_fallback_flags_missing_failure_cost():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        checks=[SkillCheck(id="sk1", stat_id="rep", failure_branch_id="b1")],
    )
    problems = check_check_fallback(event)
    assert any("failure_cost" in p for p in problems)


def test_check_check_fallback_passes_with_failure_branch_and_cost():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        checks=[SkillCheck(id="sk1", stat_id="rep", failure_branch_id="b1", failure_cost="受傷")],
    )
    assert check_check_fallback(event) == []


# --- check_scene_information --------------------------------------------


def test_check_scene_information_flags_content_free_scene():
    event = Event(id="ev1", title="t", location="l", summary="s")
    problems = check_scene_information(event)
    assert any("純填充場景" in p for p in problems)


def test_check_scene_information_passes_scene_with_clue():
    event = Event(id="ev1", title="t", location="l", summary="s", clue_ids=["c1"])
    assert check_scene_information(event) == []


def test_check_scene_information_passes_scene_with_item_effect():
    event = Event(
        id="ev1", title="t", location="l", summary="s",
        branches=[Branch(id="b1", choice_text="go", next_event_id="ev1",
                          effect_ops=[EffectOp(target_kind="item", target_id="itm1", op="give")])],
    )
    assert check_scene_information(event) == []


# --- check_scene_mix ---------------------------------------------------------


def test_check_scene_mix_passes_when_main_at_or_above_flavor():
    script = _chapter_script(events=[
        Event(id="ev1", title="t1", location="l", summary="s", scene_kind="main"),
        Event(id="ev2", title="t2", location="l", summary="s", scene_kind="flavor"),
    ])
    assert check_scene_mix(script) == []


def test_check_scene_mix_flags_flavor_heavy_script():
    script = _chapter_script(events=[
        Event(id="ev1", title="t1", location="l", summary="s", scene_kind="main"),
        Event(id="ev2", title="t2", location="l", summary="s", scene_kind="flavor"),
        Event(id="ev3", title="t3", location="l", summary="s", scene_kind="flavor"),
    ])
    problems = check_scene_mix(script)
    assert any("調味場景" in p for p in problems)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
