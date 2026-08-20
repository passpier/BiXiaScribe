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
    Chapter,
    Clue,
    DialogueLine,
    EffectOp,
    Ending,
    Event,
    ExtractionResult,
    Faction,
    FactionRelation,
    Item,
    PlayerCharacter,
    ProgressiveReveal,
    Quest,
    Region,
    Script,
    SkillCheck,
    StatCondition,
    StatThreshold,
    SubLocation,
    TruthLayer,
    Variable,
    validate_npc_introductions,
    validate_references,
    validate_stat_thresholds,
    validate_truth_layering,
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


# --- GMUD frame models: round-trips -------------------------------------


def test_faction_relation_round_trip():
    faction = Faction(
        id="f1", name="少林", alignment="正道",
        relations=[FactionRelation(faction_id="f2", stance="敵對")],
    )
    assert Faction.model_validate_json(faction.model_dump_json()) == faction


def test_region_sub_location_round_trip():
    region = Region(
        id="r1", name="洛陽", unlock_condition="",
        sub_locations=[
            SubLocation(id="sl1", name="酒樓", function="打聽消息"),
            SubLocation(id="sl2", name="醫館", function="療傷"),
        ],
    )
    assert Region.model_validate_json(region.model_dump_json()) == region


def test_truth_layer_round_trip():
    truth = TruthLayer(
        public=["江湖傳聞甲派滅門"],
        progressive=[ProgressiveReveal(id="pr1", fact="真兇是乙", reveal_chapter_id="ch2")],
        hidden=["幕後主使是丙"],
    )
    assert TruthLayer.model_validate_json(truth.model_dump_json()) == truth


def test_chapter_replaces_chapter_outline_with_new_fields():
    chapter = Chapter(
        id="ch1", title="下山", summary="s", hook="h",
        event_ids=["ev1"], converge_event_id="ev1", clue_ids=["c1"],
    )
    assert Chapter.model_validate_json(chapter.model_dump_json()) == chapter


def test_skill_check_and_ending_round_trip():
    check = SkillCheck(
        id="sk1", kind="attribute_contest", stat_id="st1", failure_branch_id="b1",
        failure_cost="受傷",
    )
    ending = Ending(
        id="e1", name="正義結局",
        stat_conditions=[StatCondition(stat_id="st1", min_value=50)],
        required_branch_ids=["b1"],
    )
    assert SkillCheck.model_validate_json(check.model_dump_json()) == check
    assert Ending.model_validate_json(ending.model_dump_json()) == ending


def test_extraction_result_defaults_include_gmud_fields():
    extraction = ExtractionResult()
    assert extraction.factions == []
    assert extraction.regions == []
    assert extraction.truth is None
    assert extraction.stat_thresholds == []
    assert extraction.clues == []
    assert extraction.endings == []


# --- validate_references(): GMUD cross-reference checks ------------------


def _gmud_script(**overrides) -> Script:
    defaults = dict(
        title="t",
        premise="p",
        player=PlayerCharacter(
            id="player", stats=[Variable(id="st1", name="聲望", initial=0, kind="stat")],
        ),
        npcs=[NPC(id="npc1", name="甲", identity="俠客", personality="剛", speech_style="直")],
        regions=[Region(id="r1", name="洛陽", sub_locations=[
            SubLocation(id="sl1", name="酒樓"), SubLocation(id="sl2", name="醫館"),
        ])],
        chapters=[Chapter(id="ch1", title="c", summary="s")],
        clues=[Clue(id="c1", name="血書", found_in_event_id="ev1")],
        events=[
            Event(id="ev1", title="t1", location="l1", summary="s1", chapter_id="ch1",
                  region_id="r1", sub_location_id="sl1", clue_ids=["c1"]),
        ],
    )
    defaults.update(overrides)
    return Script(**defaults)


def test_validate_references_accepts_valid_gmud_refs():
    assert validate_references(_gmud_script()) == []


def test_validate_references_flags_unknown_faction_relation():
    script = _gmud_script(factions=[Faction(id="f1", name="少林", relations=[
        FactionRelation(faction_id="no-such-faction"),
    ])])
    problems = validate_references(script)
    assert any("relation references unknown" in p for p in problems)


def test_validate_references_flags_unknown_npc_faction():
    script = _gmud_script()
    script.npcs[0].faction_id = "no-such-faction"
    problems = validate_references(script)
    assert any("faction_id" in p for p in problems)


def test_validate_references_flags_unknown_event_region_and_sub_location():
    script = _gmud_script()
    script.events[0].region_id = "no-such-region"
    script.events[0].sub_location_id = "no-such-sub"
    problems = validate_references(script)
    assert any("region_id" in p for p in problems)
    assert any("sub_location_id" in p for p in problems)


def test_validate_references_flags_unknown_event_chapter_and_clue():
    script = _gmud_script()
    script.events[0].chapter_id = "no-such-chapter"
    script.events[0].clue_ids = ["no-such-clue"]
    problems = validate_references(script)
    assert any("chapter_id" in p for p in problems)
    assert any("clue_ids" in p for p in problems)


def test_validate_references_flags_unknown_skill_check_targets():
    script = _gmud_script()
    script.events[0].checks = [
        SkillCheck(id="sk1", stat_id="no-such-stat", success_next_event_id="no-such-event",
                   failure_branch_id="no-such-branch", item_bypass_id="no-such-item"),
    ]
    problems = validate_references(script)
    assert any("stat_id" in p for p in problems)
    assert any("success_next_event_id" in p for p in problems)
    assert any("failure_branch_id" in p for p in problems)
    assert any("item_bypass_id" in p for p in problems)


def test_validate_references_flags_unknown_branch_payoff_and_convergence():
    script = _gmud_script()
    script.events[0].branches = [
        Branch(id="b1", choice_text="go", next_event_id="ev1",
               payoff_chapter_id="no-such-chapter", converges_to_event_id="no-such-event"),
    ]
    problems = validate_references(script)
    assert any("payoff_chapter_id" in p for p in problems)
    assert any("converges_to_event_id" in p for p in problems)


def test_validate_references_flags_unknown_stat_threshold_refs():
    script = _gmud_script(stat_thresholds=[
        StatThreshold(
            id="th1", stat_id="no-such-stat", unlocks_kind="ending", unlocks_id="no-such-ending",
        ),
        StatThreshold(id="th2", stat_id="st1", unlocks_kind="bogus_kind", unlocks_id="x"),
    ])
    problems = validate_references(script)
    assert any("th1" in p and "unknown stat_id" in p for p in problems)
    assert any("th1" in p and "ending" in p for p in problems)
    assert any("th2" in p and "unknown unlocks_kind" in p for p in problems)


def test_validate_references_flags_unknown_chapter_refs():
    script = _gmud_script(chapters=[
        Chapter(id="ch1", title="c", summary="s", converge_event_id="no-such-event",
                event_ids=["no-such-event"], clue_ids=["no-such-clue"]),
    ])
    problems = validate_references(script)
    assert any("converge_event_id" in p for p in problems)
    assert any("event_ids references unknown event" in p for p in problems)
    assert any("clue_ids references unknown clue" in p for p in problems)


def test_validate_references_flags_unknown_clue_found_in_event():
    script = _gmud_script(clues=[Clue(id="c1", name="血書", found_in_event_id="no-such-event")])
    problems = validate_references(script)
    assert any("found_in_event_id" in p for p in problems)


def test_validate_references_flags_unknown_ending_refs():
    script = _gmud_script(endings=[
        Ending(id="e1", name="結局",
               stat_conditions=[StatCondition(stat_id="no-such-stat")],
               required_branch_ids=["no-such-branch"]),
    ])
    problems = validate_references(script)
    assert any("stat_conditions references unknown" in p for p in problems)
    assert any("required_branch_ids references unknown" in p for p in problems)


def test_validate_references_flags_unknown_progressive_reveal_refs():
    script = _gmud_script(truth=TruthLayer(progressive=[
        ProgressiveReveal(id="pr1", fact="x", reveal_chapter_id="no-such-chapter",
                           reveal_event_id="no-such-event"),
    ]))
    problems = validate_references(script)
    assert any("reveal_chapter_id" in p for p in problems)
    assert any("reveal_event_id" in p for p in problems)


def test_validate_references_flags_unknown_player_token_item():
    script = _gmud_script(player=PlayerCharacter(id="player", token_item_id="no-such-item"))
    problems = validate_references(script)
    assert any("token_item_id" in p for p in problems)


# --- validate_stat_thresholds() ------------------------------------------


def test_validate_stat_thresholds_clean_when_covered_and_non_overlapping():
    script = _gmud_script(
        stat_thresholds=[
            StatThreshold(id="th1", stat_id="st1", min_value=0, max_value=49,
                          unlocks_kind="ending", unlocks_id="e1"),
            StatThreshold(id="th2", stat_id="st1", min_value=50, max_value=100,
                          unlocks_kind="ending", unlocks_id="e1"),
        ],
        endings=[Ending(id="e1", name="結局")],
        events=[
            Event(id="ev1", title="t1", location="l1", summary="s1", chapter_id="ch1",
                  region_id="r1", sub_location_id="sl1", clue_ids=["c1"],
                  branches=[Branch(
                      id="b1", choice_text="go", next_event_id="ev1",
                      effect_ops=[EffectOp(target_kind="stat", target_id="st1", op="add")],
                  )]),
        ],
    )
    assert validate_stat_thresholds(script) == []


def test_validate_stat_thresholds_flags_uncovered_stat():
    script = _gmud_script(
        events=[
            Event(id="ev1", title="t1", location="l1", summary="s1", chapter_id="ch1",
                  region_id="r1", sub_location_id="sl1", clue_ids=["c1"],
                  branches=[Branch(
                      id="b1", choice_text="go", next_event_id="ev1",
                      effect_ops=[EffectOp(target_kind="stat", target_id="st1", op="add")],
                  )]),
        ],
    )
    problems = validate_stat_thresholds(script)
    assert any("no narrative meaning" in p for p in problems)


def test_validate_stat_thresholds_flags_overlapping_ranges():
    script = _gmud_script(stat_thresholds=[
        StatThreshold(id="th1", stat_id="st1", min_value=0, max_value=60,
                      unlocks_kind="ending", unlocks_id="e1"),
        StatThreshold(id="th2", stat_id="st1", min_value=50, max_value=100,
                      unlocks_kind="ending", unlocks_id="e1"),
    ])
    problems = validate_stat_thresholds(script)
    assert any("overlapping ranges" in p for p in problems)


def test_validate_stat_thresholds_flags_threshold_unlocking_nothing():
    script = _gmud_script(stat_thresholds=[StatThreshold(id="th1", stat_id="st1")])
    problems = validate_stat_thresholds(script)
    assert any("does not unlock anything" in p for p in problems)


# --- validate_truth_layering() -------------------------------------------


def test_validate_truth_layering_clean_when_non_decreasing():
    script = _gmud_script(
        chapters=[
            Chapter(id="ch1", title="c1", summary="s"),
            Chapter(id="ch2", title="c2", summary="s"),
        ],
        truth=TruthLayer(progressive=[
            ProgressiveReveal(id="pr1", fact="x", reveal_chapter_id="ch1"),
            ProgressiveReveal(id="pr2", fact="y", reveal_chapter_id="ch2"),
        ]),
    )
    assert validate_truth_layering(script) == []


def test_validate_truth_layering_flags_decreasing_order():
    script = _gmud_script(
        chapters=[
            Chapter(id="ch1", title="c1", summary="s"),
            Chapter(id="ch2", title="c2", summary="s"),
        ],
        truth=TruthLayer(progressive=[
            ProgressiveReveal(id="pr1", fact="x", reveal_chapter_id="ch2"),
            ProgressiveReveal(id="pr2", fact="y", reveal_chapter_id="ch1"),
        ]),
    )
    problems = validate_truth_layering(script)
    assert any("non-decreasing chapter order" in p for p in problems)


def test_validate_truth_layering_silent_when_no_truth():
    script = _gmud_script()
    assert validate_truth_layering(script) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
