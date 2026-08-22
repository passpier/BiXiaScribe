"""Unit tests for the layered-generation data models added in Phase 1 of
BiXiaScribe_REFACTORING_PLAN.md (Outline/Beat/ExtractionResult/CausalPlotGraph
and their validators). No external deps, no API key needed -- mirrors
test_chunking.py's "no external deps" philosophy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.schema import (  # noqa: E402
    NPC,
    Beat,
    BeatSheet,
    CausalPlotGraph,
    Chapter,
    Event,
    ExtractionResult,
    Meta,
    Outline,
    PlotEdge,
    PlotNode,
    Script,
    Stat,
    parse_model_json,
    parse_script_json,
    validate_causal_graph,
    validate_outline_beats,
)


def _outline() -> Outline:
    return Outline(
        title="下山",
        premise="少林弟子下山查案",
        chapters=[
            Chapter(id="ch-1", title="啟程", summary="下山"),
            Chapter(id="ch-2", title="查案", summary="調查"),
        ],
    )


def test_outline_and_chapter_construct_and_round_trip() -> None:
    outline = _outline()
    dumped = outline.model_dump_json()
    reloaded = Outline.model_validate_json(dumped)
    assert reloaded == outline
    assert reloaded.chapters[0].id == "ch-1"


def test_chapter_fields_default_and_round_trip() -> None:
    chapter = Chapter(id="ch-1", title="啟程", summary="下山")
    assert chapter.loc == ""
    assert chapter.start_event == ""
    filled = Chapter(id="ch-1", title="啟程", summary="下山", loc="山門", start_event="ev-2")
    assert Chapter.model_validate_json(filled.model_dump_json()) == filled


def test_beat_defaults_and_round_trips() -> None:
    beat = Beat(id="beat-1", chapter_id="ch-1", summary="s")
    assert beat.npc_ids == []
    assert Beat.model_validate_json(beat.model_dump_json()) == beat


def test_beat_sheet_construct_and_round_trip() -> None:
    outline = _outline()
    beats = [
        Beat(id="beat-1", chapter_id="ch-1", summary="出發", npc_ids=["npc-1"]),
        Beat(
            id="beat-2",
            chapter_id="ch-2",
            summary="查案",
            npc_ids=["npc-1"],
            causal_deps=["beat-1"],
        ),
    ]
    sheet = BeatSheet(outline=outline, beats=beats)
    reloaded = BeatSheet.model_validate_json(sheet.model_dump_json())
    assert reloaded == sheet
    assert reloaded.beats[1].causal_deps == ["beat-1"]


def test_extraction_result_defaults_and_round_trip() -> None:
    result = ExtractionResult(
        npcs=[NPC(id="npc-1", name="A", personality="y", speech_style="z")],
        stat=Stat(id="var-1", name="v", init=0),
    )
    assert result.items == []
    reloaded = ExtractionResult.model_validate_json(result.model_dump_json())
    assert reloaded == result


def test_causal_plot_graph_construct_and_round_trip() -> None:
    graph = CausalPlotGraph(
        nodes=[PlotNode(id="n1"), PlotNode(id="n2")],
        edges=[PlotEdge(from_id="n1", to_id="n2", condition="var==1")],
    )
    reloaded = CausalPlotGraph.model_validate_json(graph.model_dump_json())
    assert reloaded == graph


def test_validate_causal_graph_clean_graph_returns_empty() -> None:
    graph = CausalPlotGraph(
        nodes=[PlotNode(id="n1"), PlotNode(id="n2")],
        edges=[PlotEdge(from_id="n1", to_id="n2")],
    )
    assert validate_causal_graph(graph) == []


def test_validate_causal_graph_reports_dangling_edge_endpoints() -> None:
    graph = CausalPlotGraph(
        nodes=[PlotNode(id="n1")],
        edges=[PlotEdge(from_id="n1", to_id="n-missing")],
    )
    problems = validate_causal_graph(graph)
    assert len(problems) == 1
    assert "n-missing" in problems[0]


def test_validate_causal_graph_reports_dangling_from_id() -> None:
    graph = CausalPlotGraph(
        nodes=[PlotNode(id="n1")],
        edges=[PlotEdge(from_id="n-missing", to_id="n1")],
    )
    problems = validate_causal_graph(graph)
    assert len(problems) == 1
    assert "n-missing" in problems[0]


def test_validate_outline_beats_consistent_returns_empty() -> None:
    outline = _outline()
    beats = [
        Beat(id="beat-1", chapter_id="ch-1", summary="出發"),
        Beat(id="beat-2", chapter_id="ch-2", summary="查案"),
    ]
    assert validate_outline_beats(outline, beats) == []


def test_validate_outline_beats_reports_unknown_chapter_id() -> None:
    outline = _outline()
    beats = [Beat(id="beat-1", chapter_id="ch-missing", summary="出發")]
    problems = validate_outline_beats(outline, beats)
    assert len(problems) == 1
    assert "beat-1" in problems[0]
    assert "ch-missing" in problems[0]


def test_parse_model_json_salvages_wrapped_json() -> None:
    text = "好的：\n\n```json\n" + _outline().model_dump_json() + "\n```\n完成。"
    result = parse_model_json(text, Outline)
    assert isinstance(result, Outline)
    assert result.title == "下山"


def test_parse_model_json_rejects_non_matching_schema() -> None:
    # An ExtractionResult JSON doesn't validate as an Outline (no
    # title/premise, and pydantic ignores its extra fields), so it must be
    # rejected rather than partially/incorrectly coerced.
    extraction_text = ExtractionResult(
        npcs=[NPC(id="npc-1", name="A", personality="y", speech_style="z")]
    ).model_dump_json()
    assert parse_model_json(extraction_text, Outline) is None


def test_parse_model_json_returns_none_for_no_json() -> None:
    assert parse_model_json("這裡完全沒有 JSON。", Outline) is None


def test_parse_script_json_still_works_via_parse_model_json() -> None:
    # Regression: parse_script_json must keep its original signature/return
    # type after becoming a thin wrapper around parse_model_json.
    text = "結果：\n" + Script(meta=Meta(title="x")).model_dump_json() + "\n完畢。"
    script = parse_script_json(text)
    assert isinstance(script, Script)
    assert script.meta.title == "x"


def test_event_and_beat_are_independent_models() -> None:
    # Sanity check the new models don't accidentally collide with the
    # existing Event model's field names/types.
    event = Event(id="e1", title="t", summary="s")
    beat = Beat(id="beat-1", chapter_id="ch-1", summary="s")
    assert event.id != beat.id
    assert not hasattr(beat, "dialogue")


if __name__ == "__main__":
    test_outline_and_chapter_construct_and_round_trip()
    test_chapter_fields_default_and_round_trip()
    test_beat_defaults_and_round_trips()
    test_beat_sheet_construct_and_round_trip()
    test_extraction_result_defaults_and_round_trip()
    test_causal_plot_graph_construct_and_round_trip()
    test_validate_causal_graph_clean_graph_returns_empty()
    test_validate_causal_graph_reports_dangling_edge_endpoints()
    test_validate_causal_graph_reports_dangling_from_id()
    test_validate_outline_beats_consistent_returns_empty()
    test_validate_outline_beats_reports_unknown_chapter_id()
    test_parse_model_json_salvages_wrapped_json()
    test_parse_model_json_rejects_non_matching_schema()
    test_parse_model_json_returns_none_for_no_json()
    test_parse_script_json_still_works_via_parse_model_json()
    test_event_and_beat_are_independent_models()
    print("All tests passed.")
