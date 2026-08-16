"""Unit tests for crew/metrics.py's structural quality proxies -- pure
functions over hand-built Script objects, no crew/LLM involved (mirrors
test_chunking.py's "no external deps" philosophy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.crew.metrics import continuity_metrics, script_metrics  # noqa: E402
from bixiascribe.schema import NPC, Branch, DialogueLine, Event, Script  # noqa: E402


def test_metrics_on_empty_script() -> None:
    script = Script(title="t", premise="p")
    metrics = script_metrics(script)
    assert metrics["events"] == 0
    assert metrics["npcs"] == 0
    assert metrics["dialogue_lines"] == 0
    # Ratios must be 0.0, not NaN/None, when their denominator is empty --
    # so callers can average this dict across many runs without filtering.
    assert metrics["dialogue_lines_per_event"] == 0.0
    assert metrics["events_with_dialogue_pct"] == 0.0
    assert metrics["npc_speaking_pct"] == 0.0
    assert metrics["avg_line_chars"] == 0.0


def test_metrics_detect_uneven_npc_speaking_coverage() -> None:
    # One NPC does all the talking -- exactly what the `fake` LLM backend
    # produces (see llm.py::_fake_fill_dialogue, always npcs[0]).
    script = Script(
        title="t",
        premise="p",
        npcs=[
            NPC(id="npc-1", name="A", identity="x", personality="y", speech_style="z"),
            NPC(id="npc-2", name="B", identity="x", personality="y", speech_style="z"),
        ],
        events=[
            Event(
                id="evt-1",
                title="t",
                location="l",
                summary="s",
                dialogue=[DialogueLine(npc_id="npc-1", line="你好啊")],
            ),
            Event(
                id="evt-2",
                title="t",
                location="l",
                summary="s",
                dialogue=[DialogueLine(npc_id="npc-1", line="再會")],
            ),
        ],
    )
    metrics = script_metrics(script)
    assert metrics["events"] == 2
    assert metrics["npcs"] == 2
    assert metrics["dialogue_lines"] == 2
    assert metrics["dialogue_lines_per_event"] == 1.0
    assert metrics["events_with_dialogue_pct"] == 1.0
    assert metrics["npc_speaking_pct"] == 0.5  # only npc-1 ever speaks
    assert metrics["avg_line_chars"] == 2.5  # "你好啊" (3) and "再會" (2) avg to 2.5


def test_metrics_normal_case() -> None:
    script = Script(
        title="t",
        premise="p",
        npcs=[NPC(id="npc-1", name="A", identity="x", personality="y", speech_style="z")],
        events=[
            Event(
                id="evt-1",
                title="t",
                location="l",
                summary="s",
                dialogue=[DialogueLine(npc_id="npc-1", line="一")],
                branches=[Branch(id="b1", choice_text="go", next_event_id="evt-1")],
            ),
            Event(id="evt-2", title="t", location="l", summary="s", dialogue=[]),
        ],
    )
    metrics = script_metrics(script)
    assert metrics["events"] == 2
    assert metrics["branches"] == 1
    assert metrics["events_with_dialogue_pct"] == 0.5
    assert metrics["npc_speaking_pct"] == 1.0


# --- continuity_metrics() (BiXiaScribe 重構 Phase 5) -----------------------


def test_continuity_metrics_on_empty_script() -> None:
    script = Script(title="t", premise="p")
    metrics = continuity_metrics(script)
    assert metrics == {
        "distinct_event_title_pct": 0.0,
        "repeated_dialogue_pct": 0.0,
        "prior_entity_reference_pct": 0.0,
        "connected_event_pct": 0.0,
        "self_loop_branch_pct": 0.0,
    }


def _npc(id_: str, name: str) -> NPC:
    return NPC(id=id_, name=name, identity="x", personality="y", speech_style="z")


def _memoryless_script() -> Script:
    """Three events with no visibility into each other: duplicate generic
    titles, a verbatim-repeated dialogue line, self-loop branches, and no
    event ever names an earlier NPC or location -- the symptom pattern a
    scene writer with zero prior-scene context produces."""
    npcs = [_npc("npc-1", "張三"), _npc("npc-2", "李四"), _npc("npc-3", "王五")]

    def _evt(id_: str, npc_id: str, location: str) -> Event:
        return Event(
            id=id_,
            title="初遇",
            location=location,
            summary="兩人偶然相遇。",
            dialogue=[DialogueLine(npc_id=npc_id, line="你是何人？")],
            branches=[Branch(id=f"b-{id_}", choice_text="x", next_event_id=id_)],
        )

    return Script(
        title="t",
        premise="p",
        npcs=npcs,
        events=[
            _evt("evt-1", "npc-1", "廢墟"),
            _evt("evt-2", "npc-2", "荒野"),
            _evt("evt-3", "npc-3", "荒野"),
        ],
    )


def _continuous_script() -> Script:
    """Three events that build on each other: distinct titles, unique
    dialogue, each later event calls back to an earlier NPC/location, and
    branches chain forward (never self-loop) -- the pattern a scene writer
    with full prior-scene context produces."""
    npcs = [_npc("npc-1", "張三"), _npc("npc-2", "李四"), _npc("npc-3", "王五")]

    return Script(
        title="t",
        premise="p",
        npcs=npcs,
        events=[
            Event(
                id="evt-1",
                title="下山",
                location="少林寺",
                summary="張三領命下山查案。",
                dialogue=[DialogueLine(npc_id="npc-1", line="師父保重，弟子這就下山了。")],
                branches=[Branch(id="b1", choice_text="go", next_event_id="evt-2")],
            ),
            Event(
                id="evt-2",
                title="查案",
                location="村莊",
                summary="李四向剛從少林寺下山的張三問起案情。",
                dialogue=[DialogueLine(npc_id="npc-2", line="少林寺的張三少俠，你可查得此案？")],
                branches=[Branch(id="b2", choice_text="go", next_event_id="evt-3")],
            ),
            Event(
                id="evt-3",
                title="決戰",
                location="破廟",
                summary="王五在村莊外攔住了李四所說的線索。",
                dialogue=[DialogueLine(npc_id="npc-3", line="村莊那頭的線索，果然是你。")],
                branches=[],
            ),
        ],
    )


def test_continuity_metrics_memoryless_vs_continuous() -> None:
    memoryless = continuity_metrics(_memoryless_script())
    continuous = continuity_metrics(_continuous_script())

    # Higher-is-better metrics: continuous must score higher.
    assert continuous["distinct_event_title_pct"] > memoryless["distinct_event_title_pct"]
    assert (
        continuous["prior_entity_reference_pct"] > memoryless["prior_entity_reference_pct"]
    )
    assert continuous["connected_event_pct"] > memoryless["connected_event_pct"]

    # Lower-is-better metrics: continuous must score lower.
    assert continuous["repeated_dialogue_pct"] < memoryless["repeated_dialogue_pct"]
    assert continuous["self_loop_branch_pct"] < memoryless["self_loop_branch_pct"]

    # Exact values on the memoryless fixture, since it's deliberately extreme.
    assert memoryless["prior_entity_reference_pct"] == 0.0
    assert memoryless["self_loop_branch_pct"] == 1.0
    assert memoryless["connected_event_pct"] == 0.0


def test_prior_entity_reference_excludes_present_npcs() -> None:
    # evt-2 mentions npc-1 by name, but npc-1 also speaks in evt-2 itself --
    # that must NOT count as a "prior" reference (it's not a callback, it's
    # just the current scene's own cast), and evt-2's location is unchanged
    # from evt-1's, so there's no location signal either.
    script = Script(
        title="t",
        premise="p",
        npcs=[_npc("npc-1", "張三")],
        events=[
            Event(
                id="evt-1",
                title="初遇",
                location="村口",
                summary="張三初次登場。",
                dialogue=[DialogueLine(npc_id="npc-1", line="在下張三。")],
            ),
            Event(
                id="evt-2",
                title="重逢",
                location="村口",
                summary="張三再次現身。",
                dialogue=[DialogueLine(npc_id="npc-1", line="張三又回來了。")],
            ),
        ],
    )
    metrics = continuity_metrics(script)
    assert metrics["prior_entity_reference_pct"] == 0.0


def test_script_metrics_includes_continuity_keys() -> None:
    metrics = script_metrics(_continuous_script())
    for key in (
        "distinct_event_title_pct",
        "repeated_dialogue_pct",
        "prior_entity_reference_pct",
        "connected_event_pct",
        "self_loop_branch_pct",
    ):
        assert key in metrics


if __name__ == "__main__":
    test_metrics_on_empty_script()
    test_metrics_detect_uneven_npc_speaking_coverage()
    test_metrics_normal_case()
    test_continuity_metrics_on_empty_script()
    test_continuity_metrics_memoryless_vs_continuous()
    test_prior_entity_reference_excludes_present_npcs()
    test_script_metrics_includes_continuity_keys()
    print("All tests passed.")
