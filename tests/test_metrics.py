"""Unit tests for crew/metrics.py's structural quality proxies -- pure
functions over hand-built Script objects, no crew/LLM involved (mirrors
test_chunking.py's "no external deps" philosophy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.crew.metrics import script_metrics  # noqa: E402
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


if __name__ == "__main__":
    test_metrics_on_empty_script()
    test_metrics_detect_uneven_npc_speaking_coverage()
    test_metrics_normal_case()
    print("All tests passed.")
