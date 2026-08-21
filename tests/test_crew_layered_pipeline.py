"""Runs the layered Stage 2 pipeline (拆書 -> 排場 -> 逐場寫戲 -> 校對) end-to-end
against LLM_BACKEND=fake, so it needs no API key, no network access, and
costs nothing -- mirrors test_crew_pipeline.py's approach for the legacy
pipeline.

Set LLM_BACKEND before importing anything from bixiascribe: config.py reads
it from the environment at import time, so it must be set first.
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ["LLM_BACKEND"] = "fake"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402
from bixiascribe.crew.pipeline import (  # noqa: E402
    _coerce_model,
    run_layered_pipeline,
    run_pipeline,
)
from bixiascribe.llm import (  # noqa: E402
    ROLE_BEAT_EXPANDER,
    ROLE_DIALOGUE,
    ROLE_EXTRACTOR,
    ROLE_PROOFREADER,
    ROLE_SCENE_WRITER,
    ROLE_WRITER,
    ModelChoice,
)
from bixiascribe.schema import (  # noqa: E402
    NPC,
    Beat,
    BeatSheet,
    Chapter,
    ExtractionResult,
    Outline,
    Script,
    validate_references,
)

REQUIREMENT = "少林俗家弟子奉命下山，追查一樁滅門血案背後的血衣門餘孽。"


@contextmanager
def _isolated_state_dir():
    """run_layered_pipeline() is a thin wrapper over
    crew/orchestrator.py::run_layered(), which always checkpoints to
    config.BIXIA_STATE_DIR -- without this, every run in this module (three
    of them, at REQUIREMENT's fixed slug) writes a real
    .bixia_state/<run_id>/ dir on every `pytest tests/` invocation. Same
    save/restore pattern as tests/test_generation.py's own
    _isolated_state_dir() and tests/test_orchestrator.py's."""
    original = config.BIXIA_STATE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.BIXIA_STATE_DIR = Path(tmp)
        try:
            yield
        finally:
            config.BIXIA_STATE_DIR = original


def test_layered_pipeline_produces_valid_script() -> None:
    with _isolated_state_dir():
        script, report = run_layered_pipeline(REQUIREMENT, verbose=False)

    assert isinstance(script, Script)
    assert script.title
    assert script.npcs, "extractor should have produced at least one NPC"
    assert script.events, "beat_expander/scene_writer should have produced at least one event"
    assert all(event.dialogue for event in script.events), (
        "scene_writer should fill dialogue for every event, not leave any empty"
    )
    assert validate_references(script) == []

    assert report.mode == "layered"
    assert report.scenes_generated == len(script.events)


def test_layered_pipeline_event_ids_match_beat_ids() -> None:
    # The pipeline must not trust a scene_writer's own id choice -- it
    # always overwrites Event.id with the beat's id, since that's the
    # invariant a later parallel-scenes phase depends on to avoid
    # collisions. Under LLM_BACKEND=fake, _fake_beat_sheet() always yields
    # beats beat_depart/beat_village/beat_clue.
    with _isolated_state_dir():
        script, _report = run_layered_pipeline(REQUIREMENT, verbose=False)
    event_ids = {event.id for event in script.events}
    assert event_ids == {"beat_depart", "beat_village", "beat_clue"}


def test_layered_pipeline_reports_model_override() -> None:
    models = ModelChoice(
        extractor="ex-model", beat_expander="be-model", scene_writer="sw-model"
    )
    with _isolated_state_dir():
        _script, report = run_layered_pipeline(REQUIREMENT, verbose=False, models=models)
    assert report.model_extractor == "ex-model"
    assert report.model_beat_expander == "be-model"
    assert report.model_scene_writer == "sw-model"


def test_legacy_pipeline_unaffected_by_layered_addition() -> None:
    # Regression: adding run_layered_pipeline() must not change the legacy
    # single-shot pipeline's behavior at all.
    script = run_pipeline(REQUIREMENT, verbose=False)
    assert isinstance(script, Script)
    assert validate_references(script) == []


def test_model_choice_for_role_maps_layered_roles() -> None:
    models = ModelChoice(
        writer="w", dialogue="d", proof="p",
        extractor="ex", beat_expander="be", scene_writer="sw",
    )
    assert models.for_role(ROLE_WRITER) == "w"
    assert models.for_role(ROLE_DIALOGUE) == "d"
    assert models.for_role(ROLE_PROOFREADER) == "p"
    assert models.for_role(ROLE_EXTRACTOR) == "ex"
    assert models.for_role(ROLE_BEAT_EXPANDER) == "be"
    assert models.for_role(ROLE_SCENE_WRITER) == "sw"


class _FakeOutput:
    """Stand-in for CrewOutput/TaskOutput's shared pydantic/json_dict/raw
    fields, mirrors test_crew_pipeline.py's _FakeOutput but generic over
    model_cls via _coerce_model."""

    def __init__(self, pydantic=None, json_dict=None, raw=""):
        self.pydantic = pydantic
        self.json_dict = json_dict
        self.raw = raw


def _minimal_outline_json() -> str:
    return Outline(title="t", premise="p").model_dump_json()


def test_coerce_model_prefers_pydantic_for_any_model() -> None:
    outline = Outline(title="from-pydantic", premise="p")
    output = _FakeOutput(pydantic=outline, json_dict={"title": "wrong"}, raw="{}")
    result, source = _coerce_model(output, Outline)
    assert result is outline
    assert source == "pydantic"


def test_coerce_model_falls_back_to_json_dict() -> None:
    output = _FakeOutput(pydantic=None, json_dict={"title": "from-dict", "premise": "p"})
    result, source = _coerce_model(output, Outline)
    assert isinstance(result, Outline)
    assert result.title == "from-dict"
    assert source == "json_dict"


def test_coerce_model_falls_back_to_raw_scan() -> None:
    prose_wrapped = "這是結果：\n" + _minimal_outline_json() + "\n完畢。"
    output = _FakeOutput(pydantic=None, json_dict=None, raw=prose_wrapped)
    result, source = _coerce_model(output, Outline)
    assert isinstance(result, Outline)
    assert result.title == "t"
    assert source == "raw_scan"


def test_coerce_model_returns_none_when_nothing_salvageable() -> None:
    output = _FakeOutput(pydantic=None, json_dict=None, raw="沒有 JSON")
    result, source = _coerce_model(output, Outline)
    assert result is None
    assert source is None


def test_extraction_result_and_beat_sheet_are_independent_of_script() -> None:
    # Sanity check the new intermediate models used by the layered pipeline
    # don't accidentally get treated as interchangeable with Script.
    extraction = ExtractionResult(
        npcs=[NPC(id="npc-1", name="A", identity="x", personality="y", speech_style="z")]
    )
    output = _FakeOutput(pydantic=extraction)
    result, source = _coerce_model(output, ExtractionResult)
    assert result is extraction
    assert source == "pydantic"

    outline = Outline(
        title="t", premise="p", chapters=[Chapter(id="c1", title="t", summary="s")]
    )
    beat_sheet = BeatSheet(outline=outline, beats=[Beat(id="b1", chapter_id="c1", summary="s")])
    output2 = _FakeOutput(pydantic=beat_sheet)
    result2, source2 = _coerce_model(output2, BeatSheet)
    assert result2 is beat_sheet
    assert source2 == "pydantic"


if __name__ == "__main__":
    test_layered_pipeline_produces_valid_script()
    test_layered_pipeline_event_ids_match_beat_ids()
    test_layered_pipeline_reports_model_override()
    test_legacy_pipeline_unaffected_by_layered_addition()
    test_model_choice_for_role_maps_layered_roles()
    test_coerce_model_prefers_pydantic_for_any_model()
    test_coerce_model_falls_back_to_json_dict()
    test_coerce_model_falls_back_to_raw_scan()
    test_coerce_model_returns_none_when_nothing_salvageable()
    test_extraction_result_and_beat_sheet_are_independent_of_script()
    print("All tests passed.")
