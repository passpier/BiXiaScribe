"""Runs the Stage 2 CrewAI pipeline (編劇 -> 對話 -> 校對) end-to-end against
LLM_BACKEND=fake, so it needs no API key, no network access, and costs
nothing -- mirroring test_chunking.py's "no external deps" philosophy.

Set LLM_BACKEND before importing anything from bixiascribe: config.py reads
it from the environment at import time, so it must be set first.
"""
import os
import sys
from pathlib import Path

os.environ["LLM_BACKEND"] = "fake"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from bixiascribe.crew.pipeline import (  # noqa: E402
    _coerce_script,
    run_pipeline,
    run_pipeline_with_report,
)
from bixiascribe.llm import ROLE_DIALOGUE, ROLE_PROOFREADER, ROLE_WRITER, ModelChoice  # noqa: E402
from bixiascribe.schema import (  # noqa: E402
    NPC,
    Event,
    Script,
    parse_script_json,
    validate_references,
)

REQUIREMENT = "少林俗家弟子奉命下山，追查一樁滅門血案背後的血衣門餘孽。"


def test_pipeline_produces_valid_script() -> None:
    script = run_pipeline(REQUIREMENT, verbose=False)

    # 1. Basic shape: three agents' worth of content actually landed.
    assert isinstance(script, Script)
    assert script.title
    assert script.npcs, "writer agent should have produced at least one NPC"
    assert script.events, "writer agent should have produced at least one event"

    # 2. Dialogue agent actually filled something in (not left empty by the
    #    writer's dialogue=[] skeleton).
    assert any(event.dialogue for event in script.events), (
        "dialogue agent should have filled in at least one event's dialogue"
    )

    # 3. Schema + cross-reference integrity, as checked by the proofreader
    #    stage (and re-verified here in Python -- see crew/pipeline.py).
    assert validate_references(script) == []


def test_pipeline_is_resumable_across_runs() -> None:
    # Sanity check the pipeline is a pure function of its input under the
    # fake backend (no hidden state leaking between runs / agents).
    first = run_pipeline(REQUIREMENT, verbose=False)
    second = run_pipeline(REQUIREMENT, verbose=False)
    assert first.model_dump() == second.model_dump()


def _minimal_script_json() -> str:
    return Script(title="x", premise="y").model_dump_json()


def test_parse_script_json_from_prose_wrapped_output():
    text = (
        "好的，這是修正後的劇本：\n\n```json\n"
        + _minimal_script_json()
        + "\n```\n以上完成校對。"
    )
    script = parse_script_json(text)
    assert isinstance(script, Script)
    assert script.title == "x"


def test_parse_script_json_ignores_json_schema():
    # CrewAI injects the Script JSON Schema into prompts for output_pydantic
    # tasks; its "properties" object also contains "npcs"/"events" keys, so
    # a naive "does this dict have those keys" scan would misfire on it.
    # parse_script_json must reject it since it doesn't validate as a Script.
    import json

    schema_text = json.dumps(Script.model_json_schema())
    assert parse_script_json(schema_text) is None


def test_parse_script_json_returns_none_for_no_json():
    assert parse_script_json("這裡完全沒有 JSON。") is None


class _FakeOutput:
    """Stand-in for CrewOutput/TaskOutput's shared pydantic/json_dict/raw
    fields, so _coerce_script's three-tier fallback can be tested without
    spinning up a real crew."""

    def __init__(self, pydantic=None, json_dict=None, raw=""):
        self.pydantic = pydantic
        self.json_dict = json_dict
        self.raw = raw


def test_coerce_script_prefers_pydantic():
    script = Script(title="from-pydantic", premise="p")
    output = _FakeOutput(pydantic=script, json_dict={"title": "wrong"}, raw="{}")
    result, source = _coerce_script(output)
    assert result is script
    assert source == "pydantic"


def test_coerce_script_falls_back_to_json_dict():
    output = _FakeOutput(pydantic=None, json_dict={"title": "from-dict", "premise": "p"})
    result, source = _coerce_script(output)
    assert isinstance(result, Script)
    assert result.title == "from-dict"
    assert source == "json_dict"


def test_coerce_script_falls_back_to_raw():
    prose_wrapped = "這是結果：\n" + _minimal_script_json() + "\n完畢。"
    output = _FakeOutput(pydantic=None, json_dict=None, raw=prose_wrapped)
    result, source = _coerce_script(output)
    assert isinstance(result, Script)
    assert result.title == "x"
    assert source == "raw_scan"


def test_coerce_script_returns_none_when_nothing_salvageable():
    output = _FakeOutput(pydantic=None, json_dict=None, raw="沒有 JSON")
    result, source = _coerce_script(output)
    assert result is None
    assert source is None


def test_validate_references_reports_dangling_ids():
    # No existing test exercised this failure branch: the fake LLM backend
    # always attributes dialogue to npcs[0], so it never produces a dangling
    # npc_id or next_event_id.
    script = Script(
        title="t",
        premise="p",
        npcs=[NPC(id="npc-1", name="A", identity="x", personality="y", speech_style="z")],
        events=[
            Event(
                id="event-1",
                title="t",
                location="l",
                summary="s",
                dialogue=[{"npc_id": "npc-missing", "line": "..."}],
                branches=[{"id": "b1", "choice_text": "go", "next_event_id": "event-missing"}],
            )
        ],
    )
    problems = validate_references(script)
    assert len(problems) == 2
    assert any("npc-missing" in p for p in problems)
    assert any("event-missing" in p for p in problems)


def test_model_choice_for_role_maps_each_role() -> None:
    models = ModelChoice(writer="w-model", dialogue="d-model", proof="p-model")
    assert models.for_role(ROLE_WRITER) == "w-model"
    assert models.for_role(ROLE_DIALOGUE) == "d-model"
    assert models.for_role(ROLE_PROOFREADER) == "p-model"


def test_model_choice_for_role_raises_on_unknown_role() -> None:
    models = ModelChoice()
    try:
        models.for_role("not-a-real-role")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown role")


def test_run_pipeline_with_report_reflects_model_override() -> None:
    # Under LLM_BACKEND=fake, build_llm() never looks at the model id, so
    # this only checks that the override flows through to the report --
    # not that it changes generation (that needs a real backend, exercised
    # by scripts/eval_generation.py, not this offline suite).
    models = ModelChoice(writer="w-model", dialogue="d-model", proof="p-model")
    _script, report = run_pipeline_with_report(REQUIREMENT, verbose=False, models=models)
    assert report.model_writer == "w-model"
    assert report.model_dialogue == "d-model"
    assert report.model_proof == "p-model"
    assert report.requirement == REQUIREMENT


def test_run_report_to_dict_round_trips_through_json() -> None:
    _script, report = run_pipeline_with_report(REQUIREMENT, verbose=False)
    row = report.to_dict()
    # Must survive a real json.dumps -- this is exactly the shape
    # scripts/eval_generation.py appends to its JSONL log.
    reloaded = json.loads(json.dumps(row, ensure_ascii=False))
    assert reloaded["requirement"] == REQUIREMENT
    assert reloaded["coerced_from"] == report.coerced_from


if __name__ == "__main__":
    test_pipeline_produces_valid_script()
    test_pipeline_is_resumable_across_runs()
    test_parse_script_json_from_prose_wrapped_output()
    test_parse_script_json_ignores_json_schema()
    test_parse_script_json_returns_none_for_no_json()
    test_coerce_script_prefers_pydantic()
    test_coerce_script_falls_back_to_json_dict()
    test_coerce_script_falls_back_to_raw()
    test_coerce_script_returns_none_when_nothing_salvageable()
    test_validate_references_reports_dangling_ids()
    test_model_choice_for_role_maps_each_role()
    test_model_choice_for_role_raises_on_unknown_role()
    test_run_pipeline_with_report_reflects_model_override()
    test_run_report_to_dict_round_trips_through_json()
    print("All tests passed.")
