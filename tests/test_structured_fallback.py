"""Unit tests for the structured-output-parse-failure fallback path added
alongside this fix (crew/execute.py, crew/tasks.py's `structured` flag on
every make_*_task() factory, pipeline.py's raw_scan_lenient coercion tier).

All offline: no real crewai Agent/LLM call. crew/execute.py::run_task() is
tested against plain stand-in callables/objects (not real Task instances),
since its contract is deliberately decoupled from any one task's shape --
see that module's docstring. The task-factory tests build real Task objects
(needs a real crewai Agent, backed by FakeLLM per repo convention -- see
test_script_length.py/test_guardrail_wiring.py) but never call
execute_sync(), only inspect the constructed Task's fields.
"""
import json
import os
import sys
from pathlib import Path

os.environ["LLM_BACKEND"] = "fake"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402

config.LLM_BACKEND = "fake"

from pydantic import BaseModel, ValidationError  # noqa: E402

from bixiascribe.crew import execute, scene_metrics  # noqa: E402
from bixiascribe.crew.agents import make_writer_agent  # noqa: E402
from bixiascribe.crew.pipeline import _coerce_model  # noqa: E402
from bixiascribe.crew.tasks import make_extract_task, make_writer_task  # noqa: E402
from bixiascribe.schema import ExtractionResult, Script  # noqa: E402

_AGENT = make_writer_agent()


# --- is_structured_parse_error ---------------------------------------------


def test_recognizes_pydantic_validation_error():
    class _M(BaseModel):
        x: int

    try:
        _M.model_validate({"x": "not-an-int-and-missing-other-stuff"})
    except ValidationError:
        pass
    try:
        _M.model_validate_json("{\n ")
    except ValidationError as exc:
        assert execute.is_structured_parse_error(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_recognizes_json_decode_error():
    try:
        json.loads("{\n ")
    except json.JSONDecodeError as exc:
        assert execute.is_structured_parse_error(exc)
    else:
        raise AssertionError("expected JSONDecodeError")


def test_does_not_recognize_unrelated_errors():
    assert not execute.is_structured_parse_error(RuntimeError("401 Unauthorized"))
    assert not execute.is_structured_parse_error(TimeoutError("timed out"))


# --- run_task ---------------------------------------------------------------


class _FakeTask:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.execute_sync_calls = 0

    def execute_sync(self, agent=None, **kwargs):
        self.execute_sync_calls += 1
        if self._exc is not None:
            raise self._exc
        return self._result


def _reset_execute_stats():
    execute.reset_stats()


def test_run_task_succeeds_on_first_try_without_fallback():
    _reset_execute_stats()
    calls = []
    structured_task = _FakeTask(result="ok")

    def build(structured: bool):
        calls.append(structured)
        return structured_task

    outcome = execute.run_task(build, agent=object())
    assert outcome.output == "ok"
    assert outcome.degraded is False
    assert calls == [True]
    assert structured_task.execute_sync_calls == 1
    assert execute.get_stats().count == 0


def test_run_task_falls_back_once_on_structured_parse_error():
    _reset_execute_stats()
    calls = []
    err = json.JSONDecodeError("bad", "{\n ", 2)
    failing_task = _FakeTask(exc=err)
    fallback_task = _FakeTask(result="freeform-ok")

    def build(structured: bool):
        calls.append(structured)
        return failing_task if structured else fallback_task

    outcome = execute.run_task(build, agent=object())
    assert outcome.output == "freeform-ok"
    assert outcome.degraded is True
    assert calls == [True, False]
    assert failing_task.execute_sync_calls == 1
    assert fallback_task.execute_sync_calls == 1
    assert execute.get_stats().count == 1
    assert "自由文字模式" in execute.get_stats().notes[0]


def test_run_task_reraises_non_structured_errors_without_a_second_call():
    _reset_execute_stats()
    calls = []
    failing_task = _FakeTask(exc=RuntimeError("401 Unauthorized"))

    def build(structured: bool):
        calls.append(structured)
        return failing_task

    try:
        execute.run_task(build, agent=object())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError to propagate")
    assert calls == [True]
    assert failing_task.execute_sync_calls == 1
    assert execute.get_stats().count == 0


def test_run_task_respects_structured_output_off():
    _reset_execute_stats()
    orig = config.STRUCTURED_OUTPUT
    config.STRUCTURED_OUTPUT = "off"
    try:
        calls = []
        freeform_task = _FakeTask(result="freeform-only")

        def build(structured: bool):
            calls.append(structured)
            return freeform_task

        outcome = execute.run_task(build, agent=object())
        assert outcome.output == "freeform-only"
        assert outcome.degraded is False
        assert calls == [False]
        assert execute.get_stats().count == 0
    finally:
        config.STRUCTURED_OUTPUT = orig


# --- per-scene attribution (crew/scene_metrics.py) --------------------------


def test_run_task_records_call_elapsed_against_active_scene():
    scene_metrics.reset_stats()
    _reset_execute_stats()
    structured_task = _FakeTask(result="ok")

    with scene_metrics.scene_scope("bt-a"):
        execute.run_task(lambda structured: structured_task, agent=object())

    rows = scene_metrics.get_stats().as_rows()
    assert len(rows) == 1
    assert rows[0]["beat_id"] == "bt-a"
    assert rows[0]["call_elapsed_s"] >= 0
    assert rows[0]["structured_fallbacks"] == 0


def test_run_task_records_structured_fallback_against_active_scene():
    scene_metrics.reset_stats()
    _reset_execute_stats()
    err = json.JSONDecodeError("bad", "{\n ", 2)
    failing_task = _FakeTask(exc=err)
    fallback_task = _FakeTask(result="freeform-ok")

    def build(structured: bool):
        return failing_task if structured else fallback_task

    with scene_metrics.scene_scope("bt-fallback"):
        execute.run_task(build, agent=object())

    rows = scene_metrics.get_stats().as_rows()
    assert len(rows) == 1
    assert rows[0]["beat_id"] == "bt-fallback"
    assert rows[0]["structured_fallbacks"] == 1
    # execute.py's own module-level FallbackStats (run-wide) is unaffected
    # by this per-scene bookkeeping -- both are updated from the same call.
    assert execute.get_stats().count == 1


def test_run_task_call_elapsed_is_noop_outside_any_scope():
    scene_metrics.reset_stats()
    _reset_execute_stats()
    structured_task = _FakeTask(result="ok")
    execute.run_task(lambda structured: structured_task, agent=object())
    assert scene_metrics.get_stats().as_rows() == []


# --- make_*_task(structured=...) --------------------------------------------


def test_writer_task_structured_true_is_unchanged_from_baseline():
    task = make_writer_task("REQ", _AGENT)
    assert task.output_pydantic is not None
    assert "JSON Schema" not in task.expected_output


def test_writer_task_structured_false_has_no_output_pydantic_and_inlines_schema():
    task = make_writer_task("REQ", _AGENT, structured=False)
    assert task.output_pydantic is None
    assert "JSON Schema" in task.expected_output
    assert "只輸出一個 JSON 物件" in task.expected_output
    # description (the "what to produce" half of the prompt) is untouched --
    # only expected_output carries the schema-spelled-out addition.
    assert "根據以下使用者劇情需求" in task.description


def test_extract_task_structured_false_has_no_output_pydantic():
    task = make_extract_task("REQ", _AGENT, structured=False)
    assert task.output_pydantic is None
    assert "JSON Schema" in task.expected_output


# --- pipeline._coerce_model's raw_scan_lenient tier -------------------------


def _fake_output(raw: str):
    class _Output:
        pydantic = None
        json_dict = None

    out = _Output()
    out.raw = raw
    return out


def test_coerce_model_raw_scan_lenient_salvages_missing_required_field():
    # Script.meta has no default (strict pydantic-required), so a flat
    # object missing it fails the plain raw_scan tier and needs the
    # lenient-mirror tier to salvage anything at all. Deliberately no nested
    # dict here -- parse_model_json's "keep the last validating match" scan
    # would otherwise pick a nested object over this top-level one, since
    # every dict trivially validates against an all-defaulted lenient
    # mirror (not something this test is meant to exercise).
    raw = json.dumps({"npcs": []})
    result, source = _coerce_model(_fake_output(raw), Script)
    assert source == "raw_scan_lenient"
    assert result is not None
    assert result.meta.title == ""


def test_coerce_model_returns_none_for_unparseable_text():
    result, source = _coerce_model(_fake_output("not json at all"), ExtractionResult)
    assert result is None
    assert source is None


if __name__ == "__main__":
    import inspect

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("all tests passed")
