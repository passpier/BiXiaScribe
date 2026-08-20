"""Integration test for crew/tasks.py's guardrail wiring itself (not the
pure check functions -- see test_guardrails.py for those): does
Task(guardrail=...) actually get attached under the conditions it's
supposed to, and skipped under the ones it isn't.

Runs against LLM_BACKEND=fake at import time (repo convention, see
test_script_length.py) but flips config.LLM_BACKEND/config.GUARDRAILS_ENABLED
per-test via monkeypatch + try/finally, since this is exactly the knob being
tested. Never executes a task (no real LLM call, no network) -- only
constructs Task objects and inspects task.guardrail.
"""
import os
import sys
from pathlib import Path

os.environ["LLM_BACKEND"] = "fake"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402

config.LLM_BACKEND = "fake"

from bixiascribe.crew.agents import make_writer_agent  # noqa: E402
from bixiascribe.crew.context_builder import build_session_document  # noqa: E402
from bixiascribe.crew.tasks import (  # noqa: E402
    make_beat_expand_task,
    make_extract_task,
    make_scene_write_task,
    make_writer_task,
)
from bixiascribe.schema import Beat, ExtractionResult  # noqa: E402

_AGENT = make_writer_agent()
_EXTRACTION = ExtractionResult(npcs=[], variables=[])
_BEAT = Beat(id="b1", chapter_id="c1", summary="s", npc_ids=[], causal_deps=[])
_SESSION = build_session_document(_BEAT, _EXTRACTION, [])


def _with_backend(backend: str, guardrails_enabled: bool, fn):
    orig_backend, orig_enabled = config.LLM_BACKEND, config.GUARDRAILS_ENABLED
    config.LLM_BACKEND, config.GUARDRAILS_ENABLED = backend, guardrails_enabled
    try:
        return fn()
    finally:
        config.LLM_BACKEND, config.GUARDRAILS_ENABLED = orig_backend, orig_enabled


def test_writer_task_has_no_guardrail_under_fake_backend():
    task = _with_backend("fake", True, lambda: make_writer_task("REQ", _AGENT))
    assert task.guardrail is None


def test_writer_task_has_guardrail_under_openrouter_backend():
    task = _with_backend("openrouter", True, lambda: make_writer_task("REQ", _AGENT))
    assert task.guardrail is not None
    assert task.guardrail_max_retries == config.GUARDRAIL_MAX_RETRIES


def test_writer_task_has_no_guardrail_when_disabled_via_config():
    task = _with_backend("openrouter", False, lambda: make_writer_task("REQ", _AGENT))
    assert task.guardrail is None


def test_extract_task_has_guardrail_under_openrouter_backend():
    task = _with_backend("openrouter", True, lambda: make_extract_task("REQ", _AGENT))
    assert task.guardrail is not None


def test_scene_write_task_has_guardrail_under_openrouter_backend():
    task = _with_backend(
        "openrouter",
        True,
        lambda: make_scene_write_task(_BEAT, _EXTRACTION, _AGENT, "e1", session=_SESSION),
    )
    assert task.guardrail is not None


def test_beat_expand_task_has_no_guardrail_under_fake_backend():
    task = _with_backend(
        "fake", True, lambda: make_beat_expand_task("REQ", _EXTRACTION, _AGENT)
    )
    assert task.guardrail is None


def test_beat_expand_task_has_guardrail_under_openrouter_backend():
    task = _with_backend(
        "openrouter", True, lambda: make_beat_expand_task("REQ", _EXTRACTION, _AGENT)
    )
    assert task.guardrail is not None
    assert task.guardrail_max_retries == config.GUARDRAIL_MAX_RETRIES


def test_beat_expand_task_has_no_guardrail_when_disabled_via_config():
    task = _with_backend(
        "openrouter", False, lambda: make_beat_expand_task("REQ", _EXTRACTION, _AGENT)
    )
    assert task.guardrail is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
