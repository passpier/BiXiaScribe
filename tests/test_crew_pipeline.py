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

from bixiascribe.crew.pipeline import run_pipeline  # noqa: E402
from bixiascribe.schema import Script, validate_references  # noqa: E402

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


if __name__ == "__main__":
    test_pipeline_produces_valid_script()
    test_pipeline_is_resumable_across_runs()
    print("All tests passed.")
