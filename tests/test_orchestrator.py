"""Unit tests for crew/orchestrator.py (BiXiaScribe 重構 Phase 3): checkpoint
I/O, stage detection, and resumability after a crash/corruption. Uses
hand-written StageRunners stand-ins (no crewai/LLM involved at all), so this
suite needs no LLM_BACKEND/API key and runs instantly -- mirrors
test_crew_tools.py's approach of passing plain callables instead of
monkeypatching module globals.
"""
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402
from bixiascribe.crew import orchestrator  # noqa: E402
from bixiascribe.crew.orchestrator import (  # noqa: E402
    StageRunners,
    detect_stage,
    dispatch_next,
    run_layered,
)
from bixiascribe.schema import (  # noqa: E402
    NPC,
    Beat,
    BeatSheet,
    ChapterOutline,
    Event,
    ExtractionResult,
    Outline,
    Variable,
)

REQUIREMENT = "test requirement"


@contextmanager
def _isolated_state_dir():
    """Point config.BIXIA_STATE_DIR at a throwaway tempdir for the duration
    of one test, then restore it -- orchestrator.state_dir() reads
    config.BIXIA_STATE_DIR live (not a value bound at import time), so this
    reassignment is picked up immediately, same pattern test_generation.py
    uses for config.LLM_BACKEND."""
    original = config.BIXIA_STATE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.BIXIA_STATE_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            config.BIXIA_STATE_DIR = original


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        npcs=[NPC(id="npc-1", name="A", identity="x", personality="y", speech_style="z")],
        variables=[Variable(id="v1", name="v", initial=0)],
    )


def _beat_sheet(n: int) -> BeatSheet:
    outline = Outline(
        title="t", premise="p", chapters=[ChapterOutline(id="ch-1", title="c", summary="s")]
    )
    beats = [Beat(id=f"beat-{i}", chapter_id="ch-1", summary=f"s{i}") for i in range(n)]
    return BeatSheet(outline=outline, beats=beats)


def _event_for(beat: Beat) -> Event:
    return Event(
        id=beat.id,
        title=beat.summary,
        location="",
        summary=beat.summary,
        dialogue=[{"npc_id": "npc-1", "line": "..."}],
    )


class CountingRunners:
    """Hand-written StageRunners backing implementation that counts calls
    per stage/beat and can simulate a scene failing N times before it
    succeeds -- stands in for a real crew call without any LLM/network."""

    def __init__(self, n_beats: int = 2, fail_scene_id: str | None = None, fail_times: int = 0):
        self.extract_calls = 0
        self.expand_calls = 0
        self.scene_calls: dict[str, int] = {}
        self.n_beats = n_beats
        self.fail_scene_id = fail_scene_id
        self.fail_times = fail_times
        self._fail_count = 0

    def extract(self, requirement, models, verbose):
        self.extract_calls += 1
        return _extraction(), "pydantic"

    def expand_beats(self, requirement, extraction, models, verbose):
        self.expand_calls += 1
        return _beat_sheet(self.n_beats), "pydantic"

    def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
        self.scene_calls[beat.id] = self.scene_calls.get(beat.id, 0) + 1
        if beat.id == self.fail_scene_id and self._fail_count < self.fail_times:
            self._fail_count += 1
            raise RuntimeError("simulated scene failure")
        return _event_for(beat), "pydantic"

    def as_stage_runners(self) -> StageRunners:
        return StageRunners(
            extract=self.extract, expand_beats=self.expand_beats, write_scene=self.write_scene
        )


def test_detect_stage_missing_run_is_extract() -> None:
    with _isolated_state_dir():
        assert detect_stage("no-such-run") == "extract"


def test_checkpoint_envelope_has_schema_version_and_round_trips() -> None:
    with _isolated_state_dir():
        extraction = _extraction()
        path = orchestrator._extraction_path("run-envelope")
        orchestrator.save_checkpoint(path, extraction)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == 1
        assert raw["data"]["npcs"][0]["id"] == "npc-1"

        reloaded = orchestrator.load_checkpoint(path, ExtractionResult)
        assert reloaded == extraction


def test_load_checkpoint_rejects_unknown_schema_version() -> None:
    with _isolated_state_dir():
        run_id = "run-badver"
        path = orchestrator._extraction_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 999, "data": {}}), encoding="utf-8")

        assert orchestrator.load_checkpoint(path, ExtractionResult) is None
        assert detect_stage(run_id) == "extract"


def test_dispatch_next_processes_one_scene_per_call() -> None:
    with _isolated_state_dir():
        runners = CountingRunners(n_beats=3)
        run_id = "run-granular"

        dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # extract
        dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beats
        assert detect_stage(run_id) == "scenes"

        dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # 1 scene only
        assert sum(runners.scene_calls.values()) == 1
        assert detect_stage(run_id) == "scenes"  # 2 more beats still missing


def test_run_layered_calls_extract_and_beats_exactly_once() -> None:
    with _isolated_state_dir():
        runners = CountingRunners(n_beats=3)
        script, report = run_layered(
            REQUIREMENT, run_id="run-count", runners=runners.as_stage_runners(), verbose=False
        )
        assert runners.extract_calls == 1
        assert runners.expand_calls == 1
        assert sum(runners.scene_calls.values()) == 3
        assert len(script.events) == 3
        assert report.mode == "layered"
        assert report.scenes_generated == 3


def test_resume_after_scene_crash_does_not_restart_from_extract() -> None:
    with _isolated_state_dir():
        run_id = "run-crash"
        runners = CountingRunners(n_beats=2, fail_scene_id="beat-1", fail_times=1)

        try:
            run_layered(
                REQUIREMENT, run_id=run_id, runners=runners.as_stage_runners(), verbose=False
            )
        except Exception:
            pass  # simulated crash -- the point is what's left on disk afterward

        # extraction.json and beats.json survived; only beat-1's scene is missing.
        assert detect_stage(run_id) == "scenes"
        assert runners.extract_calls == 1
        assert runners.expand_calls == 1
        assert runners.scene_calls.get("beat-0") == 1
        assert runners.scene_calls.get("beat-1") == 1  # the failed attempt

        # Resume with the same (now non-failing) runners -- extract/beats must
        # not be called again, and beat-0's already-saved scene must not be
        # regenerated either.
        script, _report = run_layered(
            REQUIREMENT, run_id=run_id, runners=runners.as_stage_runners(), verbose=False
        )
        assert detect_stage(run_id) == "done"
        assert runners.extract_calls == 1
        assert runners.expand_calls == 1
        assert runners.scene_calls == {"beat-0": 1, "beat-1": 2}
        assert len(script.events) == 2


def test_corrupted_scene_checkpoint_only_regenerates_that_scene() -> None:
    with _isolated_state_dir():
        run_id = "run-corrupt"
        runners = CountingRunners(n_beats=2)
        run_layered(REQUIREMENT, run_id=run_id, runners=runners.as_stage_runners(), verbose=False)
        assert runners.scene_calls == {"beat-0": 1, "beat-1": 1}

        # Corrupt one scene checkpoint; remove the final script.json so
        # detect_stage() doesn't short-circuit to "done" on the stale script.
        orchestrator._scene_path(run_id, "beat-0").write_text("not valid json{{{", encoding="utf-8")
        orchestrator._script_path(run_id).unlink()

        assert detect_stage(run_id) == "scenes"

        run_layered(REQUIREMENT, run_id=run_id, runners=runners.as_stage_runners(), verbose=False)
        assert detect_stage(run_id) == "done"
        # extract/beats and the *other*, still-valid scene are not redone --
        # only the corrupted one gets a second call.
        assert runners.extract_calls == 1
        assert runners.expand_calls == 1
        assert runners.scene_calls == {"beat-0": 2, "beat-1": 1}


if __name__ == "__main__":
    test_detect_stage_missing_run_is_extract()
    test_checkpoint_envelope_has_schema_version_and_round_trips()
    test_load_checkpoint_rejects_unknown_schema_version()
    test_dispatch_next_processes_one_scene_per_call()
    test_run_layered_calls_extract_and_beats_exactly_once()
    test_resume_after_scene_crash_does_not_restart_from_extract()
    test_corrupted_scene_checkpoint_only_regenerates_that_scene()
    print("All tests passed.")
