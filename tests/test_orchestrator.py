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

from bixiascribe import config, review  # noqa: E402
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
    Chapter,
    Event,
    ExtractionResult,
    Outline,
    Script,
    Variable,
)

REQUIREMENT = "test requirement"


@contextmanager
def _isolated_fake_llm_backend():
    """Force config.LLM_BACKEND="fake" for the duration of one test. Needed
    only by tests that exercise a real code path make_proofreader_agent()
    is on (the repair loop) -- unlike every other test in this file, those
    don't fully stub out agent construction via StageRunners, so without
    this they'd build a real crewai LLM against whatever LLM_BACKEND/API
    key happen to be configured in the ambient environment (a real
    OPENROUTER_API_KEY in .env would mean a real, paid call from a unit
    test). config.LLM_BACKEND is read at call time by llm.py::build_llm(),
    so this reassignment takes effect immediately, same pattern as
    _isolated_state_dir() above for config.BIXIA_STATE_DIR."""
    original = config.LLM_BACKEND
    config.LLM_BACKEND = "fake"
    try:
        yield
    finally:
        config.LLM_BACKEND = original


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
        title="t", premise="p", chapters=[Chapter(id="ch-1", title="c", summary="s")]
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
        assert raw["schema_version"] == orchestrator._SCHEMA_VERSION
        assert raw["data"]["npcs"][0]["id"] == "npc-1"

        reloaded = orchestrator.load_checkpoint(path, ExtractionResult)
        assert reloaded == extraction


def test_saved_script_checkpoint_round_trips_through_review_load_script() -> None:
    # Pins the envelope contract src/bixiascribe/review.py::load_script()
    # relies on to read a finished layered run's script.json directly out of
    # .bixia_state/ (see CLAUDE.md "Review UI + generation-from-UI") without
    # importing this module -- if save_checkpoint()'s envelope shape ever
    # changes, this test (not just review.py's own envelope-unwrap test)
    # should catch it.
    with _isolated_state_dir():
        script = Script(
            title="檢查點劇本",
            premise="p",
            npcs=[NPC(id="npc-1", name="A", identity="x", personality="y", speech_style="z")],
            events=[Event(id="e1", title="t", location="l", summary="s")],
        )
        path = orchestrator._script_path("run-script-envelope")
        orchestrator.save_checkpoint(path, script)

        reloaded = review.load_script(path)
        assert reloaded.title == "檢查點劇本"
        assert reloaded == script


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


def _chained_beat_sheet(n: int) -> BeatSheet:
    """Like _beat_sheet(), but beat-i causally depends on beat-(i-1), so
    every scene after the first has real prior-scene context to see."""
    outline = Outline(
        title="t", premise="p", chapters=[Chapter(id="ch-1", title="c", summary="s")]
    )
    beats = [
        Beat(
            id=f"beat-{i}",
            chapter_id="ch-1",
            summary=f"summary of beat {i}",
            causal_deps=[f"beat-{i - 1}"] if i > 0 else [],
        )
        for i in range(n)
    ]
    return BeatSheet(outline=outline, beats=beats)


class RecordingRunners:
    """Records the SessionDocument each write_scene call was given, so
    session_doc_max_tokens threading (Phase 5) can be asserted on directly
    instead of inferred from Event content."""

    def __init__(self, n_beats: int = 3):
        self.n_beats = n_beats
        self.sessions: dict[str, object] = {}

    def extract(self, requirement, models, verbose):
        return _extraction(), "pydantic"

    def expand_beats(self, requirement, extraction, models, verbose):
        return _chained_beat_sheet(self.n_beats), "pydantic"

    def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
        self.sessions[beat.id] = session
        return _event_for(beat), "pydantic"

    def as_stage_runners(self) -> StageRunners:
        return StageRunners(
            extract=self.extract, expand_beats=self.expand_beats, write_scene=self.write_scene
        )


def test_session_doc_max_tokens_threads_through_run_layered() -> None:
    with _isolated_state_dir():
        runners = RecordingRunners(n_beats=3)
        run_layered(
            REQUIREMENT,
            run_id="run-sdmt-tight",
            runners=runners.as_stage_runners(),
            session_doc_max_tokens=1,
        )
        # A budget of 1 token can't fit any summary alongside the character
        # cards/current_beat, so the last scene (which has 2 prior scenes to
        # see) must have dropped every one of them.
        last_session = runners.sessions["beat-2"]
        assert last_session.scene_summaries == []
        assert last_session.omitted_scene_count == 2

    with _isolated_state_dir():
        runners = RecordingRunners(n_beats=3)
        run_layered(
            REQUIREMENT,
            run_id="run-sdmt-unlimited",
            runners=runners.as_stage_runners(),
            session_doc_max_tokens=0,
        )
        last_session = runners.sessions["beat-2"]
        assert len(last_session.scene_summaries) == 2
        assert last_session.omitted_scene_count == 0


def test_two_tuple_runners_still_supported() -> None:
    """CountingRunners (used throughout this file) returns 2-tuples with no
    usage element -- _unpack_stage_result() must accept that unchanged and
    report zero usage rather than crashing."""
    with _isolated_state_dir():
        runners = CountingRunners(n_beats=2)
        _script, report = run_layered(
            REQUIREMENT, run_id="run-2tuple", runners=runners.as_stage_runners()
        )
        assert report.token_usage == {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cache_creation_tokens": 0,
            "successful_requests": 0,
        }


class UsageReportingRunners:
    """Every stage runner returns a 3-tuple with a fixed usage dict -- lets
    tests assert run_layered() actually accumulates the third element."""

    def __init__(self, n_beats: int = 2, per_call_tokens: int = 10):
        self.n_beats = n_beats
        self.per_call_tokens = per_call_tokens
        self.calls = 0

    def _usage(self):
        self.calls += 1
        return {
            "total_tokens": self.per_call_tokens,
            "prompt_tokens": self.per_call_tokens - 3,
            "cached_prompt_tokens": 0,
            "completion_tokens": 3,
            "reasoning_tokens": 0,
            "cache_creation_tokens": 0,
            "successful_requests": 1,
        }

    def extract(self, requirement, models, verbose):
        return _extraction(), "pydantic", self._usage()

    def expand_beats(self, requirement, extraction, models, verbose):
        return _beat_sheet(self.n_beats), "pydantic", self._usage()

    def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
        return _event_for(beat), "pydantic", self._usage()

    def as_stage_runners(self) -> StageRunners:
        return StageRunners(
            extract=self.extract, expand_beats=self.expand_beats, write_scene=self.write_scene
        )


def test_token_usage_accumulates_from_three_tuple_runners() -> None:
    with _isolated_state_dir():
        runners = UsageReportingRunners(n_beats=3, per_call_tokens=10)
        _script, report = run_layered(
            REQUIREMENT, run_id="run-usage", runners=runners.as_stage_runners()
        )
        # 1 extract + 1 beats + 3 scenes = 5 calls x 10 tokens each.
        assert runners.calls == 5
        assert report.token_usage["total_tokens"] == 50
        assert report.token_usage["successful_requests"] == 5


def test_partial_token_usage_survives_pipeline_error() -> None:
    """A scene_writer that always produces a dangling next_event_id makes
    the repair loop exhaust and run_layered() raise PipelineError -- the
    usage spent up to that point (including the failed repair attempts)
    must still be on exc.report."""

    class DanglingRunners(UsageReportingRunners):
        def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
            event = _event_for(beat).model_copy(
                update={
                    "branches": [
                        {
                            "id": f"br-{beat.id}",
                            "choice_text": "x",
                            "next_event_id": "no-such-event",
                        }
                    ]
                }
            )
            return event, "pydantic", self._usage()

    with _isolated_state_dir(), _isolated_fake_llm_backend():
        runners = DanglingRunners(n_beats=1, per_call_tokens=7)
        try:
            run_layered(REQUIREMENT, run_id="run-dangling", runners=runners.as_stage_runners())
            raise AssertionError("expected PipelineError")
        except orchestrator.PipelineError as exc:
            assert exc.report is not None
            assert exc.report.token_usage["total_tokens"] > 0


def test_unhandled_dispatch_crash_becomes_pipeline_error_with_report() -> None:
    """A raw (non-PipelineError) exception out of a stage runner -- standing
    in for the real-world case CLAUDE.md's Known limitations documents,
    where some OpenRouter endpoints (Decart, GMICloud) return a response
    with choices=None and trip a provider-library exception before any of
    orchestrator.py's own error handling runs -- must surface as a
    PipelineError with a report attached, not propagate as a bare
    RuntimeError and take the whole process down with it. This is what lets
    scripts/eval_generation.py's `except PipelineError` (via
    generation.py's `except PipelineError as exc:`) log ok=False instead of
    crashing the whole matrix run."""
    with _isolated_state_dir():
        run_id = "run-unhandled-crash"
        # fail_times > n's actual retry count in run_layered()'s own loop
        # (there is none for a raw exception -- unlike the coercion-failure
        # retry CountingRunners is normally used for) so this always raises.
        runners = CountingRunners(n_beats=1, fail_scene_id="beat-0", fail_times=999)

        try:
            run_layered(
                REQUIREMENT, run_id=run_id, runners=runners.as_stage_runners(), verbose=False
            )
            raise AssertionError("expected PipelineError")
        except orchestrator.PipelineError as exc:
            assert exc.report is not None
            assert "RuntimeError" in str(exc)
        except RuntimeError:
            raise AssertionError(
                "raw RuntimeError leaked out of run_layered() instead of being "
                "converted to PipelineError"
            ) from None


if __name__ == "__main__":
    test_detect_stage_missing_run_is_extract()
    test_checkpoint_envelope_has_schema_version_and_round_trips()
    test_load_checkpoint_rejects_unknown_schema_version()
    test_dispatch_next_processes_one_scene_per_call()
    test_run_layered_calls_extract_and_beats_exactly_once()
    test_resume_after_scene_crash_does_not_restart_from_extract()
    test_corrupted_scene_checkpoint_only_regenerates_that_scene()
    test_session_doc_max_tokens_threads_through_run_layered()
    test_two_tuple_runners_still_supported()
    test_token_usage_accumulates_from_three_tuple_runners()
    test_partial_token_usage_survives_pipeline_error()
    test_unhandled_dispatch_crash_becomes_pipeline_error_with_report()
    print("All tests passed.")
