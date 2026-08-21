"""Unit tests for crew/orchestrator.py's Phase 4 additions: plan_batches()
(pure causal-dependency batching) and dispatch_batch() (concurrent scene
generation within one batch). Mirrors tests/test_orchestrator.py's
conventions -- hand-written StageRunners stand-ins, no crewai/LLM involved,
no pytest features -- so this suite needs no LLM_BACKEND/API key and runs
instantly.

Also includes a regression test for crew/tools.py's RetrievalStats
bookkeeping under real thread concurrency -- tests/test_crew_tools.py only
ever calls WuxiaRetrievalTool._run() from one thread, so `_stats_lock`'s
correctness under concurrent writers has never actually been exercised
before this file.
"""
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402
from bixiascribe.crew import orchestrator, scene_metrics, tools  # noqa: E402
from bixiascribe.crew.orchestrator import (  # noqa: E402
    StageRunners,
    confirm_batch,
    detect_stage,
    dispatch_batch,
    pending_batch_ids,
    plan_batches,
    reject_batch,
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
    Variable,
)

REQUIREMENT = "test requirement"


@contextmanager
def _isolated_state_dir():
    """Same pattern as tests/test_orchestrator.py -- orchestrator.state_dir()
    reads config.BIXIA_STATE_DIR live, so this reassignment is picked up
    immediately."""
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


def _beat(id_: str, deps: list[str] | None = None) -> Beat:
    return Beat(id=id_, chapter_id="ch-1", summary=f"summary-{id_}", causal_deps=deps or [])


def _beat_sheet(beats: list[Beat]) -> BeatSheet:
    outline = Outline(
        title="t", premise="p", chapters=[Chapter(id="ch-1", title="c", summary="s")]
    )
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
    """Same shape as test_orchestrator.py's CountingRunners, plus recording
    which thread each write_scene call ran on and an optional per-call
    sleep, so tests can observe real concurrency."""

    def __init__(
        self,
        n_beats: int = 2,
        beats: list[Beat] | None = None,
        fail_scene_id: str | None = None,
        fail_times: int = 0,
        sleep_s: float = 0.0,
    ):
        self.extract_calls = 0
        self.expand_calls = 0
        self.scene_calls: dict[str, int] = {}
        self.thread_names: dict[str, str] = {}
        self.beats = beats if beats is not None else [_beat(f"beat-{i}") for i in range(n_beats)]
        self.fail_scene_id = fail_scene_id
        self.fail_times = fail_times
        self.sleep_s = sleep_s
        self._fail_count = 0
        self._fail_lock = threading.Lock()

    def extract(self, requirement, models, verbose):
        self.extract_calls += 1
        return _extraction(), "pydantic"

    def expand_beats(self, requirement, extraction, models, verbose):
        self.expand_calls += 1
        return _beat_sheet(self.beats), "pydantic"

    def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
        if self.sleep_s:
            time.sleep(self.sleep_s)
        with self._fail_lock:
            self.scene_calls[beat.id] = self.scene_calls.get(beat.id, 0) + 1
            self.thread_names[beat.id] = threading.current_thread().name
            should_fail = beat.id == self.fail_scene_id and self._fail_count < self.fail_times
            if should_fail:
                self._fail_count += 1
        if should_fail:
            raise RuntimeError("simulated scene failure")
        return _event_for(beat), "pydantic"

    def as_stage_runners(self) -> StageRunners:
        return StageRunners(
            extract=self.extract, expand_beats=self.expand_beats, write_scene=self.write_scene
        )


# --- plan_batches() -----------------------------------------------------


def test_plan_batches_independent_beats_form_one_batch() -> None:
    beats = [_beat("b0"), _beat("b1"), _beat("b2")]
    batches = plan_batches(beats)
    assert len(batches) == 1
    assert [b.id for b in batches[0]] == ["b0", "b1", "b2"]


def test_plan_batches_causal_chain_is_fully_serial() -> None:
    beats = [
        _beat("beat_depart"),
        _beat("beat_village", deps=["beat_depart"]),
        _beat("beat_clue", deps=["beat_village"]),
    ]
    batches = plan_batches(beats)
    assert [[b.id for b in batch] for batch in batches] == [
        ["beat_depart"],
        ["beat_village"],
        ["beat_clue"],
    ]


def test_plan_batches_diamond_shape() -> None:
    beats = [
        _beat("a"),
        _beat("b", deps=["a"]),
        _beat("c", deps=["a"]),
        _beat("d", deps=["b", "c"]),
    ]
    batches = plan_batches(beats)
    assert [sorted(b.id for b in batch) for batch in batches] == [["a"], ["b", "c"], ["d"]]


def test_plan_batches_unknown_dep_id_does_not_strand_beat() -> None:
    beats = [_beat("a", deps=["ghost-event-id"])]
    batches = plan_batches(beats)
    assert [b.id for b in batches[0]] == ["a"]


def test_plan_batches_cycle_degrades_to_one_beat_per_batch() -> None:
    beats = [_beat("a", deps=["b"]), _beat("b", deps=["a"])]
    batches = plan_batches(beats)
    # No progress possible -- every beat still comes out exactly once.
    all_ids = [b.id for batch in batches for b in batch]
    assert sorted(all_ids) == ["a", "b"]
    assert all(len(batch) == 1 for batch in batches)


# --- dispatch_batch() concurrency ----------------------------------------


def test_dispatch_batch_runs_independent_scenes_concurrently() -> None:
    with _isolated_state_dir():
        run_id = "run-concurrent"
        runners = CountingRunners(n_beats=4, sleep_s=0.1)

        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # extract
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beats
        assert detect_stage(run_id) == "scenes"

        start = time.monotonic()
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=4)
        elapsed = time.monotonic() - start

        assert sum(runners.scene_calls.values()) == 4
        assert detect_stage(run_id) == "proofread"
        # 4 calls * 0.1s serially would take >=0.4s; concurrently they
        # should finish in well under that.
        assert elapsed < 0.3
        assert len(set(runners.thread_names.values())) >= 2


def test_dispatch_batch_concurrency_one_matches_dispatch_next() -> None:
    with _isolated_state_dir():
        run_id = "run-serial-equiv"
        runners = CountingRunners(n_beats=3)

        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=1)
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=1)
        assert detect_stage(run_id) == "scenes"

        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=1)
        assert sum(runners.scene_calls.values()) == 1
        assert detect_stage(run_id) == "scenes"


# --- retrieval stats under real thread concurrency ------------------------


def test_retrieval_stats_correct_under_concurrent_tool_calls() -> None:
    tools.reset_stats()
    original_get_cached = tools._get_cached_collection
    original_retrieve = tools.retrieve

    def _fake_retrieve(query, top_k=3, collection=None):
        time.sleep(0.01)
        return []

    tools._get_cached_collection = lambda: "fake-collection"
    tools.retrieve = _fake_retrieve
    try:
        n_threads = 8
        queries_per_thread = 5

        def _worker(i: int) -> None:
            tool = tools.WuxiaRetrievalTool()
            for j in range(queries_per_thread):
                tool._run(f"query-{i}-{j}")

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = tools.get_stats()
        assert stats.calls == n_threads * queries_per_thread
        assert stats.failures == 0
        assert len(stats.queries) == n_threads * queries_per_thread
    finally:
        tools._get_cached_collection = original_get_cached
        tools.retrieve = original_retrieve
        tools.reset_stats()


# --- partial batch failure -------------------------------------------------


def test_partial_batch_failure_banks_successful_scenes() -> None:
    with _isolated_state_dir():
        run_id = "run-partial-fail"
        beats = [_beat("beat-0"), _beat("beat-1"), _beat("beat-2")]
        runners = CountingRunners(beats=beats, fail_scene_id="beat-1", fail_times=1)

        # extract, then beats
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=3)
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=3)
        assert detect_stage(run_id) == "scenes"

        try:
            dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=3)
        except RuntimeError:
            pass  # simulated crash mid-batch

        # beat-0 and beat-2 succeeded and were checkpointed despite beat-1's
        # failure; only beat-1 is missing.
        assert orchestrator._scene_path(run_id, "beat-0").exists()
        assert orchestrator._scene_path(run_id, "beat-2").exists()
        assert not orchestrator._scene_path(run_id, "beat-1").exists()
        assert detect_stage(run_id) == "scenes"

        # Resume: only the failed beat needs to be regenerated.
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=3)
        assert detect_stage(run_id) == "proofread"
        assert runners.scene_calls == {"beat-0": 1, "beat-1": 2, "beat-2": 1}


# --- parallel produces the same Script as serial ---------------------------


def test_parallel_and_serial_produce_identical_event_ordering() -> None:
    beats = [_beat("a"), _beat("b", deps=["a"]), _beat("c", deps=["a"])]

    with _isolated_state_dir():
        runners_serial = CountingRunners(beats=beats)
        script_serial, _ = run_layered(
            REQUIREMENT, run_id="run-serial", runners=runners_serial.as_stage_runners(),
            verbose=False, concurrency=1,
        )

    with _isolated_state_dir():
        runners_parallel = CountingRunners(beats=beats)
        script_parallel, _ = run_layered(
            REQUIREMENT, run_id="run-parallel", runners=runners_parallel.as_stage_runners(),
            verbose=False, concurrency=3,
        )

    assert [e.id for e in script_serial.events] == [e.id for e in script_parallel.events]
    assert [e.id for e in script_parallel.events] == ["a", "b", "c"]


# --- confirmation gate (Phase 4b) ------------------------------------------


def test_gate_reject_regenerates_batch_then_confirm_advances() -> None:
    with _isolated_state_dir():
        run_id = "run-gate"
        beats = [_beat("a"), _beat("b")]
        runners = CountingRunners(beats=beats)

        decisions = iter([False, True])  # reject the first offer, confirm the second

        def gate(pending_ids: list[str]) -> bool:
            assert sorted(pending_ids) == ["a", "b"]
            return next(decisions)

        script, _report = run_layered(
            REQUIREMENT, run_id=run_id, runners=runners.as_stage_runners(), verbose=False, gate=gate
        )
        assert detect_stage(run_id) == "done"
        assert sorted(e.id for e in script.events) == ["a", "b"]
        # a and b were each written twice: once for the rejected batch, once
        # for the confirmed one -- rejecting truly discards and regenerates.
        assert runners.scene_calls == {"a": 2, "b": 2}


def test_pending_scenes_survive_a_crash_and_are_reoffered_on_resume() -> None:
    with _isolated_state_dir():
        run_id = "run-gate-crash"
        beats = [_beat("a"), _beat("b")]
        runners = CountingRunners(beats=beats)

        class _Crash(Exception):
            pass

        def gate_that_crashes(pending_ids: list[str]) -> bool:
            raise _Crash("simulated process crash while awaiting confirmation")

        try:
            run_layered(
                REQUIREMENT,
                run_id=run_id,
                runners=runners.as_stage_runners(),
                verbose=False,
                gate=gate_that_crashes,
            )
        except _Crash:
            pass

        # The batch was staged but never confirmed/rejected -- still pending.
        assert sorted(pending_batch_ids(run_id)) == ["a", "b"]
        assert detect_stage(run_id) == "scenes"  # not "done": nothing promoted

        # Resume: the already-staged batch is re-offered, not regenerated.
        confirmed: list[str] = []

        def gate_confirms(pending_ids: list[str]) -> bool:
            confirmed.extend(pending_ids)
            return True

        script, _report = run_layered(
            REQUIREMENT,
            run_id=run_id,
            runners=runners.as_stage_runners(),
            verbose=False,
            gate=gate_confirms,
        )
        assert sorted(confirmed) == ["a", "b"]
        assert runners.scene_calls == {"a": 1, "b": 1}  # never regenerated
        assert detect_stage(run_id) == "done"
        assert sorted(e.id for e in script.events) == ["a", "b"]


def test_confirm_batch_and_reject_batch_are_noops_with_nothing_staged() -> None:
    with _isolated_state_dir():
        run_id = "run-gate-empty"
        assert pending_batch_ids(run_id) == []
        confirm_batch(run_id)  # must not raise
        reject_batch(run_id)  # must not raise
        assert detect_stage(run_id) == "extract"


# --- token usage accumulation under real thread concurrency (Phase 5) -----


class UsageCountingRunners(CountingRunners):
    """Like CountingRunners, but every write_scene call also returns a
    fixed-usage 3-tuple, with the same optional sleep -- lets the batch
    dispatch path be exercised under real ThreadPoolExecutor concurrency
    while reporting usage, mirroring how test_retrieval_stats_correct_
    under_concurrent_tool_calls above exercises crew/tools.py's
    _stats_lock."""

    def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
        event, source = super().write_scene(
            beat, extraction, models, verbose, target_event_id, session=session
        )
        return event, source, {"total_tokens": 1, "successful_requests": 1}


def test_token_usage_accumulation_is_thread_safe() -> None:
    with _isolated_state_dir():
        run_id = "run-usage-concurrent"
        n_beats = 8
        runners = UsageCountingRunners(n_beats=n_beats, sleep_s=0.005)

        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # extract
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beats
        assert detect_stage(run_id) == "scenes"

        totals: list[int] = []
        lock = threading.Lock()

        def _on_usage(usage) -> None:
            if usage is None:
                return
            with lock:
                totals.append(usage["total_tokens"])

        dispatch_batch(
            run_id,
            REQUIREMENT,
            runners=runners.as_stage_runners(),
            concurrency=n_beats,
            on_usage=_on_usage,
        )

        assert sum(runners.scene_calls.values()) == n_beats
        assert sum(totals) == n_beats  # not n_beats-1 or n_beats+1: no lost/duplicated updates


# --- per-scene attribution under real thread concurrency -------------------


class ScopedRunners(CountingRunners):
    """Like UsageCountingRunners, but its write_scene wraps the body in
    scene_metrics.scene_scope(beat.id) and makes a couple of
    WuxiaRetrievalTool._run() calls -- standing in for what the real
    crewai-backed _default_write_scene()/scene_writer agent do, so this
    exercises the actual production wiring (scene_metrics.record_
    retrieval_call() inside tools.py) under dispatch_batch()'s real
    ThreadPoolExecutor concurrency, not just a hand-rolled thread pool like
    test_retrieval_stats_correct_under_concurrent_tool_calls above."""

    def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
        with scene_metrics.scene_scope(beat.id):
            tool = tools.WuxiaRetrievalTool()
            for _ in range(3):
                tool._run(f"query for {beat.id}")
            event, source = super().write_scene(
                beat, extraction, models, verbose, target_event_id, session=session
            )
            return event, source, {"total_tokens": 1, "successful_requests": 1}


def test_scene_metrics_attributes_correctly_under_concurrent_batch_dispatch() -> None:
    original_get_cached = tools._get_cached_collection
    original_retrieve = tools.retrieve
    tools._get_cached_collection = lambda: "fake-collection"
    tools.retrieve = lambda query, top_k=3, collection=None: (time.sleep(0.005) or [])
    scene_metrics.reset_stats()
    try:
        with _isolated_state_dir():
            run_id = "run-scene-metrics-concurrent"
            n_beats = 6
            runners = ScopedRunners(n_beats=n_beats, sleep_s=0.005)

            dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # extract
            dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beats
            assert detect_stage(run_id) == "scenes"

            dispatch_batch(
                run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=n_beats
            )

            rows = {row["beat_id"]: row for row in scene_metrics.get_stats().as_rows()}
            assert set(rows) == {f"beat-{i}" for i in range(n_beats)}
            for row in rows.values():
                # Each beat's own 3 tool calls, never a sibling's -- if the
                # thread-local scope leaked across concurrently-running
                # scenes, some beat would see more or fewer than 3.
                assert row["retrieval_calls"] == 3
    finally:
        tools._get_cached_collection = original_get_cached
        tools.retrieve = original_retrieve
        scene_metrics.reset_stats()


if __name__ == "__main__":
    test_plan_batches_independent_beats_form_one_batch()
    test_plan_batches_causal_chain_is_fully_serial()
    test_plan_batches_diamond_shape()
    test_plan_batches_unknown_dep_id_does_not_strand_beat()
    test_plan_batches_cycle_degrades_to_one_beat_per_batch()
    test_dispatch_batch_runs_independent_scenes_concurrently()
    test_dispatch_batch_concurrency_one_matches_dispatch_next()
    test_retrieval_stats_correct_under_concurrent_tool_calls()
    test_partial_batch_failure_banks_successful_scenes()
    test_parallel_and_serial_produce_identical_event_ordering()
    test_gate_reject_regenerates_batch_then_confirm_advances()
    test_pending_scenes_survive_a_crash_and_are_reoffered_on_resume()
    test_confirm_batch_and_reject_batch_are_noops_with_nothing_staged()
    test_token_usage_accumulation_is_thread_safe()
    test_scene_metrics_attributes_correctly_under_concurrent_batch_dispatch()
    print("All tests passed.")
