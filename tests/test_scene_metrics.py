"""Unit tests for crew/scene_metrics.py -- the per-scene execution
attribution accumulator (see that module's docstring and
openspec/changes/profile-layered-pipeline-cost/design.md for why it
exists). All offline, no crewai/LLM involved: this module has no crewai
import at all."""
import concurrent.futures
import contextvars
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("LLM_BACKEND", "fake")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.crew import scene_metrics  # noqa: E402


def setup_function(_fn):
    scene_metrics.reset_stats()


# --- no-op outside any scope -------------------------------------------------


def test_record_call_elapsed_is_noop_outside_scope():
    scene_metrics.record_call_elapsed(1.0)
    assert scene_metrics.get_stats().as_rows() == []


def test_record_retrieval_call_is_noop_outside_scope():
    scene_metrics.record_retrieval_call()
    assert scene_metrics.get_stats().as_rows() == []


def test_record_structured_fallback_is_noop_outside_scope():
    scene_metrics.record_structured_fallback()
    assert scene_metrics.get_stats().as_rows() == []


def test_record_guardrail_retry_is_noop_with_no_beat_id_and_no_scope():
    scene_metrics.record_guardrail_retry()
    assert scene_metrics.get_stats().as_rows() == []


def test_record_usage_is_noop_with_empty_beat_id():
    scene_metrics.record_usage("", {"total_tokens": 10})
    assert scene_metrics.get_stats().as_rows() == []


def test_record_repair_elapsed_is_noop_with_empty_beat_id():
    scene_metrics.record_repair_elapsed("", 1.0)
    assert scene_metrics.get_stats().as_rows() == []


# --- scene_scope --------------------------------------------------------


def test_scene_scope_records_elapsed_time():
    with scene_metrics.scene_scope("bt-a"):
        time.sleep(0.01)
    rows = scene_metrics.get_stats().as_rows()
    assert len(rows) == 1
    assert rows[0]["beat_id"] == "bt-a"
    assert rows[0]["elapsed_s"] > 0


def test_scene_scope_records_elapsed_time_even_on_exception():
    class _Boom(Exception):
        pass

    try:
        with scene_metrics.scene_scope("bt-fail"):
            time.sleep(0.01)
            raise _Boom("kaboom")
    except _Boom:
        pass

    rows = scene_metrics.get_stats().as_rows()
    assert len(rows) == 1
    assert rows[0]["beat_id"] == "bt-fail"
    assert rows[0]["elapsed_s"] > 0


def test_scene_scope_restores_previous_thread_local_on_exit():
    with scene_metrics.scene_scope("bt-outer"):
        assert scene_metrics._active_beat_id() == "bt-outer"
        with scene_metrics.scene_scope("bt-inner"):
            assert scene_metrics._active_beat_id() == "bt-inner"
        assert scene_metrics._active_beat_id() == "bt-outer"
    assert scene_metrics._active_beat_id() is None


def test_recorders_attribute_to_active_scope():
    with scene_metrics.scene_scope("bt-a"):
        scene_metrics.record_call_elapsed(2.5)
        scene_metrics.record_retrieval_call()
        scene_metrics.record_retrieval_call()
        scene_metrics.record_structured_fallback()
        scene_metrics.record_guardrail_retry()
        scene_metrics.record_usage(
            "bt-a",
            {"successful_requests": 3, "reasoning_tokens": 100, "total_tokens": 500},
        )
        scene_metrics.record_repair_elapsed("bt-a", 1.5)

    rows = scene_metrics.get_stats().as_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["call_elapsed_s"] == 2.5
    assert row["retrieval_calls"] == 2
    assert row["structured_fallbacks"] == 1
    assert row["guardrail_retries"] == 1
    assert row["llm_calls"] == 3
    assert row["reasoning_tokens"] == 100
    assert row["total_tokens"] == 500
    assert row["repair_elapsed_s"] == 1.5


def test_record_guardrail_retry_accepts_explicit_beat_id_outside_scope():
    scene_metrics.record_guardrail_retry("bt-explicit")
    rows = scene_metrics.get_stats().as_rows()
    assert len(rows) == 1
    assert rows[0]["beat_id"] == "bt-explicit"
    assert rows[0]["guardrail_retries"] == 1


# --- concurrency ----------------------------------------------------------


def test_concurrent_scopes_on_separate_threads_attribute_independently():
    barrier = threading.Barrier(3)

    def _worker(beat_id: str, n_calls: int) -> None:
        with scene_metrics.scene_scope(beat_id):
            barrier.wait()
            for _ in range(n_calls):
                scene_metrics.record_retrieval_call()

    threads = [
        threading.Thread(target=_worker, args=("bt-1", 5)),
        threading.Thread(target=_worker, args=("bt-2", 9)),
    ]
    barrier_release = threading.Thread(target=barrier.wait)
    for t in threads + [barrier_release]:
        t.start()
    for t in threads + [barrier_release]:
        t.join()

    rows = {row["beat_id"]: row for row in scene_metrics.get_stats().as_rows()}
    assert rows["bt-1"]["retrieval_calls"] == 5
    assert rows["bt-2"]["retrieval_calls"] == 9


def test_recorder_attributes_across_crewai_style_threadpool_hop():
    """Reproduces the real mismatch this module used to have: crewai's own
    native-tool-calling loop (crewai/agents/crew_agent_executor.py) dispatches
    concurrent tool calls from one LLM turn via
    `ThreadPoolExecutor.submit(contextvars.copy_context().run, fn, ...)` --
    verified against a real run (.bixia_state/1787309292-req-d232acf2d8): the
    run-level retrieval_calls was 3 for the one scene generated, but that
    scene's sidecar recorded 0. A plain threading.local() (the original
    implementation) is never populated on the pool's worker thread even
    though copy_context().run() is used to submit the call -- only
    contextvars.ContextVar state crosses that hop. This test fails against a
    threading.local()-based _current and passes against the
    contextvars.ContextVar-based one."""
    scene_metrics.reset_stats()
    with scene_metrics.scene_scope("bt-tool"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                contextvars.copy_context().run, scene_metrics.record_retrieval_call
            )
            future.result()

    rows = scene_metrics.get_stats().as_rows()
    assert len(rows) == 1
    assert rows[0]["beat_id"] == "bt-tool"
    assert rows[0]["retrieval_calls"] == 1


# --- as_rows / reset_stats --------------------------------------------------


def test_as_rows_preserves_first_seen_order():
    with scene_metrics.scene_scope("bt-second"):
        pass
    with scene_metrics.scene_scope("bt-first"):
        pass
    with scene_metrics.scene_scope("bt-second"):
        pass  # re-entering an already-seen beat must not reorder it

    ids = [row["beat_id"] for row in scene_metrics.get_stats().as_rows()]
    assert ids == ["bt-second", "bt-first"]


def test_reset_stats_zeroes_state():
    with scene_metrics.scene_scope("bt-a"):
        scene_metrics.record_retrieval_call()
    assert scene_metrics.get_stats().as_rows() != []

    scene_metrics.reset_stats()
    assert scene_metrics.get_stats().as_rows() == []


# --- active_scenes -----------------------------------------------------


def test_active_scenes_reports_an_in_flight_scope():
    scene_metrics.reset_stats()
    assert scene_metrics.active_scenes() == {}
    with scene_metrics.scene_scope("bt-live"):
        active = scene_metrics.active_scenes()
        assert set(active) == {"bt-live"}
        assert active["bt-live"] >= 0.0
    assert scene_metrics.active_scenes() == {}


def test_active_scenes_materializes_row_on_entry():
    """dispatch_batch()'s .get(beat.id) lookup (orchestrator.py) now sees a
    zero-valued row for a scene that's in flight, not None -- pinning that
    behavior change."""
    scene_metrics.reset_stats()
    with scene_metrics.scene_scope("bt-live"):
        assert "bt-live" in scene_metrics.get_stats().scenes


def test_reset_stats_clears_active_scenes():
    """Leak guard -- _active is module-global across runs."""

    def _crash_mid_scope():
        with scene_metrics.scene_scope("bt-doomed"):
            raise RuntimeError("boom")

    try:
        _crash_mid_scope()
    except RuntimeError:
        pass
    # scene_scope's finally already pops on exit even when the block
    # raises, so this should already be empty -- reset_stats() is the
    # belt-and-suspenders guard for anything that somehow leaked.
    scene_metrics.reset_stats()
    assert scene_metrics.active_scenes() == {}
