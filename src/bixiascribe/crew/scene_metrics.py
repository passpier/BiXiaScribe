"""Per-scene execution attribution for the layered pipeline: elapsed time,
LLM call count, reasoning-token usage, guardrail retries, retrieval calls,
and structured-output fallbacks, broken down by which scene (beat id) each
happened during.

Why this exists: a real `script_length=long` layered run took 1h46m for 19
scenes, and per-scene output size only explained 27% of the elapsed-time
variance (r(size, elapsed) = 0.519, r^2 = 0.27 -- see
openspec/changes/profile-layered-pipeline-cost/design.md's 證據二). ~73% of
the time went somewhere the pipeline's existing run-wide-only accounting
(RunReport.elapsed_s/token_usage/retrieval_calls/structured_fallbacks)
cannot attribute to a specific scene or a specific mechanism (reasoning
tokens, guardrail retries, ReAct tool rounds, structured-output fallback
retries). This module is the fix: it lets downstream optimizations
(reasoning_effort, beat DAG prompt rewrites, agent loop/timeout caps) be
validated against a specific scene/mechanism instead of eyeballing a single
run-wide elapsed number.

Follows crew/execute.py's FallbackStats/crew/tools.py's RetrievalStats
convention: a module-level accumulator guarded by a threading.Lock, reset at
the start of one pipeline run (reset_stats()) and read back at the end
(get_stats()). No crewai import, same as guardrails.py/causal.py/
normalize.py.

Attribution across call sites (execute.run_task(), WuxiaRetrievalTool._run(),
the scene guardrail closure in crew/tasks.py) that don't otherwise share a
beat id parameter is done via a thread-local "current scene" scope
(scene_scope()) rather than threading a beat_id through every one of those
signatures. This is sound because crewai's ReAct loop, the guardrail
callback, and the tool's _run() all execute synchronously on the same
worker thread dispatch_batch()'s ThreadPoolExecutor submitted the scene to
-- concurrent scenes in one batch run on separate worker threads, and each
thread's threading.local() state is independent. Every record_*() helper is
a no-op when no scope is active on the calling thread, which is what keeps
the legacy pipeline (which never opens a scope) and every existing test
byte-for-byte unaffected."""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

_stats_lock = threading.Lock()

# Thread-local "which beat is this thread currently generating a scene for"
# marker -- set only inside scene_scope(), read by every record_*() helper
# below. Deliberately per-thread, not a single module global: dispatch_batch()
# runs several scene_scope() blocks concurrently on separate worker threads.
_current = threading.local()


class SceneMetric(BaseModel):
    """One scene's execution attribution. A pydantic BaseModel (not a plain
    dataclass, unlike FallbackStats/RetrievalStats) so
    crew/orchestrator.py::save_checkpoint() -- which wraps a BaseModel in the
    {schema_version, data} envelope -- can persist one of these as a sidecar
    checkpoint unchanged; see load_scene_metrics()'s docstring in
    orchestrator.py for why a resumed run needs that.

    elapsed_s covers the *entire* write_scene runner call for this beat,
    including any crewai-internal max_retry_limit retries, guardrail
    retries, and the structured-output free-text fallback. call_elapsed_s is
    only the one execute_sync() call that ultimately succeeded (whichever of
    the structured/free-text attempts execute.run_task() returns). Both are
    kept because they answer different questions: "how long did this scene
    take overall" vs. "how long does one real model call take" -- see
    design.md's Open Questions, resolved by keeping both rather than
    picking one."""

    beat_id: str = ""
    elapsed_s: float = 0.0
    call_elapsed_s: float = 0.0
    repair_elapsed_s: float = 0.0
    llm_calls: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    guardrail_retries: int = 0
    retrieval_calls: int = 0
    structured_fallbacks: int = 0


class SceneStats:
    """Module-level accumulator: every scene's SceneMetric, keyed by beat id
    so a repeated update (e.g. a causal-repair pass folding usage into a
    beat already recorded by _default_write_scene) mutates the same row
    instead of creating a duplicate."""

    def __init__(self) -> None:
        self.scenes: dict[str, SceneMetric] = {}

    def _get(self, beat_id: str) -> SceneMetric:
        metric = self.scenes.get(beat_id)
        if metric is None:
            metric = SceneMetric(beat_id=beat_id)
            self.scenes[beat_id] = metric
        return metric

    def as_rows(self) -> list[dict[str, Any]]:
        """JSON-safe rows in first-seen (insertion) order -- Python dicts
        preserve insertion order, and scenes are first recorded in the order
        they're dispatched, so this reads naturally without an explicit
        sort key."""
        return [metric.model_dump(mode="json") for metric in self.scenes.values()]


_stats = SceneStats()


def get_stats() -> SceneStats:
    return _stats


def reset_stats() -> None:
    """Zero the accumulator -- call once at the start of one run_layered()
    call, same convention as crew/tools.py::reset_stats() /
    crew/execute.py::reset_stats()."""
    global _stats
    with _stats_lock:
        _stats = SceneStats()


@contextmanager
def scene_scope(beat_id: str) -> Iterator[None]:
    """Mark the calling thread as "currently generating scene `beat_id`" for
    the duration of the block, and fold the block's wall-clock time into
    that beat's elapsed_s on exit -- including when the block raises, so a
    scene that ultimately fails is still attributed (see
    orchestrator.py::_default_write_scene, which raises PipelineError on a
    coercion failure after this scope has already recorded elapsed_s).

    Nested/re-entrant use for the same beat_id on the same thread is
    supported (e.g. a causal-repair pass calling record_repair_elapsed()
    from inside a still-open outer scope would be unusual but harmless);
    scoping a *different* beat_id while one is already active on this
    thread is not a case any current caller hits (each worker thread
    generates one scene start-to-finish) and simply overwrites the
    thread-local for the duration, restoring the previous value on exit."""
    previous = getattr(_current, "beat_id", None)
    _current.beat_id = beat_id
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        with _stats_lock:
            metric = _stats._get(beat_id)
            metric.elapsed_s += elapsed
        _current.beat_id = previous


def _active_beat_id() -> str | None:
    return getattr(_current, "beat_id", None)


def record_call_elapsed(sec: float) -> None:
    """Record one execute_sync() call's wall-clock time against the
    currently-active scene (crew/execute.py::run_task()). A no-op outside
    any scene_scope() -- the legacy pipeline calls execute.run_task() too,
    via its own task execution path, but never opens a scene scope."""
    beat_id = _active_beat_id()
    if beat_id is None:
        return
    with _stats_lock:
        _stats._get(beat_id).call_elapsed_s += sec


def record_repair_elapsed(beat_id: str, sec: float) -> None:
    """Record a causal-repair pass's wall-clock time against `beat_id`
    explicitly (not the thread-local scope) -- _default_repair_scene() is
    called from _validate_scene(), which may run outside the write_scene
    runner's own scene_scope() depending on caller, so it always knows and
    passes its own target beat_id rather than relying on ambient state."""
    if not beat_id:
        return
    with _stats_lock:
        _stats._get(beat_id).repair_elapsed_s += sec


def record_usage(beat_id: str, usage: dict[str, int] | None) -> None:
    """Fold a per-call token-usage delta (crew/pipeline.py::_usage_delta()'s
    output, keyed by _USAGE_FIELDS) into `beat_id`'s SceneMetric.
    successful_requests maps to llm_calls -- crewai's BaseLLM tracks that
    field per LLM instance already (see llms/base_llm.py's
    _track_token_usage_internal), so no separate ReAct-round counter is
    needed (see design.md's Open Questions)."""
    if not beat_id or not usage:
        return
    with _stats_lock:
        metric = _stats._get(beat_id)
        metric.llm_calls += int(usage.get("successful_requests", 0) or 0)
        metric.reasoning_tokens += int(usage.get("reasoning_tokens", 0) or 0)
        metric.total_tokens += int(usage.get("total_tokens", 0) or 0)


def record_guardrail_retry(beat_id: str | None = None) -> None:
    """Record one guardrail rejection against `beat_id`, or the
    thread-locally active scene if beat_id is omitted (crew/tasks.py's
    _scene_guardrail closure has `beat` in scope directly, so it always
    passes it explicitly; other callers may rely on ambient scope). A no-op
    if neither is available."""
    target = beat_id or _active_beat_id()
    if not target:
        return
    with _stats_lock:
        _stats._get(target).guardrail_retries += 1


def record_retrieval_call() -> None:
    """Record one wuxia_corpus_search call against the currently-active
    scene (crew/tools.py::WuxiaRetrievalTool._run()). A no-op outside any
    scene_scope() -- the legacy pipeline's 對話 agent also calls this tool,
    but never opens a scene scope, so its calls are correctly left
    unattributed here (crew/pipeline.py's own RetrievalStats already counts
    them at the run level)."""
    beat_id = _active_beat_id()
    if beat_id is None:
        return
    with _stats_lock:
        _stats._get(beat_id).retrieval_calls += 1


def record_structured_fallback() -> None:
    """Record one structured-output-parse-failure fallback (crew/execute.py
    ::run_task()) against the currently-active scene. A no-op outside any
    scene_scope() -- same rationale as record_retrieval_call()."""
    beat_id = _active_beat_id()
    if beat_id is None:
        return
    with _stats_lock:
        _stats._get(beat_id).structured_fallbacks += 1
