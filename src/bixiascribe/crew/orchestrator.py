"""Stateful orchestration for the layered pipeline (BiXiaScribe 重構 Phase 3).

Persists one checkpoint file per completed stage/scene under
config.BIXIA_STATE_DIR/<run_id>/, so a crashed or interrupted
run_layered() can resume without re-calling the LLM for whatever already
succeeded -- the biggest cost saving relative to the legacy pipeline, where
a failure in the 對話 stage means re-running the whole 3-Task crew including
the already-successful 編劇 stage.

Checkpoint granularity is per-stage for extract/beats, and per-scene for
scenes -- see dispatch_next()'s docstring for why. This mirrors
crew/pipeline.py's run_layered_pipeline() (Phase 2) stage-by-stage
structure; the difference here is that each stage is its own
resumable, independently-retryable unit backed by a file on disk instead
of an in-memory local variable.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .. import config, review
from ..llm import ModelChoice
from ..schema import (
    Beat,
    BeatSheet,
    CausalPlotGraph,
    Event,
    ExtractionResult,
    Script,
    SessionDocument,
    validate_causal_graph,
    validate_references,
)
from . import causal
from .agents import (
    make_beat_expander_agent,
    make_extractor_agent,
    make_proofreader_agent,
    make_scene_writer_agent,
)
from .context_builder import build_session_document
from .pipeline import (
    MAX_REPAIR_ATTEMPTS,
    PipelineError,
    RunReport,
    StepEvent,
    _coerce_model,
    _repair,
)
from .tasks import make_beat_expand_task, make_extract_task, make_scene_write_task
from .tools import get_stats, reset_stats

M = TypeVar("M", bound=BaseModel)

Stage = Literal["extract", "beats", "scenes", "proofread", "done"]

# Envelope version for checkpoint files -- {"schema_version": N, "data": {...}}.
# Bump this if a checkpointed model's shape changes in a way old checkpoints
# can't be read as; load_checkpoint() treats a version mismatch as "no
# checkpoint" (half-finished/incompatible), not a crash.
_SCHEMA_VERSION = 1


class PipelineState(BaseModel):
    run_id: str
    requirement: str
    stage: Stage = "extract"
    completed_scene_ids: list[str] = Field(default_factory=list)
    last_updated: float = 0.0


# --- Checkpoint I/O --------------------------------------------------------


def state_dir(run_id: str) -> Path:
    return config.BIXIA_STATE_DIR / run_id


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write `payload` to `path` without ever leaving a half-written file on
    disk for a concurrent detect_stage()/load_checkpoint() call to trip
    over: write to a sibling .tmp file, then os.replace() it into place
    (atomic on POSIX and Windows for same-filesystem renames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def save_checkpoint(path: Path, model: BaseModel) -> None:
    """Persist a pydantic model as a checkpoint file, wrapped in a
    {"schema_version", "data"} envelope (see _SCHEMA_VERSION)."""
    payload = {"schema_version": _SCHEMA_VERSION, "data": model.model_dump(mode="json")}
    _atomic_write_json(path, payload)


def load_checkpoint(path: Path, model_cls: type[M]) -> M | None:
    """Load and validate a checkpoint file as `model_cls`. Returns None if
    the file doesn't exist, isn't valid JSON, has a schema_version this
    process doesn't recognize, or doesn't validate as `model_cls` -- all of
    which are treated identically by detect_stage()/dispatch_next(): as "this
    stage/scene isn't done yet", not as an error to propagate."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    try:
        return model_cls.model_validate(data)
    except ValidationError:
        return None


def _state_path(run_id: str) -> Path:
    return state_dir(run_id) / "state.json"


def _extraction_path(run_id: str) -> Path:
    return state_dir(run_id) / "extraction.json"


def _beats_path(run_id: str) -> Path:
    return state_dir(run_id) / "beats.json"


def _scene_path(run_id: str, beat_id: str) -> Path:
    return state_dir(run_id) / f"scene_{beat_id}.json"


def _pending_scene_path(run_id: str, beat_id: str) -> Path:
    """A scene staged for user confirmation (Phase 4b), not yet promoted to
    _scene_path(). Kept as a distinct file (not e.g. a flag inside
    scene_<id>.json) specifically so detect_stage() -- which only ever
    looks at _scene_path() -- never needs to change: a staged-but-unconfirmed
    batch simply doesn't exist yet as far as stage detection is concerned,
    preserving Phase 3's "purely from disk, never trust in-memory state"
    invariant untouched."""
    return state_dir(run_id) / f"pending_scene_{beat_id}.json"


_PENDING_PREFIX = "pending_scene_"


def _pending_beat_id(path: Path) -> str:
    return path.name[len(_PENDING_PREFIX) : -len(".json")]


def _script_path(run_id: str) -> Path:
    return state_dir(run_id) / "script.json"


def _graph_path(run_id: str) -> Path:
    """BiXiaScribe 重構 Phase 6: the run's causal-consistency graph,
    rebuilt from scratch (see crew/causal.py::build_graph()'s docstring for
    why) every time a scene is committed. Deliberately not consulted by
    detect_stage() -- it's a derived artifact of the scene checkpoints, not
    a stage marker of its own."""
    return state_dir(run_id) / "causal_graph.json"


# --- Stage detection ---------------------------------------------------


def detect_stage(run_id: str) -> Stage:
    """Determine which stage a run is at purely from what's on disk --
    never from an in-memory PipelineState, so this is safe to call right
    after a process crash. A checkpoint that exists but fails to
    deserialize/validate (load_checkpoint returns None) counts the same as
    "not done yet", which is what makes a corrupted scene_<id>.json file
    self-heal on the next dispatch_next() call instead of wedging the run.
    """
    if _script_path(run_id).exists() and load_checkpoint(_script_path(run_id), Script) is not None:
        return "done"

    extraction = load_checkpoint(_extraction_path(run_id), ExtractionResult)
    if extraction is None:
        return "extract"

    beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
    if beat_sheet is None:
        return "beats"

    for beat in beat_sheet.beats:
        if load_checkpoint(_scene_path(run_id, beat.id), Event) is None:
            return "scenes"

    return "proofread"


def _assemble_script(run_id: str) -> Script:
    """Build the final Script from the extraction/beats/scene checkpoints
    on disk. Only valid once detect_stage() reports "proofread" or later."""
    extraction = load_checkpoint(_extraction_path(run_id), ExtractionResult)
    beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
    if extraction is None or beat_sheet is None:
        raise PipelineError(f"run {run_id!r}: extraction/beats checkpoint 缺失，無法組裝劇本")

    events: list[Event] = []
    for beat in beat_sheet.beats:
        event = load_checkpoint(_scene_path(run_id, beat.id), Event)
        if event is None:
            raise PipelineError(
                f"run {run_id!r}: 場次 {beat.id!r} 的 checkpoint 缺失，無法組裝劇本"
            )
        events.append(event)

    return Script(
        title=beat_sheet.outline.title,
        premise=beat_sheet.outline.premise,
        variables=extraction.variables,
        npcs=extraction.npcs,
        events=events,
    )


# --- Causal consistency graph (BiXiaScribe 重構 Phase 6) --------------------


def _load_graph(run_id: str) -> CausalPlotGraph | None:
    """The graph as of the last _refresh_graph() call -- None if
    CAUSAL_VALIDATION=off (never written) or no scene has committed yet."""
    return load_checkpoint(_graph_path(run_id), CausalPlotGraph)


def _refresh_graph(
    run_id: str, beat_sheet: BeatSheet, events: list[Event]
) -> CausalPlotGraph | None:
    """Rebuild and checkpoint the causal graph from every currently-
    committed scene. A no-op (returns None, writes nothing) when
    config.CAUSAL_VALIDATION == "off" -- checked at call time so tests can
    reassign it, same convention as every other config.* read in this
    module."""
    if config.CAUSAL_VALIDATION == "off":
        return None
    graph = causal.build_graph(beat_sheet, events)
    save_checkpoint(_graph_path(run_id), graph)
    return graph


# --- Token usage accounting (BiXiaScribe 重構 Phase 5) ----------------------
#
# crewai 1.15.5's TaskOutput carries no usage info, and the layered path
# never builds a Crew (so Crew.calculate_usage_metrics() isn't available
# either). But every crewai BaseLLM (the real `LLM` and our own
# llm.py::FakeLLM) exposes get_token_usage_summary(), cumulative for that
# LLM instance's lifetime -- and each _default_* runner below builds a
# fresh agent+LLM per call, so a before/after delta on that instance is
# exactly that call's usage.

_USAGE_FIELDS = (
    "total_tokens",
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cache_creation_tokens",
    "successful_requests",
)


def _llm_usage(agent: Any) -> dict[str, int] | None:
    """Best-effort snapshot of `agent.llm`'s cumulative token usage so far,
    as a plain dict. None if the agent/llm doesn't expose one (never raises
    -- usage accounting must not be able to break generation)."""
    llm = getattr(agent, "llm", None)
    getter = getattr(llm, "get_token_usage_summary", None)
    if getter is None:
        return None
    try:
        summary = getter()
        return {field: int(getattr(summary, field, 0) or 0) for field in _USAGE_FIELDS}
    except Exception:  # noqa: BLE001 -- accounting must never break a run
        return None


def _usage_delta(
    before: dict[str, int] | None, after: dict[str, int] | None
) -> dict[str, int] | None:
    """Field-wise `after - before`, clamped at 0 (a fresh LLM instance
    should start at all-zero, but this stays correct even if it doesn't).
    None if either snapshot is unavailable."""
    if before is None or after is None:
        return None
    return {field: max(0, after.get(field, 0) - before.get(field, 0)) for field in _USAGE_FIELDS}


class _UsageAccumulator:
    """Thread-safe field-wise sum of per-call usage dicts. dispatch_batch()
    dispatches write_scene calls on a ThreadPoolExecutor, so this needs the
    same kind of lock crew/tools.py's RetrievalStats uses for the same
    reason (concurrent scene generation writing to shared state)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, int] = dict.fromkeys(_USAGE_FIELDS, 0)

    def add(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        with self._lock:
            for field in _USAGE_FIELDS:
                self._totals[field] += usage.get(field, 0)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return dict(self._totals)


class _UsageByRoleAccumulator:
    """Like _UsageAccumulator, but keyed by role name (extractor/
    beat_expander/scene_writer/proof) rather than one run-wide total. This is
    what lets pricing.py compute an exact per-role cost for a layered run
    instead of falling back to a uniform token/model estimate -- see
    RunReport.token_usage_by_role. run_layered() tags each dispatch_batch()
    call with the role it's about to run (known from `stage` right before
    dispatch) rather than dispatch_next()/dispatch_batch()'s own `on_usage`
    contract carrying a role -- see run_layered()'s docstring note on why."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, dict[str, int]] = {}

    def add(self, role: str, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        with self._lock:
            bucket = self._totals.setdefault(role, dict.fromkeys(_USAGE_FIELDS, 0))
            for field in _USAGE_FIELDS:
                bucket[field] += usage.get(field, 0)

    def as_dict(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {role: dict(v) for role, v in self._totals.items()}


def _unpack_stage_result(
    result: tuple[Any, ...],
) -> tuple[Any, str | None, dict[str, int] | None]:
    """Normalize a StageRunners callable's return value to
    (obj, coerced_from, usage). The real runners below return a 3-tuple
    whose last element is a usage dict; every existing test stand-in
    (CountingRunners in tests/test_orchestrator*.py) returns a 2-tuple and
    is accepted unchanged, reporting no usage. Tolerating both is
    deliberate: StageRunners exists precisely so tests can pass plain
    callables, and forcing every stand-in to grow a third element would
    make that contract more expensive than the bookkeeping is worth."""
    if len(result) >= 3:
        obj, source, usage = result[0], result[1], result[2]
        return obj, source, usage
    obj, source = result
    return obj, source, None


# --- Default stage runners (crew-backed) -----------------------------------
#
# Each returns (result, coerced_from, usage) so callers can see which
# _coerce_model fallback level produced it (same as the legacy pipeline's
# RunReport) and how many tokens the call spent (see _llm_usage() above).
# StageRunners lets tests swap these out for counting/failing stand-ins
# without monkeypatching module globals (crew/tools.py's lock-guarded
# globals are the one place this repo already does that, and it's noted
# there as leaking process-wide -- passing callables avoids that here).


def _default_extract(
    requirement: str, models: ModelChoice, verbose: bool
) -> tuple[ExtractionResult, str | None, dict[str, int] | None]:
    agent = make_extractor_agent(verbose=verbose, models=models)
    before = _llm_usage(agent)
    task = make_extract_task(requirement, agent)
    output = task.execute_sync(agent=agent)
    usage = _usage_delta(before, _llm_usage(agent))
    result, source = _coerce_model(output, ExtractionResult)
    if result is None:
        preview = (output.raw or "")[:500]
        raise PipelineError(
            "extractor 未能輸出符合 ExtractionResult schema 的結果，原始輸出前 500 字：\n" + preview
        )
    return result, source, usage


def _default_expand_beats(
    requirement: str,
    extraction: ExtractionResult,
    models: ModelChoice,
    verbose: bool,
    *,
    script_length: str = "short",
) -> tuple[BeatSheet, str | None, dict[str, int] | None]:
    agent = make_beat_expander_agent(verbose=verbose, models=models)
    before = _llm_usage(agent)
    task = make_beat_expand_task(requirement, extraction, agent, script_length=script_length)
    output = task.execute_sync(agent=agent)
    usage = _usage_delta(before, _llm_usage(agent))
    result, source = _coerce_model(output, BeatSheet)
    if result is None:
        preview = (output.raw or "")[:500]
        raise PipelineError(
            "beat_expander 未能輸出符合 BeatSheet schema 的結果，原始輸出前 500 字：\n" + preview
        )
    return result, source, usage


def _default_write_scene(
    beat: Beat,
    extraction: ExtractionResult,
    models: ModelChoice,
    verbose: bool,
    target_event_id: str,
    *,
    session: SessionDocument | None = None,
    script_length: str = "short",
) -> tuple[Event, str | None, dict[str, int] | None]:
    agent = make_scene_writer_agent(verbose=verbose, models=models)
    before = _llm_usage(agent)
    task = make_scene_write_task(
        beat,
        extraction,
        agent,
        target_event_id=target_event_id,
        session=session,
        script_length=script_length,
    )
    output = task.execute_sync(agent=agent)
    usage = _usage_delta(before, _llm_usage(agent))
    result, source = _coerce_model(output, Event)
    if result is None:
        preview = (output.raw or "")[:500]
        raise PipelineError(
            f"scene_writer 未能輸出符合 Event schema 的結果（場次 {beat.id!r}），"
            "原始輸出前 500 字：\n" + preview
        )
    return result, source, usage


def _default_repair_scene(
    event: Event, problems: list[str], models: ModelChoice, verbose: bool
) -> tuple[Event | None, str | None, dict[str, int] | None]:
    """BiXiaScribe 重構 Phase 6: ask the proofreader agent to fix specific
    causal-consistency problems in one already-generated scene. Unlike the
    other _default_* runners, a coercion failure returns None instead of
    raising PipelineError -- _validate_scene() treats a None repair result
    the same way crew/pipeline.py's proofread repair loop treats it: skip
    this attempt, keep the best result found so far. Under
    LLM_BACKEND=fake, the proofreader's FakeLLM branch returns a Script,
    not an Event, so this always coerces to None there -- a documented,
    deliberate degrade to "warn" behavior in offline tests (see
    tests/test_causal_consistency.py)."""
    agent = make_proofreader_agent(verbose=verbose, models=models)
    before = _llm_usage(agent)
    task = causal.repair_scene_task(event, problems, agent)
    output = task.execute_sync(agent=agent, context=event.model_dump_json())
    usage = _usage_delta(before, _llm_usage(agent))
    result, source = _coerce_model(output, Event)
    return result, source, usage


@dataclass
class StageRunners:
    """The three LLM-calling units of work dispatch_next() delegates to.
    Defaults are the real crew-backed implementations above; tests pass
    counting/failing stand-ins instead of monkeypatching module globals.

    Each callable may return either a 2-tuple `(result, coerced_from)` (every
    existing test stand-in) or a 3-tuple `(result, coerced_from, usage)`
    (the real runners, since Phase 5) -- callers unpack via
    _unpack_stage_result() so both shapes work unchanged. The type hints
    below are intentionally loose (`tuple[Any, ...]`) to allow either."""

    extract: Callable[[str, ModelChoice, bool], tuple[Any, ...]] = _default_extract
    expand_beats: Callable[
        [str, ExtractionResult, ModelChoice, bool], tuple[Any, ...]
    ] = _default_expand_beats
    write_scene: Callable[..., tuple[Any, ...]] = _default_write_scene
    """Called as write_scene(beat, extraction, models, verbose,
    target_event_id, session=...). The `session` kwarg is keyword-only and
    defaulted (see _default_write_scene/make_scene_write_task), so existing
    5-positional-arg stand-ins in tests keep working unchanged; Callable[...,]
    rather than a fixed positional signature is what lets that vary."""
    repair_scene: Callable[
        [Event, list[str], ModelChoice, bool], tuple[Any, ...]
    ] = _default_repair_scene
    """BiXiaScribe 重構 Phase 6. Called as repair_scene(event, problems,
    models, verbose) by _validate_scene() when config.CAUSAL_VALIDATION is
    "repair" or "strict" and a scene has causal-consistency problems.
    Defaulting it (rather than requiring every existing StageRunners
    construction to grow a 4th field) is what keeps the three pre-Phase-6
    test stub classes (CountingRunners/RecordingRunners/
    UsageReportingRunners) working unchanged."""


def _load_completed_scenes(
    run_id: str, beat_sheet: BeatSheet, state: PipelineState
) -> list[Event]:
    """Completed scenes in completion order (state.completed_scene_ids),
    which is what build_session_document() treats "most recent" as meaning.
    Falls back to appending any scene checkpoint present on disk but not
    yet recorded in state (e.g. resumed from a run whose state.json is
    stale) so a session document never loses continuity it could have had."""
    ordered_ids = list(state.completed_scene_ids)
    known = set(ordered_ids)
    for beat in beat_sheet.beats:
        if beat.id in known:
            continue
        if load_checkpoint(_scene_path(run_id, beat.id), Event) is not None:
            ordered_ids.append(beat.id)
            known.add(beat.id)

    scenes: list[Event] = []
    for beat_id in ordered_ids:
        event = load_checkpoint(_scene_path(run_id, beat_id), Event)
        if event is not None:
            scenes.append(event)
    return scenes


def _validate_scene(
    beat_sheet: BeatSheet,
    committed: list[Event],
    event: Event,
    *,
    runners: StageRunners,
    models: ModelChoice,
    verbose: bool,
    max_repair_attempts: int,
    on_usage: Callable[[dict[str, int] | None], None] | None = None,
) -> tuple[Event, list[str], int]:
    """BiXiaScribe 重構 Phase 6: validate `event` (freshly generated, not
    yet checkpointed) against `committed` scenes' causal facts, per
    config.CAUSAL_VALIDATION (read at call time, same convention as
    _refresh_graph()):

    - "off": skip entirely, return (event, [], 0).
    - "warn": check and return whatever problems are found, never repair.
    - "repair"/"strict": on conflict, try up to max_repair_attempts targeted
      repairs via runners.repair_scene(), following the same
      monotone-non-worsening rule as crew/pipeline.py's proofread repair
      loop -- a repair is only kept if it has no more problems than the
      best attempt so far. An attempt that raises just burns that attempt
      (doesn't abort the loop), same as _repair()'s existing behavior.

    Always returns its best attempt plus whatever problems remain -- it
    never raises and never decides whether "strict" should block
    checkpointing; that's the caller's job (dispatch_next/dispatch_batch),
    since only they know whether this is a single-scene or batch context.
    """
    mode = config.CAUSAL_VALIDATION
    if mode == "off":
        return event, [], 0

    problems = causal.check_scene_consistency(beat_sheet, committed, event)
    if not problems or mode == "warn":
        return event, problems, 0

    best_event, best_problems = event, problems
    attempts = 0
    while best_problems and attempts < max_repair_attempts:
        attempts += 1
        try:
            repaired, _source, usage = _unpack_stage_result(
                runners.repair_scene(best_event, best_problems, models, verbose)
            )
        except Exception:
            continue
        if on_usage is not None:
            on_usage(usage)
        if repaired is None:
            continue
        repaired_problems = causal.check_scene_consistency(beat_sheet, committed, repaired)
        if len(repaired_problems) <= len(best_problems):
            best_event, best_problems = repaired, repaired_problems

    return best_event, best_problems, attempts


# --- Dispatch ----------------------------------------------------------


def dispatch_next(
    run_id: str,
    requirement: str,
    *,
    runners: StageRunners | None = None,
    models: ModelChoice | None = None,
    verbose: bool = False,
    session_doc_max_tokens: int | None = None,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    on_usage: Callable[[dict[str, int] | None], None] | None = None,
    on_scene_session: Callable[[SessionDocument], None] | None = None,
    on_causal: Callable[[str, list[str], int], None] | None = None,
) -> PipelineState:
    """Execute exactly one unit of work for whatever stage `run_id` is
    currently at, checkpoint the result, and return the updated state.

    Granularity: one call each for "extract" and "beats" (each is a single
    LLM call), but only *one scene* per call while in "scenes" -- not the
    whole remaining batch. That's what gives a crash mid-scenes the
    per-scene resume behavior described in the refactor plan's 5.1
    (checkpoint granularity): a crash after 2 of 5 scenes leaves exactly 2
    scene_<id>.json files on disk, and the next dispatch_next() call
    resumes at scene 3 without re-calling the LLM for 1-2.

    Never touches "proofread"/"done" -- assembling the final Script and
    running the cross-reference repair loop is run_layered()'s job (it
    reuses crew/pipeline.py's existing _repair(), the same safety net the
    legacy and Phase-2 pipelines use), since that's a bounded, cheap,
    in-memory operation, not something worth its own checkpoint file.

    Since Phase 6, a freshly-generated scene is also checked with
    _validate_scene() before it's checkpointed -- see config.CAUSAL_VALIDATION
    for the off/warn/repair/strict modes. `on_causal`, if given, is called
    once per scene as on_causal(beat_id, problems, repair_attempts).
    """
    runners = runners or StageRunners()
    models = models or ModelChoice()

    stage = detect_stage(run_id)
    state = load_checkpoint(_state_path(run_id), PipelineState)
    if state is None:
        state = PipelineState(run_id=run_id, requirement=requirement)
    state.requirement = requirement
    state.stage = stage

    if stage == "extract":
        extraction, _source, usage = _unpack_stage_result(
            runners.extract(requirement, models, verbose)
        )
        if on_usage is not None:
            on_usage(usage)
        save_checkpoint(_extraction_path(run_id), extraction)
        state.stage = "beats"

    elif stage == "beats":
        extraction = load_checkpoint(_extraction_path(run_id), ExtractionResult)
        assert extraction is not None  # detect_stage() already confirmed this
        beat_sheet, _source, usage = _unpack_stage_result(
            runners.expand_beats(requirement, extraction, models, verbose)
        )
        if on_usage is not None:
            on_usage(usage)
        save_checkpoint(_beats_path(run_id), beat_sheet)
        state.stage = "scenes"

    elif stage == "scenes":
        beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
        extraction = load_checkpoint(_extraction_path(run_id), ExtractionResult)
        assert beat_sheet is not None and extraction is not None
        target_beat = next(
            (
                b
                for b in beat_sheet.beats
                if load_checkpoint(_scene_path(run_id, b.id), Event) is None
            ),
            None,
        )
        if target_beat is not None:
            completed_scenes = _load_completed_scenes(run_id, beat_sheet, state)
            graph = _load_graph(run_id)
            session = build_session_document(
                target_beat,
                extraction,
                completed_scenes,
                beat_sheet=beat_sheet,
                graph=graph,
                max_tokens=session_doc_max_tokens,
            )
            if on_scene_session is not None:
                on_scene_session(session)
            event, _source, usage = _unpack_stage_result(
                runners.write_scene(
                    target_beat, extraction, models, verbose, target_beat.id, session=session
                )
            )
            if on_usage is not None:
                on_usage(usage)
            # A scene_writer's own id choice is never trusted -- overwrite
            # with the beat's id (Event.id == Beat.id by convention), the
            # invariant a future parallel-scenes phase relies on to avoid
            # collisions.
            if event.id != target_beat.id:
                event = event.model_copy(update={"id": target_beat.id})

            event, problems, attempts = _validate_scene(
                beat_sheet,
                completed_scenes,
                event,
                runners=runners,
                models=models,
                verbose=verbose,
                max_repair_attempts=max_repair_attempts,
                on_usage=on_usage,
            )
            if on_causal is not None:
                on_causal(target_beat.id, problems, attempts)
            if problems and config.CAUSAL_VALIDATION == "strict":
                raise PipelineError(
                    f"run {run_id!r}: 場次 {target_beat.id!r} 因果一致性校驗未通過：\n"
                    + "\n".join(problems)
                )

            save_checkpoint(_scene_path(run_id, target_beat.id), event)
            if target_beat.id not in state.completed_scene_ids:
                state.completed_scene_ids.append(target_beat.id)
            _refresh_graph(run_id, beat_sheet, [*completed_scenes, event])
        state.stage = detect_stage(run_id)  # "scenes" again if more remain, else "proofread"

    # "proofread"/"done": nothing to dispatch -- see docstring.

    state.last_updated = time.time()
    save_checkpoint(_state_path(run_id), state)
    return state


def plan_batches(beats: list[Beat]) -> list[list[Beat]]:
    """Group `beats` into ordered batches by Beat.causal_deps (Kahn-style
    level ordering): batch k holds every beat whose deps are all satisfied
    by batches 0..k-1, so no two beats in the same batch ever depend on
    each other -- dispatch_batch() below relies on that to run a batch's
    beats concurrently.

    Pure and side-effect free (no disk I/O) so it's trivial to unit test.

    Two degradations, both deliberate -- a model-generated beat sheet must
    never be able to wedge or hang an otherwise-working run:
    - A dep id that doesn't name any beat in this list (the beat_expander
      can hallucinate one, or point at an event id from a different beat
      sheet) is treated as already satisfied rather than as a block.
    - A dependency cycle (a pass that satisfies zero new beats) falls back
      to emitting each remaining beat as its own single-beat batch, in
      original order -- i.e. it degrades to Phase 3's serial behavior
      instead of raising or looping forever.

    Within a batch, beats keep their original `beats` list order, so a
    concurrency=1 caller iterating batches in order reproduces the exact
    beat sequence Phase 3 used.
    """
    known_ids = {b.id for b in beats}
    remaining = list(beats)
    done: set[str] = set()
    batches: list[list[Beat]] = []

    while remaining:
        ready = [
            b
            for b in remaining
            if all(dep in done or dep not in known_ids for dep in b.causal_deps)
        ]
        if not ready:
            # Cycle: no beat's deps are fully satisfied. Degrade to one
            # single-beat batch per remaining beat, in original order.
            for b in remaining:
                batches.append([b])
                done.add(b.id)
            remaining = []
            break

        batches.append(ready)
        done.update(b.id for b in ready)
        ready_ids = {b.id for b in ready}
        remaining = [b for b in remaining if b.id not in ready_ids]

    return batches


def dispatch_batch(
    run_id: str,
    requirement: str,
    *,
    runners: StageRunners | None = None,
    models: ModelChoice | None = None,
    verbose: bool = False,
    concurrency: int | None = None,
    stage_pending: bool = False,
    session_doc_max_tokens: int | None = None,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    on_usage: Callable[[dict[str, int] | None], None] | None = None,
    on_scene_session: Callable[[SessionDocument], None] | None = None,
    on_causal: Callable[[str, list[str], int], None] | None = None,
) -> PipelineState:
    """Like dispatch_next(), but while in the "scenes" stage, generates an
    entire causally-independent batch of scenes concurrently instead of
    just one. For every other stage ("extract"/"beats"/"proofread"/"done")
    this is a thin passthrough to dispatch_next(), since those are single
    LLM calls with nothing to parallelize.

    concurrency=1 (or config.SCENE_CONCURRENCY resolving to 1) delegates
    an un-gated "scenes" stage to dispatch_next() too, so Phase 3's
    one-scene-at-a-time behavior is reproduced exactly, not just
    approximated -- this is what keeps every existing run_layered() test
    passing unchanged.

    A scene that raises inside the batch does not stop its siblings: every
    other scene's checkpoint is still saved (partial progress survives a
    crash, same guarantee dispatch_next() already gives one scene at a
    time), and the first exception encountered is re-raised only after all
    workers have finished.

    `stage_pending=True` (Phase 4b's confirmation gate) writes each
    generated scene to _pending_scene_path() instead of _scene_path(), and
    does not touch state.completed_scene_ids -- the batch stays invisible
    to detect_stage() until a caller promotes it via confirm_batch() or
    discards it via reject_batch(). This forces `concurrency` to be
    respected even when it's 1 (no dispatch_next() passthrough), since
    dispatch_next() has no concept of staging.

    Since Phase 6, every generated scene (committed or staged-pending
    alike) is checked with _validate_scene() before being written to disk
    -- see config.CAUSAL_VALIDATION. A "strict" scene that still has
    problems after repair does not stop its siblings (same guarantee as
    any other raising scene in this function), and the causal graph
    checkpoint is only refreshed from scenes that were actually committed
    (never from staged-pending ones -- those aren't visible to
    detect_stage() yet, so they shouldn't be visible to the graph either).
    """
    runners = runners or StageRunners()
    models = models or ModelChoice()
    concurrency = concurrency if concurrency is not None else config.SCENE_CONCURRENCY

    stage = detect_stage(run_id)
    if stage != "scenes":
        return dispatch_next(
            run_id,
            requirement,
            runners=runners,
            models=models,
            verbose=verbose,
            session_doc_max_tokens=session_doc_max_tokens,
            max_repair_attempts=max_repair_attempts,
            on_usage=on_usage,
            on_scene_session=on_scene_session,
            on_causal=on_causal,
        )
    if not stage_pending and concurrency <= 1:
        return dispatch_next(
            run_id,
            requirement,
            runners=runners,
            models=models,
            verbose=verbose,
            session_doc_max_tokens=session_doc_max_tokens,
            max_repair_attempts=max_repair_attempts,
            on_usage=on_usage,
            on_scene_session=on_scene_session,
            on_causal=on_causal,
        )

    state = load_checkpoint(_state_path(run_id), PipelineState)
    if state is None:
        state = PipelineState(run_id=run_id, requirement=requirement)
    state.requirement = requirement
    state.stage = stage

    beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
    extraction = load_checkpoint(_extraction_path(run_id), ExtractionResult)
    assert beat_sheet is not None and extraction is not None  # detect_stage() confirmed this

    def _still_needed(beat: Beat) -> bool:
        if load_checkpoint(_scene_path(run_id, beat.id), Event) is not None:
            return False
        if stage_pending:
            pending_event = load_checkpoint(_pending_scene_path(run_id, beat.id), Event)
            if pending_event is not None:
                return False
        return True

    batches = plan_batches(beat_sheet.beats)
    target_batch: list[Beat] = []
    for batch in batches:
        pending = [b for b in batch if _still_needed(b)]
        if pending:
            target_batch = pending
            break

    if target_batch:
        first_error: BaseException | None = None
        # Snapshot completed scenes once, before dispatch -- not per-worker.
        # Beats within one batch are causally independent by plan_batches()'
        # contract, so siblings must not see each other's in-flight output;
        # a per-worker read would make that nondeterministic under
        # concurrency (whichever sibling happens to finish first would leak
        # into the others' context).
        completed_scenes = _load_completed_scenes(run_id, beat_sheet, state)
        graph = _load_graph(run_id)
        sessions = {
            beat.id: build_session_document(
                beat,
                extraction,
                completed_scenes,
                beat_sheet=beat_sheet,
                graph=graph,
                max_tokens=session_doc_max_tokens,
            )
            for beat in target_batch
        }
        if on_scene_session is not None:
            for session in sessions.values():
                on_scene_session(session)
        committed_events: list[Event] = []
        with ThreadPoolExecutor(max_workers=min(concurrency, len(target_batch))) as pool:
            future_to_beat = {
                pool.submit(
                    runners.write_scene,
                    beat,
                    extraction,
                    models,
                    verbose,
                    beat.id,
                    session=sessions[beat.id],
                ): beat
                for beat in target_batch
            }
            for future in as_completed(future_to_beat):
                beat = future_to_beat[future]
                try:
                    event, _source, usage = _unpack_stage_result(future.result())
                except BaseException as exc:  # noqa: BLE001 -- re-raised below, once
                    if first_error is None:
                        first_error = exc
                    continue
                if on_usage is not None:
                    on_usage(usage)
                # A scene_writer's own id choice is never trusted -- see
                # dispatch_next()'s identical rule, the invariant that keeps
                # concurrent calls from colliding on checkpoint filenames.
                if event.id != beat.id:
                    event = event.model_copy(update={"id": beat.id})

                # Validate against the pre-batch snapshot, not siblings --
                # plan_batches() guarantees no two beats in this batch
                # depend on each other, so siblings are irrelevant to this
                # scene's causal ancestors regardless of dispatch order.
                event, problems, attempts = _validate_scene(
                    beat_sheet,
                    completed_scenes,
                    event,
                    runners=runners,
                    models=models,
                    verbose=verbose,
                    max_repair_attempts=max_repair_attempts,
                    on_usage=on_usage,
                )
                if on_causal is not None:
                    on_causal(beat.id, problems, attempts)
                if problems and config.CAUSAL_VALIDATION == "strict":
                    if first_error is None:
                        first_error = PipelineError(
                            f"run {run_id!r}: 場次 {beat.id!r} 因果一致性校驗未通過：\n"
                            + "\n".join(problems)
                        )
                    continue

                if stage_pending:
                    save_checkpoint(_pending_scene_path(run_id, beat.id), event)
                else:
                    save_checkpoint(_scene_path(run_id, beat.id), event)
                    committed_events.append(event)
                    if beat.id not in state.completed_scene_ids:
                        state.completed_scene_ids.append(beat.id)

        if committed_events:
            _refresh_graph(run_id, beat_sheet, [*completed_scenes, *committed_events])

        if first_error is not None:
            state.stage = detect_stage(run_id)
            state.last_updated = time.time()
            save_checkpoint(_state_path(run_id), state)
            raise first_error

    state.stage = detect_stage(run_id)  # "scenes" again if more remain, else "proofread"
    state.last_updated = time.time()
    save_checkpoint(_state_path(run_id), state)
    return state


def pending_batch_ids(run_id: str) -> list[str]:
    """Beat ids of the batch currently staged and awaiting confirmation
    (Phase 4b) -- empty if nothing is staged. Reads the filesystem only, so
    it's safe to call right after a crash/refresh, same as detect_stage()."""
    return sorted(_pending_beat_id(p) for p in state_dir(run_id).glob(f"{_PENDING_PREFIX}*.json"))


def confirm_batch(run_id: str) -> PipelineState:
    """Promote every staged pending_scene_<id>.json to scene_<id>.json (an
    atomic os.replace() per file, via the same save_checkpoint()/
    _atomic_write_json() machinery every other checkpoint uses) and mark
    those beats completed. A no-op (returns current state unchanged) if
    nothing is staged.

    Since Phase 6, also refreshes the causal graph checkpoint from the
    newly-committed scenes -- staged scenes were already validated when
    they were generated (dispatch_batch(stage_pending=True) calls
    _validate_scene() before staging), so this only needs to make them
    visible to the graph, not re-validate them."""
    state = load_checkpoint(_state_path(run_id), PipelineState) or PipelineState(
        run_id=run_id, requirement=""
    )
    promoted = False
    for beat_id in pending_batch_ids(run_id):
        event = load_checkpoint(_pending_scene_path(run_id, beat_id), Event)
        if event is not None:
            save_checkpoint(_scene_path(run_id, beat_id), event)
            promoted = True
            if beat_id not in state.completed_scene_ids:
                state.completed_scene_ids.append(beat_id)
        _pending_scene_path(run_id, beat_id).unlink(missing_ok=True)
    if promoted:
        beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
        if beat_sheet is not None:
            committed = _load_completed_scenes(run_id, beat_sheet, state)
            _refresh_graph(run_id, beat_sheet, committed)
    state.stage = detect_stage(run_id)
    state.last_updated = time.time()
    save_checkpoint(_state_path(run_id), state)
    return state


def reject_batch(run_id: str) -> PipelineState:
    """Discard every staged pending_scene_<id>.json without promoting it --
    the next dispatch_batch(..., stage_pending=True) call regenerates the
    same batch from scratch."""
    for beat_id in pending_batch_ids(run_id):
        _pending_scene_path(run_id, beat_id).unlink(missing_ok=True)
    state = load_checkpoint(_state_path(run_id), PipelineState) or PipelineState(
        run_id=run_id, requirement=""
    )
    state.stage = detect_stage(run_id)
    state.last_updated = time.time()
    save_checkpoint(_state_path(run_id), state)
    return state


def _default_run_id(requirement: str) -> str:
    return f"{int(time.time())}-{review.requirement_slug(requirement)}"


def run_layered(
    requirement: str,
    *,
    run_id: str | None = None,
    runners: StageRunners | None = None,
    models: ModelChoice | None = None,
    verbose: bool = False,
    on_step: Callable[[StepEvent], None] | None = None,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    concurrency: int | None = None,
    gate: Callable[[list[str]], bool] | None = None,
    session_doc_max_tokens: int | None = None,
    script_length: str = "short",
) -> tuple[Script, RunReport]:
    """Run (or resume) the layered pipeline to completion, dispatching one
    stage (or, in "scenes", one causally-independent batch -- see
    plan_batches()/dispatch_batch(), Phase 4) at a time and persisting a
    checkpoint after each. Returns the same (Script, RunReport) shape as
    run_pipeline_with_report()/run_layered_pipeline(), so callers can switch
    between all three without touching downstream code.

    `run_id` identifies which `.bixia_state/<run_id>/` checkpoint directory
    to use -- pass the same run_id used by a previous, interrupted call to
    resume it (already-completed stages/scenes are read back from disk, not
    regenerated). Omit it to start a fresh run with a generated id.

    `concurrency` (default: config.SCENE_CONCURRENCY) caps how many scenes
    within one batch run at once; 1 reproduces Phase 3's exact one-scene-
    per-call sequence via dispatch_batch()'s passthrough to dispatch_next().

    `gate` (Phase 4b), if given, is called with the beat ids of each scenes
    batch right after it's generated (staged as pending_scene_<id>.json --
    see dispatch_batch(stage_pending=True)), and must return True to
    promote the batch (confirm_batch()) or False to discard and regenerate
    it (reject_batch()). It's called synchronously and may block -- that's
    the intended way for a caller (e.g. generation.GenerationJob) to pause
    a background thread until a UI user decides. `gate=None` (the default)
    skips staging entirely and behaves exactly as before Phase 4b.

    `session_doc_max_tokens` (Phase 5) is forwarded to every
    build_session_document() call made while writing scenes: `None` (the
    default) uses config.SESSION_DOC_MAX_TOKENS, `0` (or negative) disables
    trimming entirely. This is the knob the compressed-vs-untrimmed quality
    regression (docs/BiXiaScribe_REFACTORING_PLAN.md Phase 5) is built on.

    `report.token_usage` only counts LLM calls made by *this* process: a
    resumed run reads already-completed stages/scenes back from disk
    without calling the model for them, so a resumed run's token_usage is
    the incremental spend of the resume, not the run's lifetime total --
    sum the JSONL rows sharing a run_id for the lifetime figure.

    `max_repair_attempts` is also the cap on Phase 6's per-scene causal
    repair attempts (config.CAUSAL_VALIDATION), not just the final
    proofread repair loop. `report.causal_problems`/`causal_repair_attempts`
    accumulate across every scene plus the final structural check;
    `report.causal_validation` records which mode this run used.

    `script_length` (default "short", config.SCRIPT_LENGTH's default and
    byte-identical to this pipeline's pre-knob prompts) is forwarded to
    every beat_expand/scene_write task built by the *default* crew-backed
    runners -- see crew/tasks.py::_LENGTH_TARGETS. It has no effect when a
    caller supplies its own `runners` (test stand-ins own their own prompt
    behavior, or lack of it).

    `report.token_usage_by_role` buckets usage by which stage spent it
    (extractor/beat_expander/scene_writer/proof) rather than only a run-wide
    total -- run_layered() tags each dispatch_batch() call with the stage
    it's about to run (rather than changing dispatch_next()/dispatch_batch()'s
    own `on_usage` callback contract, which a test in
    tests/test_orchestrator_parallel.py calls directly with a single-arg
    callback). A "scenes" stage's bucket folds in both the scene_writer call
    and any Phase 6 causal repair calls for that batch (both use the same
    model split in practice), and the final structural proofread repair loop
    is folded into "proof".
    """
    reset_stats()
    start = time.monotonic()
    models = models or ModelChoice()
    run_id = run_id or _default_run_id(requirement)
    usage = _UsageAccumulator()
    usage_by_role = _UsageByRoleAccumulator()
    omitted_total = 0
    causal_problems_total: list[str] = []
    causal_repair_attempts_total = 0

    if runners is None:
        # Bind script_length into the default crew-backed runners here
        # (rather than letting dispatch_next()/dispatch_batch() construct
        # their own StageRunners() when given runners=None) -- that's the
        # only way this run's script_length reaches make_beat_expand_task()/
        # make_scene_write_task() without changing either function's public
        # signature. A caller-supplied `runners` is left untouched.
        runners = StageRunners(
            expand_beats=functools.partial(_default_expand_beats, script_length=script_length),
            write_scene=functools.partial(_default_write_scene, script_length=script_length),
        )

    _ROLE_BY_STAGE = {"extract": "extractor", "beats": "beat_expander", "scenes": "scene_writer"}

    def _on_usage_for(stage: str) -> Callable[[dict[str, int] | None], None]:
        role = _ROLE_BY_STAGE.get(stage, stage)

        def _on_usage(delta: dict[str, int] | None) -> None:
            usage.add(delta)
            usage_by_role.add(role, delta)

        return _on_usage

    def _on_scene_session(session: SessionDocument) -> None:
        nonlocal omitted_total
        omitted_total += session.omitted_scene_count

    def _on_causal(beat_id: str, problems: list[str], attempts: int) -> None:
        nonlocal causal_repair_attempts_total
        del beat_id  # already embedded in each problem message
        causal_repair_attempts_total += attempts
        causal_problems_total.extend(problems)

    report = RunReport(
        requirement=requirement,
        model_writer=models.writer,
        model_dialogue=models.dialogue,
        model_proof=models.proof,
        mode="layered",
        model_extractor=models.extractor,
        model_beat_expander=models.beat_expander,
        model_scene_writer=models.scene_writer,
        session_doc_max_tokens=session_doc_max_tokens,
        script_length=script_length,
    )

    step_index = 0

    def _emit(kind: str, role: str, text: str) -> None:
        nonlocal step_index
        if on_step is None:
            return
        step_index += 1
        on_step(StepEvent(kind=kind, role=role, text=text, index=step_index))

    def _finalize_report() -> None:
        stats = get_stats()
        report.elapsed_s = time.monotonic() - start
        report.retrieval_calls = stats.calls
        report.retrieval_failures = stats.failures
        report.retrieval_queries = list(stats.queries)
        report.token_usage = usage.as_dict()
        report.token_usage_by_role = usage_by_role.as_dict()
        report.session_doc_omitted_total = omitted_total
        report.causal_validation = config.CAUSAL_VALIDATION
        report.causal_problems = list(causal_problems_total)
        report.causal_repair_attempts = causal_repair_attempts_total

    stage_labels: dict[str, str] = {
        "extract": "拆書",
        "beats": "排場",
        "scenes": "寫戲",
    }

    while True:
        stage = detect_stage(run_id)
        if stage in ("proofread", "done"):
            break

        if stage == "scenes" and gate is not None:
            pending_ids = pending_batch_ids(run_id)
            if not pending_ids:
                _emit("phase", "寫戲", "開始執行（等待確認）")
                dispatch_batch(
                    run_id,
                    requirement,
                    runners=runners,
                    models=models,
                    verbose=verbose,
                    concurrency=concurrency,
                    stage_pending=True,
                    session_doc_max_tokens=session_doc_max_tokens,
                    max_repair_attempts=max_repair_attempts,
                    on_usage=_on_usage_for(stage),
                    on_scene_session=_on_scene_session,
                    on_causal=_on_causal,
                )
                pending_ids = pending_batch_ids(run_id)
            if gate(pending_ids):
                confirm_batch(run_id)
                _emit("task", "寫戲", f"已確認 {len(pending_ids)} 場")
            else:
                reject_batch(run_id)
                _emit("task", "寫戲", "已拒絕，重新生成本批")
            continue

        batch_note = ""
        if stage == "scenes":
            beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
            if beat_sheet is not None:
                for batch in plan_batches(beat_sheet.beats):
                    pending = [
                        b
                        for b in batch
                        if load_checkpoint(_scene_path(run_id, b.id), Event) is None
                    ]
                    if pending:
                        batch_note = f"（本批 {len(pending)} 場）"
                        break

        _emit("phase", stage_labels.get(stage, stage), f"開始執行{batch_note}")
        dispatch_batch(
            run_id,
            requirement,
            runners=runners,
            models=models,
            verbose=verbose,
            concurrency=concurrency,
            session_doc_max_tokens=session_doc_max_tokens,
            max_repair_attempts=max_repair_attempts,
            on_usage=_on_usage_for(stage),
            on_scene_session=_on_scene_session,
            on_causal=_on_causal,
        )
        _emit("task", stage_labels.get(stage, stage), "任務完成")

    if detect_stage(run_id) == "done":
        script = load_checkpoint(_script_path(run_id), Script)
        if script is None:  # pragma: no cover -- detect_stage() already checked this
            _finalize_report()
            raise PipelineError(
                f"run {run_id!r}: 標記為 done 但 script.json 無法讀取", report=report
            )
        beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
        report.scenes_generated = len(beat_sheet.beats) if beat_sheet else 0
        _finalize_report()
        return script, report

    _emit("phase", "校對", "開始執行")
    script = _assemble_script(run_id)
    beat_sheet = load_checkpoint(_beats_path(run_id), BeatSheet)
    report.scenes_generated = len(beat_sheet.beats) if beat_sheet else 0

    problems = validate_references(script)
    best_script, best_problems = script, problems

    attempts = 0
    if best_problems:
        proofreader = make_proofreader_agent(verbose=verbose, models=models)
        while best_problems and attempts < max_repair_attempts:
            attempts += 1
            _emit("phase", "校對", f"修補嘗試 {attempts}/{max_repair_attempts}")
            before = _llm_usage(proofreader)
            try:
                repaired, _source = _repair(best_script, best_problems, proofreader)
            except Exception:
                continue
            finally:
                # A repair attempt that raised still spent tokens.
                delta = _usage_delta(before, _llm_usage(proofreader))
                usage.add(delta)
                usage_by_role.add("proof", delta)
            if repaired is None:
                continue
            repaired_problems = validate_references(repaired)
            if len(repaired_problems) <= len(best_problems):
                best_script, best_problems = repaired, repaired_problems
    report.repair_attempts = attempts

    # Phase 6: structural causal-graph check at proofread time, on top of
    # the per-scene semantic checks already run by dispatch_next()/
    # dispatch_batch() while writing each scene. validate_causal_graph()
    # only checks edge referential integrity (schema.py), which is safe to
    # run now that every scene exists -- unlike per-scene generation time,
    # where a branch pointing at a not-yet-written scene is expected, not
    # a problem.
    structural_problems: list[str] = []
    if config.CAUSAL_VALIDATION != "off":
        final_graph = _refresh_graph(run_id, beat_sheet, best_script.events) if beat_sheet else None
        if final_graph is not None:
            structural_problems = validate_causal_graph(final_graph)
            causal_problems_total.extend(structural_problems)

    _finalize_report()

    if best_problems:
        raise PipelineError(
            "校對後的劇本仍有交叉引用錯誤：\n" + "\n".join(best_problems),
            report=report,
        )

    if structural_problems and config.CAUSAL_VALIDATION == "strict":
        raise PipelineError(
            "校對後的因果圖仍有結構錯誤：\n" + "\n".join(structural_problems),
            report=report,
        )

    save_checkpoint(_script_path(run_id), best_script)
    state = load_checkpoint(_state_path(run_id), PipelineState) or PipelineState(
        run_id=run_id, requirement=requirement
    )
    state.stage = "done"
    state.last_updated = time.time()
    save_checkpoint(_state_path(run_id), state)

    return best_script, report
