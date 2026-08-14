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

import json
import os
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
    Event,
    ExtractionResult,
    Script,
    validate_references,
)
from .agents import (
    make_beat_expander_agent,
    make_extractor_agent,
    make_proofreader_agent,
    make_scene_writer_agent,
)
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


# --- Default stage runners (crew-backed) -----------------------------------
#
# Each returns (result, coerced_from) so callers can see which _coerce_model
# fallback level produced it, same as the legacy pipeline's RunReport.
# StageRunners lets tests swap these out for counting/failing stand-ins
# without monkeypatching module globals (crew/tools.py's lock-guarded
# globals are the one place this repo already does that, and it's noted
# there as leaking process-wide -- passing callables avoids that here).


def _default_extract(
    requirement: str, models: ModelChoice, verbose: bool
) -> tuple[ExtractionResult, str | None]:
    agent = make_extractor_agent(verbose=verbose, models=models)
    task = make_extract_task(requirement, agent)
    output = task.execute_sync(agent=agent)
    result, source = _coerce_model(output, ExtractionResult)
    if result is None:
        preview = (output.raw or "")[:500]
        raise PipelineError(
            "extractor 未能輸出符合 ExtractionResult schema 的結果，原始輸出前 500 字：\n" + preview
        )
    return result, source


def _default_expand_beats(
    requirement: str, extraction: ExtractionResult, models: ModelChoice, verbose: bool
) -> tuple[BeatSheet, str | None]:
    agent = make_beat_expander_agent(verbose=verbose, models=models)
    task = make_beat_expand_task(requirement, extraction, agent)
    output = task.execute_sync(agent=agent)
    result, source = _coerce_model(output, BeatSheet)
    if result is None:
        preview = (output.raw or "")[:500]
        raise PipelineError(
            "beat_expander 未能輸出符合 BeatSheet schema 的結果，原始輸出前 500 字：\n" + preview
        )
    return result, source


def _default_write_scene(
    beat: Beat,
    extraction: ExtractionResult,
    models: ModelChoice,
    verbose: bool,
    target_event_id: str,
) -> tuple[Event, str | None]:
    agent = make_scene_writer_agent(verbose=verbose, models=models)
    task = make_scene_write_task(beat, extraction, agent, target_event_id=target_event_id)
    output = task.execute_sync(agent=agent)
    result, source = _coerce_model(output, Event)
    if result is None:
        preview = (output.raw or "")[:500]
        raise PipelineError(
            f"scene_writer 未能輸出符合 Event schema 的結果（場次 {beat.id!r}），"
            "原始輸出前 500 字：\n" + preview
        )
    return result, source


@dataclass
class StageRunners:
    """The three LLM-calling units of work dispatch_next() delegates to.
    Defaults are the real crew-backed implementations above; tests pass
    counting/failing stand-ins instead of monkeypatching module globals."""

    extract: Callable[[str, ModelChoice, bool], tuple[ExtractionResult, str | None]] = (
        _default_extract
    )
    expand_beats: Callable[
        [str, ExtractionResult, ModelChoice, bool], tuple[BeatSheet, str | None]
    ] = _default_expand_beats
    write_scene: Callable[
        [Beat, ExtractionResult, ModelChoice, bool, str], tuple[Event, str | None]
    ] = _default_write_scene


# --- Dispatch ----------------------------------------------------------


def dispatch_next(
    run_id: str,
    requirement: str,
    *,
    runners: StageRunners | None = None,
    models: ModelChoice | None = None,
    verbose: bool = False,
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
        extraction, _source = runners.extract(requirement, models, verbose)
        save_checkpoint(_extraction_path(run_id), extraction)
        state.stage = "beats"

    elif stage == "beats":
        extraction = load_checkpoint(_extraction_path(run_id), ExtractionResult)
        assert extraction is not None  # detect_stage() already confirmed this
        beat_sheet, _source = runners.expand_beats(requirement, extraction, models, verbose)
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
            event, _source = runners.write_scene(
                target_beat, extraction, models, verbose, target_beat.id
            )
            # A scene_writer's own id choice is never trusted -- overwrite
            # with the beat's id (Event.id == Beat.id by convention), the
            # invariant a future parallel-scenes phase relies on to avoid
            # collisions.
            if event.id != target_beat.id:
                event = event.model_copy(update={"id": target_beat.id})
            save_checkpoint(_scene_path(run_id, target_beat.id), event)
            if target_beat.id not in state.completed_scene_ids:
                state.completed_scene_ids.append(target_beat.id)
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
    """
    runners = runners or StageRunners()
    models = models or ModelChoice()
    concurrency = concurrency if concurrency is not None else config.SCENE_CONCURRENCY

    stage = detect_stage(run_id)
    if stage != "scenes":
        return dispatch_next(run_id, requirement, runners=runners, models=models, verbose=verbose)
    if not stage_pending and concurrency <= 1:
        return dispatch_next(run_id, requirement, runners=runners, models=models, verbose=verbose)

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
        with ThreadPoolExecutor(max_workers=min(concurrency, len(target_batch))) as pool:
            future_to_beat = {
                pool.submit(
                    runners.write_scene, beat, extraction, models, verbose, beat.id
                ): beat
                for beat in target_batch
            }
            for future in as_completed(future_to_beat):
                beat = future_to_beat[future]
                try:
                    event, _source = future.result()
                except BaseException as exc:  # noqa: BLE001 -- re-raised below, once
                    if first_error is None:
                        first_error = exc
                    continue
                # A scene_writer's own id choice is never trusted -- see
                # dispatch_next()'s identical rule, the invariant that keeps
                # concurrent calls from colliding on checkpoint filenames.
                if event.id != beat.id:
                    event = event.model_copy(update={"id": beat.id})
                if stage_pending:
                    save_checkpoint(_pending_scene_path(run_id, beat.id), event)
                else:
                    save_checkpoint(_scene_path(run_id, beat.id), event)
                    if beat.id not in state.completed_scene_ids:
                        state.completed_scene_ids.append(beat.id)

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
    nothing is staged."""
    state = load_checkpoint(_state_path(run_id), PipelineState) or PipelineState(
        run_id=run_id, requirement=""
    )
    for beat_id in pending_batch_ids(run_id):
        event = load_checkpoint(_pending_scene_path(run_id, beat_id), Event)
        if event is not None:
            save_checkpoint(_scene_path(run_id, beat_id), event)
            if beat_id not in state.completed_scene_ids:
                state.completed_scene_ids.append(beat_id)
        _pending_scene_path(run_id, beat_id).unlink(missing_ok=True)
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
    """
    reset_stats()
    start = time.monotonic()
    models = models or ModelChoice()
    run_id = run_id or _default_run_id(requirement)

    report = RunReport(
        requirement=requirement,
        model_writer=models.writer,
        model_dialogue=models.dialogue,
        model_proof=models.proof,
        mode="layered",
        model_extractor=models.extractor,
        model_beat_expander=models.beat_expander,
        model_scene_writer=models.scene_writer,
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
            try:
                repaired, _source = _repair(best_script, best_problems, proofreader)
            except Exception:
                continue
            if repaired is None:
                continue
            repaired_problems = validate_references(repaired)
            if len(repaired_problems) <= len(best_problems):
                best_script, best_problems = repaired, repaired_problems
    report.repair_attempts = attempts

    _finalize_report()

    if best_problems:
        raise PipelineError(
            "校對後的劇本仍有交叉引用錯誤：\n" + "\n".join(best_problems),
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
