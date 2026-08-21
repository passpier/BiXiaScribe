"""Structured-output-failure fallback wrapper around Task.execute_sync().

Why this exists: crewai's structured-output path (an Agent's tools=[] task
gets `response_model` forwarded straight to the provider as
`response_format`, see openai/completion.py::_handle_completion) asks the
*provider*, not this process, to emit well-formed JSON matching the wire
schema. When the provider truncates its own response mid-object (observed
against deepseek-v4-flash-0731: `content == '{\\n '`), the openai SDK's own
`client.beta.chat.completions.parse()` raises `pydantic.ValidationError`
*inside* `Task.execute_sync()`, before crew/pipeline.py::_coerce_model's
three-tier (pydantic -> json_dict -> raw_scan) rescue ever gets a chance to
run -- see docs/DESIGN_NOTES.md's 2026-08-21 "通過率修復實測記錄" for the
first occurrence of this exact failure mode (there: a dropped required
field; here: mid-object truncation -- same exception path, different
provider misbehavior).

crewai's own Agent.max_retry_limit (default 2) already retries the *same*
call shape a couple of times before giving up -- see agent/core.py::
_check_execution_error. That absorbs a one-off transient truncation, but
burns a full-price call each time and does nothing if the provider
truncates structured output systematically for this prompt shape. This
module adds one more level: on a structured-output parse failure, retry
*once* with the same task rebuilt in free-text mode (no output_pydantic;
schema spelled out in the prompt instead -- see tasks.py::
_freeform_expected_output), so the salvage logic in
pipeline.py::_coerce_model (extended with a lenient-mirror raw-scan tier
for exactly this) gets a real shot at the response. A non-structured-output
error (401/429/timeout/etc.) is never retried here -- a second call
wouldn't fix it and would just double the cost.

Fallback counts/notes are tracked as a module-level, thread-safe
accumulator -- same pattern as crew/tools.py's RetrievalStats -- since the
StageRunners callables orchestrator.py dispatches don't have a return-value
slot for this without changing every existing test stand-in's tuple shape.
Callers reset at the start of one pipeline run (reset_stats()) and read the
totals back into RunReport at the end (get_stats()), exactly like
tools.reset_stats()/get_stats().
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .. import config

_stats_lock = threading.Lock()


@dataclass
class FallbackStats:
    count: int = 0
    notes: list[str] = field(default_factory=list)


_stats = FallbackStats()


def get_stats() -> FallbackStats:
    return _stats


def reset_stats() -> None:
    """Zero the counters -- call once at the start of one pipeline run, same
    convention as crew/tools.py::reset_stats()."""
    global _stats
    with _stats_lock:
        _stats = FallbackStats()


def _record(note: str) -> None:
    with _stats_lock:
        _stats.count += 1
        _stats.notes.append(note)


# Substring markers checked in addition to isinstance(), since the exception
# surfacing from deep inside crewai's openai completion wrapper isn't
# guaranteed to be a pydantic.ValidationError/json.JSONDecodeError this
# process's own imports produce the same class object for (observed:
# litellm/openai occasionally re-wrap). Deliberately narrow -- an
# auth/rate-limit/timeout error must NOT match, or a second, equally
# billable call would be wasted retrying something a retry can't fix.
_STRUCTURED_ERROR_MARKERS = (
    "Invalid JSON",
    "validation error for Lenient",
    "json_invalid",
)


def is_structured_parse_error(exc: BaseException) -> bool:
    """True if `exc` looks like a provider-side structured-output parse
    failure worth retrying in free-text mode -- see this module's
    docstring."""
    if isinstance(exc, (ValidationError, json.JSONDecodeError)):
        return True
    text = str(exc)
    return any(marker in text for marker in _STRUCTURED_ERROR_MARKERS)


@dataclass
class ExecOutcome:
    output: Any
    degraded: bool
    note: str = ""


def run_task(
    build_task: Callable[[bool], Any],
    agent: Any,
    *,
    role_label: str = "",
    allow_fallback: bool = True,
    execute_kwargs: dict[str, Any] | None = None,
) -> ExecOutcome:
    """Execute `build_task(True)` (structured output) via
    `task.execute_sync(agent=agent, **execute_kwargs)`. On a
    structured-output parse failure (see is_structured_parse_error),
    rebuild and run `build_task(False)` (free-text) once instead of
    propagating -- any other exception (a genuine provider/auth/rate-limit
    failure) is re-raised immediately, no second call attempted.

    `build_task` takes the `structured: bool` flag make_*_task() factories
    now accept (crew/tasks.py) so this stays decoupled from any one task's
    construction signature. `execute_kwargs` covers callers that need extra
    arguments on execute_sync() beyond `agent=` (e.g.
    causal.repair_scene_task's `context=event.model_dump_json()`).
    config.STRUCTURED_OUTPUT == "off" skips the structured attempt entirely
    and goes straight to free-text -- no fallback bookkeeping in that case,
    since there was nothing to fall back from."""
    execute_kwargs = execute_kwargs or {}
    if config.STRUCTURED_OUTPUT == "off":
        task = build_task(False)
        output = task.execute_sync(agent=agent, **execute_kwargs)
        return ExecOutcome(output=output, degraded=False)

    task = build_task(True)
    try:
        output = task.execute_sync(agent=agent, **execute_kwargs)
        return ExecOutcome(output=output, degraded=False)
    except Exception as exc:  # noqa: BLE001 -- re-raised below unless it matches
        if not allow_fallback or not is_structured_parse_error(exc):
            raise
        note = (
            f"{role_label or getattr(agent, 'role', '')}："
            f"結構化輸出解析失敗（{exc}），已改用自由文字模式重試一次"
        )
        fallback_task = build_task(False)
        output = fallback_task.execute_sync(agent=agent, **execute_kwargs)
        _record(note)
        return ExecOutcome(output=output, degraded=True, note=note)
