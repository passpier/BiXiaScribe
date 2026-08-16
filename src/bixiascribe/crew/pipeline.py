"""Sequential Crew wiring: 編劇 -> 對話 -> 校對, producing one validated
Script. This is the Stage 2 entry point -- scripts/generate_script.py and
tests/test_crew_pipeline.py both call run_pipeline()."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from crewai import Crew, Process, Task
from pydantic import BaseModel

from .. import config
from ..llm import ModelChoice
from ..schema import Script, parse_model_json, validate_references
from .agents import make_dialogue_agent, make_proofreader_agent, make_writer_agent
from .tasks import make_dialogue_task, make_proofread_task, make_writer_task
from .tools import get_stats, reset_stats

M = TypeVar("M", bound=BaseModel)

# Cross-reference problems (dangling npc_id / next_event_id) are handed back
# to the 校對 agent for a targeted repair pass this many times before giving
# up. Re-running just the proofread task (not the whole crew) is deliberate:
# the writer/dialogue output isn't in question, only structural references,
# so a full re-run would triple the token cost for no benefit.
MAX_REPAIR_ATTEMPTS = 2


class PipelineError(RuntimeError):
    """Raised when the crew doesn't produce a schema-valid, reference-clean
    Script (e.g. the proofreader's final output fails validate_references()),
    or when crew.kickoff() itself fails (provider errors, timeouts, etc.).
    `report` carries whatever RunReport data was gathered before the
    failure -- token spend and tool-call counts are most useful exactly
    when a run didn't succeed."""

    def __init__(self, message: str, report: RunReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class RunReport:
    """A summary of one run_pipeline_with_report() call: which models were
    used, how long it took, how many tokens it spent, and -- most
    importantly -- whether the 對話 agent actually called
    wuxia_corpus_search at all (see crew/tools.py's RetrievalStats
    docstring for why that can silently be zero on a real model)."""

    requirement: str = ""
    model_writer: str = config.LLM_MODEL_WRITER
    model_dialogue: str = config.LLM_MODEL_DIALOGUE
    model_proof: str = config.LLM_MODEL_PROOF
    elapsed_s: float = 0.0
    token_usage: dict[str, Any] = field(default_factory=dict)
    retrieval_calls: int = 0
    retrieval_failures: int = 0
    retrieval_queries: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    coerced_from: str | None = None
    # Layered-pipeline fields (BiXiaScribe 重構 Phase 2). All default to
    # values that make a legacy run_pipeline_with_report() row indistinguishable
    # from before -- run_layered_pipeline() is the only thing that sets them.
    mode: str = "legacy"
    model_extractor: str = ""
    model_beat_expander: str = ""
    model_scene_writer: str = ""
    scenes_generated: int = 0
    # Phase 5 quality-regression fields (BiXiaScribe 重構 Phase 5). Only
    # run_layered() sets these -- a legacy run leaves both at their defaults.
    # session_doc_max_tokens mirrors run_layered()'s own argument (None =
    # config.SESSION_DOC_MAX_TOKENS was used, 0 = trimming was disabled);
    # session_doc_omitted_total is the sum of every SessionDocument's
    # omitted_scene_count across the run -- the manipulation check for the
    # compressed-vs-untrimmed experiment (see docs/BiXiaScribe_REFACTORING_
    # PLAN.md Phase 5): without it, "arm A never actually trimmed anything"
    # is invisible in the JSONL data.
    session_doc_max_tokens: int | None = None
    session_doc_omitted_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe representation of one run -- the row shape shared
        by scripts/generate_script.py's stderr report and
        scripts/eval_generation.py's JSONL log, so both stay in sync with
        whatever fields RunReport actually carries."""
        return {
            "requirement": self.requirement,
            "model_writer": self.model_writer,
            "model_dialogue": self.model_dialogue,
            "model_proof": self.model_proof,
            "elapsed_s": self.elapsed_s,
            "token_usage": self.token_usage,
            "retrieval_calls": self.retrieval_calls,
            "retrieval_failures": self.retrieval_failures,
            "retrieval_queries": self.retrieval_queries,
            "repair_attempts": self.repair_attempts,
            "coerced_from": self.coerced_from,
            "mode": self.mode,
            "model_extractor": self.model_extractor,
            "model_beat_expander": self.model_beat_expander,
            "model_scene_writer": self.model_scene_writer,
            "scenes_generated": self.scenes_generated,
            "session_doc_max_tokens": self.session_doc_max_tokens,
            "session_doc_omitted_total": self.session_doc_omitted_total,
        }


@dataclass(frozen=True)
class StepEvent:
    """One tick of pipeline progress, translated from crewai's own
    AgentAction/AgentFinish/TaskOutput types so no caller (e.g. a UI progress
    widget) needs to import crewai to consume `on_step` events.

    `kind` is "phase" for the synthetic before-kickoff/repair-loop markers
    this module emits itself, or "task" for a real task_callback firing
    (crewai's step_callback never fires for our toolless 編劇/校對 agents --
    see crew_agent_executor.py's `_invoke_loop_native_no_tools`, which skips
    it entirely -- so "task" is the only crewai-sourced kind in practice).
    `index` is 1-based and monotonic within one run_pipeline_with_report()
    call.
    """

    kind: str = "phase"
    role: str = ""
    text: str = ""
    index: int = 0


def _coerce_model(output: Any, model_cls: type[M]) -> tuple[M | None, str | None]:
    """Extract a `model_cls` instance from a CrewOutput or TaskOutput, in
    decreasing order of trust: the pydantic object CrewAI's output_pydantic
    coercion already produced, then its json_dict, then a salvage scan of
    the raw text. Real models frequently wrap their JSON in explanatory
    prose that trips up CrewAI's own coercion even though the JSON itself is
    fine -- see schema.parse_model_json for why the raw scan validates
    against the schema instead of just keyword-matching.

    Returns (obj, source) where source identifies which level produced it
    ("pydantic" / "json_dict" / "raw_scan"), or (None, None) if nothing
    validated -- source is reported in RunReport.coerced_from since a real
    model landing on "raw_scan" is itself a signal worth seeing.
    """
    if isinstance(output.pydantic, model_cls):
        return output.pydantic, "pydantic"

    if output.json_dict is not None:
        try:
            return model_cls.model_validate(output.json_dict), "json_dict"
        except Exception:
            pass

    if output.raw:
        obj = parse_model_json(output.raw, model_cls)
        if obj is not None:
            return obj, "raw_scan"

    return None, None


def _coerce_script(output: Any) -> tuple[Script | None, str | None]:
    """Thin wrapper around _coerce_model kept for backward compatibility
    with existing callers/tests -- see that function's docstring for the
    three-tier fallback logic."""
    return _coerce_model(output, Script)


def _repair(script: Script, problems: list[str], agent: Any) -> tuple[Script | None, str | None]:
    """Ask the 校對 agent to fix specific cross-reference problems in an
    otherwise-complete script, without touching event/branch structure."""
    task = Task(
        description=(
            "以下劇本經程式檢查仍有交叉引用錯誤，請逐項修正後回傳完整 Script JSON：\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n修正原則：每個 dialogue 的 npc_id 必須指向已存在的 NPC；"
            "每個 branch 的 next_event_id 必須指向已存在的 event。"
            "不要新增或刪除事件、NPC 或分支，只修正指向錯誤的欄位。"
        ),
        expected_output="修正後、交叉引用完全正確的完整 Script JSON。",
        agent=agent,
        output_pydantic=Script,
    )
    output = task.execute_sync(agent=agent, context=script.model_dump_json())
    return _coerce_script(output)


def run_pipeline(
    requirement: str,
    verbose: bool = True,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    models: ModelChoice | None = None,
) -> Script:
    """Thin wrapper around run_pipeline_with_report() for callers (existing
    tests, prior scripts) that only need the Script, not the run report."""
    script, _report = run_pipeline_with_report(
        requirement, verbose=verbose, max_repair_attempts=max_repair_attempts, models=models
    )
    return script


def run_pipeline_with_report(
    requirement: str,
    verbose: bool = True,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    models: ModelChoice | None = None,
    on_step: Callable[[StepEvent], None] | None = None,
) -> tuple[Script, RunReport]:
    """Run the 編劇 -> 對話 -> 校對 sequential crew once for a given plain-text
    劇情需求 (story requirement), returning the final validated Script plus a
    RunReport (token spend, elapsed time, and -- critically -- whether the
    對話 agent's wuxia_corpus_search tool was ever actually called).

    `models` overrides which OpenRouter model id each agent role uses --
    default None falls back to the env-configured ModelChoice() (today's
    behavior). Passing an explicit ModelChoice is what lets
    scripts/eval_generation.py A/B different per-agent splits within one
    process, without editing .env and restarting.

    `on_step`, if given, is called synchronously on this same thread with a
    StepEvent each time a task finishes (crewai's task_callback; verified to
    fire exactly once per task) plus a couple of synthetic markers before
    kickoff and around repair attempts. It is wired into `Crew(...)` only
    when non-None so existing callers get a byte-identical Crew object (and
    avoid a pydantic "function callbacks cannot be serialized" warning).
    Exceptions from `on_step` are NOT swallowed -- that's the mechanism a
    caller (e.g. the Stage 3 UI's cancel button) uses to abort a run.

    Cross-reference integrity (npc_id / next_event_id) is re-checked in
    Python via schema.validate_references() after the crew finishes, rather
    than trusted solely to the proofreader agent's own judgement -- that
    keeps the guarantee deterministic regardless of config.LLM_BACKEND. If
    problems are found, the proofreader gets up to `max_repair_attempts`
    targeted repair passes (see _repair) before this raises PipelineError.
    """
    reset_stats()
    start = time.monotonic()
    models = models or ModelChoice()
    report = RunReport(
        requirement=requirement,
        model_writer=models.writer,
        model_dialogue=models.dialogue,
        model_proof=models.proof,
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

    writer = make_writer_agent(verbose=verbose, models=models)
    dialoguer = make_dialogue_agent(verbose=verbose, models=models)
    proofreader = make_proofreader_agent(verbose=verbose, models=models)

    writer_task = make_writer_task(requirement, writer)
    dialogue_task = make_dialogue_task(dialoguer, writer_task)
    proofread_task = make_proofread_task(proofreader, dialogue_task)

    def _on_task_done(task_output: Any) -> None:
        _emit("task", getattr(task_output, "agent", ""), "任務完成")

    crew_kwargs: dict[str, Any] = {}
    if on_step is not None:
        crew_kwargs["task_callback"] = _on_task_done

    crew = Crew(
        agents=[writer, dialoguer, proofreader],
        tasks=[writer_task, dialogue_task, proofread_task],
        process=Process.sequential,
        verbose=verbose,
        **crew_kwargs,
    )

    _emit("phase", "", "開始執行")

    try:
        crew_output = crew.kickoff()
    except Exception as exc:
        _finalize_report()
        raise PipelineError(
            f"crew 執行失敗（{type(exc).__name__}）：{exc}", report=report
        ) from exc

    if crew_output.token_usage is not None:
        report.token_usage = crew_output.token_usage.model_dump()

    script, coerced_from = _coerce_script(crew_output)
    report.coerced_from = coerced_from
    if script is None:
        _finalize_report()
        preview = (crew_output.raw or "")[:500]
        raise PipelineError(
            "校對 agent 未能輸出符合 Script schema 的結果，原始輸出前 500 字：\n"
            + preview,
            report=report,
        )

    problems = validate_references(script)
    best_script, best_problems = script, problems

    attempts = 0
    while best_problems and attempts < max_repair_attempts:
        attempts += 1
        _emit("phase", "校對", f"修補嘗試 {attempts}/{max_repair_attempts}")
        # Unlike crew.kickoff() above, _repair()'s task.execute_sync() call
        # isn't wrapped by CrewAI itself -- a provider error here (e.g. a
        # persistent 400/429 from an unstable OpenRouter route) used to
        # propagate uncaught and crash the whole process instead of just
        # this run. Treat a failed repair attempt the same as a repair that
        # produced no valid Script: skip it and let the loop either retry
        # or fall through to the final PipelineError with whatever the best
        # script so far was.
        try:
            repaired, repaired_from = _repair(best_script, best_problems, proofreader)
        except Exception:
            continue
        if repaired is None:
            continue
        repaired_problems = validate_references(repaired)
        # Only keep the repair if it's no worse than what we had -- a repair
        # pass that introduces new problems isn't an improvement.
        if len(repaired_problems) <= len(best_problems):
            best_script, best_problems = repaired, repaired_problems
            report.coerced_from = repaired_from
    report.repair_attempts = attempts

    _finalize_report()

    if best_problems:
        raise PipelineError(
            "校對後的劇本仍有交叉引用錯誤：\n" + "\n".join(best_problems),
            report=report,
        )

    return best_script, report


def run_layered_pipeline(
    requirement: str,
    verbose: bool = True,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    models: ModelChoice | None = None,
    on_step: Callable[[StepEvent], None] | None = None,
) -> tuple[Script, RunReport]:
    """Run the layered 拆書 -> 排場 -> 逐場寫戲 -> 校對 pipeline (BiXiaScribe
    重構 Phase 2) once for a given plain-text 劇情需求, returning the final
    validated Script plus a RunReport -- same return shape as
    run_pipeline_with_report(), so callers can switch between the two
    without touching downstream code.

    Unlike the legacy pipeline (a single sequential Crew), each stage here
    is invoked as its own Task.execute_sync() call, sequentially: extractor
    -> beat_expander -> one scene_writer call per beat. This Phase does
    scenes strictly one at a time (parallel scene calls come in a later
    phase); it's what gives this pipeline a natural per-stage/per-scene
    checkpoint granularity that the legacy pipeline's 3-Task-per-run doesn't
    have -- see the module docstring notes on crewai's step_callback
    limitation.

    A scene_writer's returned Event.id is not trusted: it's always
    overwritten with the beat's own id (Event.id == Beat.id in this
    pipeline), since a model's own id choice can't be relied on to avoid
    collisions once scenes are generated in parallel (a later phase).

    Cross-reference integrity is re-checked and repaired exactly like
    run_pipeline_with_report() (see _repair) -- this pipeline doesn't
    reinvent that safety net, it reuses it on the assembled Script.

    As of BiXiaScribe 重構 Phase 3, this is a thin wrapper around
    crew/orchestrator.py::run_layered() -- a fresh, uncheckpointed run_id is
    used every call, so behavior/signature/return shape are unchanged from
    Phase 2. Callers that want resumable, checkpointed runs (crash recovery
    without re-spending tokens on already-completed stages) should call
    orchestrator.run_layered() directly with an explicit run_id instead.
    Imported lazily (inside this function, not at module top) to avoid a
    circular import: orchestrator.py imports several names from this module.
    """
    from .orchestrator import run_layered

    return run_layered(
        requirement,
        models=models,
        verbose=verbose,
        on_step=on_step,
        max_repair_attempts=max_repair_attempts,
    )
