"""Sequential Crew wiring: 編劇 -> 對話 -> 校對, producing one validated
Script. This is the legacy pipeline entry point. run_pipeline_with_report()
is what real callers use (scripts/generate_script.py's legacy branch,
src/bixiascribe/generation.py's "legacy" mode); run_pipeline() is a thin
Script-only wrapper kept for tests/test_crew_pipeline.py and other existing
callers that don't need the RunReport. The layered alternative
(run_layered() in crew/orchestrator.py) lives alongside this module rather
than replacing it -- see CLAUDE.md's "Script generation" section."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from crewai import Crew, Process, Task
from pydantic import BaseModel

from .. import config, wire
from ..llm import ModelChoice
from ..schema import Script, parse_model_json, validate_references
from . import guardrails, normalize
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

# --- Token usage accounting ------------------------------------------------
#
# Lives here (not crew/orchestrator.py, which imports from this module) so
# both the legacy and layered pipelines can share one implementation without
# a circular import -- orchestrator.py already depends on pipeline.py at
# module load time, so the dependency has to run this direction.
#
# crewai 1.15.5's TaskOutput carries no usage info, and per-agent LLM
# instances (the real `LLM` and our own llm.py::FakeLLM) expose
# get_token_usage_summary(), cumulative for that LLM instance's lifetime.
# Each make_*_agent() call below builds a fresh agent+LLM, so a before/after
# delta on that instance is exactly what that agent spent.

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
    # Directly-measured corpus text injected into prompts by successful
    # wuxia_corpus_search calls this run (crew/tools.py::RetrievalStats).
    # This is what CLAUDE.md's cost analysis previously had to infer from a
    # regression of prompt_tokens against retrieval_calls -- see that
    # module's RetrievalStats docstring for why only successful lookups
    # count. 0 for every existing JSONL row logged before this field
    # existed, same "field didn't exist yet" convention as the others below.
    retrieval_chars_returned: int = 0
    retrieval_chunks_returned: int = 0
    repair_attempts: int = 0
    coerced_from: str | None = None
    # Layered-pipeline fields. All default to values that make a legacy
    # run_pipeline_with_report() row indistinguishable from before -- only
    # crew/orchestrator.py's run_layered() (the real, checkpointed layered
    # entry point) sets them. run_layered_pipeline() below is a thin,
    # uncheckpointed shim over run_layered() kept for
    # tests/test_crew_layered_pipeline.py; it has no production callers.
    mode: str = "legacy"
    model_extractor: str = ""
    model_beat_expander: str = ""
    model_scene_writer: str = ""
    scenes_generated: int = 0
    # Context-compression quality-regression fields. Only run_layered() sets
    # these -- a legacy run leaves both at their defaults.
    # session_doc_max_tokens mirrors run_layered()'s own argument (None =
    # config.SESSION_DOC_MAX_TOKENS was used, 0 = trimming was disabled);
    # session_doc_omitted_total is the sum of every SessionDocument's
    # omitted_scene_count across the run -- the manipulation check for the
    # compressed-vs-untrimmed experiment (see
    # eval/context_compression_variants.json): without it, "arm A never
    # actually trimmed anything" is invisible in the JSONL data.
    session_doc_max_tokens: int | None = None
    session_doc_omitted_total: int = 0
    # Causal-consistency fields. Only run_layered() sets these -- a legacy
    # run leaves all three at their defaults, same convention as the fields
    # above.
    causal_validation: str = ""
    causal_problems: list[str] = field(default_factory=list)
    causal_repair_attempts: int = 0
    # Script-length knob (config.SCRIPT_LENGTH / Variant.script_length /
    # --script-length; see crew/tasks.py::_LENGTH_TARGETS) that produced this
    # run -- self-describing, like session_doc_max_tokens above, so a JSONL
    # row doesn't need cross-referencing against whatever the env var was at
    # run time. "short" for both legacy and layered runs made before this
    # field existed (dataclass default), matching the knob's own default.
    script_length: str = "short"
    # Cost accounting (pricing.py). cost_usd/cost_basis are computed by
    # generation.build_run_row() from token_usage(_by_role), not set here --
    # RunReport carries the raw usage; pricing is a presentation-layer
    # concern kept out of the crew/ package so pricing.py stays importable
    # without crewai. token_usage_by_role is a role-keyed usage dict: for
    # layered runs, populated by run_layered()'s per-stage usage accounting;
    # for legacy runs, populated by run_pipeline_with_report()'s per-agent
    # LLM-instance before/after delta (see this module's "Token usage
    # accounting" section above) so a legacy run with mixed per-role models
    # (e.g. eval/model_variants.json's long-prose) doesn't have to fall back
    # to pricing.py's uniform-model estimate across the whole run-wide
    # total. Empty only for legacy JSONL rows logged before this existed.
    token_usage_by_role: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Whether the dialogue/scene_writer agent(s) had wuxia_corpus_search at
    # all this run (config.RETRIEVAL_ENABLED / Variant.use_retrieval /
    # --no-retrieval). True (today's only behavior, pre-knob) for every
    # existing JSONL row read back without this field. When False,
    # retrieval_calls is expected to be 0 -- ui/app.py checks this field
    # before treating retrieval_calls == 0 as the "tool-calling silently
    # never fired" warning sign it normally is.
    retrieval_enabled: bool = True
    # Whether crew/guardrails.py's RPG-shape checks were wired onto the
    # writer/extract/scene_write tasks this run (config.GUARDRAILS_ENABLED,
    # forced off under LLM_BACKEND=fake -- see tasks.py::_guardrails_active).
    # False for every existing JSONL row logged before this field existed
    # (guardrails didn't exist then), same convention as retrieval_enabled.
    guardrails_enabled: bool = False
    guardrail_max_retries: int = 0
    # crew/normalize.py::normalize_script()'s mechanical fixes (dangling
    # next_event_id / chapter_id backfill / cleared cosmetic ids), applied
    # before validate_references() so the existing repair loops only ever
    # see problems that actually need a model. Empty for every run where
    # nothing needed fixing, and for JSONL rows logged before this existed.
    normalize_notes: list[str] = field(default_factory=list)
    # Report-only findings from the nine GMUD guardrail checks
    # (check_choice_quality/check_delayed_payoff/etc., see
    # crew/guardrails.py) that are NOT wired as Task guardrails -- retrying
    # against them was measured to make the pass rate worse (10-28 extra
    # findings per already-generated script), so they're surfaced here for
    # visibility instead of gating the run. Empty for JSONL rows logged
    # before this existed.
    quality_problems: list[str] = field(default_factory=list)

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
            "retrieval_chars_returned": self.retrieval_chars_returned,
            "retrieval_chunks_returned": self.retrieval_chunks_returned,
            "repair_attempts": self.repair_attempts,
            "coerced_from": self.coerced_from,
            "mode": self.mode,
            "model_extractor": self.model_extractor,
            "model_beat_expander": self.model_beat_expander,
            "model_scene_writer": self.model_scene_writer,
            "scenes_generated": self.scenes_generated,
            "session_doc_max_tokens": self.session_doc_max_tokens,
            "session_doc_omitted_total": self.session_doc_omitted_total,
            "causal_validation": self.causal_validation,
            "causal_problems": self.causal_problems,
            "causal_repair_attempts": self.causal_repair_attempts,
            "script_length": self.script_length,
            "token_usage_by_role": self.token_usage_by_role,
            "retrieval_enabled": self.retrieval_enabled,
            "guardrails_enabled": self.guardrails_enabled,
            "guardrail_max_retries": self.guardrail_max_retries,
            "normalize_notes": self.normalize_notes,
            "quality_problems": self.quality_problems,
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
    ("pydantic" / "json_dict" / "raw_scan" / "lenient_mirror"), or
    (None, None) if nothing validated -- source is reported in
    RunReport.coerced_from since a real model landing on "raw_scan" (or
    especially "lenient_mirror", meaning the strict wire schema's own
    required-field set genuinely wasn't satisfied) is itself a signal
    worth seeing.
    """
    if isinstance(output.pydantic, model_cls):
        return output.pydantic, "pydantic"

    # Tasks now set output_pydantic=wire.lenient_mirror(model_cls) (see
    # crew/tasks.py), so a well-formed response lands here instead of the
    # branch above -- every field on the mirror has a default, so this
    # coercion itself cannot fail on a missing field the way the strict
    # model's own structured-output parsing could.
    if isinstance(output.pydantic, wire.lenient_mirror(model_cls)):
        return wire.to_strict(output.pydantic, model_cls), "lenient_mirror"

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
            "每個 branch 的 next_event_id 必須指向已存在的 event；"
            "faction 的 relations、npc 的 faction_id、stat_threshold 的 "
            "stat_id/unlocks_id、event 的 chapter_id/region_id/"
            "sub_location_id/clue_ids、skill check 的 stat_id/"
            "success_next_event_id/failure_branch_id/item_bypass_id、"
            "branch 的 payoff_chapter_id/converges_to_event_id、chapter 的 "
            "converge_event_id/event_ids/clue_ids、clue 的 "
            "found_in_event_id、ending 的 stat_conditions/"
            "required_branch_ids、truth 的 progressive reveal 的 "
            "reveal_chapter_id/reveal_event_id、player 的 token_item_id，"
            "都必須指向已存在的對應項目。"
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
    script_length: str = "short",
) -> Script:
    """Thin wrapper around run_pipeline_with_report() for callers (existing
    tests, prior scripts) that only need the Script, not the run report."""
    script, _report = run_pipeline_with_report(
        requirement,
        verbose=verbose,
        max_repair_attempts=max_repair_attempts,
        models=models,
        script_length=script_length,
    )
    return script


def run_pipeline_with_report(
    requirement: str,
    verbose: bool = True,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    models: ModelChoice | None = None,
    on_step: Callable[[StepEvent], None] | None = None,
    script_length: str = "short",
    use_retrieval: bool | None = None,
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

    `use_retrieval` (default None -> config.RETRIEVAL_ENABLED) turns the
    對話 agent's wuxia_corpus_search tool -- and its prompt instructions to
    use it -- on or off for this run. Off means no Chroma/embedding-model
    access is needed at all, and retrieval_calls in the returned RunReport
    is expected to stay 0 (see RunReport.retrieval_enabled).
    """
    resolved_use_retrieval = (
        use_retrieval if use_retrieval is not None else config.RETRIEVAL_ENABLED
    )
    reset_stats()
    start = time.monotonic()
    models = models or ModelChoice()
    report = RunReport(
        requirement=requirement,
        model_writer=models.writer,
        model_dialogue=models.dialogue,
        model_proof=models.proof,
        script_length=script_length,
        retrieval_enabled=resolved_use_retrieval,
        guardrails_enabled=config.GUARDRAILS_ENABLED and config.LLM_BACKEND != "fake",
        guardrail_max_retries=config.GUARDRAIL_MAX_RETRIES,
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
        report.retrieval_chars_returned = stats.chars_returned
        report.retrieval_chunks_returned = stats.chunks_returned
        # Per-role token attribution (writer/dialogue/proof), same
        # before/after-delta technique crew/orchestrator.py uses for the
        # layered pipeline -- see that module's "Token usage accounting"
        # section. Each agent below holds its own fresh LLM instance for
        # this run, so `_before_usage[role]` (captured pre-kickoff, when
        # every instance is fresh) vs. a post-kickoff/post-repair snapshot
        # here is exactly that role's total spend, including any 校對
        # repair-pass calls (_repair() reuses the same `proofreader`
        # instance, so its usage accumulates into the same LLM object).
        # Without this, pricing.py can't compute an exact "by_role" cost for
        # a legacy run with mixed per-role models -- it falls back to a
        # uniform-model estimate across the whole run-wide total instead.
        for role, agent in (("writer", writer), ("dialogue", dialoguer), ("proof", proofreader)):
            delta = _usage_delta(_before_usage.get(role), _llm_usage(agent))
            if delta:
                report.token_usage_by_role[role] = delta

    writer = make_writer_agent(verbose=verbose, models=models)
    dialoguer = make_dialogue_agent(
        verbose=verbose, models=models, use_retrieval=resolved_use_retrieval
    )
    proofreader = make_proofreader_agent(verbose=verbose, models=models)
    _before_usage: dict[str, dict[str, int] | None] = {
        "writer": _llm_usage(writer),
        "dialogue": _llm_usage(dialoguer),
        "proof": _llm_usage(proofreader),
    }

    writer_task = make_writer_task(requirement, writer, script_length=script_length)
    dialogue_task = make_dialogue_task(
        dialoguer, writer_task, script_length=script_length, use_retrieval=resolved_use_retrieval
    )
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

    script, report.normalize_notes = normalize.normalize_script(script)

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

    report.quality_problems = guardrails.collect_quality_problems(best_script)

    return best_script, report


def run_layered_pipeline(
    requirement: str,
    verbose: bool = True,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    models: ModelChoice | None = None,
    on_step: Callable[[StepEvent], None] | None = None,
    script_length: str = "short",
) -> tuple[Script, RunReport]:
    """Run the layered 拆書 -> 排場 -> 逐場寫戲 -> 校對 pipeline once for a
    given plain-text 劇情需求, returning the final validated Script plus a
    RunReport -- same return shape as run_pipeline_with_report(), so callers
    can switch between the two without touching downstream code.

    Unlike the legacy pipeline (a single sequential Crew), each stage here
    is invoked as its own Task.execute_sync() call, sequentially: extractor
    -> beat_expander -> one scene_writer call per beat. It's what gives this
    pipeline a natural per-stage/per-scene checkpoint granularity that the
    legacy pipeline's 3-Task-per-run doesn't have -- see the module
    docstring notes on crewai's step_callback limitation.

    A scene_writer's returned Event.id is not trusted: it's always
    overwritten with the beat's own id (Event.id == Beat.id in this
    pipeline), since a model's own id choice can't be relied on to avoid
    collisions once scenes are generated in parallel.

    Cross-reference integrity is re-checked and repaired exactly like
    run_pipeline_with_report() (see _repair) -- this pipeline doesn't
    reinvent that safety net, it reuses it on the assembled Script.

    This is a thin wrapper around crew/orchestrator.py::run_layered() -- a
    fresh, uncheckpointed run_id is used every call. Callers that want
    resumable, checkpointed runs (crash recovery without re-spending tokens
    on already-completed stages) should call orchestrator.run_layered()
    directly with an explicit run_id instead. Imported lazily (inside this
    function, not at module top) to avoid a circular import: orchestrator.py
    imports several names from this module.

    Kept only for tests/test_crew_layered_pipeline.py: it has zero
    production callers -- generation.py, scripts/generate_script.py, and
    scripts/eval_generation.py all call orchestrator.run_layered() directly,
    since every real caller wants the run_id/gate/concurrency features this
    wrapper doesn't forward. Not removed, since deleting it would mean
    rewriting that test file for no functional gain; new code should prefer
    orchestrator.run_layered().
    """
    from .orchestrator import run_layered

    return run_layered(
        requirement,
        models=models,
        verbose=verbose,
        on_step=on_step,
        max_repair_attempts=max_repair_attempts,
        script_length=script_length,
    )
