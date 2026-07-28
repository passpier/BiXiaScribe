"""Sequential Crew wiring: 編劇 -> 對話 -> 校對, producing one validated
Script. This is the Stage 2 entry point -- scripts/generate_script.py and
tests/test_crew_pipeline.py both call run_pipeline()."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from crewai import Crew, Process, Task

from .. import config
from ..llm import ModelChoice
from ..schema import Script, parse_script_json, validate_references
from .agents import make_dialogue_agent, make_proofreader_agent, make_writer_agent
from .tasks import make_dialogue_task, make_proofread_task, make_writer_task
from .tools import get_stats, reset_stats

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
        }


def _coerce_script(output: Any) -> tuple[Script | None, str | None]:
    """Extract a Script from a CrewOutput or TaskOutput, in decreasing order
    of trust: the pydantic object CrewAI's output_pydantic coercion already
    produced, then its json_dict, then a salvage scan of the raw text. Real
    models frequently wrap their JSON in explanatory prose that trips up
    CrewAI's own coercion even though the JSON itself is fine -- see
    schema.parse_script_json for why the raw scan validates against the
    schema instead of just keyword-matching.

    Returns (script, source) where source identifies which level produced
    it ("pydantic" / "json_dict" / "raw_scan"), or (None, None) if nothing
    validated -- source is reported in RunReport.coerced_from since a real
    model landing on "raw_scan" is itself a signal worth seeing.
    """
    if isinstance(output.pydantic, Script):
        return output.pydantic, "pydantic"

    if output.json_dict is not None:
        try:
            return Script.model_validate(output.json_dict), "json_dict"
        except Exception:
            pass

    if output.raw:
        script = parse_script_json(output.raw)
        if script is not None:
            return script, "raw_scan"

    return None, None


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

    crew = Crew(
        agents=[writer, dialoguer, proofreader],
        tasks=[writer_task, dialogue_task, proofread_task],
        process=Process.sequential,
        verbose=verbose,
    )

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
        repaired, repaired_from = _repair(best_script, best_problems, proofreader)
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
