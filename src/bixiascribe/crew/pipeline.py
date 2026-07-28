"""Sequential Crew wiring: 編劇 -> 對話 -> 校對, producing one validated
Script. This is the Stage 2 entry point -- scripts/generate_script.py and
tests/test_crew_pipeline.py both call run_pipeline()."""
from __future__ import annotations

from typing import Any

from crewai import Crew, Process, Task

from ..schema import Script, parse_script_json, validate_references
from .agents import make_dialogue_agent, make_proofreader_agent, make_writer_agent
from .tasks import make_dialogue_task, make_proofread_task, make_writer_task

# Cross-reference problems (dangling npc_id / next_event_id) are handed back
# to the 校對 agent for a targeted repair pass this many times before giving
# up. Re-running just the proofread task (not the whole crew) is deliberate:
# the writer/dialogue output isn't in question, only structural references,
# so a full re-run would triple the token cost for no benefit.
MAX_REPAIR_ATTEMPTS = 2


class PipelineError(RuntimeError):
    """Raised when the crew doesn't produce a schema-valid, reference-clean
    Script (e.g. the proofreader's final output fails validate_references()),
    or when crew.kickoff() itself fails (provider errors, timeouts, etc.)."""


def _coerce_script(output: Any) -> Script | None:
    """Extract a Script from a CrewOutput or TaskOutput, in decreasing order
    of trust: the pydantic object CrewAI's output_pydantic coercion already
    produced, then its json_dict, then a salvage scan of the raw text. Real
    models frequently wrap their JSON in explanatory prose that trips up
    CrewAI's own coercion even though the JSON itself is fine -- see
    schema.parse_script_json for why the raw scan validates against the
    schema instead of just keyword-matching.
    """
    if isinstance(output.pydantic, Script):
        return output.pydantic

    if output.json_dict is not None:
        try:
            return Script.model_validate(output.json_dict)
        except Exception:
            pass

    if output.raw:
        return parse_script_json(output.raw)

    return None


def _repair(
    script: Script, problems: list[str], agent: Any, verbose: bool
) -> Script | None:
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
) -> Script:
    """Run the 編劇 -> 對話 -> 校對 sequential crew once for a given plain-text
    劇情需求 (story requirement), returning the final validated Script.

    Cross-reference integrity (npc_id / next_event_id) is re-checked in
    Python via schema.validate_references() after the crew finishes, rather
    than trusted solely to the proofreader agent's own judgement -- that
    keeps the guarantee deterministic regardless of config.LLM_BACKEND. If
    problems are found, the proofreader gets up to `max_repair_attempts`
    targeted repair passes (see _repair) before this raises PipelineError.
    """
    writer = make_writer_agent(verbose=verbose)
    dialoguer = make_dialogue_agent(verbose=verbose)
    proofreader = make_proofreader_agent(verbose=verbose)

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
        raise PipelineError(f"crew 執行失敗（{type(exc).__name__}）：{exc}") from exc

    script = _coerce_script(crew_output)
    if script is None:
        preview = (crew_output.raw or "")[:500]
        raise PipelineError(
            "校對 agent 未能輸出符合 Script schema 的結果，原始輸出前 500 字：\n"
            + preview
        )

    problems = validate_references(script)
    best_script, best_problems = script, problems

    attempts = 0
    while best_problems and attempts < max_repair_attempts:
        attempts += 1
        repaired = _repair(best_script, best_problems, proofreader, verbose)
        if repaired is None:
            continue
        repaired_problems = validate_references(repaired)
        # Only keep the repair if it's no worse than what we had -- a repair
        # pass that introduces new problems isn't an improvement.
        if len(repaired_problems) <= len(best_problems):
            best_script, best_problems = repaired, repaired_problems

    if best_problems:
        raise PipelineError(
            "校對後的劇本仍有交叉引用錯誤：\n" + "\n".join(best_problems)
        )

    return best_script
