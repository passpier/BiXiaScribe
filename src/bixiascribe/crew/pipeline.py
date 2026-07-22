"""Sequential Crew wiring: 編劇 -> 對話 -> 校對, producing one validated
Script. This is the Stage 2 entry point -- scripts/generate_script.py and
tests/test_crew_pipeline.py both call run_pipeline()."""
from __future__ import annotations

from crewai import Crew, Process

from ..schema import Script, validate_references
from .agents import make_dialogue_agent, make_proofreader_agent, make_writer_agent
from .tasks import make_dialogue_task, make_proofread_task, make_writer_task


class PipelineError(RuntimeError):
    """Raised when the crew doesn't produce a schema-valid, reference-clean
    Script (e.g. the proofreader's final output fails validate_references())."""


def run_pipeline(requirement: str, verbose: bool = True) -> Script:
    """Run the 編劇 -> 對話 -> 校對 sequential crew once for a given plain-text
    劇情需求 (story requirement), returning the final validated Script.

    Cross-reference integrity (npc_id / next_event_id) is re-checked in
    Python via schema.validate_references() after the crew finishes, rather
    than trusted solely to the proofreader agent's own judgement -- that
    keeps the guarantee deterministic regardless of config.LLM_BACKEND.
    """
    writer = make_writer_agent()
    dialoguer = make_dialogue_agent()
    proofreader = make_proofreader_agent()

    writer_task = make_writer_task(requirement, writer)
    dialogue_task = make_dialogue_task(dialoguer, writer_task)
    proofread_task = make_proofread_task(proofreader, dialogue_task)

    crew = Crew(
        agents=[writer, dialoguer, proofreader],
        tasks=[writer_task, dialogue_task, proofread_task],
        process=Process.sequential,
        verbose=verbose,
    )

    crew_output = crew.kickoff()

    script = crew_output.pydantic
    if script is None or not isinstance(script, Script):
        raise PipelineError("校對 agent 未能輸出符合 Script schema 的結果")

    problems = validate_references(script)
    if problems:
        raise PipelineError(
            "校對後的劇本仍有交叉引用錯誤：\n" + "\n".join(problems)
        )

    return script
