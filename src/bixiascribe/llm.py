"""LLM backend selection for the Stage 2 CrewAI pipeline.

Mirrors the EMBED_BACKEND fake/real split in embedding.py: config.LLM_BACKEND
controls whether agents call out to OpenRouter for real generation, or run
against a deterministic offline FakeLLM. The fake backend is what
tests/test_crew_pipeline.py uses so the three-agent pipeline can be verified
end-to-end with no API key, no network access, and no token cost.

Model calls are not tied to any single provider's SDK: the real path uses
crewai's LLM class pointed at OpenRouter's OpenAI-compatible endpoint (via
litellm's "openrouter/" model prefix), so swapping models is just an env var
change (config.LLM_MODEL / LLM_MODEL_WRITER / LLM_MODEL_DIALOGUE /
LLM_MODEL_PROOF), never a code change.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from crewai.llms.base_llm import BaseLLM

from . import config
from .schema import NPC, Branch, DialogueLine, Event, Script, Trigger, Variable

if TYPE_CHECKING:
    from pydantic import BaseModel

# Agent role names. Defined here (rather than in crew/agents.py) so both
# agents.py (which constructs Agent(role=...)) and FakeLLM (which branches
# on from_agent.role to decide what to fake) can import them without a
# circular dependency between llm.py and crew/agents.py.
ROLE_WRITER = "說書人・鐵筆生"
ROLE_DIALOGUE = "江湖代言人・柳三娘"
ROLE_PROOFREADER = "總編・青衫客"


def build_llm(role: str):
    """Return the LLM instance an Agent of `role` should use, honoring
    config.LLM_BACKEND. `role` must be one of the ROLE_* constants above --
    it picks which per-agent model env var applies on the real backend, and
    which canned behavior FakeLLM produces on the fake backend."""
    if config.LLM_BACKEND == "fake":
        return FakeLLM(model=f"fake/{role}")

    model_by_role = {
        ROLE_WRITER: config.LLM_MODEL_WRITER,
        ROLE_DIALOGUE: config.LLM_MODEL_DIALOGUE,
        ROLE_PROOFREADER: config.LLM_MODEL_PROOF,
    }
    if role not in model_by_role:
        raise ValueError(f"Unknown agent role: {role!r}")

    from crewai import LLM

    return LLM(
        model=model_by_role[role],
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.require_openrouter_key(),
    )


def _extract_script_json(messages: str | list[dict[str, Any]]) -> dict[str, Any] | None:
    """Scan the prompt so far for the most recent Script-shaped JSON object
    (e.g. a previous task's output, embedded as context by CrewAI) and parse
    it. Uses JSONDecoder.raw_decode at every '{' rather than a regex, so it
    tolerates surrounding prose/markdown instead of requiring the JSON to be
    the entire message. Returns None if nothing Script-shaped is found."""
    if isinstance(messages, str):
        text = messages
    else:
        text = "\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in messages
        )

    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "events" in obj and "npcs" in obj:
            best = obj  # keep scanning -- a later match is more recent context
    return best


def _fake_writer_script() -> Script:
    """A hand-written, schema-valid skeleton (dialogue left empty) standing
    in for what the 編劇 agent would produce from a real requirement."""
    return Script(
        title="試煉：血衣門疑雲",
        premise="一名少林俗家弟子奉命下山，追查一樁滅門血案背後的血衣門餘孽。",
        variables=[
            Variable(
                id="var_suspicion", name="懷疑度", initial=0,
                description="主角對血衣門的懷疑程度",
            ),
        ],
        npcs=[
            NPC(
                id="npc_master", name="了塵長老", identity="少林戒律院長老",
                personality="嚴厲重規矩", speech_style="佛偈式短句，常引戒律",
            ),
            NPC(
                id="npc_widow", name="柳寡婦", identity="血案倖存者",
                personality="悲憤壓抑", speech_style="市井口語，夾雜哭腔",
            ),
        ],
        events=[
            Event(
                id="evt_depart", title="下山", location="少林寺山門",
                summary="主角領命下山，了塵長老交代查案禁忌。",
                triggers=[Trigger(type="on_enter", condition="story_flag==start")],
                dialogue=[],
                branches=[
                    Branch(id="br_go", choice_text="領命下山", next_event_id="evt_village"),
                ],
            ),
            Event(
                id="evt_village", title="血案現場", location="山腳村落",
                summary="主角詢問柳寡婦滅門血案經過，發現血衣門線索。",
                triggers=[Trigger(type="on_enter", condition="")],
                dialogue=[],
                branches=[],
            ),
        ],
    )


def _fake_fill_dialogue(prior: dict[str, Any] | None) -> Script:
    """Stand-in for the 對話 agent: take the writer's script (recovered from
    context, or invented fresh if that failed) and fill each empty
    Event.dialogue with a canned line from its first NPC."""
    script = Script.model_validate(prior) if prior is not None else _fake_writer_script()

    npc_id = script.npcs[0].id if script.npcs else None
    if npc_id is not None:
        for event in script.events:
            if event.dialogue:
                continue
            event.dialogue = [
                DialogueLine(
                    npc_id=npc_id,
                    line=f"（於{event.location}）此事，且聽我道來——{event.summary}",
                    emotion="凝重",
                )
            ]
    return script


def _fake_proofread(prior: dict[str, Any] | None) -> Script:
    """Stand-in for the 校對 agent: pass the dialogue-filled script through
    unchanged (real cross-reference validation happens in Python via
    schema.validate_references(), called by crew/pipeline.py after kickoff)."""
    if prior is None:
        return _fake_fill_dialogue(None)
    return Script.model_validate(prior)


class FakeLLM(BaseLLM):
    """Deterministic, offline stand-in for crewai's real LLM classes.

    Selected via LLM_BACKEND=fake. Rather than calling out to a model, it
    looks at which agent is asking (from_agent.role) and either invents a
    first-draft Script (writer), or recovers the Script produced so far from
    the prompt context and fills in the next piece (dialogue agent fills
    dialogue; proofreader passes it through). This is what lets
    tests/test_crew_pipeline.py exercise the full three-agent wiring and
    schema validation without an API key, network access, or token cost.
    """

    def __init__(self, **data: Any) -> None:
        data.setdefault("model", "fake")
        super().__init__(**data)

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        role = getattr(from_agent, "role", "") or ""

        if role == ROLE_WRITER:
            script = _fake_writer_script()
        elif role == ROLE_DIALOGUE:
            script = _fake_fill_dialogue(_extract_script_json(messages))
        elif role == ROLE_PROOFREADER:
            script = _fake_proofread(_extract_script_json(messages))
        else:
            # Unexpected role -- return something harmless instead of
            # crashing the executor loop.
            return "Final Answer: {}"

        # Tool-bearing agents (the dialogue agent) never get a response_model
        # from the executor (see crew_agent_executor._invoke_loop_react),
        # so they must produce ReAct-style "Final Answer:" text instead of a
        # structured object.
        if response_model is not None:
            return script
        return f"Final Answer: {script.model_dump_json()}"

    def supports_function_calling(self) -> bool:
        return False
