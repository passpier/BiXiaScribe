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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from crewai.llms.base_llm import BaseLLM

from . import config
from .schema import (
    NPC,
    Beat,
    BeatSheet,
    Branch,
    Chapter,
    Clue,
    DialogueLine,
    EffectOp,
    Ending,
    Event,
    ExtractionResult,
    Faction,
    FactionRelation,
    Item,
    Outline,
    PlayerCharacter,
    ProgressiveReveal,
    Quest,
    Region,
    Script,
    SkillCheck,
    StatCondition,
    StatThreshold,
    SubLocation,
    Trigger,
    TruthLayer,
    Variable,
    parse_model_json,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

# Agent role names. Defined here (rather than in crew/agents.py) so both
# agents.py (which constructs Agent(role=...)) and FakeLLM (which branches
# on from_agent.role to decide what to fake) can import them without a
# circular dependency between llm.py and crew/agents.py.
ROLE_WRITER = "說書人・鐵筆生"
ROLE_DIALOGUE = "江湖代言人・柳三娘"
ROLE_PROOFREADER = "總編・青衫客"

# Layered-pipeline roles -- see
# crew/pipeline.py::run_layered_pipeline. Kept distinct from the three
# roles above so the legacy single-shot pipeline is unaffected.
ROLE_EXTRACTOR = "拆書人・辨物客"
ROLE_BEAT_EXPANDER = "說書人・排場先生"
ROLE_SCENE_WRITER = "江湖代言人・柳三娘（分場）"


@dataclass(frozen=True)
class ModelChoice:
    """Which OpenRouter model id each of the three agent roles should use.
    Defaults come straight from config.LLM_MODEL_* (today's env-driven
    behavior), but callers that want to A/B different per-agent splits in
    one process -- see scripts/eval_generation.py -- can construct their own
    and pass it through build_llm() / the crew/agents.py factories /
    run_pipeline_with_report() instead of editing .env and restarting."""

    writer: str = config.LLM_MODEL_WRITER
    dialogue: str = config.LLM_MODEL_DIALOGUE
    proof: str = config.LLM_MODEL_PROOF
    # Layered-pipeline roles. No dedicated config.LLM_MODEL_* env vars:
    # extractor/beat_expander reuse the writer's model split, scene_writer
    # reuses the dialogue model split, since both mirror those roles'
    # responsibilities closely enough that a dedicated knob isn't justified.
    # Adding dedicated env vars here was evaluated against real layered
    # runs and decided against -- nothing in that data suggested the reused
    # splits were a bottleneck, and a 6th independent model knob would widen
    # eval_generation.py's variant matrix without evidence it's needed.
    # Revisit once real usage says otherwise.
    extractor: str = config.LLM_MODEL_WRITER
    beat_expander: str = config.LLM_MODEL_WRITER
    scene_writer: str = config.LLM_MODEL_DIALOGUE

    def for_role(self, role: str) -> str:
        by_role = {
            ROLE_WRITER: self.writer,
            ROLE_DIALOGUE: self.dialogue,
            ROLE_PROOFREADER: self.proof,
            ROLE_EXTRACTOR: self.extractor,
            ROLE_BEAT_EXPANDER: self.beat_expander,
            ROLE_SCENE_WRITER: self.scene_writer,
        }
        if role not in by_role:
            raise ValueError(f"Unknown agent role: {role!r}")
        return by_role[role]


def build_llm(role: str, models: ModelChoice | None = None):
    """Return the LLM instance an Agent of `role` should use, honoring
    config.LLM_BACKEND. `role` must be one of the ROLE_* constants above --
    it picks which per-agent model applies on the real backend, and which
    canned behavior FakeLLM produces on the fake backend. `models` defaults
    to the env-configured ModelChoice() when omitted."""
    if config.LLM_BACKEND == "fake":
        return FakeLLM(model=f"fake/{role}")

    models = models or ModelChoice()

    from crewai import LLM

    llm_kwargs: dict[str, Any] = {
        "model": models.for_role(role),
        "base_url": config.OPENROUTER_BASE_URL,
        "api_key": config.require_openrouter_key(),
    }
    if config.LLM_MAX_TOKENS is not None:
        llm_kwargs["max_tokens"] = config.LLM_MAX_TOKENS

    # OpenRouter provider routing, forwarded as litellm's extra_body (crewai's
    # LLM spreads `additional_params` into the underlying completion() call --
    # see crewai/llm.py's `**self.additional_params`). Not verified against a
    # real OpenRouter response as of this writing -- if litellm's openrouter
    # integration doesn't honor extra_body the way this assumes, provider
    # pinning silently falls back to OpenRouter's default routing rather than
    # erroring, which is the safe failure direction for something this
    # optional. Both config knobs default unset, so this is a no-op today.
    provider_routing: dict[str, Any] = {}
    if config.LLM_PROVIDER_ONLY:
        provider_routing["only"] = config.LLM_PROVIDER_ONLY
    elif config.LLM_PROVIDER_SORT:
        provider_routing["sort"] = config.LLM_PROVIDER_SORT
    if provider_routing:
        llm_kwargs["additional_params"] = {"extra_body": {"provider": provider_routing}}

    return LLM(**llm_kwargs)


def _messages_to_text(messages: str | list[dict[str, Any]]) -> str:
    """Flatten a `call()` messages argument (a plain string, or a list of
    chat-message dicts) into one text blob for scanning."""
    if isinstance(messages, str):
        return messages
    return "\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in messages)


def _extract_script_json(messages: str | list[dict[str, Any]]) -> dict[str, Any] | None:
    """Scan the prompt so far for the most recent Script-shaped JSON object
    (e.g. a previous task's output, embedded as context by CrewAI) and parse
    it. Uses JSONDecoder.raw_decode at every '{' rather than a regex, so it
    tolerates surrounding prose/markdown instead of requiring the JSON to be
    the entire message. Returns None if nothing Script-shaped is found."""
    text = _messages_to_text(messages)

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


def _fake_factions() -> list[Faction]:
    return [
        Faction(
            id="f_shaolin", name="少林派", alignment="正道",
            relations=[FactionRelation(faction_id="f_bloodrobe", stance="敵對")],
        ),
        Faction(
            id="f_bloodrobe", name="血衣門", alignment="邪道",
            relations=[FactionRelation(faction_id="f_shaolin", stance="敵對")],
        ),
    ]


def _fake_regions() -> list[Region]:
    return [
        Region(
            id="r_village", name="山腳村落",
            sub_locations=[
                SubLocation(id="sl_teahouse", name="茶棚", function="打聽消息"),
                SubLocation(id="sl_clinic", name="醫館", function="療傷"),
            ],
        ),
    ]


def _fake_truth(reveal_chapter_id: str) -> TruthLayer:
    return TruthLayer(
        public=["血衣門曾在此地作亂"],
        progressive=[
            ProgressiveReveal(
                id="pr_1", fact="柳寡婦目睹真兇面容", reveal_chapter_id=reveal_chapter_id,
            ),
        ],
        hidden=["真兇其實是了塵長老的師弟"],
    )


def _fake_stat_thresholds() -> list[StatThreshold]:
    return [
        StatThreshold(
            id="th_rep", stat_id="rep", min_value=0, max_value=100,
            unlocks_kind="ending", unlocks_id="end_justice", description="聲望決定結局走向",
        ),
    ]


def _fake_endings() -> list[Ending]:
    return [
        Ending(
            id="end_justice", name="伸張正義", description="聲望達標，揭穿真兇",
            stat_conditions=[StatCondition(stat_id="rep", min_value=50)],
            required_branch_ids=["br_go"],
        ),
    ]


def _fake_writer_script() -> Script:
    """A hand-written, schema-valid skeleton (dialogue left empty) standing
    in for what the 編劇 agent would produce from a real requirement --
    including the GMUD world frame (factions/regions/truth/stat_thresholds/
    chapters/clues/endings) so offline tests exercise the full shape."""
    return Script(
        title="試煉：血衣門疑雲",
        premise="一名少林俗家弟子奉命下山，追查一樁滅門血案背後的血衣門餘孽。",
        theme="正邪抉擇", goal="查明滅門血案真相", tone="沉鬱肅殺",
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
                faction_id="f_shaolin", surface_motive="遵循門規辦案",
                true_motive="其實想掩蓋當年恩怨",
            ),
            NPC(
                id="npc_widow", name="柳寡婦", identity="血案倖存者",
                personality="悲憤壓抑", speech_style="市井口語，夾雜哭腔",
                surface_motive="尋求正義", true_motive="其實想復仇",
            ),
        ],
        player=PlayerCharacter(
            id="player", name="無名弟子", identity="少林俗家弟子",
            stats=[Variable(id="rep", name="聲望", initial=0, kind="stat")],
            starting_items=["itm_token"], origin="少林寺", weakness="心浮氣躁",
            token_item_id="itm_token", relation_to_core_event="奉命下山查案",
        ),
        items=[Item(id="itm_token", name="羅漢令牌", description="少林弟子下山信物")],
        quests=[
            Quest(
                id="q_investigate", name="追查血衣門", objective="查明滅門血案真相",
                giver_npc_id="npc_master", start_event_id="evt_depart",
                event_ids=["evt_depart", "evt_village"],
            ),
        ],
        factions=_fake_factions(),
        regions=_fake_regions(),
        truth=_fake_truth("ch_village"),
        stat_thresholds=_fake_stat_thresholds(),
        chapters=[
            Chapter(
                id="ch_depart", title="下山", summary="主角領命下山",
                event_ids=["evt_depart"], hook="師父神色凝重地交付密令",
                converge_event_id="evt_depart",
            ),
            Chapter(
                id="ch_village", title="查案", summary="主角查訪血案現場",
                event_ids=["evt_village"], hook="村落瀰漫著詭異的血腥氣息",
                converge_event_id="evt_village", clue_ids=["clue_bloodmark"],
            ),
        ],
        clues=[
            Clue(
                id="clue_bloodmark", name="血手印",
                found_in_event_id="evt_village", serves="指向真兇身份",
            ),
        ],
        endings=_fake_endings(),
        events=[
            Event(
                id="evt_depart", title="下山", location="少林寺山門",
                summary="主角領命下山，了塵長老交代查案禁忌。",
                chapter_id="ch_depart", scene_kind="main",
                triggers=[Trigger(type="on_enter", condition="story_flag==start")],
                dialogue=[],
                branches=[
                    Branch(
                        id="br_go", choice_text="領命下山", next_event_id="evt_village",
                        cost="告別師門，斷絕退路", immediate_feedback="了塵長老授予羅漢令牌",
                        effect_ops=[
                            EffectOp(target_kind="stat", target_id="rep", op="add", value="10"),
                        ],
                        converges_to_event_id="evt_village",
                    ),
                ],
            ),
            Event(
                id="evt_village", title="血案現場", location="山腳村落",
                summary="主角詢問柳寡婦滅門血案經過，發現血衣門線索。",
                chapter_id="ch_village", scene_kind="main",
                region_id="r_village", sub_location_id="sl_teahouse",
                clue_ids=["clue_bloodmark"],
                checks=[
                    SkillCheck(
                        id="sk_persuade", kind="attribute_contest", stat_id="rep",
                        success_next_event_id="evt_village", item_bypass_id="itm_token",
                    ),
                ],
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


# --- Layered-pipeline fakes ---------------------------------------------


def _fake_extraction() -> ExtractionResult:
    """Stand-in for the extractor agent: cast/variables/player/items/quests
    plus the GMUD world frame. References the beat ids _fake_beat_sheet()/
    _fake_scene() actually use as the final Event ids (Beat.id == Event.id
    by convention), not _fake_writer_script()'s evt_depart/evt_village
    (a separate, independent fixture for the legacy pipeline) -- so a
    layered fake-backend run's assembled Script validates cleanly."""
    return ExtractionResult(
        npcs=[
            NPC(
                id="npc_master", name="了塵長老", identity="少林戒律院長老",
                personality="嚴厲重規矩", speech_style="佛偈式短句，常引戒律",
                faction_id="f_shaolin", surface_motive="遵循門規辦案",
                true_motive="其實想掩蓋當年恩怨",
            ),
            NPC(
                id="npc_widow", name="柳寡婦", identity="血案倖存者",
                personality="悲憤壓抑", speech_style="市井口語，夾雜哭腔",
                surface_motive="尋求正義", true_motive="其實想復仇",
            ),
        ],
        variables=[
            Variable(
                id="var_suspicion", name="懷疑度", initial=0,
                description="主角對血衣門的懷疑程度",
            ),
        ],
        player=PlayerCharacter(
            id="player", name="無名弟子", identity="少林俗家弟子",
            stats=[Variable(id="rep", name="聲望", initial=0, kind="stat")],
            starting_items=["itm_token"], origin="少林寺", weakness="心浮氣躁",
            token_item_id="itm_token", relation_to_core_event="奉命下山查案",
        ),
        items=[Item(id="itm_token", name="羅漢令牌", description="少林弟子下山信物")],
        quests=[
            Quest(
                id="q_investigate", name="追查血衣門", objective="查明滅門血案真相",
                giver_npc_id="npc_master", start_event_id="beat_depart",
                event_ids=["beat_depart", "beat_village"],
            ),
        ],
        theme="正邪抉擇", goal="查明滅門血案真相", tone="沉鬱肅殺",
        factions=_fake_factions(),
        regions=_fake_regions(),
        truth=_fake_truth("ch_village"),
        stat_thresholds=_fake_stat_thresholds(),
        clues=[
            Clue(
                id="clue_bloodmark", name="血手印",
                found_in_event_id="beat_village", serves="指向真兇身份",
            ),
        ],
        endings=_fake_endings(),
    )


def _fake_beat_sheet() -> BeatSheet:
    """Stand-in for the beat_expander agent: a small outline with a causal
    chain of beats (beat_village depends on beat_depart, etc.) so tests that
    exercise topological batching have real data to work with. Chapters
    carry hook/converge_event_id/clue_ids, and each beat carries scene_kind
    -- the GMUD frame's beat-expand-stage fields."""
    outline = Outline(
        title="試煉：血衣門疑雲",
        premise="一名少林俗家弟子奉命下山，追查一樁滅門血案背後的血衣門餘孽。",
        chapters=[
            Chapter(
                id="ch_depart", title="下山", summary="主角領命下山",
                beat_ids=["beat_depart"], hook="師父神色凝重地交付密令",
                converge_event_id="beat_depart",
            ),
            Chapter(
                id="ch_village",
                title="查案",
                summary="主角查訪血案現場",
                beat_ids=["beat_village", "beat_clue"],
                hook="村落瀰漫著詭異的血腥氣息",
                converge_event_id="beat_clue", clue_ids=["clue_bloodmark"],
            ),
        ],
    )
    beats = [
        Beat(
            id="beat_depart",
            chapter_id="ch_depart",
            summary="主角領命下山，了塵長老交代查案禁忌。",
            npc_ids=["npc_master"],
            scene_kind="main",
        ),
        Beat(
            id="beat_village",
            chapter_id="ch_village",
            summary="主角詢問柳寡婦滅門血案經過。",
            npc_ids=["npc_widow"],
            causal_deps=["beat_depart"],
            scene_kind="main",
        ),
        Beat(
            id="beat_clue",
            chapter_id="ch_village",
            summary="主角在村落中發現血衣門線索。",
            npc_ids=["npc_widow"],
            causal_deps=["beat_village"],
            scene_kind="flavor",
        ),
    ]
    return BeatSheet(outline=outline, beats=beats)


def _fake_scene(beat: Beat | None) -> Event:
    """Stand-in for the scene_writer agent: expand one Beat into a
    dialogue-filled Event, including the GMUD frame's scene-writer-stage
    fields (scene_kind/chapter_id/checks/clue_ids and a branch with cost/
    immediate_feedback/converges_to_event_id on beat_depart, matching
    _fake_extraction()'s end_justice ending's required_branch_ids). The
    returned Event.id doesn't need to be authoritative -- run_layered_
    pipeline() overwrites it with the target_event_id it called scene_write
    with, since a model's own id choice can't be trusted (that invariant is
    what keeps parallel calls in a later phase from colliding)."""
    if beat is None:
        return Event(
            id="evt_unknown",
            title="未知場景",
            location="",
            summary="",
            dialogue=[DialogueLine(npc_id="npc_master", line="……", emotion="")],
        )
    npc_id = beat.npc_ids[0] if beat.npc_ids else "npc_master"
    branches: list[Branch] = []
    clue_ids: list[str] = []
    checks: list[SkillCheck] = []
    region_id = ""
    sub_location_id = ""
    if beat.id == "beat_depart":
        branches = [
            Branch(
                id="br_go", choice_text="領命下山", next_event_id="beat_village",
                cost="告別師門，斷絕退路", immediate_feedback="了塵長老授予羅漢令牌",
                effect_ops=[EffectOp(target_kind="stat", target_id="rep", op="add", value="10")],
                converges_to_event_id="beat_village",
            ),
        ]
    elif beat.id == "beat_village":
        clue_ids = ["clue_bloodmark"]
        checks = [
            SkillCheck(
                id="sk_persuade", kind="attribute_contest", stat_id="rep",
                success_next_event_id="beat_clue", item_bypass_id="itm_token",
            ),
        ]
        region_id, sub_location_id = "r_village", "sl_teahouse"
    return Event(
        id=beat.id,
        title=beat.summary[:8] or beat.id,
        location="",
        summary=beat.summary,
        chapter_id=beat.chapter_id,
        scene_kind=beat.scene_kind or "main",
        region_id=region_id,
        sub_location_id=sub_location_id,
        clue_ids=clue_ids,
        checks=checks,
        branches=branches,
        dialogue=[
            DialogueLine(
                npc_id=npc_id,
                line=f"（於此）此事，且聽我道來——{beat.summary}",
                emotion="凝重",
            )
        ],
    )


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

        obj: BaseModel
        if role == ROLE_WRITER:
            obj = _fake_writer_script()
        elif role == ROLE_DIALOGUE:
            obj = _fake_fill_dialogue(_extract_script_json(messages))
        elif role == ROLE_PROOFREADER:
            obj = _fake_proofread(_extract_script_json(messages))
        elif role == ROLE_EXTRACTOR:
            obj = _fake_extraction()
        elif role == ROLE_BEAT_EXPANDER:
            obj = _fake_beat_sheet()
        elif role == ROLE_SCENE_WRITER:
            obj = _fake_scene(parse_model_json(_messages_to_text(messages), Beat))
        else:
            # Unexpected role -- return something harmless instead of
            # crashing the executor loop.
            return "Final Answer: {}"

        # Tool-bearing agents (the dialogue/scene_writer agents) never get a
        # response_model from the executor (see
        # crew_agent_executor._invoke_loop_react), so they must produce
        # ReAct-style "Final Answer:" text instead of a structured object.
        if response_model is not None:
            return obj
        return f"Final Answer: {obj.model_dump_json()}"

    def supports_function_calling(self) -> bool:
        return False
