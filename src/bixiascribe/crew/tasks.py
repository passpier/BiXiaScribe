"""Task definitions wiring the three agents into a sequential pipeline. Each
task's output_pydantic is schema.Script, so CrewAI enforces/coerces
structured output at every stage (see ..llm for how that interacts with
tool-bearing agents like the dialogue agent)."""
from __future__ import annotations

import json
from typing import Any, TypeVar

from crewai import Agent, Task
from pydantic import BaseModel

from .. import config, length, wire
from ..schema import (
    Beat,
    BeatSheet,
    Event,
    ExtractionResult,
    Script,
    SessionDocument,
    parse_model_json,
)
from . import guardrails, normalize, scene_metrics
from .context_builder import build_session_document

M = TypeVar("M", bound=BaseModel)


def _freeform_expected_output(model_cls: type[BaseModel], base_text: str) -> str:
    """expected_output text for a task's `structured=False` variant.
    Normally the shape a model must emit is enforced provider-side, via
    output_pydantic being forwarded as `response_format` (see
    crew/execute.py's module docstring) -- it is never written into the
    prompt itself. Without output_pydantic, the shape has to be spelled out
    here instead, or the model has no way to know what fields are expected.
    Kept as a separate function (rather than folded into each task's
    `description`) so the `structured=True` path's description/
    expected_output stay byte-for-byte unchanged -- see
    tests/test_script_length.py's regression guards."""
    schema_json = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    return (
        f"{base_text}\n\n"
        "請只輸出一個 JSON 物件，不要加任何解說文字、不要用 ``` 圍欄，"
        f"必須符合以下 JSON Schema：\n{schema_json}"
    )


def _coerce_for_guardrail(output: Any, model_cls: type[M]) -> M | None:
    """Minimal duplicate of pipeline.py::_coerce_model's pydantic ->
    lenient-mirror -> json_dict -> raw-scan (-> raw-scan lenient) fallback,
    kept local to avoid a circular import (pipeline.py already imports from
    this module). A guardrail only needs the salvaged object, not
    pipeline.py's source-tier bookkeeping."""
    if isinstance(output.pydantic, model_cls):
        return output.pydantic
    if isinstance(output.pydantic, wire.lenient_mirror(model_cls)):
        return wire.to_strict(output.pydantic, model_cls)
    if output.json_dict is not None:
        try:
            return model_cls.model_validate(output.json_dict)
        except Exception:
            pass
    result = parse_model_json(output.raw, model_cls)
    if result is not None:
        return result
    # A structured=False (free-text) task's response has no provider-side
    # schema enforcement at all, so a model that skips one strict-required
    # field is more likely here than on the structured path -- the lenient
    # mirror tier (see wire.py) is what still salvages that instead of the
    # guardrail rejecting an otherwise-fine response outright.
    lenient = parse_model_json(output.raw, wire.lenient_mirror(model_cls))
    if lenient is not None:
        return wire.to_strict(lenient, model_cls)
    return None


def _guardrails_active() -> bool:
    """Guardrails must be off under LLM_BACKEND=fake -- see
    guardrails.py's module docstring: FakeLLM's canned responses can never
    satisfy an RPG-shape check, so a guardrail retry loop against it would
    just spin guardrail_max_retries on every offline/fake test run."""
    return config.GUARDRAILS_ENABLED and config.LLM_BACKEND != "fake"

# Prompt-level script-length targets (config.SCRIPT_LENGTH / Variant.script_length
# / --script-length), resolved by ..length.parse_length_spec (presets or a
# custom:key=value,... string -- see that module's docstring). "short"
# reproduces this module's original, pre-knob prompt text byte-for-byte (see
# tests/test_script_length.py) -- deliberately, since SCRIPT_LENGTH defaults
# to "short" and every existing caller must see zero prompt change unless it
# opts in. "events" feeds the legacy writer task (which produces the whole
# event list in one shot); "chapters"/"beats_per_chapter" feed the layered
# beat_expand task; "min_dialogue" feeds both dialogue-authoring tasks
# (legacy dialogue task, layered scene_write task). None of this is enforced
# anywhere -- see crew/tasks.py's module docstring and config.SCRIPT_LENGTH's
# comment for why a model can still under/over-shoot it.


def _length_target(script_length: str) -> dict[str, str]:
    return length.parse_length_spec(script_length).targets


# Shared clause asking for the RPG-shaped entities (player/items) that
# schema.py added alongside the original title/npcs/events set -- see
# CLAUDE.md's script generation section for why these are first-class
# fields now instead of something a model has to improvise (a fake
# "npc_player"/"npc_narrator" NPC, a props list that's extracted then
# dropped, etc). Injected into both the legacy writer task and the layered
# extract task, since both are where these entities first get a chance to
# exist.
_RPG_ENTITIES_CLAUSE = (
    "另外，這是一齣要「玩」的 RPG，不是純小說大綱，所以還必須包含："
    "player（玩家角色，name/origin/flaw/token）與 stat（唯一數值，"
    "含 id/name/init，如心境值/正邪值）——玩家絕對不可以放進 npcs 名冊，"
    "旁白也不要造一個假 NPC，敘述請寫進 event.summary；"
    "items（至少 1-2 件關鍵道具，每件都要有 from_event 指出"
    "在哪個 event 可以取得，或留白表示一開始就持有）。"
)

# Shared clause asking for the GMUD world frame: factions, the guide's
# 唯一數值 single-stat system, three-layer truth disclosure, and endings --
# the structural concepts a novel-outline-shaped script has no field for.
# Injected into both the legacy writer task and the layered extract task,
# matching _RPG_ENTITIES_CLAUSE's sharing pattern.
_GMUD_WORLD_CLAUSE = (
    "還要設計這個江湖的世界觀骨架："
    "factions（勢力，至少 1-2 個，含 id/name/motive 一句話動機）；"
    "truth（三層真相：public 是一開始就公開的事實，revealed 是隨劇情"
    "逐步揭露的事實列表——依揭露順序排列，hidden 是保留給作者、"
    "絕對不能提前出現在任何場景內容裡的私藏真相）；"
    "endings（至少 1-2 個結局，每個都用 min/max 界定 stat 數值區間，"
    "彼此不重疊）。"
)

# Shared clause teaching 抉擇點設計三原則 with the guide's own
# 錯誤示範/正確示範 pair, so the writer/scene_writer roles internalize what
# a real choice looks like instead of just being told "add a cost field".
_CHOICE_DESIGN_CLAUSE = (
    "每個有效果（effects/delta）的選項都要符合「抉擇點設計三原則」："
    "1) 代價（cost）——玩家真正失去了什麼，不能只是數值增減，要讓玩家覺得"
    "「這是取捨」；2) 立即回饋——選了之後馬上看得到後果（寫進下一個 event "
    "的 summary/dialogue，不需要獨立欄位）；3) 延遲回收（payoff_at）——如果"
    "效果不是當場兌現，要填上哪一章會回收這個效果。\n"
    "錯誤示範（假選擇）：「A. 立刻上前扶起受傷的少女」與「B. 立刻上前扶起"
    "受傷的少年」——兩者文字幾乎相同、數值增減方向相同（都是聲望+1），沒有"
    "任何取捨可言，這是假選擇，不允許。\n"
    "正確示範：「A. 出手救人（代價：暴露行蹤，日後被追殺，聲望+1，內力-10）」"
    "與「B. 悄悄離開（代價：錯過一條重要線索，但保住行蹤）」——兩者代價不同、"
    "指向不同的後續發展，這才是真選擇。"
)


def make_writer_task(
    requirement: str, agent: Agent, script_length: str = "short", structured: bool = True
) -> Task:
    target = _length_target(script_length)
    guardrail_kwargs: dict[str, Any] = {}
    if _guardrails_active():

        def _writer_guardrail(output):
            script = _coerce_for_guardrail(output, Script)
            if script is None:
                return False, "輸出不是合法的 Script JSON，請重新輸出完整的 Script JSON。"
            problems = guardrails.check_script_rpg(script)
            if problems:
                return False, guardrails.as_feedback(problems)
            return True, output

        guardrail_kwargs = {
            "guardrail": _writer_guardrail,
            "guardrail_max_retries": config.GUARDRAIL_MAX_RETRIES,
        }
    expected_output = "一份符合 Script schema 的 JSON，所有 event 的 dialogue 欄位皆為空陣列。"
    output_kwargs: dict[str, Any] = {"output_pydantic": wire.lenient_mirror(Script)}
    if not structured:
        expected_output = _freeform_expected_output(Script, expected_output)
        output_kwargs = {}
    return Task(
        description=(
            "根據以下使用者劇情需求，設計一份武俠 RPG 事件/分支骨架：\n\n"
            f"{requirement}\n\n"
            "產出必須包含：meta（title、theme 主題一句話、goal 玩家目標、"
            "tone 基調/氛圍）、"
            "至少 2 個 npcs（含 id/name/faction_id/role/personality/speech_style）、"
            f"至少 {target['events']} 個 events。每個 event 需含 id/title/summary、"
            "preconditions、choices（choice.next 必須對應到某個 event 的 "
            "id），並且每個 event 的 dialogue 欄位一律填空陣列 []——台詞由下一位"
            "「對話 agent」負責，你只搭骨架。至少 1 個 chapter（含 start_event "
            "指向這一章開場的 event）。"
            f"{_RPG_ENTITIES_CLAUSE}"
            f"{_GMUD_WORLD_CLAUSE}"
            f"{_CHOICE_DESIGN_CLAUSE}"
        ),
        expected_output=expected_output,
        agent=agent,
        **output_kwargs,
        **guardrail_kwargs,
    )


def make_dialogue_task(
    agent: Agent,
    context_task: Task,
    script_length: str = "short",
    use_retrieval: bool = True,
    structured: bool = True,
) -> Task:
    target = _length_target(script_length)
    retrieval_clause = (
        "使用語料庫檢索工具（wuxia_corpus_search）查詢貼近場景語感的原文"
        f"片段，再寫出至少{target['min_dialogue']} NPC 台詞填入該 event 的 dialogue 欄位。"
        if use_retrieval
        else (
            f"直接以你自己的武俠語感，寫出至少{target['min_dialogue']} "
            "NPC 台詞填入該 event 的 dialogue 欄位。"
        )
    )
    expected_output = "與輸入相同結構的 Script JSON，但每個 event 的 dialogue 都已補上台詞。"
    output_kwargs: dict[str, Any] = {"output_pydantic": wire.lenient_mirror(Script)}
    if not structured:
        expected_output = _freeform_expected_output(Script, expected_output)
        output_kwargs = {}
    return Task(
        description=(
            "上一步「編劇」產出的事件骨架見對話上下文（context）。請針對每一個 "
            "event，依照其中 NPC 的 personality/speech_style，"
            f"{retrieval_clause}"
            "不要更動編劇定下的事件結構、id、preconditions、choices——只補上台詞。"
            "回傳補完 dialogue 後的完整 Script JSON。"
        ),
        expected_output=expected_output,
        agent=agent,
        context=[context_task],
        **output_kwargs,
    )


def make_proofread_task(agent: Agent, context_task: Task, structured: bool = True) -> Task:
    expected_output = "通過檢查、可直接使用的最終 Script JSON。"
    output_kwargs: dict[str, Any] = {"output_pydantic": wire.lenient_mirror(Script)}
    if not structured:
        expected_output = _freeform_expected_output(Script, expected_output)
        output_kwargs = {}
    return Task(
        description=(
            "上一步「對話」產出的完整劇本見對話上下文（context）。請檢查："
            "1) 是否所有 dialogue.npc 都對應到存在的 NPC 或 \"player\"；"
            "2) 是否所有 choice.next 都對應到存在的 event；"
            "3) 各 NPC 台詞語氣是否符合其 speech_style 設定；"
            "4) player/items 是否齊全，每件 item 是否有事件可取得；"
            "5) npcs 名冊裡是否混入了假冒的玩家或旁白角色。"
            "若發現問題就直接修正，最終回傳一份你確認無誤的完整 Script JSON。"
        ),
        expected_output=expected_output,
        agent=agent,
        context=[context_task],
        **output_kwargs,
    )


# --- Layered-pipeline tasks --------------------------------------------


def make_extract_task(requirement: str, agent: Agent, structured: bool = True) -> Task:
    guardrail_kwargs: dict[str, Any] = {}
    if _guardrails_active():

        def _extract_guardrail(output):
            extraction = _coerce_for_guardrail(output, ExtractionResult)
            if extraction is None:
                return False, "輸出不是合法的 ExtractionResult JSON，請重新輸出完整 JSON。"
            problems = guardrails.check_extraction_rpg(extraction)
            if problems:
                return False, guardrails.as_feedback(problems)
            return True, output

        guardrail_kwargs = {
            "guardrail": _extract_guardrail,
            "guardrail_max_retries": config.GUARDRAIL_MAX_RETRIES,
        }
    expected_output = "一份符合 ExtractionResult schema 的 JSON。"
    output_kwargs: dict[str, Any] = {"output_pydantic": wire.lenient_mirror(ExtractionResult)}
    if not structured:
        expected_output = _freeform_expected_output(ExtractionResult, expected_output)
        output_kwargs = {}
    return Task(
        description=(
            "根據以下使用者劇情需求，拆出玩家角色、登場人物與關鍵世界觀：\n\n"
            f"{requirement}\n\n"
            "產出必須包含：meta（theme 主題一句話、goal 玩家目標、"
            "tone 基調/氛圍）、"
            f"{_RPG_ENTITIES_CLAUSE}"
            f"{_GMUD_WORLD_CLAUSE}"
            "不要設計事件結構或章節，那是後面「排場先生」的活兒。"
        ),
        expected_output=expected_output,
        agent=agent,
        **output_kwargs,
        **guardrail_kwargs,
    )


def make_beat_expand_task(
    requirement: str,
    extraction: ExtractionResult,
    agent: Agent,
    script_length: str = "short",
    structured: bool = True,
) -> Task:
    target = _length_target(script_length)
    guardrail_kwargs: dict[str, Any] = {}
    if _guardrails_active():

        def _beat_expand_guardrail(output):
            beat_sheet = _coerce_for_guardrail(output, BeatSheet)
            if beat_sheet is None:
                return False, "輸出不是合法的 BeatSheet JSON，請重新輸出完整 JSON。"
            problems = guardrails.check_beat_expand_rpg(beat_sheet)
            if problems:
                return False, guardrails.as_feedback(problems)
            return True, output

        guardrail_kwargs = {
            "guardrail": _beat_expand_guardrail,
            "guardrail_max_retries": config.GUARDRAIL_MAX_RETRIES,
        }
    expected_output = "一份符合 BeatSheet schema 的 JSON（outline + beats）。"
    output_kwargs: dict[str, Any] = {"output_pydantic": wire.lenient_mirror(BeatSheet)}
    if not structured:
        expected_output = _freeform_expected_output(BeatSheet, expected_output)
        output_kwargs = {}
    return Task(
        description=(
            "使用者的劇情需求：\n\n"
            f"{requirement}\n\n"
            "拆書人已整理出的人物/世界觀素材：\n\n"
            f"{extraction.model_dump_json()}\n\n"
            f"請把這段劇情排成分章大綱與逐場戲的 beat 清單：至少 {target['chapters']} 章"
            f"（chapters），每章至少 {target['beats_per_chapter']} 個 beat。每個 chapter 需含 "
            "title/summary。每個 beat 需含 id、"
            "chapter_id（對應到某個 chapter 的 id）、summary（這場戲的"
            "梗概）、npc_ids（涉及哪些登場角色，id 需來自上方"
            "素材）、causal_deps（依賴哪些前置 beat 的 id，沒有就留空陣列）。"
            "不要寫場景細節或台詞，那是後面「江湖代言人」的活兒。"
        ),
        expected_output=expected_output,
        agent=agent,
        **output_kwargs,
        **guardrail_kwargs,
    )


def make_scene_write_task(
    beat: Beat,
    extraction: ExtractionResult,
    agent: Agent,
    target_event_id: str,
    *,
    session: SessionDocument | None = None,
    script_length: str = "short",
    use_retrieval: bool = True,
    structured: bool = True,
) -> Task:
    """Unlike the legacy dialogue task, this doesn't chain via `context=
    [prior_task]` -- the scene_writer's input is a token-bounded
    SessionDocument, not the whole prior Script. If `session` isn't
    supplied, one is built from just `beat` + `extraction` (no cross-scene
    continuity), matching this task's behavior for any existing direct
    caller that doesn't pass a session.

    See crew/context_builder.py::build_session_document() for how
    character_cards/scene_summaries are ranked and trimmed to
    config.SESSION_DOC_MAX_TOKENS. `script_length` (default "short",
    byte-identical to this task's pre-knob prompt) scales the minimum
    dialogue depth asked for -- see _LENGTH_TARGETS above. `use_retrieval`
    (default True, byte-identical prompt) mirrors config.RETRIEVAL_ENABLED --
    see agents.py::make_scene_writer_agent()'s matching tools= toggle.

    The guardrail (check_scene_rpg) is scoped to session.character_cards'
    NPCs plus session.introduced_npc_ids -- it can only ever see what this
    call was actually handed, so it never penalizes a scene for not
    introducing an NPC outside its own SessionDocument."""
    target = _length_target(script_length)
    if session is None:
        session = build_session_document(beat, extraction, [])
    retrieval_clause = (
        "使用語料庫檢索工具"
        "（wuxia_corpus_search）查詢貼近場景語感的原文片段，寫出至少"
        f"{target['min_dialogue']}台詞。"
        if use_retrieval
        else f"直接以你自己的武俠語感寫出至少{target['min_dialogue']}台詞。"
    )
    guardrail_kwargs: dict[str, Any] = {}
    if _guardrails_active():
        known_npc_ids = {
            card.split("｜", 1)[0]
            for card in session.character_cards + session.player_card
            if "｜" in card
        }
        # id -> name, parsed from the same cards -- lets the guardrail repair
        # a scene_writer output that filled dialogue[].npc with the display
        # name instead of the id (see normalize.normalize_scene_npc_ids's
        # docstring for the real run this reproduces) before judging it,
        # instead of burning a retry on a purely mechanical mistake. Only
        # the "id｜name｜..." shaped cards (character_cards) carry a clean
        # name field; player_card's second segment has a trailing
        # parenthetical ("...｜name（出身：...）"), stripped here so it
        # doesn't pollute the name index.
        name_to_id: dict[str, str] = {}
        for card in session.character_cards + session.player_card:
            parts = card.split("｜")
            if len(parts) < 2 or not parts[0]:
                continue
            name_to_id[parts[1].split("（", 1)[0].strip()] = parts[0]
        introduced_npc_ids = set(session.introduced_npc_ids)

        def _scene_guardrail(output):
            event = _coerce_for_guardrail(output, Event)
            if event is None:
                return False, "輸出不是合法的 Event JSON，請重新輸出完整 Event JSON。"
            event, _ = normalize.normalize_scene_npc_ids(
                event, known_npc_ids=known_npc_ids, name_to_id=name_to_id
            )
            problems = guardrails.check_scene_rpg(
                event,
                known_npc_ids=known_npc_ids,
                introduced_npc_ids=introduced_npc_ids,
            )
            if problems:
                scene_metrics.record_guardrail_retry(beat.id)
                return False, guardrails.as_feedback(problems)
            return True, output

        guardrail_kwargs = {
            "guardrail": _scene_guardrail,
            "guardrail_max_retries": config.GUARDRAIL_MAX_RETRIES,
        }
    expected_output = "一份符合 Event schema 的 JSON，dialogue 已填台詞。"
    output_kwargs: dict[str, Any] = {"output_pydantic": wire.lenient_mirror(Event)}
    if not structured:
        expected_output = _freeform_expected_output(Event, expected_output)
        output_kwargs = {}
    return Task(
        description=(
            "請把以下這一場戲的 beat 展開成一個完整的 event。session 內含"
            "玩家/道具素材、登場 NPC 設定（含哪些已經在先前場次登場過）、"
            "勢力/章節、目前已解鎖的真相（truth_public/"
            "truth_unlocked——這是本場戲能提及的真相全部，絕對不能提及尚未"
            "解鎖的內容）、目前這場戲的 beat，以及（若有）已完成的前情"
            "場次摘要——已完成場次是本場戲不可牴觸的既定事實：\n\n"
            f"{session.model_dump_json()}\n\n"
            f'event 的 id 欄位請填 "{target_event_id}"。依每位 NPC 的 '
            f"personality/speech_style，{retrieval_clause}"
            "preconditions、choices 依 beat 的 summary、"
            "已完成場次摘要與因果合理補上，不得與已完成場次矛盾——"
            "preconditions 至少要寫出這場戲成立的前提一句話，不可留空。"
            "chapter_id/clue_ids，以及 choice 的 payoff_at，"
            "只能填 session.allowed_ids 這份清單裡列出的值——這是封閉選單，"
            "不是自由發揮，清單裡沒有合適的就留空，絕對不要自己編一個新 id；"
            "dialogue 每一段的 npc 欄位只能填 session.character_cards 裡"
            "「｜」前面的那個 id（例如 npc_innkeeper），絕對不能填「｜」後面"
            "的姓名（例如 陳掌柜）——這同樣是封閉選單，填姓名會被判定為未"
            "登場角色而整場作廢；"
            "若本場戲有需要檢定的橋段，填 check（on_pass/on_fail 兩條路線都要"
            "填，on_fail 要有 fail_cost，確保失敗也能推進劇情）；"
            "若本場戲有可蒐集的線索，填 clue_ids。"
            "若本場戲有 NPC 是第一次登場（不在已登場名單內），台詞或 "
            "summary 要交代清楚他是誰、為何在此，不可以憑空開口。"
            f"{_CHOICE_DESIGN_CLAUSE}"
        ),
        expected_output=expected_output,
        agent=agent,
        **output_kwargs,
        **guardrail_kwargs,
    )
