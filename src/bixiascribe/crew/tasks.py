"""Task definitions wiring the three agents into a sequential pipeline. Each
task's output_pydantic is schema.Script, so CrewAI enforces/coerces
structured output at every stage (see ..llm for how that interacts with
tool-bearing agents like the dialogue agent)."""
from __future__ import annotations

from crewai import Agent, Task

from ..schema import Beat, BeatSheet, Event, ExtractionResult, Script, SessionDocument
from .context_builder import build_session_document


def make_writer_task(requirement: str, agent: Agent) -> Task:
    return Task(
        description=(
            "根據以下使用者劇情需求，設計一份武俠 RPG 事件/分支骨架：\n\n"
            f"{requirement}\n\n"
            "產出必須包含：title、premise、至少 1-2 個 variables、"
            "至少 2 個 npcs（含 id/name/identity/personality/speech_style）、"
            "至少 2 個 events。每個 event 需含 id/title/location/summary、"
            "triggers、branches（branch.next_event_id 必須對應到某個 event 的 "
            "id），並且每個 event 的 dialogue 欄位一律填空陣列 []——台詞由下一位"
            "「對話 agent」負責，你只搭骨架。"
        ),
        expected_output="一份符合 Script schema 的 JSON，所有 event 的 dialogue 欄位皆為空陣列。",
        agent=agent,
        output_pydantic=Script,
    )


def make_dialogue_task(agent: Agent, context_task: Task) -> Task:
    return Task(
        description=(
            "上一步「編劇」產出的事件骨架見對話上下文（context）。請針對每一個 "
            "event，依照其中 NPC 的 identity/personality/speech_style，"
            "使用語料庫檢索工具（wuxia_corpus_search）查詢貼近場景語感的原文"
            "片段，再寫出至少一段 NPC 台詞填入該 event 的 dialogue 欄位。"
            "不要更動編劇定下的事件結構、id、觸發條件、分支——只補上台詞。"
            "回傳補完 dialogue 後的完整 Script JSON。"
        ),
        expected_output="與輸入相同結構的 Script JSON，但每個 event 的 dialogue 都已補上台詞。",
        agent=agent,
        context=[context_task],
        output_pydantic=Script,
    )


def make_proofread_task(agent: Agent, context_task: Task) -> Task:
    return Task(
        description=(
            "上一步「對話」產出的完整劇本見對話上下文（context）。請檢查："
            "1) 是否所有 dialogue.npc_id 都對應到存在的 NPC；"
            "2) 是否所有 branch.next_event_id 都對應到存在的 event；"
            "3) 各 NPC 台詞語氣是否符合其 speech_style 設定。"
            "若發現問題就直接修正，最終回傳一份你確認無誤的完整 Script JSON。"
        ),
        expected_output="通過檢查、可直接使用的最終 Script JSON。",
        agent=agent,
        context=[context_task],
        output_pydantic=Script,
    )


# --- Layered-pipeline tasks (BiXiaScribe 重構 Phase 2) --------------------


def make_extract_task(requirement: str, agent: Agent) -> Task:
    return Task(
        description=(
            "根據以下使用者劇情需求，拆出登場人物與關鍵變數：\n\n"
            f"{requirement}\n\n"
            "產出必須包含：至少 2 個 npcs（含 id/name/identity/personality/"
            "speech_style）、至少 1-2 個 variables，以及可選的 props（關鍵"
            "道具）與 branch_candidates（可能的分支種子，用一句話描述）。"
            "不要設計事件結構，那是後面「排場先生」的活兒。"
        ),
        expected_output="一份符合 ExtractionResult schema 的 JSON。",
        agent=agent,
        output_pydantic=ExtractionResult,
    )


def make_beat_expand_task(requirement: str, extraction: ExtractionResult, agent: Agent) -> Task:
    return Task(
        description=(
            "使用者的劇情需求：\n\n"
            f"{requirement}\n\n"
            "拆書人已整理出的人物/變數素材：\n\n"
            f"{extraction.model_dump_json()}\n\n"
            "請把這段劇情排成分章大綱與逐場戲的 beat 清單：至少 1-2 章"
            "（chapters），每章至少 1 個 beat，每個 beat 需含 id、"
            "chapter_id（對應到某個 chapter 的 id）、summary（這場戲的"
            "梗概）、npc_ids（涉及哪些登場角色，id 需來自上方素材）、"
            "causal_deps（依賴哪些前置 beat 的 id，沒有就留空陣列）。"
            "不要寫場景細節或台詞，那是後面「江湖代言人」的活兒。"
        ),
        expected_output="一份符合 BeatSheet schema 的 JSON（outline + beats）。",
        agent=agent,
        output_pydantic=BeatSheet,
    )


def make_scene_write_task(
    beat: Beat,
    extraction: ExtractionResult,
    agent: Agent,
    target_event_id: str,
    *,
    session: SessionDocument | None = None,
) -> Task:
    """Unlike the legacy dialogue task, this doesn't chain via `context=
    [prior_task]` -- the scene_writer's input is a token-bounded
    SessionDocument (BiXiaScribe 重構 Phase 5), not the whole prior Script.
    If `session` isn't supplied, one is built from just `beat` + `extraction`
    (no cross-scene continuity), matching this task's pre-Phase-5 behavior
    for any existing direct caller.

    See crew/context_builder.py::build_session_document() for how
    character_cards/scene_summaries are ranked and trimmed to
    config.SESSION_DOC_MAX_TOKENS."""
    if session is None:
        session = build_session_document(beat, extraction, [])
    return Task(
        description=(
            "請把以下這一場戲的 beat 展開成一個完整的 event。session 內含"
            "登場 NPC 設定、目前這場戲的 beat，以及（若有）已完成的前情"
            "場次摘要——已完成場次是本場戲不可牴觸的既定事實：\n\n"
            f"{session.model_dump_json()}\n\n"
            f'event 的 id 欄位請填 "{target_event_id}"。依每位 NPC 的 '
            "identity/personality/speech_style，使用語料庫檢索工具"
            "（wuxia_corpus_search）查詢貼近場景語感的原文片段，寫出至少"
            "一段台詞。location、triggers、branches 依 beat 的 summary、"
            "已完成場次摘要與因果合理補上，不得與已完成場次矛盾。"
        ),
        expected_output="一份符合 Event schema 的 JSON，dialogue 已填台詞。",
        agent=agent,
        output_pydantic=Event,
    )
