"""Task definitions wiring the three agents into a sequential pipeline. Each
task's output_pydantic is schema.Script, so CrewAI enforces/coerces
structured output at every stage (see ..llm for how that interacts with
tool-bearing agents like the dialogue agent)."""
from __future__ import annotations

from crewai import Agent, Task

from ..schema import Script


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
