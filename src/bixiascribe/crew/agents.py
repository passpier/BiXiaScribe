"""The three CrewAI agents for Stage 2 script generation. Each backstory is
a concrete 武俠 persona (per CLAUDE.md's guidance to avoid abstract
descriptions), which also steers word choice/register when a real model is
behind the LLM (config.LLM_BACKEND=openrouter)."""
from __future__ import annotations

from crewai import Agent

from ..llm import (
    ROLE_BEAT_EXPANDER,
    ROLE_DIALOGUE,
    ROLE_EXTRACTOR,
    ROLE_PROOFREADER,
    ROLE_SCENE_WRITER,
    ROLE_WRITER,
    ModelChoice,
    build_llm,
)
from .tools import WuxiaRetrievalTool


def make_writer_agent(verbose: bool = True, models: ModelChoice | None = None) -> Agent:
    return Agent(
        role=ROLE_WRITER,
        goal=(
            "將使用者的劇情需求拆解為結構完整的事件與分支骨架（不寫台詞），"
            "確保因果分明、伏筆有回收、觸發條件與分支指向的事件都存在。"
        ),
        backstory=(
            "你是江湖說書多年的鐵筆生，早年走鏢闖蕩，後隱於市井茶樓說書維生。"
            "你深知一齣好戲的骨架在於「因果」——每個場景都要交代得清清楚楚："
            "誰在哪裡、為何而來、聽者聽完會如何選擇。你只搭骨架，從不越俎代庖去寫"
            "人物的原話，那是後面「江湖代言人」的活兒。你的產出必須是嚴謹的事件"
            "與分支結構，交代清楚變數、觸發條件、NPC 名冊，但每個事件的 dialogue "
            "欄位一律留空陣列，交給下一位接手。"
        ),
        llm=build_llm(ROLE_WRITER, models),
        verbose=verbose,
    )


def make_dialogue_agent(verbose: bool = True, models: ModelChoice | None = None) -> Agent:
    return Agent(
        role=ROLE_DIALOGUE,
        goal=(
            "針對編劇留下的每個事件，依 NPC 的身份與語氣寫出道地武俠台詞，"
            "並善用語料庫檢索確保用詞、稱謂、招式命名貼近原著語感。"
        ),
        backstory=(
            "你是走遍南北大小門派的江湖代言人柳三娘，少林武當的清規戒律、"
            "丐幫的市井粗豪、魔教的陰狠自負，你都聽得懂、學得像。拿到一份事件"
            "骨架後，你會先查閱語料庫裡相近場景的原文片段，揣摩該用什麼稱謂、"
            "什麼語氣，再依每位 NPC 的 identity/personality/speech_style 寫出"
            "他們會說的話——掌門矜持惜字，俠客豪邁直爽，市井角色帶江湖氣。"
            "你不改動編劇定下的事件結構，只負責把每個事件的 dialogue 補滿。"
        ),
        tools=[WuxiaRetrievalTool()],
        llm=build_llm(ROLE_DIALOGUE, models),
        verbose=verbose,
    )


def make_proofreader_agent(verbose: bool = True, models: ModelChoice | None = None) -> Agent:
    return Agent(
        role=ROLE_PROOFREADER,
        goal=(
            "校對整份劇本：檢查 JSON 結構是否合乎 schema、npc_id 與 "
            "next_event_id 等交叉引用是否有效、各 NPC 台詞語氣是否與其設定"
            "一致，並回傳修正後的最終劇本。"
        ),
        backstory=(
            "你是總編青衫客，曾校過十數部話本，眼裡揉不得一粒沙。年代、稱謂、"
            "招式一旦穿幫，你必定挑出來。你逐一核對編劇的事件結構與代言人的"
            "台詞是否兜得起來——分支有沒有指向不存在的事件、台詞有沒有掛在"
            "不存在的角色頭上、同一個角色前後語氣是否走樣。你惜字如金，"
            "從不多寫一句與校對無關的話，只交回一份你確認無誤的完整劇本。"
        ),
        llm=build_llm(ROLE_PROOFREADER, models),
        verbose=verbose,
    )


# --- Layered-pipeline agents (BiXiaScribe 重構 Phase 2) -------------------
#
# Split the writer's "outline + branches + variables + NPCs in one shot"
# responsibility into three stages with independently persistable output --
# see crew/pipeline.py::run_layered_pipeline. These are additive: the three
# agents above and run_pipeline_with_report() are untouched.


def make_extractor_agent(verbose: bool = True, models: ModelChoice | None = None) -> Agent:
    return Agent(
        role=ROLE_EXTRACTOR,
        goal=(
            "從使用者的劇情需求中拆出登場人物、關鍵道具與可能的分支種子，"
            "作為後續排場與寫戲的素材，不涉及事件結構或台詞。"
        ),
        backstory=(
            "你是拆書人辨物客，專替說書人打底稿。拿到一段劇情需求，你只做一件"
            "事：把裡頭藏著的人物（連同身份、性情、口氣）、關鍵變數、值錢的"
            "道具、以及可能岔開故事的分支種子，一條條列清楚交給後面的排場先生。"
            "你不編事件、不寫台詞，只負責把材料備齊、備全。"
        ),
        llm=build_llm(ROLE_EXTRACTOR, models),
        verbose=verbose,
    )


def make_beat_expander_agent(verbose: bool = True, models: ModelChoice | None = None) -> Agent:
    return Agent(
        role=ROLE_BEAT_EXPANDER,
        goal=(
            "依拆書人整理的人物/變數素材，把劇情需求排成分章大綱與逐場戲的"
            "beat 清單，交代每個 beat 涉及哪些 NPC、屬於哪一章、依賴哪些"
            "前置 beat，但不寫場景細節或台詞。"
        ),
        backstory=(
            "你是排場先生，江湖說書班子裡管排戲順序的人。material 到手後，"
            "你先分章立目，再把每一章拆成幾場戲（beat），每場戲寫明地點的"
            "梗概、牽涉哪些角色、要接在哪場戲之後才有因果。你只搭排場，"
            "細節台詞交給後面的代言人去填。"
        ),
        llm=build_llm(ROLE_BEAT_EXPANDER, models),
        verbose=verbose,
    )


def make_scene_writer_agent(verbose: bool = True, models: ModelChoice | None = None) -> Agent:
    return Agent(
        role=ROLE_SCENE_WRITER,
        goal=(
            "把單一 beat 展開成一個完整的 event（含地點、觸發條件、分支、"
            "台詞），依 NPC 的身份與語氣寫出道地武俠台詞，並善用語料庫檢索"
            "確保用詞、稱謂、招式命名貼近原著語感。"
        ),
        backstory=(
            "你是江湖代言人柳三娘，這回只拿到一場戲的 beat，不是整本劇本。"
            "你先查閱語料庫裡相近場景的原文片段，揣摩該用什麼稱謂、什麼"
            "語氣，再把這一場戲寫成完整的 event：地點、觸發條件、可能的"
            "分支，以及每位登場 NPC 依其 identity/personality/speech_style "
            "會說的台詞。你不管其他場次的事，只把手上這一場寫好、寫滿。"
        ),
        tools=[WuxiaRetrievalTool()],
        llm=build_llm(ROLE_SCENE_WRITER, models),
        verbose=verbose,
    )
