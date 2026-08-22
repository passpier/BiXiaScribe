"""Unit tests for crew/tasks.py's SCRIPT_LENGTH prompt-target knob.

Runs against LLM_BACKEND=fake -- Task construction needs a real crewai
Agent object (task.description is computed at construction time, before any
LLM call happens), so these build real Agents backed by FakeLLM rather than
a bare stub. FakeLLM's beat sheet/scene fixtures are fixed regardless of
script_length (see llm.py::_fake_beat_sheet/_fake_scene) -- these tests
assert on the *prompt text* the task builders produce, not on how long a
fake-backend run turns out, which is the only way to test this knob
offline (crewai's Task.description is what the real model would read; the
fake LLM never looks at it).

The "short"-preset baselines below are deliberately updated (not the
original pre-GMUD prompt text) to include the GMUD world/choice-design
clauses added in this change -- see CLAUDE.md's script generation section.
They stay byte-for-byte regression guards against *unintended* prompt
drift, just against today's baseline instead of the pre-GMUD one."""
import os
import sys
from pathlib import Path

os.environ["LLM_BACKEND"] = "fake"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402

config.LLM_BACKEND = "fake"

from bixiascribe.crew.agents import make_writer_agent  # noqa: E402
from bixiascribe.crew.context_builder import build_session_document  # noqa: E402
from bixiascribe.crew.tasks import (  # noqa: E402
    make_beat_expand_task,
    make_dialogue_task,
    make_scene_write_task,
    make_writer_task,
)
from bixiascribe.schema import Beat, ExtractionResult  # noqa: E402

_AGENT = make_writer_agent()
_EXTRACTION = ExtractionResult(npcs=[])
_BEAT = Beat(id="b1", chapter_id="c1", summary="s", npc_ids=[], causal_deps=[])
_SESSION = build_session_document(_BEAT, _EXTRACTION, [])

# The exact current prompt text, byte-for-byte, for the "short" preset
# (default) -- these are the regression guards. If any of these break, a
# default (script_length="short") run's prompt has silently changed.
#
# Updated for Phase 4's flat/ID-referenced schema rewrite (see
# openspec/changes/2026-08-22-slim-script-schema-mvp) -- a deliberate
# byte-content update, not a stale guard.
_ORIG_WRITER = '根據以下使用者劇情需求，設計一份武俠 RPG 事件/分支骨架：\n\nREQ\n\n產出必須包含：meta（title、theme 主題一句話、goal 玩家目標、tone 基調/氛圍）、至少 2 個 npcs（含 id/name/faction_id/role/personality/speech_style）、至少 2 個 events。每個 event 需含 id/title/summary、preconditions、choices（choice.next 必須對應到某個 event 的 id），並且每個 event 的 dialogue 欄位一律填空陣列 []——台詞由下一位「對話 agent」負責，你只搭骨架。至少 1 個 chapter（含 start_event 指向這一章開場的 event）。另外，這是一齣要「玩」的 RPG，不是純小說大綱，所以還必須包含：player（玩家角色，name/origin/flaw/token）與 stat（唯一數值，含 id/name/init，如心境值/正邪值）——玩家絕對不可以放進 npcs 名冊，旁白也不要造一個假 NPC，敘述請寫進 event.summary；items（至少 1-2 件關鍵道具，每件都要有 from_event 指出在哪個 event 可以取得，或留白表示一開始就持有）。還要設計這個江湖的世界觀骨架：factions（勢力，至少 1-2 個，含 id/name/motive 一句話動機）；truth（三層真相：public 是一開始就公開的事實，revealed 是隨劇情逐步揭露的事實列表——依揭露順序排列，hidden 是保留給作者、絕對不能提前出現在任何場景內容裡的私藏真相）；endings（至少 1-2 個結局，每個都用 min/max 界定 stat 數值區間，彼此不重疊）。每個有效果（effects/delta）的選項都要符合「抉擇點設計三原則」：1) 代價（cost）——玩家真正失去了什麼，不能只是數值增減，要讓玩家覺得「這是取捨」；2) 立即回饋——選了之後馬上看得到後果（寫進下一個 event 的 summary/dialogue，不需要獨立欄位）；3) 延遲回收（payoff_at）——如果效果不是當場兌現，要填上哪一章會回收這個效果。\n錯誤示範（假選擇）：「A. 立刻上前扶起受傷的少女」與「B. 立刻上前扶起受傷的少年」——兩者文字幾乎相同、數值增減方向相同（都是聲望+1），沒有任何取捨可言，這是假選擇，不允許。\n正確示範：「A. 出手救人（代價：暴露行蹤，日後被追殺，聲望+1，內力-10）」與「B. 悄悄離開（代價：錯過一條重要線索，但保住行蹤）」——兩者代價不同、指向不同的後續發展，這才是真選擇。'  # noqa: E501
_ORIG_DIALOGUE = (
    "上一步「編劇」產出的事件骨架見對話上下文（context）。請針對每一個 "
    "event，依照其中 NPC 的 personality/speech_style，"
    "使用語料庫檢索工具（wuxia_corpus_search）查詢貼近場景語感的原文"
    "片段，再寫出至少一段 NPC 台詞填入該 event 的 dialogue 欄位。"
    "不要更動編劇定下的事件結構、id、preconditions、choices——只補上台詞。"
    "回傳補完 dialogue 後的完整 Script JSON。"
)
_ORIG_BEAT_EXPAND = '使用者的劇情需求：\n\nREQ\n\n拆書人已整理出的人物/世界觀素材：\n\n{"meta":null,"stat":null,"player":null,"npcs":[],"items":[],"factions":[],"truth":null,"clues":[],"endings":[]}\n\n請把這段劇情排成分章大綱與逐場戲的 beat 清單：至少 1-2 章（chapters），每章至少 1 個 beat。每個 chapter 需含 title/summary。每個 beat 需含 id、chapter_id（對應到某個 chapter 的 id）、summary（這場戲的梗概）、npc_ids（涉及哪些登場角色，id 需來自上方素材）、causal_deps（依賴哪些前置 beat 的 id，沒有就留空陣列）。不要寫場景細節或台詞，那是後面「江湖代言人」的活兒。'  # noqa: E501
_ORIG_SCENE_WRITE = '請把以下這一場戲的 beat 展開成一個完整的 event。session 內含玩家/道具素材、登場 NPC 設定（含哪些已經在先前場次登場過）、勢力/章節、目前已解鎖的真相（truth_public/truth_unlocked——這是本場戲能提及的真相全部，絕對不能提及尚未解鎖的內容）、目前這場戲的 beat，以及（若有）已完成的前情場次摘要——已完成場次是本場戲不可牴觸的既定事實：\n\n{"character_cards":[],"scene_summaries":[],"omitted_scene_count":0,"player_card":[],"item_cards":[],"introduced_npc_ids":[],"faction_cards":[],"chapter_card":[],"truth_public":[],"truth_unlocked":[],"allowed_ids":[],"current_beat":{"id":"b1","chapter_id":"c1","summary":"s","npc_ids":[],"causal_deps":[]}}\n\nevent 的 id 欄位請填 "e1"。依每位 NPC 的 personality/speech_style，使用語料庫檢索工具（wuxia_corpus_search）查詢貼近場景語感的原文片段，寫出至少一段台詞。preconditions、choices 依 beat 的 summary、已完成場次摘要與因果合理補上，不得與已完成場次矛盾——preconditions 至少要寫出這場戲成立的前提一句話，不可留空。chapter_id/clue_ids，以及 choice 的 payoff_at，只能填 session.allowed_ids 這份清單裡列出的值——這是封閉選單，不是自由發揮，清單裡沒有合適的就留空，絕對不要自己編一個新 id；若本場戲有需要檢定的橋段，填 check（on_pass/on_fail 兩條路線都要填，on_fail 要有 fail_cost，確保失敗也能推進劇情）；若本場戲有可蒐集的線索，填 clue_ids。若本場戲有 NPC 是第一次登場（不在已登場名單內），台詞或 summary 要交代清楚他是誰、為何在此，不可以憑空開口。每個有效果（effects/delta）的選項都要符合「抉擇點設計三原則」：1) 代價（cost）——玩家真正失去了什麼，不能只是數值增減，要讓玩家覺得「這是取捨」；2) 立即回饋——選了之後馬上看得到後果（寫進下一個 event 的 summary/dialogue，不需要獨立欄位）；3) 延遲回收（payoff_at）——如果效果不是當場兌現，要填上哪一章會回收這個效果。\n錯誤示範（假選擇）：「A. 立刻上前扶起受傷的少女」與「B. 立刻上前扶起受傷的少年」——兩者文字幾乎相同、數值增減方向相同（都是聲望+1），沒有任何取捨可言，這是假選擇，不允許。\n正確示範：「A. 出手救人（代價：暴露行蹤，日後被追殺，聲望+1，內力-10）」與「B. 悄悄離開（代價：錯過一條重要線索，但保住行蹤）」——兩者代價不同、指向不同的後續發展，這才是真選擇。'  # noqa: E501


def test_writer_task_default_is_byte_identical_to_pre_knob_prompt():
    task = make_writer_task("REQ", _AGENT)
    assert task.description == _ORIG_WRITER


def test_writer_task_short_is_byte_identical_to_pre_knob_prompt():
    task = make_writer_task("REQ", _AGENT, script_length="short")
    assert task.description == _ORIG_WRITER


def test_dialogue_task_default_is_byte_identical_to_pre_knob_prompt():
    prior = make_writer_task("REQ", _AGENT)
    task = make_dialogue_task(_AGENT, prior)
    assert task.description == _ORIG_DIALOGUE


def test_beat_expand_task_default_is_byte_identical_to_pre_knob_prompt():
    task = make_beat_expand_task("REQ", _EXTRACTION, _AGENT)
    assert task.description == _ORIG_BEAT_EXPAND


def test_scene_write_task_default_is_byte_identical_to_pre_knob_prompt():
    task = make_scene_write_task(_BEAT, _EXTRACTION, _AGENT, "e1", session=_SESSION)
    assert task.description == _ORIG_SCENE_WRITE


def test_unknown_script_length_falls_back_to_short():
    task = make_writer_task("REQ", _AGENT, script_length="glacial")
    assert task.description == _ORIG_WRITER


# --- medium/long actually change the prompt ---------------------------------


def test_beat_expand_task_medium_asks_for_more_chapters_and_beats():
    task = make_beat_expand_task("REQ", _EXTRACTION, _AGENT, script_length="medium")
    assert "至少 3-4 章" in task.description
    assert "每章至少 2-3 個 beat" in task.description
    assert task.description != _ORIG_BEAT_EXPAND


def test_beat_expand_task_long_asks_for_even_more():
    task = make_beat_expand_task("REQ", _EXTRACTION, _AGENT, script_length="long")
    assert "至少 5-6 章" in task.description
    assert "每章至少 3-4 個 beat" in task.description


def test_writer_task_medium_and_long_raise_event_target():
    medium = make_writer_task("REQ", _AGENT, script_length="medium")
    assert "至少 8-12 個 events" in medium.description
    long_ = make_writer_task("REQ", _AGENT, script_length="long")
    assert "至少 15-24 個 events" in long_.description


def test_scene_write_task_medium_and_long_raise_dialogue_target():
    medium = make_scene_write_task(
        _BEAT, _EXTRACTION, _AGENT, "e1", session=_SESSION, script_length="medium"
    )
    assert "寫出至少二至三段台詞" in medium.description
    long_ = make_scene_write_task(
        _BEAT, _EXTRACTION, _AGENT, "e1", session=_SESSION, script_length="long"
    )
    assert "寫出至少三段以上台詞" in long_.description


def test_dialogue_task_medium_and_long_raise_dialogue_target():
    prior = make_writer_task("REQ", _AGENT)
    medium = make_dialogue_task(_AGENT, prior, script_length="medium")
    assert "至少二至三段 NPC 台詞" in medium.description


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
