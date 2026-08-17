"""Unit tests for crew/tasks.py's SCRIPT_LENGTH prompt-target knob.

Runs against LLM_BACKEND=fake -- Task construction needs a real crewai
Agent object (task.description is computed at construction time, before any
LLM call happens), so these build real Agents backed by FakeLLM rather than
a bare stub. FakeLLM's beat sheet/scene fixtures are fixed regardless of
script_length (see llm.py::_fake_beat_sheet/_fake_scene) -- these tests
assert on the *prompt text* the task builders produce, not on how long a
fake-backend run turns out, which is the only way to test this knob
offline (crewai's Task.description is what the real model would read; the
fake LLM never looks at it)."""
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
_EXTRACTION = ExtractionResult(npcs=[], variables=[], props=[], branch_candidates=[])
_BEAT = Beat(id="b1", chapter_id="c1", summary="s", npc_ids=[], causal_deps=[])
_SESSION = build_session_document(_BEAT, _EXTRACTION, [])

# The exact original prompt text, byte-for-byte, from before SCRIPT_LENGTH
# existed -- these are the regression guards. If any of these break, a
# default (script_length="short") run's prompt has silently changed.
_ORIG_WRITER = (
    "根據以下使用者劇情需求，設計一份武俠 RPG 事件/分支骨架：\n\n"
    "REQ\n\n"
    "產出必須包含：title、premise、至少 1-2 個 variables、"
    "至少 2 個 npcs（含 id/name/identity/personality/speech_style）、"
    "至少 2 個 events。每個 event 需含 id/title/location/summary、"
    "triggers、branches（branch.next_event_id 必須對應到某個 event 的 "
    "id），並且每個 event 的 dialogue 欄位一律填空陣列 []——台詞由下一位"
    "「對話 agent」負責，你只搭骨架。"
)
_ORIG_DIALOGUE = (
    "上一步「編劇」產出的事件骨架見對話上下文（context）。請針對每一個 "
    "event，依照其中 NPC 的 identity/personality/speech_style，"
    "使用語料庫檢索工具（wuxia_corpus_search）查詢貼近場景語感的原文"
    "片段，再寫出至少一段 NPC 台詞填入該 event 的 dialogue 欄位。"
    "不要更動編劇定下的事件結構、id、觸發條件、分支——只補上台詞。"
    "回傳補完 dialogue 後的完整 Script JSON。"
)
_ORIG_BEAT_EXPAND = (
    "使用者的劇情需求：\n\n"
    "REQ\n\n"
    "拆書人已整理出的人物/變數素材：\n\n"
    f"{_EXTRACTION.model_dump_json()}\n\n"
    "請把這段劇情排成分章大綱與逐場戲的 beat 清單：至少 1-2 章"
    "（chapters），每章至少 1 個 beat，每個 beat 需含 id、"
    "chapter_id（對應到某個 chapter 的 id）、summary（這場戲的"
    "梗概）、npc_ids（涉及哪些登場角色，id 需來自上方素材）、"
    "causal_deps（依賴哪些前置 beat 的 id，沒有就留空陣列）。"
    "不要寫場景細節或台詞，那是後面「江湖代言人」的活兒。"
)
_ORIG_SCENE_WRITE = (
    "請把以下這一場戲的 beat 展開成一個完整的 event。session 內含"
    "登場 NPC 設定、目前這場戲的 beat，以及（若有）已完成的前情"
    "場次摘要——已完成場次是本場戲不可牴觸的既定事實：\n\n"
    f"{_SESSION.model_dump_json()}\n\n"
    'event 的 id 欄位請填 "e1"。依每位 NPC 的 '
    "identity/personality/speech_style，使用語料庫檢索工具"
    "（wuxia_corpus_search）查詢貼近場景語感的原文片段，寫出至少"
    "一段台詞。location、triggers、branches 依 beat 的 summary、"
    "已完成場次摘要與因果合理補上，不得與已完成場次矛盾。"
)


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
