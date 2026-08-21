"""Unit tests for the RETRIEVAL_ENABLED / Variant.use_retrieval /
--no-retrieval knob: turning wuxia_corpus_search off entirely for a run,
distinct from RETRIEVAL_MODE (hybrid vs vector-only, which stays on either
way). Runs against LLM_BACKEND=fake -- no API key, no network, no Chroma.

Mirrors tests/test_script_length.py's approach for the agent/task layer
(byte-for-byte prompt regression guard for use_retrieval=True, the default)
and tests/test_generation.py's approach for the generate()/Variant layer.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["LLM_BACKEND"] = "fake"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config, generation  # noqa: E402

config.LLM_BACKEND = "fake"

from bixiascribe.crew.agents import make_dialogue_agent, make_scene_writer_agent  # noqa: E402
from bixiascribe.crew.context_builder import build_session_document  # noqa: E402
from bixiascribe.crew.orchestrator import run_layered  # noqa: E402
from bixiascribe.crew.pipeline import run_pipeline_with_report  # noqa: E402
from bixiascribe.crew.tasks import (  # noqa: E402
    make_dialogue_task,
    make_scene_write_task,
    make_writer_task,
)
from bixiascribe.schema import Beat, ExtractionResult  # noqa: E402

_EXTRACTION = ExtractionResult(npcs=[], variables=[], props=[], branch_candidates=[])
_BEAT = Beat(id="b1", chapter_id="c1", summary="s", npc_ids=[], causal_deps=[])
_SESSION = build_session_document(_BEAT, _EXTRACTION, [])
REQUIREMENT = "少林俗家弟子奉命下山，追查一樁滅門血案背後的血衣門餘孽。"

# Byte-for-byte, from before this knob existed -- same convention as
# tests/test_script_length.py's _ORIG_* guards. use_retrieval=True must
# reproduce these exactly.
_ORIG_DIALOGUE = (
    "上一步「編劇」產出的事件骨架見對話上下文（context）。請針對每一個 "
    "event，依照其中 NPC 的 identity/personality/speech_style，"
    "使用語料庫檢索工具（wuxia_corpus_search）查詢貼近場景語感的原文"
    "片段，再寫出至少一段 NPC 台詞填入該 event 的 dialogue 欄位。"
    "不要更動編劇定下的事件結構、id、觸發條件、分支——只補上台詞。"
    "回傳補完 dialogue 後的完整 Script JSON。"
)


def test_dialogue_agent_has_no_tools_when_retrieval_disabled():
    agent = make_dialogue_agent(use_retrieval=False)
    assert agent.tools == []


def test_dialogue_agent_has_tool_when_retrieval_enabled():
    agent = make_dialogue_agent(use_retrieval=True)
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "wuxia_corpus_search"


def test_scene_writer_agent_has_no_tools_when_retrieval_disabled():
    agent = make_scene_writer_agent(use_retrieval=False)
    assert agent.tools == []


def test_dialogue_task_prompt_byte_identical_when_retrieval_enabled():
    writer_task = make_writer_task(REQUIREMENT, make_dialogue_agent())
    task = make_dialogue_task(make_dialogue_agent(), writer_task, use_retrieval=True)
    assert task.description == _ORIG_DIALOGUE


def test_dialogue_task_prompt_drops_tool_mention_when_retrieval_disabled():
    writer_task = make_writer_task(REQUIREMENT, make_dialogue_agent())
    task = make_dialogue_task(make_dialogue_agent(), writer_task, use_retrieval=False)
    assert "wuxia_corpus_search" not in task.description
    assert "語料庫" not in task.description
    assert "直接以你自己的武俠語感" in task.description


def test_scene_write_task_prompt_drops_tool_mention_when_retrieval_disabled():
    task = make_scene_write_task(
        _BEAT, _EXTRACTION, make_scene_writer_agent(), "e1", session=_SESSION,
        use_retrieval=False,
    )
    assert "wuxia_corpus_search" not in task.description
    assert "語料庫" not in task.description


def test_legacy_pipeline_reports_retrieval_disabled():
    script, report = run_pipeline_with_report(
        REQUIREMENT, verbose=False, use_retrieval=False
    )
    assert report.retrieval_enabled is False
    assert report.retrieval_calls == 0
    assert "retrieval_enabled" in report.to_dict()
    assert script.title


def test_legacy_pipeline_defaults_to_retrieval_enabled():
    _, report = run_pipeline_with_report(REQUIREMENT, verbose=False)
    assert report.retrieval_enabled is True


def test_layered_pipeline_reports_retrieval_disabled():
    # Save/restore the original value rather than reconstructing
    # config.PROJECT_ROOT / ".bixia_state" -- same convention as
    # tests/test_generation.py/test_orchestrator.py's _isolated_state_dir(),
    # so this still restores correctly if BIXIA_STATE_DIR was overridden via
    # the environment.
    original_state_dir = config.BIXIA_STATE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.BIXIA_STATE_DIR = Path(tmp)
        try:
            script, report = run_layered(
                REQUIREMENT,
                run_id="norag-test",
                verbose=False,
                use_retrieval=False,
            )
        finally:
            config.BIXIA_STATE_DIR = original_state_dir
    assert report.retrieval_enabled is False
    assert report.mode == "layered"
    assert script.title


def test_variant_round_trips_use_retrieval():
    variant = generation.Variant.from_dict(
        {"name": "v", "writer": "fake/w", "dialogue": "fake/d", "proof": "fake/p",
         "use_retrieval": False}
    )
    assert variant.use_retrieval is False


def test_variant_defaults_use_retrieval_to_none():
    variant = generation.Variant(name="v")
    assert variant.use_retrieval is None


def test_variant_ui_visible_round_trips():
    hidden = generation.Variant.from_dict({"name": "v", "ui_visible": False})
    shown = generation.Variant.from_dict({"name": "v"})
    assert hidden.ui_visible is False
    assert shown.ui_visible is True


def test_at_least_one_shipped_variant_is_hidden_from_the_ui():
    variants = generation.load_variants()
    assert any(not v.ui_visible for v in variants), (
        "expected eval/model_variants.json to hide at least one unreliable/expensive "
        "variant (long-cheap/long-mimo) from the UI's picker via ui_visible=false"
    )


def test_generate_forwards_use_retrieval_into_row():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "不檢索語料測試",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-norag",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            use_retrieval=False,
        )
        assert result.ok
        assert result.row["retrieval_enabled"] is False
        assert result.row["retrieval_calls"] == 0
