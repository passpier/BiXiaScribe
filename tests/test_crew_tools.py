"""Unit tests for WuxiaRetrievalTool (crew/tools.py) with retrieval.retrieve
monkeypatched -- no Chroma index, no embedding backend, no API key needed.

This tool previously had zero test coverage: FakeLLM.supports_function_calling()
is False, so the fake dialogue-agent path never actually invokes it, and a
broken retrieval tool would have passed CI silently. These tests exercise
_run directly instead of going through a crew.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bixiascribe.crew.tools as tools  # noqa: E402
from bixiascribe.retrieval import CollectionNotFoundError, RetrievedChunk  # noqa: E402


def _reset_collection_cache():
    tools._collection = None


def test_run_formats_retrieved_chunks():
    _reset_collection_cache()
    tools._get_cached_collection = lambda: "fake-collection"
    tools.retrieve = lambda query, top_k=3, collection=None: [
        RetrievedChunk(text="劍氣縱橫三萬里", source="笑傲江湖.txt", chunk_index=0),
        RetrievedChunk(text="風清揚傳授劍法", source="笑傲江湖.txt", chunk_index=1),
    ]
    result = tools.WuxiaRetrievalTool()._run("獨孤九劍")
    assert "[片段 1 | 來源: 笑傲江湖.txt]" in result
    assert "劍氣縱橫三萬里" in result
    assert "[片段 2 | 來源: 笑傲江湖.txt]" in result


def test_run_reports_no_results():
    _reset_collection_cache()
    tools._get_cached_collection = lambda: "fake-collection"
    tools.retrieve = lambda query, top_k=3, collection=None: []
    result = tools.WuxiaRetrievalTool()._run("不存在的招式")
    assert result == "（查無相關語料片段）"


def test_run_reports_missing_collection():
    _reset_collection_cache()

    def _raise_not_found():
        raise CollectionNotFoundError("no collection built yet")

    tools._get_cached_collection = _raise_not_found
    result = tools.WuxiaRetrievalTool()._run("獨孤九劍")
    assert "尚未建立索引" in result


def test_run_degrades_on_unexpected_error():
    # This is the path that was previously uncaught: retrieve() can also
    # fail from embedding backend load errors, Chroma internals changes,
    # etc. -- not just a missing collection. The tool must not raise, since
    # that would abort the whole dialogue-agent task over what's meant to be
    # an optional, quality-improving lookup.
    _reset_collection_cache()
    tools._get_cached_collection = lambda: "fake-collection"

    def _raise_runtime(query, top_k=3, collection=None):
        raise RuntimeError("embedding backend exploded")

    tools.retrieve = _raise_runtime
    result = tools.WuxiaRetrievalTool()._run("獨孤九劍")
    assert "檢索失敗" in result
    assert "embedding backend exploded" in result


if __name__ == "__main__":
    test_run_formats_retrieved_chunks()
    test_run_reports_no_results()
    test_run_reports_missing_collection()
    test_run_degrades_on_unexpected_error()
    print("All tests passed.")
