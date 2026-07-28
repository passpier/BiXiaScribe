"""CrewAI tool wrapping ..retrieval.retrieve(), so the 對話 (dialogue) agent
can pull 武俠語感 (period-appropriate wuxia prose/register) from the indexed
corpus while writing NPC lines."""
from __future__ import annotations

from crewai.tools import BaseTool

from ..retrieval import CollectionNotFoundError, get_query_collection, retrieve

# Module-level cache, same pattern as lexical.py's BM25 index cache:
# get_query_collection() opens a fresh Chroma collection handle, and
# retrieval.py's own docstring recommends callers that loop (as this tool
# does, once per dialogue-agent tool call) reuse one instead of reopening it
# every time.
_collection = None


def _get_cached_collection():
    global _collection
    if _collection is None:
        _collection = get_query_collection()
    return _collection


class WuxiaRetrievalTool(BaseTool):
    name: str = "wuxia_corpus_search"
    description: str = (
        "在已索引的武俠語料庫中搜尋與查詢語意相近的原文片段，"
        "用來揣摩招式命名、稱謂用語、行文語感。輸入一句話描述你想要的語感"
        "（例如角色設定或場景摘要），回傳最相關的幾個原文片段。"
    )

    def _run(self, query: str, top_k: int = 3) -> str:
        try:
            collection = _get_cached_collection()
            chunks = retrieve(query, top_k=top_k, collection=collection)
        except CollectionNotFoundError as exc:
            return f"（語料庫尚未建立索引，無法檢索：{exc}）"
        except Exception as exc:  # noqa: BLE001
            # Retrieval is meant to *help* wuxia voice, not gate the whole
            # generation on it -- a broken embedding backend or a Chroma
            # internals change (e.g. `include` defaults) shouldn't abort the
            # dialogue agent. Degrade to a message it can work around.
            return f"（語料庫檢索失敗，請改用你自己的武俠語感撰寫：{exc}）"

        if not chunks:
            return "（查無相關語料片段）"

        return "\n\n".join(
            f"[片段 {i + 1} | 來源: {chunk.source}]\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )
