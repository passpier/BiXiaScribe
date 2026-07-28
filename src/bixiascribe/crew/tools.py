"""CrewAI tool wrapping ..retrieval.retrieve(), so the 對話 (dialogue) agent
can pull 武俠語感 (period-appropriate wuxia prose/register) from the indexed
corpus while writing NPC lines."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from crewai.tools import BaseTool

from ..retrieval import CollectionNotFoundError, get_query_collection, retrieve

# Module-level cache, same pattern as lexical.py's BM25 index cache:
# get_query_collection() opens a fresh Chroma collection handle, and
# retrieval.py's own docstring recommends callers that loop (as this tool
# does, once per dialogue-agent tool call) reuse one instead of reopening it
# every time.
_collection = None
# Guards _collection's init -- see WuxiaRetrievalTool's docstring: CrewAI can
# run several wuxia_corpus_search calls from one LLM turn concurrently, and
# without this two threads could both open a fresh collection handle at once.
_collection_lock = threading.Lock()

# Guards RetrievalStats mutations below (calls/failures/queries), for the same
# concurrency reason -- plain += from multiple threads can lose increments.
_stats_lock = threading.Lock()

# Serializes the actual retrieval work (collection lookup + retrieve() call)
# across concurrent tool calls. A retrieval query is one short sentence, so
# serializing it is cheap; the alternative -- N concurrent BGE-M3 encode()
# calls sharing one loaded model -- is untested and not worth the risk versus
# the near-zero cost of queuing.
#
# Lock ordering: _retrieval_lock is always acquired first (in _run below),
# and _collection_lock / _stats_lock only ever nest inside it, never the
# other way around, so there's no cycle and thus no deadlock risk.
_retrieval_lock = threading.Lock()


def _get_cached_collection():
    global _collection
    if _collection is None:
        with _collection_lock:
            if _collection is None:
                _collection = get_query_collection()
    return _collection


@dataclass
class RetrievalStats:
    """How much (if at all) the dialogue agent actually used
    wuxia_corpus_search during a run. A real model can produce a perfectly
    valid Script while never calling the tool -- see CLAUDE.md's warning
    about LLM_MODEL_DIALOGUE needing function-calling support -- so
    `calls == 0` after a run is the signal that RAG grounding silently
    never fired."""

    calls: int = 0
    failures: int = 0
    queries: list[str] = field(default_factory=list)


_stats = RetrievalStats()


def get_stats() -> RetrievalStats:
    return _stats


def reset_stats() -> None:
    """Zero the counters. Callers that run one pipeline pass (pipeline.py)
    should call this at the start so the report reflects that run, not
    process-lifetime totals across multiple run_pipeline() calls."""
    global _stats
    with _stats_lock:
        _stats = RetrievalStats()


class WuxiaRetrievalTool(BaseTool):
    name: str = "wuxia_corpus_search"
    description: str = (
        "在已索引的武俠語料庫中搜尋與查詢語意相近的原文片段，"
        "用來揣摩招式命名、稱謂用語、行文語感。輸入一句話描述你想要的語感"
        "（例如角色設定或場景摘要），回傳最相關的幾個原文片段。"
    )

    def _run(self, query: str, top_k: int = 3) -> str:
        with _stats_lock:
            _stats.calls += 1
            _stats.queries.append(query)

        # CrewAI can fire several wuxia_corpus_search calls from one LLM turn
        # concurrently (see module-level lock docstrings above) -- serialize
        # the actual lookup so concurrent calls queue instead of racing to
        # build the embedding model / BM25 index / Chroma handle at once.
        with _retrieval_lock:
            try:
                collection = _get_cached_collection()
                chunks = retrieve(query, top_k=top_k, collection=collection)
            except CollectionNotFoundError as exc:
                with _stats_lock:
                    _stats.failures += 1
                return f"（語料庫尚未建立索引，無法檢索：{exc}）"
            except Exception as exc:  # noqa: BLE001
                # Retrieval is meant to *help* wuxia voice, not gate the whole
                # generation on it -- a broken embedding backend or a Chroma
                # internals change (e.g. `include` defaults) shouldn't abort
                # the dialogue agent. Degrade to a message it can work around.
                with _stats_lock:
                    _stats.failures += 1
                return f"（語料庫檢索失敗，請改用你自己的武俠語感撰寫：{exc}）"

        if not chunks:
            return "（查無相關語料片段）"

        return "\n\n".join(
            f"[片段 {i + 1} | 來源: {chunk.source}]\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )
