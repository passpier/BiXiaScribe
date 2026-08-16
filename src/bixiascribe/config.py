"""Central configuration for the BiXiaScribe RAG indexing pipeline.

Loads secrets from `.env` (via python-dotenv) and defines the constants
shared across chunking, embedding, and indexing.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels up from this file (src/bixiascribe/config.py -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# --- Embedding ---
# "bge-m3" (default): local, offline, no API quota. "gemini": Google's API,
# subject to free-tier rate limits/daily quota.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "bge-m3").strip().lower()

EMBED_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# Gemini backend
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_EMBED_DIM = 1536  # truncated from the model's native 3072 dims via output_dimensionality
GEMINI_EMBED_BATCH_SIZE = 100  # documents per embed_content call, to stay friendly with free-tier rate limits

# Local (BGE-M3) backend
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-m3").strip()
LOCAL_EMBED_DIM = 1024
LOCAL_EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "").strip()  # "" = auto-detect
LOCAL_EMBED_USE_FP16 = False  # safe default for CPU/MPS
LOCAL_EMBED_MAX_LENGTH = 1024
LOCAL_EMBED_BATCH_SIZE = 16  # CPU/MPS-friendly

# Active backend's dim/batch size, used by indexer/embedding code.
EMBED_MODEL = GEMINI_EMBED_MODEL if EMBED_BACKEND == "gemini" else LOCAL_EMBED_MODEL
EMBED_DIM = GEMINI_EMBED_DIM if EMBED_BACKEND == "gemini" else LOCAL_EMBED_DIM
EMBED_BATCH_SIZE = GEMINI_EMBED_BATCH_SIZE if EMBED_BACKEND == "gemini" else LOCAL_EMBED_BATCH_SIZE

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Per-file character cap applied only to webnovel-*.txt corpus files (see
# scripts/prepare_webnovel.py), not to the 金庸 novels. The webnovel dataset
# has no genre label and is ~6x the 金庸 corpus by size, mostly 玄幻/仙俠 rather
# than 武俠 -- indexing it uncapped would dilute 武俠語感 in retrieval results.
# Capping each book keeps 金庸 dominant in the index while still adding some
# webnovel 語感 variety. 0 disables the cap (index webnovel files in full).
WEBNOVEL_MAX_CHARS = int(os.environ.get("WEBNOVEL_MAX_CHARS", "1000000"))

# --- Retrieval ---
# "hybrid" (default): fuse BM25 keyword search with vector search via
# Reciprocal Rank Fusion -- helps wuxia proper nouns (獨孤九劍, 六脈神劍) where
# pure vector search under-performs exact/keyword matches. "vector": the
# original vector-only behavior.
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid").strip().lower()

# --- Chroma ---
COLLECTION_NAME = "wuxia_corpus"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"

# --- Stage 2 eval / Stage 3 review UI output layout ---
# out/ is gitignored -- these are local-only artifacts from
# scripts/eval_generation.py, read back by src/bixiascribe/review.py.
OUT_DIR = PROJECT_ROOT / "out"
EVAL_SCRIPTS_DIR = OUT_DIR / "eval"
EVAL_REQUIREMENTS_FILE = PROJECT_ROOT / "eval" / "script_requirements.txt"
RUN_LOG_GLOB = "generation_runs*.jsonl"

# --- Stage 2b: layered/stateful generation pipeline (BiXiaScribe 重構) ---
# crew/orchestrator.py persists per-stage checkpoints under BIXIA_STATE_DIR
# (gitignored, one subdirectory per run_id) so a crashed/interrupted layered
# run can resume without re-calling the LLM for already-completed stages.
_bixia_state_env = os.environ.get("BIXIA_STATE_DIR", "").strip()
BIXIA_STATE_DIR = Path(_bixia_state_env) if _bixia_state_env else PROJECT_ROOT / ".bixia_state"

# "legacy" (default): the original single-shot run_pipeline_with_report().
# "layered": crew/orchestrator.py's stateful extract -> beats -> scenes ->
# proofread pipeline (run_layered_pipeline()/run_layered()). Not yet wired
# into any caller (scripts/generate_script.py, generation.py) -- this only
# defines the switch, so both pipelines remain reachable by calling the
# right function directly regardless of this env var.
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "legacy").strip().lower()

# Phase 4: scenes within one causal-independent batch (see
# crew/orchestrator.py::plan_batches()) are generated concurrently, up to
# this many at once. Retrieval itself stays serialized by crew/tools.py's
# _retrieval_lock regardless, so this only overlaps the LLM round-trips.
# 3 is a conservative default for OpenRouter free/low-tier rate limits.
SCENE_CONCURRENCY = max(1, int(os.environ.get("SCENE_CONCURRENCY", "3") or "3"))

# Phase 5: crew/context_builder.py::build_session_document() trims the
# SessionDocument handed to each scene_writer call (character cards + prior
# scene summaries) until its estimated token count fits under this budget.
# No tokenizer dependency -- estimate_tokens() uses the same "measure
# Chinese text in characters" approach as CHUNK_SIZE above, biased high for
# CJK so the budget stays conservative rather than silently overrunning.
SESSION_DOC_MAX_TOKENS = max(1, int(os.environ.get("SESSION_DOC_MAX_TOKENS", "4000") or "4000"))

# Phase 6: real-time causal consistency validation behavior, checked after
# every scene is generated (crew/causal.py::check_scene_consistency()), not
# just at the final proofread stage.
# "off"    -- skip entirely, no causal_graph.json is produced (pre-Phase-6 behavior)
# "warn"   -- validate and record problems in RunReport, but always checkpoint the scene
# "repair" -- (default) try up to MAX_REPAIR_ATTEMPTS targeted repairs first; if
#             problems remain, record and still checkpoint (same as "warn")
# "strict" -- if problems remain after repair, refuse to checkpoint the scene and
#             raise PipelineError (the refactor plan's literal 5.4 rollback behavior)
# Any unrecognized value falls back to "repair" rather than erroring, mirroring
# how PIPELINE_MODE/LLM_BACKEND degrade rather than crash on a typo'd env var.
_causal_validation = os.environ.get("CAUSAL_VALIDATION", "repair").strip().lower()
CAUSAL_VALIDATION = _causal_validation if _causal_validation in (
    "off",
    "warn",
    "repair",
    "strict",
) else "repair"


def require_api_key() -> str:
    """Return the Gemini API key or raise a clear error if it's missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your "
            "Gemini API key (get one for free at https://aistudio.google.com/apikey)."
        )
    return GEMINI_API_KEY


# --- LLM (Stage 2: CrewAI script generation) ---
# "openrouter" (default): real generation via OpenRouter, model per-agent
# configurable below. "fake": offline canned responses, no API key/network/
# cost -- used by tests/test_crew_pipeline.py, mirrors the EMBED_BACKEND
# fake-vs-real split above.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openrouter").strip().lower()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Shared default model, with per-agent overrides so each role can be tuned
# independently (e.g. a cheaper/logic-oriented model for the writer, a
# Chinese-prose-strong model for dialogue) without touching code -- see the
# 武俠 RPG 劇本 RAG 架構方案 doc's model-assignment table.
LLM_MODEL = os.environ.get("LLM_MODEL", "openrouter/deepseek/deepseek-chat").strip()
LLM_MODEL_WRITER = os.environ.get("LLM_MODEL_WRITER", "").strip() or LLM_MODEL
LLM_MODEL_DIALOGUE = os.environ.get("LLM_MODEL_DIALOGUE", "").strip() or LLM_MODEL
LLM_MODEL_PROOF = os.environ.get("LLM_MODEL_PROOF", "").strip() or LLM_MODEL


def require_openrouter_key() -> str:
    """Return the OpenRouter API key or raise a clear error if it's missing."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your "
            "OpenRouter API key (get one at https://openrouter.ai/keys), or set "
            "LLM_BACKEND=fake to run offline (used by tests)."
        )
    return OPENROUTER_API_KEY
