"""Central configuration for the BiXiaScribe RAG indexing pipeline.

Loads secrets from `.env` (via python-dotenv) and defines the constants
shared across chunking, embedding, and indexing.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from . import length

# Project root = two levels up from this file (src/bixiascribe/config.py -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# --- Embedding ---
# Local BGE-M3, offline, no API key, no quota. EMBED_BACKEND is not an env
# var to switch anymore -- it's kept as a constant purely because
# embedding.py::CorpusEmbeddingFunction.name() bakes it into the Chroma
# collection name (see EMBED_MODEL/EMBED_DIM below), and changing that
# string would make an existing data/chroma/ collection unopenable.
EMBED_BACKEND = "bge-m3"

# Kept as distinct constants only for backward-compatible collection
# naming (see EMBED_BACKEND above) -- BGE-M3 itself ignores them.
EMBED_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# Local (BGE-M3) backend
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-m3").strip()
LOCAL_EMBED_DIM = 1024
LOCAL_EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "").strip()  # "" = auto-detect
LOCAL_EMBED_USE_FP16 = False  # safe default for CPU/MPS
LOCAL_EMBED_MAX_LENGTH = 1024
LOCAL_EMBED_BATCH_SIZE = 16  # CPU/MPS-friendly

# Used by indexer/embedding code; aliased rather than inlined so
# embedding.py::CorpusEmbeddingFunction.name()'s collection-naming string
# stays stable if these ever move.
EMBED_MODEL = LOCAL_EMBED_MODEL
EMBED_DIM = LOCAL_EMBED_DIM
EMBED_BATCH_SIZE = LOCAL_EMBED_BATCH_SIZE

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

# Whether the 對話/scene_writer agents get the wuxia_corpus_search tool at
# all -- distinct from RETRIEVAL_MODE above, which only chooses *how* an
# enabled retrieval call is scored (hybrid vs vector-only). Off entirely
# skips loading the embedding model / opening Chroma for that run, which is
# both a cost lever (retrieval is the largest token cost per CLAUDE.md's
# model-split A/B numbers -- each tool call injects several ~1000-char
# corpus excerpts into the prompt, resent every subsequent turn) and a
# quality question worth A/B'ing (does a model with strong native wuxia
# register need the corpus at all?). Same degrade-not-crash convention as
# CAUSAL_VALIDATION: any value not recognized as falsy defaults to enabled.
_retrieval_enabled = os.environ.get("RETRIEVAL_ENABLED", "true").strip().lower()
RETRIEVAL_ENABLED = _retrieval_enabled not in ("false", "0", "no", "off")

# --- Chroma ---
COLLECTION_NAME = "wuxia_corpus"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"

# --- eval / review UI output layout ---
# out/ is gitignored -- these are local-only artifacts from
# scripts/eval_generation.py, read back by src/bixiascribe/review.py.
OUT_DIR = PROJECT_ROOT / "out"
EVAL_SCRIPTS_DIR = OUT_DIR / "eval"
EVAL_REQUIREMENTS_FILE = PROJECT_ROOT / "eval" / "script_requirements.txt"
RUN_LOG_GLOB = "generation_runs*.jsonl"

# --- Layered/stateful generation pipeline ---
# crew/orchestrator.py persists per-stage checkpoints under BIXIA_STATE_DIR
# (gitignored, one subdirectory per run_id) so a crashed/interrupted layered
# run can resume without re-calling the LLM for already-completed stages.
_bixia_state_env = os.environ.get("BIXIA_STATE_DIR", "").strip()
BIXIA_STATE_DIR = Path(_bixia_state_env) if _bixia_state_env else PROJECT_ROOT / ".bixia_state"

# "legacy" (default): the original single-shot run_pipeline_with_report().
# "layered": crew/orchestrator.py's stateful extract -> beats -> scenes ->
# proofread pipeline (run_layered()), checkpointed and resumable. Read by
# generation.generate()/GenerationJob (so both the UI and
# scripts/eval_generation.py --pipeline-mode honor it) and by
# scripts/generate_script.py --pipeline-mode. Keeping this default at
# "legacy" is a zero-cost rollback lever if "layered" misbehaves in the
# wild -- both pipelines remain fully reachable regardless of this default
# by passing pipeline_mode= explicitly or --pipeline-mode on the CLI.
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "legacy").strip().lower()

# Scenes within one causal-independent batch (see
# crew/orchestrator.py::plan_batches()) are generated concurrently, up to
# this many at once. Retrieval itself stays serialized by crew/tools.py's
# _retrieval_lock regardless, so this only overlaps the LLM round-trips.
# 3 is a conservative default for OpenRouter free/low-tier rate limits.
SCENE_CONCURRENCY = max(1, int(os.environ.get("SCENE_CONCURRENCY", "3") or "3"))

# crew/context_builder.py::build_session_document() trims the
# SessionDocument handed to each scene_writer call (character cards + prior
# scene summaries) until its estimated token count fits under this budget.
# No tokenizer dependency -- estimate_tokens() uses the same "measure
# Chinese text in characters" approach as CHUNK_SIZE above, biased high for
# CJK so the budget stays conservative rather than silently overrunning.
SESSION_DOC_MAX_TOKENS = max(1, int(os.environ.get("SESSION_DOC_MAX_TOKENS", "4000") or "4000"))

# Real-time causal consistency validation behavior, checked after every
# scene is generated (crew/causal.py::check_scene_consistency()), not just
# at the final proofread stage.
# "off"    -- skip entirely, no causal_graph.json is produced
# "warn"   -- validate and record problems in RunReport, but always checkpoint the scene
# "repair" -- (default) try up to MAX_REPAIR_ATTEMPTS targeted repairs first; if
#             problems remain, record and still checkpoint (same as "warn")
# "strict" -- if problems remain after repair, refuse to checkpoint the scene and
#             raise PipelineError
# Any unrecognized value falls back to "repair" rather than erroring, mirroring
# how PIPELINE_MODE/LLM_BACKEND degrade rather than crash on a typo'd env var.
_causal_validation = os.environ.get("CAUSAL_VALIDATION", "repair").strip().lower()
CAUSAL_VALIDATION = _causal_validation if _causal_validation in (
    "off",
    "warn",
    "repair",
    "strict",
) else "repair"

# Target script length, read by crew/tasks.py's task-builder functions. "short"
# (default) is today's floor -- 1-2 chapters x 1 beat (layered) / 2 events
# (legacy), unchanged since before this knob existed. "medium"/"long" ask for
# progressively more chapters/beats/dialogue depth, or a
# "custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT" string
# with any subset of fields (missing ones derived from events) -- see
# length.py. This is a prompt-level target only: nothing in schema.py
# enforces it and no LLM output cap exists besides LLM_MAX_TOKENS above, so a
# model can under- or over-shoot it. Any unrecognized value falls back to
# "short", same degrade-not-crash convention as CAUSAL_VALIDATION --
# canonicalizing here (rather than just membership-checking) also means
# SCRIPT_LENGTH is always a resolvable value downstream.
_script_length = os.environ.get("SCRIPT_LENGTH", "short").strip()
SCRIPT_LENGTH = length.parse_length_spec(_script_length).canonical


# --- LLM (CrewAI script generation) ---
# "openrouter" (default): real generation via OpenRouter, model per-agent
# configurable below. "fake": offline canned responses, no API key/network/
# cost -- used by tests/test_crew_pipeline.py.
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

# Optional ceiling on a single LLM call's completion tokens -- unset (the
# default) passes nothing through to crewai's LLM, i.e. today's behavior:
# whatever the OpenRouter provider's own default max_completion_tokens is.
# Worth setting once SCRIPT_LENGTH=medium/long is in play: a longer target
# means longer Event JSON per scene, and a low provider-side default can
# silently truncate it before it becomes invalid JSON -- see llm.py::build_llm().
_llm_max_tokens = os.environ.get("LLM_MAX_TOKENS", "").strip()
LLM_MAX_TOKENS = int(_llm_max_tokens) if _llm_max_tokens else None

# OpenRouter provider routing (https://openrouter.ai/docs/features/provider-routing),
# forwarded via litellm's extra_body -- see llm.py::build_llm(). LLM_PROVIDER_ONLY
# pins to a comma-separated allowlist of provider slugs (e.g. a model whose
# cheapest route doesn't support tool calling, which would silently break
# wuxia_corpus_search -- see eval/model_variants.json's notes on this exact
# failure mode). LLM_PROVIDER_SORT ("price" or "throughput") only takes effect
# when LLM_PROVIDER_ONLY is unset. Both unset (default) is today's behavior:
# OpenRouter's own default routing, no pinning.
LLM_PROVIDER_ONLY = [
    p.strip() for p in os.environ.get("LLM_PROVIDER_ONLY", "").split(",") if p.strip()
]
_llm_provider_sort = os.environ.get("LLM_PROVIDER_SORT", "").strip().lower()
LLM_PROVIDER_SORT = _llm_provider_sort if _llm_provider_sort in ("price", "throughput") else ""


def require_openrouter_key() -> str:
    """Return the OpenRouter API key or raise a clear error if it's missing."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your "
            "OpenRouter API key (get one at https://openrouter.ai/keys), or set "
            "LLM_BACKEND=fake to run offline (used by tests)."
        )
    return OPENROUTER_API_KEY
