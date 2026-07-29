<div align="center">

# BiXiaScribe

**Retrieve wuxia source text with RAG, hand it to a multi-agent pipeline that writes a structured wuxia RPG script.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/passpier/BiXiaScribe?style=social)](https://github.com/passpier/BiXiaScribe/stargazers)

[繁體中文](./README.md) | [English](./README.en.md)

</div>

---

## What is this?

BiXiaScribe is a wuxia (武俠) RPG script generator. Give it one line of a story requirement
(e.g. "a Shaolin disciple leaves the temple to investigate a massacre"), and it retrieves
relevant passages from your own corpus of wuxia novels, then hands them to three specialized
LLM agents (writer → dialogue → proofreader) that produce a structured script JSON — NPCs,
events, branching choices, triggers — usable as source material for downstream game production
(e.g. RPG Maker).

## Why this architecture for wuxia scripts

Compared to just prompting ChatGPT directly for a script, BiXiaScribe differs in:

- **RAG retrieval over real source text, not just the model's imagination of "wuxia voice"** —
  it indexes your own corpus and feeds retrieved passages into the dialogue agent's prompt, so
  wording and move names stay closer to the source material.
- **A Chinese-aware chunker** — measures length in characters and prefers splitting at
  paragraph/punctuation boundaries, instead of reusing token-splitting logic built for English NLP.
- **Structured output with automated cross-reference validation** — `npc_id`/`next_event_id`
  cross-references are re-checked in Python, not just trusted because the proofreader agent
  says it's fine.
- **Local-first, runnable end-to-end at zero cost** — the default embedding backend is the
  local `bge-m3` model (offline, free, no API key); the LLM also has a `fake` mode so tests
  never make a real API call.

## Key results

**Stage 1 — hybrid retrieval vs. vector-only** (`scripts/eval_retrieval.py`, index of 14 full
金庸 novels + 11 webnovel books, 14 wuxia queries): under a strict comparison (only the single
most-relevant chunk, `--top-k 1`), both modes hit the same source-match rate, but **term-match
rate (does the top chunk actually contain the exact move/sect name?) is 75% for vector-only vs.
91.7% for hybrid** — character-bigram BM25 catches proper-noun matches vector search tends to
miss. (At the default `--top-k 5`, both saturate at 100% — this query set is too easy at that
granularity to show a gap; full results in
[`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md) *(in Chinese)*.)

**Stage 2 — 5-way per-agent model split A/B** (`scripts/eval_generation.py`, n=10/variant, 2026-07-29):

| Variant | Success | avg retrieval_calls | zero-call runs | avg tokens |
|---|---|---|---|---|
| baseline (all deepseek-chat) | 10/10 | 2.10 | 4/10 | 16,492 |
| prose-split (dialogue → qwen3-235b) | 10/10 | 0.40 | 6/10 | 13,166 |
| dialogue-control-openai (dialogue → gpt-4o-mini) | 10/10 | 3.30 | 0/10 | 28,002 |
| dialogue-control-qwen (dialogue → qwen3-30b) | 10/10 | 0.00 | 10/10 | 10,851 |
| cheap-ends (writer/proof → qwen3-30b) | 0/10 | — | — | — |

The default stays `baseline` (all three roles on `deepseek/deepseek-chat`) — fastest, cheapest,
structurally richest, and no other variant beats it on every axis. A non-obvious finding:
`retrieval_calls` shows that "the model supports function calling" is not the same guarantee as
"it reliably chooses to call the tool in a CrewAI ReAct loop" — the qwen family shows low or
zero tool-call rates in practice even where OpenRouter's metadata says tool-calling is supported.
Full analysis and line-by-line prose comparison in [`docs/MILESTONES.md`](./docs/MILESTONES.md).

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # only needed for the Gemini embedding backend or Stage 2
```

```bash
# 1. Build an index (sample corpus, done in seconds, no API key)
python scripts/build_index.py --corpus tests/sample_corpus.txt

# 2. Query the index (default: hybrid = vector + BM25)
python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3

# 3. Check the backend/API key/index are wired up before spending a token
python scripts/generate_script.py --requirement "test" --preflight-only

# 4. Generate a script (needs LLM_BACKEND=openrouter + OPENROUTER_API_KEY)
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

Drop your own `.txt` files into `data/corpus/` (not assumed UTF-8). Full commands for switching
corpora/embedding backends and comparing retrieval/model-split quality are in
[`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md) *(in Chinese)*.

## Output format

`script.json` looks roughly like this (full field definitions in
[`src/bixiascribe/schema.py`](./src/bixiascribe/schema.py)):

```json
{
  "title": "...",
  "premise": "...",
  "variables": [{ "id": "...", "name": "...", "initial": "..." }],
  "npcs": [{ "id": "...", "name": "...", "identity": "...", "personality": "...", "speech_style": "..." }],
  "events": [
    {
      "id": "...",
      "title": "...",
      "location": "...",
      "triggers": [...],
      "dialogue": [{ "npc_id": "...", "line": "...", "emotion": "..." }],
      "branches": [{ "id": "...", "choice_text": "...", "next_event_id": "..." }]
    }
  ]
}
```

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Vector store | [Chroma](https://www.trychroma.com/) (embedded `PersistentClient`, local folder) |
| Embedding | `bge-m3` ([FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), local) / `gemini-embedding-001` (Google API) |
| Multi-agent framework | [CrewAI](https://www.crewai.com/) |
| LLM routing | [OpenRouter](https://openrouter.ai/) (via CrewAI's `LLM` + litellm's `openrouter/` prefix) |
| Validation | [pydantic](https://docs.pydantic.dev/) |

Supported environments: Python ≥ 3.12 (`crewai` requires ≥ 3.10; this repo standardizes on 3.12).

## Project status

- ✅ Stage 1: Chinese-aware RAG indexing (txt → chunking → embedding → Chroma), resumable.
- ✅ Hybrid retrieval (vector + hand-rolled BM25, see Key Results above).
- ✅ Stage 2: 3-agent script generation (writer → dialogue → proofreader), structured JSON + cross-reference validation.
- ✅ Dual-backend switching (embedding/LLM both have an offline/free mode); unit tests never hit a real API.
- 📋 Streamlit UI (planned)

## Further Reading

- [`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md) *(in Chinese)* — design rationale, full command reference, install troubleshooting.
- [`docs/MILESTONES.md`](./docs/MILESTONES.md) — progress tracking, full A/B experiment data and line-by-line analysis.
- [`CLAUDE.md`](./CLAUDE.md) — architecture/interface notes written for an AI coding agent, equally useful for humans.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — local dev setup, test, and lint commands.

## Contributing

Contributions of any kind are welcome — bug reports, suggestions, or code:

- 🐛 **Found a bug?** Open an issue via the [bug report template](https://github.com/passpier/BiXiaScribe/issues/new?template=bug_report.md).
- 💡 **Have a feature idea?** Use the [feature request template](https://github.com/passpier/BiXiaScribe/issues/new?template=feature_request.md), or start a [discussion](https://github.com/passpier/BiXiaScribe/discussions).
- 🔧 **Want to contribute code?** See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## License

This project's code is licensed under the [MIT License](./LICENSE).

> ⚠️ This license covers only the source code in this repo. The wuxia novel corpus under
> `data/corpus/`, and third-party model weights such as `bge-m3` / `gemini-embedding-001`,
> are not distributed with this repo — use them under their own respective licenses.

## Contact

💬 [Discussions](https://github.com/passpier/BiXiaScribe/discussions) ・ 🐛 [Issues](https://github.com/passpier/BiXiaScribe/issues) ・ 👤 [@passpier](https://github.com/passpier)

> This is a solo side project — response times may be irregular, thanks for your patience 🙏

<div align="center">

⭐ If this project helped you, consider giving it a star!

</div>
