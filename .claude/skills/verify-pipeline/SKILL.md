---
name: verify-pipeline
description: Runs the chunking unit tests, a smoke-test index build against the sample corpus, and a sample retrieval query, to confirm the end-to-end RAG pipeline still works after a change. Use after editing chunking.py, embedding.py, indexer.py, or the build_index/test_retrieval scripts.
---

Run these steps in order from the repo root, using the project's `.venv`. Stop and report the failure if any
step errors — do not continue to the next step.

1. Unit tests for the chunker (no API key, no external deps):
   ```bash
   python tests/test_chunking.py
   ```

2. Smoke-test the full pipeline against the sample corpus (uses whatever `EMBED_BACKEND` is set in `.env`,
   defaults to local `bge-m3`, no API key needed):
   ```bash
   python scripts/build_index.py --corpus tests/sample_corpus.txt --reset
   ```
   `--reset` is used here so the smoke test always builds a clean collection instead of skipping
   already-indexed chunks.

3. Confirm retrieval works end-to-end:
   ```bash
   python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3
   ```
   Check that results come back with reasonable (low) cosine distances and non-empty previews.

Summarize pass/fail for each of the three steps at the end.
