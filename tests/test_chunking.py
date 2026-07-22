"""Pure unit tests for chunking.chunk_text — no API key, no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.chunking import chunk_text  # noqa: E402


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_returns_single_chunk():
    text = "獨孤九劍乃劍魔獨孤求敗生平絕學。"
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert chunks == [text]


def test_long_text_is_split_and_respects_chunk_size():
    text = "武功招式與內力心法紛雜多變，" * 200  # long, unspaced Chinese text
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100 + 20  # merge may slightly exceed due to overlap seed
        assert c.strip() == c
        assert c != ""


def test_overlap_carries_context_between_chunks():
    text = "。".join(f"第{i}句話內容較長用以撐滿長度需求" for i in range(50))
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert len(chunks) > 1
    # The tail of chunk N should reappear at the head of chunk N+1.
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev[-20:]
        assert nxt.startswith(tail[: min(len(tail), len(nxt))]) or tail in nxt


def test_rejects_invalid_overlap():
    try:
        chunk_text("abc", chunk_size=10, overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    # Allow running as a plain script without pytest.
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All chunking tests passed.")
