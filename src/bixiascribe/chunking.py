"""Lightweight Chinese-aware recursive text chunker.

Pure Python, no external dependencies. Designed for unspaced Chinese prose
(e.g. wuxia novels) where whitespace-based tokenization is meaningless, so
length is measured in characters and splits prefer sentence-ending
punctuation over whitespace.
"""
from __future__ import annotations

# Ordered from "most preferred split point" to "least preferred". The empty
# string as a final fallback means "split anywhere" (hard cut by character).
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ""]


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text on separator, keeping the separator attached to the
    preceding piece (so punctuation isn't lost from the chunk content)."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to every piece except the last (which had
    # nothing after it).
    pieces = [p + separator for p in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])
    return pieces


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split `text` into pieces no longer than chunk_size, trying separators
    in priority order and recursing into any piece that's still too long."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # Should not happen since "" is always the last separator, but guard anyway.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_separators = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)

    result: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            if piece:
                result.append(piece)
        else:
            result.extend(_recursive_split(piece, chunk_size, rest_separators))
    return result


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks close to chunk_size, carrying
    `overlap` characters of trailing context from one chunk into the next."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            # Seed the next chunk with the overlap tail of the current one.
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + piece
        else:
            current += piece

    if current.strip():
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split `text` into overlapping chunks of roughly
    `chunk_size` characters, preferring paragraph/sentence boundaries.

    Args:
        text: raw input text (may be unspaced Chinese prose).
        chunk_size: target max characters per chunk.
        overlap: characters of trailing context carried into the next chunk.
        separators: split-priority list; defaults to DEFAULT_SEPARATORS.

    Returns:
        List of non-empty, whitespace-trimmed chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _recursive_split(text, chunk_size, seps)
    chunks = _merge_pieces(pieces, chunk_size, overlap)
    return [c.strip() for c in chunks if c.strip()]
