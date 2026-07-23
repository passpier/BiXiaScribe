#!/usr/bin/env python3
"""CLI: pull a wuxia-leaning subset of the `wdndev/webnovel-chinese` HuggingFace
dataset and write it out as per-book .txt files into data/corpus/, so the
existing Stage 1 indexing pipeline (scripts/build_index.py) can pick it up
completely unchanged.

The dataset has ~9,000 books / ~600k rows (one row per chapter) and NO genre
label -- it's overwhelmingly contemporary web fiction (玄幻/都市/言情...), not
wuxia. Dumping it wholesale into the wuxia_corpus index would dilute 武俠語感
rather than add to it, so this script is deliberately a two-step workflow:

  1. --list-titles   : stream a sample and print book titles + row/char counts,
                        so you can eyeball which titles are actually wuxia and
                        build your own whitelist (--titles inline or --titles-file).
  2. (fetch mode)     : stream again, keep only books matching --titles /
                        --titles-file and/or --title-keywords, aggregate their chapters, and
                        write out webnovel-<title>.txt files. The `webnovel-`
                        prefix becomes the chunk `source` metadata once indexed,
                        so it's always distinguishable from the 金庸 novels
                        already in data/corpus/ -- no changes needed to
                        indexer.py, chunking.py, embedding.py, or the Stage 2
                        retrieval tool.

Usage:
    # 1. See what's in there before committing to anything
    python scripts/prepare_webnovel.py --list-titles --max-scan 50000

    # 2. Small trial run with an explicit whitelist (most accurate)
    python scripts/prepare_webnovel.py --titles "八荒剑神,傲剑凌云" \\
        --max-books 5 --max-chars-per-book 500000
    # ...or from a file, one title per line:
    python scripts/prepare_webnovel.py --titles-file my_wuxia_titles.txt

    # ...or a keyword pass (title substring match, less precise --
    #    keywords like 劍/俠 also hit 玄幻 novels, so review before indexing)
    python scripts/prepare_webnovel.py --title-keywords 劍,俠,江湖,門派,武林 \\
        --max-books 20

    # 3. Then, as usual (do NOT pass --reset -- indexing is resumable and will
    #    only embed the new webnovel- chunks):
    python scripts/build_index.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402

DATASET_NAME = "wdndev/webnovel-chinese"

# Title-substring keywords used as a fallback filter when no --titles-file is
# given. Deliberately loose (wuxia-flavored, not wuxia-exclusive) -- 玄幻/仙俠
# novels often share these words, so a --titles-file whitelist built from
# --list-titles output is the more accurate path.
DEFAULT_KEYWORDS = ["劍", "俠", "刀", "江湖", "門派", "武林", "掌門", "少林", "武當"]

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w一-鿿\-]+")


def _safe_filename(title: str) -> str:
    """Turn a book title into a filesystem-safe filename stem."""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", title.strip()).strip("_")
    return cleaned or "untitled"


def _load_titles_file(path: Path) -> set[str]:
    if not path.is_file():
        raise SystemExit(
            f"--titles-file: '{path}' not found. It must be a FILE PATH with one book "
            f"title per line. For a single inline title, use --titles '傲世丹神' instead."
        )
    titles = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        titles.add(line.rsplit("-", 1)[0].strip() if "-" in line else line)
    return titles


def _parse_inline_titles(raw: str) -> set[str]:
    """Parse a comma-separated --titles value. Accepts a bare title or a
    "書名-作者" pair (common when copying from ranking lists) -- in the
    latter case only the part before the last "-" is kept, since the
    dataset's `title` field holds just the book title, not "書名-作者"."""
    titles = set()
    for raw_title in raw.split(","):
        raw_title = raw_title.strip()
        if not raw_title:
            continue
        title = raw_title.rsplit("-", 1)[0].strip() if "-" in raw_title else raw_title
        titles.add(title)
    return titles


def _iter_dataset_rows(max_scan: int | None):
    """Yield {"title", "chapter", "text"} dicts by streaming the dataset (no
    39GB download -- rows are pulled on demand)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for this script "
            "(pip install -r requirements.txt)."
        ) from exc

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_scan is not None and i >= max_scan:
            break
        yield row


def cmd_list_titles(args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    for row in _iter_dataset_rows(args.max_scan):
        title = (row.get("title") or "").strip()
        if not title:
            continue
        counts[title] += 1
        chars[title] += len(row.get("text") or "")

    print(f"Scanned {args.max_scan} rows, found {len(counts)} distinct titles:\n")
    for title, n_chapters in counts.most_common(args.top_n):
        print(f"{n_chapters:>5} chapters  {chars[title]:>10,} chars  {title}")
    print(
        "\nPick the wuxia-leaning titles above, then re-run with --titles "
        "'title1,title2,...' or a --titles-file (one title per line)."
    )


def cmd_fetch(args: argparse.Namespace) -> None:
    whitelist = None
    if args.titles_file:
        whitelist = _load_titles_file(args.titles_file)
    if args.titles:
        whitelist = (whitelist or set()) | _parse_inline_titles(args.titles)

    keywords = (
        [k.strip() for k in args.title_keywords.split(",") if k.strip()]
        if args.title_keywords
        else (DEFAULT_KEYWORDS if whitelist is None else [])
    )

    if whitelist is None and not keywords:
        raise SystemExit(
            "Provide --titles, --titles-file, and/or --title-keywords to select books."
        )

    def matches(title: str) -> bool:
        if whitelist is not None:
            bare = title.rsplit("-", 1)[0].strip() if "-" in title else title
            if title in whitelist or bare in whitelist:
                return True
        if keywords and any(k in title for k in keywords):
            return True
        return False

    args.out_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list[str]] = {}
    char_counts: Counter[str] = Counter()
    finished: set[str] = set()

    def flush(title: str) -> None:
        text = "\n\n".join(buffers.pop(title, []))
        if len(text) < args.min_chars:
            return
        out_path = args.out_dir / f"webnovel-{_safe_filename(title)}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  wrote {out_path.name}  ({len(text):,} chars)")

    for row in _iter_dataset_rows(args.max_scan):
        if len(finished) >= args.max_books:
            break
        title = (row.get("title") or "").strip()
        if not title or title in finished or not matches(title):
            continue

        text = row.get("text") or ""
        if char_counts[title] >= args.max_chars_per_book:
            finished.add(title)
            flush(title)
            continue

        buffers.setdefault(title, []).append(text)
        char_counts[title] += len(text)

    # Flush any books still under their cap when scanning ended.
    for title in list(buffers):
        finished.add(title)
        flush(title)

    print(f"\nDone. {len(finished)} book(s) written to {args.out_dir}.")
    print("Next: python scripts/build_index.py   (no --reset needed -- resumable)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=20000,
        help="Max rows to stream through (dataset has ~600k rows total; default: 20000).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.CORPUS_DIR,
        help=f"Output directory for webnovel-*.txt files (default: {config.CORPUS_DIR}).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list-titles",
        action="store_true",
        help="Scan and print title/chapter/char counts instead of writing files.",
    )
    parser.add_argument(
        "--top-n", type=int, default=50, help="With --list-titles, how many titles to print."
    )

    parser.add_argument(
        "--titles",
        help=(
            "Comma-separated inline whitelist of exact book titles "
            "(e.g. '八荒剑神,傲剑凌云'). A trailing '-作者' on any entry is "
            "stripped automatically, since the dataset's title field has no "
            "author suffix. Combine with --titles-file if you have both."
        ),
    )
    parser.add_argument(
        "--titles-file",
        type=Path,
        help=(
            "Whitelist FILE PATH, one exact book title per line "
            "(not an inline list -- use --titles for that)."
        ),
    )
    parser.add_argument(
        "--title-keywords",
        help=(
            "Comma-separated title substrings to match (e.g. 劍,俠,江湖). Used as a "
            f"fallback when --titles-file is omitted (default: {','.join(DEFAULT_KEYWORDS)})."
        ),
    )
    parser.add_argument(
        "--max-books", type=int, default=20, help="Max number of books to write out."
    )
    parser.add_argument(
        "--max-chars-per-book",
        type=int,
        default=2_000_000,
        help="Per-book character cap, to bound output size.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=10_000,
        help="Skip books shorter than this many characters (likely incomplete/noise).",
    )

    args = parser.parse_args()

    if args.list_titles:
        cmd_list_titles(args)
    else:
        cmd_fetch(args)


if __name__ == "__main__":
    main()
