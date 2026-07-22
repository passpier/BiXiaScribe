#!/usr/bin/env python3
"""CLI: chunk + embed + index a corpus (file or directory of .txt) into Chroma.

Usage:
    python scripts/build_index.py [--corpus PATH] [--reset]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402
from bixiascribe.indexer import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=config.CORPUS_DIR,
        help=f"Path to a .txt file or a directory of .txt files (default: {config.CORPUS_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the collection from scratch instead of upserting.",
    )
    args = parser.parse_args()

    build_index(args.corpus, reset=args.reset)


if __name__ == "__main__":
    main()
