#!/usr/bin/env python3
"""CLI: run the Stage 2 CrewAI pipeline (編劇 -> 對話 -> 校對) once and print
(or save) the resulting Script JSON.

Requires an existing Chroma index (see scripts/build_index.py) for the
對話 agent's corpus retrieval to have anything to work with, and
LLM_BACKEND=openrouter + OPENROUTER_API_KEY set for real generation.

Usage:
    python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案"
    python scripts/generate_script.py --requirement "..." --out script.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.crew.pipeline import PipelineError, run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", required=True, help="劇情需求 (Chinese OK).")
    parser.add_argument(
        "--out", type=Path, default=None, help="Write the Script JSON here instead of stdout."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress CrewAI's per-agent verbose output."
    )
    args = parser.parse_args()

    try:
        script = run_pipeline(args.requirement, verbose=not args.quiet)
    except (PipelineError, RuntimeError) as exc:
        print(f"生成失敗：{exc}", file=sys.stderr)
        sys.exit(1)

    payload = script.model_dump_json(indent=2, exclude_none=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out} ({len(script.events)} events, {len(script.npcs)} npcs).")
    else:
        print(payload)


if __name__ == "__main__":
    main()
