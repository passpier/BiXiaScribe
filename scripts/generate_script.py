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

from bixiascribe import config  # noqa: E402
from bixiascribe.crew.pipeline import (  # noqa: E402
    PipelineError,
    RunReport,
    run_pipeline_with_report,
)
from bixiascribe.generation import preflight  # noqa: E402


def _print_report(report: RunReport) -> None:
    print("--- run report ---", file=sys.stderr)
    print(
        f"models: writer={report.model_writer} dialogue={report.model_dialogue} "
        f"proof={report.model_proof}",
        file=sys.stderr,
    )
    print(f"elapsed: {report.elapsed_s:.1f}s", file=sys.stderr)
    if report.token_usage:
        print(f"token usage: {report.token_usage}", file=sys.stderr)
    print(f"coerced from: {report.coerced_from}", file=sys.stderr)
    print(f"repair attempts: {report.repair_attempts}", file=sys.stderr)
    print(
        f"retrieval: {report.retrieval_calls} call(s), "
        f"{report.retrieval_failures} failure(s)",
        file=sys.stderr,
    )
    if report.retrieval_calls == 0:
        print(
            "⚠ 對話 agent 從未呼叫 wuxia_corpus_search — 本次生成沒有語料佐證 "
            "(check that LLM_MODEL_DIALOGUE supports function calling/tool use).",
            file=sys.stderr,
        )
    elif report.retrieval_queries:
        print(f"retrieval queries: {report.retrieval_queries}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", required=True, help="劇情需求 (Chinese OK).")
    parser.add_argument(
        "--out", type=Path, default=None, help="Write the Script JSON here instead of stdout."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress CrewAI's per-agent verbose output."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight checks and print resolved model ids, then exit (no tokens spent).",
    )
    args = parser.parse_args()

    problems = preflight()
    if args.preflight_only:
        print(
            f"models: writer={config.LLM_MODEL_WRITER} dialogue={config.LLM_MODEL_DIALOGUE} "
            f"proof={config.LLM_MODEL_PROOF}"
        )
        if problems:
            print("Problems found:")
            for p in problems:
                print(f"  - {p}")
            sys.exit(1)
        print("Preflight OK.")
        return

    if problems:
        for p in problems:
            print(f"生成前檢查失敗：{p}", file=sys.stderr)
        sys.exit(1)

    try:
        script, report = run_pipeline_with_report(args.requirement, verbose=not args.quiet)
    except (PipelineError, RuntimeError) as exc:
        print(f"生成失敗：{exc}", file=sys.stderr)
        if isinstance(exc, PipelineError) and exc.report is not None:
            _print_report(exc.report)
        sys.exit(1)

    payload = script.model_dump_json(indent=2, exclude_none=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out} ({len(script.events)} events, {len(script.npcs)} npcs).")
    else:
        print(payload)

    _print_report(report)


if __name__ == "__main__":
    main()
