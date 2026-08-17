#!/usr/bin/env python3
"""CLI: run the Stage 2 (legacy) or Stage 2b (layered) generation pipeline
once and print (or save) the resulting Script JSON.

Requires an existing Chroma index (see scripts/build_index.py) for the
對話/scene_writer agent's corpus retrieval to have anything to work with, and
LLM_BACKEND=openrouter + OPENROUTER_API_KEY set for real generation.

Usage:
    python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案"
    python scripts/generate_script.py --requirement "..." --out script.json

    # Stage 2b layered pipeline (see CLAUDE.md "Stage 2b"), checkpointed
    # under .bixia_state/<run_id>/ -- resumable if interrupted:
    python scripts/generate_script.py --requirement "..." --pipeline-mode layered
    python scripts/generate_script.py --requirement "..." --pipeline-mode layered \\
        --run-id 1786720605-req-0728cc739f   # resume a specific run
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402
from bixiascribe.crew.orchestrator import run_layered  # noqa: E402
from bixiascribe.crew.pipeline import (  # noqa: E402
    PipelineError,
    RunReport,
    run_pipeline_with_report,
)
from bixiascribe.generation import preflight  # noqa: E402
from bixiascribe.review import requirement_slug  # noqa: E402


def _print_report(report: RunReport) -> None:
    print("--- run report ---", file=sys.stderr)
    if report.mode == "layered":
        print(
            f"models: extractor={report.model_extractor} "
            f"beat_expander={report.model_beat_expander} "
            f"scene_writer={report.model_scene_writer}",
            file=sys.stderr,
        )
    else:
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
            "⚠ 對話/scene_writer agent 從未呼叫 wuxia_corpus_search — 本次生成沒有語料佐證 "
            "(check that the dialogue/scene_writer model supports function calling/tool use).",
            file=sys.stderr,
        )
    elif report.retrieval_queries:
        print(f"retrieval queries: {report.retrieval_queries}", file=sys.stderr)
    if report.mode == "layered":
        print(f"scenes generated: {report.scenes_generated}", file=sys.stderr)
        print(
            f"session doc: max_tokens={report.session_doc_max_tokens} "
            f"omitted_total={report.session_doc_omitted_total}",
            file=sys.stderr,
        )
        print(
            f"causal validation ({report.causal_validation}): "
            f"{len(report.causal_problems)} problem(s), "
            f"{report.causal_repair_attempts} repair attempt(s)",
            file=sys.stderr,
        )


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
    parser.add_argument(
        "--pipeline-mode",
        choices=("legacy", "layered"),
        default=None,
        help="'legacy' (default, config.PIPELINE_MODE) or 'layered' (Stage 2b, "
        "checkpointed under .bixia_state/<run_id>/). See CLAUDE.md 'Stage 2b'.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Layered-mode only: resume a specific .bixia_state/<run_id>/ checkpoint "
        "instead of starting a fresh run. Ignored (with a warning) in legacy mode.",
    )
    args = parser.parse_args()

    mode = (args.pipeline_mode or config.PIPELINE_MODE).strip().lower()
    if args.run_id and mode != "layered":
        print(
            f"⚠ --run-id={args.run_id!r} 被忽略——目前 pipeline_mode={mode!r}，"
            "--run-id 只在 --pipeline-mode layered 下有意義。",
            file=sys.stderr,
        )

    problems = preflight()
    if args.preflight_only:
        print(f"pipeline_mode: {mode}")
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

    resolved_run_id = None
    try:
        if mode == "layered":
            # run_layered() doesn't hand the resolved run_id back on its
            # own (RunReport has no such field) -- resolve it the same way
            # generation.generate() does so it can be printed for --run-id
            # resume on the next invocation.
            resolved_run_id = args.run_id or (
                f"{int(time.time())}-{requirement_slug(args.requirement)}"
            )
            script, report = run_layered(
                args.requirement, run_id=resolved_run_id, verbose=not args.quiet
            )
        else:
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
    if resolved_run_id:
        print(f"run_id: {resolved_run_id}  (pass --run-id to resume)", file=sys.stderr)


if __name__ == "__main__":
    main()
