#!/usr/bin/env python3
"""CLI: run the Stage 2 CrewAI pipeline across a {model variant} x
{requirement} matrix, append one machine-readable row per run to a JSONL
log, and print a side-by-side aggregate comparison -- the Stage 2
counterpart to scripts/eval_retrieval.py's vector-vs-hybrid comparison.

This exists to answer two questions from docs/MILESTONES.md's "Gap to
product-grade" list without re-reading old terminal scrollback:
  1. Which per-agent model split (see eval/model_variants.json) actually
     produces better structural quality per token spent?
  2. Is retrieval_calls == 0 (the dialogue agent silently never calling
     wuxia_corpus_search -- see CLAUDE.md) a one-off or a pattern across
     variants/requirements?

Deliberately structural-metrics-only (see crew/metrics.py's docstring) --
no LLM-as-judge prose scoring. Reading actual generated dialogue for 武俠
語感 is still a human job; this script saves each run's script to
out/eval/ specifically so that read is easy to do afterward.

Usage:
    # check every variant's ids + key + index, no tokens spent
    python scripts/eval_generation.py --dry-run

    # run the full matrix (default: all variants x all requirements)
    python scripts/eval_generation.py

    # run a subset
    python scripts/eval_generation.py --variants baseline,prose-split --repeat 1

    # re-print the aggregate table from a past run without spending anything
    python scripts/eval_generation.py --from-jsonl out/generation_runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_script import preflight  # noqa: E402

from bixiascribe import config  # noqa: E402
from bixiascribe.crew.metrics import script_metrics  # noqa: E402
from bixiascribe.crew.pipeline import PipelineError, run_pipeline_with_report  # noqa: E402
from bixiascribe.llm import ModelChoice  # noqa: E402
from bixiascribe.review import load_jsonl, requirement_slug  # noqa: E402

DEFAULT_VARIANTS_FILE = Path(__file__).resolve().parents[1] / "eval" / "model_variants.json"
DEFAULT_REQUIREMENTS_FILE = config.EVAL_REQUIREMENTS_FILE
DEFAULT_JSONL = config.OUT_DIR / "generation_runs.jsonl"
DEFAULT_SCRIPTS_DIR = config.EVAL_SCRIPTS_DIR


def _load_variants(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_requirements(path: Path) -> list[str]:
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


def dry_run(variants: list[dict]) -> int:
    """Preflight-only check: real backend/key/index problems, plus that
    every variant's model ids are non-empty. No tokens spent."""
    problems = preflight()
    for variant in variants:
        for role in ("writer", "dialogue", "proof"):
            if not variant.get(role):
                problems.append(f"variant {variant['name']!r} missing {role!r} model id")

    if problems:
        print("Problems found:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"Preflight OK -- {len(variants)} variant(s) ready:")
    for v in variants:
        print(f"  - {v['name']}: writer={v['writer']} dialogue={v['dialogue']} proof={v['proof']}")
    return 0


def run_matrix(
    variants: list[dict],
    requirements: list[str],
    repeat: int,
    jsonl_path: Path,
    scripts_dir: Path,
    verbose: bool,
) -> list[dict]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = len(variants) * len(requirements) * repeat
    done = 0

    with jsonl_path.open("a", encoding="utf-8") as f:
        for variant in variants:
            models = ModelChoice(
                writer=variant["writer"], dialogue=variant["dialogue"], proof=variant["proof"]
            )
            for requirement in requirements:
                for rep in range(repeat):
                    done += 1
                    print(
                        f"[{done}/{total}] variant={variant['name']!r} "
                        f"requirement={requirement[:20]!r}... rep={rep}",
                        file=sys.stderr,
                    )
                    row = _run_one(
                        variant["name"], requirement, models, scripts_dir, verbose, rep
                    )
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()

    return rows


def _run_one(
    variant_name: str,
    requirement: str,
    models: ModelChoice,
    scripts_dir: Path,
    verbose: bool,
    rep: int = 0,
) -> dict:
    row: dict = {"variant": variant_name, "ts": time.time(), "ok": False, "error": None}
    try:
        script, report = run_pipeline_with_report(requirement, verbose=verbose, models=models)
    except PipelineError as exc:
        row["error"] = str(exc)
        if exc.report is not None:
            row.update(exc.report.to_dict())
        return row

    row["ok"] = True
    row.update(report.to_dict())
    row.update(script_metrics(script))

    # rep > 0 gets a suffix so `--repeat N` doesn't silently overwrite the same
    # file N times, keeping only the last rep's script -- reading these by hand
    # is the only way 武俠語感 gets judged (see module docstring), so losing
    # earlier reps loses evidence. rep 0 keeps the original filename so past
    # single-repeat runs' paths stay stable.
    suffix = f"__rep{rep}" if rep > 0 else ""
    out_path = scripts_dir / f"{variant_name}__{requirement_slug(requirement)}{suffix}.json"
    out_path.write_text(script.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
    row["script_path"] = str(out_path)
    return row


def print_aggregate(rows: list[dict]) -> None:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    print("=" * 70)
    print(f"Aggregate over {len(rows)} run(s), {len(by_variant)} variant(s):\n")

    for variant, variant_rows in by_variant.items():
        ok_rows = [r for r in variant_rows if r.get("ok")]
        n = len(variant_rows)
        n_ok = len(ok_rows)
        print(f"[{variant}] {n_ok}/{n} succeeded")
        if not ok_rows:
            continue

        def _mean(key: str, rows: list[dict] = ok_rows) -> float:
            vals = [r[key] for r in rows if key in r and r[key] is not None]
            return statistics.mean(vals) if vals else float("nan")

        zero_retrieval = sum(1 for r in ok_rows if r.get("retrieval_calls") == 0)
        coerced_counts = Counter(r.get("coerced_from") for r in ok_rows)
        # RunReport already captures the actual query strings passed to
        # wuxia_corpus_search (crew/tools.py's RetrievalStats.queries) -- surface
        # them so a nonzero retrieval_calls can be told apart from a token
        # gesture vs. a substantive lookup, without opening each JSONL row by hand.
        all_queries = [q for r in ok_rows for q in (r.get("retrieval_queries") or [])]

        print(f"    elapsed_s        avg={_mean('elapsed_s'):.1f}")
        print(
            f"    retrieval_calls  avg={_mean('retrieval_calls'):.2f}  "
            f"zero-call runs={zero_retrieval}/{n_ok}"
        )
        if all_queries:
            preview = "; ".join(all_queries[:5])
            more = f" (+{len(all_queries) - 5} more)" if len(all_queries) > 5 else ""
            print(f"    retrieval_queries {preview}{more}")
        print(f"    repair_attempts  avg={_mean('repair_attempts'):.2f}")
        print(f"    coerced_from     {dict(coerced_counts)}")
        print(
            f"    events={_mean('events'):.1f}  npcs={_mean('npcs'):.1f}  "
            f"branches={_mean('branches'):.1f}  dialogue_lines={_mean('dialogue_lines'):.1f}"
        )
        print(
            f"    events_with_dialogue_pct={_mean('events_with_dialogue_pct'):.1%}  "
            f"npc_speaking_pct={_mean('npc_speaking_pct'):.1%}  "
            f"avg_line_chars={_mean('avg_line_chars'):.1f}"
        )
        token_totals = [
            r["token_usage"].get("total_tokens")
            for r in ok_rows
            if r.get("token_usage") and r["token_usage"].get("total_tokens") is not None
        ]
        if token_totals:
            print(f"    total_tokens     avg={statistics.mean(token_totals):.0f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants-file", type=Path, default=DEFAULT_VARIANTS_FILE)
    parser.add_argument("--requirements-file", type=Path, default=DEFAULT_REQUIREMENTS_FILE)
    parser.add_argument(
        "--variants", default=None, help="Comma-separated subset of variant names to run."
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeats per (variant, requirement).")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--scripts-dir", type=Path, default=DEFAULT_SCRIPTS_DIR)
    parser.add_argument("--verbose", action="store_true", help="Show CrewAI's per-agent output.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preflight checks only, no tokens spent."
    )
    parser.add_argument(
        "--from-jsonl",
        type=Path,
        default=None,
        help="Skip running anything; print the aggregate table from an existing JSONL log.",
    )
    args = parser.parse_args()

    if args.from_jsonl:
        rows = load_jsonl(args.from_jsonl)
        print_aggregate(rows)
        return

    all_variants = _load_variants(args.variants_file)
    if args.variants:
        wanted = {name.strip() for name in args.variants.split(",")}
        all_variants = [v for v in all_variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in all_variants}
        if missing:
            print(f"Unknown variant name(s): {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        sys.exit(dry_run(all_variants))

    problems = preflight()
    if problems:
        for p in problems:
            print(f"生成前檢查失敗：{p}", file=sys.stderr)
        sys.exit(1)

    requirements = _load_requirements(args.requirements_file)
    rows = run_matrix(
        all_variants, requirements, args.repeat, args.jsonl, args.scripts_dir, args.verbose
    )
    print_aggregate(rows)
    print(f"Rows appended to {args.jsonl}; scripts saved under {args.scripts_dir}/")


if __name__ == "__main__":
    main()
