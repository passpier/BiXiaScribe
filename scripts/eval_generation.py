#!/usr/bin/env python3
"""CLI: run the Stage 2 CrewAI pipeline across a {model variant} x
{requirement} matrix, append one machine-readable row per run to a JSONL
log, and print a side-by-side aggregate comparison -- the Stage 2
counterpart to scripts/eval_retrieval.py's vector-vs-hybrid comparison.

This exists to answer two recurring questions without re-reading old
terminal scrollback:
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
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config, length, pricing  # noqa: E402
from bixiascribe.generation import Variant, generate, preflight  # noqa: E402
from bixiascribe.review import load_jsonl  # noqa: E402

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


def _format_session_doc_max_tokens(value: object) -> str:
    if value is None:
        return f"env default ({config.SESSION_DOC_MAX_TOKENS})"
    if isinstance(value, int) and value <= 0:
        return "unlimited"
    return str(value)


def _estimate_matrix_cost(
    variants: list[dict], n_requirements: int, repeat: int, pipeline_mode: str
) -> tuple[float, list[str]]:
    """Rough pre-spend cost estimate for the whole matrix, scaled from
    out/generation_runs*.jsonl's recorded historical mean token usage per
    mode/script_length (falling back to a fixed guess when no history
    exists for a given script_length -- an unrun combination, e.g. "long",
    has no historical mean to scale from). Returns (total_usd, warnings) --
    never raises; a missing price just contributes $0 and a warning, since
    this is a before-you-spend sanity check, not a billing system."""
    # Historical means from out/generation_runs_phase_c.jsonl (legacy) and
    # out/generation_runs_phase5.jsonl (layered), both mode="short" (today's
    # baseline length, events=2 -> events_scale=1.0). No historical data
    # exists yet for medium/long/custom -- scale linearly off the resolved
    # events target as a rough guess (see length.py::LengthSpec.events_scale).
    _BASE_TOKENS = {
        "legacy": {"prompt": 13000, "completion": 3200},
        "layered": {"prompt": 23000, "completion": 4500},
    }

    warnings: list[str] = []
    prices = pricing.load_prices()
    base = _BASE_TOKENS.get(pipeline_mode, _BASE_TOKENS["legacy"])
    total = 0.0
    for variant_row in variants:
        variant = Variant.from_dict(variant_row)
        scale = length.parse_length_spec(
            variant.script_length or config.SCRIPT_LENGTH
        ).events_scale
        models = variant.to_model_choice()
        model_ids = {models.writer, models.dialogue, models.proof}
        priced = [prices[m] for m in model_ids if m in prices]
        if not priced:
            warnings.append(f"variant {variant.name!r}: no price entry for any of {model_ids}")
            continue
        chosen_price = priced[0] if len(priced) == 1 else min(
            priced, key=lambda p: p.prompt_usd_per_1m + p.completion_usd_per_1m
        )
        per_run = chosen_price.cost(base["prompt"] * scale, base["completion"] * scale)
        total += per_run * n_requirements * repeat
    return total, warnings


def dry_run(variants: list[dict], pipeline_mode: str, requirements: list[str], repeat: int) -> int:
    """Preflight-only check: real backend/key/index problems, plus that
    every variant's model ids are non-empty. No tokens spent.

    Also prints the resolved pipeline_mode/SCENE_CONCURRENCY/per-variant
    session_doc_max_tokens/script_length, all six per-agent-role model ids
    (not just writer/dialogue/proof -- the layered-only extractor/
    beat_expander/scene_writer roles used to silently fall back to
    config.LLM_MODEL_* regardless of what a variant set, which is exactly
    the bug this printout exists to make visible), and a rough pre-spend
    cost estimate for the whole matrix (see _estimate_matrix_cost()) so a
    caller can confirm the matrix before spending anything.
    Warns rather than blocks -- these are footguns for the
    compressed-vs-untrimmed quality regression, not hard errors for every
    other use of this script:
    - pipeline_mode="layered" with SCENE_CONCURRENCY > 1: dispatch_batch()
      snapshots completed_scenes once per batch (before dispatch), so if
      the beat_expander emits beats with no causal_deps, every scene in the
      first batch would see zero prior scenes regardless of
      session_doc_max_tokens -- silently nullifying the comparison.
    - a variant sets session_doc_max_tokens while pipeline_mode="legacy":
      the knob is layered-only and would be silently ignored.
    """
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

    print(
        f"Preflight OK -- pipeline_mode={pipeline_mode}  "
        f"SCENE_CONCURRENCY={config.SCENE_CONCURRENCY}  "
        f"SCRIPT_LENGTH default={config.SCRIPT_LENGTH}"
    )
    if pipeline_mode == "layered" and config.SCENE_CONCURRENCY > 1:
        print(
            "  WARNING: SCENE_CONCURRENCY > 1 with pipeline_mode=layered -- "
            "dispatch_batch() snapshots completed scenes once per batch, so a "
            "beat sheet with no causal_deps can put every scene in one batch "
            "with zero prior-scene visibility, regardless of "
            "session_doc_max_tokens. Set SCENE_CONCURRENCY=1 for a context-"
            "compression comparison to be meaningful."
        )
    print(f"{len(variants)} variant(s) ready:")
    for v in variants:
        variant = Variant.from_dict(v)
        models = variant.to_model_choice()
        sdmt = v.get("session_doc_max_tokens")
        script_length = variant.script_length or config.SCRIPT_LENGTH
        print(f"  - {v['name']}  script_length={script_length}")
        print(
            f"      writer={models.writer}  dialogue={models.dialogue}  proof={models.proof}"
        )
        print(
            f"      extractor={models.extractor}  beat_expander={models.beat_expander}  "
            f"scene_writer={models.scene_writer}"
        )
        print(f"      session_doc_max_tokens={_format_session_doc_max_tokens(sdmt)}")
        if sdmt is not None and pipeline_mode != "layered":
            print(
                f"      WARNING: variant {v['name']!r} sets session_doc_max_tokens "
                f"but pipeline_mode={pipeline_mode!r} -- this is layered-only and "
                "will be silently ignored."
            )

    total_cost, cost_warnings = _estimate_matrix_cost(
        variants, len(requirements), repeat, pipeline_mode
    )
    for w in cost_warnings:
        print(f"  WARNING: {w}")
    print(
        f"Estimated matrix cost: ~${total_cost:.4f} for "
        f"{len(variants)} variant(s) x {len(requirements)} requirement(s) x {repeat} rep(s) "
        f"(rough, scaled from historical token means -- not a quote)."
    )
    return 0


def run_matrix(
    variants: list[dict],
    requirements: list[str],
    repeat: int,
    jsonl_path: Path,
    scripts_dir: Path,
    verbose: bool,
    pipeline_mode: str | None = None,
    script_length: str | None = None,
) -> list[dict]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = len(variants) * len(requirements) * repeat
    done = 0

    with jsonl_path.open("a", encoding="utf-8") as f:
        for variant_row in variants:
            variant = Variant.from_dict(variant_row)
            for requirement in requirements:
                for rep in range(repeat):
                    done += 1
                    print(
                        f"[{done}/{total}] variant={variant.name!r} "
                        f"requirement={requirement[:20]!r}... rep={rep}",
                        file=sys.stderr,
                    )
                    row = _run_one(
                        variant,
                        requirement,
                        scripts_dir,
                        verbose,
                        rep,
                        pipeline_mode,
                        script_length,
                    )
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()

    return rows


def _run_one(
    variant: Variant,
    requirement: str,
    scripts_dir: Path,
    verbose: bool,
    rep: int = 0,
    pipeline_mode: str | None = None,
    script_length: str | None = None,
) -> dict:
    # Thin adapter over bixiascribe.generation.generate(), which now owns the
    # persistence + row-shape logic shared with the Stage 3 UI's "生成" mode.
    # jsonl_path=None: run_matrix() below owns the open file handle and does
    # its own crash-resilient write-then-flush per row, so generate() must
    # not also append -- two writers on one file would interleave/duplicate.
    # rep=rep (not None) preserves this harness's `--repeat` semantics
    # (rep 0 keeps the original filename; rep>0 gets a suffix; re-running
    # with the same rep deliberately overwrites) rather than the UI's
    # auto-next-free-rep behavior. pipeline_mode is forwarded straight
    # through (None falls back to config.PIPELINE_MODE, same as generate()).
    result = generate(
        requirement,
        variant,
        variant_name=variant.name,
        rep=rep,
        verbose=verbose,
        scripts_dir=scripts_dir,
        jsonl_path=None,
        pipeline_mode=pipeline_mode,
        script_length=script_length,
    )
    return result.row


def _row_cost(row: dict) -> tuple[float | None, str]:
    """cost_usd/cost_basis for one row, computed by generation.py's
    build_run_row() for any row written after this feature shipped --
    but *retroactively* priced here for older rows (out/generation_runs*.jsonl
    logs from before cost_usd existed) so --from-jsonl can price historical
    data without re-running anything. See pricing.estimate_cost()'s
    docstring for the by_role/uniform/uniform_lower_bound/unknown_price
    basis values this can return."""
    if "cost_usd" in row:
        return row.get("cost_usd"), row.get("cost_basis") or ""
    if row.get("mode") == "layered":
        models = {
            "extractor": row.get("model_extractor") or "",
            "beat_expander": row.get("model_beat_expander") or "",
            "scene_writer": row.get("model_scene_writer") or "",
            "proof": row.get("model_proof") or "",
        }
    else:
        models = {
            "writer": row.get("model_writer") or "",
            "dialogue": row.get("model_dialogue") or "",
            "proof": row.get("model_proof") or "",
        }
    return pricing.estimate_cost(
        row.get("token_usage"), models, token_usage_by_role=row.get("token_usage_by_role")
    )


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

        # Cost accounting (pricing.py) -- computed here (not just read from
        # the row) so --from-jsonl can price older logs written before
        # cost_usd existed; see _row_cost()'s docstring.
        costs: list[float] = []
        cost_bases: Counter[str] = Counter()
        unit_costs: list[dict[str, float | None]] = []
        for r in ok_rows:
            cost, basis = _row_cost(r)
            cost_bases[basis] += 1
            if cost is not None:
                costs.append(cost)
                unit_costs.append(pricing.quality_unit_costs(cost, r))
        if costs:
            print(
                f"    cost_usd         avg=${statistics.mean(costs):.4f}  "
                f"total=${sum(costs):.4f}  basis={dict(cost_bases)}"
            )
            per_event = [u["usd_per_event"] for u in unit_costs if u["usd_per_event"] is not None]
            per_line = [
                u["usd_per_dialogue_line"]
                for u in unit_costs
                if u["usd_per_dialogue_line"] is not None
            ]
            if per_event or per_line:
                pe = f"${statistics.mean(per_event):.4f}" if per_event else "n/a"
                pl = f"${statistics.mean(per_line):.4f}" if per_line else "n/a"
                print(f"    usd_per_event    avg={pe}   usd_per_dialogue_line avg={pl}")
        elif any(b != "" for b in cost_bases):
            print(f"    cost_usd         no priceable rows (basis={dict(cost_bases)})")

        # Continuity metrics -- gated on presence so aggregating an
        # older JSONL log (rows written before these keys existed) doesn't
        # print a row of "nan%". review.py's overview_rows() recomputes
        # script_metrics() from disk, so only *this* harness's stored rows
        # can be missing them, not the UI.
        if any("repeated_dialogue_pct" in r for r in ok_rows):
            print(
                f"    continuity       distinct_title={_mean('distinct_event_title_pct'):.1%}  "
                f"repeated_dialogue={_mean('repeated_dialogue_pct'):.1%}  "
                f"prior_ref={_mean('prior_entity_reference_pct'):.1%}  "
                f"connected={_mean('connected_event_pct'):.1%}  "
                f"self_loop={_mean('self_loop_branch_pct'):.1%}"
            )
        if any(r.get("session_doc_omitted_total") is not None for r in ok_rows):
            omitted_vals = [
                r["session_doc_omitted_total"]
                for r in ok_rows
                if r.get("session_doc_omitted_total") is not None
            ]
            max_tokens_vals = {r.get("session_doc_max_tokens") for r in ok_rows}
            print(
                f"    session_doc      max_tokens={sorted(max_tokens_vals, key=str)}  "
                f"omitted_total avg={statistics.mean(omitted_vals):.1f}"
            )
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
    parser.add_argument(
        "--pipeline-mode",
        choices=("legacy", "layered"),
        default=None,
        help=(
            "Which pipeline to run every variant against (default: "
            "config.PIPELINE_MODE, i.e. the PIPELINE_MODE env var). The "
            "compressed-vs-untrimmed quality regression needs 'layered' here "
            "for Variant.session_doc_max_tokens to have any effect."
        ),
    )
    parser.add_argument(
        "--script-length",
        default=None,
        help=(
            "Override every variant's script_length for this run: "
            "'short'/'medium'/'long', or a custom "
            "'custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT' "
            "string (default: each variant's own script_length, falling back "
            "to config.SCRIPT_LENGTH, i.e. the SCRIPT_LENGTH env var). See "
            "length.py."
        ),
    )
    args = parser.parse_args()

    if args.from_jsonl:
        rows = load_jsonl(args.from_jsonl)
        print_aggregate(rows)
        return

    pipeline_mode = (args.pipeline_mode or config.PIPELINE_MODE).strip().lower()

    all_variants = _load_variants(args.variants_file)
    if args.variants:
        wanted = {name.strip() for name in args.variants.split(",")}
        all_variants = [v for v in all_variants if v["name"] in wanted]
        missing = wanted - {v["name"] for v in all_variants}
        if missing:
            print(f"Unknown variant name(s): {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

    requirements = _load_requirements(args.requirements_file)

    if args.dry_run:
        sys.exit(dry_run(all_variants, pipeline_mode, requirements, args.repeat))

    problems = preflight()
    if problems:
        for p in problems:
            print(f"生成前檢查失敗：{p}", file=sys.stderr)
        sys.exit(1)

    rows = run_matrix(
        all_variants,
        requirements,
        args.repeat,
        args.jsonl,
        args.scripts_dir,
        args.verbose,
        pipeline_mode=args.pipeline_mode,
        script_length=args.script_length,
    )
    print_aggregate(rows)
    print(f"Rows appended to {args.jsonl}; scripts saved under {args.scripts_dir}/")


if __name__ == "__main__":
    main()
