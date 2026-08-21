"""Pre-run / in-run cost and time estimation -- the missing half of
pricing.py's accounting. pricing.py answers "what did a *finished* run
cost"; this module answers "what will/does this run cost and how long will
it take", before or while tokens are actually being spent.

Deliberately dependency-free like pricing.py/review.py (no crewai, no
streamlit import) so it's the one estimation path shared by
scripts/eval_generation.py's `--dry-run`, scripts/generate_script.py's
preflight/pre-spend printout, and ui/app.py's 生成 mode (via
generation.py's thin wrappers) -- see CLAUDE.md's "Pre-run / in-run
estimation" section for the basis-priority rationale and the real-run
finding that motivated it.

Why this exists (concrete finding, not speculation): a real Streamlit
`layered`/`long` run against `deepseek-v4-flash-0731`
(.bixia_state/1787309292-req-d232acf2d8) was cancelled after 19 minutes and
$0.0039 spent, with only 1 of 30 beats generated. Extrapolating the
already-generated stage costs gave ~$0.065 and ~2 hours to finish -- money
was never the issue, but nothing told the user *that* before they had
already waited 19 minutes to find out. Separately, that run's 30 beats
formed a single linear causal chain (plan_batches() returns 30 batches of
width 1), so SCENE_CONCURRENCY had zero effect on it -- exposing that
number (`parallelism`) is as important as the cost/time numbers themselves.

`basis` values, most to least trustworthy (mirrors pricing.estimate_cost()'s
own basis-string convention):
- "measured_run": this run's own already-completed scenes/stages, read back
  from scene_metrics.py sidecars via the caller (this module does no
  checkpoint I/O itself -- see `measured` param below).
- "history_mode_length": out/generation_runs*.jsonl rows matching the same
  pipeline_mode AND script_length (and, loosely, similar models).
- "history_mode": rows matching pipeline_mode only, token/time priors scaled
  by length.LengthSpec.events_scale to the requested script_length.
- "prior": no matching history at all -- a fixed, documented-provenance
  default (see _PRIOR_STAGE_TOKENS below), never silently $0.
- "unknown_price": token counts exist but no eval/model_prices.json entry
  covers the models involved -- cost_usd/seconds are still computed when
  possible (seconds never depends on price), only cost_usd/cost_low/
  cost_high are None. Never guess $0 for a missing price -- pricing.py's
  own docstring explains why that's actively misleading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config, length, pricing

# Roles making up one legacy-pipeline run and one layered-pipeline run's
# non-scene stages -- kept as tuples (not sets) so aggregation order is
# stable/reproducible.
_LEGACY_ROLES = ("writer", "dialogue", "proof")
_LAYERED_STAGE_ROLES = ("extractor", "beat_expander")
_LAYERED_SCENE_ROLE = "scene_writer"

# Fixed-provenance fallback token priors, used only when no historical JSONL
# row matches at all (basis="prior"). Sourced from the two real (non-fake)
# scene_meta_*.json sidecars this project has as of 2026-08-21
# (.bixia_state/1787305976-req-4e61dbe167, .bixia_state/1787309292-
# req-d232acf2d8: ~21.5-21.9k total tokens, ~2 LLM calls, ~226-229s per
# scene against deepseek-v4-flash-0731) plus that same run's extract/
# beat_expand stage token_usage_by_role entries. This replaces
# scripts/eval_generation.py's old _BASE_TOKENS constant, which had drifted
# out of sync with its own docstring (claimed to scale from historical
# means; didn't). n is small (2 real scenes) -- notes on a "prior"-basis
# RunEstimate always say so; prefer basis="history_*" whenever
# out/generation_runs*.jsonl has anything to scale from instead.
_PRIOR_STAGE_TOKENS: dict[str, dict[str, float]] = {
    "extractor": {"prompt": 4276, "completion": 2904, "cached": 0, "seconds": 35},
    "beat_expander": {"prompt": 3168, "completion": 5410, "cached": 0, "seconds": 45},
    "scene_writer": {"prompt": 13614, "completion": 8295, "cached": 3584, "seconds": 227},
    "writer": {"prompt": 3318, "completion": 18039, "cached": 2515, "seconds": 60},
    "dialogue": {
        "prompt": 205686 / 7, "completion": 29457 / 7, "cached": 173069 / 7, "seconds": 90,
    },
    "proof": {"prompt": 25503 / 2, "completion": 24954 / 2, "cached": 0, "seconds": 60},
}

# A layered beat sheet's real beat count consistently exceeds its
# script_length target's nominal `events` upper bound -- e.g.
# .bixia_state/1787309292-req-d232acf2d8 (script_length=long, events target
# "15-24") produced 30 beats, because every chapter also gets its own
# converge beat and length.py's `events` target doesn't account for those.
# Only used to guess a scene count when no real beats.json exists yet
# (basis="prior"/"history_mode" without a `scenes` override) -- always
# superseded by a real beat count the moment one exists (see
# generation.GenerationJob.estimate()).
_SCENES_PER_EVENTS_TARGET = 1.3


@dataclass(frozen=True)
class RunEstimate:
    """Best-effort projection of one run's remaining/total scenes, cost, and
    wall-clock time. Every numeric field is `None` (never a fabricated 0)
    when it can't be computed -- same "unknown, not free" convention
    pricing.estimate_cost() already uses."""

    scenes: int | None = None
    batches: int | None = None
    parallelism: float | None = None
    cost_usd: float | None = None
    cost_low: float | None = None
    cost_high: float | None = None
    seconds: float | None = None
    seconds_low: float | None = None
    seconds_high: float | None = None
    basis: str = "unknown_price"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _StagePrior:
    prompt: float
    completion: float
    cached: float
    seconds: float
    basis: str


def _has_real_usage(row: dict[str, Any]) -> bool:
    """False for a row whose token_usage is all-zero -- the concrete
    signature of an `LLM_BACKEND=fake` run (verified against a real one
    logged during this module's own manual UI testing: `ok=True`,
    `elapsed_s` real, every token count 0, since FakeLLM never calls a
    priced API). Such a row is a legitimate, successful run in every other
    sense (review.py/the UI correctly show it), but it must never be
    mistaken for real spend data here -- an `estimate_run()` matching only
    fake rows would report a confident-looking $0.00/0min "history_mode"
    estimate instead of falling through to `prior`, which is a worse
    failure than having no history at all (a caller can't distinguish "this
    really is free" from "the history is fake-backend noise")."""
    usage = row.get("token_usage") or {}
    if (usage.get("total_tokens") or 0) > 0:
        return True
    by_role = row.get("token_usage_by_role") or {}
    return any((u or {}).get("total_tokens") for u in by_role.values())


def load_history(
    glob: str = config.RUN_LOG_GLOB, out_dir: Path = config.OUT_DIR
) -> list[dict[str, Any]]:
    """Every `ok=True` row with real (non-zero) token usage across every
    out/generation_runs*.jsonl file, most recent last. A thin wrapper around
    review.load_jsonl() (not a fresh reimplementation) filtered to
    successful, real-usage runs -- a failed run's token/elapsed numbers
    don't represent what a normal run costs, and an all-zero-usage row is an
    `LLM_BACKEND=fake` test artifact (see _has_real_usage()), not a $0 real
    run. Returns [] if out/ or no matching files exist yet -- same "no data
    yet, not an error" convention review.py's own loaders use."""
    from .review import load_jsonl  # local import: avoids a hard dependency

    rows: list[dict[str, Any]] = []
    if not out_dir.exists():
        return rows
    for path in sorted(out_dir.glob(glob)):
        try:
            rows.extend(
                r for r in load_jsonl(path) if r.get("ok") and _has_real_usage(r)
            )
        except (OSError, ValueError):
            continue
    return rows


def _role_tokens(row: dict[str, Any], role: str) -> tuple[float, float, float] | None:
    """(prompt, completion, cached) for one role in one history row, or None
    when there's no *real* signal to average in. Both "key missing" and
    "prompt==0 and completion==0" are treated as absent -- a genuine LLM
    call never produces zero prompt tokens, so an all-zero entry is a row
    where usage simply wasn't recorded (observed in practice: a
    LLM_BACKEND=fake smoke-test run with a real elapsed_s but
    token_usage/token_usage_by_role left all-zero) rather than a real
    zero-cost call. Averaging it in anyway would silently drag a stage's
    estimated cost/time toward 0 -- exactly the "fabricated free" outcome
    this module's docstring says never to produce."""
    by_role = row.get("token_usage_by_role") or {}
    usage = by_role.get(role)
    if not usage:
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    if prompt == 0 and completion == 0:
        return None
    cached = usage.get("cached_prompt_tokens") or 0
    return float(prompt), float(completion), float(cached)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stage_prior_from_history(
    history: list[dict[str, Any]], role: str, *, per_scene: bool = False
) -> _StagePrior | None:
    """Average (prompt, completion, cached) tokens and seconds for one role
    across every matching history row. `per_scene=True` divides the role's
    totals by that row's `scenes_generated` (scene_writer's usage is summed
    across every scene in the run, not per-scene) -- rows with
    scenes_generated == 0 are skipped for that role since the average would
    be undefined."""
    prompts: list[float] = []
    completions: list[float] = []
    cacheds: list[float] = []
    seconds: list[float] = []
    for row in history:
        tokens = _role_tokens(row, role)
        if tokens is None:
            continue
        prompt, completion, cached = tokens
        divisor = 1.0
        if per_scene:
            n_scenes = row.get("scenes_generated") or 0
            if n_scenes <= 0:
                continue
            divisor = float(n_scenes)
        prompts.append(prompt / divisor)
        completions.append(completion / divisor)
        cacheds.append(cached / divisor)
        elapsed = row.get("elapsed_s")
        if elapsed:
            # Whole-run elapsed_s isn't per-role -- approximate this role's
            # share by its own token share of the run's total, a rough but
            # non-fabricated signal (never used when scene_metrics-derived
            # per-scene timing is available -- see estimate_remaining()).
            total_tokens = (row.get("token_usage") or {}).get("total_tokens") or 0
            role_tokens = prompt + completion
            if total_tokens > 0:
                seconds.append(elapsed * role_tokens / total_tokens / divisor)
    if not prompts:
        return None
    return _StagePrior(
        prompt=_mean(prompts),
        completion=_mean(completions),
        cached=_mean(cacheds),
        seconds=_mean(seconds) if seconds else 0.0,
        basis="history_mode",
    )


def _stage_prior(
    history: list[dict[str, Any]],
    role: str,
    *,
    mode: str,
    script_length: str,
    per_scene: bool = False,
) -> _StagePrior:
    exact = [
        r
        for r in history
        if r.get("mode") == mode and (r.get("script_length") or "") == script_length
    ]
    prior = _stage_prior_from_history(exact, role, per_scene=per_scene)
    if prior is not None:
        return _StagePrior(
            prior.prompt, prior.completion, prior.cached, prior.seconds, "history_mode_length"
        )

    same_mode = [r for r in history if r.get("mode") == mode]
    prior = _stage_prior_from_history(same_mode, role, per_scene=per_scene)
    if prior is not None:
        scale = length.parse_length_spec(script_length).events_scale
        # Only the token counts scale with length -- a per-scene prior is
        # already length-independent (one scene's tokens don't grow just
        # because the script has more scenes), so only the non-per-scene
        # (extract/beat_expand/legacy-role) priors are scaled here.
        if per_scene:
            return prior
        return _StagePrior(
            prior.prompt * scale, prior.completion * scale, prior.cached * scale,
            prior.seconds * scale, "history_mode",
        )

    default = _PRIOR_STAGE_TOKENS.get(role)
    if default is None:
        return _StagePrior(0.0, 0.0, 0.0, 0.0, "prior")
    return _StagePrior(
        default["prompt"], default["completion"], default["cached"], default["seconds"], "prior"
    )


def _price_stage(
    prior: _StagePrior, model_id: str, prices: dict[str, pricing.ModelPrice]
) -> float | None:
    price = prices.get(model_id) if model_id else None
    if price is None:
        return None
    return price.cost(prior.prompt, prior.completion, int(prior.cached))


def _combine_basis(bases: list[str]) -> str:
    """Worst-case (least trustworthy) basis among several stage bases, in
    the priority order this module documents at the top of the file."""
    order = ["measured_run", "history_mode_length", "history_mode", "prior", "unknown_price"]
    present = [b for b in bases if b in order]
    if not present:
        return "unknown_price"
    return max(present, key=order.index)


def _scene_count_from_target(script_length: str) -> int:
    events = length.parse_length_spec(script_length).events
    head = events.split("-", 1)[-1].strip()
    try:
        n = int(head)
    except ValueError:
        n = int(length.PRESETS["short"]["events"])
    return max(1, round(n * _SCENES_PER_EVENTS_TARGET))


def estimate_run(
    *,
    pipeline_mode: str,
    script_length: str,
    models: dict[str, str],
    scene_concurrency: int = 1,
    history: list[dict[str, Any]] | None = None,
    prices: dict[str, pricing.ModelPrice] | None = None,
    batch_widths: list[int] | None = None,
    scenes: int | None = None,
) -> RunEstimate:
    """Estimate a whole run's scenes/cost/time before (or independent of)
    any real progress. `batch_widths`, if given, is the caller's own
    plan_batches()-derived list of per-batch beat counts -- this module
    never re-derives causal-dependency scheduling itself (that's
    crew/orchestrator.py::plan_batches()'s job); omit it to estimate as if
    every beat were its own batch (parallelism=1.0, which real runs have
    shown is common -- see this module's docstring)."""
    prices = prices if prices is not None else pricing.load_prices()
    history = history if history is not None else load_history()

    if pipeline_mode == "layered":
        if scenes is not None:
            n_scenes = scenes
        elif batch_widths is not None:
            n_scenes = sum(batch_widths)
        else:
            n_scenes = _scene_count_from_target(script_length)
        widths = batch_widths if batch_widths is not None else [1] * n_scenes
        n_batches = len(widths)
        parallelism = (n_scenes / n_batches) if n_batches else None

        extract_prior = _stage_prior(
            history, "extractor", mode="layered", script_length=script_length
        )
        beats_prior = _stage_prior(
            history, "beat_expander", mode="layered", script_length=script_length
        )
        scene_prior = _stage_prior(
            history, "scene_writer", mode="layered", script_length=script_length, per_scene=True
        )

        extract_cost = _price_stage(extract_prior, models.get("extractor", ""), prices)
        beats_cost = _price_stage(beats_prior, models.get("beat_expander", ""), prices)
        scene_cost = _price_stage(scene_prior, models.get("scene_writer", ""), prices)

        costs = [extract_cost, beats_cost]
        notes: list[str] = []
        total_cost: float | None
        if scene_cost is None:
            total_cost = None
            notes.append(f"scene_writer 模型 {models.get('scene_writer', '(未設定)')!r} 無定價資料")
        elif any(c is None for c in costs):
            total_cost = None
            missing = [
                r for r, c in zip(("extractor", "beat_expander"), costs) if c is None
            ]
            notes.append(f"{'/'.join(missing)} 模型無定價資料")
        else:
            total_cost = sum(c for c in costs if c is not None) + scene_cost * n_scenes

        seconds = extract_prior.seconds + beats_prior.seconds + sum(
            math.ceil(w / max(1, scene_concurrency)) * scene_prior.seconds for w in widths
        )
        basis = _combine_basis([extract_prior.basis, beats_prior.basis, scene_prior.basis])
        if total_cost is None:
            basis = "unknown_price"

        if parallelism is not None and abs(parallelism - 1.0) < 1e-9 and scene_concurrency > 1:
            notes.append(
                f"beat 依賴鏈近乎線性（並行度 {parallelism:.1f}x），"
                f"SCENE_CONCURRENCY={scene_concurrency} 對這個結構可能沒有實際加速效果"
            )

        return RunEstimate(
            scenes=n_scenes,
            batches=n_batches,
            parallelism=parallelism,
            cost_usd=total_cost,
            cost_low=total_cost * 0.7 if total_cost is not None else None,
            cost_high=total_cost * 1.5 if total_cost is not None else None,
            seconds=seconds,
            seconds_low=seconds * 0.7,
            seconds_high=seconds * 1.5,
            basis=basis,
            notes=tuple(notes),
        )

    # legacy: one writer + one dialogue + one proof call, no scene concept.
    priors = {
        role: _stage_prior(history, role, mode="legacy", script_length=script_length)
        for role in _LEGACY_ROLES
    }
    role_costs = {
        role: _price_stage(priors[role], models.get(role, ""), prices) for role in _LEGACY_ROLES
    }
    missing = [role for role, c in role_costs.items() if c is None]
    total_cost = None if missing else sum(role_costs.values())
    seconds = sum(priors[role].seconds for role in _LEGACY_ROLES)
    basis = _combine_basis([priors[role].basis for role in _LEGACY_ROLES])
    notes = (f"{'/'.join(missing)} 模型無定價資料",) if missing else ()
    if missing:
        basis = "unknown_price"

    return RunEstimate(
        scenes=None,
        batches=None,
        parallelism=None,
        cost_usd=total_cost,
        cost_low=total_cost * 0.7 if total_cost is not None else None,
        cost_high=total_cost * 1.5 if total_cost is not None else None,
        seconds=seconds,
        seconds_low=seconds * 0.7,
        seconds_high=seconds * 1.5,
        basis=basis,
        notes=notes,
    )


def estimate_remaining(
    *,
    total_scenes: int,
    completed_metrics: list[dict[str, Any]],
    remaining_batch_widths: list[int],
    scene_concurrency: int,
    scene_model: str,
    prices: dict[str, pricing.ModelPrice] | None = None,
    fallback: RunEstimate | None = None,
) -> RunEstimate:
    """Estimate the cost/time still needed to finish a layered run already
    in progress, using the *actual* measured per-scene cost/time of its own
    completed_metrics (crew/scene_metrics.py::SceneMetric-shaped dicts, e.g.
    from crew/orchestrator.py::load_scene_metrics()) rather than any
    historical prior -- this is the "measured_run" basis, the most
    trustworthy tier. Falls back to `fallback` (normally an estimate_run()
    call) when no scenes have completed yet."""
    prices = prices if prices is not None else pricing.load_prices()
    n_remaining = max(0, total_scenes - len(completed_metrics))

    if not completed_metrics or n_remaining == 0:
        if n_remaining == 0:
            return RunEstimate(
                scenes=0, batches=0, parallelism=None,
                cost_usd=0.0, cost_low=0.0, cost_high=0.0,
                seconds=0.0, seconds_low=0.0, seconds_high=0.0,
                basis="measured_run", notes=("已無剩餘場次",),
            )
        return fallback if fallback is not None else RunEstimate()

    prior = _completed_scene_stage_prior(completed_metrics)
    scene_cost = _price_stage(prior, scene_model, prices)

    widths = remaining_batch_widths or [1] * n_remaining
    n_batches = len(widths)
    parallelism = (n_remaining / n_batches) if n_batches else None
    seconds = sum(math.ceil(w / max(1, scene_concurrency)) * prior.seconds for w in widths)

    total_cost = scene_cost * n_remaining if scene_cost is not None else None
    basis = "measured_run" if scene_cost is not None else "unknown_price"
    notes = () if scene_cost is not None else (f"模型 {scene_model!r} 無定價資料",)

    return RunEstimate(
        scenes=n_remaining,
        batches=n_batches,
        parallelism=parallelism,
        cost_usd=total_cost,
        cost_low=total_cost * 0.85 if total_cost is not None else None,
        cost_high=total_cost * 1.25 if total_cost is not None else None,
        seconds=seconds,
        seconds_low=seconds * 0.85,
        seconds_high=seconds * 1.25,
        basis=basis,
        notes=notes,
    )


def _completed_scene_stage_prior(completed_metrics: list[dict[str, Any]]) -> _StagePrior:
    """A `measured_run`-basis prior built from real, already-completed
    scenes of *this* run. SceneMetric has no prompt/completion split (only
    `total_tokens`), so cost is priced by treating the whole total as
    completion-rate token count divided evenly -- a deliberate
    simplification documented here rather than hidden: this slightly
    *overestimates* cost for a model whose completion rate is pricier than
    its prompt rate (true for every model in eval/model_prices.json today),
    which is the safer direction to be wrong in for a spend estimate."""
    totals = [m.get("total_tokens", 0) or 0 for m in completed_metrics]
    seconds = [m.get("elapsed_s", 0) or 0 for m in completed_metrics]
    avg_total = _mean(totals)
    avg_seconds = _mean(seconds)
    # Split evenly between "prompt" and "completion" buckets for pricing.py's
    # cost() signature -- see docstring above for why this is a deliberate,
    # conservative approximation rather than a real split.
    return _StagePrior(
        prompt=avg_total / 2, completion=avg_total / 2, cached=0.0,
        seconds=avg_seconds, basis="measured_run",
    )


__all__ = [
    "RunEstimate",
    "load_history",
    "estimate_run",
    "estimate_remaining",
]
