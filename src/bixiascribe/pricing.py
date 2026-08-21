"""Cost accounting for Stage 2/2b generation runs -- the missing piece
CLAUDE.md's "Comparing per-agent model splits" section flagged: this repo
records `token_usage` in every RunReport/JSONL row but never converts it to
money, so a question like "is the pricier dialogue model actually worth it"
had no answer besides eyeballing raw token counts. This module is that
conversion, kept deliberately dependency-free (no crewai/litellm import) so
it's importable from review.py (streamlit-free, per that module's own
constraint) as well as generation.py and the eval harness.

Two things this module is careful about, both because the data it's fed is
real and imperfect:

1. A price lookup can miss (an unpriced/typo'd model id, or a variant that
   mixes models CLAUDE.md's own eval_generation.py doesn't attribute to
   specific roles for legacy runs). Every function here returns `None` for
   the cost rather than guessing, and reports *why* via a `basis` string
   ("by_role" | "uniform" | "unknown_price") -- see estimate_cost()'s
   docstring. A caller that silently treated a missing price as $0 would
   make "this model is free" a plausible-looking bug.
2. Quality-per-dollar, not just total dollars, is the point (see this
   project's docs/BENCHMARKS.md for why raw token counts alone don't answer
   "was the pricier model worth it"). quality_unit_costs() divides a cost
   by crew/metrics.py::script_metrics() counts the caller already has --
   guarding every division against a zero denominator (a failed run has
   zero events) instead of raising ZeroDivisionError.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

DEFAULT_PRICES_FILE = config.PROJECT_ROOT / "eval" / "model_prices.json"

# Same usage field names crew/orchestrator.py's _UsageAccumulator tracks --
# kept in sync manually (no shared import) since orchestrator.py deliberately
# has no dependency on this module.
_PROMPT_FIELD = "prompt_tokens"
_COMPLETION_FIELD = "completion_tokens"
_CACHED_PROMPT_FIELD = "cached_prompt_tokens"
_TOTAL_FIELD = "total_tokens"


@dataclass(frozen=True)
class ModelPrice:
    """One eval/model_prices.json entry. `provider` is the OpenRouter
    provider slug this snapshot was pinned to ("" = OpenRouter's own default
    routing, unpinned -- what every historical JSONL row actually ran
    under).

    `cache_read_usd_per_1m` / `cache_write_usd_per_1m` are OpenRouter's
    prompt-caching rates for this endpoint (cache_write is recorded for
    completeness/future use -- cost() below only ever bills a cache *read*,
    since cache_creation_tokens isn't priced by any caller yet). Both are
    None for a snapshot taken before scripts/refresh_prices.py learned to
    fetch them, or for a provider/model that doesn't support prompt caching
    at all -- cost() treats None the same as "no cache discount available"
    and bills those tokens at the full prompt rate rather than guessing."""

    model_id: str
    prompt_usd_per_1m: float
    completion_usd_per_1m: float
    provider: str = ""
    context_length: int | None = None
    max_completion_tokens: int | None = None
    supports_tools: bool = True
    fetched_at: str = ""
    cache_read_usd_per_1m: float | None = None
    cache_write_usd_per_1m: float | None = None

    def cost(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> float:
        """`cached_tokens` (from token_usage's cached_prompt_tokens) is a
        subset of `prompt_tokens`, not additional to it -- those tokens are
        billed at cache_read_usd_per_1m instead of the full prompt rate when
        that rate is known. `cached_tokens=0` (the default) makes this
        byte-for-byte the pre-cache-accounting formula, so every existing
        caller that doesn't pass it is unaffected."""
        cached_tokens = max(0, min(cached_tokens, prompt_tokens))
        fresh_tokens = prompt_tokens - cached_tokens
        cache_rate = (
            self.cache_read_usd_per_1m
            if self.cache_read_usd_per_1m is not None
            else self.prompt_usd_per_1m
        )
        return (
            fresh_tokens * self.prompt_usd_per_1m
            + cached_tokens * cache_rate
            + completion_tokens * self.completion_usd_per_1m
        ) / 1_000_000

    def cache_priced(self, cached_tokens: int) -> bool:
        """False when `cached_tokens` were actually billed at the full
        prompt rate above because this entry has no known cache_read rate --
        i.e. cost() is a safe (never-underestimates) placeholder for those
        tokens, not a real cache-discounted price. True whenever there's
        nothing ambiguous to flag (no cached tokens at all, or a real rate
        was applied)."""
        return cached_tokens <= 0 or self.cache_read_usd_per_1m is not None


def load_prices(path: Path = DEFAULT_PRICES_FILE) -> dict[str, ModelPrice]:
    """Parse eval/model_prices.json into {model_id: ModelPrice}. Returns {}
    if the file is missing (mirrors review.py's loaders' "no file yet" ==
    empty convention, not an error) -- every caller below already treats a
    missing price as `basis="unknown_price"`, not a crash. Top-level and
    per-entry keys starting with "_" (e.g. "_comment", "_note") are metadata
    for humans reading the file, not model ids/fields, and are skipped."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    prices: dict[str, ModelPrice] = {}
    for model_id, entry in raw.items():
        if model_id.startswith("_") or not isinstance(entry, dict):
            continue
        try:
            prices[model_id] = ModelPrice(
                model_id=model_id,
                prompt_usd_per_1m=float(entry["prompt_usd_per_1m"]),
                completion_usd_per_1m=float(entry["completion_usd_per_1m"]),
                provider=entry.get("provider") or "",
                context_length=entry.get("context_length"),
                max_completion_tokens=entry.get("max_completion_tokens"),
                supports_tools=bool(entry.get("supports_tools", True)),
                fetched_at=entry.get("fetched_at") or "",
                cache_read_usd_per_1m=(
                    float(entry["cache_read_usd_per_1m"])
                    if entry.get("cache_read_usd_per_1m") is not None
                    else None
                ),
                cache_write_usd_per_1m=(
                    float(entry["cache_write_usd_per_1m"])
                    if entry.get("cache_write_usd_per_1m") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int] | None:
    """(prompt_tokens, completion_tokens, cached_prompt_tokens) from a usage
    dict, or None if prompt/completion is missing/unusable -- a failed run's
    `token_usage` is `{}` (legacy) or all-zero (layered), and both should
    fall through to "nothing to price" rather than a spurious $0.00.
    `cached_prompt_tokens` defaults to 0 (not required) since it postdates
    every historical JSONL row -- absent means "no cache accounting for this
    row", not "zero cache tokens were spent"."""
    if not usage:
        return None
    prompt = usage.get(_PROMPT_FIELD)
    completion = usage.get(_COMPLETION_FIELD)
    if prompt is None or completion is None:
        return None
    cached = int(usage.get(_CACHED_PROMPT_FIELD) or 0)
    return int(prompt), int(completion), cached


def _price_run(price: ModelPrice, tokens: tuple[int, int, int]) -> tuple[float, bool]:
    """(usd, cache_priced) for one (prompt, completion, cached) tuple against
    one ModelPrice -- see ModelPrice.cost()/.cache_priced()'s docstrings.
    `usd` is always a real number (never guessed-free); `cache_priced` is
    False when cached tokens had to be billed at the full prompt rate for
    lack of a known cache_read rate, so estimate_cost() can flag that in
    `basis` instead of a caller silently believing the cache discount was
    applied."""
    return price.cost(*tokens), price.cache_priced(tokens[2])


def estimate_cost(
    token_usage: dict[str, Any] | None,
    models: dict[str, str],
    *,
    token_usage_by_role: dict[str, dict[str, Any]] | None = None,
    prices: dict[str, ModelPrice] | None = None,
) -> tuple[float | None, str]:
    """Best-effort cost of one run, in USD.

    `models` maps role name -> OpenRouter model id (e.g.
    {"writer": ..., "dialogue": ..., "proof": ...} for a legacy run, or the
    six-role layered set) -- the same shape RunReport.to_dict()'s
    model_writer/model_dialogue/... fields already carry, just collected by
    the caller into a dict.

    Returns (cost_usd, basis):
    - `token_usage_by_role` given and covers every role with a nonzero
      count: ("by_role") -- exact, since each role's actual spend is priced
      against its own model rather than assumed to share the total.
    - Otherwise, if every role in `models` resolves to the *same* model id:
      ("uniform") -- also exact, since there's only one price to apply to
      the run-wide `token_usage` total (this is every current variant in
      eval/model_variants.json except long-prose, its one mixed-model entry).
    - Otherwise (mixed models, no per-role usage): the run-wide total is
      still priced against whichever role's model has the *lowest* price
      per token, i.e. a lower-bound estimate -- ("uniform_lower_bound").
      Call twice (min/max role) for a range, as CLAUDE.md's discussion of
      this exact ambiguity describes.
    - No usable token_usage, or none of the models referenced have a price
      entry: (None, "unknown_price").

    Any of the three priced bases above gets a "_uncached_price_missing"
    suffix (e.g. "by_role_uncached_price_missing") when the run actually
    spent cached_prompt_tokens but the priced model's ModelPrice entry has
    no cache_read_usd_per_1m -- those tokens were still billed (at the full
    prompt rate, see ModelPrice.cost()), never silently treated as free, but
    the returned cost_usd is an overestimate for that portion until
    scripts/refresh_prices.py learns this model's cache rate.
    """
    prices = prices if prices is not None else load_prices()

    if token_usage_by_role:
        total = 0.0
        any_priced = False
        cache_unpriced = False
        for role, usage in token_usage_by_role.items():
            tokens = _usage_tokens(usage)
            if tokens is None:
                continue
            model_id = models.get(role)
            price = prices.get(model_id) if model_id else None
            if price is None:
                continue
            any_priced = True
            usd, cache_priced = _price_run(price, tokens)
            total += usd
            if not cache_priced:
                cache_unpriced = True
        if any_priced:
            basis = "by_role" + ("_uncached_price_missing" if cache_unpriced else "")
            return total, basis

    tokens = _usage_tokens(token_usage)
    if tokens is None:
        return None, "unknown_price"

    distinct_models = sorted(set(models.values()) - {""})
    if not distinct_models:
        return None, "unknown_price"

    if len(distinct_models) == 1:
        price = prices.get(distinct_models[0])
        if price is None:
            return None, "unknown_price"
        usd, cache_priced = _price_run(price, tokens)
        basis = "uniform" + ("_uncached_price_missing" if not cache_priced else "")
        return usd, basis

    priced = [prices[m] for m in distinct_models if m in prices]
    if not priced:
        return None, "unknown_price"
    cheapest = min(priced, key=lambda p: p.prompt_usd_per_1m + p.completion_usd_per_1m)
    usd, cache_priced = _price_run(cheapest, tokens)
    basis = "uniform_lower_bound" + ("_uncached_price_missing" if not cache_priced else "")
    return usd, basis


def quality_unit_costs(
    cost_usd: float | None, metrics: dict[str, Any]
) -> dict[str, float | None]:
    """usd_per_event / usd_per_dialogue_line / usd_per_1k_dialogue_chars from
    a run's cost and its crew/metrics.py::script_metrics() dict -- the
    "is the pricier model actually worth it" numbers, not just a total.
    Every value is None (not a ZeroDivisionError, not 0.0 -- 0.0 would read
    as "free") when cost_usd is None or the relevant count is zero (a failed
    run, or one script_metrics() couldn't compute)."""
    events = metrics.get("events") or 0
    dialogue_lines = metrics.get("dialogue_lines") or 0
    avg_line_chars = metrics.get("avg_line_chars") or 0.0
    dialogue_chars = dialogue_lines * avg_line_chars

    if cost_usd is None:
        return {
            "usd_per_event": None,
            "usd_per_dialogue_line": None,
            "usd_per_1k_dialogue_chars": None,
        }
    return {
        "usd_per_event": cost_usd / events if events else None,
        "usd_per_dialogue_line": cost_usd / dialogue_lines if dialogue_lines else None,
        "usd_per_1k_dialogue_chars": (
            cost_usd / (dialogue_chars / 1000) if dialogue_chars else None
        ),
    }


__all__ = [
    "DEFAULT_PRICES_FILE",
    "ModelPrice",
    "load_prices",
    "estimate_cost",
    "quality_unit_costs",
]
