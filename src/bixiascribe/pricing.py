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
   project's README "關鍵數據" for why raw token counts alone don't answer
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
_TOTAL_FIELD = "total_tokens"


@dataclass(frozen=True)
class ModelPrice:
    """One eval/model_prices.json entry. `provider` is the OpenRouter
    provider slug this snapshot was pinned to ("" = OpenRouter's own default
    routing, unpinned -- what every historical JSONL row actually ran
    under)."""

    model_id: str
    prompt_usd_per_1m: float
    completion_usd_per_1m: float
    provider: str = ""
    context_length: int | None = None
    max_completion_tokens: int | None = None
    supports_tools: bool = True
    fetched_at: str = ""

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_usd_per_1m
            + completion_tokens * self.completion_usd_per_1m
        ) / 1_000_000


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
            )
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int] | None:
    """(prompt_tokens, completion_tokens) from a usage dict, or None if
    either is missing/unusable -- a failed run's `token_usage` is `{}`
    (legacy) or all-zero (layered), and both should fall through to
    "nothing to price" rather than a spurious $0.00."""
    if not usage:
        return None
    prompt = usage.get(_PROMPT_FIELD)
    completion = usage.get(_COMPLETION_FIELD)
    if prompt is None or completion is None:
        return None
    return int(prompt), int(completion)


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
      eval/model_variants.json except the retired mixed-model ones).
    - Otherwise (mixed models, no per-role usage): the run-wide total is
      still priced against whichever role's model has the *lowest* price
      per token, i.e. a lower-bound estimate -- ("uniform_lower_bound").
      Call twice (min/max role) for a range, as CLAUDE.md's discussion of
      this exact ambiguity describes.
    - No usable token_usage, or none of the models referenced have a price
      entry: (None, "unknown_price").
    """
    prices = prices if prices is not None else load_prices()

    if token_usage_by_role:
        total = 0.0
        any_priced = False
        for role, usage in token_usage_by_role.items():
            tokens = _usage_tokens(usage)
            if tokens is None:
                continue
            model_id = models.get(role)
            price = prices.get(model_id) if model_id else None
            if price is None:
                continue
            any_priced = True
            total += price.cost(*tokens)
        if any_priced:
            return total, "by_role"

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
        return price.cost(*tokens), "uniform"

    priced = [prices[m] for m in distinct_models if m in prices]
    if not priced:
        return None, "unknown_price"
    cheapest = min(priced, key=lambda p: p.prompt_usd_per_1m + p.completion_usd_per_1m)
    return cheapest.cost(*tokens), "uniform_lower_bound"


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
