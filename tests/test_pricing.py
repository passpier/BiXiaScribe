"""Unit tests for bixiascribe.pricing -- entirely offline, no network, no
LLM_BACKEND dependency (pricing.py imports neither crewai nor litellm)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import pricing  # noqa: E402

REAL_PRICES_FILE = Path(__file__).resolve().parents[1] / "eval" / "model_prices.json"


def _prices_file(entries: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- kept open for the test's lifetime
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(entries, tmp, ensure_ascii=False)
    tmp.close()
    return Path(tmp.name)


def test_load_prices_parses_the_real_file():
    prices = pricing.load_prices(REAL_PRICES_FILE)
    assert "openrouter/deepseek/deepseek-chat" in prices
    p = prices["openrouter/deepseek/deepseek-chat"]
    assert p.prompt_usd_per_1m > 0
    assert p.completion_usd_per_1m > 0


def test_load_prices_skips_underscore_keys_and_missing_file():
    path = _prices_file(
        {
            "_comment": "not a model",
            "openrouter/x": {
                "_note": "ignored per-entry field",
                "prompt_usd_per_1m": 1.0,
                "completion_usd_per_1m": 2.0,
            },
        }
    )
    prices = pricing.load_prices(path)
    assert set(prices) == {"openrouter/x"}

    assert pricing.load_prices(Path("/nonexistent/model_prices.json")) == {}


def test_load_prices_skips_malformed_entries_without_raising():
    path = _prices_file(
        {
            "openrouter/good": {"prompt_usd_per_1m": 1.0, "completion_usd_per_1m": 2.0},
            "openrouter/bad": {"prompt_usd_per_1m": "not-a-number"},
            "openrouter/also_bad": "not-a-dict",
        }
    )
    prices = pricing.load_prices(path)
    assert set(prices) == {"openrouter/good"}


def test_model_price_cost_is_linear_in_tokens():
    price = pricing.ModelPrice(
        model_id="m", prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0
    )
    assert price.cost(1_000_000, 0) == 1.0
    assert price.cost(0, 1_000_000) == 2.0
    assert price.cost(500_000, 500_000) == 1.5


def test_model_price_cost_bills_cached_tokens_at_cache_rate_when_known():
    price = pricing.ModelPrice(
        model_id="m",
        prompt_usd_per_1m=1.0,
        completion_usd_per_1m=2.0,
        cache_read_usd_per_1m=0.1,
    )
    # 1M prompt tokens, 400k of them cached: 600k fresh @ $1 + 400k cached @ $0.1
    assert price.cost(1_000_000, 0, cached_tokens=400_000) == 0.6 + 0.04
    assert price.cache_priced(400_000) is True
    assert price.cache_priced(0) is True  # nothing cached -> nothing ambiguous


def test_model_price_cost_falls_back_to_full_rate_when_cache_rate_unknown():
    price = pricing.ModelPrice(model_id="m", prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)
    # No cache_read_usd_per_1m -- cached tokens still billed (at the full
    # prompt rate), never silently free, but cache_priced() must say so.
    assert price.cost(1_000_000, 0, cached_tokens=400_000) == 1.0
    assert price.cache_priced(400_000) is False


def test_model_price_cost_clamps_cached_tokens_to_prompt_tokens():
    price = pricing.ModelPrice(
        model_id="m", prompt_usd_per_1m=1.0, completion_usd_per_1m=0.0, cache_read_usd_per_1m=0.1
    )
    # cached_tokens > prompt_tokens shouldn't go negative on fresh_tokens.
    assert price.cost(1_000_000, 0, cached_tokens=5_000_000) == 0.1


def test_load_prices_parses_cache_rates_when_present():
    path = _prices_file(
        {
            "openrouter/x": {
                "prompt_usd_per_1m": 1.0,
                "completion_usd_per_1m": 2.0,
                "cache_read_usd_per_1m": 0.1,
                "cache_write_usd_per_1m": 1.25,
            },
            "openrouter/y": {"prompt_usd_per_1m": 1.0, "completion_usd_per_1m": 2.0},
        }
    )
    prices = pricing.load_prices(path)
    assert prices["openrouter/x"].cache_read_usd_per_1m == 0.1
    assert prices["openrouter/x"].cache_write_usd_per_1m == 1.25
    # An entry with no cache fields at all must stay None, not 0.0 -- 0.0
    # would read as "cache reads are free" instead of "unknown".
    assert prices["openrouter/y"].cache_read_usd_per_1m is None
    assert prices["openrouter/y"].cache_write_usd_per_1m is None


def test_estimate_cost_by_role_applies_cache_discount_when_price_known():
    prices = {
        "openrouter/m": pricing.ModelPrice(
            "openrouter/m", 1.0, 2.0, cache_read_usd_per_1m=0.1
        ),
    }
    cost, basis = pricing.estimate_cost(
        None,
        {"writer": "openrouter/m"},
        token_usage_by_role={
            "writer": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 0,
                "cached_prompt_tokens": 400_000,
            },
        },
        prices=prices,
    )
    assert basis == "by_role"
    assert cost == 0.6 + 0.04


def test_estimate_cost_flags_basis_when_cache_rate_unavailable():
    prices = {"openrouter/m": pricing.ModelPrice("openrouter/m", 1.0, 2.0)}
    cost, basis = pricing.estimate_cost(
        None,
        {"writer": "openrouter/m"},
        token_usage_by_role={
            "writer": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 0,
                "cached_prompt_tokens": 400_000,
            },
        },
        prices=prices,
    )
    assert basis == "by_role_uncached_price_missing"
    assert cost == 1.0  # still billed in full, just flagged as an overestimate

    cost2, basis2 = pricing.estimate_cost(
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "cached_prompt_tokens": 400_000,
        },
        {"writer": "openrouter/m", "dialogue": "openrouter/m"},
        prices=prices,
    )
    assert basis2 == "uniform_uncached_price_missing"
    assert cost2 == 1.0


def test_estimate_cost_uniform_when_all_roles_share_one_model():
    prices = {"openrouter/m": pricing.ModelPrice("openrouter/m", 1.0, 2.0)}
    cost, basis = pricing.estimate_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        {"writer": "openrouter/m", "dialogue": "openrouter/m", "proof": "openrouter/m"},
        prices=prices,
    )
    assert basis == "uniform"
    assert cost == 3.0


def test_estimate_cost_by_role_is_exact_for_mixed_models():
    prices = {
        "openrouter/cheap": pricing.ModelPrice("openrouter/cheap", 1.0, 1.0),
        "openrouter/pricey": pricing.ModelPrice("openrouter/pricey", 10.0, 10.0),
    }
    cost, basis = pricing.estimate_cost(
        {"prompt_tokens": 100, "completion_tokens": 0},  # ignored when by_role succeeds
        {
            "writer": "openrouter/cheap",
            "dialogue": "openrouter/pricey",
            "proof": "openrouter/cheap",
        },
        token_usage_by_role={
            "extractor": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
            "scene_writer": {"prompt_tokens": 0, "completion_tokens": 1_000_000},
        },
        prices=prices,
    )
    # role names in token_usage_by_role must be looked up in `models`, so map
    # extractor/scene_writer explicitly for this exact-cost path.
    cost2, basis2 = pricing.estimate_cost(
        None,
        {"extractor": "openrouter/cheap", "scene_writer": "openrouter/pricey"},
        token_usage_by_role={
            "extractor": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
            "scene_writer": {"prompt_tokens": 0, "completion_tokens": 1_000_000},
        },
        prices=prices,
    )
    assert basis2 == "by_role"
    assert cost2 == 1.0 + 10.0
    # sanity: the first call's mismatched role names mean no role in
    # token_usage_by_role resolves to a priced model (models has
    # writer/dialogue/proof, token_usage_by_role has extractor/scene_writer),
    # so it falls through to the run-wide token_usage path instead -- and
    # since writer/dialogue map to two distinct models, that's a lower bound.
    assert basis == "uniform_lower_bound"
    assert cost == 0.0001  # 100 prompt tokens priced against the cheaper model


def test_estimate_cost_mixed_models_no_role_usage_falls_back_to_lower_bound():
    prices = {
        "openrouter/cheap": pricing.ModelPrice("openrouter/cheap", 1.0, 1.0),
        "openrouter/pricey": pricing.ModelPrice("openrouter/pricey", 10.0, 10.0),
    }
    cost, basis = pricing.estimate_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        {"writer": "openrouter/cheap", "dialogue": "openrouter/pricey"},
        prices=prices,
    )
    assert basis == "uniform_lower_bound"
    assert cost == 1.0  # priced against the cheaper of the two models


def test_estimate_cost_unknown_model_returns_none_not_zero():
    cost, basis = pricing.estimate_cost(
        {"prompt_tokens": 1000, "completion_tokens": 1000},
        {"writer": "openrouter/does-not-exist"},
        prices={},
    )
    assert cost is None
    assert basis == "unknown_price"


def test_estimate_cost_empty_usage_returns_none():
    prices = {"openrouter/m": pricing.ModelPrice("openrouter/m", 1.0, 2.0)}
    cost, basis = pricing.estimate_cost(None, {"writer": "openrouter/m"}, prices=prices)
    assert cost is None
    assert basis == "unknown_price"

    cost2, basis2 = pricing.estimate_cost({}, {"writer": "openrouter/m"}, prices=prices)
    assert cost2 is None
    assert basis2 == "unknown_price"


def test_estimate_cost_no_models_at_all_returns_unknown():
    cost, basis = pricing.estimate_cost(
        {"prompt_tokens": 100, "completion_tokens": 100}, {}, prices={}
    )
    assert cost is None
    assert basis == "unknown_price"


def test_quality_unit_costs_normal_case():
    metrics = {"events": 4, "dialogue_lines": 8, "avg_line_chars": 25.0}
    result = pricing.quality_unit_costs(1.0, metrics)
    assert result["usd_per_event"] == 0.25
    assert result["usd_per_dialogue_line"] == 0.125
    # 8 lines * 25 chars = 200 chars = 0.2k chars -> 1.0 / 0.2 == 5.0
    assert result["usd_per_1k_dialogue_chars"] == 5.0


def test_quality_unit_costs_zero_denominators_return_none_not_crash():
    metrics = {"events": 0, "dialogue_lines": 0, "avg_line_chars": 0.0}
    result = pricing.quality_unit_costs(1.0, metrics)
    assert result == {
        "usd_per_event": None,
        "usd_per_dialogue_line": None,
        "usd_per_1k_dialogue_chars": None,
    }


def test_quality_unit_costs_none_cost_returns_all_none():
    metrics = {"events": 4, "dialogue_lines": 8, "avg_line_chars": 25.0}
    result = pricing.quality_unit_costs(None, metrics)
    assert all(v is None for v in result.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
