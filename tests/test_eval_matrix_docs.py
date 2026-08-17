"""Lightweight, offline regression guard: every eval/model_variants.json
entry's six per-agent-role model ids must have a price entry in
eval/model_prices.json. Catches the silent failure mode where a new
variant is added but nobody runs scripts/refresh_prices.py for its models
-- src/bixiascribe/pricing.py then reports cost_basis="unknown_price"
instead of a real number, and that's easy to miss in a wall of aggregate
output. No API key, no network, no Chroma.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import generation, pricing  # noqa: E402


def test_every_variant_role_model_has_a_price_entry():
    prices = pricing.load_prices()
    variants = generation.load_variants()
    missing = []
    for v in variants:
        models = v.to_model_choice()
        for role in ("writer", "dialogue", "proof", "extractor", "beat_expander", "scene_writer"):
            model_id = getattr(models, role)
            if model_id and model_id not in prices:
                missing.append(f"{v.name}.{role}={model_id}")
    assert not missing, f"missing eval/model_prices.json entries: {missing}"
