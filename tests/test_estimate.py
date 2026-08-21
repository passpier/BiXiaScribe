"""Unit tests for bixiascribe.estimate -- entirely offline, no network, no
LLM_BACKEND dependency (estimate.py imports neither crewai nor litellm; its
review import is lazy and only touches the real out/ dir when a caller
doesn't pass `history` explicitly, which every test here does)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import estimate, pricing  # noqa: E402

_PRICES = {
    "openrouter/writer": pricing.ModelPrice("openrouter/writer", 1.0, 2.0),
    "openrouter/dialogue": pricing.ModelPrice("openrouter/dialogue", 1.0, 2.0),
    "openrouter/proof": pricing.ModelPrice("openrouter/proof", 1.0, 2.0),
    "openrouter/scene": pricing.ModelPrice("openrouter/scene", 1.0, 2.0),
    "openrouter/extract": pricing.ModelPrice("openrouter/extract", 1.0, 2.0),
    "openrouter/beat": pricing.ModelPrice("openrouter/beat", 1.0, 2.0),
}

_LEGACY_MODELS = {
    "writer": "openrouter/writer",
    "dialogue": "openrouter/dialogue",
    "proof": "openrouter/proof",
}
_LAYERED_MODELS = {
    "extractor": "openrouter/extract",
    "beat_expander": "openrouter/beat",
    "scene_writer": "openrouter/scene",
}


def _legacy_history_row(elapsed_s: float = 300.0) -> dict:
    return {
        "ok": True,
        "mode": "legacy",
        "script_length": "short",
        "elapsed_s": elapsed_s,
        "token_usage": {"total_tokens": 3000 + 1500},
        "token_usage_by_role": {
            "writer": {"prompt_tokens": 1000, "completion_tokens": 500},
            "dialogue": {"prompt_tokens": 2000, "completion_tokens": 800},
            "proof": {"prompt_tokens": 500, "completion_tokens": 200},
        },
    }


def _layered_history_row(scenes: int = 3, per_scene_prompt: float = 1000.0) -> dict:
    return {
        "ok": True,
        "mode": "layered",
        "script_length": "short",
        "elapsed_s": 900.0,
        "scenes_generated": scenes,
        "token_usage": {"total_tokens": 20000},
        "token_usage_by_role": {
            "extractor": {"prompt_tokens": 1000, "completion_tokens": 500},
            "beat_expander": {"prompt_tokens": 1000, "completion_tokens": 700},
            # scene_writer usage is a run-wide total across every scene, per
            # _stage_prior_from_history's per_scene=True division.
            "scene_writer": {
                "prompt_tokens": per_scene_prompt * scenes,
                "completion_tokens": 500 * scenes,
            },
        },
    }


def test_no_price_entry_returns_none_not_zero():
    result = estimate.estimate_run(
        pipeline_mode="legacy",
        script_length="short",
        models=_LEGACY_MODELS,
        history=[],
        prices={},
    )
    assert result.cost_usd is None
    assert result.basis == "unknown_price"
    assert result.notes


def test_prior_basis_when_no_history():
    result = estimate.estimate_run(
        pipeline_mode="legacy",
        script_length="short",
        models=_LEGACY_MODELS,
        history=[],
        prices=_PRICES,
    )
    assert result.basis == "prior"
    assert result.cost_usd is not None
    assert result.cost_usd > 0
    assert result.seconds is not None
    assert result.seconds > 0


def test_history_mode_length_exact_match_used_when_available():
    history = [_legacy_history_row(elapsed_s=123.0)]
    result = estimate.estimate_run(
        pipeline_mode="legacy",
        script_length="short",
        models=_LEGACY_MODELS,
        history=history,
        prices=_PRICES,
    )
    assert result.basis == "history_mode_length"
    # writer: 1000*1 + 500*2 = 2000; dialogue: 2000+1600=3600; proof: 500+400=900
    assert round(result.cost_usd, 6) == round((2000 + 3600 + 900) / 1_000_000, 6)


def test_history_mode_only_scales_by_events_when_length_differs():
    history = [_legacy_history_row()]
    result = estimate.estimate_run(
        pipeline_mode="legacy",
        script_length="medium",  # no "medium" rows in history -> falls back
        models=_LEGACY_MODELS,
        history=history,
        prices=_PRICES,
    )
    assert result.basis == "history_mode"


def test_layered_scenes_and_batches_from_batch_widths():
    history = [_layered_history_row(scenes=3)]
    result = estimate.estimate_run(
        pipeline_mode="layered",
        script_length="short",
        models=_LAYERED_MODELS,
        history=history,
        prices=_PRICES,
        scenes=6,
        batch_widths=[3, 3],
        scene_concurrency=3,
    )
    assert result.scenes == 6
    assert result.batches == 2
    assert result.parallelism == 3.0
    assert result.cost_usd is not None
    assert result.seconds is not None


def test_layered_scene_level_prior_is_not_double_scaled_by_events():
    """A single scene's own token/time cost must not additionally scale by
    events_scale on top of the scene *count* already scaling with it --
    otherwise a script_length=long matrix estimate compounds both factors
    (this was caught estimating ~85 real hours for a 6-variant matrix off a
    single ~220s real sample). Compare a "short" and "long" run with the
    same `scenes` count and no history (prior basis): per-scene component of
    cost/time must be identical, only the scene count differs across calls,
    so holding scenes fixed must hold total cost/time fixed too."""
    short_result = estimate.estimate_run(
        pipeline_mode="layered",
        script_length="short",
        models=_LAYERED_MODELS,
        history=[],
        prices=_PRICES,
        scenes=5,
    )
    long_result = estimate.estimate_run(
        pipeline_mode="layered",
        script_length="long",
        models=_LAYERED_MODELS,
        history=[],
        prices=_PRICES,
        scenes=5,
    )
    # extractor/beat_expander setup stages may scale with script_length, but
    # the scene_writer component (5 identical scenes either way) must not --
    # so the two totals should differ by at most the setup-stage delta, not
    # by anything proportional to events_scale on the scene component too.
    assert long_result.seconds is not None and short_result.seconds is not None
    scene_prior_seconds = estimate._PRIOR_STAGE_TOKENS["scene_writer"]["seconds"]
    # Both must contain exactly 5x the (unscaled) per-scene prior seconds.
    assert short_result.seconds >= 5 * scene_prior_seconds
    assert long_result.seconds >= 5 * scene_prior_seconds
    # The difference between long and short must come only from the setup
    # stages (extractor/beat_expander), not from the scene component being
    # additionally multiplied by events_scale (~7.5x for long vs short).
    diff = long_result.seconds - short_result.seconds
    assert diff < 5 * scene_prior_seconds  # nowhere near a 7.5x scene blowup


def test_layered_no_batch_widths_assumes_serial_and_warns():
    result = estimate.estimate_run(
        pipeline_mode="layered",
        script_length="short",
        models=_LAYERED_MODELS,
        history=[],
        prices=_PRICES,
        scenes=5,
        scene_concurrency=3,  # should have no effect without batch_widths
    )
    assert result.parallelism == 1.0
    assert result.batches == 5
    assert any("SCENE_CONCURRENCY" in n or "並行" in n for n in result.notes)


def test_estimate_remaining_falls_back_when_nothing_completed():
    fallback = estimate.RunEstimate(scenes=10, cost_usd=1.0, seconds=100.0, basis="prior")
    result = estimate.estimate_remaining(
        total_scenes=10,
        completed_metrics=[],
        remaining_batch_widths=[],
        scene_concurrency=1,
        scene_model="openrouter/scene",
        prices=_PRICES,
        fallback=fallback,
    )
    assert result is fallback


def test_estimate_remaining_projects_from_measured_scenes():
    completed = [
        {"beat_id": "b0", "elapsed_s": 200.0, "total_tokens": 20000},
        {"beat_id": "b1", "elapsed_s": 220.0, "total_tokens": 21000},
    ]
    result = estimate.estimate_remaining(
        total_scenes=10,
        completed_metrics=completed,
        remaining_batch_widths=[1] * 8,
        scene_concurrency=1,
        scene_model="openrouter/scene",
        prices=_PRICES,
    )
    assert result.basis == "measured_run"
    assert result.scenes == 8
    assert result.seconds is not None
    # 8 remaining scenes * mean(200,220)=210s
    assert round(result.seconds, 1) == round(8 * 210.0, 1)
    assert result.cost_usd is not None
    assert result.cost_usd > 0


def test_estimate_remaining_no_price_returns_none_cost_but_keeps_seconds():
    completed = [{"beat_id": "b0", "elapsed_s": 200.0, "total_tokens": 20000}]
    result = estimate.estimate_remaining(
        total_scenes=3,
        completed_metrics=completed,
        remaining_batch_widths=[1, 1],
        scene_concurrency=1,
        scene_model="openrouter/does-not-exist",
        prices=_PRICES,
    )
    assert result.cost_usd is None
    assert result.seconds is not None
    assert any("無定價" in n for n in result.notes)


def test_estimate_remaining_zero_remaining_is_a_real_zero_not_a_guess():
    completed = [{"beat_id": f"b{i}", "elapsed_s": 200.0, "total_tokens": 20000} for i in range(3)]
    result = estimate.estimate_remaining(
        total_scenes=3,
        completed_metrics=completed,
        remaining_batch_widths=[],
        scene_concurrency=1,
        scene_model="openrouter/scene",
        prices=_PRICES,
    )
    assert result.scenes == 0
    assert result.cost_usd == 0.0
    assert result.seconds == 0.0


def test_all_zero_token_history_row_is_excluded_not_averaged_in():
    """Reproduces a real contamination case: a LLM_BACKEND=fake UI
    smoke-test run landed in real out/generation_runs_ui.jsonl with a real
    elapsed_s (33.8s) but token_usage/token_usage_by_role left entirely
    zeroed out. Averaging that row in as if it were a real zero-cost,
    near-instant scene would silently drag the seconds/cost estimate for
    every other layered/short run toward 0 -- exactly the "fabricated free"
    outcome this module must never produce. The all-zero row must be
    excluded from the average, not treated as a valid (if cheap) sample."""
    zero_row = {
        "ok": True,
        "mode": "layered",
        "script_length": "short",
        "scenes_generated": 3,
        "elapsed_s": 33.8,
        "token_usage": {"total_tokens": 0},
        "token_usage_by_role": {
            "extractor": {"prompt_tokens": 0, "completion_tokens": 0},
            "beat_expander": {"prompt_tokens": 0, "completion_tokens": 0},
            "scene_writer": {"prompt_tokens": 0, "completion_tokens": 0},
        },
    }
    result = estimate.estimate_run(
        pipeline_mode="layered",
        script_length="short",
        models=_LAYERED_MODELS,
        history=[zero_row],
        prices=_PRICES,
        scenes=3,
    )
    # No usable history row survives -> falls all the way to "prior", not a
    # zero-second "history_mode_length" estimate built from the zero row.
    assert result.basis == "prior"
    assert result.seconds is not None and result.seconds > 0
    assert result.cost_usd is not None and result.cost_usd > 0


def test_never_returns_zero_cost_for_a_missing_price():
    # A missing price must read as None, never a plausible-looking 0.0.
    result = estimate.estimate_run(
        pipeline_mode="layered",
        script_length="short",
        models={"scene_writer": "openrouter/unknown"},
        history=[],
        prices={},
    )
    assert result.cost_usd is None
    assert result.cost_usd != 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
