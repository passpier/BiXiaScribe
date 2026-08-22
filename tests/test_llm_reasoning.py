"""Unit tests for llm.py::build_llm()'s reasoning_effort wiring -- mocks
crewai.LLM itself so these run without a real OpenRouter key/network call,
verifying "default" stays a byte-identical no-op (design.md 決策二) and other
values pass through crewai's native reasoning_effort field unchanged,
coexisting with provider-routing additional_params (see llm.py's own
docstring on why extra_body was avoided)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crewai  # noqa: E402

from bixiascribe import config  # noqa: E402
from bixiascribe.llm import ROLE_WRITER, ModelChoice, build_llm  # noqa: E402


class _CapturedLLM:
    """Stand-in for crewai.LLM that just remembers its kwargs."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _CapturedLLM.last_kwargs = kwargs


def _with_real_backend(fn):
    """Run `fn` with config.LLM_BACKEND temporarily set to "openrouter" (so
    build_llm() takes the real, non-FakeLLM path), a fake API key (so
    require_openrouter_key() doesn't raise), and crewai.LLM monkeypatched to
    _CapturedLLM -- restoring all three afterward."""
    original_backend = config.LLM_BACKEND
    original_key = config.OPENROUTER_API_KEY
    original_llm = crewai.LLM
    config.LLM_BACKEND = "openrouter"
    config.OPENROUTER_API_KEY = "test-key"
    crewai.LLM = _CapturedLLM
    try:
        fn()
    finally:
        config.LLM_BACKEND = original_backend
        config.OPENROUTER_API_KEY = original_key
        crewai.LLM = original_llm


def test_default_reasoning_effort_passes_no_kwarg():
    def run():
        _CapturedLLM.last_kwargs = None
        models = ModelChoice(reasoning_effort="default")
        build_llm(ROLE_WRITER, models)
        assert "reasoning_effort" not in _CapturedLLM.last_kwargs

    _with_real_backend(run)


def test_none_low_medium_high_pass_through_unchanged():
    def run():
        for effort in ("none", "low", "medium", "high"):
            _CapturedLLM.last_kwargs = None
            models = ModelChoice(reasoning_effort=effort)
            build_llm(ROLE_WRITER, models)
            assert _CapturedLLM.last_kwargs["reasoning_effort"] == effort

    _with_real_backend(run)


def test_garbage_reasoning_effort_normalizes_to_default_no_kwarg():
    def run():
        _CapturedLLM.last_kwargs = None
        models = ModelChoice(reasoning_effort="garbage")
        build_llm(ROLE_WRITER, models)
        assert "reasoning_effort" not in _CapturedLLM.last_kwargs

    _with_real_backend(run)


def test_reasoning_effort_coexists_with_provider_routing():
    def run():
        original_only = config.LLM_PROVIDER_ONLY
        config.LLM_PROVIDER_ONLY = ["DeepInfra"]
        try:
            _CapturedLLM.last_kwargs = None
            models = ModelChoice(reasoning_effort="high")
            build_llm(ROLE_WRITER, models)
            kwargs = _CapturedLLM.last_kwargs
            assert kwargs["reasoning_effort"] == "high"
            assert kwargs["additional_params"]["extra_body"]["provider"]["only"] == ["DeepInfra"]
        finally:
            config.LLM_PROVIDER_ONLY = original_only

    _with_real_backend(run)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
