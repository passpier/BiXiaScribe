"""Unit tests for wire.py's lenient-mirror round trip. No LLM/network
involved -- pure pydantic model manipulation, mirroring
tests/test_guardrails.py's split between pure-function and integration
tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import wire  # noqa: E402
from bixiascribe.schema import (  # noqa: E402
    Choice,
    Event,
    Meta,
    Script,
)


def test_lenient_mirror_has_no_required_fields():
    mirror = wire.lenient_mirror(Event)
    assert all(not info.is_required() for info in mirror.model_fields.values())


def test_lenient_mirror_accepts_missing_required_field():
    mirror = wire.lenient_mirror(Event)
    # No 'id' at all -- would raise against the strict Event.
    obj = mirror.model_validate({})
    assert obj is not None


def test_missing_next_survives_as_empty_string():
    """The exact failure observed against a real deepseek-v4-flash-0731 run
    (see docs/DESIGN_NOTES.md's phase-zero probe): a choice with every
    other field filled in, but no `next` key at all."""
    mirror = wire.lenient_mirror(Event)
    data = {
        "id": "ev1",
        "title": "t",
        "summary": "s",
        "choices": [
            {
                "id": "b1",
                "text": "x",
                "payoff_at": "ch1",
            }
        ],
    }
    obj = mirror.model_validate(data)
    strict = wire.to_strict(obj, Event)
    assert isinstance(strict, Event)
    assert strict.choices[0].next == ""
    assert strict.choices[0].payoff_at == "ch1"


def test_full_script_round_trips_identically():
    script = Script(
        meta=Meta(title="t"),
        events=[
            Event(
                id="ev1",
                title="a",
                summary="s",
                choices=[Choice(id="b1", text="x", next="ev1")],
            )
        ],
    )
    mirror = wire.lenient_mirror(Script)
    lenient = mirror.model_validate(script.model_dump())
    round_tripped = wire.to_strict(lenient, Script)
    assert round_tripped.model_dump() == script.model_dump()


def test_lenient_mirror_is_cached_per_model():
    assert wire.lenient_mirror(Event) is wire.lenient_mirror(Event)


def test_to_strict_of_none_uses_all_defaults():
    strict = wire.to_strict(None, Event)
    assert strict.id == ""
    assert strict.choices == []


if __name__ == "__main__":
    test_lenient_mirror_has_no_required_fields()
    test_lenient_mirror_accepts_missing_required_field()
    test_missing_next_survives_as_empty_string()
    test_full_script_round_trips_identically()
    test_lenient_mirror_is_cached_per_model()
    test_to_strict_of_none_uses_all_defaults()
    print("All tests passed.")
