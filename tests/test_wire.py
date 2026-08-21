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
    Branch,
    Event,
    Script,
    Variable,
)


def test_lenient_mirror_has_no_required_fields():
    mirror = wire.lenient_mirror(Event)
    assert all(not info.is_required() for info in mirror.model_fields.values())


def test_lenient_mirror_accepts_missing_required_field():
    mirror = wire.lenient_mirror(Event)
    # No 'id'/'title'/'location'/'summary' at all -- would raise against
    # the strict Event.
    obj = mirror.model_validate({})
    assert obj is not None


def test_missing_next_event_id_survives_as_empty_string():
    """The exact failure observed against a real deepseek-v4-flash-0731 run
    (see docs/DESIGN_NOTES.md's phase-zero probe): a branch with every
    other field filled in, but no next_event_id key at all."""
    mirror = wire.lenient_mirror(Event)
    data = {
        "id": "ev1",
        "title": "t",
        "location": "l",
        "summary": "s",
        "branches": [
            {
                "id": "b1",
                "choice_text": "x",
                "converges_to_event_id": "ev-ch1-converge",
            }
        ],
    }
    obj = mirror.model_validate(data)
    strict = wire.to_strict(obj, Event)
    assert isinstance(strict, Event)
    assert strict.branches[0].next_event_id == ""
    assert strict.branches[0].converges_to_event_id == "ev-ch1-converge"


def test_full_script_round_trips_identically():
    script = Script(
        title="t",
        premise="p",
        variables=[Variable(id="v1", name="n", initial=1)],
        events=[
            Event(
                id="ev1",
                title="a",
                location="l",
                summary="s",
                branches=[Branch(id="b1", choice_text="x", next_event_id="ev1")],
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
    assert strict.branches == []


if __name__ == "__main__":
    test_lenient_mirror_has_no_required_fields()
    test_lenient_mirror_accepts_missing_required_field()
    test_missing_next_event_id_survives_as_empty_string()
    test_full_script_round_trips_identically()
    test_lenient_mirror_is_cached_per_model()
    test_to_strict_of_none_uses_all_defaults()
    print("All tests passed.")
