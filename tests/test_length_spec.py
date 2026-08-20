"""Unit tests for length.py's preset/custom script-length parsing. No
network/API/crewai dependency -- length.py is deliberately dependency-free
(see its module docstring)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.length import parse_length_spec  # noqa: E402


def test_short_preset():
    spec = parse_length_spec("short")
    assert spec.targets == {
        "events": "2",
        "chapters": "1-2",
        "beats_per_chapter": "1",
        "min_dialogue": "一段",
        "scene_mix": "主要場景略多於調味場景",
    }
    assert spec.canonical == "short"


def test_medium_preset():
    spec = parse_length_spec("medium")
    assert spec.targets["events"] == "8-12"
    assert spec.canonical == "medium"


def test_long_preset():
    spec = parse_length_spec("long")
    assert spec.targets["events"] == "15-24"
    assert spec.canonical == "long"


def test_default_empty_value_falls_back_to_short():
    spec = parse_length_spec("")
    assert spec.canonical == "short"


def test_preset_case_insensitive():
    spec = parse_length_spec("MEDIUM")
    assert spec.canonical == "medium"


def test_full_custom_spec_uses_exact_values():
    spec = parse_length_spec(
        "custom:events=20,chapters=4,beats_per_chapter=5,min_dialogue=兩段以上,"
        "scene_mix=主要場景遠多於調味場景"
    )
    assert spec.targets == {
        "events": "20",
        "chapters": "4",
        "beats_per_chapter": "5",
        "min_dialogue": "兩段以上",
        "scene_mix": "主要場景遠多於調味場景",
    }
    assert spec.preset is None


def test_partial_custom_spec_derives_missing_fields_from_events():
    spec = parse_length_spec("custom:events=20")
    assert spec.targets["events"] == "20"
    assert spec.targets["chapters"] == "4"
    assert spec.targets["beats_per_chapter"] == "5"
    assert spec.targets["min_dialogue"] == "三段以上"
    assert spec.targets["scene_mix"] == "主要場景略多於調味場景，約 3:2"


def test_partial_custom_spec_overrides_only_given_fields():
    spec = parse_length_spec("custom:events=20,chapters=10")
    assert spec.targets["chapters"] == "10"
    assert spec.targets["beats_per_chapter"] == "2"


def test_custom_spec_without_events_defaults_events_to_short_baseline():
    spec = parse_length_spec("custom:chapters=6")
    assert spec.targets["events"] == "2"
    assert spec.targets["chapters"] == "6"


def test_malformed_custom_string_falls_back_to_short():
    spec = parse_length_spec("custom:evnets=20")
    assert spec.canonical == "short"


def test_custom_non_numeric_events_falls_back_to_short():
    spec = parse_length_spec("custom:events=lots")
    assert spec.canonical == "short"


def test_unrecognized_non_preset_string_falls_back_to_short():
    spec = parse_length_spec("glacial")
    assert spec.canonical == "short"


def test_canonical_round_trips_through_parse_again():
    spec = parse_length_spec("custom:events=20")
    round_tripped = parse_length_spec(spec.canonical)
    assert round_tripped.targets == spec.targets
    assert round_tripped.canonical == spec.canonical


def test_preset_canonical_round_trips():
    spec = parse_length_spec("long")
    round_tripped = parse_length_spec(spec.canonical)
    assert round_tripped.targets == spec.targets


def test_events_scale_short_baseline_is_one():
    assert parse_length_spec("short").events_scale == 1.0


def test_events_scale_medium():
    assert parse_length_spec("medium").events_scale == 4.0


def test_events_scale_uses_range_lower_bound():
    spec = parse_length_spec("long")
    assert spec.events_scale == 15 / 2


def test_events_scale_custom():
    spec = parse_length_spec("custom:events=20")
    assert spec.events_scale == 10.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
