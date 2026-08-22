"""Dependency-free parsing/resolution of a script-length target (config.
SCRIPT_LENGTH / --script-length / Variant.script_length). Zero imports from
config, crew/, or crewai, so both config.py (validating SCRIPT_LENGTH at
import time, crewai-free) and crew/tasks.py (building prompts, crewai-heavy)
can depend on this without a circular import -- see design.md's "A
dependency-free length.py module owns parsing".

Accepts either a bare preset name (short/medium/long) or a
`custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT` string
with any subset of the four fields; omitted fields are derived from
`events`. Any unparseable/unrecognized value falls back to `short`, matching
the degrade-not-crash convention used elsewhere (CAUSAL_VALIDATION,
PIPELINE_MODE).

Phase 4 (2026-08-22, see openspec/changes/2026-08-22-slim-script-schema-mvp)
dropped the fifth target, `scene_mix` -- Event.scene_kind/Beat.scene_kind no
longer exist to measure a main:flavor ratio against."""
from __future__ import annotations

import math
from dataclasses import dataclass

PRESETS: dict[str, dict[str, str]] = {
    "short": {
        "events": "2",
        "chapters": "1-2",
        "beats_per_chapter": "1",
        "min_dialogue": "一段",
    },
    "medium": {
        "events": "8-12",
        "chapters": "3-4",
        "beats_per_chapter": "2-3",
        "min_dialogue": "二至三段",
    },
    "long": {
        "events": "15-24",
        "chapters": "5-6",
        "beats_per_chapter": "3-4",
        "min_dialogue": "三段以上",
    },
}

_CUSTOM_FIELDS = ("events", "chapters", "beats_per_chapter", "min_dialogue")

# Per-field disclosure for the UI's 自訂篇幅 inputs, grounded in the verified
# crew/tasks.py call sites that actually read each field -- not every field
# affects both pipeline modes' *output*, and filling in the wrong one for the
# active mode silently does nothing to the generated script (see
# proposal.md's "Why"). `affects` is one of _AFFECTS_LABEL's keys below.
FIELD_HELP: dict[str, dict[str, str]] = {
    "events": {
        "label": "events（事件數）",
        "affects": "legacy",
        "help": (
            "只餵給 legacy pipeline 的編劇任務（crew/tasks.py:184），決定一次性生成的 "
            "event 數量下限。layered pipeline 的場次數由 chapters/beats_per_chapter 決定，"
            "不讀這個欄位——但無論哪種模式，events 都還是會影響 LengthSpec.events_scale，"
            "也就是預估場次/成本/時間的估算基準。"
        ),
    },
    "chapters": {
        "label": "chapters（章數）",
        "affects": "layered",
        "help": "只餵給 layered pipeline 的排場任務（crew/tasks.py:337），決定分章大綱的章數下限。",
    },
    "beats_per_chapter": {
        "label": "beats_per_chapter（每章場次數）",
        "affects": "layered",
        "help": "只餵給 layered pipeline 的排場任務（crew/tasks.py:338），決定每章 beat 數下限。",
    },
    "min_dialogue": {
        "label": "min_dialogue（最少台詞段落）",
        "affects": "both",
        "help": (
            "兩種管線都會讀：legacy 的對話任務（crew/tasks.py:210,213）與 layered 的分場"
            "對話任務（crew/tasks.py:388,390），決定每場對話的最少段落數。"
        ),
    },
}

# UI 警告後綴用的標籤——對照 FIELD_HELP 的 affects 值。
_AFFECTS_LABEL: dict[str, str] = {
    "legacy": "⚠ 僅 legacy",
    "layered": "⚠ 僅 layered",
    "both": "",
}


def _events_lower_bound(events: str) -> int:
    """"15-24" -> 15, "2" -> 2. Falls back to short's own baseline (2) if
    unparseable -- callers only reach this after events is known non-empty."""
    head = events.split("-", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return int(PRESETS["short"]["events"])


def _derive_from_events(events: int) -> dict[str, str]:
    chapters = max(1, math.ceil(events / 5))
    beats_per_chapter = max(1, math.ceil(events / chapters))
    if events <= 4:
        min_dialogue = "一段"
    elif events <= 14:
        min_dialogue = "二至三段"
    else:
        min_dialogue = "三段以上"
    return {
        "events": str(events),
        "chapters": str(chapters),
        "beats_per_chapter": str(beats_per_chapter),
        "min_dialogue": min_dialogue,
    }


def _parse_custom(body: str) -> dict[str, str] | None:
    """Parses the part after "custom:". Returns None (caller falls back to
    short) if no field parses at all or `events` is present but non-numeric."""
    fields: dict[str, str] = {}
    for part in body.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if key in _CUSTOM_FIELDS and value:
            fields[key] = value

    if not fields:
        return None

    if "events" in fields:
        try:
            events = int(fields["events"])
        except ValueError:
            return None
    else:
        events = int(PRESETS["short"]["events"])

    resolved = _derive_from_events(events)
    if "chapters" in fields:
        resolved["chapters"] = fields["chapters"]
        try:
            chapters = int(fields["chapters"])
            resolved["beats_per_chapter"] = str(max(1, math.ceil(events / chapters)))
        except (ValueError, ZeroDivisionError):
            pass
    resolved.update({k: v for k, v in fields.items() if k != "chapters"})
    return resolved


@dataclass(frozen=True)
class LengthSpec:
    events: str
    chapters: str
    beats_per_chapter: str
    min_dialogue: str
    preset: str | None = None

    @property
    def targets(self) -> dict[str, str]:
        return {
            "events": self.events,
            "chapters": self.chapters,
            "beats_per_chapter": self.beats_per_chapter,
            "min_dialogue": self.min_dialogue,
        }

    @property
    def canonical(self) -> str:
        if self.preset is not None:
            return self.preset
        return (
            f"custom:events={self.events},chapters={self.chapters},"
            f"beats_per_chapter={self.beats_per_chapter},"
            f"min_dialogue={self.min_dialogue}"
        )

    @property
    def events_scale(self) -> float:
        baseline = _events_lower_bound(PRESETS["short"]["events"])
        return _events_lower_bound(self.events) / baseline


def _spec_from_preset(name: str) -> LengthSpec:
    target = PRESETS[name]
    return LengthSpec(
        events=target["events"],
        chapters=target["chapters"],
        beats_per_chapter=target["beats_per_chapter"],
        min_dialogue=target["min_dialogue"],
        preset=name,
    )


def parse_length_spec(value: str) -> LengthSpec:
    value = (value or "").strip()
    lowered = value.lower()
    if lowered in PRESETS:
        return _spec_from_preset(lowered)

    if lowered.startswith("custom:"):
        parsed = _parse_custom(value[len("custom:"):])
        if parsed is not None:
            return LengthSpec(
                events=parsed["events"],
                chapters=parsed["chapters"],
                beats_per_chapter=parsed["beats_per_chapter"],
                min_dialogue=parsed["min_dialogue"],
                preset=None,
            )

    return _spec_from_preset("short")
