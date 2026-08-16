"""Unit tests for crew/context_builder.py (BiXiaScribe 重構 Phase 5):
estimate_tokens()'s CJK-aware heuristic, and build_session_document()'s
budget/priority behavior. No LLM/network involved -- this module never
calls an LLM -- so no LLM_BACKEND setup is needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402
from bixiascribe.crew.context_builder import (  # noqa: E402
    build_session_document,
    estimate_tokens,
)
from bixiascribe.schema import (  # noqa: E402
    NPC,
    Beat,
    BeatSheet,
    ChapterOutline,
    Event,
    ExtractionResult,
    Outline,
    Variable,
    parse_model_json,
)


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        npcs=[
            NPC(
                id="npc-1", name="甲", identity="俠客",
                personality="剛烈", speech_style="直來直去",
            ),
            NPC(
                id="npc-2", name="乙", identity="掌門",
                personality="沉穩", speech_style="文縐縐",
            ),
        ],
        variables=[Variable(id="v1", name="v", initial=0)],
    )


def _beat(id_: str, deps: list[str] | None = None, npc_ids: list[str] | None = None) -> Beat:
    return Beat(
        id=id_,
        chapter_id="ch-1",
        summary=f"summary-{id_}",
        npc_ids=npc_ids or [],
        causal_deps=deps or [],
    )


def _beat_sheet(beats: list[Beat]) -> BeatSheet:
    outline = Outline(
        title="t", premise="p", chapters=[ChapterOutline(id="ch-1", title="c", summary="s")]
    )
    return BeatSheet(outline=outline, beats=beats)


def _event_for(beat: Beat, summary: str | None = None) -> Event:
    return Event(
        id=beat.id,
        title=beat.summary,
        location="",
        summary=summary if summary is not None else beat.summary,
        dialogue=[{"npc_id": "npc-1", "line": "..."}],
    )


# --- estimate_tokens() ---------------------------------------------------


def test_estimate_tokens_pure_cjk_counts_one_per_char() -> None:
    text = "獨孤九劍六脈神劍"
    assert estimate_tokens(text) == len(text)


def test_estimate_tokens_pure_ascii_is_quarter_per_char_ceiling() -> None:
    text = "a" * 20
    assert estimate_tokens(text) == 5


def test_estimate_tokens_mixed_sums_both_parts() -> None:
    text = "俠客" + "abcd" * 2  # 2 CJK chars + 8 ASCII chars
    assert estimate_tokens(text) == 2 + 2


def test_estimate_tokens_empty_string_is_zero() -> None:
    assert estimate_tokens("") == 0


# --- build_session_document(): NPC filter parity --------------------------


def test_npc_filter_selects_only_beat_named_npcs() -> None:
    extraction = _extraction()
    beat = _beat("beat-a", npc_ids=["npc-2"])
    doc = build_session_document(beat, extraction, [])
    assert len(doc.character_cards) == 1
    assert "npc-2" in doc.character_cards[0]


def test_npc_filter_falls_back_to_all_npcs_when_no_match() -> None:
    extraction = _extraction()
    beat = _beat("beat-a", npc_ids=["npc-does-not-exist"])
    doc = build_session_document(beat, extraction, [])
    assert len(doc.character_cards) == len(extraction.npcs)


# --- build_session_document(): budget (Phase 5 gate 1) --------------------


def test_document_fits_under_max_tokens_with_many_scenes() -> None:
    extraction = _extraction()
    beats = [_beat(f"beat-{i}") for i in range(50)]
    beat_sheet = _beat_sheet(beats)
    completed = [_event_for(b, summary="很長的場景摘要內容" * 10) for b in beats[:40]]
    current = _beat("beat-current")

    doc = build_session_document(
        current, extraction, completed, beat_sheet=beat_sheet, max_tokens=500
    )
    assert estimate_tokens(doc.model_dump_json()) <= 500


def test_document_respects_custom_max_tokens_config_default() -> None:
    # No explicit max_tokens -- falls back to config.SESSION_DOC_MAX_TOKENS.
    extraction = _extraction()
    beats = [_beat(f"beat-{i}") for i in range(20)]
    completed = [_event_for(b, summary="場景" * 50) for b in beats]
    current = _beat("beat-current")

    doc = build_session_document(current, extraction, completed, beat_sheet=_beat_sheet(beats))
    assert estimate_tokens(doc.model_dump_json()) <= config.SESSION_DOC_MAX_TOKENS


# --- build_session_document(): non-growth (Phase 5 gate 2) ----------------


def test_document_size_does_not_grow_unboundedly_with_scene_count() -> None:
    extraction = _extraction()
    current = _beat("beat-current")

    def _doc_for(n: int):
        beats = [_beat(f"beat-{i}") for i in range(n)]
        completed = [_event_for(b, summary="場景摘要內容範例文字") for b in beats]
        return build_session_document(
            current, extraction, completed, beat_sheet=_beat_sheet(beats), max_tokens=800
        )

    doc_small = _doc_for(3)
    doc_medium = _doc_for(30)
    doc_large = _doc_for(200)

    size_small = len(doc_small.model_dump_json())
    size_medium = len(doc_medium.model_dump_json())
    size_large = len(doc_large.model_dump_json())

    # Once the budget is saturated, growing the input scene count further
    # must not keep growing the serialized document.
    assert size_medium <= size_small + 1500  # some growth expected before saturation
    assert size_large <= size_medium * 1.5  # must not keep scaling with input size


def test_omitted_scene_count_accounts_for_every_dropped_scene() -> None:
    extraction = _extraction()
    beats = [_beat(f"beat-{i}") for i in range(30)]
    completed = [_event_for(b, summary="場景摘要" * 20) for b in beats]
    current = _beat("beat-current")

    doc = build_session_document(
        current, extraction, completed, beat_sheet=_beat_sheet(beats), max_tokens=400
    )
    assert doc.omitted_scene_count == len(completed) - len(doc.scene_summaries)
    assert doc.omitted_scene_count > 0  # this test's inputs must actually exceed budget


# --- build_session_document(): causal priority ----------------------------


def test_causal_ancestor_survives_truncation_over_unrelated_older_scene() -> None:
    extraction = _extraction()
    ancestor = _beat("beat-ancestor")
    unrelated = _beat("beat-unrelated")
    current = _beat("beat-current", deps=["beat-ancestor"])
    beat_sheet = _beat_sheet([ancestor, unrelated, current])

    # Completion order: unrelated first (older), ancestor second (newer) --
    # if priority were purely recency, "unrelated" would win. It must not.
    completed = [
        _event_for(unrelated, summary="不相關的場景" * 30),
        _event_for(ancestor, summary="因果前置場景" * 30),
    ]

    doc = build_session_document(
        current, extraction, completed, beat_sheet=beat_sheet, max_tokens=230
    )
    ids_kept = [s.split("｜", 1)[0] for s in doc.scene_summaries]
    assert "beat-ancestor" in ids_kept
    assert "beat-unrelated" not in ids_kept


def test_transitive_causal_ancestors_are_included() -> None:
    extraction = _extraction()
    grandparent = _beat("beat-gp")
    parent = _beat("beat-parent", deps=["beat-gp"])
    current = _beat("beat-current", deps=["beat-parent"])
    beat_sheet = _beat_sheet([grandparent, parent, current])
    completed = [_event_for(grandparent), _event_for(parent)]

    doc = build_session_document(
        current, extraction, completed, beat_sheet=beat_sheet, max_tokens=4000
    )
    ids_kept = [s.split("｜", 1)[0] for s in doc.scene_summaries]
    assert "beat-gp" in ids_kept
    assert "beat-parent" in ids_kept


def test_no_beat_sheet_treats_all_scenes_as_non_ancestors() -> None:
    extraction = _extraction()
    other = _beat("beat-other")
    current = _beat("beat-current")
    completed = [_event_for(other)]

    # Without a beat_sheet, causal ancestry can't be computed -- must not
    # raise, must still include the scene as a (recency-ranked) summary.
    doc = build_session_document(current, extraction, completed)
    assert len(doc.scene_summaries) == 1


# --- build_session_document(): FakeLLM compatibility (regression guard) ---


def test_current_beat_recoverable_via_parse_model_json() -> None:
    """llm.py::FakeLLM's scene_writer branch recovers the target beat by
    scanning the prompt text for a JSON object that validates as Beat. If
    current_beat were serialized as a doubly-encoded string instead of a
    typed nested object, this would return None and every offline test
    would silently regress to FakeLLM's "beat is None" fallback."""
    extraction = _extraction()
    beat = _beat("beat-target", npc_ids=["npc-1"])
    doc = build_session_document(beat, extraction, [])

    recovered = parse_model_json(doc.model_dump_json(), Beat)
    assert recovered is not None
    assert recovered.id == "beat-target"


def test_never_drops_character_cards_or_current_beat() -> None:
    extraction = _extraction()
    beats = [_beat(f"beat-{i}") for i in range(20)]
    completed = [_event_for(b, summary="場景" * 100) for b in beats]
    current = _beat("beat-current", npc_ids=["npc-1", "npc-2"])

    # An unreasonably tight budget: even if it can't be honored, character
    # cards/current_beat must survive intact.
    doc = build_session_document(
        current, extraction, completed, beat_sheet=_beat_sheet(beats), max_tokens=1
    )
    assert len(doc.character_cards) == len(extraction.npcs)
    assert doc.current_beat.id == "beat-current"


if __name__ == "__main__":
    test_estimate_tokens_pure_cjk_counts_one_per_char()
    test_estimate_tokens_pure_ascii_is_quarter_per_char_ceiling()
    test_estimate_tokens_mixed_sums_both_parts()
    test_estimate_tokens_empty_string_is_zero()
    test_npc_filter_selects_only_beat_named_npcs()
    test_npc_filter_falls_back_to_all_npcs_when_no_match()
    test_document_fits_under_max_tokens_with_many_scenes()
    test_document_respects_custom_max_tokens_config_default()
    test_document_size_does_not_grow_unboundedly_with_scene_count()
    test_omitted_scene_count_accounts_for_every_dropped_scene()
    test_causal_ancestor_survives_truncation_over_unrelated_older_scene()
    test_transitive_causal_ancestors_are_included()
    test_no_beat_sheet_treats_all_scenes_as_non_ancestors()
    test_current_beat_recoverable_via_parse_model_json()
    test_never_drops_character_cards_or_current_beat()
    print("All tests passed.")
