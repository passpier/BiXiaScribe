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
    Chapter,
    Event,
    ExtractionResult,
    Faction,
    Outline,
    Stat,
    Truth,
    parse_model_json,
)


def _extraction(**overrides) -> ExtractionResult:
    defaults = dict(
        npcs=[
            NPC(id="npc-1", name="甲", personality="剛烈", speech_style="直來直去"),
            NPC(id="npc-2", name="乙", personality="沉穩", speech_style="文縐縐"),
        ],
        stat=Stat(id="v1", name="v", init=0),
    )
    defaults.update(overrides)
    return ExtractionResult(**defaults)


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
        title="t", premise="p", chapters=[Chapter(id="ch-1", title="c", summary="s")]
    )
    return BeatSheet(outline=outline, beats=beats)


def _event_for(beat: Beat, summary: str | None = None) -> Event:
    return Event(
        id=beat.id,
        title=beat.summary,
        summary=summary if summary is not None else beat.summary,
        dialogue=[{"npc": "npc-1", "line": "..."}],
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

    # 340 (not 280): SessionDocument's faction_cards/chapter_card/
    # truth_public/truth_unlocked fields (GMUD world context) add fixed
    # per-document overhead even when unused (empty-list JSON), tightening
    # the effective budget for scene_summaries.
    doc = build_session_document(
        current, extraction, completed, beat_sheet=beat_sheet, max_tokens=340
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


# --- build_session_document(): max_tokens<=0 disables trimming (Phase 5) --


def test_max_tokens_zero_disables_trimming() -> None:
    extraction = _extraction()
    beats = [_beat(f"beat-{i}") for i in range(20)]
    completed = [_event_for(b, summary="場景摘要" * 30) for b in beats]
    current = _beat("beat-current")

    doc = build_session_document(
        current, extraction, completed, beat_sheet=_beat_sheet(beats), max_tokens=0
    )
    assert doc.omitted_scene_count == 0
    assert len(doc.scene_summaries) == len(beats)


def test_negative_max_tokens_treated_as_unlimited() -> None:
    extraction = _extraction()
    beats = [_beat(f"beat-{i}") for i in range(20)]
    completed = [_event_for(b, summary="場景摘要" * 30) for b in beats]
    current = _beat("beat-current")

    doc = build_session_document(
        current, extraction, completed, beat_sheet=_beat_sheet(beats), max_tokens=-1
    )
    assert doc.omitted_scene_count == 0
    assert len(doc.scene_summaries) == len(beats)


def test_max_tokens_none_still_reads_config_at_call_time() -> None:
    # Regression guard: max_tokens=None must fall back to whatever
    # config.SESSION_DOC_MAX_TOKENS is *at call time*, not a value captured
    # earlier -- same pattern test_document_respects_custom_max_tokens_
    # config_default() above already relies on, made explicit here.
    original = config.SESSION_DOC_MAX_TOKENS
    try:
        extraction = _extraction()
        beats = [_beat(f"beat-{i}") for i in range(20)]
        completed = [_event_for(b, summary="場景摘要" * 30) for b in beats]
        current = _beat("beat-current")

        config.SESSION_DOC_MAX_TOKENS = 1
        doc_tight = build_session_document(
            current, extraction, completed, beat_sheet=_beat_sheet(beats)
        )
        assert doc_tight.omitted_scene_count > 0

        config.SESSION_DOC_MAX_TOKENS = 100_000
        doc_loose = build_session_document(
            current, extraction, completed, beat_sheet=_beat_sheet(beats)
        )
        assert doc_loose.omitted_scene_count == 0
    finally:
        config.SESSION_DOC_MAX_TOKENS = original


# --- build_session_document(): GMUD world cards ---------------------------


def test_faction_cards_populated():
    extraction = _extraction(
        factions=[Faction(id="f1", name="少林", motive="肅清邪魔")],
    )
    beat = _beat("beat-a")
    doc = build_session_document(beat, extraction, [])
    assert any("f1" in c for c in doc.faction_cards)


def test_chapter_card_reflects_current_beat_chapter():
    extraction = _extraction()
    outline = Outline(
        title="t", premise="p",
        chapters=[Chapter(id="ch-1", title="啟程", summary="師父遇害")],
    )
    beat = Beat(id="beat-a", chapter_id="ch-1", summary="s")
    beat_sheet = BeatSheet(outline=outline, beats=[beat])
    doc = build_session_document(beat, extraction, [], beat_sheet=beat_sheet)
    assert len(doc.chapter_card) == 1
    assert "師父遇害" in doc.chapter_card[0]


def test_chapter_card_empty_without_beat_sheet():
    extraction = _extraction()
    beat = _beat("beat-a")
    doc = build_session_document(beat, extraction, [])
    assert doc.chapter_card == []


def test_truth_public_always_visible():
    extraction = _extraction(truth=Truth(public="江湖傳聞甲派滅門", hidden="幕後主使是丙"))
    beat = _beat("beat-a")
    doc = build_session_document(beat, extraction, [])
    assert doc.truth_public == ["江湖傳聞甲派滅門"]


def test_truth_unlocked_only_includes_reveals_up_to_current_chapter():
    outline = Outline(
        title="t", premise="p",
        chapters=[
            Chapter(id="ch-1", title="c1", summary="s"),
            Chapter(id="ch-2", title="c2", summary="s"),
        ],
    )
    extraction = _extraction(truth=Truth(
        revealed=["早期真相", "後期真相"],
        hidden="幕後黑手",
    ))
    # chapter index 0 (ch-1) -> unlocks revealed[:0] = nothing yet
    beat = Beat(id="beat-a", chapter_id="ch-1", summary="s")
    beat_sheet = BeatSheet(outline=outline, beats=[beat])
    doc = build_session_document(beat, extraction, [], beat_sheet=beat_sheet)
    assert doc.truth_unlocked == []

    # chapter index 1 (ch-2) -> unlocks revealed[:1] = the first fact only
    beat2 = Beat(id="beat-b", chapter_id="ch-2", summary="s")
    beat_sheet2 = BeatSheet(outline=outline, beats=[beat2])
    doc2 = build_session_document(beat2, extraction, [], beat_sheet=beat_sheet2)
    assert doc2.truth_unlocked == ["早期真相"]


def test_hidden_truth_never_appears_in_any_built_session_document():
    outline = Outline(
        title="t", premise="p",
        chapters=[
            Chapter(id="ch-1", title="c1", summary="s"),
            Chapter(id="ch-2", title="c2", summary="s"),
        ],
    )
    extraction = _extraction(truth=Truth(
        public="公開事實",
        revealed=["逐步真相"],
        hidden="絕對機密幕後黑手",
    ))
    for chapter_id in ("ch-1", "ch-2"):
        beat = Beat(id=f"beat-{chapter_id}", chapter_id=chapter_id, summary="s")
        beat_sheet = BeatSheet(outline=outline, beats=[beat])
        doc = build_session_document(beat, extraction, [], beat_sheet=beat_sheet)
        assert "絕對機密幕後黑手" not in doc.model_dump_json()


def test_gmud_cards_survive_tight_budget_alongside_current_beat():
    extraction = _extraction(
        factions=[Faction(id="f1", name="少林")],
        truth=Truth(public="公開事實"),
    )
    beats = [_beat(f"beat-{i}") for i in range(20)]
    completed = [_event_for(b, summary="場景" * 100) for b in beats]
    current = _beat("beat-current")

    doc = build_session_document(
        current, extraction, completed, beat_sheet=_beat_sheet(beats), max_tokens=1
    )
    assert doc.faction_cards == ["f1｜少林："]
    assert doc.truth_public == ["公開事實"]


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
    test_max_tokens_zero_disables_trimming()
    test_negative_max_tokens_treated_as_unlimited()
    test_max_tokens_none_still_reads_config_at_call_time()
    print("All tests passed.")
