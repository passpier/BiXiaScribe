"""Unit tests for crew/causal.py (BiXiaScribe 重構 Phase 6): fact
normalization, graph construction, per-scene consistency checking, and the
config.CAUSAL_VALIDATION-driven wiring into crew/orchestrator.py's
dispatch_next()/dispatch_batch()/confirm_batch()/run_layered(). Mirrors
tests/test_orchestrator.py's conventions -- hand-written StageRunners
stand-ins, no crewai/LLM involved except where the fake-backend degrade is
the point of the test -- so this suite needs no real API key and runs
instantly.
"""
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402
from bixiascribe.crew import orchestrator  # noqa: E402
from bixiascribe.crew.causal import (  # noqa: E402
    Fact,
    build_graph,
    check_scene_consistency,
    event_to_node,
    facts_conflict,
    parse_fact,
)
from bixiascribe.crew.orchestrator import (  # noqa: E402
    StageRunners,
    detect_stage,
    dispatch_batch,
    dispatch_next,
    run_layered,
)
from bixiascribe.crew.pipeline import PipelineError  # noqa: E402
from bixiascribe.schema import (  # noqa: E402
    NPC,
    Beat,
    BeatSheet,
    Branch,
    ChapterOutline,
    Event,
    ExtractionResult,
    Outline,
    Trigger,
    Variable,
    validate_references,
)

REQUIREMENT = "test requirement"


@contextmanager
def _isolated_state_dir():
    """Same pattern as tests/test_orchestrator.py -- orchestrator.state_dir()
    reads config.BIXIA_STATE_DIR live, so this reassignment is picked up
    immediately."""
    original = config.BIXIA_STATE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.BIXIA_STATE_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            config.BIXIA_STATE_DIR = original


@contextmanager
def _isolated_causal_mode(mode: str):
    """Force config.CAUSAL_VALIDATION for the duration of one test --
    config.CAUSAL_VALIDATION is read at call time throughout orchestrator.py
    (_refresh_graph()/_validate_scene()/run_layered()'s proofread check),
    same convention as _isolated_fake_llm_backend()/_isolated_state_dir()
    in tests/test_orchestrator.py."""
    original = config.CAUSAL_VALIDATION
    config.CAUSAL_VALIDATION = mode
    try:
        yield
    finally:
        config.CAUSAL_VALIDATION = original


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        npcs=[NPC(id="npc-1", name="A", identity="x", personality="y", speech_style="z")],
        variables=[Variable(id="v1", name="v", initial=0)],
    )


def _beat(id_: str, deps: list[str] | None = None) -> Beat:
    return Beat(id=id_, chapter_id="ch-1", summary=f"summary-{id_}", causal_deps=deps or [])


def _beat_sheet(beats: list[Beat]) -> BeatSheet:
    outline = Outline(
        title="t", premise="p", chapters=[ChapterOutline(id="ch-1", title="c", summary="s")]
    )
    return BeatSheet(outline=outline, beats=beats)


def _event_for(
    beat: Beat, *, triggers: list[Trigger] | None = None, branches: list[Branch] | None = None
) -> Event:
    return Event(
        id=beat.id,
        title=beat.summary,
        location="",
        summary=beat.summary,
        triggers=triggers or [],
        dialogue=[{"npc_id": "npc-1", "line": "..."}],
        branches=branches or [],
    )


# --- parse_fact / facts_conflict --------------------------------------------


def test_parse_fact_recognizes_predicate_after_time_marker() -> None:
    fact = parse_fact("柳寡婦 已死亡")
    assert fact == Fact(subject="柳寡婦", predicate="死亡", negated=False)


def test_parse_fact_recognizes_predicate_with_no_marker() -> None:
    fact = parse_fact("柳寡婦 存活")
    assert fact == Fact(subject="柳寡婦", predicate="存活", negated=False)


def test_parse_fact_recognizes_negation_prefix() -> None:
    fact = parse_fact("任務未完成")
    assert fact is not None
    assert fact.subject == "任務"
    assert fact.negated is True


def test_parse_fact_recognizes_relation_operator() -> None:
    fact = parse_fact("flag_x == true")
    assert fact == Fact(subject="flag_x", predicate="true", negated=False)


def test_parse_fact_negation_operator_sets_negated() -> None:
    fact = parse_fact("flag_x != true")
    assert fact == Fact(subject="flag_x", predicate="true", negated=True)


def test_parse_fact_unrecognized_text_returns_none() -> None:
    assert parse_fact("這句話沒有任何可辨識的述詞") is None
    assert parse_fact("") is None
    assert parse_fact("   ") is None


def test_facts_conflict_same_predicate_opposite_polarity() -> None:
    a = Fact(subject="柳寡婦", predicate="死亡", negated=False)
    b = Fact(subject="柳寡婦", predicate="死亡", negated=True)
    assert facts_conflict(a, b)
    assert facts_conflict(b, a)


def test_facts_conflict_antonym_pair_same_polarity() -> None:
    alive = Fact(subject="柳寡婦", predicate="存活", negated=False)
    dead = Fact(subject="柳寡婦", predicate="死亡", negated=False)
    assert facts_conflict(alive, dead)


def test_facts_conflict_different_subject_never_conflicts() -> None:
    a = Fact(subject="柳寡婦", predicate="死亡", negated=False)
    b = Fact(subject="血衣門", predicate="死亡", negated=False)
    assert not facts_conflict(a, b)


def test_facts_conflict_unrelated_predicates_do_not_conflict() -> None:
    a = Fact(subject="柳寡婦", predicate="憤怒", negated=False)
    b = Fact(subject="柳寡婦", predicate="存活", negated=False)
    assert not facts_conflict(a, b)


# --- build_graph -------------------------------------------------------------


def test_event_to_node_pulls_from_triggers_and_branches() -> None:
    beat = _beat("beat-a")
    event = _event_for(
        beat,
        triggers=[Trigger(type="on_enter", condition="柳寡婦 存活")],
        branches=[Branch(id="b1", choice_text="x", effects="柳寡婦 死亡", next_event_id="beat-a")],
    )
    node = event_to_node(event)
    assert node.id == "beat-a"
    assert node.preconditions == ["柳寡婦 存活"]
    assert node.postconditions == ["柳寡婦 死亡"]


def test_build_graph_edges_from_causal_deps_and_branches() -> None:
    beat_a = _beat("beat-a")
    beat_b = _beat("beat-b", deps=["beat-a"])
    beat_sheet = _beat_sheet([beat_a, beat_b])
    event_a = _event_for(
        beat_a,
        branches=[Branch(id="b1", choice_text="x", next_event_id="beat-b")],
    )
    event_b = _event_for(beat_b)

    graph = build_graph(beat_sheet, [event_a, event_b])

    assert {n.id for n in graph.nodes} == {"beat-a", "beat-b"}
    edge_pairs = {(e.from_id, e.to_id) for e in graph.edges}
    assert ("beat-a", "beat-b") in edge_pairs  # from Beat.causal_deps
    assert ("beat-a", "beat-b") in edge_pairs  # also from the branch flow (same pair, fine)


def test_build_graph_skips_deps_and_branches_pointing_at_uncommitted_events() -> None:
    beat_a = _beat("beat-a", deps=["beat-missing"])
    beat_sheet = _beat_sheet([beat_a])
    event_a = _event_for(
        beat_a,
        branches=[Branch(id="b1", choice_text="x", next_event_id="evt-missing")],
    )

    graph = build_graph(beat_sheet, [event_a])

    assert graph.nodes == [event_to_node(event_a)]
    assert graph.edges == []


# --- check_scene_consistency (the plan's specified fixture) -----------------


def test_check_scene_consistency_catches_ancestor_postcondition_conflict() -> None:
    """scene A's branch effect kills off 柳寡婦; scene B (causal_deps=[A])
    has a trigger precondition that assumes 柳寡婦 is still alive. The
    legacy validate_references() gate can't see this at all -- it only
    checks npc_id/next_event_id existence."""
    beat_a = _beat("beat-a")
    beat_b = _beat("beat-b", deps=["beat-a"])
    beat_sheet = _beat_sheet([beat_a, beat_b])

    event_a = _event_for(
        beat_a,
        branches=[Branch(id="b1", choice_text="x", effects="柳寡婦 死亡", next_event_id="beat-b")],
    )
    event_b = _event_for(
        beat_b,
        triggers=[Trigger(type="on_enter", condition="柳寡婦 存活")],
    )

    problems = check_scene_consistency(beat_sheet, [event_a], event_b)
    assert len(problems) == 1
    assert "beat-b" in problems[0]
    assert "beat-a" in problems[0]

    # The old gate genuinely sees nothing wrong with this script.
    from bixiascribe.schema import Script

    script = Script(
        title="t",
        premise="p",
        npcs=_extraction().npcs,
        variables=[],
        events=[event_a, event_b],
    )
    assert validate_references(script) == []


def test_check_scene_consistency_unrelated_subject_is_not_flagged() -> None:
    beat_a = _beat("beat-a")
    beat_b = _beat("beat-b", deps=["beat-a"])
    beat_sheet = _beat_sheet([beat_a, beat_b])

    event_a = _event_for(
        beat_a,
        branches=[Branch(id="b1", choice_text="x", effects="血衣門 潰散", next_event_id="beat-b")],
    )
    event_b = _event_for(
        beat_b,
        triggers=[Trigger(type="on_enter", condition="柳寡婦 存活")],
    )

    assert check_scene_consistency(beat_sheet, [event_a], event_b) == []


def test_check_scene_consistency_no_ancestor_deps_is_never_flagged() -> None:
    beat_a = _beat("beat-a")
    beat_sheet = _beat_sheet([beat_a])
    event_a = _event_for(beat_a, triggers=[Trigger(type="on_enter", condition="柳寡婦 死亡")])
    assert check_scene_consistency(beat_sheet, [], event_a) == []


# --- orchestrator wiring -----------------------------------------------------


class ConflictingRunners:
    """Two beats, beat-a -> beat-b, where beat-a's committed scene kills
    off 柳寡婦 and beat-b's freshly-written scene assumes she's alive --
    the same conflict as the fixture above, driven through dispatch_next()/
    dispatch_batch() instead of calling check_scene_consistency() directly.
    An optional `repair_fixes` flag makes repair_scene() return a
    conflict-free replacement, for exercising the "repair" mode's happy
    path."""

    def __init__(self, repair_fixes: bool = False):
        self.beats = [_beat("beat-a"), _beat("beat-b", deps=["beat-a"])]
        self.repair_calls = 0
        self.repair_fixes = repair_fixes

    def extract(self, requirement, models, verbose):
        return _extraction(), "pydantic"

    def expand_beats(self, requirement, extraction, models, verbose):
        return _beat_sheet(self.beats), "pydantic"

    def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
        if beat.id == "beat-a":
            branch = Branch(id="b1", choice_text="x", effects="柳寡婦 死亡", next_event_id="beat-b")
            event = _event_for(beat, branches=[branch])
        else:
            event = _event_for(beat, triggers=[Trigger(type="on_enter", condition="柳寡婦 存活")])
        return event, "pydantic"

    def repair_scene(self, event, problems, models, verbose):
        self.repair_calls += 1
        if not self.repair_fixes:
            return None, None, None
        fixed = event.model_copy(update={"triggers": []})
        return fixed, "pydantic", None

    def as_stage_runners(self) -> StageRunners:
        return StageRunners(
            extract=self.extract,
            expand_beats=self.expand_beats,
            write_scene=self.write_scene,
            repair_scene=self.repair_scene,
        )


def _advance_to_scene_b(run_id: str, runners: ConflictingRunners) -> None:
    """Drive dispatch_next() through extract -> beats -> beat-a's scene,
    leaving only beat-b's scene (the conflicting one) to be dispatched by
    the caller."""
    dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # extract
    dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beats
    dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beat-a
    assert detect_stage(run_id) == "scenes"


def test_off_mode_never_builds_graph_or_reports_problems() -> None:
    with _isolated_state_dir() as tmp, _isolated_causal_mode("off"):
        run_id = "run-off"
        runners = ConflictingRunners()
        _advance_to_scene_b(run_id, runners)
        dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beat-b
        assert detect_stage(run_id) == "proofread"
        assert not (Path(tmp) / run_id / "causal_graph.json").exists()


def test_warn_mode_checkpoints_scene_and_reports_problem() -> None:
    with _isolated_state_dir(), _isolated_causal_mode("warn"):
        run_id = "run-warn"
        runners = ConflictingRunners()
        _advance_to_scene_b(run_id, runners)

        problems_seen: list[str] = []

        def on_causal(beat_id, problems, attempts):
            problems_seen.extend(problems)

        dispatch_next(
            run_id, REQUIREMENT, runners=runners.as_stage_runners(), on_causal=on_causal
        )
        assert detect_stage(run_id) == "proofread"  # scene still checkpointed
        assert problems_seen  # but the conflict was reported
        assert runners.repair_calls == 0  # warn never repairs


def test_strict_mode_blocks_checkpoint_and_raises() -> None:
    with _isolated_state_dir(), _isolated_causal_mode("strict"):
        run_id = "run-strict"
        runners = ConflictingRunners()
        _advance_to_scene_b(run_id, runners)

        try:
            dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())
            raise AssertionError("expected PipelineError")
        except PipelineError:
            pass

        # beat-b's scene was never written; beat-a's earlier progress survives.
        assert orchestrator.load_checkpoint(
            orchestrator._scene_path(run_id, "beat-b"), Event
        ) is None
        assert orchestrator.load_checkpoint(
            orchestrator._scene_path(run_id, "beat-a"), Event
        ) is not None
        assert detect_stage(run_id) == "scenes"


def test_repair_mode_uses_fix_and_clears_problems() -> None:
    with _isolated_state_dir(), _isolated_causal_mode("repair"):
        run_id = "run-repair"
        runners = ConflictingRunners(repair_fixes=True)
        _advance_to_scene_b(run_id, runners)

        problems_seen: list[list[str]] = []
        attempts_seen: list[int] = []

        def on_causal(beat_id, problems, attempts):
            problems_seen.append(problems)
            attempts_seen.append(attempts)

        dispatch_next(
            run_id, REQUIREMENT, runners=runners.as_stage_runners(), on_causal=on_causal
        )
        assert detect_stage(run_id) == "proofread"
        assert problems_seen[-1] == []  # repaired down to zero problems
        assert attempts_seen[-1] == 1
        assert runners.repair_calls == 1


def test_repair_mode_default_runner_degrades_to_warn_under_fake_backend() -> None:
    """Without an injected repair_scene stub, the default runner
    (orchestrator._default_repair_scene) builds a real proofreader agent.
    Under LLM_BACKEND=fake, FakeLLM's proofreader branch returns a Script,
    not an Event, so _coerce_model always fails and repair_scene returns
    None -- exactly like _repair()'s existing "None means skip" contract.
    The scene should still end up checkpointed with its problems intact
    (mode="repair" falls back to "warn" behavior when repair can't help),
    not silently lost."""
    original_backend = config.LLM_BACKEND
    config.LLM_BACKEND = "fake"
    try:
        with _isolated_state_dir(), _isolated_causal_mode("repair"):
            run_id = "run-repair-fake"
            runners = ConflictingRunners()  # no repair_scene stub -> uses the default

            class _NoRepairStub(ConflictingRunners):
                def as_stage_runners(self) -> StageRunners:
                    return StageRunners(
                        extract=self.extract,
                        expand_beats=self.expand_beats,
                        write_scene=self.write_scene,
                    )

            runners = _NoRepairStub()
            _advance_to_scene_b(run_id, runners)
            dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())
            assert detect_stage(run_id) == "proofread"
            assert orchestrator.load_checkpoint(
                orchestrator._scene_path(run_id, "beat-b"), Event
            ) is not None
    finally:
        config.LLM_BACKEND = original_backend


def test_dispatch_batch_strict_failure_does_not_block_siblings() -> None:
    """A three-beat batch where beat-c is independent of the conflicting
    beat-a/beat-b pair: beat-c must still get checkpointed even though
    beat-b's strict failure raises."""

    class BatchConflictingRunners(ConflictingRunners):
        def __init__(self, repair_fixes: bool = False):
            super().__init__(repair_fixes=repair_fixes)
            self.beats = [_beat("beat-a"), _beat("beat-b", deps=["beat-a"]), _beat("beat-c")]

        def write_scene(self, beat, extraction, models, verbose, target_event_id, *, session=None):
            if beat.id == "beat-c":
                return _event_for(beat), "pydantic"
            return super().write_scene(
                beat, extraction, models, verbose, target_event_id, session=session
            )

    with _isolated_state_dir(), _isolated_causal_mode("strict"):
        run_id = "run-batch-strict"
        runners = BatchConflictingRunners()

        dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # extract
        dispatch_next(run_id, REQUIREMENT, runners=runners.as_stage_runners())  # beats
        # beat-a and beat-c are both in the first batch (only beat-b depends
        # on beat-a); dispatch that batch fully before beat-b's turn.
        dispatch_batch(run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=2)
        assert orchestrator.load_checkpoint(
            orchestrator._scene_path(run_id, "beat-c"), Event
        ) is not None
        assert orchestrator.load_checkpoint(
            orchestrator._scene_path(run_id, "beat-a"), Event
        ) is not None

        try:
            dispatch_batch(
                run_id, REQUIREMENT, runners=runners.as_stage_runners(), concurrency=2
            )
            raise AssertionError("expected PipelineError")
        except PipelineError:
            pass

        assert orchestrator.load_checkpoint(
            orchestrator._scene_path(run_id, "beat-b"), Event
        ) is None


def test_confirm_batch_refreshes_graph_from_promoted_scenes() -> None:
    with _isolated_state_dir() as tmp, _isolated_causal_mode("warn"):
        run_id = "run-confirm-graph"
        runners = ConflictingRunners()
        gate_calls: list[list[str]] = []

        def gate(pending_ids: list[str]) -> bool:
            gate_calls.append(pending_ids)
            return True

        script, report = run_layered(
            REQUIREMENT,
            run_id=run_id,
            runners=runners.as_stage_runners(),
            verbose=False,
            gate=gate,
            concurrency=2,
        )
        assert gate_calls  # the gate was consulted at least once
        assert (Path(tmp) / run_id / "causal_graph.json").exists()
        assert report.mode == "layered"
        assert report.causal_validation == "warn"
        # The A-kills-柳寡婦/B-assumes-alive conflict should have surfaced
        # somewhere in the run's accumulated causal_problems.
        assert any("柳寡婦" in p for p in report.causal_problems)


if __name__ == "__main__":
    test_parse_fact_recognizes_predicate_after_time_marker()
    test_parse_fact_recognizes_predicate_with_no_marker()
    test_parse_fact_recognizes_negation_prefix()
    test_parse_fact_recognizes_relation_operator()
    test_parse_fact_negation_operator_sets_negated()
    test_parse_fact_unrecognized_text_returns_none()
    test_facts_conflict_same_predicate_opposite_polarity()
    test_facts_conflict_antonym_pair_same_polarity()
    test_facts_conflict_different_subject_never_conflicts()
    test_facts_conflict_unrelated_predicates_do_not_conflict()
    test_event_to_node_pulls_from_triggers_and_branches()
    test_build_graph_edges_from_causal_deps_and_branches()
    test_build_graph_skips_deps_and_branches_pointing_at_uncommitted_events()
    test_check_scene_consistency_catches_ancestor_postcondition_conflict()
    test_check_scene_consistency_unrelated_subject_is_not_flagged()
    test_check_scene_consistency_no_ancestor_deps_is_never_flagged()
    test_off_mode_never_builds_graph_or_reports_problems()
    test_warn_mode_checkpoints_scene_and_reports_problem()
    test_strict_mode_blocks_checkpoint_and_raises()
    test_repair_mode_uses_fix_and_clears_problems()
    test_repair_mode_default_runner_degrades_to_warn_under_fake_backend()
    test_dispatch_batch_strict_failure_does_not_block_siblings()
    test_confirm_batch_refreshes_graph_from_promoted_scenes()
    print("All tests passed.")
