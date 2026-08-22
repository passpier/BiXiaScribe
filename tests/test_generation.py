"""Unit tests for bixiascribe.generation -- run entirely offline against
LLM_BACKEND=fake (crew/pipeline.py's FakeLLM), no API key, no network, no
Chroma index, and (checked mechanically below) no streamlit import.

LLM_BACKEND is read from the environment at import time (see config.py), so
it's set both as an env var *and* patched onto the already-imported config
module below -- belt and braces, since an earlier-sorting test module may
have imported bixiascribe.config first.

Deliberately does not touch the real (gitignored) out/ directory -- every
test that writes files uses tempfile.TemporaryDirectory().
"""
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["LLM_BACKEND"] = "fake"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import config  # noqa: E402

config.LLM_BACKEND = "fake"

from bixiascribe import generation, review  # noqa: E402
from bixiascribe.crew import scene_metrics  # noqa: E402
from bixiascribe.crew.pipeline import StepEvent, run_pipeline_with_report  # noqa: E402

REAL_VARIANTS_FILE = Path(__file__).resolve().parents[1] / "eval" / "model_variants.json"


@contextmanager
def _isolated_state_dir():
    """Phase 4b's layered-mode tests write real .bixia_state/<run_id>/
    checkpoints -- point config.BIXIA_STATE_DIR at a throwaway tempdir for
    the duration of one test, same pattern as
    tests/test_orchestrator.py/test_orchestrator_parallel.py."""
    original = config.BIXIA_STATE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.BIXIA_STATE_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            config.BIXIA_STATE_DIR = original


def test_generation_module_does_not_import_streamlit():
    # generation.py joins review.py under the same no-streamlit rule (see
    # CONTRIBUTING.md) -- checked both by source scan and by what's actually
    # been imported, so a transitive import would also trip this.
    source = Path(generation.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "streamlit" not in sys.modules


def test_load_variants_parses_the_real_variants_file():
    variants = generation.load_variants(REAL_VARIANTS_FILE)
    names = {v.name for v in variants}
    assert "baseline" in names
    for v in variants:
        assert v.writer and v.dialogue and v.proof


def test_ui_variant_names_are_parseable_by_review():
    variants = generation.load_variants(REAL_VARIANTS_FILE)
    for v in variants:
        prefixed = generation.ui_variant_name(v.name)
        assert "__" not in prefixed
        name = f"{prefixed}__req-abc123__rep1.json"
        parsed_variant, parsed_slug, parsed_rep = review.parse_script_filename(name)
        assert parsed_variant == prefixed
        assert parsed_slug == "req-abc123"
        assert parsed_rep == 1


def test_preflight_flags_fake_backend():
    problems = generation.preflight(check_index=False)
    assert any("LLM_BACKEND=fake" in p for p in problems)


def test_generate_writes_script_and_run_row():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "測試需求",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-test",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
        )
        assert result.ok
        assert result.rep == 0
        assert result.script_path is not None
        assert result.script_path.name.startswith("ui-test__")
        assert result.script_path.is_file()

        row = result.row
        expected_keys = set(result.report.to_dict()) | {
            "variant",
            "ts",
            "ok",
            "error",
            "script_path",
            "events",
            "npcs",
        }
        assert expected_keys <= set(row)
        assert row["error"] is None
        assert row["variant"] == "ui-test"


def test_generate_second_run_gets_rep_suffix():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        variant = generation.Variant(
            name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"
        )

        first = generation.generate(
            "同一個需求", variant, variant_name="ui-rep", rep=None,
            scripts_dir=scripts_dir, jsonl_path=None,
        )
        second = generation.generate(
            "同一個需求", variant, variant_name="ui-rep", rep=None,
            scripts_dir=scripts_dir, jsonl_path=None,
        )
        assert first.rep == 0
        assert second.rep == 1
        assert first.script_path.is_file()
        assert second.script_path.is_file()
        assert first.script_path != second.script_path


def test_generate_with_no_jsonl_path_writes_no_log():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        generation.generate(
            "免紀錄需求",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-nolog",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
        )
        assert not list(Path(tmp).glob("*.jsonl"))


def test_generated_artifacts_are_discoverable_by_review():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        scripts_dir = out_dir / "eval"
        jsonl_path = out_dir / "generation_runs_ui.jsonl"

        generation.generate(
            "可被發現的需求",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-discover",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=jsonl_path,
        )

        records = review.discover_scripts(
            scripts_dir=scripts_dir,
            out_dir=out_dir,
            requirements_file=out_dir / "nope.txt",
            state_dir=out_dir / "no_such_state",
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.source == "jsonl"
        assert rec.variant == "ui-discover"
        assert rec.requirement == "可被發現的需求"

        rows = review.overview_rows(records)
        assert rows[0]["events"] > 0


def test_run_pipeline_with_report_emits_progress_events():
    events = []
    run_pipeline_with_report("進度事件測試", verbose=False, on_step=events.append)
    assert len(events) >= 3
    assert {e.kind for e in events} >= {"task"}


def test_job_runs_to_completion():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        job = generation.GenerationJob(
            "背景工作測試",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            scripts_dir=scripts_dir,
            jsonl_path=None,
        )
        job.start()
        snap = job.join(timeout=60)
        assert snap.status == "done"
        assert snap.phase_index == 3
        assert snap.elapsed_s > 0
        assert snap.result is not None
        assert snap.result.ok
        assert generation.is_running() is False


def test_on_step_phase_event_updates_phase_in_layered_mode():
    """Regression test for the original complaint: 拆書/排場/寫戲 used to
    keep showing the *previous* stage's label for their entire duration
    because only "task" (stage-end) events updated the displayed phase."""
    with tempfile.TemporaryDirectory() as tmp:
        job = generation.GenerationJob(
            "階段事件測試",
            generation.Variant(
                name="test", extractor="fake/e", beat_expander="fake/b",
                scene_writer="fake/s",
            ),
            scripts_dir=Path(tmp) / "eval",
            jsonl_path=None,
            pipeline_mode="layered",
        )
        job._on_step(StepEvent(kind="phase", role="拆書", text="開始執行"))
        snap = job.snapshot()
        assert snap.phase == "拆書"
        assert snap.stage == "拆書"
        assert snap.phase_index == 0

        job._on_step(StepEvent(kind="task", role="拆書", text="任務完成"))
        assert job.snapshot().phase_index == 1


def test_on_step_phase_event_ignored_in_legacy_mode():
    with tempfile.TemporaryDirectory() as tmp:
        job = generation.GenerationJob(
            "legacy 階段事件測試",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            scripts_dir=Path(tmp) / "eval",
            jsonl_path=None,
            pipeline_mode="legacy",
        )
        job._on_step(StepEvent(kind="phase", role="不該顯示", text="x"))
        snap = job.snapshot()
        assert snap.phase == ""
        assert snap.phase_index == 0

        job._on_step(StepEvent(kind="task", role="", text="x"))
        snap = job.snapshot()
        assert snap.phase == "編劇・鐵筆生"
        assert snap.phase_index == 1


def test_job_snapshot_new_fields_default_to_legacy_safe_values():
    snap = generation.JobSnapshot()
    assert snap.stage == ""
    assert snap.scenes_done == 0
    assert snap.scenes_total == 0
    assert snap.active_scene_ids == ()
    assert snap.active_scene_elapsed_s == 0.0
    assert snap.guardrail_retries == 0


def test_snapshot_reports_live_scene_progress():
    scene_metrics.reset_stats()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            job = generation.GenerationJob(
                "場次進度測試",
                generation.Variant(
                    name="test", extractor="fake/e", beat_expander="fake/b",
                    scene_writer="fake/s",
                ),
                scripts_dir=Path(tmp) / "eval",
                jsonl_path=None,
                pipeline_mode="layered",
            )
            with scene_metrics.scene_scope("b1"):
                snap = job.snapshot()
                assert snap.active_scene_ids == ("b1",)
                assert snap.active_scene_elapsed_s >= 0.0
            snap = job.snapshot()
            assert snap.active_scene_ids == ()
            assert snap.scenes_done == 1
    finally:
        scene_metrics.reset_stats()


def test_snapshot_counts_guardrail_retries():
    scene_metrics.reset_stats()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            job = generation.GenerationJob(
                "guardrail 重試計數測試",
                generation.Variant(
                    name="test", extractor="fake/e", beat_expander="fake/b",
                    scene_writer="fake/s",
                ),
                scripts_dir=Path(tmp) / "eval",
                jsonl_path=None,
                pipeline_mode="layered",
            )
            with scene_metrics.scene_scope("b1"):
                scene_metrics.record_guardrail_retry("b1")
            assert job.snapshot().guardrail_retries == 1
    finally:
        scene_metrics.reset_stats()


def test_scenes_total_read_is_cached():
    """Regression guard for the 1-second-poll pressure _scenes_total()
    exists to avoid -- once a beat sheet is found, load_beat_sheet() must
    not be called again for the rest of the run."""
    from bixiascribe.crew.orchestrator import save_checkpoint, state_dir
    from bixiascribe.schema import Beat, BeatSheet, Outline

    with _isolated_state_dir(), tempfile.TemporaryDirectory() as tmp:
        job = generation.GenerationJob(
            "scenes_total 快取測試",
            generation.Variant(
                name="test", extractor="fake/e", beat_expander="fake/b",
                scene_writer="fake/s",
            ),
            scripts_dir=Path(tmp) / "eval",
            jsonl_path=None,
            pipeline_mode="layered",
        )
        run_id = job._run_id
        state_dir(run_id).mkdir(parents=True, exist_ok=True)
        sheet = BeatSheet(
            outline=Outline(title="t", chapters=[]),
            beats=[Beat(id="b1", chapter_id="c1", summary="s")],
        )
        save_checkpoint(state_dir(run_id) / "beats.json", sheet)

        calls = []
        real_load_beat_sheet = generation.load_beat_sheet

        def _counting_load_beat_sheet(rid):
            calls.append(rid)
            return real_load_beat_sheet(rid)

        generation.load_beat_sheet = _counting_load_beat_sheet
        try:
            for _ in range(5):
                job.snapshot()
        finally:
            generation.load_beat_sheet = real_load_beat_sheet
        assert len(calls) == 1


def test_job_start_is_rejected_while_another_run_holds_the_lock():
    acquired = generation._run_lock.acquire(blocking=False)
    assert acquired
    try:
        try:
            generation.generate("被鎖住的需求", generation.Variant(name="t"))
            raise AssertionError("expected GenerationBusyError")
        except generation.GenerationBusyError:
            pass
    finally:
        generation._run_lock.release()


def test_cancel_before_start_reports_cancelled():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        job = generation.GenerationJob(
            "取消測試",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            scripts_dir=scripts_dir,
            jsonl_path=None,
        )
        job.cancel()
        job.start()
        snap = job.join(timeout=60)
        assert snap.status == "cancelled"
        assert not list(scripts_dir.glob("*.json")) if scripts_dir.is_dir() else True


def test_generate_layered_mode_writes_script_and_checkpoints():
    with _isolated_state_dir(), tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "分層測試需求",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-layered",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            pipeline_mode="layered",
        )
        assert result.ok
        assert result.report.mode == "layered"
        assert result.run_id  # non-empty only for layered runs
        assert result.script_path.is_file()
        assert (config.BIXIA_STATE_DIR / result.run_id / "script.json").is_file()


def test_generate_legacy_mode_leaves_run_id_empty():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "傳統管線需求",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-legacy",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            pipeline_mode="legacy",
        )
        assert result.ok
        assert result.report.mode == "legacy"
        assert result.run_id == ""


def _wait_for_gate(
    job: "generation.GenerationJob", timeout: float = 30
) -> "generation.JobSnapshot":
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = job.snapshot()
        if snap.awaiting_confirmation or snap.status != "running":
            return snap
        time.sleep(0.02)
    raise AssertionError("timed out waiting for confirmation gate")


def test_layered_job_stages_a_batch_and_waits_for_confirmation():
    with _isolated_state_dir(), tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        job = generation.GenerationJob(
            "確認閘門測試",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            scripts_dir=scripts_dir,
            jsonl_path=None,
            pipeline_mode="layered",
        )
        job.start()

        snap = _wait_for_gate(job)
        assert snap.awaiting_confirmation
        assert snap.pending_scene_ids  # first batch: beat_depart

        # FakeLLM's beat sheet is a 3-beat causal chain (beat_depart ->
        # beat_village -> beat_clue), so each is its own batch -- confirm
        # each one as it's offered until the run finishes.
        confirmed_batches: list[tuple[str, ...]] = []
        while True:
            snap = job.snapshot()
            if snap.status != "running":
                break
            if snap.awaiting_confirmation:
                confirmed_batches.append(snap.pending_scene_ids)
                job.confirm_batch()
            time.sleep(0.02)

        final = job.join(timeout=30)
        assert final.status == "done"
        assert final.result is not None
        assert final.result.ok
        assert not final.awaiting_confirmation
        assert confirmed_batches == [("beat_depart",), ("beat_village",), ("beat_clue",)]


def test_layered_job_reject_batch_regenerates_it():
    with _isolated_state_dir(), tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        job = generation.GenerationJob(
            "拒絕重生測試",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            scripts_dir=scripts_dir,
            jsonl_path=None,
            pipeline_mode="layered",
        )
        job.start()

        first = _wait_for_gate(job)
        assert first.awaiting_confirmation
        assert first.pending_scene_ids == ("beat_depart",)
        job.reject_batch()

        second = _wait_for_gate(job)
        assert second.awaiting_confirmation
        assert second.pending_scene_ids == ("beat_depart",)  # re-offered, not skipped
        job.confirm_batch()

        # Confirm the remaining batches to let the run finish.
        while True:
            snap = job.snapshot()
            if snap.status != "running":
                break
            if snap.awaiting_confirmation:
                job.confirm_batch()
            time.sleep(0.02)

        final = job.join(timeout=30)
        assert final.status == "done"
        assert final.result.ok


def test_cancel_wins_while_awaiting_confirmation():
    with _isolated_state_dir(), tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        jsonl_path = Path(tmp) / "runs.jsonl"
        job = generation.GenerationJob(
            "閘門取消測試",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            scripts_dir=scripts_dir,
            jsonl_path=jsonl_path,
            pipeline_mode="layered",
        )
        job.start()

        snap = _wait_for_gate(job)
        assert snap.awaiting_confirmation

        job.cancel()
        snap = job.join(timeout=30)
        assert snap.status == "cancelled"

        # A run cancelled at the batch-confirmation gate must still leave a
        # priceable row behind for whatever the first scene already spent --
        # see generation.generate()'s GenerationCancelled handler and
        # orchestrator.run_layered()'s gate()-call guard. Before that fix,
        # this row was never written at all.
        assert snap.result is not None
        assert not snap.result.ok
        assert snap.result.row
        assert jsonl_path.is_file()
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["ok"] is False
        assert rows[0]["mode"] == "layered"
        assert "cost_usd" in rows[0]
        assert "cost_basis" in rows[0]


# --- Phase 5: session_doc_max_tokens threading -----------------------------


def test_variant_round_trips_session_doc_max_tokens():
    variant = generation.Variant.from_dict(
        {"name": "v", "writer": "w", "dialogue": "d", "proof": "p", "session_doc_max_tokens": 0}
    )
    assert variant.session_doc_max_tokens == 0

    variant_default = generation.Variant.from_dict(
        {"name": "v", "writer": "w", "dialogue": "d", "proof": "p"}
    )
    assert variant_default.session_doc_max_tokens is None


def test_layered_generate_accepts_session_doc_max_tokens():
    with _isolated_state_dir(), tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "壓縮測試需求",
            generation.Variant(
                name="test",
                writer="fake/w",
                dialogue="fake/d",
                proof="fake/p",
                session_doc_max_tokens=0,
            ),
            variant_name="ui-ctx",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            pipeline_mode="layered",
        )
        assert result.ok
        assert result.report.session_doc_max_tokens == 0
        # build_run_row() merges script_metrics() (5 continuity keys) into
        # the row, and RunReport.to_dict() carries session_doc_omitted_total.
        for key in (
            "distinct_event_title_pct",
            "repeated_dialogue_pct",
            "prior_entity_reference_pct",
            "connected_event_pct",
            "self_loop_branch_pct",
        ):
            assert key in result.row
        assert "token_usage" in result.row
        assert "session_doc_omitted_total" in result.row


# --- Six-role model split (layered-mode fallback fix) ----------------------


def test_to_model_choice_fills_all_six_roles_from_writer_dialogue():
    # Before this fix, layered mode's extractor/beat_expander/scene_writer
    # silently ignored a variant's models entirely (see generation.Variant's
    # docstring) -- this is the regression test for that fix: writer/
    # dialogue must propagate to the three layered-only roles when a variant
    # doesn't set them explicitly.
    variant = generation.Variant(name="v", writer="fake/w", dialogue="fake/d", proof="fake/p")
    models = variant.to_model_choice()
    assert models.extractor == "fake/w"
    assert models.beat_expander == "fake/w"
    assert models.scene_writer == "fake/d"


def test_to_model_choice_honors_explicit_layered_roles():
    variant = generation.Variant(
        name="v",
        writer="fake/w",
        dialogue="fake/d",
        proof="fake/p",
        extractor="fake/x",
        beat_expander="fake/b",
        scene_writer="fake/s",
    )
    models = variant.to_model_choice()
    assert models.extractor == "fake/x"
    assert models.beat_expander == "fake/b"
    assert models.scene_writer == "fake/s"


def test_layered_generate_uses_variants_scene_writer_model():
    # End-to-end: a layered run's RunReport.model_scene_writer must reflect
    # the variant's dialogue model, not config.LLM_MODEL_DIALOGUE.
    with _isolated_state_dir(), tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "分層模型測試需求",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-model-fix",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            pipeline_mode="layered",
        )
        assert result.ok
        assert result.report.model_scene_writer == "fake/d"
        assert result.report.model_extractor == "fake/w"
        assert result.report.model_beat_expander == "fake/w"


# --- no-retired-key regression guard -----------------------------------------


def test_real_variants_file_has_no_retired_key():
    # eval/model_variants.json used to carry retired: true entries as
    # long-form evidence for why a model was dropped -- removed 2026-08-17
    # since stale model judgments rot fast and only cost context on every
    # read. Don't let that pattern creep back in.
    rows = json.loads(REAL_VARIANTS_FILE.read_text(encoding="utf-8"))
    assert all("retired" not in row for row in rows)


# --- script_length threading -------------------------------------------------


def test_variant_round_trips_script_length():
    variant = generation.Variant.from_dict(
        {"name": "v", "writer": "w", "dialogue": "d", "proof": "p", "script_length": "long"}
    )
    assert variant.script_length == "long"

    variant_default = generation.Variant.from_dict(
        {"name": "v", "writer": "w", "dialogue": "d", "proof": "p"}
    )
    assert variant_default.script_length is None


def test_variant_round_trips_custom_script_length_string():
    variant = generation.Variant.from_dict(
        {
            "name": "v",
            "writer": "w",
            "dialogue": "d",
            "proof": "p",
            "script_length": "custom:events=20,chapters=4",
        }
    )
    assert variant.script_length == "custom:events=20,chapters=4"


def test_generate_resolves_script_length_explicit_over_variant_over_config():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        variant = generation.Variant(
            name="test", writer="fake/w", dialogue="fake/d", proof="fake/p", script_length="medium"
        )
        # explicit script_length= wins over variant.script_length
        result = generation.generate(
            "篇幅測試需求",
            variant,
            variant_name="ui-length",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            script_length="long",
        )
        assert result.ok
        assert result.report.script_length == "long"

        # no explicit arg: falls back to variant.script_length
        result2 = generation.generate(
            "篇幅測試需求2",
            variant,
            variant_name="ui-length2",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
        )
        assert result2.report.script_length == "medium"


def test_generate_records_canonicalized_custom_script_length():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        variant = generation.Variant(
            name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"
        )
        result = generation.generate(
            "篇幅測試需求3",
            variant,
            variant_name="ui-length3",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            script_length="custom:events=20",
        )
        assert result.ok
        assert result.report.script_length == (
            "custom:events=20,chapters=4,beats_per_chapter=5,min_dialogue=三段以上"
        )


# --- cost accounting (pricing.py wiring) ------------------------------------


def test_build_run_row_always_carries_cost_keys():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "成本測試需求",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-cost",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
        )
        assert result.ok
        assert "cost_usd" in result.row
        assert "cost_basis" in result.row
        # "fake/w" etc. have no eval/model_prices.json entry, so this must
        # degrade to unknown_price/None -- never a fabricated number.
        assert result.row["cost_usd"] is None
        assert result.row["cost_basis"] == "unknown_price"


# --- run-resume (GenerationJob(run_id=...)) --------------------------------


def test_generation_job_run_id_resumes_existing_checkpoint():
    with _isolated_state_dir() as state_dir, tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        variant = generation.Variant(
            name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"
        )

        job1 = generation.GenerationJob(
            "續跑測試需求", variant, scripts_dir=scripts_dir, jsonl_path=None,
            pipeline_mode="layered",
        )
        job1.start()
        snap = _wait_for_gate(job1)
        assert snap.awaiting_confirmation
        run_id = snap.run_id
        job1.confirm_batch()
        # Wait for either the next gate (another batch staged) or the run
        # finishing on its own, then cancel to leave a partial checkpoint.
        _wait_for_gate(job1)
        job1.cancel()
        job1.join(timeout=30)

        run_dir = state_dir / run_id
        assert run_dir.is_dir()
        scene_files_before = sorted(p.name for p in run_dir.glob("scene_*.json"))
        assert scene_files_before  # the confirmed batch committed at least one scene
        dirs_before = {p.name for p in state_dir.iterdir()}

        job2 = generation.GenerationJob(
            "續跑測試需求", variant, scripts_dir=scripts_dir, jsonl_path=None,
            pipeline_mode="layered", run_id=run_id,
        )
        assert job2.snapshot().run_id == run_id
        job2.start()
        while True:
            s = job2.snapshot()
            if s.status != "running":
                break
            if s.awaiting_confirmation:
                job2.confirm_batch()
            time.sleep(0.02)
        final = job2.join(timeout=30)
        assert final.status == "done"

        dirs_after = {p.name for p in state_dir.iterdir()}
        assert dirs_before == dirs_after  # resumed the same dir, minted no second one
        for name in scene_files_before:
            assert (run_dir / name).exists()


# --- reasoning_effort -------------------------------------------------------


def test_variant_round_trips_reasoning_effort():
    row = {"name": "v", "reasoning_effort": "high"}
    variant = generation.Variant.from_dict(row)
    assert variant.reasoning_effort == "high"
    assert variant.to_model_choice().reasoning_effort == "high"


def test_variant_reasoning_effort_falls_back_to_config_default():
    variant = generation.Variant(name="v")
    assert variant.to_model_choice().reasoning_effort == config.REASONING_EFFORT


def test_generate_reasoning_effort_explicit_arg_wins_over_variant():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "推理強度測試",
            generation.Variant(
                name="test", writer="fake/w", dialogue="fake/d", proof="fake/p",
                reasoning_effort="low",
            ),
            variant_name="ui-reasoning",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            reasoning_effort="high",
        )
        assert result.ok
        assert result.report.reasoning_effort == "high"


def test_generate_reasoning_effort_variant_wins_over_config_default():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "推理強度測試2",
            generation.Variant(
                name="test", writer="fake/w", dialogue="fake/d", proof="fake/p",
                reasoning_effort="medium",
            ),
            variant_name="ui-reasoning2",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
        )
        assert result.ok
        assert result.report.reasoning_effort == "medium"


def test_generate_run_row_carries_reasoning_effort():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        result = generation.generate(
            "推理強度測試3",
            generation.Variant(name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"),
            variant_name="ui-reasoning3",
            rep=0,
            scripts_dir=scripts_dir,
            jsonl_path=None,
            reasoning_effort="none",
        )
        assert result.ok
        assert result.row["reasoning_effort"] == "none"


# --- _cost_models() dedup against review.role_keys_for_mode() ---------------


def test_cost_models_key_set_matches_role_keys_for_mode():
    for pipeline_mode in ("legacy", "layered"):
        with tempfile.TemporaryDirectory() as tmp:
            scripts_dir = Path(tmp) / "eval"
            ctx = _isolated_state_dir() if pipeline_mode == "layered" else _null_context()
            with ctx:
                result = generation.generate(
                    "cost_models 測試",
                    generation.Variant(
                        name="test", writer="fake/w", dialogue="fake/d", proof="fake/p"
                    ),
                    variant_name=f"ui-costmodels-{pipeline_mode}",
                    rep=0,
                    scripts_dir=scripts_dir,
                    jsonl_path=None,
                    pipeline_mode=pipeline_mode,
                )
            assert result.ok
            assert set(generation._cost_models(result.report)) == set(
                review.role_keys_for_mode(pipeline_mode)
            )


@contextmanager
def _null_context():
    yield


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
