"""Unit tests for bixiascribe.review -- pure functions over a tmp dir of
synthetic fixtures. No API key, no network, no external deps, and (checked
mechanically below) no streamlit import -- this module backs the Stage 3
review UI but must stay swappable to a different frontend later.

Deliberately does not touch the real (gitignored) out/ directory -- a clean
checkout must be able to run this file with no eval artifacts present.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import review  # noqa: E402


def _script_json(title: str = "測試劇本", n_events: int = 1) -> dict:
    return {
        "meta": {"title": title, "theme": "theme"},
        "npcs": [
            {
                "id": "npc1",
                "name": "張三",
                "personality": "剛直",
                "speech_style": "直率",
            }
        ],
        "events": [
            {
                "id": f"event{i}",
                "title": f"事件{i}",
                "summary": "summary",
                "preconditions": [],
                "dialogue": [{"npc": "npc1", "line": "在下張三"}],
                "choices": [],
            }
            for i in range(n_events)
        ],
    }


def _write_run_row(**overrides) -> dict:
    row = {
        "variant": "baseline",
        "ts": 100.0,
        "ok": True,
        "error": None,
        "requirement": "少林弟子下山查一樁滅門案",
        "model_writer": "m1",
        "model_dialogue": "m2",
        "model_proof": "m3",
        "elapsed_s": 1.5,
        "token_usage": {"total_tokens": 100},
        "retrieval_calls": 1,
        "retrieval_failures": 0,
        "retrieval_queries": ["query1"],
        "repair_attempts": 0,
        "coerced_from": "pydantic",
    }
    row.update(overrides)
    return row


# ---- requirement_slug / parse_script_filename ----


def test_requirement_slug_matches_checked_in_filenames():
    # Pinned against the actual filenames in out/eval/ as of Phase C, so
    # moving _slug out of scripts/eval_generation.py is provably a
    # behavior-preserving refactor.
    assert review.requirement_slug("少林弟子下山查一樁滅門案") == "req-30f25bc507"
    assert review.requirement_slug(
        "丐幫弟子護送一份密函上武當，途中遭黑道劫殺，須查出洩密者"
    ) == "req-d5cacd93f4"


def test_requirement_slug_ascii_path_and_truncation():
    assert review.requirement_slug("hello world!") == "helloworld"
    long_text = "a" * 40
    assert review.requirement_slug(long_text, max_len=10) == "a" * 10


def test_parse_script_filename_plain_and_rep():
    assert review.parse_script_filename("baseline__req-abc123.json") == (
        "baseline",
        "req-abc123",
        0,
    )
    assert review.parse_script_filename("baseline__req-abc123__rep1.json") == (
        "baseline",
        "req-abc123",
        1,
    )


def test_parse_script_filename_tolerates_underscores_in_slug():
    variant, slug, rep = review.parse_script_filename("a__b__req_x__rep2.json")
    assert variant == "a"
    assert slug == "b__req_x"
    assert rep == 2


def test_parse_script_filename_rejects_nameless_variant():
    try:
        review.parse_script_filename("noseparator.json")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---- load_jsonl ----


def test_load_jsonl_skips_blank_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n   \n', encoding="utf-8")
        rows = review.load_jsonl(path)
        assert rows == [{"a": 1}, {"a": 2}]


# ---- load_run_records / latest_run_by_path ----


def test_load_run_records_merges_multiple_logs_and_tags_source():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        (out_dir / "generation_runs.jsonl").write_text(
            json.dumps(_write_run_row(script_path="/x/a.json")) + "\n", encoding="utf-8"
        )
        (out_dir / "generation_runs_2.jsonl").write_text(
            json.dumps(_write_run_row(script_path="/x/b.json", ts=200.0)) + "\n",
            encoding="utf-8",
        )
        runs = review.load_run_records(out_dir)
        assert len(runs) == 2
        assert {r.source_log for r in runs} == {"generation_runs.jsonl", "generation_runs_2.jsonl"}


def test_run_record_from_partial_row_uses_empty_defaults():
    row = {"variant": "cheap-ends", "ts": 50.0, "ok": False, "error": "boom"}
    run = review.RunRecord.from_row(row, source_log="log.jsonl")
    assert run.variant == "cheap-ends"
    assert run.ok is False
    assert run.error == "boom"
    assert run.script_path == ""
    assert run.token_usage == {}
    assert run.retrieval_queries == ()
    assert run.coerced_from == ""
    assert run.scene_metrics == ()


def test_latest_run_by_path_keeps_newest_ts():
    older = review.RunRecord.from_row(
        _write_run_row(script_path="/x/a.json", ts=100.0), "log1.jsonl"
    )
    newer = review.RunRecord.from_row(
        _write_run_row(script_path="/x/a.json", ts=200.0), "log2.jsonl"
    )
    best = review.latest_run_by_path([older, newer])
    assert best["/x/a.json"] is newer


# ---- discover_scripts ----


def _fixture(tmp: Path) -> None:
    scripts_dir = tmp / "eval"
    scripts_dir.mkdir()
    out_dir = tmp
    eval_dir = tmp / "eval_reqs"
    eval_dir.mkdir()

    # 1. a script with a matching run row
    matched_path = scripts_dir / "baseline__req-30f25bc507.json"
    matched_path.write_text(json.dumps(_script_json(), ensure_ascii=False), encoding="utf-8")

    # 2. an orphan script with no matching row
    orphan_path = scripts_dir / "prose-split__req-a09fa62fc7.json"
    orphan_path.write_text(
        json.dumps(_script_json(n_events=2), ensure_ascii=False), encoding="utf-8"
    )

    # 3. a rep-suffixed script, also matched
    rep_path = scripts_dir / "baseline__req-30f25bc507__rep1.json"
    rep_path.write_text(json.dumps(_script_json(), ensure_ascii=False), encoding="utf-8")

    requirements_file = eval_dir / "script_requirements.txt"
    requirements_file.write_text(
        "少林弟子下山查一樁滅門案\n"
        "# comment line\n"
        "青樓歌女身懷絕世劍法，為報滅門之仇隻身闖蕩江湖仇殺仇家\n",
        encoding="utf-8",
    )

    rows = [
        _write_run_row(script_path=str(matched_path.resolve()), ts=100.0),
        # an older, duplicate row for the same path -- should be superseded
        _write_run_row(script_path=str(matched_path.resolve()), ts=50.0),
        _write_run_row(
            script_path=str(rep_path.resolve()), ts=150.0, requirement="少林弟子下山查一樁滅門案"
        ),
        # a failed run with no script_path at all
        {
            "variant": "cheap-ends",
            "ts": 300.0,
            "ok": False,
            "error": "structured output parse failure",
            "requirement": "測試需求",
        },
        # a blank line to make sure load_jsonl tolerates it
    ]
    (out_dir / "generation_runs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n",
        encoding="utf-8",
    )


def test_discover_scripts_joins_run_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _fixture(tmp_path)
        records = review.discover_scripts(
            scripts_dir=tmp_path / "eval",
            out_dir=tmp_path,
            requirements_file=tmp_path / "eval_reqs" / "script_requirements.txt",
            state_dir=tmp_path / "no_such_state",
        )
        matched = [r for r in records if r.source == "jsonl"]
        assert len(matched) == 2
        assert all(r.requirement == "少林弟子下山查一樁滅門案" for r in matched)


def test_discover_scripts_falls_back_to_filename_and_requirements_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _fixture(tmp_path)
        records = review.discover_scripts(
            scripts_dir=tmp_path / "eval",
            out_dir=tmp_path,
            requirements_file=tmp_path / "eval_reqs" / "script_requirements.txt",
            state_dir=tmp_path / "no_such_state",
        )
        orphan = [r for r in records if r.source == "filename"]
        assert len(orphan) == 1
        assert orphan[0].variant == "prose-split"
        assert orphan[0].requirement == "青樓歌女身懷絕世劍法，為報滅門之仇隻身闖蕩江湖仇殺仇家"


def test_discover_scripts_surfaces_failed_runs_without_a_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _fixture(tmp_path)
        records = review.discover_scripts(
            scripts_dir=tmp_path / "eval",
            out_dir=tmp_path,
            requirements_file=tmp_path / "eval_reqs" / "script_requirements.txt",
            state_dir=tmp_path / "no_such_state",
        )
        run_only = [r for r in records if r.source == "run-only"]
        assert len(run_only) == 1
        assert run_only[0].path is None
        assert run_only[0].variant == "cheap-ends"
        assert run_only[0].run.error == "structured output parse failure"


def test_discover_scripts_on_missing_dirs_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        records = review.discover_scripts(
            scripts_dir=tmp_path / "nonexistent",
            out_dir=tmp_path / "also_nonexistent",
            requirements_file=tmp_path / "nope.txt",
            state_dir=tmp_path / "no_such_state",
        )
        assert records == []


def test_group_by_requirement_and_find_record():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _fixture(tmp_path)
        records = review.discover_scripts(
            scripts_dir=tmp_path / "eval",
            out_dir=tmp_path,
            requirements_file=tmp_path / "eval_reqs" / "script_requirements.txt",
            state_dir=tmp_path / "no_such_state",
        )
        groups = review.group_by_requirement(records)
        assert "少林弟子下山查一樁滅門案" in groups
        assert len(groups["少林弟子下山查一樁滅門案"]) == 2

        rec = review.find_record(records, "少林弟子下山查一樁滅門案", "baseline", 0)
        assert rec is not None
        assert rec.rep == 0
        rec_rep1 = review.find_record(records, "少林弟子下山查一樁滅門案", "baseline", 1)
        assert rec_rep1 is not None
        assert rec_rep1.rep == 1


def test_npc_names_and_event_titles_resolve_ids():
    script = review.Script.model_validate(_script_json(n_events=2))
    names = review.npc_names(script)
    titles = review.event_titles(script)
    assert names == {"npc1": "張三"}
    assert titles == {"event0": "事件0", "event1": "事件1"}


def test_npc_names_includes_player_when_present():
    data = _script_json(n_events=1)
    data["player"] = {"name": "你"}
    script = review.Script.model_validate(data)
    names = review.npc_names(script)
    assert names["player"] == "你"
    assert names["npc1"] == "張三"


def test_chapter_names_resolves_ids():
    data = _script_json(n_events=1)
    data["chapters"] = [{"id": "ch1", "title": "第一章", "summary": "s"}]
    script = review.Script.model_validate(data)
    assert review.chapter_names(script) == {"ch1": "第一章"}


def test_clue_names_resolves_ids():
    data = _script_json(n_events=1)
    data["clues"] = [{"id": "c1", "name": "血手印"}]
    script = review.Script.model_validate(data)
    assert review.clue_names(script) == {"c1": "血手印"}


def test_faction_names_resolves_ids():
    data = _script_json(n_events=1)
    data["factions"] = [{"id": "f1", "name": "少林派"}]
    script = review.Script.model_validate(data)
    assert review.faction_names(script) == {"f1": "少林派"}


def test_overview_rows_include_gmud_metric_keys():
    script_data = _script_json(n_events=1)
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp)
        path = scripts_dir / "v1__slug.json"
        path.write_text(json.dumps(script_data), encoding="utf-8")
        records = review.discover_scripts(
            scripts_dir=scripts_dir,
            out_dir=Path(tmp) / "no-such-out",
            state_dir=Path(tmp) / "no_such_state",
        )
        rows = review.overview_rows(records)
        assert rows
        for key in (
            "branches_with_cost_pct",
            "branches_with_payoff_pct",
            "checks_with_fallback_pct",
            "events_with_clue_pct",
            "faction_count",
            "ending_count",
        ):
            assert key in rows[0]


def test_overview_rows_recompute_metrics_from_file_not_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        scripts_dir = tmp_path / "eval"
        scripts_dir.mkdir()
        script_path = scripts_dir / "baseline__req-x.json"
        script_path.write_text(
            json.dumps(_script_json(n_events=3), ensure_ascii=False), encoding="utf-8"
        )
        (tmp_path / "generation_runs.jsonl").write_text(
            json.dumps(
                _write_run_row(
                    script_path=str(script_path.resolve()),
                    requirement="測試",
                )
                | {"events": 99}  # stale/wrong count, must be ignored
            )
            + "\n",
            encoding="utf-8",
        )
        records = review.discover_scripts(
            scripts_dir=scripts_dir,
            out_dir=tmp_path,
            requirements_file=tmp_path / "nonexistent.txt",
            state_dir=tmp_path / "no_such_state",
        )
        rows = review.overview_rows(records)
        assert len(rows) == 1
        assert rows[0]["events"] == 3  # recomputed from the file, not the stale JSONL 99


def test_overview_rows_run_only_records_have_continuity_keys():
    # A run-only record (path=None, e.g. a run that crashed before ever
    # writing a script file) must still carry the 5 Phase 5 continuity keys
    # at their 0.0 default, so st.dataframe(overview_rows(...)) never gets
    # ragged columns between file-backed and run-only rows.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _fixture(tmp_path)
        records = review.discover_scripts(
            scripts_dir=tmp_path / "eval",
            out_dir=tmp_path,
            requirements_file=tmp_path / "eval_reqs" / "script_requirements.txt",
            state_dir=tmp_path / "no_such_state",
        )
        run_only = [r for r in records if r.source == "run-only"]
        assert run_only and run_only[0].path is None

        rows = review.overview_rows(run_only)
        assert len(rows) == 1
        for key in (
            "distinct_event_title_pct",
            "repeated_dialogue_pct",
            "prior_entity_reference_pct",
            "connected_event_pct",
            "self_loop_branch_pct",
        ):
            assert rows[0][key] == 0.0


def test_load_script_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "script.json"
        path.write_text(json.dumps(_script_json(), ensure_ascii=False), encoding="utf-8")
        script = review.load_script(path)
        assert script.meta.title == "測試劇本"


def test_load_script_unwraps_checkpoint_envelope():
    # crew/orchestrator.py::save_checkpoint() wraps every checkpointed model
    # (including the final Script) as {"schema_version": N, "data": {...}}.
    # load_script() must transparently unwrap that -- detected structurally,
    # never by importing orchestrator.py (see review.py's module docstring).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "script.json"
        envelope = {"schema_version": 4, "data": _script_json(title="檢查點劇本")}
        path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        script = review.load_script(path)
        assert script.meta.title == "檢查點劇本"


# ---- discover_checkpoint_runs ----


def _write_checkpoint(
    state_dir: Path,
    run_id: str,
    *,
    stage: str = "done",
    requirement: str = "少林弟子下山查一樁滅門案",
    last_updated: float = 100.0,
    completed_scene_ids: list[str] | None = None,
    with_script: bool = True,
    schema_version: int = 4,
) -> Path:
    run_dir = state_dir / run_id
    run_dir.mkdir(parents=True)
    state_payload = {
        "schema_version": schema_version,
        "data": {
            "run_id": run_id,
            "requirement": requirement,
            "stage": stage,
            "completed_scene_ids": completed_scene_ids or ["bt1", "bt2"],
            "last_updated": last_updated,
        },
    }
    (run_dir / "state.json").write_text(json.dumps(state_payload, ensure_ascii=False))
    if with_script:
        script_payload = {"schema_version": schema_version, "data": _script_json(n_events=2)}
        (run_dir / "script.json").write_text(json.dumps(script_payload, ensure_ascii=False))
    return run_dir


def test_discover_checkpoint_runs_picks_up_done_run_with_script():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        _write_checkpoint(state_dir, "run-a")
        records = review.discover_checkpoint_runs(state_dir=state_dir)
        assert len(records) == 1
        rec = records[0]
        assert rec.source == "checkpoint"
        assert rec.variant == "checkpoint:run-a"
        assert rec.path is not None and rec.path.name == "script.json"
        assert rec.run is not None
        assert rec.run.mode == "layered"
        assert rec.run.run_id == "run-a"
        assert rec.run.scenes_generated == 2
        assert rec.requirement == "少林弟子下山查一樁滅門案"


def test_discover_checkpoint_runs_skips_unfinished_and_scriptless():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        _write_checkpoint(state_dir, "still-running", stage="scenes")
        _write_checkpoint(state_dir, "done-no-script", with_script=False)
        _write_checkpoint(state_dir, "done-and-ready")
        records = review.discover_checkpoint_runs(state_dir=state_dir)
        assert [r.run.run_id for r in records] == ["done-and-ready"]


def test_discover_checkpoint_runs_orders_newest_first_and_respects_limit():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        _write_checkpoint(state_dir, "old", last_updated=10.0)
        _write_checkpoint(state_dir, "mid", last_updated=50.0)
        _write_checkpoint(state_dir, "new", last_updated=90.0)
        records = review.discover_checkpoint_runs(state_dir=state_dir, limit=2)
        assert [r.run.run_id for r in records] == ["new", "mid"]


def test_discover_checkpoint_runs_tolerates_corrupt_state_json():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        good = _write_checkpoint(state_dir, "good")
        bad_dir = state_dir / "corrupt"
        bad_dir.mkdir()
        (bad_dir / "state.json").write_text("not json", encoding="utf-8")
        (bad_dir / "script.json").write_text("{}", encoding="utf-8")
        records = review.discover_checkpoint_runs(state_dir=state_dir)
        assert len(records) == 1
        assert records[0].path == good / "script.json"


def test_discover_checkpoint_runs_on_missing_dir_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        records = review.discover_checkpoint_runs(state_dir=Path(tmp) / "nonexistent")
        assert records == []


def test_discover_scripts_merges_checkpoints_by_default_and_can_be_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _fixture(tmp_path)
        state_dir = tmp_path / "state"
        _write_checkpoint(state_dir, "layered-run")

        with_checkpoints = review.discover_scripts(
            scripts_dir=tmp_path / "eval",
            out_dir=tmp_path,
            requirements_file=tmp_path / "eval_reqs" / "script_requirements.txt",
            state_dir=state_dir,
        )
        without_checkpoints = review.discover_scripts(
            scripts_dir=tmp_path / "eval",
            out_dir=tmp_path,
            requirements_file=tmp_path / "eval_reqs" / "script_requirements.txt",
            state_dir=state_dir,
            include_checkpoints=False,
        )
        assert any(r.source == "checkpoint" for r in with_checkpoints)
        assert not any(r.source == "checkpoint" for r in without_checkpoints)
        assert len(with_checkpoints) == len(without_checkpoints) + 1


def test_run_record_from_row_reads_layered_fields():
    row = _write_run_row(
        mode="layered",
        run_id="run-123",
        model_extractor="m-extract",
        model_beat_expander="m-expand",
        model_scene_writer="m-scene",
        scenes_generated=7,
    )
    run = review.RunRecord.from_row(row, source_log="log.jsonl")
    assert run.mode == "layered"
    assert run.run_id == "run-123"
    assert run.model_extractor == "m-extract"
    assert run.model_beat_expander == "m-expand"
    assert run.model_scene_writer == "m-scene"
    assert run.scenes_generated == 7


def test_run_record_from_row_reads_scene_metrics():
    row = _write_run_row(
        mode="layered",
        scene_metrics=[
            {"beat_id": "bt-a", "elapsed_s": 12.5, "llm_calls": 2},
            {"beat_id": "bt-b", "elapsed_s": 3.1, "llm_calls": 1},
        ],
    )
    run = review.RunRecord.from_row(row, source_log="log.jsonl")
    assert run.scene_metrics == (
        {"beat_id": "bt-a", "elapsed_s": 12.5, "llm_calls": 2},
        {"beat_id": "bt-b", "elapsed_s": 3.1, "llm_calls": 1},
    )


def test_overview_rows_include_mode_column():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        _write_checkpoint(state_dir, "run-a")
        records = review.discover_checkpoint_runs(state_dir=state_dir)
        rows = review.overview_rows(records)
        assert rows[0]["mode"] == "layered"
        assert rows[0]["events"] == 2  # recomputed from the real checkpoint script.json


def test_review_module_does_not_import_streamlit():
    # The module's docstring *talks about* streamlit (why it's decoupled
    # from it); this only checks that nothing actually imports it.
    source = Path(review.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "streamlit" not in sys.modules


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
