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
        "title": title,
        "premise": "premise",
        "variables": [],
        "npcs": [
            {
                "id": "npc1",
                "name": "張三",
                "identity": "俠客",
                "personality": "剛直",
                "speech_style": "直率",
            }
        ],
        "events": [
            {
                "id": f"event{i}",
                "title": f"事件{i}",
                "location": "洛陽",
                "summary": "summary",
                "triggers": [],
                "dialogue": [{"npc_id": "npc1", "line": "在下張三", "emotion": ""}],
                "branches": [],
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
        )
        rows = review.overview_rows(records)
        assert len(rows) == 1
        assert rows[0]["events"] == 3  # recomputed from the file, not the stale JSONL 99


def test_load_script_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "script.json"
        path.write_text(json.dumps(_script_json(), ensure_ascii=False), encoding="utf-8")
        script = review.load_script(path)
        assert script.title == "測試劇本"


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
