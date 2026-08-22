"""Unit tests for bixiascribe.library -- delete/export/import, all against
tmp-dir fixtures. No API key, no network, no real out/ or .bixia_state/
directory touched."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import library, review  # noqa: E402
from bixiascribe.schema import Script  # noqa: E402


def _script_json(title: str = "測試劇本") -> dict:
    return {
        "meta": {"title": title, "theme": "theme"},
        "npcs": [
            {"id": "npc1", "name": "張三", "personality": "剛直", "speech_style": "直率"}
        ],
        "events": [
            {
                "id": "event0",
                "title": "事件0",
                "summary": "summary",
                "preconditions": [],
                "dialogue": [{"npc": "npc1", "line": "在下張三"}],
                "choices": [],
            }
        ],
    }


def _write_run_row(**overrides) -> dict:
    row = {
        "variant": "baseline",
        "ts": 100.0,
        "ok": True,
        "error": None,
        "requirement": "少林弟子下山查一樁滅門案",
        "script_path": "",
    }
    row.update(overrides)
    return row


def test_export_bytes_matches_generation_serialization():
    script = Script.model_validate(_script_json())
    expected = script.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    assert library.export_bytes(script) == expected


def test_delete_jsonl_sourced_record_leaves_jsonl_untouched_and_reappears_run_only():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        scripts_dir.mkdir()
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        script_path = scripts_dir / "baseline__req-abc.json"
        script_path.write_text(json.dumps(_script_json()), encoding="utf-8")

        jsonl_path = out_dir / "generation_runs.jsonl"
        row = _write_run_row(script_path=str(script_path))
        jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        records = review.discover_scripts(
            scripts_dir=scripts_dir, out_dir=out_dir, include_checkpoints=False
        )
        rec = next(r for r in records if r.source == "jsonl")

        library.delete_record(rec, scripts_dir=scripts_dir)
        assert not script_path.exists()
        assert jsonl_path.exists()
        assert jsonl_path.read_text(encoding="utf-8").strip() != ""

        records_after = review.discover_scripts(
            scripts_dir=scripts_dir, out_dir=out_dir, include_checkpoints=False
        )
        run_only = [r for r in records_after if r.source == "run-only"]
        assert len(run_only) == 1
        assert run_only[0].path is None


def test_delete_checkpoint_record_removes_whole_run_directory():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        run_dir = state_dir / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "data": {
                        "run_id": "run-a",
                        "requirement": "req",
                        "stage": "done",
                        "completed_scene_ids": ["e0"],
                        "last_updated": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "script.json").write_text(
            json.dumps({"schema_version": 4, "data": _script_json()}), encoding="utf-8"
        )
        records = review.discover_checkpoint_runs(state_dir=state_dir)
        rec = records[0]
        library.delete_record(rec, state_dir=state_dir)
        assert not run_dir.exists()


def test_delete_out_of_bounds_raises_value_error():
    with tempfile.TemporaryDirectory() as tmp:
        outside = Path(tmp) / "outside.json"
        outside.write_text(json.dumps(_script_json()), encoding="utf-8")
        rec = review.ScriptRecord(
            key="k", path=outside, variant="v", slug="s", rep=0,
            requirement="r", run=None, source="filename",
        )
        try:
            library.delete_record(rec, state_dir=Path(tmp) / "state")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for out-of-bounds delete")
        assert outside.exists()


def test_import_script_writes_discoverable_file():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        payload = json.dumps(_script_json()).encode("utf-8")
        out_path = library.import_script(
            payload, variant="imported", requirement="測試需求", scripts_dir=scripts_dir
        )
        assert out_path.exists()
        variant, slug, rep = review.parse_script_filename(out_path.name)
        assert variant == "imported"
        assert rep == 0

        records = review.discover_scripts(
            scripts_dir=scripts_dir, out_dir=Path(tmp) / "out", include_checkpoints=False
        )
        assert any(r.path == out_path for r in records)


def test_import_script_repeated_requirement_gets_rep1():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        payload = json.dumps(_script_json()).encode("utf-8")
        first = library.import_script(
            payload, variant="v", requirement="同一份需求", scripts_dir=scripts_dir
        )
        second = library.import_script(
            payload, variant="v", requirement="同一份需求", scripts_dir=scripts_dir
        )
        assert first != second
        _, _, rep1 = review.parse_script_filename(second.name)
        assert rep1 == 1


def test_import_script_accepts_checkpoint_envelope():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        payload = json.dumps({"schema_version": 4, "data": _script_json()}).encode("utf-8")
        out_path = library.import_script(
            payload, variant="v", requirement="req", scripts_dir=scripts_dir
        )
        assert out_path.exists()
        # The written file is the unwrapped Script, not the envelope.
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert "schema_version" not in written


def test_import_script_rejects_malformed_json_and_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        try:
            library.import_script(
                b"not json", variant="v", requirement="req", scripts_dir=scripts_dir
            )
        except library.ImportRejected:
            pass
        else:
            raise AssertionError("expected ImportRejected")
        assert not scripts_dir.exists() or list(scripts_dir.glob("*.json")) == []


def test_import_script_rejects_schema_invalid_payload():
    with tempfile.TemporaryDirectory() as tmp:
        scripts_dir = Path(tmp) / "eval"
        payload = json.dumps({"not_a_script": True}).encode("utf-8")
        try:
            library.import_script(
                payload, variant="v", requirement="req", scripts_dir=scripts_dir
            )
        except library.ImportRejected:
            pass
        else:
            raise AssertionError("expected ImportRejected")
        assert not scripts_dir.exists() or list(scripts_dir.glob("*.json")) == []


def test_library_module_does_not_import_streamlit():
    source = Path(library.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "streamlit" not in sys.modules


def test_load_ad_hoc_returns_adhoc_source():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "some__file.json"
        path.write_text(json.dumps(_script_json()), encoding="utf-8")
        script, rec = library.load_ad_hoc(path)
        assert rec.source == "adhoc"
        assert rec.run is None
        assert script.meta.title == "測試劇本"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
