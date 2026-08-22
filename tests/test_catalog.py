"""Unit tests for bixiascribe.catalog -- curated model/role/reasoning-effort
metadata joined against pricing.py at load time. No network/API dependency."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe import catalog  # noqa: E402
from bixiascribe.generation import DEFAULT_VARIANTS_FILE  # noqa: E402


def test_catalog_and_prices_ids_are_bidirectionally_consistent():
    cat = catalog.load_catalog()
    from bixiascribe import pricing

    prices = pricing.load_prices()
    catalog_ids = set(cat.models)
    price_ids = set(prices)
    assert catalog_ids == price_ids, (
        f"catalog/model_prices id mismatch: "
        f"catalog-only={catalog_ids - price_ids}, prices-only={price_ids - catalog_ids}"
    )


def test_every_model_variants_entry_has_a_catalog_entry():
    cat = catalog.load_catalog()
    variants = json.loads(DEFAULT_VARIANTS_FILE.read_text(encoding="utf-8"))
    model_ids = set()
    for row in variants:
        for key in ("writer", "dialogue", "proof", "extractor", "beat_expander", "scene_writer"):
            value = row.get(key)
            if value:
                model_ids.add(value)
    missing = model_ids - set(cat.models)
    assert not missing, f"model_variants.json references uncataloged model ids: {missing}"


def test_recommended_roles_are_valid_role_keys():
    cat = catalog.load_catalog()
    valid_roles = set(cat.roles)
    for info in cat.models.values():
        for role in info.recommended_roles:
            assert role in valid_roles, f"{info.model_id} recommends unknown role {role!r}"


def test_describe_degrades_gracefully_on_unknown_id():
    cat = catalog.load_catalog()
    info = cat.describe("openrouter/does-not-exist/model")
    assert info.status == "untested"
    assert info.label == "openrouter/does-not-exist/model"
    assert info.price is None


def test_describe_empty_id():
    cat = catalog.load_catalog()
    info = cat.describe("")
    assert info.status == "untested"


def test_missing_catalog_file_returns_empty_catalog():
    with tempfile.TemporaryDirectory() as tmp:
        cat = catalog.load_catalog(path=Path(tmp) / "nonexistent.json")
        assert cat.models == {}
        assert cat.roles == {}
        assert cat.reasoning_efforts == {}


def test_corrupt_catalog_file_returns_empty_catalog():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "catalog.json"
        path.write_text("not json", encoding="utf-8")
        cat = catalog.load_catalog(path=path)
        assert cat.models == {}


def test_selectable_excludes_unusable_includes_baseline():
    cat = catalog.load_catalog()
    ids = {info.model_id for info in cat.selectable()}
    assert "openrouter/z-ai/glm-5.2" not in ids
    assert "openrouter/deepseek/deepseek-chat" in ids


def test_selectable_filters_by_role():
    cat = catalog.load_catalog()
    writer_models = cat.selectable("writer")
    assert all("writer" in info.recommended_roles for info in writer_models)


def test_normalize_reasoning_effort_falls_back_to_default():
    assert catalog.normalize_reasoning_effort(None) == "default"
    assert catalog.normalize_reasoning_effort("") == "default"
    assert catalog.normalize_reasoning_effort("garbage") == "default"
    assert catalog.normalize_reasoning_effort("HIGH") == "high"


def test_roles_for_mode():
    cat = catalog.load_catalog()
    layered_roles = {info.role for info in cat.roles_for_mode("layered")}
    assert "extractor" in layered_roles
    assert "writer" not in layered_roles


def test_catalog_module_does_not_import_streamlit():
    source = Path(catalog.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "streamlit" not in sys.modules


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
