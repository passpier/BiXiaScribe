"""Curated model/role/reasoning-effort catalog -- the single source of truth
for which models are tested/usable/deprecated, joined against
eval/model_prices.json at load time so price data is never duplicated (see
eval/model_catalog.json's own _comment and design.md's Decisions).

Deliberately dependency-free apart from config+pricing (never imports
llm.py/crewai) so it stays importable from review.py/generation.py without
pulling crewai into those streamlit-free modules, same rationale as
pricing.py's own module docstring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config, pricing

DEFAULT_CATALOG_FILE = config.PROJECT_ROOT / "eval" / "model_catalog.json"

_VALID_REASONING_EFFORTS = ("default", "none", "low", "medium", "high")


@dataclass(frozen=True)
class ModelInfo:
    """One model's catalog entry, joined against its eval/model_prices.json
    ModelPrice (if any). `price` is None for a model with no pricing entry --
    callers must not treat that as free, same convention as pricing.py."""

    model_id: str
    label: str
    description: str = ""
    status: str = "untested"  # tested | baseline | unusable | untested
    recommended_roles: tuple[str, ...] = ()
    price: pricing.ModelPrice | None = None


@dataclass(frozen=True)
class RoleInfo:
    role: str
    label: str
    modes: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ReasoningEffortInfo:
    value: str
    label: str
    note: str = ""


@dataclass(frozen=True)
class Catalog:
    models: dict[str, ModelInfo] = field(default_factory=dict)
    roles: dict[str, RoleInfo] = field(default_factory=dict)
    reasoning_efforts: dict[str, ReasoningEffortInfo] = field(default_factory=dict)

    def describe(self, model_id: str) -> ModelInfo:
        """Never raises. A model_id with no catalog entry (e.g. a historical
        run using a model since removed from the catalog) degrades to a
        description using the raw id as its own label and status="untested"
        rather than an error -- see model-catalog spec's "Unknown model
        degrades gracefully" scenario."""
        if not model_id:
            return ModelInfo(model_id="", label="—", status="untested")
        info = self.models.get(model_id)
        if info is not None:
            return info
        return ModelInfo(model_id=model_id, label=model_id, status="untested")

    def selectable(self, role: str | None = None) -> list[ModelInfo]:
        """Models excluding status="unusable", optionally filtered to those
        that recommend `role`. Order follows insertion order of the catalog
        file."""
        result = [info for info in self.models.values() if info.status != "unusable"]
        if role is not None:
            result = [info for info in result if role in info.recommended_roles]
        return result

    def roles_for_mode(self, mode: str) -> list[RoleInfo]:
        return [info for info in self.roles.values() if mode in info.modes]


def model_label(info: ModelInfo) -> str:
    """format_func-friendly display string: label plus a price segment
    (omitted entirely when unpriced, rather than showing a misleading $0)."""
    if info.price is None:
        return info.label
    p = info.price
    return f"{info.label}（${p.prompt_usd_per_1m:.2f}/${p.completion_usd_per_1m:.2f} per 1M）"


def normalize_reasoning_effort(value: str | None) -> str:
    """Validate-and-fallback to "default", imitating config.py's
    CAUSAL_VALIDATION pattern -- any unrecognized/empty value degrades to
    "default" (a no-op) rather than raising or propagating garbage."""
    value = (value or "").strip().lower()
    return value if value in _VALID_REASONING_EFFORTS else "default"


def load_catalog(
    path: Path = DEFAULT_CATALOG_FILE,
    prices: dict[str, pricing.ModelPrice] | None = None,
) -> Catalog:
    """Parse eval/model_catalog.json into a Catalog, joined against
    eval/model_prices.json (or a caller-supplied `prices` dict, e.g. for
    tests). Missing/corrupt catalog file -> empty Catalog (same
    degrade-not-crash convention as pricing.load_prices()), not an error."""
    if not path.exists():
        return Catalog()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Catalog()
    if not isinstance(raw, dict):
        return Catalog()

    prices = prices if prices is not None else pricing.load_prices()

    models: dict[str, ModelInfo] = {}
    for model_id, entry in (raw.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        models[model_id] = ModelInfo(
            model_id=model_id,
            label=entry.get("label") or model_id,
            description=entry.get("description") or "",
            status=entry.get("status") or "untested",
            recommended_roles=tuple(entry.get("recommended_roles") or ()),
            price=prices.get(model_id),
        )

    roles: dict[str, RoleInfo] = {}
    for role_key, entry in (raw.get("roles") or {}).items():
        if not isinstance(entry, dict):
            continue
        roles[role_key] = RoleInfo(
            role=role_key,
            label=entry.get("label") or role_key,
            modes=tuple(entry.get("modes") or ()),
            note=entry.get("note") or "",
        )

    reasoning_efforts: dict[str, ReasoningEffortInfo] = {}
    for value, entry in (raw.get("reasoning_efforts") or {}).items():
        if not isinstance(entry, dict):
            continue
        reasoning_efforts[value] = ReasoningEffortInfo(
            value=value,
            label=entry.get("label") or value,
            note=entry.get("note") or "",
        )

    return Catalog(models=models, roles=roles, reasoning_efforts=reasoning_efforts)


__all__ = [
    "DEFAULT_CATALOG_FILE",
    "ModelInfo",
    "RoleInfo",
    "ReasoningEffortInfo",
    "Catalog",
    "model_label",
    "normalize_reasoning_effort",
    "load_catalog",
]
