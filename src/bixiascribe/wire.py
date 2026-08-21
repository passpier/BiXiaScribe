"""Lenient mirrors of the schema.py models used as CrewAI's `output_pydantic`,
in place of the strict models themselves.

Why this exists: crewai's structured-output path
(`crewai.utilities.converter.generate_model_description`) does not send the
strict model's own JSON schema to the provider -- it runs
`ensure_all_properties_required()` on it first, which marks *every* field
required regardless of whether schema.py gave it a default, and sets
`additionalProperties: false`/`strict: true`. Measured against `Event`:
pydantic-level required = 4 fields (id/title/location/summary), wire-level
required = all 14 top-level fields plus every nested Branch (11/11) and
SkillCheck (8/8) field. If the provider's structured-output enforcement is
imperfect (not guaranteed for every OpenRouter route) and the model omits
even one of those -- e.g. `Branch.next_event_id`, observed dropped even when
the prompt explicitly says it's still required alongside the new
`converges_to_event_id` field -- `openai.lib._parsing._completions.
parse_chat_completion`'s `model_validate_json` raises `ValidationError`
*inside* `Task.execute_sync()`, before `crew/pipeline.py::_coerce_model`'s
three-tier (pydantic -> json_dict -> raw_scan) rescue ever gets a chance to
run. That rescue logic is effectively dead code against this failure mode.

`lenient_mirror(Model)` builds a same-shaped model where every field
(including every nested model's fields, recursively) has a default --
`ensure_all_properties_required` still marks all of *its* fields required on
the wire, but every one of those wire-required fields already has a
schema-legal empty value (`""`/`0`/`False`/`[]`/a recursively-empty nested
model) for a corner-cutting model to fall back on, and a genuinely-missing
key no longer raises: pydantic just applies the mirror's own default. This
does not shrink the wire schema (still just as many bytes, still all
required) -- it only converts "field absent" from a hard failure into
inspectable data. Shrinking the schema itself is a separate, later lever
(see CLAUDE.md's "Script generation" section on script length/complexity).

`to_strict(obj, Model)` converts a filled-in mirror instance back to the
real strict model, substituting a type-appropriate empty value for any
field that is still genuinely missing (never fabricating content -- an
empty string/list is not new information, just a placeholder the existing
repair loops and crew/normalize.py can act on).

Pure module: no crewai/config import, same dependency-free convention as
length.py/crew/causal.py, so it can be imported from crew/tasks.py without
pulling in the LLM stack.
"""
from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, create_model

_MIRROR_CACHE: dict[type[BaseModel], type[BaseModel]] = {}


def _is_union(ann: Any) -> bool:
    origin = get_origin(ann)
    return origin is Union or origin is types.UnionType


def _is_model(ann: Any) -> bool:
    return isinstance(ann, type) and issubclass(ann, BaseModel)


def _non_none_args(ann: Any) -> list[Any]:
    return [a for a in get_args(ann) if a is not type(None)]


def lenient_mirror(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Return a cached, structurally-mirrored version of `model_cls` where
    every field (recursively, for nested models/lists-of-models) has a
    default, so nothing is required at the pydantic level. crewai's own
    `ensure_all_properties_required()` still marks every field required on
    the wire schema it sends the provider -- this only ensures each of
    those wire-required fields already has a legal fallback value, so a
    model that omits one produces valid-but-empty data instead of a hard
    `ValidationError` inside `Task.execute_sync()`."""
    if model_cls in _MIRROR_CACHE:
        return _MIRROR_CACHE[model_cls]

    # `schema.py` uses `from __future__ import annotations` and some models
    # (e.g. Script -> Chapter) reference a class defined later in the file,
    # so `model_fields[name].annotation` can still be an unresolved
    # ForwardRef until pydantic lazily rebuilds the strict model itself.
    # `get_type_hints` forces resolution against the model's own module
    # namespace instead of relying on that having already happened.
    hints = get_type_hints(model_cls)
    fields: dict[str, tuple[Any, Any]] = {}
    for name in model_cls.model_fields:
        lenient_ann, default = _lenient_field(hints[name])
        fields[name] = (lenient_ann, default)

    mirror = create_model(  # type: ignore[call-overload]
        f"Lenient{model_cls.__name__}",
        __base__=BaseModel,
        **fields,
    )
    _MIRROR_CACHE[model_cls] = mirror
    return mirror


def _lenient_field(ann: Any) -> tuple[Any, Any]:
    """Return (lenient_annotation, default) for one field's annotation."""
    origin = get_origin(ann)

    if origin is list:
        (item,) = get_args(ann) or (Any,)
        if _is_model(item):
            return list[lenient_mirror(item)], []  # type: ignore[misc]
        return ann, []

    if _is_union(ann):
        args = _non_none_args(ann)
        if len(args) == 1 and _is_model(args[0]):
            return lenient_mirror(args[0]) | None, None
        if _includes_none(ann):
            return ann, None
        return ann | None, None

    if _is_model(ann):
        return lenient_mirror(ann) | None, None

    return ann | None, None


def _includes_none(ann: Any) -> bool:
    return type(None) in get_args(ann)


def _empty_for_annotation(ann: Any) -> Any:
    """A schema-legal, content-free value for `ann` -- never fabricated
    narrative content, just the same zero value schema.py's own
    `Field(default_factory=...)`/`= ""` conventions already use."""
    origin = get_origin(ann)

    if origin is list:
        return []

    if _is_union(ann):
        args = _non_none_args(ann)
        if args:
            return _empty_for_annotation(args[0])
        return None

    if _is_model(ann):
        return _empty_model(ann)

    if ann is str:
        return ""
    if ann is bool:
        return False
    if ann is int:
        return 0
    if ann is float:
        return 0.0
    return None


def _empty_model(model_cls: type[BaseModel]) -> BaseModel:
    hints = get_type_hints(model_cls)
    kwargs = {
        name: _empty_for_annotation(hints[name])
        for name, info in model_cls.model_fields.items()
        if info.is_required()
    }
    return model_cls(**kwargs)


def to_strict(obj: BaseModel | dict[str, Any] | None, model_cls: type[BaseModel]) -> BaseModel:
    """Convert a lenient-mirror instance (or an equivalent dict) back into
    a real `model_cls` instance. Any field still missing/None after the
    mirror -- genuinely absent from the model's output, not merely
    optional in schema.py -- is filled with `_empty_for_annotation`'s
    placeholder rather than raising, so this never fails on incomplete
    data; the caller (crew/normalize.py, the existing repair loops) is
    what decides whether an empty placeholder is acceptable or needs a
    targeted fix."""
    data = obj.model_dump() if isinstance(obj, BaseModel) else (obj or {})
    return _coerce_dict(data, model_cls)


def _coerce_dict(data: dict[str, Any], model_cls: type[BaseModel]) -> BaseModel:
    hints = get_type_hints(model_cls)
    kwargs: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if name not in data or data[name] is None:
            if info.is_required():
                kwargs[name] = _empty_for_annotation(hints[name])
            continue
        kwargs[name] = _coerce_value(data[name], hints[name])
    return model_cls(**kwargs)


def _coerce_value(value: Any, ann: Any) -> Any:
    origin = get_origin(ann)

    if origin is list:
        (item,) = get_args(ann) or (Any,)
        if _is_model(item) and isinstance(value, list):
            return [_coerce_dict(v, item) if isinstance(v, dict) else v for v in value]
        return value

    if _is_union(ann):
        args = _non_none_args(ann)
        if len(args) == 1 and _is_model(args[0]) and isinstance(value, dict):
            return _coerce_dict(value, args[0])
        return value

    if _is_model(ann) and isinstance(value, dict):
        return _coerce_dict(value, ann)

    return value
