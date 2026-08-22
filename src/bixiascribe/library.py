"""Mutating operations (delete/export/import) over discovered script
records -- kept out of review.py deliberately, since that module's own
docstring commits to being a read-only index that ~35 existing tests build
on (see design.md's 決策三). Depends on review.py (reading/naming) and
generation.py (naming conventions/rep allocation), one-directional.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .generation import next_rep, script_path_for
from .review import ScriptRecord, load_script, parse_script_filename, requirement_slug
from .schema import Script


@dataclass(frozen=True)
class DeletePlan:
    """What delete_record()/plan_delete() would remove: a single file for a
    jsonl/filename-sourced record, or a whole checkpoint directory for a
    checkpoint-sourced one. `kind` is "file" or "directory"."""

    kind: str
    targets: tuple[Path, ...] = field(default_factory=tuple)


def _check_within(target: Path, *roots: Path) -> None:
    resolved = target.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise ValueError(f"路徑不在允許刪除的目錄之內：{target}")


def plan_delete(
    rec: ScriptRecord,
    *,
    state_dir: Path = config.BIXIA_STATE_DIR,
    scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
) -> DeletePlan:
    """What deleting `rec` would remove, without touching disk. Raises
    ValueError if `rec` has no backing path, or if the resolved target falls
    outside scripts_dir/state_dir -- see script-library spec's "Delete
    refuses a path outside managed directories" scenario."""
    if rec.path is None:
        raise ValueError(f"此紀錄沒有對應的檔案，無法刪除：{rec.key!r}")

    if rec.source == "checkpoint":
        target_dir = rec.path.parent
        _check_within(target_dir, state_dir)
        return DeletePlan(kind="directory", targets=(target_dir,))

    _check_within(rec.path, scripts_dir)
    return DeletePlan(kind="file", targets=(rec.path,))


def delete_record(
    rec: ScriptRecord,
    *,
    state_dir: Path = config.BIXIA_STATE_DIR,
    scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
) -> DeletePlan:
    """Delete whatever plan_delete() plans, never touching
    out/generation_runs*.jsonl -- a jsonl-sourced record's run row survives
    and reappears as source="run-only" on the next discovery pass (see
    script-library spec)."""
    plan = plan_delete(rec, state_dir=state_dir, scripts_dir=scripts_dir)
    for target in plan.targets:
        if plan.kind == "directory":
            import shutil

            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    return plan


def export_bytes(script: Script) -> bytes:
    """Byte-identical to generation.py's own out/eval/ serialization, so a
    round-tripped export/import produces the exact same file content a real
    run would have written."""
    return script.model_dump_json(indent=2, exclude_none=False).encode("utf-8")


def export_filename(rec: ScriptRecord) -> str:
    suffix = f"__rep{rec.rep}" if rec.rep else ""
    slug = rec.slug or requirement_slug(rec.requirement) or rec.key
    variant = rec.variant or "adhoc"
    return f"{variant}__{slug}{suffix}.json"


class ImportRejected(ValueError):
    """Raised by validate_import()/import_script() for any payload that
    isn't valid JSON, or is valid JSON that doesn't validate as a Script
    (directly, or after unwrapping a checkpoint envelope)."""


def validate_import(payload: bytes) -> Script:
    """Parse+validate an uploaded payload as a Script, unwrapping a
    {"schema_version", "data"} checkpoint envelope the same structural way
    review.load_script() does. Every failure mode (bad JSON, wrong shape,
    schema validation error) is wrapped as ImportRejected with a
    user-facing reason."""
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImportRejected(f"不是有效的 JSON：{exc}") from exc

    if isinstance(raw, dict) and set(raw.keys()) == {"schema_version", "data"}:
        data = raw["data"]
        if isinstance(data, dict):
            raw = data

    try:
        return Script.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a user-facing reason
        raise ImportRejected(f"不符合 Script schema：{exc}") from exc


def import_script(
    payload: bytes,
    *,
    variant: str,
    requirement: str,
    rep: int | None = None,
    scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
) -> Path:
    """Validate `payload` and write it into scripts_dir under the standard
    {variant}__{slug}[__repN].json naming convention (generation.py's own
    script_path_for()/next_rep()), so it's discoverable by
    review.discover_scripts() with no additional code. Raises ImportRejected
    (writes nothing) for an invalid payload."""
    script = validate_import(payload)
    resolved_rep = rep if rep is not None else next_rep(
        scripts_dir, variant, requirement_slug(requirement)
    )
    out_path = script_path_for(requirement, variant, resolved_rep, scripts_dir)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(export_bytes(script).decode("utf-8"), encoding="utf-8")
    return out_path


def load_ad_hoc(path: Path) -> tuple[Script, ScriptRecord]:
    """Load a Script from an arbitrary path for one-time viewing, without
    copying it into scripts_dir. Returns a ScriptRecord with
    source="adhoc"/run=None so it renders via the same logic as any
    discovered record but is never treated as a persisted artifact (no
    delete button -- see design.md's path-escape mitigation)."""
    script = load_script(path)
    try:
        variant, slug, rep = parse_script_filename(path.name)
    except ValueError:
        variant, slug, rep = "(adhoc)", path.stem, 0
    rec = ScriptRecord(
        key=f"adhoc:{path}",
        path=path,
        variant=variant,
        slug=slug,
        rep=rep,
        requirement="",
        run=None,
        source="adhoc",
    )
    return script, rec


__all__ = [
    "DeletePlan",
    "plan_delete",
    "delete_record",
    "export_bytes",
    "export_filename",
    "ImportRejected",
    "validate_import",
    "import_script",
    "load_ad_hoc",
]
