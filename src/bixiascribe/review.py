"""Read-only index over the Stage 2 eval artifacts (out/eval/*.json +
out/generation_runs*.jsonl), built for the Stage 3 review UI (ui/app.py).

Deliberately pure Python -- no streamlit import anywhere in this module --
so the Stage 3 "臨時駕駛艙" (per the 武俠 RPG 劇本 RAG 架構方案 doc, Streamlit
is meant to be swapped for a Tauri desktop app later) can be replaced without
touching this data layer, and so this module stays unit-testable against a
tmp dir (see tests/test_review.py) without ever importing streamlit.

Two things this module is careful about, both learned from the real out/
data rather than assumed:

1. `out/eval/*.json` files carry no metadata of their own (no variant, no
   requirement) -- the only join key back to a run's metadata is
   `script_path` inside a `out/generation_runs*.jsonl` row. Since
   scripts/eval_generation.py's `rep > 0` suffix means only the *first* two
   reps of a (variant, requirement) pair get distinct files, a script file
   can be the target of more than one JSONL row across separate invocations
   (crash-resilient append-only logs); `latest_run_by_path` keeps the most
   recent one by `ts`.
2. `script_metrics()` counts embedded in a JSONL row can go stale: a rep can
   overwrite a script file after the row was written. `overview_rows` never
   trusts those counts -- it always recomputes `script_metrics()` from the
   file currently on disk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .crew.metrics import script_metrics
from .schema import Script

# variant names in eval/model_variants.json use hyphens, never "__" -- so
# splitting a script filename's stem on "__" and taking the first segment as
# the variant is safe even though the *slug* half can itself contain "__"
# (requirement_slug keeps "_" as a valid ascii_only character).
_REP_SUFFIX_RE = re.compile(r"rep(\d+)")


def requirement_slug(text: str, max_len: int = 24) -> str:
    """Filesystem-safe stand-in for a requirement string, used in
    out/eval/*.json filenames. Falls back to a content hash (not Python's
    hash(), which is randomized per-process) for non-ASCII text -- the
    normal case, since these are Chinese 劇情需求 -- so the same requirement
    maps to the same filename across separate invocations.

    This is the same function scripts/eval_generation.py used to define as
    a private `_slug` helper; it now lives here so both the eval harness and
    the review UI share one implementation.
    """
    ascii_only = re.sub(r"[^\w-]", "", text.encode("ascii", "ignore").decode("ascii"))
    if ascii_only:
        return ascii_only[:max_len]
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"req-{digest}"


def load_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts, skipping blank lines."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_script_filename(name: str) -> tuple[str, str, int]:
    """Recover (variant, slug, rep) from an out/eval/*.json filename written
    by scripts/eval_generation.py's `{variant}__{slug}[__rep{N}].json`
    pattern, without assuming the slug itself is free of "__" (it isn't
    always -- requirement_slug's ascii_only path keeps underscores).

    Raises ValueError if `name` doesn't look like that pattern at all (no
    "__" separator) -- callers should treat that as "can't parse this
    file", not silently guess.
    """
    stem = name[:-len(".json")] if name.endswith(".json") else name
    parts = stem.split("__")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"can't parse variant/slug from filename: {name!r}")

    rep = 0
    if len(parts) > 2:
        m = _REP_SUFFIX_RE.fullmatch(parts[-1])
        if m:
            rep = int(m.group(1))
            parts = parts[:-1]

    variant = parts[0]
    slug = "__".join(parts[1:])
    if not slug:
        raise ValueError(f"can't parse variant/slug from filename: {name!r}")
    return variant, slug, rep


@dataclass(frozen=True)
class RunRecord:
    """One row from out/generation_runs*.jsonl, normalized to always-present
    fields (empty string/0/() instead of missing keys or None), mirroring
    schema.py's "no Optional; emptiness means absent" convention.

    Deliberately does NOT carry script_metrics() fields (events, npcs,
    branches, dialogue_lines, ...) even though scripts/eval_generation.py's
    rows include them -- those can be stale (see module docstring); a
    ScriptRecord's metrics are always recomputed from the file on disk.
    """

    variant: str = ""
    ts: float = 0.0
    ok: bool = False
    error: str = ""
    requirement: str = ""
    model_writer: str = ""
    model_dialogue: str = ""
    model_proof: str = ""
    elapsed_s: float = 0.0
    token_usage: dict[str, Any] = field(default_factory=dict)
    retrieval_calls: int = 0
    retrieval_failures: int = 0
    retrieval_queries: tuple[str, ...] = ()
    repair_attempts: int = 0
    coerced_from: str = ""
    script_path: str = ""
    source_log: str = ""
    # Cost accounting (pricing.py), computed by generation.build_run_row()
    # and stored on the row (not recomputed here) -- unlike script_metrics(),
    # which overview_rows() always recomputes from disk, a run's actual
    # token_usage isn't reconstructable after the fact, so the row's stored
    # figure is the only one that can exist. None (not 0.0) when
    # cost_basis=="unknown_price" -- see pricing.estimate_cost()'s docstring
    # for why a missing price must never silently read as "free".
    cost_usd: float | None = None
    cost_basis: str = ""
    # Whether this run had wuxia_corpus_search available at all
    # (config.RETRIEVAL_ENABLED / Variant.use_retrieval). True for every row
    # logged before this field existed (dataclass default + .get() below),
    # matching the knob's own pre-existing "on" behavior -- same convention
    # as script_length's "short" default.
    retrieval_enabled: bool = True
    # Whether crew/guardrails.py's RPG-shape checks were wired onto this
    # run's tasks (config.GUARDRAILS_ENABLED). False for every row logged
    # before this field existed (guardrails didn't exist then) -- unlike
    # retrieval_enabled, "off" is the historically-accurate default here.
    guardrails_enabled: bool = False
    guardrail_max_retries: int = 0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict, source_log: str) -> RunRecord:
        return cls(
            variant=row.get("variant") or "",
            ts=row.get("ts") or 0.0,
            ok=bool(row.get("ok")),
            error=row.get("error") or "",
            requirement=row.get("requirement") or "",
            model_writer=row.get("model_writer") or "",
            model_dialogue=row.get("model_dialogue") or "",
            model_proof=row.get("model_proof") or "",
            elapsed_s=row.get("elapsed_s") or 0.0,
            token_usage=row.get("token_usage") or {},
            retrieval_calls=row.get("retrieval_calls") or 0,
            retrieval_failures=row.get("retrieval_failures") or 0,
            retrieval_queries=tuple(row.get("retrieval_queries") or ()),
            repair_attempts=row.get("repair_attempts") or 0,
            coerced_from=row.get("coerced_from") or "",
            script_path=row.get("script_path") or "",
            cost_usd=row.get("cost_usd"),
            cost_basis=row.get("cost_basis") or "",
            retrieval_enabled=row.get("retrieval_enabled", True),
            guardrails_enabled=row.get("guardrails_enabled", False),
            guardrail_max_retries=row.get("guardrail_max_retries") or 0,
            source_log=source_log,
            raw=row,
        )


@dataclass(frozen=True)
class ScriptRecord:
    """One browsable unit in the review UI: either a script file (joined to
    its run metadata when available) or a failed run that never produced a
    file at all (`path is None`, `source == "run-only"`) -- the latter is
    how e.g. the `cheap-ends` variant's structured-output failures become
    visible instead of silently vanishing because there's no JSON to open.
    """

    key: str
    path: Path | None
    variant: str
    slug: str
    rep: int
    requirement: str
    run: RunRecord | None
    source: str  # "jsonl" | "filename" | "run-only"


def load_run_records(
    out_dir: Path = config.OUT_DIR, glob: str = config.RUN_LOG_GLOB
) -> list[RunRecord]:
    """Load every row from every out/generation_runs*.jsonl file found under
    out_dir, tagging each with the log file it came from. Returns [] if
    out_dir doesn't exist (e.g. a clean checkout before any eval run)."""
    if not out_dir.is_dir():
        return []
    records: list[RunRecord] = []
    for path in sorted(out_dir.glob(glob)):
        for row in load_jsonl(path):
            records.append(RunRecord.from_row(row, source_log=path.name))
    return records


def latest_run_by_path(runs: list[RunRecord]) -> dict[str, RunRecord]:
    """Dedupe RunRecords by script_path, keeping the one with the largest
    `ts` (a script_path can be written by more than one run across separate
    eval_generation.py invocations; the newest one describes what's
    currently on disk)."""
    best: dict[str, RunRecord] = {}
    for run in runs:
        if not run.script_path:
            continue
        current = best.get(run.script_path)
        if current is None or run.ts >= current.ts:
            best[run.script_path] = run
    return best


def load_requirement_texts(path: Path = config.EVAL_REQUIREMENTS_FILE) -> dict[str, str]:
    """slug -> full requirement text, for recovering requirement text for
    script files that have no matching JSONL row. Uses the same parse rules
    as scripts/eval_generation.py's requirement-file loader (strip, skip
    blank lines and '#' comments)."""
    if not path.is_file():
        return {}
    texts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        texts[requirement_slug(line)] = line
    return texts


def discover_scripts(
    scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
    out_dir: Path = config.OUT_DIR,
    requirements_file: Path = config.EVAL_REQUIREMENTS_FILE,
) -> list[ScriptRecord]:
    """Build the full browsable list: every out/eval/*.json file (joined to
    its run metadata when found), plus every failed run that never produced
    a file. Never drops a script file, even if its filename can't be parsed
    or its run metadata can't be found -- an unrecognized file still shows
    up as `variant="(unknown)"` rather than disappearing."""
    runs = load_run_records(out_dir)
    by_path = latest_run_by_path(runs)
    by_basename = {Path(p).name: run for p, run in by_path.items()}
    requirement_texts = load_requirement_texts(requirements_file)

    records: list[ScriptRecord] = []
    matched_paths: set[str] = set()

    if scripts_dir.is_dir():
        for file_path in sorted(scripts_dir.glob("*.json")):
            resolved = str(file_path.resolve())
            run = by_path.get(resolved) or by_basename.get(file_path.name)

            if run is not None:
                matched_paths.add(run.script_path)
                try:
                    variant, slug, rep = parse_script_filename(file_path.name)
                except ValueError:
                    variant, slug, rep = run.variant or "(unknown)", "", 0
                records.append(
                    ScriptRecord(
                        key=file_path.stem,
                        path=file_path,
                        variant=run.variant or variant,
                        slug=slug,
                        rep=rep,
                        requirement=run.requirement,
                        run=run,
                        source="jsonl",
                    )
                )
                continue

            try:
                variant, slug, rep = parse_script_filename(file_path.name)
            except ValueError:
                variant, slug, rep = "(unknown)", file_path.stem, 0
            records.append(
                ScriptRecord(
                    key=file_path.stem,
                    path=file_path,
                    variant=variant,
                    slug=slug,
                    rep=rep,
                    requirement=requirement_texts.get(slug, ""),
                    run=None,
                    source="filename",
                )
            )

    # Runs that never produced a surviving file (failed before writing, or
    # the file was later deleted) -- still worth showing, error and all.
    for run in runs:
        if run.script_path and run.script_path in matched_paths:
            continue
        if run.script_path and Path(run.script_path).is_file():
            # Points at a file that exists but wasn't picked up above (e.g.
            # scripts_dir differs from where the log says it wrote to) --
            # don't duplicate it as a run-only record.
            continue
        slug = requirement_slug(run.requirement) if run.requirement else ""
        records.append(
            ScriptRecord(
                key=f"{run.variant}@{run.ts:.0f}",
                path=None,
                variant=run.variant,
                slug=slug,
                rep=0,
                requirement=run.requirement,
                run=run,
                source="run-only",
            )
        )

    return records


def load_script(path: Path) -> Script:
    """Load and validate a Script from an out/eval/*.json file."""
    return Script.model_validate_json(path.read_text(encoding="utf-8"))


def group_by_requirement(records: list[ScriptRecord]) -> dict[str, list[ScriptRecord]]:
    """Group records by requirement text (falling back to slug when the
    requirement text couldn't be recovered), preserving first-seen order."""
    groups: dict[str, list[ScriptRecord]] = {}
    for rec in records:
        key = rec.requirement or rec.slug or rec.key
        groups.setdefault(key, []).append(rec)
    return groups


def variant_names(records: list[ScriptRecord]) -> list[str]:
    return sorted({rec.variant for rec in records})


def find_record(
    records: list[ScriptRecord], requirement_key: str, variant: str, rep: int
) -> ScriptRecord | None:
    for rec in records:
        key = rec.requirement or rec.slug or rec.key
        if key == requirement_key and rec.variant == variant and rec.rep == rep:
            return rec
    return None


def npc_names(script: Script) -> dict[str, str]:
    """npc_id -> display name, plus the player's own id if present --
    dialogue.npc_id is allowed to reference script.player.id (see
    schema.validate_references()), so without this a player line would
    render as an "unknown speaker" warning in the UI."""
    names = {npc.id: npc.name for npc in script.npcs}
    if script.player:
        names[script.player.id] = script.player.name or "玩家"
    return names


def event_titles(script: Script) -> dict[str, str]:
    """event_id -> title."""
    return {event.id: event.title for event in script.events}


def quest_names(script: Script) -> dict[str, str]:
    """quest_id -> display name."""
    return {quest.id: quest.name for quest in script.quests}


def overview_rows(records: list[ScriptRecord]) -> list[dict]:
    """One flat dict per record, for a st.dataframe overview table. Metrics
    are always recomputed from the script file on disk (never trusted from
    a possibly-stale JSONL row) -- records with no file (source ==
    "run-only") get metric fields left at 0."""
    rows = []
    for rec in records:
        run = rec.run
        metrics: dict[str, Any] = {
            "events": 0,
            "npcs": 0,
            "variables": 0,
            "branches": 0,
            "dialogue_lines": 0,
            "dialogue_lines_per_event": 0.0,
            "events_with_dialogue_pct": 0.0,
            "npc_speaking_pct": 0.0,
            "avg_line_chars": 0.0,
            # Continuity metrics (crew/metrics.py::continuity_metrics()).
            "distinct_event_title_pct": 0.0,
            "repeated_dialogue_pct": 0.0,
            "prior_entity_reference_pct": 0.0,
            "connected_event_pct": 0.0,
            "self_loop_branch_pct": 0.0,
        }
        if rec.path is not None:
            try:
                metrics.update(script_metrics(load_script(rec.path)))
            except Exception:
                pass

        total_tokens = None
        if run is not None and run.token_usage:
            total_tokens = run.token_usage.get("total_tokens")

        row = {
            "variant": rec.variant,
            "requirement": rec.requirement or rec.slug,
            "rep": rec.rep,
            "ok": run.ok if run is not None else (rec.path is not None),
            "error": run.error if run is not None else "",
            "source": rec.source,
            "retrieval_calls": run.retrieval_calls if run is not None else None,
            "total_tokens": total_tokens,
            "cost_usd": run.cost_usd if run is not None else None,
            "cost_basis": run.cost_basis if run is not None else "",
            "elapsed_s": run.elapsed_s if run is not None else None,
            "coerced_from": run.coerced_from if run is not None else "",
            "script_path": rec.path and str(rec.path),
            **metrics,
        }
        rows.append(row)
    return rows


__all__ = [
    "RunRecord",
    "ScriptRecord",
    "requirement_slug",
    "load_jsonl",
    "parse_script_filename",
    "load_run_records",
    "latest_run_by_path",
    "load_requirement_texts",
    "discover_scripts",
    "load_script",
    "group_by_requirement",
    "variant_names",
    "find_record",
    "npc_names",
    "event_titles",
    "overview_rows",
]
