"""Trigger Stage 2 generation from a caller, not just the CLI scripts --
the data/threading layer behind the Stage 3 UI's "生成" mode (ui/app.py).

Deliberately streamlit-free, exactly like review.py (see that module's
docstring and tests/test_review.py's mechanical enforcement) -- one more
reason the Stage 3 "臨時駕駛艙" can be swapped for a Tauri desktop app later
without rewriting anything in src/.

Two things this module is careful about:

1. `preflight()` (moved here verbatim from scripts/generate_script.py, which
   now just imports it) must stay separate from `generate()` -- preflight
   treats LLM_BACKEND=fake as a blocking problem (real generation shouldn't
   run against canned output by accident), but the whole test suite for this
   module runs generate() under LLM_BACKEND=fake. If generate() called
   preflight() internally, no offline test could ever call it.
2. A real run takes 126-240s and can fail mid-run (see crew/pipeline.py's
   docstring). GenerationJob runs one in a background thread so a caller
   (the Streamlit UI) can show a live elapsed clock and a working cancel
   button instead of freezing for minutes -- CrewAI's step_callback never
   fires for our toolless agents (see crew/pipeline.py's StepEvent
   docstring), so a blocking call could only ever repaint ~3 times over
   those minutes, not tick a clock.

UI-triggered scripts are written under out/eval/ alongside
scripts/eval_generation.py's, but with a "ui-" variant prefix and to a
separate out/generation_runs_ui.jsonl -- config.RUN_LOG_GLOB
("generation_runs*.jsonl") already globs it, so review.py's discovery picks
up UI runs automatically without any changes there, while keeping them out
of the eval harness's A/B aggregate.
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import catalog, config, embedding, estimate, length, pricing
from .crew import scene_metrics
from .crew.metrics import script_metrics
from .crew.orchestrator import _SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION
from .crew.orchestrator import (
    load_beat_sheet,
    load_pending_scenes,
    load_scene_context,
    load_scene_metrics,
    plan_batches,
    run_layered,
)
from .crew.pipeline import MAX_REPAIR_ATTEMPTS, PipelineError, RunReport, StepEvent
from .crew.pipeline import run_pipeline_with_report as _run_pipeline_with_report
from .llm import ModelChoice
from .retrieval import CollectionNotFoundError, get_query_collection
from .review import parse_script_filename, requirement_slug, role_keys_for_mode
from .schema import Event, Script

DEFAULT_VARIANTS_FILE = config.PROJECT_ROOT / "eval" / "model_variants.json"
UI_RUN_LOG = config.OUT_DIR / "generation_runs_ui.jsonl"


class GenerationBusyError(RuntimeError):
    """Raised by generate()/GenerationJob.start() when another generation is
    already running in this process. Only one run at a time is supported --
    crew/tools.py's retrieval-stats counter is a rebindable module global,
    and crewai itself keeps a process-wide event bus, so two concurrent
    crews would corrupt each other's bookkeeping."""


class GenerationCancelled(RuntimeError):
    """Raised (internally, via on_step, or via a "layered" run's `gate`
    callback while parked awaiting batch confirmation) to unwind a run whose
    GenerationJob was cancelled. Surfaces to callers as
    JobSnapshot(status="cancelled"), not as a failed GenerationResult.

    `row`, if set by generate() before re-raising, is the JSONL row it
    still managed to write for whatever partial RunReport was recovered via
    run_layered()'s `on_report` hook -- same "attach whatever was gathered"
    convention as PipelineError.report. `None` when generate() had no
    RunReport to price at all (e.g. cancelled before the pipeline started)."""

    row: dict[str, Any] | None = None


# Guards against two concurrent generate() calls in one process (e.g. two
# browser tabs hitting the same Streamlit server). Non-blocking acquire --
# a second caller gets GenerationBusyError immediately rather than queuing.
_run_lock = threading.Lock()


def is_running() -> bool:
    """True if a generate()/GenerationJob run currently holds _run_lock."""
    acquired = _run_lock.acquire(blocking=False)
    if acquired:
        _run_lock.release()
        return False
    return True


@dataclass(frozen=True)
class Variant:
    """One named model split -- the same shape as a row in
    eval/model_variants.json, loaded here so the UI's variant picker and
    scripts/eval_generation.py share one representation.

    `extractor`/`beat_expander`/`scene_writer` are layered-only roles, kept
    optional (default "") for the same reason session_doc_max_tokens is
    optional below: most existing eval/model_variants.json entries only ever
    ran in "legacy" mode, so requiring them would be a breaking schema
    change. `to_model_choice()` falls back to writer/dialogue (llm.py's own
    ModelChoice class-default convention: extractor/beat_expander <- writer,
    scene_writer <- dialogue) when left blank -- this is also the fix for
    the bug that motivated adding these fields at all: before this, a
    layered-mode variant's writer/dialogue/proof model ids were read, but
    ModelChoice's extractor/beat_expander/scene_writer defaults are class
    attributes bound to config.LLM_MODEL_WRITER/_DIALOGUE at import time, so
    a variant's chosen models never actually reached layered-mode agents."""

    name: str = ""
    note: str = ""
    writer: str = ""
    dialogue: str = ""
    proof: str = ""
    extractor: str = ""
    beat_expander: str = ""
    scene_writer: str = ""
    # Quality-regression knob: forwarded to run_layered()'s
    # session_doc_max_tokens (None = config default, 0 = never trim -- see
    # crew/context_builder.py::build_session_document()'s docstring). Kept
    # last/optional so eval/model_variants.json's existing entries (and any
    # positional Variant(...) construction) are unaffected.
    session_doc_max_tokens: int | None = None
    # Script-length target (config.SCRIPT_LENGTH / --script-length; see
    # crew/tasks.py::_LENGTH_TARGETS). None = fall back to
    # config.SCRIPT_LENGTH, same three-level resolution as
    # session_doc_max_tokens above (see generate()'s docstring).
    script_length: str | None = None
    # Whether this variant's dialogue/scene_writer agent gets
    # wuxia_corpus_search at all. None = fall back to
    # config.RETRIEVAL_ENABLED, same three-level resolution as
    # script_length above.
    use_retrieval: bool | None = None
    # UI picker visibility only -- eval_generation.py/load_variants() still
    # loads and can run every row regardless of this flag (see
    # ui/app.py:273's filter). Lets eval/model_variants.json keep an
    # unreliable/expensive variant (e.g. long-cheap/long-mimo, see their
    # notes) reproducible for the CLI harness without it cluttering the
    # browser's variant dropdown.
    ui_visible: bool = True
    # Global reasoning-effort setting (config.REASONING_EFFORT /
    # llm.ModelChoice.reasoning_effort). None = fall back to
    # config.REASONING_EFFORT, same three-level resolution as
    # session_doc_max_tokens/script_length above -- appended last per that
    # same convention.
    reasoning_effort: str | None = None

    def to_model_choice(self) -> ModelChoice:
        return ModelChoice(
            writer=self.writer,
            dialogue=self.dialogue,
            proof=self.proof,
            extractor=self.extractor or self.writer,
            beat_expander=self.beat_expander or self.writer,
            scene_writer=self.scene_writer or self.dialogue,
            reasoning_effort=self.reasoning_effort or config.REASONING_EFFORT,
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Variant:
        return cls(
            name=row.get("name", ""),
            note=row.get("note", ""),
            writer=row.get("writer", ""),
            dialogue=row.get("dialogue", ""),
            proof=row.get("proof", ""),
            extractor=row.get("extractor", ""),
            beat_expander=row.get("beat_expander", ""),
            scene_writer=row.get("scene_writer", ""),
            session_doc_max_tokens=row.get("session_doc_max_tokens"),
            script_length=row.get("script_length"),
            use_retrieval=row.get("use_retrieval"),
            ui_visible=row.get("ui_visible", True),
            reasoning_effort=row.get("reasoning_effort"),
        )


def load_variants(path: Path = DEFAULT_VARIANTS_FILE) -> list[Variant]:
    """Parse eval/model_variants.json into Variant objects. Raises if the
    file is missing/malformed -- unlike review.py's loaders, there's no
    silent-empty fallback here since a UI picker with zero variants is a
    configuration bug worth surfacing, not a normal empty state."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Variant.from_dict(row) for row in rows]


def preflight(check_index: bool = True, check_embedding: bool = True) -> list[str]:
    """Checks worth running before spending a single token. Returns a list
    of human-readable problems (empty = clear to run).

    Moved here (verbatim logic) from scripts/generate_script.py so both the
    CLI and the Stage 3 UI share one implementation; that script now just
    imports this function. `check_index=False` skips the Chroma probe --
    useful for offline tests, which have no index and don't care about it.
    `check_embedding=False` skips the local bge-m3/torch-version probe --
    both are also skipped by callers that already know retrieval is disabled
    for this run (see ui/app.py's 不檢索語料庫 checkbox / Variant.use_retrieval),
    since a run with retrieval off never touches Chroma or the embedding
    model at all.
    """
    problems = []

    if config.LLM_BACKEND == "fake":
        problems.append(
            "LLM_BACKEND=fake -- this will produce the same canned output "
            "tests/test_crew_pipeline.py uses, not real generation. Set "
            "LLM_BACKEND=openrouter in .env for a real run."
        )
    elif config.LLM_BACKEND == "openrouter":
        try:
            config.require_openrouter_key()
        except RuntimeError as exc:
            problems.append(str(exc))
    else:
        problems.append(f"Unknown LLM_BACKEND={config.LLM_BACKEND!r}.")

    if check_embedding:
        # Catches the "streamlit resolved to the wrong (system/anaconda)
        # Python interpreter, whose torch is too old for bge-m3" failure
        # mode *before* a run starts, instead of only discovering it via
        # crew/tools.py's WuxiaRetrievalTool silently degrading every
        # retrieval call into a "檢索失敗" text message for the whole run
        # (see that module's docstring on why it never raises).
        embed_problem = embedding.check_backend_env()
        if embed_problem:
            problems.append(embed_problem)

    if check_index:
        try:
            get_query_collection()
        except CollectionNotFoundError as exc:
            problems.append(
                f"No Chroma index found -- the 對話 agent's wuxia_corpus_search tool "
                f"will have nothing to retrieve from: {exc}"
            )

    return problems


def estimate_for_form(
    variant: Variant,
    *,
    pipeline_mode: str | None = None,
    script_length: str | None = None,
    use_retrieval: bool | None = None,
    prices: dict[str, pricing.ModelPrice] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> estimate.RunEstimate:
    """Pre-run cost/time estimate for the UI's 生成 mode form, before
    "開始生成" is even clicked -- the same three-level resolution (explicit
    arg > variant field > config default) generate()/GenerationJob already
    use for script_length/use_retrieval, so the estimate reflects exactly
    what a real run with these settings would resolve to. Thin wrapper over
    bixiascribe.estimate.estimate_run() -- no checkpoint/disk state to read
    yet since nothing has been generated, so this is always basis in
    {"history_mode_length", "history_mode", "prior", "unknown_price"}, never
    "measured_run" (see GenerationJob.estimate() for that, once a run is
    actually in progress).

    `prices`/`history`, if given, are passed straight through to
    estimate_run() instead of it re-reading eval/model_prices.json (no
    cache) / out/generation_runs*.jsonl on every call -- callers making
    several calls in a row (e.g. ui/app.py's per-preset comparison table)
    should load these once and hoist them in."""
    mode = (pipeline_mode or config.PIPELINE_MODE).strip().lower()
    resolved_length = script_length or variant.script_length or config.SCRIPT_LENGTH
    models_obj = variant.to_model_choice()
    if mode == "layered":
        models = {
            "extractor": models_obj.extractor,
            "beat_expander": models_obj.beat_expander,
            "scene_writer": models_obj.scene_writer,
        }
    else:
        models = {
            "writer": models_obj.writer,
            "dialogue": models_obj.dialogue,
            "proof": models_obj.proof,
        }
    return estimate.estimate_run(
        pipeline_mode=mode,
        script_length=resolved_length,
        models=models,
        scene_concurrency=config.SCENE_CONCURRENCY,
        prices=prices,
        history=history,
    )


def ui_variant_name(base: str) -> str:
    """"baseline" -> "ui-baseline". Prefixing keeps a UI-triggered run from
    ever landing on the same out/eval/ filename as an eval_generation.py
    run of the same variant+requirement, and strips "__" so the name stays
    safe as review.parse_script_filename's variant/slug separator (variant
    names in eval/model_variants.json never contain "__", but an ad-hoc
    "自訂" name typed into the UI might)."""
    return f"ui-{base}".replace("__", "-")


def next_rep(scripts_dir: Path, variant_name: str, slug: str) -> int:
    """The smallest rep not already used by an out/eval/*.json file for this
    (variant_name, slug) pair -- so repeated UI runs of the same requirement
    never clobber an earlier one (unlike the eval harness's explicit
    `--repeat`, where overwriting rep 0 on re-invocation is deliberate)."""
    if not scripts_dir.is_dir():
        return 0
    used = set()
    for file_path in scripts_dir.glob("*.json"):
        try:
            v, s, rep = parse_script_filename(file_path.name)
        except ValueError:
            continue
        if v == variant_name and s == slug:
            used.add(rep)
    rep = 0
    while rep in used:
        rep += 1
    return rep


def script_path_for(
    requirement: str,
    variant_name: str,
    rep: int = 0,
    scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
) -> Path:
    """Same naming convention as scripts/eval_generation.py's `_run_one`:
    {variant}__{slug}[__rep{N}].json."""
    slug = requirement_slug(requirement)
    suffix = f"__rep{rep}" if rep > 0 else ""
    return scripts_dir / f"{variant_name}__{slug}{suffix}.json"


_REPORT_ROLE_FIELD = {
    "writer": "model_writer",
    "dialogue": "model_dialogue",
    "proof": "model_proof",
    "extractor": "model_extractor",
    "beat_expander": "model_beat_expander",
    "scene_writer": "model_scene_writer",
}


def _cost_models(report: RunReport) -> dict[str, str]:
    """Role -> model id, whichever set report.mode actually used -- the
    shape pricing.estimate_cost()'s `models` argument wants. Built from
    review.role_keys_for_mode() so this role set can never diverge from
    review.run_role_models()'s display (see design.md's 實測四)."""
    return {
        role: getattr(report, _REPORT_ROLE_FIELD[role])
        for role in role_keys_for_mode(report.mode)
    }


def build_run_row(
    variant_name: str,
    report: RunReport | None,
    script: Script | None = None,
    error: str | None = None,
    script_path: Path | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """Same row shape as scripts/eval_generation.py's `_run_one` produces,
    so review.RunRecord.from_row (and therefore ui/app.py's _render_run_meta)
    works unchanged regardless of which harness wrote the row.

    Also computes cost_usd/cost_basis (pricing.estimate_cost(), against
    whatever token_usage/token_usage_by_role the report carries) and, when a
    script is available, the usd_per_event/usd_per_dialogue_line/
    usd_per_1k_dialogue_chars quality-unit-cost trio (pricing.
    quality_unit_costs()) -- see pricing.py's module docstring for why both
    exist rather than just a total. A row with no report or a report with no
    priceable token_usage still gets cost_usd=None/cost_basis="unknown_price"
    rather than missing keys, so every row (including a PipelineError's
    partial report) has the same shape for review.py/ui/app.py to read."""
    row: dict[str, Any] = {
        "variant": variant_name,
        "ts": ts if ts is not None else time.time(),
        "ok": error is None,
        "error": error,
    }
    cost_usd: float | None = None
    cost_basis = "unknown_price"
    if report is not None:
        row.update(report.to_dict())
        cost_usd, cost_basis = pricing.estimate_cost(
            report.token_usage,
            _cost_models(report),
            token_usage_by_role=report.token_usage_by_role,
        )
    row["cost_usd"] = cost_usd
    row["cost_basis"] = cost_basis
    if script is not None:
        metrics = script_metrics(script)
        row.update(metrics)
        row.update(pricing.quality_unit_costs(cost_usd, metrics))
    if script_path is not None:
        row["script_path"] = str(script_path)
    return row


def append_run_row(row: dict[str, Any], jsonl_path: Path = UI_RUN_LOG) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


@dataclass(frozen=True)
class GenerationResult:
    """The outcome of one generate() call -- both the success and failure
    cases carry `row` (the JSONL row, written or not depending on
    `jsonl_path`) so a caller always has something to render, even for a
    PipelineError with a partial RunReport."""

    ok: bool = False
    variant: str = ""
    requirement: str = ""
    script: Script | None = None
    report: RunReport | None = None
    error: str = ""
    script_path: Path | None = None
    rep: int = 0
    row: dict[str, Any] = field(default_factory=dict)
    # Empty for the legacy pipeline, the .bixia_state/<run_id>/
    # checkpoint directory's name for a "layered" run -- what a caller
    # would pass back to orchestrator.run_layered(run_id=...) to resume.
    run_id: str = ""


def generate(
    requirement: str,
    variant: Variant | None = None,
    *,
    variant_name: str = "",
    rep: int | None = None,
    verbose: bool = False,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    on_step: Callable[[StepEvent], None] | None = None,
    scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
    jsonl_path: Path | None = UI_RUN_LOG,
    cancel_check: Callable[[], bool] | None = None,
    pipeline_mode: str | None = None,
    run_id: str | None = None,
    gate: Callable[[list[str]], bool] | None = None,
    session_doc_max_tokens: int | None = None,
    script_length: str | None = None,
    use_retrieval: bool | None = None,
    reasoning_effort: str | None = None,
) -> GenerationResult:
    """Run one generation and persist the result, sharing exactly the
    row/filename conventions scripts/eval_generation.py uses.

    `variant` (with its own .name) is the normal path; `variant_name`
    overrides just the name when `variant` is given (or names an
    already-prefixed ad-hoc variant when it isn't -- e.g.
    scripts/eval_generation.py's callers, which pass their own ModelChoice
    via `variant` and want the *unprefixed* eval variant name preserved).

    `rep=None` (the UI's case) auto-picks the next free rep for this
    (variant, slug) so repeated runs never overwrite each other; passing an
    explicit int (the eval harness's case) preserves today's
    deliberate-overwrite-on-repeat semantics.

    `jsonl_path=None` returns the row without appending anywhere, so a
    caller that owns its own open file handle (run_matrix's crash-resilient
    append-with-flush loop) keeps doing exactly that instead of two writers
    racing on the same file.

    Only catches PipelineError, matching scripts/eval_generation.py's
    current crash-vs-row semantics: a PipelineError becomes ok=False with
    whatever partial RunReport it carried; any other exception propagates.

    Does NOT call preflight() -- see this module's docstring for why.

    `pipeline_mode` (default config.PIPELINE_MODE) selects
    "legacy" (crew/pipeline.py's run_pipeline_with_report(), unchanged) or
    "layered" (crew/orchestrator.py's run_layered(), checkpointed
    + batched). `run_id`/`gate` are layered-only: `run_id` names the
    `.bixia_state/<run_id>/` checkpoint directory to use or resume (a fresh
    one is generated if omitted), and `gate` is forwarded straight to
    run_layered()'s batch-confirmation callback (see that function's
    docstring) -- both are silently ignored in "legacy" mode, which has no
    checkpoint directory or batch concept.

    `session_doc_max_tokens` (layered-only) overrides how far each
    scene's SessionDocument is trimmed: an explicit argument here wins over
    `variant.session_doc_max_tokens`, which in turn wins over `None` (fall
    back to config.SESSION_DOC_MAX_TOKENS). Forwarded to run_layered() only
    in "layered" mode; silently ignored in "legacy" mode, same as `gate`.

    `script_length` resolves the same three-level way: an explicit argument
    here wins over `variant.script_length`, which wins over `None` (fall
    back to config.SCRIPT_LENGTH, i.e. "short" -- today's behavior).
    Forwarded to both "legacy" and "layered" pipelines (unlike
    session_doc_max_tokens, which is layered-only) since crew/tasks.py's
    length targets apply to both.

    `use_retrieval` resolves the same three-level way: an explicit argument
    here wins over `variant.use_retrieval`, which wins over `None` (fall
    back to config.RETRIEVAL_ENABLED). Forwarded to both pipelines.

    `reasoning_effort` resolves the same three-level way: an explicit
    argument here wins over `variant.reasoning_effort`, which wins over
    config.REASONING_EFFORT. Canonicalized via
    catalog.normalize_reasoning_effort() and forwarded to every agent role
    via ModelChoice.reasoning_effort (see llm.py::build_llm()), then
    recorded on RunReport.reasoning_effort so runs at different effort
    levels are comparable.
    """
    variant = variant or Variant()
    name = variant_name or variant.name
    mode = (pipeline_mode or config.PIPELINE_MODE).strip().lower()
    resolved_reasoning_effort = catalog.normalize_reasoning_effort(
        reasoning_effort if reasoning_effort is not None else variant.reasoning_effort
    )
    models = dataclasses.replace(
        variant.to_model_choice(), reasoning_effort=resolved_reasoning_effort
    )
    resolved_session_doc_max_tokens = (
        session_doc_max_tokens
        if session_doc_max_tokens is not None
        else variant.session_doc_max_tokens
    )
    resolved_use_retrieval = (
        use_retrieval if use_retrieval is not None else variant.use_retrieval
    )
    raw_script_length = (
        script_length
        if script_length is not None
        else (variant.script_length if variant.script_length is not None else config.SCRIPT_LENGTH)
    )
    # Canonicalize (fully-resolved preset name, or a fully-resolved
    # custom:... string with every field filled in) so RunReport.script_length
    # / the JSONL row is self-describing without cross-referencing whatever
    # partial input (e.g. "custom:events=20") was configured at run time.
    resolved_script_length = length.parse_length_spec(raw_script_length).canonical

    if not _run_lock.acquire(blocking=False):
        raise GenerationBusyError("已有一個生成正在執行，請稍候再試。")
    try:
        resolved_rep = rep if rep is not None else next_rep(
            scripts_dir, name, requirement_slug(requirement)
        )
        out_path = script_path_for(requirement, name, resolved_rep, scripts_dir)
        resolved_run_id = run_id or f"{int(time.time())}-{requirement_slug(requirement)}"

        # Captured via run_layered()'s on_report hook so a layered run that
        # exits by raising something other than PipelineError (the gate
        # callback unwinding on GenerationCancelled, or any other
        # unconverted exception -- see CLAUDE.md's "Known limitations" for
        # a real example) still leaves a priceable partial RunReport behind.
        # Legacy mode has no equivalent hook: its cancellation/crash paths
        # already surface as PipelineError via crew.kickoff()'s own
        # try/except, so there's already a row-writing path for them.
        latest_report: RunReport | None = None

        def _capture_report(r: RunReport) -> None:
            nonlocal latest_report
            latest_report = r

        try:
            if mode == "layered":
                script, report = run_layered(
                    requirement,
                    run_id=resolved_run_id,
                    models=models,
                    verbose=verbose,
                    on_step=on_step,
                    max_repair_attempts=max_repair_attempts,
                    gate=gate,
                    session_doc_max_tokens=resolved_session_doc_max_tokens,
                    script_length=resolved_script_length,
                    use_retrieval=resolved_use_retrieval,
                    on_report=_capture_report,
                )
            else:
                script, report = _run_pipeline_with_report(
                    requirement,
                    verbose=verbose,
                    max_repair_attempts=max_repair_attempts,
                    models=models,
                    on_step=on_step,
                    script_length=resolved_script_length,
                    use_retrieval=resolved_use_retrieval,
                )
        except PipelineError as exc:
            row = build_run_row(name, exc.report, error=str(exc))
            if jsonl_path is not None:
                append_run_row(row, jsonl_path)
            return GenerationResult(
                ok=False,
                variant=name,
                requirement=requirement,
                error=str(exc),
                rep=resolved_rep,
                row=row,
                run_id=resolved_run_id if mode == "layered" else "",
            )
        except GenerationCancelled as exc:
            # A "layered" run cancelled while parked in `gate` (e.g. the UI's
            # 取消 button during batch confirmation) -- write a row for
            # whatever was spent before the cancellation, then re-raise
            # unchanged so GenerationJob._run() still sees GenerationCancelled
            # and reports status="cancelled", not "failed".
            row = build_run_row(name, latest_report, error="使用者取消本次生成")
            if jsonl_path is not None:
                append_run_row(row, jsonl_path)
            exc.row = row
            raise
        except Exception as exc:  # noqa: BLE001 -- record spend, then re-raise as-is
            # Anything run_layered()/run_pipeline_with_report() didn't
            # convert to PipelineError on its own (e.g. the OpenRouter
            # choices=None crash CLAUDE.md's "Known limitations" documents
            # for layered mode) -- still worth a row so the tokens already
            # spent aren't silently lost, but the exception's own type/
            # message is preserved for whatever already handles it upstream.
            row = build_run_row(name, latest_report, error=str(exc))
            if jsonl_path is not None:
                append_run_row(row, jsonl_path)
            raise

        scripts_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(script.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
        row = build_run_row(name, report, script=script, script_path=out_path)
        if jsonl_path is not None:
            append_run_row(row, jsonl_path)

        return GenerationResult(
            ok=True,
            variant=name,
            requirement=requirement,
            script=script,
            report=report,
            script_path=out_path,
            rep=resolved_rep,
            row=row,
            run_id=resolved_run_id if mode == "layered" else "",
        )
    finally:
        _run_lock.release()


@dataclass(frozen=True)
class JobSnapshot:
    """Immutable point-in-time view of a GenerationJob, handed to a poller
    (e.g. ui/app.py's st.fragment) via GenerationJob.snapshot(). Never
    mutated in place -- each snapshot() call builds a fresh one under the
    job's lock, so a caller holding an old snapshot never sees it change
    underneath it."""

    status: str = "pending"  # pending | running | done | failed | cancelled
    phase: str = ""
    phase_index: int = 0
    # 3 for "legacy" (the fixed 編劇/對話/校對 task count); 0 ("unknown step
    # count") for "layered", where the number of scene batches depends on
    # the beat sheet a run itself produces -- see GenerationJob's docstring.
    # A caller dividing phase_index/phase_total for a progress bar must
    # guard against phase_total == 0.
    phase_total: int = 3
    elapsed_s: float = 0.0
    log: tuple[str, ...] = ()
    result: GenerationResult | None = None
    error: str = ""
    # True while a "layered" run has staged a scene batch
    # (pending_scene_<id>.json files) and is blocked waiting for
    # GenerationJob.confirm_batch()/reject_batch(). Always False in
    # "legacy" mode, which has no batch concept.
    awaiting_confirmation: bool = False
    pending_scene_ids: tuple[str, ...] = ()
    # "" for "legacy"; the .bixia_state/<run_id>/ checkpoint directory name
    # for "layered", fixed for the job's whole lifetime (set at construction,
    # not just once the run finishes) so a caller can resume/inspect it
    # even while the job is still running.
    run_id: str = ""

    # --- layered live scene progress (all zero/empty in "legacy" mode,
    # which never opens a crew.scene_metrics.scene_scope()) ---
    # The same Chinese stage label as `phase` (拆書/排場/寫戲/校對), but
    # updated on a "phase" StepEvent (stage *start*) as well as "task"
    # (stage *end*) -- see _on_step()'s docstring for why `phase` itself
    # keeps its older "last completed task" meaning for backward compat.
    stage: str = ""
    # Committed-scene count for this run, seeded from a resumed run's
    # already-on-disk scene checkpoints. 0 until the beat sheet exists.
    scenes_done: int = 0
    scenes_total: int = 0
    # Beat ids with a scene_scope() currently open on some worker thread --
    # a tuple, not a scalar "current scene", because dispatch_batch() runs
    # a concurrency>1 pool; every historical beat sheet on this machine
    # happens to be a linear chain (width 1), but a future non-linear one
    # must not make this lie by only reporting one id.
    active_scene_ids: tuple[str, ...] = ()
    active_scene_elapsed_s: float = 0.0
    guardrail_retries: int = 0


_PHASE_LABELS = {1: "編劇・鐵筆生", 2: "對話・柳三娘", 3: "校對・青衫客"}
_LOG_CAP = 50


class GenerationJob:
    """Runs one generate() call on a background thread so a caller can poll
    progress (JobSnapshot) instead of blocking for the 126-240s a real run
    takes. The worker thread only ever mutates this job's own fields under
    self._lock -- never streamlit's session_state or st.* -- so a UI caller
    is responsible for its own polling/rerendering (see ui/app.py).

    `pipeline_mode` (default config.PIPELINE_MODE) selects "legacy" or
    "layered" -- see generate()'s docstring. Only "layered" jobs ever stage
    a batch/wait at confirm_batch()/reject_batch(); a "legacy" job's
    confirm_batch()/reject_batch() calls are harmless no-ops since
    JobSnapshot.awaiting_confirmation can never be True for one.

    The confirmation gate uses the same cooperative-boundary mechanism
    cancel() already relies on: _gate() blocks the worker thread on a
    threading.Event that only confirm_batch()/reject_batch()/cancel() ever
    set, so cancel() called while a batch is awaiting confirmation still
    wins -- it wakes the same wait with no decision recorded, which _gate()
    treats as a cancellation.
    """

    def __init__(
        self,
        requirement: str,
        variant: Variant,
        *,
        verbose: bool = False,
        scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
        jsonl_path: Path | None = UI_RUN_LOG,
        pipeline_mode: str | None = None,
        script_length: str | None = None,
        use_retrieval: bool | None = None,
        run_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._requirement = requirement
        self._variant = variant
        self._verbose = verbose
        self._scripts_dir = scripts_dir
        self._jsonl_path = jsonl_path
        self._mode = (pipeline_mode or config.PIPELINE_MODE).strip().lower()
        self._script_length = script_length
        self._use_retrieval = use_retrieval
        self._reasoning_effort = reasoning_effort
        # `run_id`, when given, resumes an existing .bixia_state/<run_id>/
        # checkpoint directory instead of always minting a fresh one -- see
        # run-resume spec. Only meaningful in "layered" mode, same as the
        # freshly-minted id below.
        self._run_id = run_id or (
            f"{int(time.time())}-{requirement_slug(requirement)}" if self._mode == "layered" else ""
        )

        self._lock = threading.Lock()
        self._status = "pending"
        self._phase = ""
        self._stage = ""
        self._phase_index = 0
        self._log: list[str] = []
        self._result: GenerationResult | None = None
        self._error = ""
        self._start: float | None = None
        self._cancel_requested = False
        self._thread: threading.Thread | None = None

        self._awaiting_confirmation = False
        self._pending_scene_ids: tuple[str, ...] = ()
        self._gate_event = threading.Event()
        self._gate_decision: bool | None = None

        # estimate() cache -- see that method's docstring for why: it's
        # called from ui/app.py's 1-second-interval progress fragment, and
        # reading beats.json/scene_meta_*.json sidecars off disk on every
        # tick would mean a stat() burst once a second for the whole run.
        self._estimate_cache: estimate.RunEstimate | None = None
        self._estimate_cache_ts: float = 0.0

        # scenes_total() cache -- same 1-second-poll pressure as the
        # estimate cache above, plus this one is a permanent cache once
        # non-zero: a run's beat count never changes after the beat sheet
        # is written, so there's no reason to keep re-reading beats.json
        # for the rest of the run.
        self._scenes_total_cache: int = 0
        self._scenes_resumed_cache: int = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GenerationJob.start() called more than once.")
        self._start = time.monotonic()
        with self._lock:
            self._status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _on_step(self, event: StepEvent) -> None:
        with self._lock:
            if self._cancel_requested:
                raise GenerationCancelled("使用者取消了本次生成。")
            if self._mode == "layered":
                # run_layered() emits a "phase" StepEvent at each stage's
                # *start* and a "task" one at its *end* (crew/orchestrator.py),
                # both carrying a readable Chinese stage label as `role`
                # (拆書/排場/寫戲/校對). The displayed phase must follow both,
                # not just "task" -- otherwise 拆書(35s)/排場(45s)/each scene
                # (~227s) shows the *previous* stage's name for its entire
                # duration, which was the original "no feedback until the
                # final error" complaint. phase_index still only counts
                # completed tasks -- there's no fixed total to cap it
                # against like legacy's 3 (the batch count varies per run).
                if event.role:
                    self._phase = event.role
                    self._stage = event.role
                if event.kind == "task":
                    self._phase_index += 1
            else:
                if event.kind == "task":
                    self._phase_index = min(self._phase_index + 1, 3)
                    self._phase = _PHASE_LABELS.get(self._phase_index, "")
            label = event.role or event.text
            self._log.append(f"{label}：{event.text}" if event.role else event.text)
            self._log = self._log[-_LOG_CAP:]

    def _gate(self, pending_ids: list[str]) -> bool:
        """Passed as run_layered()'s `gate` for a "layered" job: blocks this
        worker thread until confirm_batch()/reject_batch()/cancel() is
        called from another thread (e.g. the UI's main thread handling a
        button click)."""
        with self._lock:
            if self._cancel_requested:
                raise GenerationCancelled("使用者取消了本次生成。")
            self._awaiting_confirmation = True
            self._pending_scene_ids = tuple(pending_ids)
            self._gate_event.clear()
            self._gate_decision = None
        self._gate_event.wait()
        with self._lock:
            self._awaiting_confirmation = False
            decision = self._gate_decision
        if decision is None:  # woken by cancel() with no decision recorded
            raise GenerationCancelled("使用者取消了本次生成。")
        return decision

    def _run(self) -> None:
        try:
            result = generate(
                self._requirement,
                self._variant,
                verbose=self._verbose,
                on_step=self._on_step,
                scripts_dir=self._scripts_dir,
                jsonl_path=self._jsonl_path,
                pipeline_mode=self._mode,
                run_id=self._run_id or None,
                gate=self._gate if self._mode == "layered" else None,
                script_length=self._script_length,
                use_retrieval=self._use_retrieval,
                reasoning_effort=self._reasoning_effort,
            )
        except GenerationCancelled as exc:
            with self._lock:
                self._status = "cancelled"
                # generate() attaches whatever row it managed to write for a
                # "layered" run cancelled at the batch-confirmation gate
                # (see its own GenerationCancelled handler) -- carry it on
                # a GenerationResult so the UI's cancelled-state panel has
                # something to render the already-spent cost from, same
                # shape as the failed-run panel already reads.
                self._result = GenerationResult(
                    ok=False,
                    variant=self._variant.name,
                    requirement=self._requirement,
                    error=str(exc),
                    row=exc.row or {},
                    run_id=self._run_id,
                )
            return
        except Exception as exc:  # noqa: BLE001 -- surfaced via snapshot(), not raised
            with self._lock:
                self._status = "failed"
                self._error = str(exc)
            return

        with self._lock:
            self._result = result
            self._status = "done" if result.ok else "failed"
            if not result.ok:
                self._error = result.error

    def cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True
        # Wake a worker thread parked in _gate() so cancellation isn't stuck
        # waiting on a confirm_batch()/reject_batch() call that may never come.
        self._gate_event.set()

    def confirm_batch(self) -> None:
        """Approve the currently staged batch (JobSnapshot.pending_scene_ids)
        and let the worker thread continue. A no-op if nothing is staged."""
        with self._lock:
            if not self._awaiting_confirmation:
                return
            self._gate_decision = True
        self._gate_event.set()

    def reject_batch(self) -> None:
        """Discard the currently staged batch -- the worker regenerates it.
        A no-op if nothing is staged."""
        with self._lock:
            if not self._awaiting_confirmation:
                return
            self._gate_decision = False
        self._gate_event.set()

    def pending_scenes(self) -> list[Event]:
        """The full Event content of the batch currently staged (empty
        outside "layered" mode or when nothing is staged) -- for a caller
        (ui/app.py's batch-confirmation panel) that wants to show the actual
        台詞/分支 instead of just JobSnapshot.pending_scene_ids. Reads
        checkpoint files directly, same as the job's own gate/cancel
        mechanism reads run_id off self -- no lock needed since this is a
        disk read, not a mutation of the job's in-memory fields."""
        if self._mode != "layered" or not self._run_id:
            return []
        return load_pending_scenes(self._run_id)

    def scene_context(self) -> tuple[dict[str, str], dict[str, str]]:
        """(npc_id -> name, event_id -> title) for this job's run so far --
        see orchestrator.load_scene_context()'s docstring. Empty dicts
        outside "layered" mode."""
        if self._mode != "layered" or not self._run_id:
            return {}, {}
        return load_scene_context(self._run_id)

    def estimate(self, *, cache_seconds: float = 5.0) -> estimate.RunEstimate:
        """Live cost/time estimate for this job -- what ui/app.py's progress
        fragment shows next to the elapsed-time clock, and what the
        batch-confirmation panel shows as "how much more if I confirm this".

        Outside "layered" mode, or before this run's beats.json checkpoint
        exists yet, falls back to estimate_for_form()'s pre-run guess (no
        real progress to measure from yet). Once real scenes have committed
        (crew/orchestrator.py::load_scene_metrics()), switches to
        estimate.estimate_remaining()'s basis="measured_run" -- this run's
        own actually-observed per-scene cost/time, not a cross-run average.

        Cached for `cache_seconds` (default 5s): this is called from
        ui/app.py's @st.fragment(run_every=1.0) progress panel, and without
        a cache that would mean re-reading beats.json + every
        scene_meta_*.json sidecar off disk once a second for the run's
        whole duration."""
        now = time.monotonic()
        with self._lock:
            if self._estimate_cache is not None and now - self._estimate_cache_ts < cache_seconds:
                return self._estimate_cache

        fallback = estimate_for_form(
            self._variant,
            pipeline_mode=self._mode,
            script_length=self._script_length,
            use_retrieval=self._use_retrieval,
        )
        if self._mode != "layered" or not self._run_id:
            result = fallback
        else:
            beat_sheet = load_beat_sheet(self._run_id)
            if beat_sheet is None:
                result = fallback
            else:
                completed_metrics = load_scene_metrics(self._run_id)
                completed_ids = {m["beat_id"] for m in completed_metrics if m.get("beat_id")}
                remaining_beats = [b for b in beat_sheet.beats if b.id not in completed_ids]
                remaining_batch_widths = [len(batch) for batch in plan_batches(remaining_beats)]
                scene_model = self._variant.to_model_choice().scene_writer
                result = estimate.estimate_remaining(
                    total_scenes=len(beat_sheet.beats),
                    completed_metrics=completed_metrics,
                    remaining_batch_widths=remaining_batch_widths,
                    scene_concurrency=config.SCENE_CONCURRENCY,
                    scene_model=scene_model,
                    fallback=fallback,
                )

        with self._lock:
            self._estimate_cache = result
            self._estimate_cache_ts = now
        return result

    @property
    def done(self) -> bool:
        with self._lock:
            return self._status in ("done", "failed", "cancelled")

    def _scenes_total(self) -> tuple[int, int]:
        """(total beats, already-completed-before-this-process-started
        beats) for this layered run, or (0, 0) before beats.json exists.
        Permanently cached once non-zero (self._lock-guarded) -- a run's
        beat count never changes after the beat sheet is written, and
        snapshot() is polled once a second by ui/app.py's progress
        fragment, so this must not stat/parse beats.json on every tick.

        The "resumed" count matters because crew.scene_metrics's live
        accumulator only knows about scopes opened by *this* process --
        a resumed run's already-committed scenes never open a fresh scope,
        so scenes_done would otherwise undercount and the progress bar
        would jump backward on resume."""
        if self._mode != "layered" or not self._run_id:
            return 0, 0
        with self._lock:
            if self._scenes_total_cache:
                return self._scenes_total_cache, self._scenes_resumed_cache
        sheet = load_beat_sheet(self._run_id)
        if sheet is None:
            return 0, 0
        total = len(sheet.beats)
        resumed = len(load_scene_metrics(self._run_id))
        with self._lock:
            self._scenes_total_cache = total
            self._scenes_resumed_cache = resumed
        return total, resumed

    def snapshot(self) -> JobSnapshot:
        # Read the live per-scene accumulator (its own lock) BEFORE taking
        # self._lock -- self._lock is a plain, non-reentrant
        # threading.Lock, and _scenes_total() below also acquires it.
        scenes_done = 0
        active_ids: tuple[str, ...] = ()
        active_elapsed = 0.0
        retries = 0
        scenes_total = 0
        if self._mode == "layered":
            stats = scene_metrics.get_stats().scenes
            active = scene_metrics.active_scenes()
            active_ids = tuple(sorted(active))
            active_elapsed = max(active.values(), default=0.0)
            retries = sum(m.guardrail_retries for m in stats.values())
            scenes_total, resumed = self._scenes_total()
            # A scene counts as "done" once its scope has closed (whether
            # it ultimately committed or failed the run is about to end
            # anyway), plus whatever this run already had on disk before
            # this process's accumulator started tracking it.
            scenes_done = resumed + sum(1 for bid in stats if bid not in active)

        with self._lock:
            elapsed = time.monotonic() - self._start if self._start is not None else 0.0
            return JobSnapshot(
                status=self._status,
                phase=self._phase,
                phase_index=self._phase_index,
                phase_total=0 if self._mode == "layered" else 3,
                elapsed_s=elapsed,
                log=tuple(self._log),
                result=self._result,
                error=self._error,
                awaiting_confirmation=self._awaiting_confirmation,
                pending_scene_ids=self._pending_scene_ids,
                run_id=self._run_id,
                stage=self._stage,
                scenes_done=scenes_done,
                scenes_total=scenes_total,
                active_scene_ids=active_ids,
                active_scene_elapsed_s=active_elapsed,
                guardrail_retries=retries,
            )

    def join(self, timeout: float | None = None) -> JobSnapshot:
        """Block until the job reaches a terminal state -- for headless
        callers (tests) rather than the polling UI path."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self.snapshot()


__all__ = [
    "DEFAULT_VARIANTS_FILE",
    "UI_RUN_LOG",
    "CHECKPOINT_SCHEMA_VERSION",
    "GenerationBusyError",
    "GenerationCancelled",
    "Variant",
    "GenerationResult",
    "JobSnapshot",
    "GenerationJob",
    "load_variants",
    "preflight",
    "estimate_for_form",
    "ui_variant_name",
    "next_rep",
    "script_path_for",
    "build_run_row",
    "append_run_row",
    "is_running",
    "generate",
]
