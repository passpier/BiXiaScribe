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

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, embedding, length, pricing
from .crew.metrics import script_metrics
from .crew.orchestrator import load_pending_scenes, load_scene_context, run_layered
from .crew.pipeline import MAX_REPAIR_ATTEMPTS, PipelineError, RunReport, StepEvent
from .crew.pipeline import run_pipeline_with_report as _run_pipeline_with_report
from .llm import ModelChoice
from .retrieval import CollectionNotFoundError, get_query_collection
from .review import parse_script_filename, requirement_slug
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
    """Raised (internally, via on_step) to unwind a run whose GenerationJob
    was cancelled. Surfaces to callers as JobSnapshot(status="cancelled"),
    not as a failed GenerationResult."""


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

    def to_model_choice(self) -> ModelChoice:
        return ModelChoice(
            writer=self.writer,
            dialogue=self.dialogue,
            proof=self.proof,
            extractor=self.extractor or self.writer,
            beat_expander=self.beat_expander or self.writer,
            scene_writer=self.scene_writer or self.dialogue,
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


def _cost_models(report: RunReport) -> dict[str, str]:
    """Role -> model id, whichever set report.mode actually used -- the
    shape pricing.estimate_cost()'s `models` argument wants."""
    if report.mode == "layered":
        return {
            "extractor": report.model_extractor,
            "beat_expander": report.model_beat_expander,
            "scene_writer": report.model_scene_writer,
            "proof": report.model_proof,
        }
    return {
        "writer": report.model_writer,
        "dialogue": report.model_dialogue,
        "proof": report.model_proof,
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
    """
    variant = variant or Variant()
    name = variant_name or variant.name
    models = variant.to_model_choice()
    mode = (pipeline_mode or config.PIPELINE_MODE).strip().lower()
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
    ) -> None:
        self._requirement = requirement
        self._variant = variant
        self._verbose = verbose
        self._scripts_dir = scripts_dir
        self._jsonl_path = jsonl_path
        self._mode = (pipeline_mode or config.PIPELINE_MODE).strip().lower()
        self._script_length = script_length
        self._use_retrieval = use_retrieval
        self._run_id = (
            f"{int(time.time())}-{requirement_slug(requirement)}" if self._mode == "layered" else ""
        )

        self._lock = threading.Lock()
        self._status = "pending"
        self._phase = ""
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
            if event.kind == "task":
                if self._mode == "layered":
                    # crew/orchestrator.py's run_layered() already emits a
                    # readable Chinese stage label as `role` (拆書/排場/寫戲/
                    # 校對); the batch count varies per run, so there's no
                    # fixed total to cap phase_index against like legacy's 3.
                    self._phase_index += 1
                    self._phase = event.role or self._phase
                else:
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
            )
        except GenerationCancelled:
            with self._lock:
                self._status = "cancelled"
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

    def scene_context(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """(npc_id -> name, event_id -> title, quest_id -> name) for this
        job's run so far -- see orchestrator.load_scene_context()'s
        docstring. Empty dicts outside "layered" mode."""
        if self._mode != "layered" or not self._run_id:
            return {}, {}, {}
        return load_scene_context(self._run_id)

    @property
    def done(self) -> bool:
        with self._lock:
            return self._status in ("done", "failed", "cancelled")

    def snapshot(self) -> JobSnapshot:
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
    "GenerationBusyError",
    "GenerationCancelled",
    "Variant",
    "GenerationResult",
    "JobSnapshot",
    "GenerationJob",
    "load_variants",
    "preflight",
    "ui_variant_name",
    "next_rep",
    "script_path_for",
    "build_run_row",
    "append_run_row",
    "is_running",
    "generate",
]
