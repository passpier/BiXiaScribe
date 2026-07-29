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

from . import config
from .crew.metrics import script_metrics
from .crew.pipeline import MAX_REPAIR_ATTEMPTS, PipelineError, RunReport, StepEvent
from .crew.pipeline import run_pipeline_with_report as _run_pipeline_with_report
from .llm import ModelChoice
from .retrieval import CollectionNotFoundError, get_query_collection
from .review import parse_script_filename, requirement_slug
from .schema import Script

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
    """One named {writer, dialogue, proof} model split -- the same shape as
    a row in eval/model_variants.json, loaded here so the UI's variant
    picker and scripts/eval_generation.py share one representation."""

    name: str = ""
    note: str = ""
    writer: str = ""
    dialogue: str = ""
    proof: str = ""

    def to_model_choice(self) -> ModelChoice:
        return ModelChoice(writer=self.writer, dialogue=self.dialogue, proof=self.proof)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Variant:
        return cls(
            name=row.get("name", ""),
            note=row.get("note", ""),
            writer=row.get("writer", ""),
            dialogue=row.get("dialogue", ""),
            proof=row.get("proof", ""),
        )


def load_variants(path: Path = DEFAULT_VARIANTS_FILE) -> list[Variant]:
    """Parse eval/model_variants.json into Variant objects. Raises if the
    file is missing/malformed -- unlike review.py's loaders, there's no
    silent-empty fallback here since a UI picker with zero variants is a
    configuration bug worth surfacing, not a normal empty state."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Variant.from_dict(row) for row in rows]


def preflight(check_index: bool = True) -> list[str]:
    """Checks worth running before spending a single token. Returns a list
    of human-readable problems (empty = clear to run).

    Moved here (verbatim logic) from scripts/generate_script.py so both the
    CLI and the Stage 3 UI share one implementation; that script now just
    imports this function. `check_index=False` skips the Chroma probe --
    useful for offline tests, which have no index and don't care about it.
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
    works unchanged regardless of which harness wrote the row."""
    row: dict[str, Any] = {
        "variant": variant_name,
        "ts": ts if ts is not None else time.time(),
        "ok": error is None,
        "error": error,
    }
    if report is not None:
        row.update(report.to_dict())
    if script is not None:
        row.update(script_metrics(script))
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
    """
    variant = variant or Variant()
    name = variant_name or variant.name
    models = variant.to_model_choice()

    if not _run_lock.acquire(blocking=False):
        raise GenerationBusyError("已有一個生成正在執行，請稍候再試。")
    try:
        resolved_rep = rep if rep is not None else next_rep(
            scripts_dir, name, requirement_slug(requirement)
        )
        out_path = script_path_for(requirement, name, resolved_rep, scripts_dir)

        try:
            script, report = _run_pipeline_with_report(
                requirement,
                verbose=verbose,
                max_repair_attempts=max_repair_attempts,
                models=models,
                on_step=on_step,
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
    phase_total: int = 3
    elapsed_s: float = 0.0
    log: tuple[str, ...] = ()
    result: GenerationResult | None = None
    error: str = ""


_PHASE_LABELS = {1: "編劇・鐵筆生", 2: "對話・柳三娘", 3: "校對・青衫客"}
_LOG_CAP = 50


class GenerationJob:
    """Runs one generate() call on a background thread so a caller can poll
    progress (JobSnapshot) instead of blocking for the 126-240s a real run
    takes. The worker thread only ever mutates this job's own fields under
    self._lock -- never streamlit's session_state or st.* -- so a UI caller
    is responsible for its own polling/rerendering (see ui/app.py)."""

    def __init__(
        self,
        requirement: str,
        variant: Variant,
        *,
        verbose: bool = False,
        scripts_dir: Path = config.EVAL_SCRIPTS_DIR,
        jsonl_path: Path | None = UI_RUN_LOG,
    ) -> None:
        self._requirement = requirement
        self._variant = variant
        self._verbose = verbose
        self._scripts_dir = scripts_dir
        self._jsonl_path = jsonl_path

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
                self._phase_index = min(self._phase_index + 1, 3)
                self._phase = _PHASE_LABELS.get(self._phase_index, "")
            label = event.role or event.text
            self._log.append(f"{label}：{event.text}" if event.role else event.text)
            self._log = self._log[-_LOG_CAP:]

    def _run(self) -> None:
        try:
            result = generate(
                self._requirement,
                self._variant,
                verbose=self._verbose,
                on_step=self._on_step,
                scripts_dir=self._scripts_dir,
                jsonl_path=self._jsonl_path,
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
                phase_total=3,
                elapsed_s=elapsed,
                log=tuple(self._log),
                result=self._result,
                error=self._error,
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
