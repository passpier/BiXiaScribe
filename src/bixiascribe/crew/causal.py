"""Real-time causal consistency validation for the layered pipeline.

Before this module, the only cross-reference safety net was the final
proofread stage's schema.validate_references(), which checks npc_id/
next_event_id existence -- not whether one scene's outcome contradicts a
fact another scene already established (e.g. scene A kills off an NPC,
scene B has that same NPC alive and talking). validate_causal_graph()
(schema.py) only checks structural edge integrity (does from_id/to_id point
at a real node), which is a different, narrower check than what's needed
here.

This module is pure/offline (no I/O, no LLM, no crewai import) so it's
cheap to unit test in isolation, matching the style of
orchestrator.py::plan_batches(). It builds a CausalPlotGraph mechanically
from Event fields already produced by the legacy Event schema -- no schema
change, no extra prompt/token cost:

- PlotNode.preconditions  <- event.triggers[*].condition
- PlotNode.postconditions <- event.branches[*].effects
- PlotEdge (beat order)   <- Beat.causal_deps
- PlotEdge (branch flow)  <- Event.branches[*].next_event_id

The graph is always rebuilt from scratch from every committed scene rather
than mutated incrementally -- rebuilding is a cheap pure function, and it
means the confirm/reject/resume/corrupted-checkpoint paths never need their
own incremental-correctness logic, mirroring detect_stage()'s "always
re-derive from disk, never trust in-memory state" invariant.

Semantic conflict detection compares free-form Chinese condition/effect
strings via a conservative normalizer (parse_fact) plus a small curated
antonym table -- unparseable text yields no Fact and therefore never
produces a false-positive block. Only one consistency rule is implemented:
a candidate scene's precondition must not conflict with its nearest causal
ancestor's postcondition (see check_scene_consistency's docstring for why
postcondition-vs-postcondition isn't checked).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai import Task

from ..schema import Beat, BeatSheet, CausalPlotGraph, Event, PlotEdge, PlotNode

if TYPE_CHECKING:
    from crewai import Agent


# --- Fact normalization ----------------------------------------------------

# Relation operators that split a condition/effect string into subject and
# predicate unambiguously. Order matters: multi-char operators must be
# checked before their single-char prefixes (!= before =).
_RELATION_OPS = ("!=", "==", ">=", "<=", "=", "：", ":", "是")

# CJK negation prefixes/particles stripped from (and detected in) the
# predicate side of a fact. Checked longest-first so "不曾" doesn't get
# mis-split by a bare "不".
_NEGATIONS = ("不曾", "沒有", "未曾", "未", "沒", "無", "非", "不")

# Time-marker particles with no polarity meaning of their own -- stripped
# so "已死亡" and "死亡" normalize to the same predicate.
_TIME_MARKERS = ("業已", "已經", "已", "尚未", "尚", "仍然", "仍")

# Curated wuxia-flavored antonym pairs. Each entry is a frozenset of two
# predicates that are mutually exclusive states -- deliberately small and
# reviewable, not an attempt at general NLU. Extend as false negatives turn
# up in real generations.
_ANTONYMS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"存活", "死亡"}),
        frozenset({"生還", "身亡"}),
        frozenset({"得到", "失去"}),
        frozenset({"取得", "遺失"}),
        frozenset({"開啟", "關閉"}),
        frozenset({"結盟", "決裂"}),
        frozenset({"信任", "懷疑"}),
        frozenset({"痊癒", "重傷"}),
        frozenset({"在場", "離去"}),
        frozenset({"成功", "失敗"}),
        frozenset({"清醒", "昏迷"}),
    }
)

# Standalone predicate words with no antonym partner in the table above,
# recognized the same way by the tail-matching fallback -- just never
# treated as conflicting via the antonym-pair rule (only same-predicate
# opposite-polarity, e.g. "完成" vs "未完成", can conflict for these).
_EXTRA_PREDICATES: frozenset[str] = frozenset({"完成"})

# Every predicate word the tail-matching fallback (step 3 below) can
# recognize at the end of a string with no relation operator present.
_KNOWN_PREDICATES: frozenset[str] = (
    frozenset(word for pair in _ANTONYMS for word in pair) | _EXTRA_PREDICATES
)


class Fact:
    """One normalized (subject, predicate, negated) reading of a condition/
    effect string. Frozen/hashable-by-value via __eq__ for test convenience,
    but deliberately not a pydantic model -- this is an internal parsing
    result, never serialized."""

    __slots__ = ("subject", "predicate", "negated")

    def __init__(self, subject: str, predicate: str, negated: bool) -> None:
        self.subject = subject
        self.predicate = predicate
        self.negated = negated

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fact):
            return NotImplemented
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.negated == other.negated
        )

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid only
        sign = "!" if self.negated else ""
        return f"Fact({self.subject!r}, {sign}{self.predicate!r})"


def _strip_time_markers(text: str) -> str:
    for marker in _TIME_MARKERS:
        if text.startswith(marker):
            return text[len(marker) :]
    return text


def parse_fact(text: str) -> Fact | None:
    """Conservatively normalize a free-form condition/effect string into a
    Fact. Returns None whenever the text can't be confidently split into a
    (subject, predicate) pair -- an unparseable string must never produce a
    finding, since that would risk blocking a real run on a false positive.
    """
    text = text.strip().strip("\"'　")
    if not text:
        return None

    for op in _RELATION_OPS:
        idx = text.find(op)
        if idx <= 0 or idx + len(op) >= len(text):
            continue
        subject = text[:idx].strip()
        predicate_raw = text[idx + len(op) :].strip()
        if not subject or not predicate_raw:
            continue
        negated = op == "!="
        predicate = _strip_time_markers(predicate_raw)
        for neg in _NEGATIONS:
            if predicate.startswith(neg):
                negated = not negated
                predicate = predicate[len(neg) :]
                break
        predicate = predicate.strip()
        if not predicate:
            continue
        return Fact(subject=subject, predicate=predicate, negated=negated)

    # No relation operator -- fall back to tail-matching a known predicate
    # word (from the antonym table + _EXTRA_PREDICATES) at the end of the
    # string. Chinese state descriptions often stack a time marker and/or a
    # negation particle immediately before the predicate (e.g. "已死亡",
    # "未完成"), so strip them off the *end* of `head` repeatedly -- unlike
    # the relation-operator branch above, where the predicate side is
    # everything after the operator and markers only ever appear as a
    # leading prefix of that substring.
    for predicate in sorted(_KNOWN_PREDICATES, key=len, reverse=True):
        if not text.endswith(predicate):
            continue
        head = text[: -len(predicate)]
        negated = False
        stripped = True
        while stripped:
            stripped = False
            for marker in _TIME_MARKERS:
                if head.endswith(marker):
                    head = head[: -len(marker)]
                    stripped = True
                    break
            else:
                for neg in _NEGATIONS:
                    if head.endswith(neg):
                        head = head[: -len(neg)]
                        negated = not negated
                        stripped = True
                        break
        subject = head.strip()
        if not subject:
            continue
        return Fact(subject=subject, predicate=predicate, negated=negated)

    return None


def facts_conflict(a: Fact, b: Fact) -> bool:
    """Two facts conflict iff they share a subject and either assert the
    same predicate with opposite polarity, or assert a known antonym pair
    with the same polarity."""
    if a.subject != b.subject:
        return False
    if a.predicate == b.predicate:
        return a.negated != b.negated
    pair = frozenset({a.predicate, b.predicate})
    if pair in _ANTONYMS:
        return a.negated == b.negated
    return False


# --- Graph construction -----------------------------------------------------


def _field(obj: Any, name: str) -> str:
    """Read `name` off `obj`, tolerating a plain dict as well as a pydantic
    model instance -- Event.model_copy(update={...}) doesn't re-validate
    nested fields, so a caller that passes raw dicts for triggers/branches
    (as some test fixtures do) leaves them as dicts, not Trigger/Branch
    instances. Never raises; missing/wrong-type data just reads as ""."""
    if isinstance(obj, dict):
        value = obj.get(name, "")
    else:
        value = getattr(obj, name, "")
    return value if isinstance(value, str) else ""


def _effect_op_texts(branch: Any) -> list[str]:
    """Render `branch.effect_ops` (structured, validate_references()-
    checked) as fact-shaped strings ("target_id：op=value") -- an additive
    postcondition source alongside Branch.effects' free text, so causal
    consistency checking isn't blind to structured effect data. Tolerates a
    raw dict for `branch`/each op, same convention as _field()."""
    ops = (
        branch.get("effect_ops", [])
        if isinstance(branch, dict)
        else getattr(branch, "effect_ops", [])
    )
    texts: list[str] = []
    for op in ops:
        if isinstance(op, dict):
            target_id = op.get("target_id", "")
            op_name = op.get("op", "")
            value = op.get("value", "")
        else:
            target_id, op_name, value = op.target_id, op.op, op.value
        if not target_id or not op_name:
            continue
        suffix = f"={value}" if value else ""
        texts.append(f"{target_id}：{op_name}{suffix}")
    return texts


def event_to_node(event: Event) -> PlotNode:
    """One PlotNode per Event: preconditions from its triggers' conditions,
    postconditions from its branches' effects (both the free-text `effects`
    summary and the structured `effect_ops`, unioned -- see
    _effect_op_texts()). Blank strings are skipped -- triggers/branches
    without a meaningful condition/effect (the common case) contribute
    nothing."""
    preconditions = [c for t in event.triggers if (c := _field(t, "condition")).strip()]
    postconditions = [e for b in event.branches if (e := _field(b, "effects")).strip()]
    for branch in event.branches:
        postconditions.extend(_effect_op_texts(branch))
    return PlotNode(id=event.id, preconditions=preconditions, postconditions=postconditions)


def build_graph(beat_sheet: BeatSheet, events: list[Event]) -> CausalPlotGraph:
    """Rebuild the full CausalPlotGraph from a BeatSheet and every
    currently-committed Event. Always a full rebuild, never an incremental
    mutation -- see module docstring for why."""
    nodes = [event_to_node(event) for event in events]

    event_ids = {event.id for event in events}
    beats_by_id = {beat.id: beat for beat in beat_sheet.beats}

    edges: list[PlotEdge] = []
    for beat in beat_sheet.beats:
        if beat.id not in event_ids:
            continue
        for dep in beat.causal_deps:
            if dep in beats_by_id and dep in event_ids:
                edges.append(PlotEdge(from_id=dep, to_id=beat.id))

    for event in events:
        for branch in event.branches:
            next_event_id = _field(branch, "next_event_id")
            if next_event_id in event_ids:
                edges.append(
                    PlotEdge(
                        from_id=event.id,
                        to_id=next_event_id,
                        condition=_field(branch, "condition"),
                    )
                )
        # SkillCheck outcomes: a successful check moves play to
        # success_next_event_id, same edge shape as a branch's next_event_id
        # -- an additional flow path build_graph() previously had no way to
        # see. failure_branch_id names a Branch.id, not an Event.id, so its
        # own edge is already captured by the branches loop above.
        for check in event.checks:
            success_id = _field(check, "success_next_event_id")
            if success_id in event_ids:
                check_id = _field(check, "id")
                edges.append(
                    PlotEdge(
                        from_id=event.id,
                        to_id=success_id,
                        condition=f"check {check_id} success" if check_id else "check success",
                    )
                )

    return CausalPlotGraph(nodes=nodes, edges=edges)


# --- Consistency check -------------------------------------------------------


def _nearest_ancestors(beat_id: str, beats_by_id: dict[str, Beat]) -> list[str]:
    """Direct causal_deps of beat_id (its nearest ancestors), not the full
    transitive closure -- check_scene_consistency only compares against the
    immediately preceding causal fact, matching how build_session_document's
    "most recent ancestor wins" priority already treats continuity."""
    beat = beats_by_id.get(beat_id)
    if beat is None:
        return []
    return list(beat.causal_deps)


def check_scene_consistency(
    beat_sheet: BeatSheet, committed: list[Event], candidate: Event
) -> list[str]:
    """Check whether `candidate`'s preconditions conflict with any nearest
    causal ancestor's postconditions among `committed` scenes. Returns a
    list of human-readable problem descriptions (empty = consistent), in
    the same style as schema.validate_references().

    Only checks precondition-vs-ancestor-postcondition, not postcondition-
    vs-postcondition -- scenes are expected to change state over time, so
    flagging every state change as a "conflict" would be almost entirely
    false positives. Scenes are generated in causal order, so only the
    ancestor direction needs checking at generation time.
    """
    beats_by_id = {beat.id: beat for beat in beat_sheet.beats}
    ancestor_ids = set(_nearest_ancestors(candidate.id, beats_by_id))
    if not ancestor_ids:
        return []

    committed_by_id = {event.id: event for event in committed}
    candidate_facts = [
        fact
        for fact in (parse_fact(_field(t, "condition")) for t in candidate.triggers)
        if fact is not None
    ]
    if not candidate_facts:
        return []

    problems: list[str] = []
    for ancestor_id in ancestor_ids:
        ancestor = committed_by_id.get(ancestor_id)
        if ancestor is None:
            continue
        ancestor_facts = [
            fact
            for fact in (parse_fact(_field(b, "effects")) for b in ancestor.branches)
            if fact is not None
        ]
        for cand_fact in candidate_facts:
            for anc_fact in ancestor_facts:
                if facts_conflict(cand_fact, anc_fact):
                    problems.append(
                        f"event {candidate.id!r}: precondition "
                        f"{cand_fact.subject}{'!' if cand_fact.negated else ''}"
                        f"{cand_fact.predicate!r} 與前置場次 {ancestor_id!r} 造成的結果 "
                        f"{anc_fact.subject}{'!' if anc_fact.negated else ''}"
                        f"{anc_fact.predicate!r} 矛盾"
                    )

    return problems


# --- Directed repair ---------------------------------------------------------


def repair_scene_task(event: Event, problems: list[str], agent: Agent) -> Task:
    """Ask the proofreader agent to fix specific causal-consistency
    problems in one already-complete Event, without touching its overall
    structure. Mirrors crew/pipeline.py::_repair()'s shape (problem list in
    the description, event JSON via context=, output_pydantic=Event)."""
    task = Task(
        description=(
            "以下這一場戲經因果一致性檢查，與前置場次的結果有矛盾，請逐項修正後"
            "回傳完整 Event JSON：\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n修正原則：調整 triggers/branches 的條件或效果文字，讓其不再與"
            "前置場次矛盾；不要更動 event 的 id、location、dialogue 內容，也"
            "不要新增或刪除 branch/trigger 的數量。"
        ),
        expected_output="修正後、與前置場次不再矛盾的完整 Event JSON。",
        agent=agent,
        output_pydantic=Event,
    )
    return task
