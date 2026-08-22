## MODIFIED Requirements

### Requirement: Faction representation
The system SHALL allow a script to declare factions, each with a single one-sentence motive, so
NPC and plot motivation can be grounded in a concrete power structure without maintaining a
separate inter-faction relations matrix.

#### Scenario: NPC affiliated with a declared faction
- **WHEN** an NPC declares a `faction_id` matching a declared faction
- **THEN** the reference resolves, and cross-reference validation reports no problem

#### Scenario: NPC affiliated with an unknown faction
- **WHEN** an NPC's `faction_id` does not match any declared faction id
- **THEN** reference validation reports the dangling reference

### Requirement: Single player stat with resolution endings
The system SHALL track exactly one numeric player stat (e.g. 心境值/正邪值), and let the script
declare endings whose selection condition is a value range over that single stat, so the game has
one unambiguous progress value instead of a general-purpose variable/threshold system.

#### Scenario: Ending selected by the single stat's value
- **WHEN** the player stat's final value falls within an ending's declared `min`/`max` range
- **THEN** that ending is the one selected

#### Scenario: Overlapping or gapped ending ranges
- **WHEN** two declared endings' `min`/`max` ranges overlap, or the ranges declared across all
  endings leave part of the stat's possible value space uncovered
- **THEN** ending-range validation reports the overlap or gap

### Requirement: Three-layer truth disclosure
The system SHALL let a script separate its underlying truth into three layers — a fact known from
the start (`public`), an ordered list of facts revealed progressively as chapters advance
(`revealed`), and a fact reserved for the author that must never appear in generated scene content
before its designated reveal point (`hidden`) — where progressive-reveal pacing is expressed by
the `revealed` list's own order rather than by binding each entry to an explicit chapter id.

#### Scenario: Hidden truth is withheld from scene generation
- **WHEN** a scene is generated for any chapter
- **THEN** the hidden fact is not made available as input to that scene's generation, regardless
  of whether an automated check would also catch it appearing in the output

#### Scenario: A hidden fact leaks into scene content
- **WHEN** a generated event's summary or dialogue contains the hidden truth fact's substance
- **THEN** truth-pacing validation reports the premature disclosure

### Requirement: Chapter structure with a starting event
The system SHALL let a script organize events into chapters, each with a single starting event id
that begins the chapter and a location, forming a linear location chain, so the engine can
resolve "what location/event begins this chapter" without maintaining a separate convergence-point
concept.

#### Scenario: Chapter with a resolvable starting event
- **WHEN** a chapter declares a `start_event`
- **THEN** that id resolves to a declared event

#### Scenario: Chapter with a dangling starting event
- **WHEN** a chapter's `start_event` does not match any declared event id
- **THEN** reference validation reports the dangling reference

### Requirement: Clue tracking
The system SHALL let a script declare clues, each tied to the event where it can be found
(`from_event`), so investigative scenes carry information rather than being pure narrative
filler.

#### Scenario: Clue resolves to a real event
- **WHEN** a clue declares `from_event`
- **THEN** that id resolves to a declared event

#### Scenario: Event with no information payoff
- **WHEN** an event unlocks no clue and no choice with a non-empty effect description
- **THEN** scene-information validation reports the event as a content-free scene

### Requirement: Single skill check with failure fallback
The system SHALL let an event declare at most one skill check, with a pass route (`on_pass`) and
a fail route (`on_fail`) plus a failure-cost description, so a failed check advances the story
instead of ending it, without needing a check-kind taxonomy or item-bypass alternative.

#### Scenario: Check with a failure route
- **WHEN** an event's check declares both `on_fail` and a non-empty `fail_cost`
- **THEN** check-fallback validation reports no problem for that event

#### Scenario: Check with no failure route
- **WHEN** an event's check declares no `on_fail` or no `fail_cost`
- **THEN** check-fallback validation reports the check as a possible dead end

### Requirement: Choices carry cost and numeric effect
The system SHALL require every choice with a non-empty free-text effect description to also
declare a cost (what the player gives up, not merely a numeric delta), and, when its effect is a
delayed payoff rather than an immediate one, a payoff chapter id, so choices are real
value/resource tradeoffs rather than choices that differ only in degree.

#### Scenario: Choice with cost and delta
- **WHEN** a choice declares a non-empty `cost` and a non-zero `delta`
- **THEN** choice-quality validation reports no problem for that choice

#### Scenario: Choice missing a cost
- **WHEN** a choice has a non-empty effect description but no declared cost
- **THEN** choice-quality validation reports the choice as missing a real tradeoff

#### Scenario: Two choices differ only in degree
- **WHEN** two choices within the same event have highly similar choice text and the same-signed
  `delta`, with no distinct cost from one another
- **THEN** choice-quality validation flags the pair as a false choice (假選擇)

#### Scenario: Payoff chapter declared
- **WHEN** a choice declares `payoff_at`
- **THEN** that id resolves to a declared chapter

## REMOVED Requirements

### Requirement: Stat-value threshold table
**Reason**: replaced by the single-stat/ending-range model above — a general threshold table
mapping arbitrary value ranges to arbitrary unlock targets (branch/event/npc_attitude/ending) was
never used for anything but ending selection in practice, and added a wire-required nested object
per range with no offsetting narrative value for this project's MVP scope.
**Migration**: existing scripts carrying `stat_thresholds` keep loading — the field is simply
unknown to the new `Script` model and is dropped on read (pydantic's default `extra="ignore"`).

### Requirement: Region and sub-location structure
**Reason**: already removed from code in the 2026-08-21 schema-slimming pass (`Region`/
`SubLocation` classes deleted); this spec entry was left behind as drift and is removed now to
match. No equivalent concept exists in `武俠劇本資料庫Schema設計.md`'s flat schema — chapter/event
location is expressed by `Chapter.loc` (a single string) instead.
**Migration**: not applicable — the code-level removal already shipped and existing scripts with
`regions`/`sub_locations` already silently drop those fields.

### Requirement: Chapter structure with hook and convergence
**Reason**: replaced by "Chapter structure with a starting event" above. `hook` is dropped
(narrative framing folded into `Chapter.summary`); `converge_event_id` and the per-branch
`converges_to_event_id`/convergence validation are dropped — the "every branch path eventually
reconverges" guarantee is replaced by the simpler `payoff_at` (a chapter id) annotation on
individual choices, matching the flat schema's ID-reference style rather than a graph-reachability
guarantee.
**Migration**: existing scripts' `chapters[].hook`/`.converge_event_id` and
`branches[].converges_to_event_id` are dropped on read.

### Requirement: Skill checks with failure fallback (multi-check, item-bypass variant)
**Reason**: replaced by "Single skill check with failure fallback" above — an event carries at
most one check (a single object, not a list), with no check-kind taxonomy (attribute
contest/dice/probability) and no item-bypass alternative route, matching the guide's "單一判定機制"
principle.
**Migration**: existing scripts' `events[].checks` (a list) is dropped on read; a script authored
under the new schema uses `events[].check` (a single optional object) instead.

### Requirement: Branch choices carry cost, feedback, and delayed payoff (branch/effect_ops variant)
**Reason**: replaced by "Choices carry cost and numeric effect" above. `effect_ops` (a structured
multi-target-kind effect list: variable/stat/item, each with its own op/value) is replaced by a
single `delta: int` plus a free-text `effects` description — this project only ever tracked one
numeric stat, so the general multi-target effect system added wire cost with no corresponding use.
`immediate_feedback` is dropped — its content is folded into the next event's own dialogue/summary
rather than being a separate annotation field.
**Migration**: existing scripts' `branches[].effect_ops`/`.immediate_feedback` are dropped on
read; `.cost`/`.payoff_description` map conceptually to the new `.cost`/`.payoff_at` but are not
auto-migrated (different shape — `payoff_description` was free text, `payoff_at` is a chapter id).

### Requirement: Main and flavor scene balance
**Reason**: `Event.scene_kind`/`Beat.scene_kind` (main/flavor classification) is removed — the
guide's example scripts never use a scene-kind taxonomy; this was assessed as over-design for
MVP scope in the same review that produced `武俠劇本資料庫Schema設計.md`. If main/flavor pacing
distinction is needed later, `Chapter.kind` was noted as a possible reintroduction point, not
resurrecting the per-event field.
**Migration**: existing scripts' `events[].scene_kind` and the corresponding `length.py`
`scene_mix` target are dropped; see the `script-length` capability's matching removal.
