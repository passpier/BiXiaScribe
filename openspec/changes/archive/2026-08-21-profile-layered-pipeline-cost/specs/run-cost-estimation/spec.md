## Purpose

Gives every generation entry point (the review UI's 生成 form, a running job, and both CLI
scripts) a cost (USD) and wall-clock time estimate before and during a run, so a caller can decide
whether to spend the tokens before waiting out the run to find out. Built on top of
`scene-generation-observability`'s per-scene attribution: once a layered run has generated at
least one scene, its own measured per-scene cost/time replaces a generic historical guess for
whatever scenes remain.

Motivated by a real run (`.bixia_state/1787309292-req-d232acf2d8`, `layered`/`script_length=long`
against `deepseek-v4-flash-0731`): cancelled after 19 minutes and $0.0039 spent, with only 1 of 30
beats generated. Extrapolating that run's own already-completed stages gave ~$0.065 and ~2 hours
to finish — money was never the blocker, but nothing surfaced that number before 19 minutes had
already been spent finding out. Separately, that run's 30 beats formed a single linear causal
chain (`plan_batches()` returned 30 batches of width 1), so `SCENE_CONCURRENCY` had no effect on
it — an estimate that only reports a dollar figure and hides this shape would still leave a caller
guessing why a "3x concurrent" run took as long as a serial one.

## ADDED Requirements

### Requirement: An estimate is available before a run starts
Before a generation run is dispatched, the system SHALL provide a cost (USD) and wall-clock time
estimate for the pipeline mode, script length, and model selection the run would use.

#### Scenario: Estimating from the generation form before submission
- **WHEN** a user has chosen a pipeline mode, script length, model variant, and retrieval setting
  in the generation UI, before clicking to start a run
- **THEN** the system shows a cost and time estimate reflecting those exact settings

#### Scenario: Estimating via the CLI before spending tokens
- **WHEN** a caller runs the generation CLI's preflight check, or is about to start a real run
- **THEN** the estimate is printed before any model call is made

### Requirement: An estimate names its own data source and confidence
Every estimate SHALL be labeled with the basis it was computed from, ranked from most to least
trustworthy: this run's own measured progress, matching historical runs of the same pipeline mode
and script length, historical runs of the same pipeline mode scaled to a different length, or a
fixed fallback when no historical data exists at all.

#### Scenario: No historical data exists yet
- **WHEN** no past run matches the requested pipeline mode at all
- **THEN** the estimate is still produced, using a documented fixed fallback, and is labeled as
  such rather than presented as if it were measured from real usage

#### Scenario: An exact historical match exists
- **WHEN** past runs exist for the same pipeline mode and script length
- **THEN** the estimate is computed from those runs' recorded token usage and elapsed time, and is
  labeled as a closer match than a fallback or cross-length estimate

### Requirement: An unpriced model is reported as unknown, never as free
When a run's cost cannot be computed because a selected model has no price entry, the system SHALL
report the cost as unavailable rather than as zero.

#### Scenario: A model with no price entry
- **WHEN** an estimate is requested for a model id that has no entry in the price catalog
- **THEN** the reported cost is absent/unknown, not a fabricated $0.00, and the estimate explains
  why

### Requirement: A layered run's batch structure is reflected in the time estimate
When the causal-dependency batch structure for a layered run's scenes is known (or can be derived
from an already-generated beat sheet), the time estimate SHALL account for it rather than assuming
every scene runs sequentially, and SHALL surface the resulting parallelism.

#### Scenario: A beat sheet with no useful parallelism
- **WHEN** a layered run's beats form a causal chain with no batch wider than one scene
- **THEN** the time estimate reflects fully serial execution regardless of the configured
  concurrency limit, and the estimate notes that the configured concurrency has no effect on this
  run's structure

#### Scenario: A beat sheet with real parallelism
- **WHEN** a layered run's beats include batches of multiple mutually-independent scenes
- **THEN** the time estimate is shorter than a fully serial estimate would be, in proportion to the
  configured concurrency limit and the batch widths involved

### Requirement: An in-progress run's estimate is corrected by its own measured progress
Once a layered run has completed at least one scene, the estimate for its remaining work SHALL be
computed from that run's own measured per-scene cost and time rather than from historical or
fallback data, and SHALL be labeled as such.

#### Scenario: Refining an estimate mid-run
- **WHEN** a layered run has completed some scenes and has scenes remaining
- **THEN** the estimate for the remaining scenes uses this run's own average measured cost and time
  per completed scene, and is labeled as measured rather than historical or fallback

#### Scenario: A run with nothing completed yet
- **WHEN** a layered run has not yet completed any scene
- **THEN** its estimate falls back to the pre-run estimate for its configured settings

#### Scenario: A run with nothing left to do
- **WHEN** every scene a layered run needs has already completed
- **THEN** the remaining estimate reports zero additional cost and time
