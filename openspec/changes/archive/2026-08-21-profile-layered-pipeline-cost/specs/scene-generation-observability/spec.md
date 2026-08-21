## Purpose

Gives the layered generation pipeline per-scene execution attribution — elapsed time, LLM call
count, reasoning-token usage, guardrail retries, and retrieval calls broken down by scene — so
that later optimizations to the pipeline's cost and latency can be validated against a specific
mechanism instead of only a run-wide total.

## ADDED Requirements

### Requirement: Per-scene execution attribution is recorded
For every scene a layered generation run produces, the system SHALL record, keyed by that
scene's identity: elapsed wall-clock time, number of LLM calls made, reasoning-token usage,
number of guardrail retries, and number of retrieval calls made while generating that scene.

#### Scenario: A scene generated without any retry
- **WHEN** a layered run generates a scene that succeeds on its first attempt, with no guardrail
  rejection and no structured-output fallback
- **THEN** the run's recorded attribution includes exactly one entry for that scene, with elapsed
  time and LLM-call count reflecting that single attempt

#### Scenario: A scene that required a guardrail retry
- **WHEN** a scene's generation is rejected by a quality guardrail and retried before succeeding
- **THEN** that scene's recorded attribution reflects at least one guardrail retry, and its
  elapsed time and LLM-call count include the rejected attempt

#### Scenario: A scene generated concurrently with others in the same batch
- **WHEN** several scenes with no causal dependency on each other are generated concurrently in
  one batch
- **THEN** each scene's recorded attribution reflects only that scene's own execution, not another
  concurrently-generated scene's

### Requirement: Attribution survives a resumed run
A layered run resumed from a checkpoint SHALL report attribution for every completed scene,
including scenes that were generated in an earlier process before the resume, not only scenes
generated after the resume.

#### Scenario: Resuming an interrupted run
- **WHEN** a layered run is interrupted after completing some scenes and later resumed by run id
- **THEN** the resumed run's final report includes attribution for the scenes completed before the
  interruption as well as the scenes completed after resuming

### Requirement: Partial attribution survives a failed run
When a layered run fails before completing every scene, the system SHALL still report attribution
for every scene that did complete before the failure.

#### Scenario: A run that fails partway through scene generation
- **WHEN** a layered run generates several scenes successfully and then fails on a later scene
- **THEN** the error raised for that run carries attribution for the scenes that completed before
  the failure

### Requirement: Attribution is available in the persisted run log and review UI
Per-scene attribution recorded for a layered run SHALL be included in that run's persisted log
record, and SHALL be viewable through the review UI's execution-record view for that run.

#### Scenario: Reading a past run's attribution from the log
- **WHEN** a layered run's log record is read back (e.g. via the evaluation harness or the review
  UI) after the run has finished
- **THEN** the per-scene attribution recorded during that run is present in the record

#### Scenario: Viewing attribution in the review UI
- **WHEN** a user opens a layered run's execution record in the review UI
- **THEN** the per-scene attribution is displayed in a form that can be sorted by elapsed time, so
  the slowest scenes in that run can be identified

### Requirement: Absence of attribution degrades gracefully
A run or log record that predates this capability, or that was produced by the legacy (non-
layered) pipeline, SHALL be read back with empty per-scene attribution rather than an error.

#### Scenario: Reading a legacy-mode run's record
- **WHEN** a run record produced by the legacy pipeline mode is read
- **THEN** its per-scene attribution is empty, and reading it does not raise an error

#### Scenario: Reading a log record written before this capability existed
- **WHEN** a persisted run log record written before per-scene attribution existed is read back
- **THEN** its per-scene attribution is empty, and reading it does not raise an error
