## ADDED Requirements

### Requirement: Discover resumable runs
The system SHALL discover interrupted layered-pipeline checkpoints — those with a readable
`state.json` but no assembled `script.json` — separately from finished-run discovery.

#### Scenario: Interrupted run listed as resumable
- **WHEN** a `.bixia_state/<run_id>/` directory has a readable `state.json` and no `script.json`
- **THEN** it appears in the resumable-runs listing with its stage, requirement, completed scene
  count, last-updated time, and the checkpoint envelope's schema version

#### Scenario: Finished run excluded from resumable listing
- **WHEN** a `.bixia_state/<run_id>/` directory's `state.json` reports `stage == "done"` and a
  `script.json` exists
- **THEN** it does not appear in the resumable-runs listing (it belongs to the existing
  finished-checkpoint listing instead)

### Requirement: Resume gated on schema version
The system SHALL refuse to resume a checkpoint whose schema version does not match the currently
running pipeline's schema version, since a version mismatch causes silent full-cost regeneration
rather than an actual resume.

#### Scenario: Matching version allowed
- **WHEN** a user selects a resumable run whose checkpoint schema version equals the current
  pipeline's schema version
- **THEN** the system proceeds to resume generation using that run's id

#### Scenario: Mismatched version refused
- **WHEN** a user selects a resumable run whose checkpoint schema version does not equal the
  current pipeline's schema version
- **THEN** the system refuses the resume with an explicit error stating that resuming will not
  reuse any completed stage and will regenerate from scratch at full cost, and does not proceed
  with that run id

### Requirement: Resumed run reuses the checkpointed requirement
The system SHALL reuse the requirement text stored in a resumed checkpoint's state rather than
accepting an edited one, since already-completed stages were generated against the original text.

#### Scenario: Requirement locked on resume
- **WHEN** a user selects a run to resume
- **THEN** the requirement field is prefilled from that run's checkpointed state and is not
  editable for that resume
