## ADDED Requirements

### Requirement: Curated model catalog
The system SHALL maintain a curated catalog of model metadata (display name, description, tested
status, recommended roles) separate from and joined against pricing data, so pricing stays owned by
a single regenerable file.

#### Scenario: Catalog entry never duplicates price fields
- **WHEN** the catalog describes a model that also has a pricing entry
- **THEN** the catalog's own JSON does not repeat price/context-length/tool-support fields — those
  are joined in at load time from the pricing data

#### Scenario: Unknown model degrades gracefully
- **WHEN** a model id is looked up that has no catalog entry (e.g. from a historical run using a
  model since removed from the catalog)
- **THEN** the system returns a description using the raw model id as the display name and an
  "untested" status, rather than raising an error

### Requirement: Role-to-model mapping reflects actual pipeline usage
The system SHALL expose, for a given pipeline mode, the exact set of agent roles that mode uses —
shared by cost accounting and by display — so the two cannot diverge.

#### Scenario: Layered mode includes the proofreader role
- **WHEN** the role set for `"layered"` pipeline mode is requested
- **THEN** it includes `"proof"` in addition to `"extractor"`, `"beat_expander"`, and
  `"scene_writer"`, reflecting that causal repair invokes the proofreader agent on every scene

#### Scenario: Legacy mode excludes layered-only roles
- **WHEN** the role set for `"legacy"` pipeline mode is requested
- **THEN** it includes only `"writer"`, `"dialogue"`, `"proof"` and excludes
  `"extractor"`/`"beat_expander"`/`"scene_writer"`

### Requirement: Selectable models are restricted to usable ones
The system SHALL exclude models marked as unusable from selection surfaces while still allowing
them to be described for historical-record rendering.

#### Scenario: Unusable model excluded from selection, still describable
- **WHEN** a model is marked `status="unusable"` in the catalog
- **THEN** it does not appear in the list of selectable models
- **AND** a historical run record that used it still renders its description correctly

### Requirement: Global reasoning-effort setting
The system SHALL support a single, run-wide (not per-role) reasoning-effort setting that is
forwarded to the underlying LLM client and recorded on the resulting run's metadata.

#### Scenario: Default effort is a no-op
- **WHEN** no reasoning-effort value is configured anywhere (env, CLI, variant, UI)
- **THEN** the system does not forward any reasoning-effort parameter to the LLM client, and the
  request is identical to a request made before this setting existed

#### Scenario: Non-default effort recorded on the run
- **WHEN** a run is started with a reasoning-effort value other than the default
- **THEN** that value is forwarded to the LLM client for every agent role in the run
- **AND** the value is recorded on the run's log row so it can be compared against other runs
