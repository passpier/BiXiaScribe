## MODIFIED Requirements

### Requirement: Preset script-length values
The system SHALL support three named presets — `short`, `medium`, `long` — each resolving to a
fixed set of five prompt targets: event count, chapter count, beats per chapter, minimum dialogue
depth, and main/flavor scene mix ratio.

#### Scenario: Default preset
- **WHEN** no script-length value is configured anywhere (env, CLI flag, variant, UI)
- **THEN** the system resolves the `short` preset

#### Scenario: Named preset selected
- **WHEN** a script-length value of `medium` or `long` is supplied
- **THEN** the system resolves that preset's five targets, including a scene-mix target expressing
  that main scenes should be somewhat more numerous than flavor scenes

### Requirement: Custom script-length specification
The system SHALL accept a custom script-length specification as a string of the form
`custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT,scene_mix=TEXT`, allowing any
subset of the five fields to be provided.

#### Scenario: Fully specified custom value
- **WHEN** a custom value supplies all five fields (`events`, `chapters`, `beats_per_chapter`,
  `min_dialogue`, `scene_mix`)
- **THEN** the system uses exactly those five values as the prompt targets

#### Scenario: Partially specified custom value derives the rest from events
- **WHEN** a custom value supplies `events` but omits one or more of `chapters`,
  `beats_per_chapter`, `min_dialogue`, `scene_mix`
- **THEN** the system derives the missing fields from `events` (chapters and beats per chapter
  scaled proportionally to `events`, minimum dialogue depth and scene mix tiered by `events`) and
  the resolved specification includes concrete values for all five fields

#### Scenario: Unparseable custom value falls back to short
- **WHEN** a script-length value starts with `custom:` but cannot be parsed into any valid field
  (e.g. malformed syntax, non-numeric `events`)
- **THEN** the system resolves the `short` preset instead of raising an error, consistent with how
  other unrecognized configuration values (e.g. `CAUSAL_VALIDATION`, `PIPELINE_MODE`) degrade
  rather than crash

#### Scenario: Unrecognized non-custom value falls back to short
- **WHEN** a script-length value is neither one of the three presets nor a `custom:`-prefixed
  string
- **THEN** the system resolves the `short` preset

### Requirement: Resolved script-length is recorded self-describingly
A generation run's recorded report (and any persisted log row derived from it) SHALL record the
fully-resolved script-length specification — with all five fields filled in — rather than the raw
possibly-partial input string, so a logged run's target length can be read without
cross-referencing the configuration that was active at run time.

#### Scenario: Logged run with a partial custom length
- **WHEN** a generation run used a custom script-length value that specified only `events`
- **THEN** the run's recorded script-length is the fully-resolved form including the derived
  `chapters`, `beats_per_chapter`, `min_dialogue`, and `scene_mix` values
