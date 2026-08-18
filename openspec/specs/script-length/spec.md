## Purpose

Defines how a script-length target (how many events/chapters/beats/dialogue-depth a
generation run should aim for) is expressed, validated, and resolved into the prompt-level
targets consumed by script generation, across every entry point that sets it (`.env`, CLI
flags, the review UI's generation mode, and eval-harness variant definitions).

## Requirements

### Requirement: Preset script-length values
The system SHALL support three named presets — `short`, `medium`, `long` — each resolving to
a fixed set of four prompt targets: event count, chapter count, beats per chapter, and
minimum dialogue depth.

#### Scenario: Default preset
- **WHEN** no script-length value is configured anywhere (env, CLI flag, variant, UI)
- **THEN** the system resolves the `short` preset

#### Scenario: Named preset selected
- **WHEN** a script-length value of `medium` or `long` is supplied
- **THEN** the system resolves that preset's four targets unchanged from today's behavior

### Requirement: Custom script-length specification
The system SHALL accept a custom script-length specification as a string of the form
`custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT`, allowing any subset of
the four fields to be provided.

#### Scenario: Fully specified custom value
- **WHEN** a custom value supplies all four fields (`events`, `chapters`,
  `beats_per_chapter`, `min_dialogue`)
- **THEN** the system uses exactly those four values as the prompt targets

#### Scenario: Partially specified custom value derives the rest from events
- **WHEN** a custom value supplies `events` but omits one or more of `chapters`,
  `beats_per_chapter`, `min_dialogue`
- **THEN** the system derives the missing fields from `events` (chapters and beats per
  chapter scaled proportionally to `events`, minimum dialogue depth tiered by `events`)
  and the resolved specification includes concrete values for all four fields

#### Scenario: Unparseable custom value falls back to short
- **WHEN** a script-length value starts with `custom:` but cannot be parsed into any valid
  field (e.g. malformed syntax, non-numeric `events`)
- **THEN** the system resolves the `short` preset instead of raising an error, consistent
  with how other unrecognized configuration values (e.g. `CAUSAL_VALIDATION`,
  `PIPELINE_MODE`) degrade rather than crash

#### Scenario: Unrecognized non-custom value falls back to short
- **WHEN** a script-length value is neither one of the three presets nor a `custom:`-prefixed
  string
- **THEN** the system resolves the `short` preset

### Requirement: Script-length applies uniformly across entry points
Every entry point that lets a user choose a script length (the `SCRIPT_LENGTH` environment
variable, the `--script-length` CLI flag on script-generation and evaluation commands, an
eval-harness variant's script-length field, and the review UI's generation mode) SHALL accept
both the named presets and the custom syntax, and SHALL resolve them identically.

#### Scenario: CLI custom flag
- **WHEN** a script-generation CLI command is invoked with
  `--script-length "custom:events=20,chapters=4"`
- **THEN** the generation run uses the resolved custom targets (chapters=4 as given,
  beats_per_chapter and min_dialogue derived from events=20)

#### Scenario: UI custom selection
- **WHEN** a user in the review UI's generation mode selects the "自訂" (custom) length
  option and fills in one or more of the four fields
- **THEN** the triggered generation run uses the resolved custom targets, the same as if the
  equivalent custom string had been passed via CLI or `.env`

#### Scenario: Eval variant custom length
- **WHEN** an eval-harness variant definition sets its script-length field to a custom string
- **THEN** every run of that variant across the requirement matrix uses the resolved custom
  targets

### Requirement: Resolved script-length is recorded self-describingly
A generation run's recorded report (and any persisted log row derived from it) SHALL record
the fully-resolved script-length specification — with all four fields filled in — rather than
the raw possibly-partial input string, so a logged run's target length can be read without
cross-referencing the configuration that was active at run time.

#### Scenario: Logged run with a partial custom length
- **WHEN** a generation run used a custom script-length value that specified only `events`
- **THEN** the run's recorded script-length is the fully-resolved form including the derived
  `chapters`, `beats_per_chapter`, and `min_dialogue` values

### Requirement: Selecting a UI variant preserves its declared script length
When the review UI's generation mode is used to trigger a run based on a predefined variant
(not the "自訂" model option), the triggered run SHALL use that variant's own declared
script-length (and any other per-variant generation settings it declares), not silently fall
back to the default.

#### Scenario: Predefined variant with its own script length
- **WHEN** a user selects a predefined variant that declares a non-default script-length in
  the review UI's generation mode, without overriding the length selector
- **THEN** the triggered generation run uses that variant's declared script-length
