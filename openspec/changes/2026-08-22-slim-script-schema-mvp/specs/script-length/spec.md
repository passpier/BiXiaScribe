## MODIFIED Requirements

### Requirement: Preset script-length values
The system SHALL support three named presets — `short`, `medium`, `long` — each resolving to a
fixed set of four prompt targets: event count, chapter count, beats per chapter, and minimum
dialogue depth.

#### Scenario: Default preset
- **WHEN** no script-length value is configured anywhere (env, CLI flag, variant, UI)
- **THEN** the system resolves the `short` preset

#### Scenario: Named preset selected
- **WHEN** a script-length value of `medium` or `long` is supplied
- **THEN** the system resolves that preset's four targets

### Requirement: Custom script-length specification
The system SHALL accept a custom script-length specification as a string of the form
`custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT`, allowing any subset of the
four fields to be provided.

#### Scenario: Fully specified custom value
- **WHEN** a custom value supplies all four fields (`events`, `chapters`, `beats_per_chapter`,
  `min_dialogue`)
- **THEN** the system uses exactly those four values as the prompt targets

#### Scenario: Partially specified custom value derives the rest from events
- **WHEN** a custom value supplies `events` but omits one or more of `chapters`,
  `beats_per_chapter`, `min_dialogue`
- **THEN** the system derives the missing fields from `events` (chapters and beats per chapter
  scaled proportionally to `events`, minimum dialogue depth tiered by `events`) and the resolved
  specification includes concrete values for all four fields

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

## REMOVED Requirements

### Requirement: Main/flavor scene-mix target
**Reason**: `Event.scene_kind`/`Beat.scene_kind` is removed by the `script-frame` capability's
matching change (no main/flavor classification exists to measure a target ratio against), so
`LengthSpec`'s fifth resolved field (`scene_mix`) has nothing left to describe.
**Migration**: `LengthSpec.targets`/resolved specs from before this change carried a `scene_mix`
string; callers reading the resolved spec dict should stop expecting that key. No persisted data
depends on it (it was a prompt-target knob, not stored script data).
