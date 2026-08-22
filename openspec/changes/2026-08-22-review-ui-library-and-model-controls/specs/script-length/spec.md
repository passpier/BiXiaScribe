## MODIFIED Requirements

### Requirement: Custom script-length specification
The system SHALL accept a custom script-length specification as a string of the form
`custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT`, allowing any subset of the
four fields to be provided, and SHALL disclose to the user which pipeline mode(s) each field
actually affects.

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

#### Scenario: Field-level pipeline-mode disclosure
- **WHEN** a user is shown the custom-length input fields in an interactive UI
- **THEN** each field discloses which pipeline mode(s) actually consume it — `events` affects only
  the legacy pipeline's prompt target (though it still affects the estimated scene count/cost for
  either pipeline mode via length scaling), `chapters` and `beats_per_chapter` affect only the
  layered pipeline's prompt target, and `min_dialogue` affects both pipelines' prompt targets
- **AND** a field that has no effect on the currently-selected pipeline mode's *output* is visibly
  marked as such
