## ADDED Requirements

### Requirement: Delete a script record
The system SHALL allow deleting a discovered script record's underlying file(s) without modifying
`out/generation_runs*.jsonl`.

#### Scenario: Deleting a jsonl/filename-sourced record
- **WHEN** a user deletes a `ScriptRecord` whose `source` is `"jsonl"` or `"filename"`
- **THEN** the system removes only the script's JSON file at `rec.path`
- **AND** the corresponding row in `out/generation_runs*.jsonl` is left untouched

#### Scenario: Deleting a checkpoint record
- **WHEN** a user deletes a `ScriptRecord` whose `source` is `"checkpoint"`
- **THEN** the system removes the entire `.bixia_state/<run_id>/` directory backing that record

#### Scenario: Deleted record's run reappears as run-only
- **WHEN** a script file backed by a `out/generation_runs*.jsonl` row is deleted
- **THEN** a subsequent discovery pass surfaces that run as a `source="run-only"` record with
  `path=None`, still carrying its original cost and error information

#### Scenario: Delete refuses a path outside managed directories
- **WHEN** a delete is attempted against a path that does not resolve inside the configured eval
  scripts directory or checkpoint state directory
- **THEN** the system raises an error and does not remove anything

#### Scenario: Bulk delete of unreadable scripts
- **WHEN** a user requests bulk deletion of all currently-unreadable script records
- **THEN** the system deletes each one individually per the scenarios above and reports how many
  were removed

### Requirement: Export a script as JSON
The system SHALL allow downloading any successfully-loaded script's JSON representation.

#### Scenario: Exported bytes match the canonical serialization
- **WHEN** a user exports a loaded `Script`
- **THEN** the exported bytes are identical to `script.model_dump_json(indent=2, exclude_none=False)`

### Requirement: Import a script by upload
The system SHALL allow importing a script by uploading a JSON file, which is validated and copied
into the eval scripts directory under the standard naming convention.

#### Scenario: Valid script uploaded
- **WHEN** a user uploads a JSON payload that validates as a `Script` (directly, or after unwrapping
  a `{"schema_version", "data"}` checkpoint envelope) along with a variant name and requirement text
- **THEN** the system writes it to the eval scripts directory using the existing
  `{variant}__{slug}[__repN].json` naming convention, choosing the next unused rep for that
  (variant, slug) pair
- **AND** the imported file is discoverable by the existing script-discovery logic with no
  additional code

#### Scenario: Invalid payload rejected
- **WHEN** a user uploads a payload that is not valid JSON, or is valid JSON that does not validate
  as a `Script`
- **THEN** the system rejects the import with a user-facing reason and writes no file

### Requirement: Load a script ad hoc
The system SHALL allow loading a script from an arbitrary filesystem path for one-time viewing
without copying it into the eval scripts directory.

#### Scenario: Ad hoc load produces a viewable record
- **WHEN** a user supplies a path to a file that validates as a `Script`
- **THEN** the system returns a script record usable by the same rendering logic as any other
  discovered record, marked with a distinct source so it is not treated as a persisted artifact
