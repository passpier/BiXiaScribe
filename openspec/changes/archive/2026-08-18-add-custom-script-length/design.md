## Context

`config.SCRIPT_LENGTH` and `Variant.script_length` are plain strings that flow, unchanged in
type, all the way to `crew/tasks.py::_LENGTH_TARGETS`'s lookup and into `RunReport.script_length`
/ JSONL rows. See proposal.md - Why for the motivation. `crew/tasks.py` imports `crewai`;
`config.py` must stay importable without it (many callers, including tests, import `config`
standalone). The Streamlit generation mode (`ui/app.py`) currently has no length control at all.

## Goals / Non-Goals

**Goals:**
- Let a script-length value be either one of the three existing presets or a custom
  `events`/`chapters`/`beats_per_chapter`/`min_dialogue` specification, resolved identically
  everywhere it's consumed.
- Keep `script_length`'s type a plain `str` end-to-end — no schema/JSONL migration.
- Keep `tests/test_script_length.py`'s byte-for-byte prompt regression guards passing
  unchanged for the `short`/`medium`/`long` presets.

**Non-Goals:**
- Enforcing script length as a hard cap — this remains a prompt-level target only, same as
  today (`LLM_MAX_TOKENS` stays the one real ceiling).
- Any change to `Script`/`Event`/`Beat` schema fields.
- Editing/save-back of generated scripts (out of scope per CLAUDE.md's project description,
  unrelated to this change anyway).

## Decisions

### A dependency-free `length.py` module owns parsing
`crew/tasks.py` imports `crewai`; `config.py` is imported by many crewai-free contexts
(tests, scripts, the UI). Parsing logic goes into a new `src/bixiascribe/length.py` with zero
imports from `config`, `crew/`, or `crewai`, so both `config.py` (validating `SCRIPT_LENGTH`
at import time) and `crew/tasks.py` (building prompts) can depend on it without a cycle, and
so `ui/app.py`/`eval_generation.py` can validate/preview a custom string before spending a
token.

Alternative considered: keep parsing inline in `crew/tasks.py` and have `config.py` do its
own separate lightweight validation (regex only, no shared derivation logic). Rejected —
would let the "derive missing fields from events" logic drift out of sync between the two
call sites, which is exactly the kind of silent inconsistency this project's existing
conventions (e.g. `generation.py`'s `script_length` three-level resolution comment) try to
avoid.

### Custom syntax: `custom:key=value,...`
Chosen over a JSON blob or four separate env vars because:
- It stays a single string, so every existing plumbing point (`Variant.script_length: str |
  None`, `--script-length` argparse, `RunReport.script_length: str`) needs zero type changes.
- It's still comfortably typeable in `.env` and a CLI flag, unlike JSON needing shell-escaping.
- Partial specification (`custom:events=20`) is directly expressible; a fixed-arity
  positional syntax (`custom:20:4:3:三段以上`) would force users to always supply all four
  fields in the derived case.

### Deriving unspecified fields from `events`
When `events` is given but other fields are omitted, derive:
- `chapters = ceil(events / 5)` (short=2→1-2, medium≈10→3-4, long≈20→5-6 — roughly matches
  today's presets' events-to-chapters ratio)
- `beats_per_chapter = ceil(events / chapters)`
- `min_dialogue`: tiered by `events` — `<=4` → `一段`, `<=14` → `二至三段`, else `三段以上`
  (matches the three presets' existing thresholds at their own event counts)

If `events` itself is also omitted (e.g. `custom:chapters=6` alone), `events` defaults to the
`short` preset's `2` before the above derivation runs, so every field always ends up
concretely resolved — this is what "canonical" form means for `RunReport`/JSONL logging.

Alternative considered: require every custom value to be fully specified, erroring on partial
input. Rejected per the spec's "partial specification" requirement — plan.md explicitly calls
for `events=20` alone to be usable, and erroring here would be a worse UX than the
project's established degrade-not-crash pattern.

### `_LENGTH_TARGETS`'s three presets move into `length.py`, `tasks.py::_length_target()` becomes a thin wrapper
`_length_target(script_length)` becomes `parse_length_spec(script_length).targets` — same
returned dict shape (`{"events": ..., "chapters": ..., "beats_per_chapter": ...,
"min_dialogue": ...}`), so every `make_*_task()` call site and its prompt text is untouched.
This is what keeps `tests/test_script_length.py` green without modification for the preset
cases.

### `eval_generation.py`'s cost-estimate scale becomes events-derived
`_LENGTH_SCALE = {"short": 1.0, "medium": 4.0, "long": 8.0}` becomes
`LengthSpec.events_scale = events / 2` (short's own `events=2` as the baseline `1.0`). This
changes medium's scale from `4.0` to `4.0` (10/2, matches) and long's from `8.0` to `8.0`
(assuming long resolves to `events=16`; the existing preset's `events` field is `"15-24"`, a
range — `events_scale` uses the range's lower bound `15` → `7.5`, a ~6% change from today's
`8.0`). This is an acceptable drift in a "rough pre-spend guess" (the module's own docstring
already calls it that) in exchange for one formula instead of a hardcoded table that a new
preset/custom value would otherwise need a manual entry for.

### UI: `ui_variant` construction is fixed to carry `script_length`/`session_doc_max_tokens`
`ui/app.py`'s existing `generation.Variant(name=..., note=..., writer=..., dialogue=...,
proof=...)` reconstruction (around ui/app.py:305) drops these two fields today. This is a
pre-existing bug independent of the new selector — without the fix, adding a length selector
that defaults to "use the variant's own value" would be silently overridden back to
`config.SCRIPT_LENGTH` the moment a predefined (non-自訂) variant is chosen.

### UI length selector UX
A `st.selectbox` with options `["short", "medium", "long", "自訂"]`, mirroring the existing
模型變體 pattern (`ui/app.py:274-289`). Selecting 自訂 opens an expander with four
`st.text_input`/`st.number_input` fields (events, chapters, beats_per_chapter, min_dialogue)
that get joined into the `custom:...` string client-side before being attached to the
`ui_variant` passed to `GenerationJob`. Defaulting the selector to whatever the currently
selected model variant declares (or `short` if none) keeps it consistent with the "predefined
variant's own script length is preserved" requirement.

## Risks / Trade-offs

- [Risk] `events_scale`'s formula changes `long`'s cost estimate by ~6% from today's hardcoded
  `8.0` → Mitigation: this only affects a pre-spend estimate printed by `--dry-run`, not
  actual billing; documented above and callable out in the PR description.
- [Risk] A custom `min_dialogue` free-text field lets a user type something nonsensical (e.g.
  empty string, or English text in a otherwise-Chinese prompt) → Mitigation: out of scope to
  validate prose content — same trust level as the existing presets' Chinese-language values,
  which are also just interpolated into the prompt unchecked.
- [Risk] Silent fallback to `short` on an unparseable custom string could surprise a user who
  made a typo (e.g. `custom:evnets=20`) → Mitigation: matches the project's existing
  degrade-not-crash convention (`CAUSAL_VALIDATION`/`PIPELINE_MODE`); `generate_script.py`
  already prints the resolved `script_length` to stderr after every run, and the UI will show
  the resolved canonical string too, so the silent fallback is still visible after the fact.

## Migration Plan

No data migration — this is additive parsing logic layered under an unchanged `str` type.
Existing `.env` files, CLI invocations, and `eval/model_variants.json` entries using
`short`/`medium`/`long` continue to work identically (verified by the byte-for-byte prompt
regression tests). Rollout is a normal code change with no feature flag needed, since the new
`custom:` syntax is purely additive and any prior data with a non-preset string already fell
back to `short` before this change too.
