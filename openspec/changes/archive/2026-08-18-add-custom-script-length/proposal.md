## Why

`SCRIPT_LENGTH` only accepts three fixed presets (`short`/`medium`/`long`), which map to
four prompt-level targets in `crew/tasks.py::_LENGTH_TARGETS` (`events`, `chapters`,
`beats_per_chapter`, `min_dialogue`). Users who want something in between or beyond those
presets (e.g. "20 events, 4 chapters, 3 beats per chapter") have no way to express it.
The Streamlit generation mode (`ui/app.py`'s 生成 mode) currently has no length control at
all — it's only reachable indirectly via the `.env` `SCRIPT_LENGTH` var before starting the
app. This change adds a fourth, user-specified option across every entry point that already
resolves a script length.

## What Changes

- Add a `custom:events=N,chapters=N,beats_per_chapter=N,min_dialogue=TEXT` string syntax,
  parsed by a new dependency-free module (`src/bixiascribe/length.py`) that both `config.py`
  and `crew/tasks.py` can use without a circular import.
- Custom syntax allows partial specification (e.g. `custom:events=20`) — unspecified fields
  are derived from `events`. An unparseable custom string falls back to `short`, matching the
  existing degrade-not-crash convention used by `CAUSAL_VALIDATION`/`PIPELINE_MODE`.
- `.env`'s `SCRIPT_LENGTH` and CLI `--script-length` (`generate_script.py`,
  `eval_generation.py`) accept the custom syntax in addition to the three presets.
- `eval/model_variants.json`'s `Variant.script_length` field accepts the same syntax (no
  schema change needed — it's already a free-form string).
- Streamlit's 生成 mode gets a new "劇本篇幅" selector (short/medium/long/自訂) with an
  expander of four input fields when 自訂 is picked, mirroring the existing 模型變體 → 自訂
  UX pattern.
- Fix: `ui/app.py`'s `ui_variant` construction currently drops the selected variant's
  `script_length`/`session_doc_max_tokens` when building the UI-run variant — this change
  carries them forward so picking a variant that declares its own `script_length` (e.g.
  `long-cheap`) actually uses it.
- `RunReport.script_length` / the JSONL row continues to be a plain string, now holding the
  canonicalized custom spec (fully-resolved, all four fields filled in) when a custom length
  was used, so a logged run is self-describing without cross-referencing the env var at run
  time.

## Capabilities

### New Capabilities
- `script-length`: parsing, validation, and resolution of the script-length target
  (preset or custom) that flows from `.env`/CLI/UI/eval-variants into the prompt text
  `crew/tasks.py`'s task builders produce, and into `RunReport`/JSONL for logging.

### Modified Capabilities
(none — no existing capability spec covers this behavior yet)

## Impact

- New: `src/bixiascribe/length.py`, `tests/test_length_spec.py`
- Modified: `src/bixiascribe/crew/tasks.py` (`_LENGTH_TARGETS`/`_length_target` become a thin
  wrapper over `length.py`, prompt text unchanged — `tests/test_script_length.py`'s
  byte-for-byte regression guards must stay green), `src/bixiascribe/config.py`
  (`SCRIPT_LENGTH` validation), `scripts/generate_script.py` and `scripts/eval_generation.py`
  (`--script-length` argparse validation, `eval_generation.py`'s `_LENGTH_SCALE` cost-estimate
  table becomes derived from `events` instead of a fixed short/medium/long lookup), `ui/app.py`
  (新增篇幅選擇器 + `ui_variant` construction fix)
- No schema changes to `Script`/`Event`/`Beat`/`RunReport`'s field types — `script_length`
  remains a plain string end to end, preserving JSONL backward compatibility.
- Docs: `CLAUDE.md`'s "Script length" section, `.env.example`, `README.md`'s length-related
  notes.
