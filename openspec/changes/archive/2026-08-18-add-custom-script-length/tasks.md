## 1. Core `length.py` module

- [x] 1.1 Create `src/bixiascribe/length.py` with a frozen `LengthSpec` dataclass
      (`events`, `chapters`, `beats_per_chapter`, `min_dialogue`) and the three preset
      dicts moved from `crew/tasks.py::_LENGTH_TARGETS`
- [x] 1.2 Implement `parse_length_spec(value: str) -> LengthSpec`: resolve `short`/`medium`/
      `long` presets, parse `custom:key=value,...` syntax, derive omitted fields from
      `events` (chapters/beats_per_chapter proportionally, min_dialogue tiered), fall back
      to `short` on any unparseable input (including malformed `custom:` values and
      unrecognized non-preset strings)
- [x] 1.3 Implement `LengthSpec.targets` (dict shape matching today's `_LENGTH_TARGETS`
      values) and `LengthSpec.canonical` (fully-resolved `custom:events=...,chapters=...,
      beats_per_chapter=...,min_dialogue=...` string, or the bare preset name for
      short/medium/long)
- [x] 1.4 Implement `LengthSpec.events_scale` (`events / 2`, using a range's lower bound
      when `events` is a range like `"15-24"`)
- [x] 1.5 Write `tests/test_length_spec.py`: preset resolution, full custom spec, partial
      custom spec derivation, malformed custom string falls back to short, unrecognized
      non-preset string falls back to short, `canonical` round-trips through `parse_length_spec`
      again unchanged

## 2. Wire `length.py` into existing prompt/config code without changing prompt text

- [x] 2.1 Update `src/bixiascribe/crew/tasks.py::_length_target()` to call
      `length.parse_length_spec(script_length).targets` instead of the local
      `_LENGTH_TARGETS` dict; remove the now-redundant local `_LENGTH_TARGETS`/
      `_length_target` definitions once callers are confirmed unaffected
- [x] 2.2 Update `src/bixiascribe/config.py`'s `SCRIPT_LENGTH` resolution to validate via
      `length.parse_length_spec` (accept anything that parses, including custom syntax)
      instead of the current three-value membership check
- [x] 2.3 Run `python tests/test_script_length.py` and confirm all byte-for-byte prompt
      regression assertions still pass unchanged

## 3. CLI entry points

- [x] 3.1 `scripts/generate_script.py`: drop `--script-length`'s `choices=(...)` restriction,
      validate the given value via `length.parse_length_spec` up front, update `--help` text
      with a custom-syntax example
- [x] 3.2 `scripts/eval_generation.py`: same argparse change for its `--script-length` flag
- [x] 3.3 `scripts/eval_generation.py::_estimate_matrix_cost()`: replace the hardcoded
      `_LENGTH_SCALE` dict lookup with `length.parse_length_spec(...).events_scale`
- [x] 3.4 Confirm `python scripts/generate_script.py --script-length 'custom:events=20'
      --requirement "測試" --preflight-only` and `python scripts/eval_generation.py --dry-run
      --script-length 'custom:events=20,chapters=4'` both run without error

## 4. Eval variants

- [x] 4.1 Confirm (no code change expected) that `eval/model_variants.json` entries can set
      `script_length` to a custom string end-to-end via `Variant.from_dict` /
      `Variant.to_model_choice` — add a regression test case if `generation.py`'s existing
      tests don't already cover a custom string passing through unchanged

## 5. Streamlit UI

- [x] 5.1 Fix `ui/app.py`'s `ui_variant` construction (around the 開始生成 button handler) to
      carry forward the selected variant's `script_length` and `session_doc_max_tokens`
      instead of dropping them
- [x] 5.2 Add a "劇本篇幅" `st.selectbox` (`short`/`medium`/`long`/`自訂`) to the 生成 mode,
      defaulting to the currently selected model variant's own `script_length` (or `short`
      if the variant doesn't declare one)
- [x] 5.3 When `自訂` is selected, show an expander with four inputs (events, chapters,
      beats_per_chapter, min_dialogue) mirroring the existing 模型變體→自訂 expander pattern;
      join the filled inputs into a `custom:...` string
- [x] 5.4 Attach the resolved script-length string to the `ui_variant`/`GenerationJob` call
      so it overrides the variant's own value only when the user actually changed the
      selector away from the variant's default
- [x] 5.5 Manually verify with `LLM_BACKEND=fake streamlit run ui/app.py`: run one offline
      generation with a custom length and confirm the resulting JSONL row's `script_length`
      is the canonical fully-resolved string

## 6. Docs and full verification

- [x] 6.1 Update `CLAUDE.md`'s "Script length" section to document the `custom:` syntax
- [x] 6.2 Update `.env.example`'s `SCRIPT_LENGTH` comment with a custom-syntax example
- [x] 6.3 Update `README.md` if it documents the three-preset-only behavior anywhere
- [x] 6.4 Run `pytest tests/` and `ruff check .`, confirm both clean
