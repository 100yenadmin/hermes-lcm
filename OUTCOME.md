# Pack A / PR-3 outcome

## Result

Implemented the agent-facing `lcm_recent` retrieval tool on
`feat/temporal-lcm-recent`. Natural periods are parsed into UTC `[start, end)`
windows; ready day/week/month rollups are served newest-first; and missing,
stale, disabled, or sub-day rollup paths transparently fall back to existing
depth-0 summary nodes in the same window. Serving is bounded, read-only, and
makes no LLM calls.

## File map

- `rollup_periods.py` adds the table-driven natural-time parser for `today`,
  `yesterday`, `Nd`, `week`, `month`, `date:YYYY-MM-DD`, and `last Nh`, including
  clean validation for invalid and out-of-range inputs.
- `tools.py` adds ready-rollup selection through
  `RollupStore.ready_rollups_for_window`, stale detection, the time-bounded
  leaf-summary fallback, provenance, stable newest-first sections, limit
  clamping, per-rollup token lines, and the shared 20,000-character response
  ceiling.
- `schemas.py` declares the required `period` argument plus `conversation` and
  `global` scopes.
- `engine.py`, `__init__.py`, and `plugin.yaml` add only the public-tool
  import/list/dispatch/registration plumbing needed to make the new handler
  agent-facing. No engine lifecycle, compaction, builder, or `/lcm` behavior
  changed.
- `README.md` and `docs/retrieval-tools.md` document the tool, three usage
  examples, UTC semantics, provenance, and transparent degradation behavior.
- `tests/test_lcm_recent.py` covers every period form, invalid inputs,
  ready/stale/disabled/sub-day modes, empty windows, scope, limit/order/character
  bounds, and provenance. Existing registration contract fixtures were updated
  for the new public tool.

## Acceptance

- Focused `lcm_recent` plus public registration/packaging contracts: **66
  passed**.
- `ruff check .`: **clean**.
- `git diff --check`: **clean**.
- Full pytest through the repository host-stub harness: **1652 passed, 1
  skipped, 12 xfailed, 3 failed**. An isolated-`HOME` run on untouched
  `/Volumes/LEXAR/repos/hermes-lcm-upstream-ro` at `31675c6` reports **1606
  passed, 1 skipped, 12 xfailed, 3 failed**, with the exact same failures:
  - `TestEngineABC::test_positive_preflight_clears_prior_noop_status`
  - `test_path_containment_within_allowed_base`
  - `test_configured_externalization_path_inside_allowed_base_accepted`
- `PYTHON=python3 scripts/validate_release.sh --full --keep-going --output
  /Volumes/LEXAR/Codex/hermes-lcm-temporal-pr3/validate-release`: compile,
  shell, diff, focused pytest (**409 passed**), benchmark smoke, stress smoke,
  and release stress all passed. Full and low-FD pytest each reported only the
  same three approved baseline failures (**1652 passed, 1 skipped, 12 xfailed,
  3 failed**). The low-FD gate completed normally and did not wedge. The
  validator exits 1 only because it does not encode the approved macOS baseline
  exception. Checklist:
  `/Volumes/LEXAR/Codex/hermes-lcm-temporal-pr3/validate-release/validation-checklist.md`.

## Contract notes

- `conversation` resolves rollups and fallback summaries against the active
  session; `global` uses the global rollup scope and all-session fallback.
- Calendar day windows use ready daily rollups. `week` and `month` select only
  the exact calendar aggregate starting at the requested UTC boundary.
- Any non-ready row in the selected rollup window forces the leaf-summary
  fallback; sub-day windows never attempt rollup serving.
- The spec's allowed-file list excluded the repository's hard-coded public tool
  surfaces. Minimal registration-only edits were required so `lcm_recent` is
  reachable and the synchronized tool-contract suite remains valid.
- `SPEC.md` remains the untracked dispatch input and was not committed. No push
  or GitHub CLI operation was performed.
