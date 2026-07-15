# Pack A / PR-4 outcome

## Result

Implemented the operator-facing temporal-rollup surfaces on
`feat/temporal-introspection`. `lcm_inspect` now always exposes a stable,
read-only `temporal_rollups` block; `/lcm rollups` renders the same current-session
data; and `/lcm rollups rebuild <day|week|month|all> [date]` marks normalized UTC
targets stale and synchronously attempts no more than
`rollup_builds_per_pass` builds.

## File map

- `tools.py` adds the shared rollup status payload: enabled flag,
  ready/stale/building/failed counts for day/week/month, oldest stale age,
  last-build cursors/timestamps, and last error. Disabled, empty, unbound, and
  query-degraded paths retain a well-formed zero/null shape. Inspect/status
  perform metadata-only SQLite reads and make no LLM calls.
- `command.py` adds `/lcm rollups`, bounded synchronous rebuild dispatch,
  UTC day/week/month normalization, per-target outcomes, flag-off handling,
  help text, and reuse of the engine's existing summary circuit breaker and
  spend guard.
- `docs/operator-guide.md` documents enablement, all rollup tuning controls,
  inspect/status/rebuild operations, UTC/default-date semantics, degradation,
  and LLM-spend safeguards.
- `docs/retrieval-tools.md` cross-links `lcm_recent` degradation behavior to the
  operator workflow.
- `tests/test_rollup_introspection.py` covers enabled-with-data, enabled-empty,
  disabled, human status, no-LLM inspect/status behavior, bounded rebuilds, and
  disabled rebuild refusal.

No changes were made to `engine.py`, `rollup_builder.py`, or `rollup_store.py`.

## Acceptance

- Focused introspection/command plus stacked temporal store, builder, and
  retrieval suites: **140 passed**.
- `ruff check .`: **clean**.
- `git diff --check`: **clean**.
- Full feature-branch pytest through the repository host-stub harness: **1660
  passed, 1 skipped, 12 xfailed, 3 failed**.
- Full pytest against untouched
  `/Volumes/LEXAR/repos/hermes-lcm-upstream-ro` at `31675c6`, using an isolated
  `HOME`: **1606 passed, 1 skipped, 12 xfailed, 3 failed**. The feature branch
  and upstream baseline have the exact same failures:
  - `TestEngineABC::test_positive_preflight_clears_prior_noop_status`
  - `test_path_containment_within_allowed_base`
  - `test_configured_externalization_path_inside_allowed_base_accepted`
- `PYTHON=python3 scripts/validate_release.sh --full --keep-going --output
  /Volumes/LEXAR/Codex/hermes-lcm-temporal-pr4/validate-release`: compile,
  shell, diff, focused pytest (**409 passed**), benchmark smoke, stress smoke,
  and release stress all passed. Full and low-FD pytest each reported only the
  same three approved baseline failures (**1660 passed, 1 skipped, 12 xfailed,
  3 failed**). The validator exits 1 only because it does not encode the
  approved macOS baseline exception. Checklist:
  `/Volumes/LEXAR/Codex/hermes-lcm-temporal-pr4/validate-release/validation-checklist.md`.

## Contract notes

- Rollup status and rebuilds are scoped to the foreground session reported by
  `current_session_id`; no cross-session rollup content is exposed.
- A supplied rebuild date is interpreted as UTC. Without a date, the current
  UTC date is used. Week targets normalize to Monday, month targets to day one,
  and `all` attempts day then week then month within the shared build limit.
- Rebuild targets beyond the configured bound remain stale and are printed as
  not attempted. Missing source summaries produce a clean per-target outcome
  instead of an exception.
- Extending `lcm_inspect` follows the lower-footprint default in the dispatch.
  If the maintainer prefers a separate `lcm_rollup_debug` tool, the shared
  `_temporal_rollups_status` payload can be wrapped by a new handler without
  changing its queries or the `/lcm` surface.
- `SPEC.md` remains the untracked dispatch input and was not committed. No push
  or GitHub CLI operation was performed.
