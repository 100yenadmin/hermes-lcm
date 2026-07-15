# Pack A / PR-1 outcome

## Result

Implemented the temporal-rollup storage substrate with no engine, compaction,
builder, or tool wiring. The new config flag remains off by default, and no
runtime path constructs or calls `RollupStore`.

## File map

- `db_bootstrap.py:30` bumps `SCHEMA_VERSION` from 5 to 6.
- `db_bootstrap.py:272-306` creates `lcm_rollups`, the partial ready-period
  index, `lcm_rollup_sources`, and `lcm_rollup_state` idempotently.
- `db_bootstrap.py:740-745` installs and records `v6_temporal_rollups` through
  the shared migration runner.
- `rollup_store.py:27-299` adds connection/bootstrap handling, locked-write
  diagnostics, transactional CRUD/state transitions, day-to-week/month
  staleness, window reads, cursor state, source replacement, and app-level
  source purge integrity.
- `config.py:299-302` maps the four new `LCM_*` environment variables.
- `config.py:481-486` adds the inert/default-off `LCMConfig` fields.
- `tests/test_rollup_store.py:52-333` adds 12 tests for migration idempotency,
  v5 upgrade, newer-version refusal, CRUD/rebuild, app-level source integrity,
  inclusive scoped reads, staleness cascade, uniqueness, cursors, purge, and
  config defaults/environment overrides.
- `tests/test_lcm_core.py:26-30,1697,1746,2714,3294,3346` replaces five stale
  schema-version-5 literals with the shared `SCHEMA_VERSION` constant required
  by the mandated v6 bump.

## Acceptance

- `pytest -q tests/test_rollup_store.py`: **12 passed**.
- Full pytest through the repository's host-stub harness: **1618 passed, 1
  skipped, 12 xfailed, 3 failed**. The same three failures reproduce on the
  untouched `/Volumes/LEXAR/repos/hermes-lcm-upstream-ro` at `31675c6` (**1606
  passed, 1 skipped, 12 xfailed, 3 failed**):
  - `TestEngineABC::test_positive_preflight_clears_prior_noop_status`
  - `test_path_containment_within_allowed_base`
  - `test_configured_externalization_path_inside_allowed_base_accepted`
- `ruff check .`: **clean**.
- `PYTHON=python3 scripts/validate_release.sh --full --keep-going --output
  /tmp/validate-a1.log`: all compile, shell, diff, focused pytest (**409
  passed**), benchmark, smoke-stress, and release-stress gates passed. Full and
  low-FD pytest each reported only the same three approved baseline failures
  (**1618 passed, 1 skipped, 12 xfailed, 3 failed**); the low-FD gate completed
  normally and did not wedge. The validator exits 1 because it does not encode
  the dispatch's macOS baseline exception. Checklist:
  `/tmp/validate-a1.log/validation-checklist.md`.

## Convention notes

- The required schema bump made five existing `test_lcm_core.py` assertions
  stale. They were changed only to use the repository's shared version
  constant; no test behavior beyond version tracking changed.
- `lcm_rollup_sources` deliberately has no SQL foreign key, matching the
  existing app-level summary-source convention. Purge and source replacement
  are transactional in `RollupStore`.
- No unresolved schema/store convention questions remain.
