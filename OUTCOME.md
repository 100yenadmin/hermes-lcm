# Pack B / PR-1 outcome

## Result

Implemented the default-off embedding storage substrate and the fixed #386
compute ladder: lazy NumPy full scan, bounded most-recent pure-Python scan, and
`none` coverage when no current profile or vectors exist. The module is not
wired into the engine, providers, HTTP, backfill, or tools.

## File map

- `db_bootstrap.py:30-31` bumps the schema to v6 and isolates the embedding
  migration version so stacking requires changing only the two version
  constants.
- `db_bootstrap.py:273-306` creates the profile, metadata, partial metadata
  index, and float32-BLOB vector tables idempotently.
- `db_bootstrap.py:740-748` installs and records `v6_embeddings` through the
  shared migration runner without changing the future-schema refusal path.
- `config.py:295-296,458-460` adds the two scalar environment mappings and the
  inert defaults (`embeddings_enabled=False`, bounded rows `2000`).
- `vector_store.py:38-48` adds the list-compatible KNN result with result-level
  coverage.
- `vector_store.py:51-177` adds connection/bootstrap handling, config loading,
  locked transactional writes, current-profile selection, immutable profile
  registration, and dimension locking.
- `vector_store.py:179-264` normalizes and float32-packs embeddings, replaces
  vector and metadata rows transactionally, copies summary source-token counts,
  invalidates the matrix cache on writes, and purges hard-deleted node vectors.
- `vector_store.py:266-375` implements the `(model, max(rowid), row_count)`
  NumPy matrix cache and the bounded most-recent array-module fallback.
- `vector_store.py:377-475` applies summary existence/suppression, time, and
  conversation filters after vector scoring with filter-aware overfetch, then
  returns the requested coverage.
- `tests/test_vector_store.py:64-395` adds 14 tests covering migrations,
  version guard, profile semantics, hand-computed cosine ordering,
  normalization, NumPy absence, bounded-window enforcement, suppression,
  purge, filters, cache invalidation, no-coverage behavior, and config.
- `tests/test_lcm_core.py:26-30,1697,1746,2714,3294,3346` replaces five stale
  schema-v5 literals with `SCHEMA_VERSION`, the same mechanical compatibility
  edit required by Pack A's v6 bump.

## Acceptance

- `pytest -q tests/test_vector_store.py`: **14 passed**.
- Canonical standalone full pytest (repository host-stub harness, isolated
  database): **1620 passed, 1 skipped, 12 xfailed, 3 failed**. The same three
  failures reproduce on untouched
  `/Volumes/LEXAR/repos/hermes-lcm-upstream-ro` at `31675c6` (**1606 passed, 1
  skipped, 12 xfailed, 3 failed**):
  - `TestEngineABC::test_positive_preflight_clears_prior_noop_status`
  - `test_path_containment_within_allowed_base`
  - `test_configured_externalization_path_inside_allowed_base_accepted`
- `ruff check .`: **clean**.
- `PYTHON=python3 scripts/validate_release.sh --full --keep-going --output
  /tmp/validate-b1.log`: compile, shell, diff, focused pytest (**409 passed**),
  benchmark smoke, stress smoke, and release stress all passed. Both full and
  low-FD pytest gates wedged in the dispatch-named Barrier migration race with
  frozen logs and were terminated after exceeding 10 minutes, so the validator
  correctly exits 1. The concurrent migration test then passed in isolation
  three consecutive times (0.02-0.03 seconds each), and a fresh bounded full
  run completed with only the three approved baseline failures above. Evidence:
  `/tmp/validate-b1.log/validation-checklist.md`.

## Schema collision / stacking

Pack A and Pack B independently claim schema v6 (`v6_temporal_rollups` versus
`v6_embeddings`). Before stacking, the orchestrator must choose the order and
renumber the second migration to v7. For this branch that is a two-line version
change at `db_bootstrap.py:30-31`; the migration step name is derived from the
embedding migration constant.

## Convention notes

- Base `31675c6` has no `summary_nodes.suppressed_at` column even though the
  dispatch requires suppression filtering. The post-scan join detects and
  enforces that exact column when present (the test supplies the forward schema
  shape), while remaining valid on the pinned base. Hard-deleted summaries are
  always excluded by the join, and app-level purge removes their stored rows.
- The current summary schema names the conversation field `session_id`; KNN's
  `conversation_ids` filter therefore targets `summary_nodes.session_id`.
  `since` uses `COALESCE(latest_at, created_at)`, matching DAG recency semantics.
- The only scope addition beyond the four implementation/test files named in
  the dispatch is the mechanical `tests/test_lcm_core.py` schema-constant edit
  required for an independently green schema bump; it matches Pack A exactly.
- No provider, HTTP, backfill, engine, compaction, externalization, or tool
  wiring was added.
