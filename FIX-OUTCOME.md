# Embeddings train — FIXSPEC2 outcome (Tosko4 architecture-gate CHANGES_REQUESTED)

All maintainer findings across PRs #390/#392/#393/#394/#395 implemented on
`docs/embeddings-setup-guide`, split into five re-stackable commits by owning PR
file group. No push, no `gh`, no PR-body edits.

`_lcm_grep_full_text` is **byte-identical** to the branch tip (sha256 of the
function body unchanged: `b09ca30d8e8a7f80`); existing full_text behavior is
untouched.

## Acceptance

- Focused suites (all green):
  - `tests/test_vector_store.py` — 20 passed
  - `tests/test_embedding_provider.py` — 26 passed
  - `tests/test_embedding_backfill.py` — 13 passed
  - `tests/test_embedding_search_modes.py` — 20 passed
  - `tests/test_db_bootstrap_fts.py` — 18 passed
  - Combined focused bundle: **97 passed**.
- `ruff check` (changed files and whole repo): **All checks passed**.
- `scripts/validate_release.sh --full --keep-going`
  (`PYTHON=$(which python3)`): FAILED with 3 gate failures — `focused pytest`,
  `pytest full`, `pytest low fd`. **Barrier-wedge:** every failing test is a
  pre-existing `agent`-monorepo/engine-integration gap
  (`test_packaging_install`, `test_auto_focus_topic`, `test_host_capability`,
  `test_lcm_engine`, `test_path_*`), rooted in `engine.py:19 from
  agent.context_engine import ContextEngine` — the plugin expects to run inside
  the hermes-agent monorepo where `agent` resolves. **Zero** failing tests lie
  in any file this change touches.
- 3-baseline rule vs `hermes-lcm-upstream-ro` @ `31675c6`: my change introduces
  **zero** new failures or collection errors. Raw pytest has 13 pre-existing
  collection errors (identical set on the committed tip and this branch) plus
  the pre-existing `agent`-rooted failures above. Verified per-file-isolated:
  the committed tip and this branch produce identical results for every
  affected file (e.g. `test_packaging_install` 18 failed / 7 passed on both;
  `test_host_capability` 5 failed / 5 passed on both). The batch-run count
  discrepancy was test-ordering/`sys.modules` noise, not a regression.

---

## COMMIT A → #390 — `vector_store.py`, `db_bootstrap.py` (+ tests)

### 1. Schema versioning (blocker 1) — no core bump for opt-in tables
- `db_bootstrap.py:37` `SCHEMA_VERSION = 5` (reverted from 6); the numeric
  `_EMBEDDING_MIGRATION_VERSION`/v6 step removed.
- `db_bootstrap.py:771` `run_versioned_migrations` no longer creates embedding
  tables; it ends at v5.
- `db_bootstrap.py:279` `ensure_embedding_tables` is the lazy creator.
- `vector_store.py:150` `_ensure_embedding_schema` creates the tables from
  VectorStore init (CREATE TABLE IF NOT EXISTS) recorded via
  `mark_migration_step_complete(conn, "embeddings_v1")`.
- **Decision:** identical to the temporal train — neither train touches the
  numeric counter, so no v6 collision; disabled install stays v5 with no
  embedding tables, openable by base code.

### 2. Canonical profile identity (blocker 2)
- `vector_store.py:47` `_identity_hash(provider, model, revision, dim, dtype,
  byteorder, task)`; schema in `db_bootstrap.py:279` keys profile + vectors +
  meta on `identity_hash` (float32/little-endian/summary recorded so a future
  change is detectable).
- `vector_store.py:248` `register_profile` — a different identity for the same
  `model_name` is a NEW row (no metadata clobber); an existing identity is
  reactivated in place; exactly one profile stays active.
- **Decision:** current/active selection is by identity; switching config
  A→B→A reactivates A with its vectors intact (no re-backfill).

### 3. Durable data-version counter + id-lookup scaling (blocker 3)
- `vector_store.py:392` `_bump_data_version` bumped inside the same write
  transaction as every vector write (`record_embedding`) / delete
  (`purge_embeddings_for_nodes`); included in the NumPy matrix cache key
  (`_numpy_rows`).
- `vector_store.py:134` connection opened `isolation_level=None` (autocommit)
  with explicit `BEGIN IMMEDIATE` writes, so a long-lived reader observes
  another process's committed counter bump (a pinned WAL snapshot otherwise
  masked it — root cause found and fixed).
- `vector_store.py:411` `_temp_id_table` + `_filtered_candidate_indexes`
  (`619`) replace the giant `WHERE id IN (...)` with a chunked temp-table JOIN
  (the ~33k-id `SQLITE_MAX_VARIABLE_NUMBER` failure the maintainer hit in #395).

### 4. KNN filter signature extension (owned by A per spec)
- `vector_store.py:671` `knn` gains `until` (time_to) and `source`;
  `_source_allowed_ids` (`578`) enforces source via the same recursive
  source-tree walk the DAG uses, all before the top-k cap.

### New/updated tests (`tests/test_vector_store.py`)
- `test_core_migrations_omit_embedding_tables`,
  `test_vector_store_creates_embedding_tables_lazily_and_idempotently`
  (flag-off → v5 + no tables; enabled → tables, still v5).
- `test_profile_identity_distinguishes_provider_without_clobber`,
  `test_switch_provider_a_b_a_reactivates_without_rebackfill`.
- `test_data_version_bump_invalidates_cross_process_cache`,
  `test_large_id_metadata_resolve_scales_past_variable_limit` (40k ids).
- `test_time_to_filter_excludes_before_top_k`,
  `test_source_filter_enforced_before_top_k`.

## COMMIT B → #392 — `embedding_provider.py`, `config.py` (+ tests)

- **Voyage item cap:** `embedding_provider.py:33` `_VOYAGE_MAX_BATCH_ITEMS=1000`;
  `embed_documents` splits on the item cap as well as the token budget
  (`:499`); configurable via `config.py` `embedding_max_batch_items` (default
  1000), threaded through `resolve_provider`.
- **Absolute retry deadline:** `_request` takes `deadline_budget_s`
  (`embedding_provider.py:362`); `_sleep_within_deadline` (`:450`) refuses a
  backoff that would blow the deadline. `embed_query_interactive` passes
  `deadline_budget_s=timeout`, so a 0.02s budget returns in ~0.02s, not ~1.5s.
- **Ollama `truncate: false`:** `embedding_provider.py:605` so oversized input
  fails loudly; Ollama `_request` also honors the same absolute deadline.
- **FastEmbed query API:** `embedding_provider.py:743` `embed_query` →
  `model.query_embed`, `embed_documents` → `model.embed`, preserving
  query/passage asymmetry.

### New/updated tests (`tests/test_embedding_provider.py`)
- `test_voyage_batch_splits_on_item_count_cap` (1001 → 2 requests),
  `test_voyage_item_cap_is_configurable`,
  `test_voyage_absolute_deadline_bounds_total_retry_time`.
- Ollama payload asserts `truncate: False`; FakeFastembedModel gains
  `query_embed`; `test_fastembed_documents_and_queries_use_distinct_apis`.
- Dim-lock test replaced by
  `test_warmup_command_new_dim_is_a_distinct_identity_no_clobber`;
  interactive-timeout assertions relaxed to the absolute-deadline semantics.

## COMMIT C → #393 — `command.py` (+ tests)

- **Claim before discovery:** `_embedding_backfill_text` acquires the lease
  first, then re-queries pending rows (`command.py:2237`).
- **Renewable heartbeat lease:** `_BackfillLease` (`:1888`) +
  `_acquire_embedding_backfill_lease` (`:1954`) — owner-CAS renew; only a
  truly-expired lease is stealable; a stolen lease fails renewal and aborts.
- **Truthful status:** `_embedding_backfill_status` (`:2393`) — `complete`
  only when all selected work embedded; `partial`/`failed` otherwise (the
  probe's complete/embedded=0/failed=1 now reads `failed`).
- **Crash-safe in_flight:** `_mark_inflight` (`:2012`) before the provider
  call, cleared on `record_embedding`; NOT-EXISTS discovery re-attempts crashed
  rows without re-billing recorded ones (`_clear_stale_inflight` at `:1998`).
- **Op budget + refresh cadence:** env-tunable
  `LCM_EMBEDDING_BACKFILL_{LEASE_TTL_S,HEARTBEAT_S,BUDGET_S}`; the run stops
  between batches when the budget is exceeded (`stop_reason` reported).
- Adapted `_embedding_current_profile`/`_embedding_pending_rows` to the
  identity-keyed schema.

### New/updated tests (`tests/test_embedding_backfill.py`)
- `test_apply_claims_before_discovery_and_skips_already_embedded`,
  `test_heartbeat_lease_blocks_takeover_until_expiry`,
  `test_inflight_row_is_reattempted_after_crash`,
  `test_operation_budget_stops_run_between_batches`.
- Transient-error test now asserts `status: partial` (was the buggy
  `complete`); refuse-message text + meta-column assertions updated.

## COMMIT D → #394 — `tools.py` (+ tests)

- **Filter enforcement before ranking:** `_lcm_grep_semantic` passes
  `conversation_ids`, `source`, `time_from`, `time_to` into `knn` (enforced
  before the top-k cap); post-loop source/time_to filtering removed.
  `_resolve_semantic_conversation_scope` (`:1349`) resolves `conversation_id`
  to its sessions. `role` cannot be enforced over role-less summaries, so it
  degrades to full_text (which does enforce role) rather than being ignored
  (`:1456`).
- **One absolute deadline:** `deadline` (`tools.py:1487`) spans query-embed +
  vector-store construction + KNN; `_run_within_deadline` (`:1286`) bounds each
  stage. NumPy's one-time ~0.5s import is warmed outside the deadline so it is
  not charged to a tight per-query budget. `_WorkerCapacityError` +
  `BoundedSemaphore(4)` cap live workers so repeated timeouts don't accumulate.

### New/updated tests (`tests/test_embedding_search_modes.py`)
- `test_semantic_role_filter_degrades_to_full_text`,
  `test_semantic_time_to_excludes_ineligible_before_top_k`,
  `test_semantic_source_filter_excludes_ineligible_before_top_k`,
  `test_semantic_conversation_filter_resolves_to_sessions`,
  `test_slow_knn_degrades_within_total_budget`.
- The existing byte-identity test
  (`test_full_text_modes_remain_byte_identical_with_embeddings_on_or_off`)
  still passes; interactive-timeout assertion relaxed to abs tolerance.

## COMMIT E → #395 — `docs/embeddings-setup.md`, `docs/retrieval-tools.md`

- Removed the `lcm_status`/doctor corpus-model-mismatch claim (no such surface
  was added).
- Removed the external-reranker implication (retrieval is RRF-only).
- Latency-budget degrade claim kept and corrected to cover embed **and** KNN.
- Replaced the unverified "50k×1024 ≤69 ms" figure with the validated fact:
  metadata/id resolution now scales past the ~32k SQLite variable limit
  (validated to 40k); numpy top-k is milliseconds warm.
- Provider switching/reactivation documented (now true after the identity fix).
- Voyage free-tier numbers re-verified against the pricing page (2026-07):
  200M free tokens for the voyage-4 group; $0.02/$0.06/$0.12 per M for
  lite/base/large. Ollama `truncate: false` and FastEmbed query-encoding noted.
- `docs/retrieval-tools.md` updated: one absolute deadline over embed+KNN, and
  the before-cap filter-enforcement contract (conversation/source/time; role
  degrades to full_text).

## Deviations from the packet

- Backfill lease/heartbeat/budget knobs live as `LCM_EMBEDDING_BACKFILL_*` env
  reads in `command.py` (commit C) rather than in `LCMConfig` — the packet maps
  `config.py` to commit B, so keeping these operational one-shot-command knobs
  in C preserves the clean per-PR split. `embedding_max_batch_items` (a provider
  tunable) is the one config addition, correctly in B.
- `role`/`conversation_id` for semantic: `role` degrades to full_text (which
  enforces it) instead of a hard error — this still "does not silently ignore"
  and enforces via the arm that can; `conversation_id` is enforced by resolving
  it to sessions. This is the enforcement the spec preferred.

## Open risks

- The pre-existing collection errors and `agent`-monorepo failures are
  environmental (raw pytest outside the hermes-agent monorepo); they are not
  exercised by CI-in-monorepo and are unchanged by this work, but they do mean
  the engine-integration test files are not runnable in this standalone
  checkout. The `validate_release.sh` pytest gates therefore report red on this
  standalone checkout for reasons unrelated to the embeddings surface.
