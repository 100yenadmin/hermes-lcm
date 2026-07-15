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

---

# FIXSPEC3 — codex-bot round-2 findings (branch docs/embeddings-setup-guide)

Per-item disposition against the CURRENT re-stacked heads. Baseline for the
"0 new failures" rule is upstream-ro @ `31675c6`: raw pytest is **38 failed /
636 passed / 13 collection errors**, every failure/error rooted in the missing
`agent` monorepo (engine-integration + packaging/host/path/benchmarking). After
this work: **38 failed / 647 passed / 13 errors** — the failed-test set and the
collection-error-file set are **byte-identical** to baseline (0 new; +11 passing
from the new tests). `_lcm_grep_full_text` byte-identity preserved:
sha256 `fbf5ece961667d5bbf5673e85f2428d7948d66bd4f1c537976226232a47d8bad`.

## Dispositions

1. **latest_at / messages.source legacy-DB crash — FIXED (A).** `vector_store._filtered_candidate_indexes`
   used `COALESCE(sn.latest_at, sn.created_at)` unconditionally (only `suppressed_at`
   was PRAGMA-guarded); `_source_allowed_ids` referenced `m.source` unconditionally.
   A VectorStore-only worker DB predating the DAG `latest_at` / MessageStore
   `source` migrations crashed time-/source-scoped KNN. Now: `recency_expr`
   falls back to `created_at` when `latest_at` is absent; `_source_allowed_ids`
   PRAGMA-detects `messages.source` and skips the source filter when absent.
2. **node_id CAST defeats the INTEGER PK index — FIXED (A).** `record_embedding`
   (`WHERE CAST(node_id AS TEXT) = ?`) and both candidate JOINs (`_filtered_candidate_indexes`,
   `_source_allowed_ids`) now bind/cast the *param* (`node_id = ?` / `sn.node_id = CAST(t.id AS INTEGER)`),
   so the `INTEGER PRIMARY KEY` index is used. `_as_node_id` coerces non-integer
   ids to None (never a match) instead of scanning.
3. **Bounded scan sorted every vector — FIXED (A).** Replaced `_bounded_rows`
   (`ORDER BY m.embedded_at DESC, v.rowid DESC LIMIT ?` → `USE TEMP B-TREE FOR
   LAST TERM`) with `_candidate_ids_by_recency` (`ORDER BY m.embedded_at DESC`
   served directly by `idx_lcm_embedding_meta_identity_embedded_at`, EXPLAIN
   confirms no temp B-tree) + `_load_vectors_for_ids`.
4. **Per-call scratch temp tables — FIXED (A).** `_temp_id_table` now uses a
   `uuid4`-suffixed name and `DROP`s in `finally`; two overlapping candidate
   sets no longer clobber (the old fixed name + `DELETE` wiped the sibling set).
5. **Resolve active profile under the write lock — FIXED (A).** `record_embedding`
   now resolves the profile + normalizes the vector INSIDE `_write_transaction`,
   so a concurrent active-provider switch cannot let it read identity A and
   write under B.
6. **Bounded path applied the limit before the filter — FIXED (A).** The
   no-numpy `knn` branch now enumerates all candidate ids in recency order,
   filters (`_filtered_candidate_indexes`), THEN takes the most-recent
   `bounded_scan_rows` survivors and loads only their vectors — a filtered match
   outside the recent window is no longer lost. (numpy path was already correct.)
7. **KNN resolved by model_id alone — FIXED (A + D).** `knn(..., provider=...)`
   + `_resolve_profile(model, provider=...)` resolve by the full configured
   identity; `tools._run_knn` passes `provider=provider.provider_id`. Switching
   provider A→B for one model name now scores against B's vectors (or degrades
   to FTS when B is unbackfilled: coverage `none`).
8. **purge_embeddings_for_nodes unwired — FIXED wiring (A); orphan-filter ALREADY-PRESENT.**
   The orphan defense (a filtered match must exist in `summary_nodes`) is already
   enforced by the inner JOIN in `_filtered_candidate_indexes`/`_source_allowed_ids`,
   which runs on every knn — proven by `test_orphaned_embeddings_are_not_ranked_and_purge_reclaims`.
   The purge is now wired: `engine.on_session_reset` collects the to-be-deleted
   node_ids (`dag.session_node_ids{,_below_depth}`) and calls the new guarded
   `engine._purge_embeddings_for_nodes` (no-op unless `embeddings_enabled`);
   `command._delete_clean_candidates_atomically` (doctor clean apply) captures
   deleted node_ids and calls the same helper after commit.
9. **Redact scalar error strings — ALREADY-FIXED (no change).** `embedding_provider._scrub_response_body`
   already maps EVERY string value to `[REDACTED]` (lines 276-277), so a scalar
   `error`/`message`/nested detail echoing input is never logged raw; status is
   logged separately as a formatted int.
10. **Backfill tripped the per-minute spend guard — FIXED (B).** `resolve_provider(config, for_backfill=True)`
    builds the provider with `EmbeddingSpendGuard(max_calls=0)` (guard disabled,
    circuit breaker retained); `command._embedding_backfill_apply` passes
    `for_backfill=True`. Interactive query embedding keeps the 60/min guard.
11. **Enforce role in semantic — ALREADY-FIXED (no change).** `tools._lcm_grep_semantic`
    already degrades to full_text when `role` is set (line ~1455). Reinforced by
    the item-12/13 contract guards below.
12 & 13. **Conversation-lane + raw-only contract — FIXED together (D).** The
    advertised `schemas.LCM_GREP` contract returns raw-message hits only for
    broader scopes (`all`/`session`) and for `role`/`time_from`/`time_to`, and
    `conversation_id` is a message-lane filter that `full_text` treats as
    raw-only. A summary node has no single lane and is cross-session, so the
    semantic arm now **degrades to the raw full_text path** (which filters at the
    message-row level — the correct lane filter) whenever `session_scope != current`,
    a time bound is set, or `conversation_id` is set — instead of the round-2
    behavior of resolving conversation→sessions and returning cross-session /
    wrong-lane summary hits. This aligns the semantic contract with full_text +
    the maintainer (Tosko4 #394) review. Round-2 tests that asserted the old
    (contract-violating) behavior were updated to assert the degradation.

## Per-commit file:line manifest

- **A (#390 vector_store/purge):** `vector_store.py` (`_resolve_profile`+provider ~215;
  `_as_node_id` ~257; `record_embedding` resolve-under-lock + int lookup ~330-370;
  `_temp_id_table` uuid+drop ~426; `_candidate_ids_by_recency`/`_load_vectors_for_ids`
  ~540-600; `_source_allowed_ids` source-guard + int join ~600-640; `_filtered_candidate_indexes`
  latest_at-guard + int join ~660-720; `knn` provider + filter-before-bound ~780-860);
  `dag.py` (`session_node_ids`/`session_node_ids_below_depth` ~255); `engine.py`
  (`on_session_reset` purge ~2977; `_purge_embeddings_for_nodes` ~2983);
  `command.py` (`_delete_clean_candidates_atomically` purge ~1440); `tests/test_vector_store.py` (+9 tests).
- **B (#392 provider/backfill):** `embedding_provider.py` (`resolve_provider(for_backfill=)` ~772);
  `command.py` (`resolve_provider(..., for_backfill=True)` ~2257); `tests/test_embedding_provider.py`
  (+1 test); `tests/test_embedding_backfill.py` (stubs accept `**_kw`).
- **D (#394 tools search):** `tools.py` (`_lcm_grep_semantic` raw-only degrade guards ~1455;
  `_run_knn` provider ~1511); `tests/test_embedding_search_modes.py` (2 rewritten + 1 new).
- **E (#395 docs):** `docs/retrieval-tools.md` (semantic filter contract ~75); `FIX-OUTCOME.md`.

## Counts

13 items: **10 fixed** (1-8, 10, 12&13 counted as the 11th/12th sub-fix under one
change), **3 already-fixed** (9, 11, and the orphan-filter half of 8). 13 new/updated
tests. ruff clean. 0 new pytest failures vs baseline; validate_release fails only
on the 14 pre-existing `test_packaging_install` agent-monorepo wedge tests (all in
the baseline failed set), every other gate green.
