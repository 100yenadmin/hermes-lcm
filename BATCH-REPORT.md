# V2 re-baseline batch report

- Branch: `batch/v2-rebaseline`
- Baseline: `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32`
- Binding spec: `SPEC-V2-REBASELINE-BATCH.md`
- Binding architecture decisions: `fork/docs/program-architecture:bench/DECISIONS-R3-UPSTREAM-ARCH.md`
- Upstream follow-up sources: `a53276cf896280fab183cfcc811c9fd5bb49c522`, `ae0f06847247aeb8abb5b7a055bdd14bc0039ff7`
- Git mutation: none; all changes remain unstaged in the working tree.

## Validate-then-port table

Each upstream hunk was checked against the current `fork/main` baseline before editing. No fix was already present or superseded in full.

| Source | Fix | Current-main validation | Disposition |
|---|---|---|---|
| `a53276c` | `evidence_compiler.py`: finite coverage counts distinct grounded `exact_ref` values | Current code still used `len(validated)`, so duplicate claims over one grounded ref could certify coverage. | **ported** |
| `a53276c` | `query_view_store.py`: hit CAS checks generation/rowcount and re-reads readiness | Current hit update had no generation predicate or post-update readiness confirmation. | **ported** |
| `a53276c` | `adaptive_retrieval.py`: persisted slot refs intersect the selected evidence set | Current manifest persisted every resolved slot ref, including unselected evidence. | **ported** |
| `a53276c` | `trajectory_store.py`: reject an existing incompatible trajectory schema before repair DDL | Current constructor ran `_init_schema()` before validating the stored trajectory schema version. | **ported** |
| `a53276c` | `vector_store.py`: deadline-expired paths do not start `COUNT(*)` | The original sites were still missing the fix, while #184 had added resident and streaming paths with the same invariant. | **ported**, adapted to current resident, binary-prescreen, candidate-enumeration, streaming, summary, and chunk paths |
| `ae0f068` | `adaptive_retrieval.py`: `mark_failed` cleanup is best-effort | Current exception handler allowed a cleanup failure to replace the original build failure. | **ported** |

## Unit results

### Unit 1: six upstream follow-up fixes

Command:

```text
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub python3 -m pytest -q tests/test_adaptive_retrieval.py::test_persisted_slot_refs_include_only_selected_evidence tests/test_adaptive_retrieval.py::test_query_view_cleanup_cannot_replace_the_build_failure tests/test_evidence_compiler.py::test_duplicate_grounded_ref_cannot_certify_finite_coverage tests/test_query_view_store.py::test_hit_confirmation_rechecks_generation_after_source_mutation tests/test_trajectory_store.py::test_newer_trajectory_schema_is_rejected_before_fts_repair tests/test_vector_store.py::test_full_scan_budget_stops_early_and_reports_bounded tests/test_vector_store.py::test_full_scan_budget_includes_candidate_enumeration tests/test_vector_store.py::test_full_scan_absolute_deadline_stops_between_batches tests/test_vector_store.py::test_resident_deadline_does_not_start_a_count_query tests/test_prescreen_flip_blackout.py::test_deadline_bounds_a_synced_binary_summary_prescreen tests/test_int8_two_stage_knn.py::test_chunk_deadline_bounds_a_synced_binary_prescreen
```

Result: **11 passed**.

### Unit 2: D-ARCH-1 event-dedupe certification

Command:

```text
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub python3 -m pytest -q tests/test_evidence_contract.py::test_finite_enumeration_distinguishes_same_entity_events_by_date tests/test_evidence_contract.py::test_finite_enumeration_returns_dated_and_undated_count_uncertified tests/test_evidence_contract.py::test_finite_enumeration_collapses_repeated_undated_mentions
```

Result: **3 passed**.

### Unit 3: D-ARCH-2 adjacency-reserve backfill

Command:

```text
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub python3 -m pytest -q tests/test_trajectory_store.py::test_unused_adjacency_reserve_backfills_the_full_ranked_limit tests/test_trajectory_store.py::test_partial_adjacency_reserve_backfills_in_rank_order tests/test_trajectory_store.py::test_full_adjacency_reserve_keeps_the_existing_composition
```

Result: **3 passed**.

### Unit 4: D-ARCH-3 date-anchor trust boundary

Command:

```text
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub python3 -m pytest -q tests/test_reasoning.py::test_compute_accepts_caller_anchor_only_when_it_agrees_with_sidecar tests/test_reasoning.py::test_compute_sidecar_overrides_disagreeing_caller_anchor tests/test_reasoning.py::test_compute_without_sidecar_marks_temporal_result_low_trust
```

Result: **3 passed**.

### Unit 5: batch-2 `how long ago` operand fix

Command:

```text
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub python3 -m pytest -q tests/test_reasoning.py::test_planner_uses_explicit_cardinality_and_interval_units
```

Result: **1 passed**. The regression proves that `how long ago` plans one evidence-date operand plus the question-date anchor.

## Affected-file validation

Spec acceptance files:

```text
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub python3 -m pytest -q tests/test_trajectory_store.py tests/test_evidence_contract.py tests/test_reasoning.py
```

Result: **81 passed**.

Upstream-port affected test files:

```text
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub python3 -m pytest -q tests/test_adaptive_retrieval.py tests/test_evidence_compiler.py tests/test_query_view_store.py tests/test_vector_store.py tests/test_prescreen_flip_blackout.py tests/test_int8_two_stage_knn.py
```

Result: **166 passed**.

Changed-file lint:

```text
ruff check adaptive_retrieval.py evidence_compiler.py query_view_store.py reasoning.py requirements_compiler.py tools.py trajectory_store.py vector_store.py tests/test_adaptive_retrieval.py tests/test_evidence_compiler.py tests/test_evidence_contract.py tests/test_int8_two_stage_knn.py tests/test_prescreen_flip_blackout.py tests/test_query_view_store.py tests/test_reasoning.py tests/test_trajectory_store.py tests/test_vector_store.py
```

Result: **all checks passed**.

## Decision fidelity

- D-ARCH-1: dated candidates dedupe by the existing event key plus resolved date; undated candidates retain the existing collapse. Counts with any undated contributor are returned with `finite_coverage=false`.
- D-ARCH-2: adjacency remains first priority for its reserve; unused slots return to ranked candidates in rank order.
- D-ARCH-3: the engine occurrence-date sidecar is the trusted anchor; disagreement overrides the caller and is noted; absence returns a low-trust, uncertified temporal result.
- Batch-2: `how long ago` uses one evidence-date operand and the question-date anchor.
- Deviations from the binding D-ARCH text: **none**.

## Proof boundary

This report proves the unstaged working-tree implementation, focused unit behavior, affected test-file results, and changed-file lint only. It does not prove merge, remote CI, paired V2 benchmark results, V1 sanity-slice results, release, deployment, or runtime adoption.
