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

All commands below use repository-relative test paths and run from the repository
root. `AGENT_STUB_PATH` is operator-local for standalone checkouts that need the
`hermes_lcm` package shim; an installed checkout may omit the `PYTHONPATH` prefix.

### Unit 1: six upstream follow-up fixes

Command:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_adaptive_retrieval.py::test_persisted_slot_refs_include_only_selected_evidence tests/test_adaptive_retrieval.py::test_query_view_cleanup_cannot_replace_the_build_failure tests/test_evidence_compiler.py::test_duplicate_grounded_ref_cannot_certify_finite_coverage tests/test_query_view_store.py::test_hit_confirmation_rechecks_generation_after_source_mutation tests/test_trajectory_store.py::test_newer_trajectory_schema_is_rejected_before_fts_repair tests/test_vector_store.py::test_full_scan_budget_stops_early_and_reports_bounded tests/test_vector_store.py::test_full_scan_budget_includes_candidate_enumeration tests/test_vector_store.py::test_full_scan_absolute_deadline_stops_between_batches tests/test_vector_store.py::test_resident_deadline_does_not_start_a_count_query tests/test_prescreen_flip_blackout.py::test_deadline_bounds_a_synced_binary_summary_prescreen tests/test_int8_two_stage_knn.py::test_chunk_deadline_bounds_a_synced_binary_prescreen
```

Result: **11 passed**.

### Unit 2: D-ARCH-1 event-dedupe certification

Command:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_evidence_contract.py::test_finite_enumeration_distinguishes_same_entity_events_by_date tests/test_evidence_contract.py::test_finite_enumeration_returns_dated_and_undated_count_uncertified tests/test_evidence_contract.py::test_finite_enumeration_collapses_repeated_undated_mentions
```

Result: **3 passed**.

### Unit 3: D-ARCH-2 adjacency-reserve backfill

Command:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_trajectory_store.py::test_unused_adjacency_reserve_backfills_the_full_ranked_limit tests/test_trajectory_store.py::test_partial_adjacency_reserve_backfills_in_rank_order tests/test_trajectory_store.py::test_full_adjacency_reserve_keeps_the_existing_composition
```

Result: **3 passed**.

### Unit 4: D-ARCH-3 date-anchor trust boundary

Command:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_reasoning.py::test_compute_accepts_caller_anchor_only_when_it_agrees_with_sidecar tests/test_reasoning.py::test_compute_sidecar_overrides_disagreeing_caller_anchor tests/test_reasoning.py::test_compute_without_sidecar_marks_temporal_result_low_trust
```

Result: **3 passed**.

### Unit 5: batch-2 `how long ago` operand fix

Command:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_reasoning.py::test_planner_uses_explicit_cardinality_and_interval_units
```

Result: **1 passed**. The regression proves that `how long ago` plans one evidence-date operand plus the question-date anchor.

## Affected-file validation

Spec acceptance files:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_trajectory_store.py tests/test_evidence_contract.py tests/test_reasoning.py
```

Result: **81 passed**.

Upstream-port affected test files:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_adaptive_retrieval.py tests/test_evidence_compiler.py tests/test_query_view_store.py tests/test_vector_store.py tests/test_prescreen_flip_blackout.py tests/test_int8_two_stage_knn.py
```

Result: **166 passed**.

Changed-file lint:

```text
ruff check adaptive_retrieval.py evidence_compiler.py query_view_store.py reasoning.py requirements_compiler.py tools.py trajectory_store.py vector_store.py tests/test_adaptive_retrieval.py tests/test_evidence_compiler.py tests/test_evidence_contract.py tests/test_int8_two_stage_knn.py tests/test_prescreen_flip_blackout.py tests/test_query_view_store.py tests/test_reasoning.py tests/test_trajectory_store.py tests/test_vector_store.py
```

Result: **all checks passed**.

## PR #190 CI fix

The reported CI command was reproduced locally before editing. It failed with
the expected six failures: one chunk-vector coverage-total regression, four
stale composition-policy expectations, and one stale state-semantic expansion
expectation.

- `vector_store.py`: scalar and vectorized scans now return deadline expiry
  separately from other bounded-stop causes. Message and chunk paths skip
  `COUNT(*)` only after actual deadline/budget expiry; an unscorable live vector
  remains bounded while still reporting the corpus total. The existing
  `test_exact_scan_does_not_overstate_coverage_for_unscorable_live_vector`
  contract was not modified.
- `tests/test_trajectory_composition_policies.py`: the default, Policy A,
  Policy D, and hybrid tests now cite D-ARCH-2 and assert that default backfill
  re-admits the lexical winner while each policy still promotes or protects it
  through its own mechanism.
- `tests/test_trajectory_state_semantic_expansion.py`: the full-pool test now
  cites D-ARCH-2 and pins the deterministic 14-state composition: incumbent
  rank order is preserved and the newly admitted semantic state fills the
  unused reserve.

Focused CI command:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest tests/test_chunk_vector_store.py tests/test_trajectory_composition_policies.py tests/test_trajectory_state_semantic_expansion.py -q
```

Result after the fix: **60 passed**.

The batch acceptance command above was rerun after the fix: **81 passed**.
The upstream-port affected-file command above was also rerun: **166 passed**.

CI-fix lint:

```text
ruff check vector_store.py tests/test_trajectory_composition_policies.py tests/test_trajectory_state_semantic_expansion.py
```

Result: **all checks passed**.

## Round 2 review dispositions

Review mode: **address**, Round 2 delta only. Candidate identity:
PR `#190`, branch `batch/v2-rebaseline`, local base/head before this unstaged
batch `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32` /
`cdc95c13e688cfa1b7af714c4d3a9661a65cf0a4`. No Git mutation or GitHub write
was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3679976787` | P1 | **fixed** — finite-scan candidates are availability-checked before event-date handling, real host `observed_at` takes precedence over a sidecar as the availability boundary, and grounding always receives the historical `as_of` when present. Regression: an undated event observed after the question date is excluded and cannot produce a count. |
| `3680021268` | Major | **fixed** — `_validate_existing_schema_version` returns only when `lcm_trajectory_corpora` is absent; malformed existing tables propagate `sqlite3.OperationalError`. Regression proves a missing `schema_version` column raises before any FTS rebuild. |
| `3679973891` | P2 | **fixed** — the grounding/operand pairing now uses `zip(..., strict=True)`, so cardinality drift fails instead of truncating silently. |
| `3679976780` | P2 | **fixed** — `_finite_event_key` reserves the 300-character budget for the date suffix. Regression proves an over-300-character base yields distinct keys for two dates and preserves both suffixes. |
| `3679976783` | P2 | **fixed** — temporal certification now requires the sidecar anchor to parse successfully. A malformed sidecar uses D-ARCH-3's absent/unusable-sidecar cell: `anchor_trust=low_trust`, `temporal_certified=false`; it does not retain `engine_sidecar` trust. |
| `3679973895` | P3 | **documented only** — comments at both certification sites name caller-evidence `exact_ref` uniqueness and engine finite-scan `dedupe_key` uniqueness and state that the surfaces intentionally differ. |
| `3680021253` | Minor | **fixed in this report** — machine-local `PYTHONPATH` values were replaced by the operator-local `AGENT_STUB_PATH` convention while all test targets remain repository-relative. |

Round-2 proof:

- Focused Round-2 regressions: **8 passed**.
- Exact CI slice: **60 passed**.
- Batch acceptance suite: **85 passed**.
- Upstream-affected suite: **166 passed**.
- Changed-file `ruff check`: **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-2 disposition gate;
**ADVANCE** to the orchestrator handoff. This does not claim remote exact-head
CI, merge readiness, merge, release, or runtime proof for the unstaged delta.

## Round 3 review dispositions

Review mode: **address**, binding Round-3 disposition batch. Candidate identity:
PR `#190`, branch `batch/v2-rebaseline`, local base/head before this unstaged
batch `e5acbbf26715a1f2e721abcdf7d180c5dd1d8d32` /
`0c8bea9a19e032b3b1261a8ff3d4931458228d92`. Each finding body was read through
the requested per-comment `gh api` route before implementation. No Git mutation
or GitHub write was performed.

| Comment ID | Priority | Disposition |
|---|---|---|
| `3680071381` | P1 | **fixed** — an uncertified finite count now injects an explicit `UNCERTIFIED` disclosure that says an undated contributor prevents exhaustive certification for the requested window and instructs the answerer to preserve that disclosure. Regression proves a dated+undated mix carries the warning while a fully dated set does not. |
| `3680071370` | P2 | **fixed** — adjacency-reserve backfill now consults and updates the same per-trajectory count used by nucleus and adjacency selection. Regression creates a backfill candidate that would become the third hit under `diversity_cap=2` and proves it is not admitted. |
| `3680071376` | P2 | **fixed** — `how long ago` planning extracts explicitly requested days, weeks, months, or years; execution supports complete years; and a question with no unit uses the coarsest non-zero complete unit. Answer-level regressions cover all four explicit units and the no-unit path. |
| `3680077589` | P2 | **fixed** — the effective scan stop remains bounded by `budget_s`, but `deadline_expired` now identifies only the caller's absolute deadline. Summary and chunk scan paths preserve `total` after budget-only scan or enumeration expiry, while absolute-deadline paths still avoid `COUNT(*)`. The F44 change is limited to stop-cause classification and the two mirrored consumers. |
| `3679973901` | P3 | **documented and verified** — the selector-stage write site now names the `None` = not applicable, `False` = uncertified, `True` = certified contract and forbids bool coercion. Repo-wide consumer grep found only direct JSON serialization; no dashboard or serializer coerces the value. Regression proves the non-temporal selector stage serializes `None` as JSON `null`, distinct from the existing uncertified `false` coverage. |

Round-3 focused regressions:

```text
PYTHONPATH="${AGENT_STUB_PATH}" python3 -m pytest -q tests/test_evidence_contract.py::test_finite_enumeration_distinguishes_same_entity_events_by_date tests/test_evidence_contract.py::test_finite_enumeration_returns_dated_and_undated_count_uncertified tests/test_trajectory_store.py::test_adjacency_reserve_backfill_preserves_diversity_cap tests/test_reasoning.py::test_how_long_ago_answer_honors_explicit_unit tests/test_reasoning.py::test_how_long_ago_answer_without_unit_uses_coarsest_fit tests/test_reasoning.py::test_public_compute_tool_reports_stages_and_discards_mutated_candidate tests/test_vector_store.py::test_full_scan_budget_stops_early_and_reports_bounded tests/test_vector_store.py::test_full_scan_budget_includes_candidate_enumeration
```

Result: **11 passed**.

Round-3 acceptance proof:

- Exact CI slice above: **60 passed**.
- Batch acceptance suite above: **91 passed**.
- Upstream-affected suite above: **166 passed**.
- Full changed/affected-file `ruff check`: **all checks passed**.
- `git diff --check`: **clean**.

Delivery checkpoint: **COMPLETE** for the named local Round-3 disposition gate;
**ADVANCE** to the orchestrator handoff. This does not claim remote exact-head
CI for the unstaged delta, merge readiness, merge, release, or runtime proof.

## Decision fidelity

- D-ARCH-1: dated candidates dedupe by the existing event key plus resolved date; undated candidates retain the existing collapse. Counts with any undated contributor are returned with `finite_coverage=false`.
- D-ARCH-2: adjacency remains first priority for its reserve; unused slots return to ranked candidates in rank order.
- D-ARCH-3: the engine occurrence-date sidecar is the trusted anchor; disagreement overrides the caller and is noted; absence returns a low-trust, uncertified temporal result.
- Batch-2: `how long ago` uses one evidence-date operand and the question-date anchor.
- Deviations from the binding D-ARCH text: **none**.

## Proof boundary

This report proves the unstaged working-tree implementation, focused unit behavior, affected test-file results, and changed-file lint only. It does not prove merge, remote CI, paired V2 benchmark results, V1 sanity-slice results, release, deployment, or runtime adoption.
