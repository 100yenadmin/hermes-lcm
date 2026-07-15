# Pack B / PR-4 outcome

## Result

Implemented semantic and hybrid `lcm_grep` retrieval for hermes-lcm #386 on
`feat/embedding-search-modes`, stacked on embedding-backfill head `44326e6`.
The default and explicit `full_text` paths retain the historical serialized
payload byte-for-byte; semantic and hybrid behavior is opt-in through the new
`mode` enum.

No engine, vector-store, provider, config, migration, push, or GitHub operation
was included. The supplied untracked `SPEC.md` remains untracked and is not part
of the implementation commit.

## File map

- `tools.py:1050-1260` preserves the existing FTS implementation behind an
  internal full-text entry point without changing its default output.
- `tools.py:1263-1473` adds confidence-band mapping, a hard wall-clock query
  embedding budget, provider/auth classification, FTS degradation flags,
  profile KNN lookup, coverage surfacing, bounded snippets, and provenance.
- `tools.py:1476-1604` adds the 50-to-500 candidate policy, node-id union/dedup,
  two-arm RRF with `k=60`, fused rank metadata, and public mode dispatch.
- `schemas.py:3-26` exposes `full_text`, `semantic`, and `hybrid` in the tool
  schema while keeping `full_text` as the default.
- `scripts/eval_retrieval_recall.py:1-284` builds the fixed-seed 60-summary
  corpus, implements the hash/topic mock embedder, evaluates all three modes,
  prints Markdown or stable JSON, and double-gates optional live-provider runs.
- `tests/fixtures/retrieval_recall_queries.json` commits 30 labeled queries
  across exact-term, paraphrase, and multi-hop-ish strata.
- `tests/test_embedding_search_modes.py:1-337` covers semantic ordering,
  confidence/coverage, timeout and missing-provider degradation, auth errors,
  RRF math and dedup, candidate caps, bounded snippets, full-text byte identity,
  and deterministic recall evaluation.
- `docs/retrieval-tools.md:39-109` documents modes, latency/degradation behavior,
  confidence and coverage fields, RRF, candidate limits, and offline/live eval
  commands.

RRF is the default and only fusion mechanism in this car. A Voyage
`rerank-2.5` enhancement remains a possible later change; no external reranker
or new dependency is called here.

## Recall smoke artifact

The offline command is deterministic and exits zero:

```text
python3 scripts/eval_retrieval_recall.py

| Mode | Stratum | Recall@5 | Recall@10 |
|---|---|---:|---:|
| full_text | exact-term | 1.000 | 1.000 |
| full_text | paraphrase | 0.100 | 0.100 |
| full_text | multi-hop-ish | 1.000 | 1.000 |
| semantic | exact-term | 1.000 | 1.000 |
| semantic | paraphrase | 1.000 | 1.000 |
| semantic | multi-hop-ish | 1.000 | 1.000 |
| hybrid | exact-term | 1.000 | 1.000 |
| hybrid | paraphrase | 1.000 | 1.000 |
| hybrid | multi-hop-ish | 1.000 | 1.000 |
```

These are synthetic smoke values, not production-provider benchmark claims.
The focused test runs the JSON form twice, asserts byte-for-byte determinism,
and enforces hybrid recall greater than or equal to FTS recall on paraphrases.

## Acceptance

Focused mode/provider/vector/schema bundle on the final tree:

```text
50 passed, 67 warnings in 0.82s
```

The 11 new tests also pass alone. The final full pytest run adds exactly 11
passes over PR-3's recorded 1650-pass stack result:

```text
3 failed, 1661 passed, 1 skipped, 12 xfailed, 1310 warnings in 41.22s
```

The only failures are the SPEC-approved baseline IDs:

```text
tests/test_lcm_engine.py::TestEngineABC::test_positive_preflight_clears_prior_noop_status
tests/test_path_containment.py::test_path_containment_within_allowed_base
tests/test_path_security.py::test_configured_externalization_path_inside_allowed_base_accepted
```

All three reproduced unchanged on the clean untouched upstream checkout
`/Volumes/LEXAR/repos/hermes-lcm-upstream-ro` at exact
`31675c6ed54abd747578b4da4b1589d81b18384a`:

```text
3 failed, 60 warnings in 0.30s
```

Static and artifact checks on the final tree:

```text
ruff check .
All checks passed!

git diff --check
<no output; exit 0>

python3 -m py_compile tools.py schemas.py scripts/eval_retrieval_recall.py tests/test_embedding_search_modes.py
<no output; exit 0>
```

The full validator packet is at
`/Volumes/LEXAR/Codex/hermes-lcm-embedding-search-modes-validation-20260715/validation-checklist.md`.
The first invocation stopped at its preflight because this machine has
`python3`, not `python`; rerunning with `PYTHON=python3` completed normally with
no wedge. Diff/compile/shell, focused pytest (409 passed), benchmark smoke,
stress smoke, and release stress all passed. Release stress reports
`failure_count: 0`.

The validator's normal and low-FD full-pytest gates exited nonzero only for the
same three upstream-reproduced baseline failures:

```text
normal: 3 failed, 1661 passed, 1 skipped, 12 xfailed in 47.45s
low-FD: 3 failed, 1661 passed, 1 skipped, 12 xfailed in 42.27s
```

Across the validator and final-tree reruns, there was no new feature failure,
benchmark failure, stress failure, provider network call, model download, or
live profile mutation.

## Scope and readiness

The implementation diff is limited to `tools.py`, `schemas.py`, the new eval
script and query fixture, `docs/retrieval-tools.md`, the focused test file, and
this required outcome report. This establishes local acceptance subject only to
the exact upstream-reproduced three-test baseline. It does not claim push,
remote CI, review, merge, release, or deployed-runtime proof.
