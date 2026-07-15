# PR #383 merge-readiness outcome

## Identity and scope

- Branch: `feat/externalized-payload-search`
- Rebased upstream base: `31675c6ed54abd747578b4da4b1589d81b18384a`
- Rebased feature head before the fixes: `b515056`
- Remote mutations: none (`git push`, `gh`, and PR-body edits were not run)
- Evidence bundle: `/tmp/validate-383.log/validation-checklist.md`

## Changes

- `tools.py:51-88`: separated incomparable search-rank scales with an explicit externalized-result tier for `relevance` and `hybrid` sorting.
- `tools.py:218-220,1263-1382`: auto-discovery now probes session metadata before applying the 256 owned-file cap. Metadata probes are separately bounded at 4,096 files and report `discovery_files`, `discovery_limit`, and `discovery_truncated`; the full read still rechecks ownership.
- `externalize.py:648-727`: closed the final-component lstat/open race with `os.open`, `O_NOFOLLOW` where available, `os.fstat`, regular-file validation, and `(st_dev, st_ino)` identity comparison before any payload read.
- `tests/test_lcm_engine.py:25726-25819`: added message-only ordering, cross-type tier, and >256 foreign-session discovery regressions.
- `tests/test_ingest_protection.py:501-542`: added a deterministic path-replacement race regression for descriptor identity validation.

## Sort-tier design

History results (messages and summaries) remain tier 0, so their existing comparator fields and relative ordering are unchanged. Externalized payload hits are tier 1 for `relevance` and `hybrid`; within that tier, the existing `byte_position` remains the native rank, so earlier payload matches win without comparing byte offsets to FTS5 BM25 ranks. The existing hybrid summary override stays ahead of the tier field. `recent` sorting is unchanged because timestamps share a common scale.

The regression compares the serialized message-result sequence from `content_scope=history` with the message subset from `content_scope=both` for both relevance and hybrid modes. A second synthetic-rank regression proves an externalized byte position of zero cannot outrank a history result solely because of the old scalar mismatch.

## TOCTOU decision

Fixed rather than documented as accepted risk. The implementation fit the requested small patch: pre-open `os.lstat`, final-component no-follow open where supported, immediate `os.fstat`, regular-file/type validation, and device/inode identity equality. Once validated, all reads use the opened descriptor, so later pathname replacement cannot redirect the read. On platforms without `O_NOFOLLOW`, the descriptor identity check remains the fallback.

## Validation

### Focused regressions

Command used the repository's standalone import bootstrap because this machine has `python3` but no `python` shim:

```text
python3 - tests/test_lcm_engine.py::TestHandleGrepExternalizedPayloads tests/test_ingest_protection.py::test_externalized_search_prefix_rejects_path_replaced_during_open -q
```

Output:

```text
.............                                                            [100%]
13 passed, 61 warnings in 0.37s
```

### Full pytest

Output on this branch:

```text
3 failed, 1620 passed, 1 skipped, 12 xfailed, 1304 warnings in 62.64s
```

The three failures are the SPEC-approved macOS baseline:

```text
tests/test_lcm_engine.py::TestEngineABC::test_positive_preflight_clears_prior_noop_status
tests/test_path_containment.py::test_path_containment_within_allowed_base
tests/test_path_security.py::test_configured_externalization_path_inside_allowed_base_accepted
```

They were rerun on the untouched upstream reference checkout `/Volumes/LEXAR/repos/hermes-lcm-upstream-ro` at exact head `31675c6ed54abd747578b4da4b1589d81b18384a` and reproduced unchanged:

```text
3 failed, 60 warnings in 3.70s
```

### Ruff and whitespace

```text
$ ruff check .
All checks passed!

$ git diff --check
<no output; exit 0>
```

### Full release validator

Command:

```text
PYTHON=python3 scripts/validate_release.sh --full --keep-going --output /tmp/validate-383.log
```

Passed gates:

```text
git diff checks
python compileall
script py_compile
shell syntax
focused pytest: 410 passed
benchmark smoke
stress smoke
stress release
```

The validator reported only two nonzero gates: `pytest full` and `pytest low fd`. Each had the same upstream-confirmed three macOS baseline failures and otherwise reported:

```text
3 failed, 1620 passed, 1 skipped, 12 xfailed
```

No benchmark or stress failures were reported. Full gate-by-gate output is in `/tmp/validate-383.log/validation-checklist.md`.

### Advisory review

The standard `codex-review --mode local` helper recursed into nested review helpers and was terminated without producing findings. A direct, tool-free Codex review of the exact four-file patch then completed with:

```text
No actionable correctness issues are evident in the provided patch.
```

No review-triggered code changes were required.

## Proof boundary

This establishes local merge readiness against upstream `31675c6`, subject to the three verified macOS baseline failures above. It does not claim pushed-head, GitHub CI, review-thread, merge, release, or deployed-runtime proof; those were outside the explicit no-push/no-`gh` scope.
