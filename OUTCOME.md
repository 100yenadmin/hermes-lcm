# Pack B / PR-3 outcome

## Result

Implemented the dry-run-first `/lcm embed backfill` command for hermes-lcm
#386 on `feat/embedding-backfill`, stacked on provider head `9a45932`. The
command discovers missing current-profile depth-0 summaries, previews bounded
work without provider calls or database writes, and applies resumable embedding
batches behind an expiring single-flight claim.

No provider, vector-store, engine, tool, schema, migration, push, or GitHub
operation was included. The supplied untracked `SPEC.md` remains untracked and
is not part of the implementation commit.

## File map

- `command.py:58-74` defines the 10-minute metadata-table claim, bounded
  command batch size, Voyage token caps used for estimates, and cost rates.
- `command.py:1742-1823` opens dry-run discovery read-only, selects newest
  depth-0 rows through a current-profile correlated `NOT EXISTS`, and computes
  provider-aware batch and cost estimates.
- `command.py:1826-1875` atomically acquires the single-flight claim with
  `BEGIN IMMEDIATE`, permits stale takeover, and releases only the matching
  owner value.
- `command.py:1878-2141` parses `--apply` / `--limit`, emits the shared report,
  calls providers outside write transactions, maps provider over-cap skips,
  commits each vector row independently, aborts on Voyage auth errors, and
  continues after transient batch errors.
- `command.py:2195-2200` routes `/lcm embed backfill` and advertises it in help.
- `docs/operator-guide.md:167-169,525-527,572-612` documents configuration,
  dry-run/apply workflow, estimates, single-flight behavior, and safe resume.
- `tests/test_embedding_backfill.py:1-353` adds nine mocked-provider tests for
  dry-run purity and local/Voyage cost math, batching/meta, idempotence,
  limit/newest ordering, per-row
  isolation, auth abort and claim cleanup, transient continuation, over-cap
  reporting, fresh/stale claims, and refusal messages.

## Acceptance

Focused feature test:

```text
python3 -m pytest -q tests/test_embedding_backfill.py
9 passed
```

Canonical host-stub provider/vector/command integration bundle:

```text
119 passed
```

This is nine more tests than the stacked provider branch's recorded 110-test
bundle. Full pytest likewise moved from the provider branch's recorded 1641
passes to 1650 passes:

```text
3 failed, 1650 passed, 1 skipped, 12 xfailed, 1308 warnings in 45.34s
```

The only failures are the SPEC-approved baseline IDs:

```text
tests/test_lcm_engine.py::TestEngineABC::test_positive_preflight_clears_prior_noop_status
tests/test_path_containment.py::test_path_containment_within_allowed_base
tests/test_path_security.py::test_configured_externalization_path_inside_allowed_base_accepted
```

All three were rerun on the clean untouched upstream checkout
`/Volumes/LEXAR/repos/hermes-lcm-upstream-ro` at exact
`31675c6ed54abd747578b4da4b1589d81b18384a` and reproduced unchanged:

```text
3 failed, 60 warnings in 0.46s
```

Static checks:

```text
ruff check .
All checks passed!

git diff --check
<no output; exit 0>
```

The last wholly completed validator packet is at
`/Volumes/LEXAR/Codex/hermes-lcm-embedding-backfill-validation-20260715-final/validation-checklist.md`.
After the final over-cap cost-estimate correction, a fresh validator run at
`/Volumes/LEXAR/Codex/hermes-lcm-embedding-backfill-validation-20260715-final2/`
repassed diff, compile, shell, focused pytest, benchmark smoke, and stress smoke,
then wedged at 8% in its first full pytest child (sleeping at 0% CPU with no log
progress for more than two minutes). The wedge playbook terminated only that
validator/pytest process. The remaining gates were rerun directly on the final
tree:

- canonical full pytest: 3 approved failures, 1650 passed, 1 skipped, 12
  xfailed, in 45.34s
- low-FD full pytest at `ulimit -n 1024`: the same 3 approved failures, 1650
  passed, 1 skipped, 12 xfailed, in 40.72s
- release stress: `failure_count: 0`, with evidence under
  `/Volumes/LEXAR/Codex/hermes-lcm-embedding-backfill-validation-20260715-final2-stress-release/`

Across the completed packet and final-state wedge recovery, the only test
failures are the exact upstream-reproduced baseline. No new feature failure,
benchmark failure, stress failure, provider network call, or live profile
mutation was observed.

## Scope and readiness

The final diff is limited to `command.py`, `docs/operator-guide.md`, the new
focused test file, and this required outcome report. This establishes local
acceptance subject to the exact upstream-reproduced three-test baseline. It
does not claim push, remote CI, review, merge, release, or deployed-runtime
proof.
