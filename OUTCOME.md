# Pack B / PR-2 outcome

## Result

Implemented the embedding provider abstraction for hermes-lcm #386 with
default-dormant configuration, raw-HTTP Voyage and Ollama clients, explicit
never-lazy FastEmbed model warmup, provider-local resilience guards, and the
`/lcm embed warmup` operator command. No engine, tool, externalization,
backfill, search-mode, or vector-store-internal changes were made.

## File map

- `embedding_provider.py` defines the provider protocol and operator-readable
  error types, injectable standard-library HTTP transport, provider-local
  circuit breaker and call-rate spend guard, and provider resolution.
- `embedding_provider.py` implements Voyage document/query asymmetry,
  `truncation=false`, 80,000-token request batching, 27,000-token per-document
  skip-and-report handling, bounded retries, Retry-After budgeting, classified
  errors, and response-body PII scrubbing.
- `embedding_provider.py` implements Ollama's batched `/api/embed` request with
  configurable base URL and network-error-only retries.
- `embedding_provider.py` isolates the optional FastEmbed import and uses
  `local_files_only=True` on normal calls. Only `FastembedProvider.warmup()` may
  construct with downloads enabled; uncached normal use points operators to
  `/lcm embed warmup`.
- `config.py` adds inert provider/model defaults, Ollama base URL, and the
  three-second interactive embedding timeout to `LCMConfig` and
  `ENV_FIELD_SPECS`.
- `command.py` adds `/lcm embed warmup`: FastEmbed explicitly downloads while
  Voyage/Ollama run one probe, the returned dimension is registered through
  `VectorStore.register_profile`, and success/error output stays
  operator-readable.
- `tests/test_embedding_provider.py` adds 21 all-mocked provider/config/command
  tests, including batching boundaries, over-cap skipping, Retry-After,
  retry/error taxonomy, PII scrubbing, never-lazy FastEmbed behavior,
  dimension locking, circuit cooldown, and spend limiting.

## Acceptance

- `pytest -q tests/test_embedding_provider.py`: **21 passed**.
- Canonical host-stub focused provider/vector/command bundle: **110 passed**.
- `ruff check .`: **clean**.
- Canonical host-stub full pytest: **1641 passed, 1 skipped, 12 xfailed, 3
  failed**. The failures are the same approved baseline recorded by Pack B
  PR-1 before this change:
  - `TestEngineABC::test_positive_preflight_clears_prior_noop_status`
  - `test_path_containment_within_allowed_base`
  - `test_configured_externalization_path_inside_allowed_base_accepted`
- `PYTHON=python3 scripts/validate_release.sh --full --keep-going`: compile,
  shell, diff, focused pytest (**409 passed**), benchmark smoke, stress smoke,
  and release stress passed. Full and low-FD pytest each completed with **1641
  passed, 1 skipped, 12 xfailed, 3 baseline failures**, so the validator exits
  1 under its strict no-baseline-exception policy.
- Validation evidence:
  `/Volumes/LEXAR/Codex/hermes-lcm-embedding-providers-validation-20260715/validation-checklist.md`.

## Scope and safety notes

- Every provider test injects or monkeypatches transport/model loading; the test
  suite performs no live provider call and downloads no model.
- `VOYAGE_API_KEY` is read only from the process environment and is never stored
  in config or logs.
- Provider resolution performs no network access or dimension discovery.
- The existing untracked dispatch input `SPEC.md` was preserved and was not
  added to the commit.
