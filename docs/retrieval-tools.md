# Retrieval tools reference

Use this page when you need the exact LCM tool contract or archive-migration notes. For install, activation, and runtime configuration, start with [Operator guide](operator-guide.md).

## Agent Tools

Use these tools for current-session recall after compaction. Use `session_search`
for earlier separate sessions or broad cross-session history.

| Tool | Use |
|------|-----|
| `lcm_grep` | Search current-session raw messages and summaries. `mode='full_text'` is the byte-compatible default; `mode='semantic'` searches embedded summaries; `mode='hybrid'` combines full-text and semantic ranks with RRF. Opt into `session_scope='all'` or `session_scope='session'` (with `session_id`) for bounded archive recovery over rows already present in `lcm.db`, including externally backfilled rows that may carry source strings such as `openclaw-lcm:*`; broader scopes return raw-message hits only in full-text mode. Raw-message filters `role`, `time_from`, and `time_to` are pushed into the full-text query; when any of them is supplied, full-text summary hits are omitted so the filter contract stays exact. Use `session_search` for earlier separate sessions or broad cross-session recall. |
| `lcm_load_session` | Load one ordered raw-message transcript page for an explicit `session_id`. This is not search: it returns raw rows in `store_id` order, bounded by `limit`, with per-message content bounded by `max_content_chars`, and continues with `after_store_id` from `next_cursor`. |
| `lcm_describe` | Inspect the current-session DAG or preview an `externalized_ref` without loading full content. |
| `lcm_expand` | Recover source messages, child summaries, or externalized payloads with pagination. Use `store_id` to fetch a single raw message regardless of session, suitable for drilling into a cross-session `lcm_grep` result. |
| `lcm_expand_query` | Answer a question using expanded current-session LCM context while returning a bounded answer. |
| `lcm_status` | Show runtime health, context pressure, config, source lineage, and lifecycle stats. |
| `lcm_inspect` | Read-only operator inventory for current-session lineage, message/frontier metadata, fresh tail, externalized refs/readability, compaction skip/no-op reasons, and matched ignore/stateless patterns. It returns metadata only; use `lcm_load_session`/`lcm_expand` when you need content. |
| `lcm_doctor` | Run database, FTS, lifecycle, config, and context-pressure diagnostics. |

### Retrieval contract

LCM retrieval tools default to current-session scope. `lcm_grep` accepts
`session_scope='all'` or `session_scope='session'` as an explicit opt-in for
bounded archive search over rows already present in `lcm.db` (raw-message hits
only). Once a session id is known, `lcm_load_session` can enumerate that session's
raw transcript in chronological `store_id` pages without a search query. Use
Hermes `session_search` for broad cross-session history outside the LCM database.

Within the current session, `source` filters raw rows directly and filters
summary nodes by descendant raw-message source lineage. `unknown` is a real
source value, not a wildcard. Legacy blank-source rows are treated as `unknown`.
`role`, `time_from`, and `time_to` are raw-message filters and are applied in the
message search query before result limiting. `time_from` and `time_to` accept Unix
seconds or timezone-aware ISO 8601 strings; naive ISO strings are rejected so the
same query means the same thing across machines. When a raw-message filter is
active, `lcm_grep` returns raw rows only and reports `summary_results_omitted`.

### Full-text, semantic, and hybrid modes

`lcm_grep` defaults to `mode='full_text'`. Omitting `mode` or setting it to
`full_text` uses the historical FTS path without changing its serialized result.
The existing `sort` argument still controls ordering inside that full-text arm;
it is separate from the retrieval `mode`.

`mode='semantic'` embeds the query with the configured provider's query path
(preserving query/document input asymmetry), then searches the current embedding
profile with cosine KNN. Semantic hits are summary nodes and retain the normal
`node_id`, depth, session, time-window, `expand_hint`, and bounded 300-character
snippet provenance. Each hit adds `score`/`cosine_score` and a
`confidence`/`confidence_band` value:

| Cosine score | Confidence |
|---:|---|
| `>= 0.65` | `high` |
| `>= 0.50` | `medium` |
| `>= 0.35` | `low` |
| `< 0.35` | `noise` |

The response-level `coverage` value comes directly from the vector store:
`full` means the complete current profile was scanned, `bounded` means the
dependency-free bounded-scan fallback was used, and `none` means no usable
profile/vector coverage was available.

Semantic and hybrid requests run under one absolute wall-clock deadline from
`embedding_query_timeout_s` (3 seconds by default), started at `lcm_grep` entry.
The same deadline covers provider resolution, query embedding, optional NumPy
import, bounded KNN, result hydration, FTS fallback, both hybrid arms, and
fusion. A disabled/missing provider or transient semantic failure degrades to
the existing full-text result only when enough time remains, adding
`degraded_to_fts: true`, `degraded_reason`, and `coverage: 'none'`. Fallback
uses independent read-only SQLite connections with progress interruption. If
the absolute deadline is exhausted, the tool returns an explicit `timeout`
error and does not start another fallback or hybrid arm. Provider
authentication failures also remain operator-readable rather than being
silently hidden.

`source` first uses the SQL-bounded candidate window, then verifies descendant
source lineage within that window before ranking. This avoids corpus-sized
lineage enumeration but means source-filtered semantic coverage is explicitly
`bounded`, not a claim that source was applied before the global corpus bound.
The recursive lineage walk has its own hard work cap. If that cap is exceeded,
or a legacy DB lacks the `messages.source` column, provenance is
`unverifiable_provenance` and the filter **fails closed** rather than treating
"can't check" as "all allowed". The remaining raw-message filters are
governed by the advertised contract: `role`, `time_from`, `time_to`,
`conversation_id`, and broader `session_scope` values (`all`/`session`) all
return raw-message hits only. Because a summary node has no single role/lane and
is cross-session, the semantic arm degrades to the raw full-text path (which
enforces those filters at the message-row level and reports `degraded_to_fts`)
whenever any of them is supplied, rather than emitting summary hits that would
violate the contract. The semantic arm therefore produces summary hits only for
a current-session query with no role/time/conversation filter (`source` still
allowed).

`mode='hybrid'` runs both arms and deduplicates shared summary nodes by
`node_id`. It fuses ranks with reciprocal-rank fusion only:

```text
rrf_score = sum(1 / (60 + rank))
```

Hybrid hits surface `fts_rank` and/or `semantic_rank`, plus `rrf_score`.
Semantic hits also retain `semantic_score` and `confidence`. Both arms inspect
`min(500, max(50, limit * 3))` candidates before the public result limit is
applied. If the semantic arm fails while deadline remains, hybrid returns its
already-computed FTS results with the same degradation fields. If the FTS arm
consumes the deadline, the semantic arm is not started; if fusion exhausts it,
an explicit timeout is returned. No external reranker is called.

### Deterministic recall smoke evaluation

The committed offline harness builds a fixed-seed, 60-summary synthetic corpus
and evaluates 30 labeled queries across exact-term, paraphrase, and
multi-hop-ish strata. Its mock embedder produces hash-based unit vectors with
topic clustering, so it requires no network, model download, or credential:

```bash
python3 scripts/eval_retrieval_recall.py
```

The command prints recall@5 and recall@10 per mode and stratum as a Markdown
table. `--json` emits stable machine-readable output. These values are a
deterministic smoke proof, not a claim about production-provider quality.

Real-provider numbers will differ. A live rerun is deliberately double-gated
because it can make network calls and incur provider cost: configure
`LCM_EMBEDDING_PROVIDER` and `LCM_EMBEDDING_MODEL` (plus the provider credential
or local endpoint), then run:

```bash
LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER=1 \
  python3 scripts/eval_retrieval_recall.py --live-provider
```

Carried-over summary nodes can become current-session content after `/new`, but
their source eligibility still comes from the descendant raw messages. Expanding
a carried-over current-session node recovers those original raw message sources
even when the sources still belong to the previous session.

### Lossless raw recovery contract

Tool responses are bounded so one retrieval call cannot flood the main context.
Lossless recovery means raw content is stored with stable source lineage and can
be recovered in deterministic pages.

- `lcm_expand(node_id=...)` pages immediate sources with `source_offset` and `source_limit`
- `lcm_load_session(session_id=...)` pages ordered raw session rows with `after_store_id` and `next_cursor`; each row includes bounded content plus truncation metadata, and large individual rows can be recovered with `lcm_expand(store_id=...)` using `content_offset`
- oversized raw messages continue with `content_offset`
- `lcm_expand(externalized_ref=...)` pages payload content with `content_offset`
- `lcm_expand_query` uses `context_max_tokens` for auxiliary context and reports truncation/pagination hints when needed

### lossless-claw/OpenClaw import utility

`hermes-lcm` includes an opt-in operator script for backfilling raw message rows from a lossless-claw/OpenClaw LCM SQLite database into the local hermes-lcm SQLite store:

```bash
python scripts/import_lossless_claw.py \
  --source-db ~/.openclaw/path/to/lcm.db \
  --target-db ~/.hermes/lcm.db \
  --agent sammy
```

The script is intentionally conservative:

- dry-run is the default; pass `--apply` to write
- run it against an explicit target DB path, preferably while Hermes is stopped for that profile
- writes create a timestamped target DB backup first when the target already exists
- only raw messages are imported; summary DAG import is out of scope
- imported rows keep explicit provenance in `session_id` and `source`, for example `openclaw-lcm:agent:sammy:<source-session>`
- the default provenance identity is the concrete source `conversations.session_id`, preserving source session boundaries even when many conversations share one `session_key`
- pass `--session-identity session_key` only when you intentionally want conversations with the same source session key grouped into one imported LCM session
- reruns are idempotent for the same `--import-id`; the default `import_id` is path-derived, so pass a stable `--import-id` if you may import the same copied DB from different paths
- changing `--agent`, `--namespace`, or `--session-identity` under the same `--import-id` is treated as the same import and will skip already-tracked source messages; use a new `--import-id` for a different mapping
- no OpenClaw config or separate secret tables are imported, but raw transcripts and tool payloads are imported and may contain sensitive user data

This is a local archive migration path. It does not make LCM a general memory provider, and it does not change the current-session retrieval contract for agent tools.

## Related references

- [Embeddings setup — free and local options](./embeddings-setup.md) — provider configuration for the `semantic` / `hybrid` modes

- [Operator guide](operator-guide.md)
- [Architecture notes](architecture.md)
- [Benchmarking and stress checks](../benchmarks/README.md)
