# Embeddings setup — free and local options

Semantic and hybrid retrieval are **opt-in** and need an embedding provider. You have three good
options, two of which cost nothing. If you configure nothing, nothing changes — retrieval stays
FTS-only exactly as before.

## TL;DR

| Option | Cost | Signup | Install | Best for |
|---|---|---|---|---|
| Voyage AI | free tier (200M tokens for the voyage-4 group), then ~$0.02–0.12 per million tokens | yes (no credit card) | none | best quality, zero local footprint |
| fastembed | $0 | no | `pip install fastembed` (ONNX, no torch) | local default — no accounts, no daemon |
| Ollama | $0 | no | Ollama app/daemon | you already run Ollama |

All providers feed the same store. Switching provider/model is one config change plus a backfill;
each provider+model keeps its own vectors keyed by a canonical identity, so switching **back** to a
previously-registered provider reactivates its vectors with no re-backfill (see *Switching or
removing providers*).

## Option 1 — Voyage AI (free tier)

The Voyage-4 generation (`voyage-4`, `voyage-4-large`, `voyage-4-lite`) carries **200 million free
tokens for that model group**, and signup does not require a credit card. For a personal LCM corpus
(thousands of summaries), that free allotment covers initial backfill and years of queries; past it,
embedding costs $0.02/M (`voyage-4-lite`), $0.06/M (`voyage-4`), or $0.12/M (`voyage-4-large`). A
query embeds exactly one short vector, so a semantic or hybrid query costs a fraction of a cent.
Voyage documents its allotments and rate tiers on its
[pricing](https://docs.voyageai.com/docs/pricing) and
[rate-limits](https://docs.voyageai.com/docs/rate-limits) pages — treat those as the source of
truth; the numbers above were verified 2026-07.

```bash
export VOYAGE_API_KEY=...           # from dash.voyageai.com
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=voyage
export LCM_EMBEDDING_MODEL=voyage-4-lite   # or voyage-4 / voyage-4-large
/lcm embed warmup                   # probes the API, registers model + dimensions
/lcm embed backfill                 # dry run: shows counts + estimated tokens, writes nothing
/lcm embed backfill --apply         # embeds your history in bounded batches
```

Notes: requests are batched under Voyage's caps — both the token budget and the 1000-item
per-request cap; over-length documents are skipped and reported, never silently truncated;
rate-limit responses are honored with bounded waits under one absolute per-operation deadline.

## Option 2 — fastembed (local, no signup, recommended local default)

[fastembed](https://github.com/qdrant/fastembed) runs ONNX models on CPU with no PyTorch and no
external service — just a pip package.

```bash
pip install fastembed
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=fastembed
export LCM_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # 384-dim, compact and quick on CPU
/lcm embed warmup     # downloads the model ONCE, explicitly (a few hundred MB incl. onnxruntime)
/lcm embed backfill --apply
```

The model download happens **only** during `warmup` — never lazily during a query or an agent turn.
If you skip warmup, semantic search simply stays off and the tools tell you why. Queries use the
model's query-specific encoding (distinct from document encoding) so query/passage asymmetry is
preserved.

## Option 3 — Ollama (local daemon)

If you already run [Ollama](https://ollama.com), use its embeddings endpoint:

```bash
ollama pull nomic-embed-text        # 768-dim; mxbai-embed-large and bge-m3 also work
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=ollama
export LCM_EMBEDDING_MODEL=nomic-embed-text
# LCM_OLLAMA_BASE_URL defaults to http://localhost:11434
/lcm embed warmup && /lcm embed backfill --apply
```

Ollama requests set `truncate: false`, so an input that exceeds the model's context fails loudly
rather than being silently truncated to a misleading embedding.

## What you get

`lcm_grep` gains two modes on top of the existing ones:

- `semantic` — paraphrase-tolerant vector search; local providers make this $0
- `hybrid` — keyword ∪ semantic, fused with reciprocal-rank fusion (RRF); the best
  "have we discussed X?" mode. (Fusion is RRF only — there is no external reranker.)

Both degrade **transparently**: if the provider is down, not configured, or the operation runs past
its latency budget, results come from full-text search with an explicit `degraded_to_fts` flag,
never an error. The latency budget is a single absolute deadline that bounds the **semantic attempt**
— query embedding **and** the vector search (KNN). It is a semantic-arm-only budget: once it is
exhausted the query degrades to `full_text`, and that fallback then runs to **completion** (a
synchronous local SQLite path that cannot be preempted mid-query), so the budget is a bound on the
semantic attempt rather than an end-to-end guarantee over the fallback. `full_text` mode itself is
unchanged and byte-for-byte identical to prior behavior.

Search filters (`conversation_id`, `source`, `time_from`, `time_to`) are enforced inside the vector
search before the top-k cap, so an ineligible high-scoring vector never displaces an eligible one. A
`source` filter **fails closed** when provenance cannot be verified (e.g. a legacy DB whose messages
lack a `source` column): it returns no semantic hit rather than a false positive, degrading to
`full_text`. `role` is a raw-message dimension summaries do not carry, so a `role` filter degrades a
semantic query to `full_text` (which does enforce role) rather than being ignored.

## Performance & footprint

- Vector math is dependency-free by default; installing **numpy** (optional) accelerates large
  corpora — the top-k scan is milliseconds warm once numpy is loaded.
- Metadata/id resolution uses a temp-table join rather than a giant `IN (...)` list, so it scales
  past the SQLite host-parameter limit that previously failed near ~32k ids (validated to 40k).
- Without numpy, search scans the most recent `LCM_EMBEDDING_BOUNDED_SCAN_ROWS` vectors (default
  2,000) and reports `coverage: bounded`. The candidate enumeration is bounded at the SQL layer
  (`ORDER BY` recency `+ LIMIT`), so a large corpus never materializes every id in host memory.

## Switching or removing providers

Change provider/model → run `/lcm embed warmup` (registers the new profile as the current identity)
→ `/lcm embed backfill --apply` (embeds under the new identity; the previous model's vectors are
kept separate and never mixed). Every vector is published under the exact identity that produced it
— the identity is captured at provider-resolution time and carried through the write, so switching
the active provider A→B mid-backfill can never rebind an A-vector onto B. Because each identity
`(provider, model, revision, dim, dtype, byteorder, task)` owns its own vectors, switching **back**
to a previously-registered provider reactivates it with its existing vectors — no re-backfill needed. Disable everything with
`LCM_EMBEDDINGS_ENABLED=false` — data stays, behavior reverts to FTS-only instantly.
