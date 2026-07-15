# Embeddings setup — free and local options

Semantic and hybrid retrieval are **opt-in** and need an embedding provider. You have three good
options, two of which cost nothing. If you configure nothing, nothing changes — retrieval stays
FTS-only exactly as before.

## TL;DR

| Option | Cost | Signup | Install | Best for |
|---|---|---|---|---|
| Voyage AI | free tier (200M tokens per model group), then ~$0.0001–0.001/query | yes (no credit card) | none | best quality, zero local footprint |
| fastembed | $0 | no | `pip install fastembed` (ONNX, no torch) | local default — no accounts, no daemon |
| Ollama | $0 | no | Ollama app/daemon | you already run Ollama |

All providers feed the same store; switching later means one config change plus a re-backfill
(`lcm_status`/doctor will tell you when the corpus model doesn't match the configured model).

## Option 1 — Voyage AI (free tier)

The Voyage-4 generation (`voyage-4`, `voyage-4-large`, `voyage-4-lite`) and the `rerank-2.5`
reranker each carry **200 million free tokens per model group**, and signup does not require a
credit card. For a personal LCM corpus (thousands of summaries), that free allotment covers initial
backfill and years of queries; past it, a hybrid query costs roughly $0.001 and a semantic-only
query roughly $0.0001. Voyage documents its allotments and rate tiers on its
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

Notes: requests are batched under Voyage's caps; over-length documents are skipped and reported,
never silently truncated; rate-limit responses are honored with bounded waits.

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
If you skip warmup, semantic search simply stays off and the tools tell you why.

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

## What you get

`lcm_grep` gains two modes on top of the existing ones:

- `semantic` — paraphrase-tolerant vector search; local providers make this $0
- `hybrid` — keyword ∪ semantic, fused (and, on Voyage, optionally reranked); the best
  "have we discussed X?" mode

Both degrade **transparently**: if the provider is down, slow past its latency budget, or not
configured, results come from full-text search with an explicit `degraded` flag — never an error.

## Performance & footprint

- Vector math is dependency-free by default; installing **numpy** (optional) accelerates large
  corpora (measured: top-50 over 50k×1024 vectors ≤69 ms cold, ~2 ms warm).
- Without numpy, search scans the most recent 2,000 vectors (~20–40 ms) and reports
  `coverage: bounded`.
- `sqlite-vec` can be enabled as an accelerator where it works, but note it cannot load on macOS
  stock/system Python (Apple compiles SQLite without extension loading) — which is why it is not the
  default path.

## Switching or removing providers

Change provider/model → run `/lcm embed warmup` (registers the new profile) → `/lcm embed backfill
--apply` (re-embeds; the old model's vectors are kept separate and never mixed). Disable everything
with `LCM_EMBEDDINGS_ENABLED=false` — data stays, behavior reverts to FTS-only instantly.
