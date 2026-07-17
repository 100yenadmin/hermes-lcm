"""Offline LongMemEval retrieval-quality harness for hermes-lcm.

This module ingests each LongMemEval question's conversation history into a
fresh temporary LCM store (reusing ``store``/``dag``/``vector_store`` APIs
directly, with no live Hermes host), builds one deterministic summary per
session, optionally backfills summary embeddings, then scores each retrieval
arm against the dataset's labeled evidence sessions.

It is retrieval-only: LongMemEval labels the evidence session(s) per question,
so recall@k / NDCG@k are computable offline without an LLM judge.

Dataset: LongMemEval_S (Wu et al., ICLR 2025), canonical Hugging Face dataset
``xiaowu0162/longmemeval``, file ``longmemeval_s``, pinned to a fixed revision
(see :data:`DATASET_REPO_ID` / :data:`DATASET_REVISION`). The dataset is
downloaded once by an explicit operator command and never during a run.

Export hygiene mirrors ``scripts/lcm_benchmark.py``: output is aggregate-only.
It contains no transcript content, session ids, or local paths.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .standalone import ensure_agent_context_engine_importable

_REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_VERSION = 1
SCHEMA_VERSION = 1
RRF_K = 60

# Canonical LongMemEval dataset coordinates. The revision is PINNED so a run is
# reproducible: `longmemeval_s` has been byte-stable since it was introduced;
# this is the current `main` commit at implementation time (2026-07-17).
DATASET_REPO_ID = "xiaowu0162/longmemeval"
DATASET_REVISION = "2ec2a557f339b6c0369619b1ed5793734cc87533"
DATASET_FILENAME = "longmemeval_s"

PROVIDERS = ("stub", "fastembed", "voyage", "ollama")
ARMS = ("fts", "summary_vectors", "hybrid_rrf", "hybrid_rerank")

# LongMemEval `question_type` -> reported category label. Abstention questions
# (``question_id`` ends with ``_abs``) are excluded from recall scoring and
# reported separately as an ``abstention`` count.
CATEGORY_LABELS = {
    "single-session-user": "single-session-user",
    "single-session-assistant": "single-session-assistant",
    "single-session-preference": "single-session-preference",
    "multi-session": "multi-session",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge-update",
}

_WHITESPACE_RE = re.compile(r"\s+")
_STUB_MODEL = "stub-hash-64"
_STUB_DIM = 64


def _ensure_hermes_lcm_package() -> None:
    """Make this source checkout importable as ``hermes_lcm`` (no plugin registration)."""
    ensure_agent_context_engine_importable()
    if "hermes_lcm" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm",
        _REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(_REPO_ROOT)],
    )
    if spec is None:
        raise RuntimeError("could not create hermes_lcm package spec")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(_REPO_ROOT)]
    module.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = module


# --------------------------------------------------------------------------- #
# Deterministic stub embedder (pure plumbing; scores are meaningless with it).
# --------------------------------------------------------------------------- #


class StubEmbedder:
    """Hash-based unit vectors for offline plumbing tests. No provider calls."""

    provider_id = "stub"

    def __init__(self, dim: int = _STUB_DIM):
        self.model_id = _STUB_MODEL
        self.dim = int(dim)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _WHITESPACE_RE.sub(" ", str(text).lower()).split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            vector[-1] = 1.0
            return vector
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _fastembed_cache_dir() -> str | None:
    """Cache dir for FastEmbed models, honoring an env override.

    The provider default is ``~/.cache/fastembed``; ``LCM_LONGMEMEVAL_FASTEMBED_CACHE``
    (or ``FASTEMBED_CACHE_PATH``) redirects it, e.g. to a roomy volume.
    """
    import os

    override = os.environ.get("LCM_LONGMEMEVAL_FASTEMBED_CACHE") or os.environ.get(
        "FASTEMBED_CACHE_PATH"
    )
    return override or None


def resolve_harness_provider(provider: str, model: str, *, timeout: float = 120.0):
    """Return a WARMED embedder for ``provider``. ``stub`` stays fully offline.

    Non-stub providers are warmed up once here so ``.dim`` is populated (FastEmbed
    reports dim only after the first embed) and any model download happens before
    the scoring loop rather than inside a per-question deadline.
    """
    if provider == "stub":
        return StubEmbedder()
    _ensure_hermes_lcm_package()
    if not model:
        raise ValueError(f"--model is required for --provider {provider}")
    if provider in {"fastembed", "fast-embed"}:
        from hermes_lcm.embedding_provider import EmbeddingSpendGuard, FastembedProvider

        # max_calls=0 disables the per-minute call-rate guard, matching the
        # bulk-backfill contract (resolve_provider(for_backfill=True)); the
        # harness embeds thousands of summaries in one pass.
        embedder = FastembedProvider(
            model,
            cache_dir=_fastembed_cache_dir(),
            timeout=timeout,
            spend_guard=EmbeddingSpendGuard(max_calls=0),
        )
        embedder.warmup()
        return embedder
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.embedding_provider import resolve_provider

    config = LCMConfig(
        embedding_provider=provider,
        embedding_model=model,
        embedding_backfill_timeout_s=timeout,
    )
    resolved = resolve_provider(config, for_backfill=True)
    if resolved is None:
        raise ValueError(f"could not resolve embedding provider {provider!r}")
    if int(getattr(resolved, "dim", 0)) == 0:
        resolved.embed_query("warmup")
    return resolved


# --------------------------------------------------------------------------- #
# Dataset loading + question shape.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Question:
    """A single LongMemEval question with its haystack and labeled evidence."""

    question_id: str
    question_type: str
    question: str
    haystack_session_ids: list[str]
    haystack_sessions: list[list[dict[str, Any]]]
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def category(self) -> str:
        return CATEGORY_LABELS.get(self.question_type, self.question_type)


def parse_question(raw: dict[str, Any]) -> Question:
    return Question(
        question_id=str(raw["question_id"]),
        question_type=str(raw["question_type"]),
        question=str(raw["question"]),
        haystack_session_ids=[str(s) for s in raw.get("haystack_session_ids", [])],
        haystack_sessions=list(raw.get("haystack_sessions", [])),
        answer_session_ids=[str(s) for s in raw.get("answer_session_ids", [])],
    )


def load_questions(path: str | Path, *, limit: int | None = None) -> list[Question]:
    """Load LongMemEval questions from the downloaded JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LongMemEval dataset must be a JSON array of questions")
    questions = [parse_question(row) for row in data]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        questions = questions[:limit]
    return questions


def evidence_sessions(question: Question) -> set[str]:
    """Session-level evidence set (empty for abstention questions)."""
    if question.is_abstention:
        return set()
    return set(question.answer_session_ids)


def evidence_turns(question: Question) -> set[tuple[str, int]]:
    """Turn-level evidence: ``(session_id, turn_index)`` where ``has_answer``."""
    turns: set[tuple[str, int]] = set()
    if question.is_abstention:
        return turns
    for session_id, session in zip(question.haystack_session_ids, question.haystack_sessions):
        for index, turn in enumerate(session):
            if isinstance(turn, dict) and turn.get("has_answer"):
                turns.add((str(session_id), index))
    return turns


# --------------------------------------------------------------------------- #
# Deterministic summarization stub (no LLM, content-derived, offline).
# --------------------------------------------------------------------------- #


def deterministic_session_summary(turns: Sequence[dict[str, Any]], *, max_chars: int = 1200) -> str:
    """Condense a session's turns into a deterministic, content-bearing summary.

    Lexical content is preserved (collapsed whitespace, truncated) so the FTS
    and embedding arms both see meaningful text. No provider is consulted.
    """
    parts: list[str] = []
    for turn in turns:
        role = str(turn.get("role", "unknown")) if isinstance(turn, dict) else "unknown"
        content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
        parts.append(f"{role}: {content}")
    condensed = _WHITESPACE_RE.sub(" ", " ".join(parts)).strip()
    return condensed[:max_chars]


# --------------------------------------------------------------------------- #
# Metric math (pure, testable).
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Set recall@k: fraction of relevant items present in the top-k retrieved."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(dict.fromkeys(retrieved))[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance NDCG@k over a deduplicated ranked list."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(dict.fromkeys(retrieved))[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def percentiles(values: Sequence[float], points: Sequence[int] = (50, 90, 99)) -> dict[str, float]:
    """Nearest-rank percentiles for latency reporting."""
    ordered = sorted(values)
    result: dict[str, float] = {}
    for point in points:
        if not ordered:
            result[f"p{point}"] = 0.0
            continue
        rank = max(1, math.ceil(point / 100 * len(ordered)))
        result[f"p{point}"] = round(ordered[min(rank, len(ordered)) - 1], 3)
    return result


# --------------------------------------------------------------------------- #
# Retrieval arms.
# --------------------------------------------------------------------------- #


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _unit(values: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        return list(values)
    return [value / magnitude for value in values]


def _dedup_sessions(session_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(session_ids))


# A LongMemEval question is a natural-language sentence; SQLite FTS5 MATCH ANDs
# its tokens, so a single non-matching word (e.g. "what") zeroes the arm. A
# standard lexical retrieval arm ORs the salient query terms (BM25-style), so we
# build the MATCH query from the repo's own term extractor minus light stopwords.
_FTS_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
        "my", "of", "on", "or", "that", "the", "to", "was", "were", "what",
        "when", "where", "which", "who", "why", "with", "you", "your",
    }
)


def build_fts_query(question: str) -> str:
    """Build an OR-of-terms FTS5 MATCH query from a natural-language question.

    Each term is reduced to a bareword (alphanumerics/underscore only) so no FTS5
    operator character survives to trip a syntax error and force the LIKE
    fallback; empty and stopword tokens are dropped.
    """
    _ensure_hermes_lcm_package()
    from hermes_lcm.search_query import extract_search_terms

    barewords: list[str] = []
    for term in extract_search_terms(question):
        cleaned = re.sub(r"\W+", "", term, flags=re.UNICODE)
        if cleaned and cleaned.lower() not in _FTS_STOPWORDS:
            barewords.append(cleaned)
    return " OR ".join(dict.fromkeys(barewords))


def fts_sessions(store, query: str, fetch: int) -> list[str]:
    """Rank evidence sessions by the raw-message FTS arm (``store.search``)."""
    match_query = build_fts_query(query)
    if not match_query:
        return []
    rows = store.search(match_query, session_id=None, limit=fetch)
    return _dedup_sessions(str(row.get("session_id", "")) for row in rows)


def vector_sessions(vector_store, dag, query_vec, model, provider, fetch: int) -> list[str]:
    """Rank evidence sessions by the summary-vector arm (``vector_store.knn``)."""
    result = vector_store.knn(query_vec, k=fetch, model=model, provider=provider)
    sessions: list[str] = []
    for embedded_id, _score, _kind in result:
        node = dag.get_node(int(embedded_id))
        if node is not None:
            sessions.append(str(node.session_id))
    return _dedup_sessions(sessions)


def rrf_fuse(*ranked_lists: Sequence[str]) -> list[str]:
    """Reciprocal-rank fusion over per-arm session rankings (``RRF_K`` = 60)."""
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in ranked_lists:
        for rank, session_id in enumerate(ranked, start=1):
            scores[session_id] = scores.get(session_id, 0.0) + 1.0 / (RRF_K + rank)
            best_rank[session_id] = min(best_rank.get(session_id, rank), rank)
    return sorted(scores, key=lambda sid: (-scores[sid], best_rank[sid], sid))


def rerank_by_cosine(
    sessions: Sequence[str], query_vec, session_vectors: dict[str, list[float]]
) -> list[str]:
    """Rerank fused candidates by cosine to the query embedding.

    This is a deterministic embedding-cosine reranker over the fused candidate
    pool, a placeholder for a real cross-encoder. Per the MemDelta caveat we
    only ever compare it against *this* configuration, never as a universal
    verdict.
    """
    normalized_query = _unit(list(query_vec))

    def score(session_id: str) -> float:
        vector = session_vectors.get(session_id)
        if vector is None:
            return -math.inf
        return _dot(normalized_query, vector)

    return sorted(sessions, key=lambda sid: (-score(sid), sid))


# --------------------------------------------------------------------------- #
# Per-question ingest + evaluation.
# --------------------------------------------------------------------------- #


@dataclass
class ArmSamples:
    recalls: dict[int, list[float]] = field(default_factory=lambda: {1: [], 5: [], 10: []})
    ndcg10: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)


def _new_arm_samples() -> dict[str, ArmSamples]:
    return {arm: ArmSamples() for arm in ARMS}


def evaluate_question(
    question: Question,
    provider_embedder,
    *,
    provider_name: str,
    tmp_dir: Path,
    embeddings_enabled: bool,
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    """Ingest one question into a fresh store and score every retrieval arm.

    Returns ``{arm: {"recall@1", "recall@5", "recall@10", "ndcg@10", "latency_ms"}}``.
    """
    _ensure_hermes_lcm_package()
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.dag import SummaryDAG, SummaryNode
    from hermes_lcm.store import MessageStore
    from hermes_lcm.vector_store import VectorStore

    db_path = tmp_dir / f"{_safe(question.question_id)}.db"
    model = provider_embedder.model_id
    dim = int(provider_embedder.dim)
    config = LCMConfig(
        database_path=str(db_path),
        embeddings_enabled=embeddings_enabled,
        embedding_provider=provider_name,
        embedding_model=model,
    )
    store = MessageStore(str(db_path), ingest_protection_config=config)
    dag = SummaryDAG(str(db_path))
    vector_store = VectorStore(str(db_path), config=config)
    session_vectors: dict[str, list[float]] = {}
    try:
        if embeddings_enabled:
            vector_store.register_profile(model, provider_name, dim)
            identity = vector_store.capture_identity(model, provider=provider_name)

        for session_id, session in zip(question.haystack_session_ids, question.haystack_sessions):
            messages = [
                {
                    "role": str(turn.get("role", "user")) if isinstance(turn, dict) else "user",
                    "content": turn.get("content", "") if isinstance(turn, dict) else str(turn),
                }
                for turn in session
            ]
            if messages:
                store.append_batch(
                    session_id, messages, source="benchmark", conversation_id=session_id
                )
            summary_text = deterministic_session_summary(session)
            node_id = dag.add_node(
                SummaryNode(
                    session_id=session_id,
                    depth=0,
                    summary=summary_text,
                    token_count=len(summary_text.split()),
                    source_token_count=sum(len(m["content"].split()) for m in messages),
                    source_type="messages",
                    created_at=float(len(session_vectors) + 1),
                )
            )
            if embeddings_enabled:
                vector = provider_embedder.embed_documents([summary_text])[0]
                vector_store.record_embedding(
                    str(node_id), "summary", model, vector, identity=identity
                )
                session_vectors[session_id] = _unit(list(vector))

        relevant = evidence_sessions(question)
        query_vec = provider_embedder.embed_query(question.question) if embeddings_enabled else None
        fetch = max(top_k * 5, 50)

        fts_ranked = _timed(lambda: fts_sessions(store, question.question, fetch))
        if embeddings_enabled:
            vector_ranked = _timed(
                lambda: vector_sessions(vector_store, dag, query_vec, model, provider_name, fetch)
            )
        else:
            vector_ranked = ([], 0.0)

        hybrid_ranked = _timed(lambda: rrf_fuse(fts_ranked[0], vector_ranked[0]))
        rerank_ranked = _timed(
            lambda: rerank_by_cosine(hybrid_ranked[0], query_vec, session_vectors)
            if embeddings_enabled
            else list(hybrid_ranked[0])
        )

        ranked_by_arm = {
            "fts": fts_ranked,
            "summary_vectors": vector_ranked,
            "hybrid_rrf": hybrid_ranked,
            "hybrid_rerank": rerank_ranked,
        }
        scored: dict[str, dict[str, float]] = {}
        for arm, (ranked, elapsed_ms) in ranked_by_arm.items():
            scored[arm] = {
                "recall@1": recall_at_k(ranked, relevant, 1),
                "recall@5": recall_at_k(ranked, relevant, 5),
                "recall@10": recall_at_k(ranked, relevant, 10),
                "ndcg@10": ndcg_at_k(ranked, relevant, 10),
                "latency_ms": elapsed_ms,
            }
        return scored
    finally:
        vector_store.close()
        dag.close()
        store.close()


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._") or "question"


# --------------------------------------------------------------------------- #
# Aggregation + report.
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def run_harness(
    questions: Sequence[Question],
    *,
    provider_name: str,
    model: str,
    tmp_dir: Path,
    embeddings_enabled: bool | None = None,
) -> dict[str, Any]:
    """Run every arm over every question and return an aggregate-only report."""
    if embeddings_enabled is None:
        embeddings_enabled = provider_name != "none"
    embedder = resolve_harness_provider(provider_name, model)

    by_category: dict[str, dict[str, ArmSamples]] = {}
    overall = _new_arm_samples()
    scored_count = 0
    abstention_count = 0

    for question in questions:
        if question.is_abstention:
            abstention_count += 1
            continue
        scored = evaluate_question(
            question,
            embedder,
            provider_name=provider_name,
            tmp_dir=tmp_dir,
            embeddings_enabled=embeddings_enabled,
        )
        scored_count += 1
        category = question.category
        bucket = by_category.setdefault(category, _new_arm_samples())
        for arm, metrics in scored.items():
            for k in (1, 5, 10):
                bucket[arm].recalls[k].append(metrics[f"recall@{k}"])
                overall[arm].recalls[k].append(metrics[f"recall@{k}"])
            bucket[arm].ndcg10.append(metrics["ndcg@10"])
            overall[arm].ndcg10.append(metrics["ndcg@10"])
            bucket[arm].latency_ms.append(metrics["latency_ms"])
            overall[arm].latency_ms.append(metrics["latency_ms"])

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transcript_contents_included": False,
        "dataset": {
            "name": "LongMemEval_S",
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "file": DATASET_FILENAME,
        },
        "provider": provider_name,
        "model": model,
        "embeddings_enabled": embeddings_enabled,
        "question_count": len(questions),
        "scored_count": scored_count,
        "abstention_excluded": abstention_count,
        "arms": {
            arm: _arm_report(overall[arm]) for arm in ARMS
        },
        "per_category": {
            category: {arm: _arm_report(samples[arm]) for arm in ARMS}
            for category, samples in sorted(by_category.items())
        },
    }


def _arm_report(samples: ArmSamples) -> dict[str, Any]:
    return {
        "recall@1": _mean(samples.recalls[1]),
        "recall@5": _mean(samples.recalls[5]),
        "recall@10": _mean(samples.recalls[10]),
        "ndcg@10": _mean(samples.ndcg10),
        "n": len(samples.ndcg10),
        "latency_ms": percentiles(samples.latency_ms),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Aggregate-only markdown table of overall per-arm recall/NDCG."""
    lines = [
        f"# LongMemEval_S retrieval — provider={report['provider']} "
        f"model={report['model'] or 'n/a'}",
        "",
        f"scored={report['scored_count']} abstention_excluded={report['abstention_excluded']} "
        f"dataset={report['dataset']['repo_id']}@{report['dataset']['revision'][:7]}",
        "",
        "| Arm | R@1 | R@5 | R@10 | NDCG@10 | p50 ms | p90 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = report["arms"][arm]
        lines.append(
            f"| {arm} | {row['recall@1']:.3f} | {row['recall@5']:.3f} | "
            f"{row['recall@10']:.3f} | {row['ndcg@10']:.3f} | "
            f"{row['latency_ms']['p50']:.1f} | {row['latency_ms']['p90']:.1f} |"
        )
    return "\n".join(lines)
