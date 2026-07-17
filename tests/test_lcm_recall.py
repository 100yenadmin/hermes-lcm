"""Tests for lcm_recall — the cross-conversation forever-memory recall tool.

Seeds summaries + chunks + raw messages across three synthetic sessions and
asserts the fused pipeline recalls cross-session WITHOUT a session filter, that
scope_bias and recency are soft ranking boosts (never filters), that chunk hits
dedupe against FTS by store_id, that rerank failures skip silently, and that the
degrade matrix (embeddings-off) still returns the FTS arm.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

CURRENT = "session-cur"


class MockProvider:
    provider_id = "mock"
    model_id = "mock-model"
    dim = 2

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = list(vector)
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self.vector)


@pytest.fixture
def recall_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "recall.db"),
        embeddings_enabled=True,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_query_timeout_s=2.0,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(tmp_path),
        current_session_id=CURRENT,
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


def _add_summary(engine, summary, *, session_id, created_at, latest_at=None):
    return engine._dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary=summary,
            token_count=20,
            source_token_count=40,
            source_ids=[],
            source_type="messages",
            created_at=created_at,
            earliest_at=created_at,
            latest_at=latest_at if latest_at is not None else created_at,
            expand_hint=f"Expand {summary[:20]}",
        )
    )


def _seed_summary_vectors(engine, rows, *, provider="mock"):
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("mock-model", provider, 2)
        identity = store.capture_identity("mock-model", provider=provider)
        for node_id, vector in rows:
            store.record_embedding(str(node_id), "summary", "mock-model", vector, identity=identity)
    finally:
        store.close()


def _chunk_identity():
    return EmbeddingIdentity.canonical("mock", "mock-model", "", 2, "float32", "little", "chunk")


def _seed_chunk_vectors(engine, rows):
    """rows: (store_id, chunk_index, char_start, char_end, vector)."""
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("mock-model", "mock", 2, task="chunk")
        identity = _chunk_identity()
        for store_id, chunk_index, char_start, char_end, vector in rows:
            store.record_chunk_embedding(
                f"{store_id}:{chunk_index}",
                "mock-model",
                vector,
                store_id=store_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                token_estimate=5,
                identity=identity,
            )
    finally:
        store.close()


def _recall(engine, monkeypatch, provider=None, **args):
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: provider or MockProvider())
    payload = json.loads(lcm_tools.lcm_recall({"query": "kanban dashboard sprint", **args}, engine=engine))
    return payload


def test_recall_returns_cross_session_summaries_without_a_filter(recall_engine, monkeypatch):
    other_a = _add_summary(recall_engine, "kanban board dashboard sprint plan", session_id="session-a", created_at=10.0)
    other_b = _add_summary(recall_engine, "fleet archive sprint board", session_id="session-b", created_at=11.0)
    here = _add_summary(recall_engine, "unrelated current note", session_id=CURRENT, created_at=12.0)
    _seed_summary_vectors(
        recall_engine,
        [(other_a, [1.0, 0.0]), (other_b, [0.9, 0.436]), (here, [0.0, 1.0])],
    )

    payload = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=0.0, limit=5)

    node_ids = [hit["node_id"] for hit in payload["hits"]]
    assert node_ids[0] == other_a and other_b in node_ids
    # The strongest hits come from OTHER conversations — no session filter applied.
    sessions = {hit["session_id"] for hit in payload["hits"][:2]}
    assert sessions == {"session-a", "session-b"}
    assert all(hit["kind"] == "summary" for hit in payload["hits"])
    assert payload["provenance"]["arms_run"] == ["summary"]


def test_scope_bias_boosts_current_conversation_without_filtering(recall_engine, monkeypatch):
    cross = _add_summary(recall_engine, "cross conversation kanban", session_id="session-a", created_at=5.0)
    here = _add_summary(recall_engine, "current conversation kanban", session_id=CURRENT, created_at=5.0)
    # cross scores higher (rank 1); current is rank 2.
    _seed_summary_vectors(recall_engine, [(cross, [1.0, 0.0]), (here, [0.95, 0.312])])

    neutral = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=0.0, limit=5)
    biased = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=1.0, limit=5)

    assert [h["node_id"] for h in neutral["hits"][:2]] == [cross, here]
    # A full scope bias lifts the current-conversation hit above the cross one,
    # yet the cross hit is still returned (boost, not filter).
    assert biased["hits"][0]["node_id"] == here
    assert cross in {h["node_id"] for h in biased["hits"]}


def test_recency_boost_moves_ranking(recall_engine, monkeypatch):
    old_strong = _add_summary(recall_engine, "old kanban board", session_id="session-a", created_at=1.0, latest_at=1.0)
    new_weak = _add_summary(recall_engine, "new kanban board", session_id="session-b", created_at=1.0, latest_at=time.time())
    # old_strong scores higher on cosine (rank 1) but is ancient; new_weak is rank 2 but fresh.
    _seed_summary_vectors(recall_engine, [(old_strong, [1.0, 0.0]), (new_weak, [0.95, 0.312])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", scope_bias=0.0, limit=5)

    assert payload["hits"][0]["node_id"] == new_weak
    assert old_strong in {h["node_id"] for h in payload["hits"]}


def test_chunk_hit_dedupes_against_fts_by_store_id(recall_engine, monkeypatch):
    store_id = recall_engine._store.append(
        CURRENT, {"role": "user", "content": "kanban dashboard sprint verbatim detail"}
    )
    _seed_chunk_vectors(recall_engine, [(store_id, 0, 0, 39, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    excerpt_hits = [h for h in payload["hits"] if h.get("store_id") == store_id]
    assert len(excerpt_hits) == 1
    hit = excerpt_hits[0]
    # The same message surfaced via both arms fuses into one entry carrying the
    # chunk span, and the expand handle points at the exact offset.
    assert set(hit["arms"]) == {"fts", "chunk"}
    assert hit["chunk_span"]["char_start"] == 0
    assert "content_offset=0" in hit["expand_hint"]


def test_include_verbatim_excludes_summaries(recall_engine, monkeypatch):
    node = _add_summary(recall_engine, "kanban summary only", session_id="session-a", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint raw"})

    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=10)

    assert "summary" not in payload["provenance"]["arms_run"]
    assert all(h["kind"] == "message_excerpt" for h in payload["hits"])


def test_embeddings_off_degrades_to_fts_arm(recall_engine, monkeypatch):
    recall_engine._config.embeddings_enabled = False
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint fallback"})

    payload = _recall(recall_engine, monkeypatch, include="all", limit=10)

    assert payload["degraded"] is True
    assert "disabled" in payload["degraded_reason"]
    assert payload["provenance"]["coverage"].get("summary") == "disabled"
    assert payload["hits"]
    assert all(h["kind"] == "message_excerpt" for h in payload["hits"])


def test_empty_vector_corpora_reports_coverage_none(recall_engine, monkeypatch):
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint only fts"})

    payload = _recall(recall_engine, monkeypatch, include="all", limit=10)

    # Vector corpora are empty (no summaries/chunks seeded) -> coverage none, but
    # the FTS arm still returns a hit, so the tool never bare-errors.
    assert payload["provenance"]["coverage"].get("summary") == "none"
    assert payload["degraded"] is True
    assert payload["hits"]


def test_rerank_disabled_by_default(recall_engine, monkeypatch):
    node = _add_summary(recall_engine, "kanban rerank off", session_id="session-a", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=5)
    assert payload["provenance"]["rerank"] == "disabled"


def test_rerank_skips_silently_on_non_voyage_provider(recall_engine, monkeypatch):
    recall_engine._config.rerank_enabled = True
    node = _add_summary(recall_engine, "kanban rerank skip", session_id="session-a", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(node, [1.0, 0.0])])

    payload = _recall(recall_engine, monkeypatch, include="summaries", limit=5)
    assert payload["provenance"]["rerank"].startswith("skipped")
    assert payload["hits"]  # order preserved from RRF, not dropped


def test_rerank_applies_and_reorders_with_voyage_provider(recall_engine, monkeypatch):
    recall_engine._config.rerank_enabled = True
    a = _add_summary(recall_engine, "kanban alpha", session_id="session-a", created_at=5.0)
    b = _add_summary(recall_engine, "kanban beta", session_id="session-b", created_at=5.0)
    # a is RRF rank 1, b rank 2 (seeded under the voyage identity the rerank
    # provider resolves KNN against).
    _seed_summary_vectors(recall_engine, [(a, [1.0, 0.0]), (b, [0.95, 0.312])], provider="voyage")

    class RerankProvider(MockProvider):
        provider_id = "voyage"

        def rerank(self, query, documents, *, top_k=None, timeout, model="rerank-2.5-lite"):
            # Flip relevance: the LAST document scores highest (index i -> score i),
            # returned in descending-relevance order as the real API does.
            return sorted(
                ((i, float(i)) for i in range(len(documents))), key=lambda item: -item[1]
            )

    payload = _recall(
        recall_engine, monkeypatch, provider=RerankProvider(), include="summaries", scope_bias=0.0, limit=5
    )
    assert payload["provenance"]["rerank"] == "applied"
    assert payload["hits"][0]["node_id"] == b


def test_rerank_failure_falls_back_to_rrf_order(recall_engine, monkeypatch):
    recall_engine._config.rerank_enabled = True
    a = _add_summary(recall_engine, "kanban gamma", session_id="session-a", created_at=5.0)
    b = _add_summary(recall_engine, "kanban delta", session_id="session-b", created_at=5.0)
    _seed_summary_vectors(recall_engine, [(a, [1.0, 0.0]), (b, [0.95, 0.312])], provider="voyage")

    class BrokenRerank(MockProvider):
        provider_id = "voyage"

        def rerank(self, *args, **kwargs):
            raise RuntimeError("rerank endpoint down")

    payload = _recall(
        recall_engine, monkeypatch, provider=BrokenRerank(), include="summaries", scope_bias=0.0, limit=5
    )
    assert payload["provenance"]["rerank"].startswith("skipped")
    assert payload["hits"][0]["node_id"] == a  # RRF order intact


def test_limit_is_capped_and_reported(recall_engine, monkeypatch):
    recall_engine._store.append(CURRENT, {"role": "user", "content": "kanban dashboard sprint cap"})
    payload = _recall(recall_engine, monkeypatch, include="verbatim", limit=1000)
    assert payload["limit"] == 25
    assert payload["limit_clamped_from"] == 1000


def test_missing_query_is_rejected(recall_engine):
    payload = json.loads(lcm_tools.lcm_recall({"query": "   "}, engine=recall_engine))
    assert "error" in payload


def test_recall_scans_full_corpus_not_grep_recency_window(recall_engine, monkeypatch):
    """Recall must NOT inherit grep's 2000-recent bound, or 'all time' truncates."""
    recall_engine._config.recall_scan_rows = 25_000
    recall_engine._config.embedding_bounded_scan_rows = 2_000
    observed: list[int] = []

    from hermes_lcm.vector_store import KNNResult

    class BoundCapturingStore:
        def __init__(self, *_args, bounded_scan_rows=None, **_kwargs):
            observed.append(bounded_scan_rows)

        def knn(self, *_args, **_kwargs):
            return KNNResult(coverage="none")

        def knn_chunks(self, *_args, **_kwargs):
            return KNNResult(coverage="none")

        def close(self):
            pass

    monkeypatch.setattr(lcm_tools, "VectorStore", BoundCapturingStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    json.loads(lcm_tools.lcm_recall({"query": "anything", "include": "all"}, engine=recall_engine))

    # Both vector arms request the large recall bound, never the small grep one.
    assert observed and all(bound == 25_000 for bound in observed)
