from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_lcm.embedding_provider as embedding_provider
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.embedding_provider import VoyageError
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import KNNResult, VectorStore


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
def semantic_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "semantic.db"),
        embeddings_enabled=True,
        embedding_provider="ollama",
        embedding_model="mock-model",
        embedding_query_timeout_s=0.1,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        current_session_id="session-a",
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


def _add_summary(engine, summary: str, *, created_at: float, source_ids=None) -> int:
    return engine._dag.add_node(
        SummaryNode(
            session_id="session-a",
            depth=0,
            summary=summary,
            token_count=20,
            source_token_count=40,
            source_ids=list(source_ids or []),
            source_type="messages",
            created_at=created_at,
            earliest_at=created_at,
            latest_at=created_at,
            expand_hint=f"Expand {summary[:20]}",
        )
    )


def _seed_vectors(engine, rows):
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("mock-model", "mock", 2)
        for node_id, vector in rows:
            store.record_embedding(str(node_id), "summary", "mock-model", vector)
    finally:
        store.close()


def test_semantic_happy_path_orders_by_cosine_and_surfaces_confidence_coverage(
    semantic_engine,
    monkeypatch,
):
    scores = [0.7, 0.55, 0.4, 0.2]
    node_ids = [
        _add_summary(
            semantic_engine,
            f"semantic result {index}",
            created_at=float(index + 1),
        )
        for index in range(len(scores))
    ]
    _seed_vectors(
        semantic_engine,
        [
            (node_id, [score, math.sqrt(1.0 - score * score)])
            for node_id, score in zip(node_ids, scores)
        ],
    )
    provider = MockProvider()
    monkeypatch.setattr(embedding_provider, "resolve_provider", lambda _config: provider)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: provider)

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "meaning-preserving query", "mode": "semantic", "limit": 4},
            engine=semantic_engine,
        )
    )

    assert [hit["node_id"] for hit in payload["results"]] == node_ids
    assert [hit["confidence"] for hit in payload["results"]] == [
        "high",
        "medium",
        "low",
        "noise",
    ]
    assert [hit["confidence_band"] for hit in payload["results"]] == [
        "high",
        "medium",
        "low",
        "noise",
    ]
    assert [hit["cosine_score"] for hit in payload["results"]] == pytest.approx(scores)
    assert payload["coverage"] in {"full", "bounded"}
    assert payload["degraded_to_fts"] is False
    assert provider.queries == ["meaning-preserving query"]


def test_semantic_timeout_degrades_to_fts_within_budget(semantic_engine, monkeypatch):
    semantic_engine._config.embedding_query_timeout_s = 0.02
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "needle survives provider timeout"},
    )

    class SlowProvider(MockProvider):
        def embed_query(self, text):
            time.sleep(0.1)
            return super().embed_query(text)

    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: SlowProvider())
    started = time.monotonic()
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "needle", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert time.monotonic() - started < 0.09
    assert payload["degraded_to_fts"] is True
    assert payload["coverage"] == "none"
    assert payload["results"][0]["type"] == "message"
    assert "latency budget" in payload["degraded_reason"]


def test_timeout_worker_is_daemon_and_provider_call_is_bounded(
    semantic_engine,
    monkeypatch,
):
    semantic_engine._config.embedding_query_timeout_s = 0.02
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "daemon timeout fallback"},
    )
    release = threading.Event()
    observed_timeouts: list[float] = []

    class BlockingProvider(MockProvider):
        def embed_query_interactive(self, _text, *, timeout):
            observed_timeouts.append(timeout)
            release.wait(1.0)
            return list(self.vector)

    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: BlockingProvider())
    try:
        payload = json.loads(
            lcm_tools.lcm_grep(
                {"query": "daemon", "mode": "semantic"},
                engine=semantic_engine,
            )
        )

        live_workers = [
            thread
            for thread in threading.enumerate()
            if thread.name == "lcm-query-embed" and thread.is_alive()
        ]
        assert payload["degraded_to_fts"] is True
        assert observed_timeouts == [pytest.approx(0.02)]
        assert live_workers
        assert all(thread.daemon for thread in live_workers)
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "lcm-query-embed":
                thread.join(timeout=0.2)


def test_semantic_missing_provider_degrades_to_fts(semantic_engine, monkeypatch):
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "local fallback marker"},
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: None)

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "marker", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert payload["degraded_to_fts"] is True
    assert payload["results"][0]["type"] == "message"
    assert "not configured" in payload["degraded_reason"]


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_none_vector_coverage_degrades_to_fts(
    semantic_engine,
    monkeypatch,
    mode,
):
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "coverage fallback marker"},
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "coverage", "mode": mode},
            engine=semantic_engine,
        )
    )

    assert payload["mode"] == mode
    assert payload["coverage"] == "none"
    assert payload["degraded_to_fts"] is True
    assert payload["results"][0]["type"] == "message"
    assert "coverage=none" in payload["degraded_reason"]


def test_provider_is_cached_until_provider_or_model_changes(
    semantic_engine,
    monkeypatch,
):
    resolved: list[MockProvider] = []

    def factory(config):
        provider = MockProvider()
        provider.model_id = config.embedding_model
        resolved.append(provider)
        return provider

    monkeypatch.setattr(lcm_tools, "resolve_provider", factory)

    for query in ("first", "second"):
        json.loads(
            lcm_tools.lcm_grep(
                {"query": query, "mode": "semantic"},
                engine=semantic_engine,
            )
        )

    assert len(resolved) == 1
    assert resolved[0].queries == ["first", "second"]

    semantic_engine._config.embedding_model = "changed-model"
    json.loads(
        lcm_tools.lcm_grep(
            {"query": "third", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert len(resolved) == 2
    assert resolved[1].model_id == "changed-model"
    assert resolved[1].queries == ["third"]


def test_semantic_auth_error_is_operator_readable_and_does_not_degrade(
    semantic_engine,
    monkeypatch,
):
    class AuthProvider(MockProvider):
        def embed_query(self, _text):
            raise VoyageError("auth", "bad credentials", status_code=401)

    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: AuthProvider())
    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "anything", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert "authentication failed" in payload["error"].lower()
    assert "degraded_to_fts" not in payload


def test_hybrid_rrf_deduplicates_nodes_and_rewards_both_arms(
    semantic_engine,
    monkeypatch,
):
    message_id = semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "fusion appears in a raw record"},
    )
    both_node = _add_summary(
        semantic_engine,
        "fusion appears in this embedded summary",
        created_at=2.0,
        source_ids=[message_id],
    )
    semantic_only = _add_summary(
        semantic_engine,
        "conceptual neighbor without the lexical term",
        created_at=1.0,
    )
    _seed_vectors(
        semantic_engine,
        [(both_node, [1.0, 0.0]), (semantic_only, [0.8, 0.6])],
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "fusion", "mode": "hybrid", "sort": "relevance", "limit": 10},
            engine=semantic_engine,
        )
    )

    node_hits = [hit for hit in payload["results"] if hit.get("node_id") == both_node]
    assert len(node_hits) == 1
    both_hit = node_hits[0]
    raw_hit = next(hit for hit in payload["results"] if hit.get("store_id") == message_id)
    assert both_hit["fts_rank"] >= 1 and both_hit["semantic_rank"] == 1
    assert both_hit["rrf_score"] == pytest.approx(
        1 / (60 + both_hit["fts_rank"]) + 1 / 61
    )
    assert both_hit["rrf_score"] > raw_hit["rrf_score"]
    assert payload["fusion"] == "rrf"
    assert payload["rrf_k"] == 60


@pytest.mark.parametrize(("limit", "expected_candidates"), [(1, 50), (40, 120), (200, 500)])
def test_hybrid_limit_controls_bounded_candidate_overfetch(
    semantic_engine,
    monkeypatch,
    limit,
    expected_candidates,
):
    observed: list[int] = []

    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def knn(self, _query, *, k, **_kwargs):
            observed.append(k)
            return KNNResult(coverage="full")

        def close(self):
            pass

    monkeypatch.setattr(lcm_tools, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "candidate cap", "mode": "hybrid", "limit": limit},
            engine=semantic_engine,
        )
    )

    assert "error" not in payload
    assert observed == [expected_candidates]
    assert payload["limit"] == limit


def test_semantic_snippets_are_bounded(semantic_engine, monkeypatch):
    node_id = _add_summary(semantic_engine, "x" * 1_000, created_at=1.0)
    _seed_vectors(semantic_engine, [(node_id, [1.0, 0.0])])
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())

    payload = json.loads(
        lcm_tools.lcm_grep(
            {"query": "bounded", "mode": "semantic"},
            engine=semantic_engine,
        )
    )

    assert len(payload["results"][0]["snippet"]) == 300


def test_full_text_modes_remain_byte_identical_with_embeddings_on_or_off(semantic_engine):
    semantic_engine._store.append(
        "session-a",
        {"role": "user", "content": "byte stable history result"},
    )

    for sort in ("recency", "relevance", "hybrid"):
        args = {"query": "stable", "sort": sort}
        semantic_engine._config.embeddings_enabled = False
        disabled = lcm_tools.lcm_grep(args, engine=semantic_engine)
        explicit = lcm_tools.lcm_grep({**args, "mode": "full_text"}, engine=semantic_engine)
        semantic_engine._config.embeddings_enabled = True
        enabled = lcm_tools.lcm_grep(args, engine=semantic_engine)
        assert disabled == explicit == enabled


def test_recall_eval_is_deterministic_and_hybrid_beats_fts_on_paraphrases():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "eval_retrieval_recall.py"
    first = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    second = subprocess.run(
        [sys.executable, str(script), "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert first == second
    metrics = json.loads(first)
    assert metrics["hybrid"]["paraphrase"]["recall@5"] >= metrics["full_text"]["paraphrase"]["recall@5"]
    assert metrics["hybrid"]["paraphrase"]["recall@10"] >= metrics["full_text"]["paraphrase"]["recall@10"]
