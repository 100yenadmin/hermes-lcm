from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.embedding_provider import VoyageError
from hermes_lcm.vector_store import VectorStore


class FakeProvider:
    provider_id = "ollama"
    model_id = "model-a"

    def __init__(self, *, dim: int = 2):
        self.dim = dim
        self.calls: list[list[str]] = []
        self.last_skipped_documents: list[int] = []

    def embed_documents(self, texts):
        current = list(texts)
        self.calls.append(current)
        self.last_skipped_documents = []
        return [[float(index + 1), 1.0] for index, _text in enumerate(current)]


@pytest.fixture(autouse=True)
def deterministic_token_count(monkeypatch):
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))


def _engine(tmp_path, *, enabled: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "backfill.db"
    config = LCMConfig(
        database_path=str(db_path),
        embeddings_enabled=enabled,
        embedding_provider="ollama",
        embedding_model="model-a",
    )
    return SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=db_path),
    )


def _seed(engine, count: int, *, register: bool = True) -> list[int]:
    dag = SummaryDAG(engine._store.db_path)
    try:
        node_ids = [
            dag.add_node(SummaryNode(
                session_id="session-a",
                depth=0,
                summary=f"summary-{index}",
                source_token_count=100 + index,
                created_at=float(index + 1),
                latest_at=float(index + 1),
            ))
            for index in range(count)
        ]
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=1,
            summary="not a leaf",
            created_at=10_000.0,
        ))
    finally:
        dag.close()
    if register:
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile("model-a", "ollama", 2)
        finally:
            store.close()
    return node_ids


def _meta_ids(engine) -> list[str]:
    conn = sqlite3.connect(engine._store.db_path)
    try:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT embedded_id FROM lcm_embedding_meta ORDER BY CAST(embedded_id AS INTEGER)"
            ).fetchall()
        ]
    finally:
        conn.close()


def _claim_value(engine):
    conn = sqlite3.connect(engine._store.db_path)
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,),
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def test_dry_run_reports_counts_tokens_and_cost_without_calls_or_writes(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    before = engine._store.db_path.read_bytes()

    result = handle_lcm_command("embed backfill --limit 2", engine)

    assert "status: dry-run" in result
    assert "pending: 3" in result
    assert "selected: 2" in result
    assert "estimated_tokens: 18" in result
    assert "estimated_cost_usd: $0.000000" in result
    assert "tokens_consumed: 0" in result
    assert provider.calls == []
    assert _meta_ids(engine) == []
    assert engine._store.db_path.read_bytes() == before

    voyage_engine = _engine(tmp_path / "voyage")
    _seed(voyage_engine, 2)
    conn = sqlite3.connect(voyage_engine._store.db_path)
    conn.execute(
        """
        UPDATE lcm_embedding_profile
        SET model_name = 'voyage-4-lite', provider = 'voyage'
        WHERE model_name = 'model-a'
        """
    )
    conn.commit()
    conn.close()
    voyage_engine._config.embedding_provider = "voyage"
    voyage_engine._config.embedding_model = "voyage-4-lite"
    monkeypatch.setattr(
        command_mod,
        "count_tokens",
        lambda text: 10_000 if str(text).endswith("0") else 30_000,
    )

    voyage_result = handle_lcm_command("embed backfill", voyage_engine)

    assert "estimated_tokens: 40000" in voyage_result
    assert "estimated_batches: 1" in voyage_result
    assert "estimated_cost_usd: $0.000200" in voyage_result
    assert provider.calls == []


def test_apply_batches_records_correct_meta_and_is_idempotent(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 35)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    first = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 35" in first
    assert "remaining: 0" in first
    assert [len(batch) for batch in provider.calls] == [32, 3]
    assert _meta_ids(engine) == [str(node_id) for node_id in node_ids]
    conn = sqlite3.connect(engine._store.db_path)
    try:
        rows = conn.execute(
            """
            SELECT m.embedded_kind, p.model_name, p.provider, m.source_token_count
            FROM lcm_embedding_meta m
            JOIN lcm_embedding_profile p ON p.identity_hash = m.identity_hash
            ORDER BY CAST(m.embedded_id AS INTEGER)
            """
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("summary", "model-a", "ollama", 100 + index) for index in range(35)
    ]

    second = handle_lcm_command("embed backfill --apply", engine)
    assert "selected: 0" in second
    assert "embedded: 0" in second
    assert [len(batch) for batch in provider.calls] == [32, 3]


def test_apply_limit_embeds_newest_rows_first(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 4)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --limit 2 --apply", engine)

    assert "embedded: 2" in result
    assert "remaining: 2" in result
    assert _meta_ids(engine) == [str(node_ids[-2]), str(node_ids[-1])]


def test_per_row_record_failure_does_not_lose_rest_of_batch(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    original = VectorStore.record_embedding

    def fail_one(self, embedded_id, kind, model, vector):
        if str(embedded_id) == str(node_ids[1]):
            raise sqlite3.OperationalError("synthetic row failure")
        return original(self, embedded_id, kind, model, vector)

    monkeypatch.setattr(VectorStore, "record_embedding", fail_one)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 2" in result
    assert "failed: 1" in result
    assert f"node_id={node_ids[1]} reason=record_error:synthetic row failure" in result
    assert _meta_ids(engine) == [str(node_ids[0]), str(node_ids[2])]


def test_auth_error_aborts_immediately_and_releases_claim(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, 35)
    provider = FakeProvider()

    def auth_error(_texts):
        provider.calls.append(["attempt"])
        raise VoyageError("auth", "bad credentials", status_code=401)

    provider.provider_id = "voyage"
    engine._config.embedding_provider = "voyage"
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        "UPDATE lcm_embedding_profile SET provider = 'voyage' WHERE model_name = 'model-a'"
    )
    conn.commit()
    conn.close()
    provider.embed_documents = auth_error
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "status: error" in result
    assert "provider authentication failed" in result
    assert len(provider.calls) == 1
    assert _claim_value(engine) is None


def test_transient_provider_error_skips_batch_and_continues(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, 33)
    provider = FakeProvider()
    original = provider.embed_documents

    def transient_once(texts):
        if not provider.calls:
            provider.calls.append(list(texts))
            raise VoyageError("network", "temporary network failure")
        return original(texts)

    provider.embed_documents = transient_once
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    # A failed batch must NOT be reported as complete — the run only partially
    # embedded the discovered work.
    assert "status: partial" in result
    assert "embedded: 1" in result
    assert "failed: 32" in result
    assert "remaining: 32" in result
    assert len(provider.calls) == 2
    assert _claim_value(engine) is None


def test_provider_overcap_rows_are_skipped_and_left_pending(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 3)
    provider = FakeProvider()

    def skip_middle(texts):
        provider.calls.append(list(texts))
        provider.last_skipped_documents = [1]
        return [[1.0, 1.0], [2.0, 1.0]]

    provider.embed_documents = skip_middle
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 2" in result
    assert "skipped_overcap: 1" in result
    assert "remaining: 1" in result
    assert f"node_id={node_ids[1]} reason=provider_document_token_cap" in result


def test_fresh_claim_refuses_second_worker_but_stale_claim_is_overridden(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    _seed(engine, 1)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        (
            command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,
            json.dumps({"owner": "first", "claimed_at": time.time()}),
        ),
    )
    conn.commit()
    conn.close()

    refused = handle_lcm_command("embed backfill --apply", engine)
    assert "status: refused" in refused
    assert "holds the lease" in refused
    assert provider.calls == []

    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        "UPDATE metadata SET value = ? WHERE key = ?",
        (
            json.dumps({"owner": "stale", "claimed_at": time.time() - 601}),
            command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,
        ),
    )
    conn.commit()
    conn.close()
    applied = handle_lcm_command("embed backfill --apply", engine)
    assert "status: complete" in applied
    assert "embedded: 1" in applied
    assert _claim_value(engine) is None


def test_apply_claims_before_discovery_and_skips_already_embedded(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    # Another writer embeds the newest row before this run claims + discovers.
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.record_embedding(str(node_ids[-1]), "summary", "model-a", [1.0, 1.0])
    finally:
        store.close()

    result = handle_lcm_command("embed backfill --apply", engine)

    # Discovery runs AFTER the lease is claimed, so the already-embedded newest
    # row is excluded rather than re-sent to the provider.
    assert "selected: 2" in result
    assert "embedded: 2" in result
    sent = [doc for batch in provider.calls for doc in batch]
    assert "summary-2" not in sent
    assert set(_meta_ids(engine)) == {str(node_id) for node_id in node_ids}


def test_heartbeat_lease_blocks_takeover_until_expiry(tmp_path):
    db_path = tmp_path / "lease.db"
    store = VectorStore(db_path)
    try:
        conn = store.connection
        command_mod._ensure_inflight_table(conn)
        lease = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_000.0
        )
        assert lease is not None
        # A second worker cannot take a live lease within the TTL.
        assert command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_100.0
        ) is None
        # A heartbeat near the original expiry extends the lease.
        assert lease.renew(now=1_590.0, force=True) is True
        assert command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_595.0
        ) is None
        # Only once the lease is truly expired (past last heartbeat + TTL) can a
        # second worker steal it — and the original can no longer renew.
        stolen = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_590.0 + 601.0
        )
        assert stolen is not None
        assert lease.renew(now=1_590.0 + 602.0, force=True) is False
    finally:
        store.close()


def test_inflight_row_is_reattempted_after_crash(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 2)
    provider = FakeProvider()

    def crash(texts):
        provider.calls.append(list(texts))
        raise VoyageError("network", "provider crashed mid-batch")

    provider.embed_documents = crash
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    first = handle_lcm_command("embed backfill --apply", engine)
    # Nothing recorded; both rows are left marked in_flight.
    assert "status: failed" in first
    assert "embedded: 0" in first
    assert "in_flight: 2" in first
    assert _meta_ids(engine) == []

    healthy = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: healthy)
    second = handle_lcm_command("embed backfill --apply", engine)
    # The in_flight rows are re-discovered and embedded; markers clear.
    assert "status: complete" in second
    assert "embedded: 2" in second
    assert "in_flight: 0" in second
    assert _meta_ids(engine) == [str(node_id) for node_id in node_ids]


def test_operation_budget_stops_run_between_batches(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, 40)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    monkeypatch.setenv("LCM_EMBEDDING_BACKFILL_BUDGET_S", "1")

    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        # First call is the run start; every later call is well past the budget.
        return 0.0 if calls["n"] == 1 else 1_000.0

    monkeypatch.setattr(command_mod.time, "monotonic", fake_monotonic)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "stop_reason: op_budget_exhausted" in result
    assert "status: partial" in result
    assert "embedded: 0" in result
    assert provider.calls == []


def test_disabled_and_missing_profile_refuse_cleanly(tmp_path):
    disabled = _engine(tmp_path / "disabled", enabled=False)
    assert "embeddings are disabled" in handle_lcm_command(
        "embed backfill", disabled
    )

    missing = _engine(tmp_path / "missing")
    _seed(missing, 1, register=False)
    result = handle_lcm_command("embed backfill --apply", missing)
    assert "status: refused" in result
    assert "no current embedding profile" in result
    assert "/lcm embed warmup" in result
