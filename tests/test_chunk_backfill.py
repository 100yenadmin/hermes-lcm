from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
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
    # len-based token count keeps chunk-size math simple and deterministic.
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))
    import hermes_lcm.chunking as chunking
    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(str(text)))


def _engine(tmp_path, *, enabled: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "backfill.db"
    config = LCMConfig(
        database_path=str(db_path),
        embeddings_enabled=enabled,
        embedding_provider="ollama",
        embedding_model="model-a",
    )
    return SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=db_path))


def _seed_messages(engine, rows, *, register: bool = True):
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            store_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO messages(store_id, session_id, source, role, content, timestamp) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    if register:
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile("model-a", "ollama", 2, task="chunk")
        finally:
            store.close()


def _chunk_meta_ids(engine) -> list[str]:
    conn = sqlite3.connect(engine._store.db_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_chunk_meta'"
        ).fetchone()
        if exists is None:
            return []
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT chunk_id FROM lcm_chunk_meta ORDER BY chunk_id"
            ).fetchall()
        ]
    finally:
        conn.close()


def _user_msgs(n, *, start=1):
    return [
        (i, "sess-a", "history", "user", "u" * 60, float(i))
        for i in range(start, start + n)
    ]


class TestChunkDryRun:
    def test_reports_pending_without_writes(self, monkeypatch, tmp_path):
        engine = _engine(tmp_path)
        _seed_messages(engine, _user_msgs(3))
        provider = FakeProvider()
        monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)
        before = engine._store.db_path.read_bytes()

        result = handle_lcm_command("embed backfill --corpus chunks", engine)

        assert "corpus: chunks" in result
        assert "policy: conversational" in result
        assert "status: dry-run" in result
        assert "pending: 3" in result
        assert provider.calls == []
        assert _chunk_meta_ids(engine) == []
        assert engine._store.db_path.read_bytes() == before

    def test_estimates_without_registered_profile(self, tmp_path):
        engine = _engine(tmp_path)
        _seed_messages(engine, _user_msgs(2), register=False)
        result = handle_lcm_command("embed backfill --corpus chunks", engine)
        assert "status: dry-run" in result
        assert "pending: 2" in result

    def test_policy_heads_includes_tool_heads(self, tmp_path):
        engine = _engine(tmp_path)
        rows = _user_msgs(1) + [(2, "sess-a", "history", "tool", "t" * 60, 2.0)]
        _seed_messages(engine, rows)
        conv = handle_lcm_command("embed backfill --corpus chunks --policy conversational", engine)
        heads = handle_lcm_command("embed backfill --corpus chunks --policy heads", engine)
        assert "pending: 1" in conv  # tool skipped under conversational
        assert "pending: 2" in heads  # tool head added under heads

    def test_disabled_is_refused(self, tmp_path):
        engine = _engine(tmp_path, enabled=False)
        _seed_messages(engine, _user_msgs(1), register=False)
        result = handle_lcm_command("embed backfill --corpus chunks", engine)
        assert "status: refused" in result


class TestChunkApply:
    def test_records_meta_and_is_idempotent(self, monkeypatch, tmp_path):
        engine = _engine(tmp_path)
        _seed_messages(engine, _user_msgs(3))
        provider = FakeProvider()
        monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

        first = handle_lcm_command("embed backfill --corpus chunks --apply", engine)
        assert "embedded: 3" in first
        assert _chunk_meta_ids(engine) == ["1:0", "2:0", "3:0"]

        second = handle_lcm_command("embed backfill --corpus chunks --apply", engine)
        assert "pending: 0" in second
        assert "embedded: 0" in second
        assert _chunk_meta_ids(engine) == ["1:0", "2:0", "3:0"]

    def test_lease_and_uncertain_retry(self, monkeypatch, tmp_path):
        engine = _engine(tmp_path)
        _seed_messages(engine, _user_msgs(2))
        provider = FakeProvider()
        monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

        # Force a durable uncertain marker for chunk "1:0".
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.ensure_chunk_schema()
            conn = store.connection
            command_mod._ensure_inflight_table(conn)
            identity = str(conn.execute(
                "SELECT identity_hash FROM lcm_embedding_profile WHERE task='chunk' AND active=1"
            ).fetchone()[0])
            conn.execute(
                "INSERT INTO lcm_embedding_backfill_inflight("
                "embedded_id, identity_hash, lease_id, generation, claimed_at, "
                "state, request_id, updated_at, last_error) "
                "VALUES('1:0', ?, 'prior', 1, 1, 'uncertain', 'prior-req', 1, 'unknown')",
                (identity,),
            )
            conn.commit()
        finally:
            store.close()

        # Ordinary apply must NOT auto-retry the uncertain chunk.
        ordinary = handle_lcm_command("embed backfill --corpus chunks --apply", engine)
        assert "1:0" not in _chunk_meta_ids(engine)
        assert "2:0" in _chunk_meta_ids(engine)
        assert "embedded: 1" in ordinary

        # Explicit authorization re-embeds only the uncertain chunk.
        retry = handle_lcm_command(
            "embed backfill --corpus chunks --apply --retry-uncertain", engine
        )
        assert "1:0" in _chunk_meta_ids(engine)
        assert "embedded: 1" in retry
