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


class TestChunkRetryUncertainSpans:
    def test_rebuild_chunk_document_returns_real_span(self, tmp_path):
        from hermes_lcm.chunking import chunk_message

        engine = _engine(tmp_path)
        content = "some substantial user output token " * 40
        _seed_messages(engine, [(1, "sess-a", "history", "user", content, 1.0)])
        expected = chunk_message(1, "user", content, policy="conversational")[0]

        conn = sqlite3.connect(engine._store.db_path)
        try:
            rebuilt = command_mod._rebuild_chunk_document(
                conn, expected.chunk_id, "conversational"
            )
        finally:
            conn.close()

        assert rebuilt is not None
        text, tokens, char_start, char_end = rebuilt
        # The real char span is carried, not the old (0, 0) placeholder.
        assert (char_start, char_end) == (expected.char_start, expected.char_end)
        assert char_end > char_start

    def test_authorized_uncertain_rows_persist_real_span_not_zero(self, tmp_path):
        from hermes_lcm.chunking import chunk_message

        engine = _engine(tmp_path)
        content = "another long verbatim user payload body " * 30
        _seed_messages(engine, [(1, "sess-a", "history", "user", content, 1.0)])
        expected = chunk_message(1, "user", content, policy="conversational")[0]

        # _chunk_authorized_uncertain_rows only string-matches identity_hash (no
        # profile join), so any stable value the inflight row shares works here.
        identity = "test-chunk-identity"

        conn = sqlite3.connect(engine._store.db_path)
        try:
            from hermes_lcm import db_bootstrap

            db_bootstrap.ensure_embedding_tables(conn)
            db_bootstrap.ensure_chunk_tables(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lcm_embedding_backfill_inflight (
                    embedded_id TEXT, identity_hash TEXT, state TEXT,
                    updated_at REAL,
                    PRIMARY KEY(embedded_id, identity_hash)
                )
                """
            )
            conn.execute(
                "INSERT INTO lcm_embedding_backfill_inflight"
                "(embedded_id, identity_hash, state, updated_at) VALUES(?, ?, 'uncertain', 1.0)",
                (expected.chunk_id, identity),
            )
            conn.commit()
            count, documents, meta = command_mod._chunk_authorized_uncertain_rows(
                conn, identity, "conversational", 10
            )
        finally:
            conn.close()

        assert count == 1
        _sid, _idx, char_start, char_end = meta[expected.chunk_id]
        assert (char_start, char_end) == (expected.char_start, expected.char_end)
        assert char_end > char_start


class TestChunkRawTextConsentGate:
    def _voyage_engine(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        db_path = tmp_path / "backfill.db"
        config = LCMConfig(
            database_path=str(db_path),
            embeddings_enabled=True,
            embedding_provider="voyage",
            embedding_model="voyage-3",
        )
        engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=db_path))
        _seed_messages(engine, _user_msgs(2), register=False)
        store = VectorStore(db_path, config=config)
        try:
            store.register_profile("voyage-3", "voyage", 2, task="chunk")
        finally:
            store.close()
        return engine

    def test_cloud_apply_refused_without_confirm_flag(self, monkeypatch, tmp_path):
        engine = self._voyage_engine(tmp_path)
        provider = FakeProvider()
        monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

        out = handle_lcm_command("embed backfill --corpus chunks --apply", engine)

        assert "status: refused" in out
        assert "RAW, VERBATIM" in out
        assert "--confirm-raw-text" in out
        # Nothing was sent to the cloud provider and nothing was written.
        assert provider.calls == []
        assert _chunk_meta_ids(engine) == []

    def test_cloud_apply_proceeds_with_confirm_flag(self, monkeypatch, tmp_path):
        engine = self._voyage_engine(tmp_path)
        provider = FakeProvider()
        # Match the registered voyage chunk profile so the apply gets past the
        # provider/profile consistency check and actually embeds.
        provider.provider_id = "voyage"
        provider.model_id = "voyage-3"
        monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

        out = handle_lcm_command(
            "embed backfill --corpus chunks --apply --confirm-raw-text", engine
        )

        assert "status: refused" not in out
        assert _chunk_meta_ids(engine)  # chunks were embedded

    def test_local_provider_exempt_from_gate(self, monkeypatch, tmp_path):
        engine = _engine(tmp_path)  # ollama (local)
        _seed_messages(engine, _user_msgs(2))
        provider = FakeProvider()
        monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

        out = handle_lcm_command("embed backfill --corpus chunks --apply", engine)

        assert "status: refused" not in out
        assert _chunk_meta_ids(engine)

    def test_confirm_flag_rejected_for_summary_corpus(self, tmp_path):
        engine = _engine(tmp_path)
        out = handle_lcm_command("embed backfill --confirm-raw-text", engine)
        assert "only applies to the chunk corpus" in out


class TestBothCorpusNextHint:
    def test_both_dry_run_emits_single_coherent_next_hint(self, monkeypatch, tmp_path):
        engine = _engine(tmp_path)
        _seed_messages(engine, _user_msgs(2))
        provider = FakeProvider()
        monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

        out = handle_lcm_command("embed backfill --corpus both", engine)

        # Exactly one next-hint, and it names the actual `--corpus both` command.
        assert out.count("next: run") == 1
        assert "next: run `/lcm embed backfill --corpus both --apply`" in out
        # The two contradictory per-corpus hints are gone.
        assert "run `/lcm embed backfill --apply`" not in out
        assert "run `/lcm embed backfill --corpus chunks --apply`" not in out
        assert provider.calls == []


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

    def test_dry_run_display_honors_explicit_context_model(self, tmp_path):
        # H3: an explicit voyage context model is the chunk-model intent and the
        # dry-run display must equal it (single resolution path) rather than
        # forcing the voyage-context-4 mapping — this is what apply would use.
        engine = _engine(tmp_path)
        engine._config.embedding_provider = "voyage"
        engine._config.embedding_model = "voyage-context-3"
        _seed_messages(engine, _user_msgs(1), register=False)
        result = handle_lcm_command("embed backfill --corpus chunks", engine)
        assert "model: voyage-context-3" in result

    def test_dry_run_display_maps_plain_voyage_to_context_default(self, tmp_path):
        # A plain (non-context) voyage model has no explicit chunk-model intent,
        # so the voyage-context-4 mapping applies.
        engine = _engine(tmp_path)
        engine._config.embedding_provider = "voyage"
        engine._config.embedding_model = "voyage-3"
        _seed_messages(engine, _user_msgs(1), register=False)
        result = handle_lcm_command("embed backfill --corpus chunks", engine)
        assert "model: voyage-context-4" in result

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
