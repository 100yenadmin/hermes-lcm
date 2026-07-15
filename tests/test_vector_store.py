from __future__ import annotations

import math
import sqlite3
from array import array

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.vector_store import VectorStore
import hermes_lcm.vector_store as vector_store_module


EMBEDDING_TABLES = {
    "lcm_embedding_profile",
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
}
MIGRATION_STEP = "embeddings_v1"


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


@pytest.fixture
def stores(tmp_path):
    db_path = tmp_path / "vectors.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path)
    try:
        yield dag, store
    finally:
        store.close()
        dag.close()


def _add_summary(
    dag: SummaryDAG,
    *,
    session_id: str = "conversation-a",
    source_token_count: int = 100,
    created_at: float = 1.0,
) -> int:
    return dag.add_node(
        SummaryNode(
            session_id=session_id,
            summary=f"summary for {session_id} at {created_at}",
            source_token_count=source_token_count,
            created_at=created_at,
            earliest_at=created_at,
            latest_at=created_at,
        )
    )


def test_core_migrations_omit_embedding_tables(tmp_path):
    """A disabled install stays at schema_version 5 with no embedding tables.

    Embedding tables are opt-in and never created by the core migration path,
    so a base build can open the DB and the numeric counter stays free for the
    temporal train (no v6 collision).
    """
    conn = sqlite3.connect(tmp_path / "core_only.db")
    try:
        db_bootstrap.run_versioned_migrations(conn)
        conn.commit()

        assert db_bootstrap.SCHEMA_VERSION == 5
        assert db_bootstrap.get_schema_version(conn) == 5
        assert not (EMBEDDING_TABLES & _table_names(conn))
        marker = conn.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name = ?",
            (MIGRATION_STEP,),
        ).fetchall()
        assert marker == []
    finally:
        conn.close()


def test_vector_store_creates_embedding_tables_lazily_and_idempotently(tmp_path):
    """VectorStore materializes the opt-in tables on first use, still at v5."""
    db_path = tmp_path / "idempotent.db"
    first = VectorStore(db_path)
    first.close()
    # Re-opening must not duplicate the marker or fail on existing tables.
    store = VectorStore(db_path)
    try:
        assert EMBEDDING_TABLES <= _table_names(store.connection)
        assert db_bootstrap.get_schema_version(store.connection) == 5
        steps = store.connection.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name = ?",
            (MIGRATION_STEP,),
        ).fetchall()
        assert [tuple(row) for row in steps] == [(MIGRATION_STEP,)]
        index_sql = store.connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND name = 'idx_lcm_embedding_meta_identity_embedded_at'
            """
        ).fetchone()[0]
        assert "WHERE archived = 0" in index_sql
    finally:
        store.close()


def test_vector_store_upgrades_previous_schema_version(tmp_path):
    db_path = tmp_path / "previous.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(db_bootstrap.SCHEMA_VERSION - 1),),
    )
    conn.commit()
    conn.close()

    store = VectorStore(db_path)
    try:
        assert EMBEDDING_TABLES <= _table_names(store.connection)
        assert db_bootstrap.get_schema_version(store.connection) == db_bootstrap.SCHEMA_VERSION
        completed = store.connection.execute(
            "SELECT completed_at FROM lcm_migration_state WHERE step_name = ?",
            (MIGRATION_STEP,),
        ).fetchone()
        assert completed is not None
    finally:
        store.close()


def test_vector_store_refuses_newer_schema_before_configuring_connection(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "future.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(db_bootstrap.SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()

    configure_called = False

    def fail_if_called(conn):
        nonlocal configure_called
        configure_called = True
        raise AssertionError("configure_connection should not run for future schemas")

    monkeypatch.setattr(vector_store_module, "configure_connection", fail_if_called)

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        VectorStore(db_path)
    assert configure_called is False
    check = sqlite3.connect(db_path)
    try:
        assert _table_names(check) == {"metadata"}
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        check.close()


def test_profile_identity_distinguishes_provider_without_clobber(tmp_path):
    """Same model_name under two providers is two profiles, not a silent overwrite."""
    store = VectorStore(tmp_path / "profiles.db")
    try:
        identity_a = store.register_profile("model-a", "provider-a", 3)
        identity_b = store.register_profile("model-a", "provider-b", 3)
        assert identity_a != identity_b

        rows = store.connection.execute(
            """
            SELECT provider, dim, active
            FROM lcm_embedding_profile
            WHERE model_name = 'model-a'
            ORDER BY provider
            """
        ).fetchall()
        # Both provider rows survive; provider-a's metadata was not clobbered.
        assert [tuple(r) for r in rows] == [
            ("provider-a", 3, 0),
            ("provider-b", 3, 1),
        ]
        # A different dim is a different identity (new row), not a locked error.
        identity_c = store.register_profile("model-a", "provider-a", 4)
        assert identity_c not in {identity_a, identity_b}
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lcm_embedding_profile WHERE model_name = 'model-a'"
        ).fetchone()[0] == 3
    finally:
        store.close()


def test_switch_provider_a_b_a_reactivates_without_rebackfill(stores):
    """Switching config A→B→A reactivates A's profile with its vectors intact."""
    dag, store = stores
    node_a = _add_summary(dag, created_at=1.0)
    node_b = _add_summary(dag, created_at=2.0)
    store.register_profile("shared-model", "provider-a", 3)
    store.record_embedding(node_a, "summary", "shared-model", [1.0, 0.0, 0.0])

    # Switch to provider B (same model name); A is retained but deactivated.
    store.register_profile("shared-model", "provider-b", 3)
    store.record_embedding(node_b, "summary", "shared-model", [0.0, 1.0, 0.0])
    current = store._current_profile()
    assert current["provider"] == "provider-b"

    # Switch back to A: no re-backfill, A's vector must resolve again.
    store.register_profile("shared-model", "provider-a", 3)
    current = store._current_profile()
    assert current["provider"] == "provider-a"
    result = store.knn([1.0, 0.0, 0.0])
    assert [row[0] for row in result] == [str(node_a)]
    # Only exactly one profile is active at a time.
    active = store.connection.execute(
        "SELECT COUNT(*) FROM lcm_embedding_profile WHERE active = 1 AND archived_at IS NULL"
    ).fetchone()[0]
    assert active == 1


def test_record_and_knn_match_hand_computed_cosines(stores):
    dag, store = stores
    axis_x = _add_summary(dag, source_token_count=11, created_at=1.0)
    diagonal = _add_summary(dag, source_token_count=22, created_at=2.0)
    axis_y = _add_summary(dag, source_token_count=33, created_at=3.0)
    store.register_profile("three-d", "local", 3)
    store.record_embedding(axis_x, "summary", "three-d", [1.0, 0.0, 0.0])
    store.record_embedding(diagonal, "summary", "three-d", [1.0, 1.0, 0.0])
    store.record_embedding(axis_y, "summary", "three-d", [0.0, 1.0, 0.0])

    result = store.knn([1.0, 0.0, 0.0], k=3, model="three-d")

    assert [row[0] for row in result] == [str(axis_x), str(diagonal), str(axis_y)]
    assert [row[1] for row in result] == pytest.approx(
        [1.0, 1.0 / math.sqrt(2.0), 0.0],
        abs=1e-6,
    )
    assert [row[2] for row in result] == ["summary", "summary", "summary"]
    assert result.coverage in {"full", "bounded"}
    token_counts = store.connection.execute(
        """
        SELECT embedded_id, source_token_count
        FROM lcm_embedding_meta
        ORDER BY CAST(embedded_id AS INTEGER)
        """
    ).fetchall()
    assert [tuple(row) for row in token_counts] == [
        (str(axis_x), 11),
        (str(diagonal), 22),
        (str(axis_y), 33),
    ]


def test_record_normalizes_vector_before_packing(stores):
    dag, store = stores
    node_id = _add_summary(dag)
    store.register_profile("normalized", "local", 3)
    store.record_embedding(node_id, "summary", "normalized", [3.0, 4.0, 0.0])

    blob = store.connection.execute(
        "SELECT vec FROM lcm_embedding_vectors WHERE embedded_id = ?",
        (str(node_id),),
    ).fetchone()[0]
    unpacked = array("f")
    unpacked.frombytes(blob)
    assert list(unpacked) == pytest.approx([0.6, 0.8, 0.0], abs=1e-6)
    result = store.knn([3.0, 4.0, 0.0], model="normalized")
    assert result[0][1] == pytest.approx(1.0, abs=1e-6)


def test_numpy_absent_uses_bounded_scan_with_same_top_k(stores, monkeypatch):
    dag, store = stores
    first = _add_summary(dag, created_at=1.0)
    second = _add_summary(dag, created_at=2.0)
    third = _add_summary(dag, created_at=3.0)
    store.register_profile("fallback", "local", 3)
    store.record_embedding(first, "summary", "fallback", [1.0, 0.0, 0.0])
    store.record_embedding(second, "summary", "fallback", [1.0, 1.0, 0.0])
    store.record_embedding(third, "summary", "fallback", [0.0, 1.0, 0.0])

    def unavailable():
        raise ImportError("numpy not installed")

    monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
    result = store.knn([1.0, 0.0, 0.0], k=3, model="fallback")

    assert result.coverage == "bounded"
    assert [row[0] for row in result] == [str(first), str(second), str(third)]
    assert [row[1] for row in result] == pytest.approx(
        [1.0, 1.0 / math.sqrt(2.0), 0.0],
        abs=1e-6,
    )


def test_bounded_scan_only_scores_most_recent_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "bounded.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=2)
    try:
        oldest = _add_summary(dag, created_at=1.0)
        recent_a = _add_summary(dag, created_at=2.0)
        recent_b = _add_summary(dag, created_at=3.0)
        store.register_profile("bounded", "local", 3)
        # Backfill writes newest content first, so recent content can have the
        # lowest vector rowids. The bounded window must use embedded_at instead.
        store.record_embedding(recent_a, "summary", "bounded", [0.0, 1.0, 0.0])
        store.record_embedding(recent_b, "summary", "bounded", [0.0, 0.0, 1.0])
        store.record_embedding(oldest, "summary", "bounded", [1.0, 0.0, 0.0])
        store.connection.executemany(
            "UPDATE lcm_embedding_meta SET embedded_at = ? WHERE embedded_id = ?",
            [
                ("2026-07-15T03:00:00+00:00", str(recent_b)),
                ("2026-07-15T02:00:00+00:00", str(recent_a)),
                ("2026-07-15T01:00:00+00:00", str(oldest)),
            ],
        )
        store.connection.commit()

        def unavailable():
            raise ImportError("numpy not installed")

        monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
        result = store.knn([1.0, 0.0, 0.0], k=3, model="bounded")

        assert result.coverage == "bounded"
        assert {row[0] for row in result} == {str(recent_a), str(recent_b)}
        assert str(oldest) not in {row[0] for row in result}
    finally:
        store.close()
        dag.close()


def test_suppressed_summaries_are_filtered_and_purge_removes_embeddings(stores):
    dag, store = stores
    suppressed = _add_summary(dag, created_at=1.0)
    kept = _add_summary(dag, created_at=2.0)
    store.register_profile("suppression", "local", 3)
    store.record_embedding(suppressed, "summary", "suppression", [1.0, 0.0, 0.0])
    store.record_embedding(kept, "summary", "suppression", [0.0, 1.0, 0.0])
    store.connection.execute("ALTER TABLE summary_nodes ADD COLUMN suppressed_at TEXT")
    store.connection.execute(
        "UPDATE summary_nodes SET suppressed_at = '2026-07-15' WHERE node_id = ?",
        (suppressed,),
    )
    store.connection.commit()

    result = store.knn([1.0, 0.0, 0.0], k=2, model="suppression")
    assert [row[0] for row in result] == [str(kept)]

    assert store.purge_embeddings_for_nodes([kept, kept]) == 1
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_embedding_meta WHERE embedded_id = ?",
        (str(kept),),
    ).fetchone()[0] == 0
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_embedding_vectors WHERE embedded_id = ?",
        (str(kept),),
    ).fetchone()[0] == 0


def test_filter_overfetch_uses_summary_time_and_session_columns(stores):
    dag, store = stores
    old_a = _add_summary(dag, session_id="conversation-a", created_at=10.0)
    new_a = _add_summary(dag, session_id="conversation-a", created_at=20.0)
    new_b = _add_summary(dag, session_id="conversation-b", created_at=30.0)
    store.register_profile("filters", "local", 3)
    store.record_embedding(old_a, "summary", "filters", [1.0, 0.0, 0.0])
    store.record_embedding(new_a, "summary", "filters", [0.9, 0.1, 0.0])
    store.record_embedding(new_b, "summary", "filters", [0.8, 0.2, 0.0])

    result = store.knn(
        [1.0, 0.0, 0.0],
        k=1,
        model="filters",
        since=15.0,
        conversation_ids=["conversation-a"],
    )
    assert [row[0] for row in result] == [str(new_a)]


def test_filters_are_applied_before_score_top_k_truncation(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    store.register_profile("filter-before-top-k", "local", 2)

    for index in range(501):
        unfiltered = _add_summary(
            dag,
            session_id="conversation-other",
            created_at=float(index + 1),
        )
        store.record_embedding(
            unfiltered,
            "summary",
            "filter-before-top-k",
            [1.0, 0.0],
        )

    filtered_ids = [
        _add_summary(
            dag,
            session_id="conversation-target",
            created_at=1_000.0 + index,
        )
        for index in range(2)
    ]
    for node_id in filtered_ids:
        store.record_embedding(
            node_id,
            "summary",
            "filter-before-top-k",
            [0.0, 1.0],
        )

    result = store.knn(
        [1.0, 0.0],
        k=2,
        model="filter-before-top-k",
        conversation_ids=["conversation-target"],
    )

    assert result.coverage == "full"
    assert {row[0] for row in result} == {str(node_id) for node_id in filtered_ids}


def test_matrix_cache_is_invalidated_on_write(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    first = _add_summary(dag, created_at=1.0)
    second = _add_summary(dag, created_at=2.0)
    store.register_profile("cache", "local", 3)
    store.record_embedding(first, "summary", "cache", [1.0, 0.0, 0.0])

    initial = store.knn([0.0, 1.0, 0.0], model="cache")
    assert initial[0][0] == str(first)
    assert store._matrix_cache

    store.record_embedding(second, "summary", "cache", [0.0, 1.0, 0.0])
    assert store._matrix_cache == {}
    updated = store.knn([0.0, 1.0, 0.0], model="cache")
    assert updated[0][0] == str(second)


def test_time_to_filter_excludes_before_top_k(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    store.register_profile("time-to", "local", 2)
    # 501 high-scoring but too-new vectors must not consume the top-k slots.
    for index in range(501):
        too_new = _add_summary(dag, created_at=10_000.0 + index)
        store.record_embedding(too_new, "summary", "time-to", [1.0, 0.0])
    eligible = _add_summary(dag, created_at=5.0)
    store.record_embedding(eligible, "summary", "time-to", [0.0, 1.0])

    result = store.knn([1.0, 0.0], k=2, model="time-to", until=100.0)
    assert result.coverage == "full"
    assert [row[0] for row in result] == [str(eligible)]


def test_source_filter_enforced_before_top_k(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    conn = store.connection
    conn.execute("CREATE TABLE IF NOT EXISTS messages (store_id INTEGER PRIMARY KEY, source TEXT)")
    conn.execute("INSERT INTO messages(store_id, source) VALUES (1, 'keep'), (2, 'drop')")
    conn.commit()
    store.register_profile("source-filter", "local", 2)

    # Many high-scoring vectors from the wrong source must be excluded before cap.
    for index in range(300):
        wrong = dag.add_node(
            SummaryNode(
                session_id="conversation-a",
                summary=f"wrong {index}",
                source_ids=[2],
                created_at=float(index + 1),
            )
        )
        store.record_embedding(wrong, "summary", "source-filter", [1.0, 0.0])
    right = dag.add_node(
        SummaryNode(
            session_id="conversation-a",
            summary="right",
            source_ids=[1],
            created_at=1_000.0,
        )
    )
    store.record_embedding(right, "summary", "source-filter", [0.0, 1.0])

    result = store.knn([1.0, 0.0], k=3, model="source-filter", source="keep")
    assert result.coverage == "full"
    assert [row[0] for row in result] == [str(right)]


def test_data_version_bump_invalidates_cross_process_cache(stores, tmp_path):
    pytest.importorskip("numpy")
    dag, store = stores
    first = _add_summary(dag, created_at=1.0)
    second = _add_summary(dag, created_at=2.0)
    store.register_profile("shared", "local", 3)
    store.record_embedding(first, "summary", "shared", [1.0, 0.0, 0.0])

    # Process A opens its own connection and warms its matrix cache.
    process_a = VectorStore(store.db_path)
    try:
        warmed = process_a.knn([0.0, 1.0, 0.0], model="shared")
        assert [row[0] for row in warmed] == [str(first)]
        assert process_a._matrix_cache

        # Process B writes a new vector, bumping the durable data_version in the
        # same transaction. max_rowid/row_count alone would not reveal an
        # in-place rewrite, but the counter forces process A to reload.
        store.record_embedding(second, "summary", "shared", [0.0, 1.0, 0.0])

        refreshed = process_a.knn([0.0, 1.0, 0.0], model="shared")
        assert refreshed[0][0] == str(second)
    finally:
        process_a.close()


def test_large_id_metadata_resolve_scales_past_variable_limit(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    store.register_profile("bulk", "local", 2)
    conn = store.connection
    # Insert 40k summary+vector rows directly for speed; a giant WHERE id IN
    # (...) resolve would raise "too many SQL variables" at ~33k on this runtime.
    now = 1.0
    vec = array("f", store._normalized([1.0, 0.0], expected_dim=2)).tobytes()
    identity = store._current_profile()["identity_hash"]
    for node_id in range(1, 40_001):
        conn.execute(
            "INSERT INTO summary_nodes(node_id, session_id, depth, summary, "
            "source_token_count, source_ids, source_type, created_at, "
            "earliest_at, latest_at) VALUES (?, 'conversation-a', 0, 's', 1, "
            "'[]', 'messages', ?, ?, ?)",
            (node_id, now, now, now),
        )
    conn.executemany(
        "INSERT INTO lcm_embedding_vectors(embedded_id, identity_hash, vec) VALUES (?, ?, ?)",
        [(str(node_id), identity, vec) for node_id in range(1, 40_001)],
    )
    conn.executemany(
        "INSERT INTO lcm_embedding_meta(embedded_id, embedded_kind, identity_hash, "
        "embedded_at, source_token_count, archived) VALUES (?, 'summary', ?, '2026', 1, 0)",
        [(str(node_id), identity) for node_id in range(1, 40_001)],
    )
    conn.commit()
    store._matrix_cache.clear()

    result = store.knn(
        [1.0, 0.0],
        k=5,
        model="bulk",
        conversation_ids=["conversation-a"],
    )
    assert result.coverage == "full"
    assert len(result) == 5


def test_no_profile_or_vectors_returns_none_coverage(tmp_path):
    store = VectorStore(tmp_path / "none.db")
    try:
        no_profile = store.knn([1.0, 0.0, 0.0])
        assert no_profile == []
        assert no_profile.coverage == "none"

        store.register_profile("empty", "local", 3)
        no_vectors = store.knn([1.0, 0.0, 0.0])
        assert no_vectors == []
        assert no_vectors.coverage == "none"
    finally:
        store.close()


def test_embedding_config_defaults_are_inert_and_read_environment(monkeypatch):
    defaults = LCMConfig()
    assert defaults.embeddings_enabled is False
    assert defaults.embedding_bounded_scan_rows == 2_000

    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDING_BOUNDED_SCAN_ROWS", "123")
    configured = LCMConfig.from_env()
    assert configured.embeddings_enabled is True
    assert configured.embedding_bounded_scan_rows == 123

    store = VectorStore(":memory:")
    try:
        assert store.bounded_scan_rows == 123
    finally:
        store.close()
