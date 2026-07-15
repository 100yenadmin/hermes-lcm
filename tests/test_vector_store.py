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
MIGRATION_STEP = f"v{db_bootstrap.SCHEMA_VERSION}_embeddings"


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


def test_embedding_migration_is_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "idempotent.db")
    try:
        db_bootstrap.run_versioned_migrations(conn)
        db_bootstrap.run_versioned_migrations(conn)
        conn.commit()

        assert EMBEDDING_TABLES <= _table_names(conn)
        assert db_bootstrap.get_schema_version(conn) == db_bootstrap.SCHEMA_VERSION
        steps = conn.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name = ?",
            (MIGRATION_STEP,),
        ).fetchall()
        assert steps == [(MIGRATION_STEP,)]
        index_sql = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND name = 'idx_lcm_embedding_meta_model_embedded_at'
            """
        ).fetchone()[0]
        assert "WHERE archived = 0" in index_sql
    finally:
        conn.close()


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


def test_profile_registration_locks_dimension_and_is_immutable(tmp_path):
    store = VectorStore(tmp_path / "profiles.db")
    try:
        store.register_profile("model-a", "provider-a", 3)
        store.register_profile("model-a", "provider-b", 3)

        row = store.connection.execute(
            "SELECT provider, dim FROM lcm_embedding_profile WHERE model_name = 'model-a'"
        ).fetchone()
        assert tuple(row) == ("provider-a", 3)
        with pytest.raises(ValueError, match="locked at 3"):
            store.register_profile("model-a", "provider-a", 4)
    finally:
        store.close()


def test_current_profile_uses_newest_active_unarchived(stores):
    dag, store = stores
    first = _add_summary(dag, created_at=1.0)
    second = _add_summary(dag, created_at=2.0)
    store.register_profile("older", "local", 3)
    store.register_profile("newer", "local", 3)
    store.connection.execute(
        "UPDATE lcm_embedding_profile SET registered_at = '2026-01-01' WHERE model_name = 'older'"
    )
    store.connection.execute(
        "UPDATE lcm_embedding_profile SET registered_at = '2026-02-01' WHERE model_name = 'newer'"
    )
    store.connection.commit()
    store.record_embedding(first, "summary", "older", [1.0, 0.0, 0.0])
    store.record_embedding(second, "summary", "newer", [0.0, 1.0, 0.0])

    assert store.knn([0.0, 1.0, 0.0])[0][0] == str(second)

    store.connection.execute(
        "UPDATE lcm_embedding_profile SET archived_at = '2026-03-01' WHERE model_name = 'newer'"
    )
    store.connection.commit()
    assert store.knn([1.0, 0.0, 0.0])[0][0] == str(first)

    store.connection.execute(
        "UPDATE lcm_embedding_profile SET active = 0 WHERE model_name = 'older'"
    )
    store.connection.commit()
    result = store.knn([1.0, 0.0, 0.0])
    assert result == []
    assert result.coverage == "none"


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
