from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.rollup_store import RollupStore


ROLLUP_TABLES = {"lcm_rollups", "lcm_rollup_sources", "lcm_rollup_state"}


@pytest.fixture
def rollup_store(tmp_path):
    store = RollupStore(tmp_path / "rollups.db")
    try:
        yield store
    finally:
        store.close()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _ready_rollup(
    store: RollupStore,
    period_kind: str,
    period_start: str,
    scope: str = "global",
    *,
    source_node_ids: list[int] | None = None,
) -> int:
    rollup_id = store.upsert_building(period_kind, period_start, scope)
    store.mark_ready(
        rollup_id,
        f"{period_kind} summary for {period_start}",
        42,
        source_node_ids or [],
        f"fingerprint-{period_kind}-{period_start}-{scope}",
    )
    return rollup_id


def test_temporal_rollup_migration_is_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "idempotent.db")
    try:
        db_bootstrap.run_versioned_migrations(conn)
        db_bootstrap.run_versioned_migrations(conn)
        conn.commit()

        assert ROLLUP_TABLES <= _table_names(conn)
        assert db_bootstrap.get_schema_version(conn) == db_bootstrap.SCHEMA_VERSION
        steps = conn.execute(
            """
            SELECT step_name
            FROM lcm_migration_state
            WHERE step_name = 'v6_temporal_rollups'
            """
        ).fetchall()
        assert steps == [("v6_temporal_rollups",)]
        index_sql = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_lcm_rollups_ready_period'
            """
        ).fetchone()[0]
        assert "WHERE status = 'ready'" in index_sql
    finally:
        conn.close()


def test_rollup_store_upgrades_previous_schema_version(tmp_path):
    db_path = tmp_path / "previous.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(db_bootstrap.SCHEMA_VERSION - 1),),
    )
    conn.commit()
    conn.close()

    store = RollupStore(db_path)
    try:
        assert ROLLUP_TABLES <= _table_names(store.connection)
        assert db_bootstrap.get_schema_version(store.connection) == db_bootstrap.SCHEMA_VERSION
        completed = store.connection.execute(
            "SELECT completed_at FROM lcm_migration_state WHERE step_name = ?",
            ("v6_temporal_rollups",),
        ).fetchone()
        assert completed is not None
    finally:
        store.close()


def test_rollup_store_refuses_newer_schema_before_configuring_connection(
    tmp_path, monkeypatch
):
    import hermes_lcm.rollup_store as rollup_store_module

    db_path = tmp_path / "future.db"
    conn = sqlite3.connect(db_path)
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

    monkeypatch.setattr(rollup_store_module, "configure_connection", fail_if_called)

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        RollupStore(db_path)
    assert configure_called is False
    check = sqlite3.connect(db_path)
    try:
        assert _table_names(check) == {"metadata"}
    finally:
        check.close()


def test_rollup_crud_round_trip_and_rebuild(rollup_store):
    rollup_id = rollup_store.upsert_building(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    ) == {
        "rollup_id": rollup_id,
        "period_kind": "day",
        "period_start": "2026-07-15",
        "scope": "conversation:conv-1",
        "summary": None,
        "token_count": None,
        "status": "building",
        "built_at": None,
        "source_fingerprint": None,
        "error": None,
        "source_node_ids": [],
    }

    rollup_store.mark_ready(
        rollup_id,
        "Daily summary",
        123,
        [9, 3, 9],
        "fingerprint-1",
    )
    ready = rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert ready["summary"] == "Daily summary"
    assert ready["token_count"] == 123
    assert ready["status"] == "ready"
    assert ready["built_at"] is not None
    assert ready["source_fingerprint"] == "fingerprint-1"
    assert ready["source_node_ids"] == [3, 9]

    rebuilt_id = rollup_store.upsert_building(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert rebuilt_id == rollup_id
    rebuilding = rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert rebuilding["status"] == "building"
    assert rebuilding["summary"] is None
    assert rebuilding["source_node_ids"] == []

    rollup_store.mark_failed(rollup_id, "summarizer unavailable")
    failed = rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert failed["status"] == "failed"
    assert failed["error"] == "summarizer unavailable"


def test_mark_ready_rejects_unknown_rollup_without_orphan_sources(rollup_store):
    with pytest.raises(ValueError, match="unknown rollup_id"):
        rollup_store.mark_ready(999, "missing", 1, [10], "fingerprint")

    count = rollup_store.connection.execute(
        "SELECT COUNT(*) FROM lcm_rollup_sources"
    ).fetchone()[0]
    assert count == 0


def test_ready_rollups_for_window_is_inclusive_and_scoped(rollup_store):
    _ready_rollup(rollup_store, "day", "2026-07-01", "global")
    second = _ready_rollup(rollup_store, "day", "2026-07-15", "global")
    third = _ready_rollup(rollup_store, "day", "2026-07-31", "global")
    _ready_rollup(rollup_store, "day", "2026-07-15", "conversation:other")
    failed = rollup_store.upsert_building("day", "2026-07-20", "global")
    rollup_store.mark_failed(failed, "failed")

    rows = rollup_store.ready_rollups_for_window(
        "day", "2026-07-15", "2026-07-31", "global"
    )

    assert [row["rollup_id"] for row in rows] == [second, third]
    assert [row["period_start"] for row in rows] == ["2026-07-15", "2026-07-31"]


def test_mark_stale_for_day_cascades_to_containing_week_and_month(rollup_store):
    scope = "conversation:conv-1"
    affected_ids = {
        _ready_rollup(rollup_store, "day", "2026-07-15", scope),
        _ready_rollup(rollup_store, "week", "2026-07-13", scope),
        _ready_rollup(rollup_store, "month", "2026-07-01", scope),
    }
    untouched_ids = {
        _ready_rollup(rollup_store, "day", "2026-07-16", scope),
        _ready_rollup(rollup_store, "week", "2026-07-06", scope),
        _ready_rollup(rollup_store, "month", "2026-07-01", "global"),
    }

    assert rollup_store.mark_stale_for_day("2026-07-15", scope) == 3

    statuses = dict(
        rollup_store.connection.execute(
            "SELECT rollup_id, status FROM lcm_rollups"
        ).fetchall()
    )
    assert {statuses[rollup_id] for rollup_id in affected_ids} == {"stale"}
    assert {statuses[rollup_id] for rollup_id in untouched_ids} == {"ready"}


def test_rollup_unique_constraint_is_kind_start_scope(rollup_store):
    rollup_store.connection.execute(
        """
        INSERT INTO lcm_rollups(period_kind, period_start, scope)
        VALUES('day', '2026-07-15', 'global')
        """
    )
    rollup_store.connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        rollup_store.connection.execute(
            """
            INSERT INTO lcm_rollups(period_kind, period_start, scope)
            VALUES('day', '2026-07-15', 'global')
            """
        )
    rollup_store.connection.rollback()

    rollup_store.connection.execute(
        """
        INSERT INTO lcm_rollups(period_kind, period_start, scope)
        VALUES('week', '2026-07-15', 'global')
        """
    )
    rollup_store.connection.commit()


def test_cursor_round_trip(rollup_store):
    assert rollup_store.get_cursor("day") is None

    rollup_store.set_cursor("day", "2026-07-15", built_at="2026-07-16T00:00:00Z")
    assert rollup_store.get_cursor("day") == "2026-07-15"
    state = rollup_store.connection.execute(
        "SELECT last_build_cursor, last_built_at FROM lcm_rollup_state WHERE period_kind = 'day'"
    ).fetchone()
    assert tuple(state) == ("2026-07-15", "2026-07-16T00:00:00Z")

    rollup_store.set_cursor("day", "2026-07-16")
    assert rollup_store.get_cursor("day") == "2026-07-16"


def test_purge_rollups_for_sources_removes_rollups_and_source_rows(rollup_store):
    first = _ready_rollup(
        rollup_store, "day", "2026-07-15", source_node_ids=[1, 2]
    )
    second = _ready_rollup(
        rollup_store, "day", "2026-07-16", source_node_ids=[2, 3]
    )
    kept = _ready_rollup(
        rollup_store, "day", "2026-07-17", source_node_ids=[4]
    )

    assert rollup_store.purge_rollups_for_sources([2, 2]) == 2
    remaining_rollups = rollup_store.connection.execute(
        "SELECT rollup_id FROM lcm_rollups ORDER BY rollup_id"
    ).fetchall()
    remaining_sources = rollup_store.connection.execute(
        "SELECT rollup_id, node_id FROM lcm_rollup_sources"
    ).fetchall()
    assert [tuple(row) for row in remaining_rollups] == [(kept,)]
    assert [tuple(row) for row in remaining_sources] == [(kept, 4)]
    assert rollup_store.get_rollup("day", "2026-07-15", "global") is None
    assert rollup_store.get_rollup("day", "2026-07-16", "global") is None
    assert {first, second}.isdisjoint({kept})
    assert rollup_store.purge_rollups_for_sources([]) == 0
    assert rollup_store.purge_rollups_for_sources([999]) == 0


def test_temporal_rollup_config_defaults_are_inert():
    config = LCMConfig()

    assert config.temporal_rollups_enabled is False
    assert config.rollup_daily_target_tokens == 5_000
    assert config.rollup_daily_max_tokens == 15_000
    assert config.rollup_aggregate_max_tokens == 20_000


def test_temporal_rollup_config_reads_environment(monkeypatch):
    monkeypatch.setenv("LCM_TEMPORAL_ROLLUPS_ENABLED", "true")
    monkeypatch.setenv("LCM_ROLLUP_DAILY_TARGET_TOKENS", "6000")
    monkeypatch.setenv("LCM_ROLLUP_DAILY_MAX_TOKENS", "16000")
    monkeypatch.setenv("LCM_ROLLUP_AGGREGATE_MAX_TOKENS", "21000")

    config = LCMConfig.from_env()

    assert config.temporal_rollups_enabled is True
    assert config.rollup_daily_target_tokens == 6_000
    assert config.rollup_daily_max_tokens == 16_000
    assert config.rollup_aggregate_max_tokens == 21_000
