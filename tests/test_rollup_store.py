from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.rollup_store import RollupBuildToken, RollupStore


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
    token = store.upsert_building(period_kind, period_start, scope)
    store.mark_ready(
        token,
        f"{period_kind} summary for {period_start}",
        42,
        source_node_ids or [],
        f"fingerprint-{period_kind}-{period_start}-{scope}",
    )
    return token.rollup_id


def test_core_migrations_do_not_create_rollup_tables_or_bump_schema(tmp_path):
    # The opt-in rollup tables are NOT part of the core numeric schema: a base
    # (feature-off) startup must leave schema_version at SCHEMA_VERSION and create
    # no lcm_rollups* tables, so a base build keeps opening the DB.
    conn = sqlite3.connect(tmp_path / "core.db")
    try:
        db_bootstrap.run_versioned_migrations(conn)
        db_bootstrap.run_versioned_migrations(conn)
        conn.commit()

        assert db_bootstrap.get_schema_version(conn) == db_bootstrap.SCHEMA_VERSION
        assert db_bootstrap.SCHEMA_VERSION == 5
        assert ROLLUP_TABLES.isdisjoint(_table_names(conn))
        # No numeric v6 step is recorded; the rollup tables use a named step.
        steps = conn.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name LIKE 'v6%' OR step_name = 'temporal_rollups_v1'"
        ).fetchall()
        assert steps == []
    finally:
        conn.close()


def test_rollup_store_lazily_creates_tables_without_bumping_schema(tmp_path):
    # A DB already at the core schema version (feature previously off) gains the
    # rollup tables + the NAMED migration step when RollupStore opens it, while
    # schema_version stays put so a base build still opens it.
    db_path = tmp_path / "enable.db"
    conn = sqlite3.connect(db_path)
    db_bootstrap.run_versioned_migrations(conn)
    conn.commit()
    conn.close()
    assert db_bootstrap.get_schema_version(sqlite3.connect(db_path)) == db_bootstrap.SCHEMA_VERSION

    store = RollupStore(db_path)
    try:
        assert ROLLUP_TABLES <= _table_names(store.connection)
        assert db_bootstrap.get_schema_version(store.connection) == db_bootstrap.SCHEMA_VERSION
        completed = store.connection.execute(
            "SELECT completed_at FROM lcm_migration_state WHERE step_name = 'temporal_rollups_v1'"
        ).fetchone()
        assert completed is not None
        index_sql = store.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_lcm_rollups_ready_period'"
        ).fetchone()[0]
        assert "WHERE status = 'ready'" in index_sql
        pending_index_sql = store.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_lcm_rollups_pending'"
        ).fetchone()[0]
        assert "WHERE status IN ('stale', 'failed')" in pending_index_sql
    finally:
        store.close()


def test_base_build_opens_enabled_rollup_db_without_raising(tmp_path, monkeypatch):
    # After the feature is enabled (rollup tables present, schema still at base),
    # a simulated base build (SCHEMA_VERSION unchanged) must open the DB and its
    # own migrations must not raise SchemaVersionTooNewError.
    db_path = tmp_path / "enabled.db"
    store = RollupStore(db_path)
    store.close()

    reopen = sqlite3.connect(db_path)
    try:
        db_bootstrap.refuse_schema_version_too_new(reopen)
        db_bootstrap.run_versioned_migrations(reopen)
        assert db_bootstrap.get_schema_version(reopen) == db_bootstrap.SCHEMA_VERSION
    finally:
        reopen.close()


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
    token = rollup_store.upsert_building(
        "day", "2026-07-15", "conversation:conv-1"
    )
    rollup_id = token.rollup_id
    initial = rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert initial.pop("built_at") is not None
    assert initial == {
        "rollup_id": rollup_id,
        "period_kind": "day",
        "period_start": "2026-07-15",
        "scope": "conversation:conv-1",
        "summary": None,
        "token_count": None,
        "status": "building",
        "source_fingerprint": None,
        "error": None,
        "generation": 0,
        "source_node_ids": [],
    }

    assert rollup_store.mark_ready(
        token,
        "Daily summary",
        123,
        [9, 3, 9],
        "fingerprint-1",
    ) is True
    ready = rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert ready["summary"] == "Daily summary"
    assert ready["token_count"] == 123
    assert ready["status"] == "ready"
    assert ready["built_at"] is not None
    assert ready["source_fingerprint"] == "fingerprint-1"
    assert ready["source_node_ids"] == [3, 9]

    rebuilt = rollup_store.upsert_building(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert rebuilt.rollup_id == rollup_id
    rebuilding = rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert rebuilding["status"] == "building"
    assert rebuilding["summary"] == "Daily summary"
    assert rebuilding["token_count"] == 123
    assert rebuilding["built_at"] is not None
    # The rebuild claim preserves the last-known-good lineage (A2): sources stay
    # queryable until mark_ready swaps them, so a concurrent purge can still find
    # and re-stale the rollup mid-rebuild.
    assert rebuilding["source_node_ids"] == [3, 9]

    rollup_store.mark_failed(rollup_id, "summarizer unavailable")
    failed = rollup_store.get_rollup(
        "day", "2026-07-15", "conversation:conv-1"
    )
    assert failed["status"] == "failed"
    assert failed["error"] == "summarizer unavailable"
    assert failed["summary"] == "Daily summary"
    assert failed["token_count"] == 123


def test_mark_ready_rejects_unknown_rollup_without_orphan_sources(rollup_store):
    with pytest.raises(ValueError, match="unknown rollup_id"):
        rollup_store.mark_ready(RollupBuildToken(999, 0), "missing", 1, [10], "fingerprint")

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
    rollup_store.mark_failed(failed.rollup_id, "failed")

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


def test_mark_stale_for_day_creates_missing_maintenance_rows(rollup_store):
    scope = "conversation:never-built"

    assert rollup_store.mark_stale_for_day("2026-07-15", scope) == 3

    rows = rollup_store.connection.execute(
        """
        SELECT period_kind, period_start, status, summary
        FROM lcm_rollups
        WHERE scope = ?
        ORDER BY period_kind
        """,
        (scope,),
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("day", "2026-07-15", "stale", None),
        ("month", "2026-07-01", "stale", None),
        ("week", "2026-07-13", "stale", None),
    ]


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


def test_purge_rollups_for_sources_restales_affected_windows(rollup_store):
    first = _ready_rollup(
        rollup_store, "day", "2026-07-15", source_node_ids=[1, 2]
    )
    second = _ready_rollup(
        rollup_store, "day", "2026-07-16", source_node_ids=[2, 3]
    )
    kept = _ready_rollup(
        rollup_store, "day", "2026-07-17", source_node_ids=[4]
    )

    # A hard delete at/before the build cursor would orphan the window (cursor
    # advanced, no pending row -> never rebuilt). Purge instead re-stales the
    # affected periods so maintenance rebuilds them from remaining sources.
    assert rollup_store.purge_rollups_for_sources([2, 2]) == 2

    first_row = rollup_store.get_rollup("day", "2026-07-15", "global")
    second_row = rollup_store.get_rollup("day", "2026-07-16", "global")
    kept_row = rollup_store.get_rollup("day", "2026-07-17", "global")
    assert first_row["status"] == "stale" and first_row["summary"] is None
    assert second_row["status"] == "stale" and second_row["summary"] is None
    # Generation advanced so an in-flight build cannot publish purged content.
    assert first_row["generation"] == 1 and second_row["generation"] == 1
    # The rollup with no purged source is left untouched (still ready).
    assert kept_row["status"] == "ready"
    assert kept_row["source_node_ids"] == [4]
    # Affected rollups' source rows are cleared (repopulated on rebuild); only the
    # kept rollup's sources remain.
    remaining_sources = rollup_store.connection.execute(
        "SELECT rollup_id, node_id FROM lcm_rollup_sources ORDER BY rollup_id, node_id"
    ).fetchall()
    assert [tuple(row) for row in remaining_sources] == [(kept, 4)]
    assert {first, second}.isdisjoint({kept})
    assert rollup_store.purge_rollups_for_sources([]) == 0
    assert rollup_store.purge_rollups_for_sources([999]) == 0


def test_invalidation_mid_build_supersedes_old_mark_ready(rollup_store):
    scope = "conversation:conv-1"
    token = rollup_store.upsert_building("day", "2026-07-15", scope)

    # An invalidation arrives while the build is in flight, advancing generation.
    rollup_store.mark_stale_for_day("2026-07-15", scope)

    # The stale builder's late publish is a no-op; the newer stale state stands.
    assert rollup_store.mark_ready(token, "stale build", 5, [1], "fp") is False
    row = rollup_store.get_rollup("day", "2026-07-15", scope)
    assert row["status"] == "stale"
    assert row["summary"] is None
    assert row["generation"] == 1


def test_reclaim_stale_building_reclaims_expired_lease_and_blocks_publish(rollup_store):
    scope = "conversation:conv-1"
    token = rollup_store.upsert_building("day", "2026-07-15", scope)

    # Nothing is reclaimed while the lease is valid.
    assert rollup_store.reclaim_stale_building(now="1999-01-01T00:00:00+00:00") == 0
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "building"

    # A far-future 'now' makes the lease expired: the crashed build is reclaimed.
    assert rollup_store.reclaim_stale_building(now="2999-01-01T00:00:00+00:00") == 1
    reclaimed = rollup_store.get_rollup("day", "2026-07-15", scope)
    assert reclaimed["status"] == "stale"
    assert reclaimed["generation"] == 1

    # The original (crashed) builder returning late cannot publish over it.
    assert rollup_store.mark_ready(token, "zombie", 5, [1], "fp") is False
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "stale"


def test_upsert_stale_seeds_missing_rows_and_leaves_building_alone(rollup_store):
    scope = "conversation:conv-1"

    assert rollup_store.upsert_stale("month", "2026-07-01", scope) == 1
    seeded = rollup_store.get_rollup("month", "2026-07-01", scope)
    assert seeded["status"] == "stale"
    assert seeded["summary"] is None

    # A currently-building row is not disturbed by an upsert_stale.
    rollup_store.upsert_building("day", "2026-07-15", scope)
    rollup_store.upsert_stale("day", "2026-07-15", scope)
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "building"


def test_stale_aggregates_for_day_targets_week_and_month_only(rollup_store):
    scope = "conversation:conv-1"
    day_id = _ready_rollup(rollup_store, "day", "2026-07-15", scope)
    week_id = _ready_rollup(rollup_store, "week", "2026-07-13", scope)
    month_id = _ready_rollup(rollup_store, "month", "2026-07-01", scope)

    assert rollup_store.stale_aggregates_for_day("2026-07-15", scope) == 2

    statuses = dict(
        rollup_store.connection.execute(
            "SELECT rollup_id, status FROM lcm_rollups"
        ).fetchall()
    )
    assert statuses[day_id] == "ready"
    assert statuses[week_id] == "stale"
    assert statuses[month_id] == "stale"


def test_cursor_state_is_per_scope(rollup_store):
    rollup_store.set_cursor("day", "cursor-a", "scope-a")
    rollup_store.set_cursor("day", "cursor-b", "scope-b")

    assert rollup_store.get_cursor("day", "scope-a") == "cursor-a"
    assert rollup_store.get_cursor("day", "scope-b") == "cursor-b"
    assert rollup_store.get_cursor("day", "scope-missing") is None


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


# --- FIXSPEC3 generation/lease-model + index + purge additions -----------------


def test_claim_advances_generation_so_racing_claims_get_distinct_leases(rollup_store):
    # Two builders racing the same stale period must get DISTINCT tokens so only
    # the latest claimant can publish (maintainer #387 blocker 1).
    scope = "conversation:conv-1"
    rollup_store.upsert_stale("day", "2026-07-15", scope)
    first = rollup_store.upsert_building("day", "2026-07-15", scope)
    second = rollup_store.upsert_building("day", "2026-07-15", scope)

    assert second.generation == first.generation + 1
    assert first.generation != second.generation
    # The older claimant is superseded; only the latest lease publishes.
    assert rollup_store.mark_ready(first, "stale build", 1, [1], "fp-old") is False
    assert rollup_store.mark_ready(second, "fresh build", 1, [2], "fp-new") is True
    row = rollup_store.get_rollup("day", "2026-07-15", scope)
    assert row["status"] == "ready"
    assert row["summary"] == "fresh build"
    assert row["source_node_ids"] == [2]


def test_a_fresh_claim_starts_at_generation_zero(rollup_store):
    token = rollup_store.upsert_building("day", "2026-07-15", "scope-x")
    assert token.generation == 0


def test_mark_failed_generation_guard_rejects_superseded_failure(rollup_store):
    # A superseded builder's late exception must not flip a newer stale/ready row
    # to failed (maintainer #387 blocker 2).
    scope = "conversation:conv-1"
    token = rollup_store.upsert_building("day", "2026-07-15", scope)
    rollup_store.mark_stale_for_day("2026-07-15", scope)  # advances generation

    assert rollup_store.mark_failed(token.rollup_id, "late boom", generation=token.generation) is False
    row = rollup_store.get_rollup("day", "2026-07-15", scope)
    assert row["status"] == "stale"
    assert row["error"] is None

    # An unguarded call still writes (used to seed a known failure directly).
    assert rollup_store.mark_failed(token.rollup_id, "seeded") is True
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "failed"


def test_defer_incomplete_is_token_guarded(rollup_store):
    # Releasing an incomplete aggregate back to stale must not erase a newer
    # aggregate published by another builder (maintainer #387 blocker 3).
    scope = "session-agg"
    token = rollup_store.upsert_building("week", "2026-07-13", scope)
    rollup_store.mark_stale_for_day("2026-07-15", scope)  # supersedes the week

    assert rollup_store.defer_incomplete(token, "incomplete: superseded") is False

    token2 = rollup_store.upsert_building("week", "2026-07-13", scope)
    assert rollup_store.defer_incomplete(token2, "incomplete: 2 dailies") is True
    row = rollup_store.get_rollup("week", "2026-07-13", scope)
    assert row["status"] == "stale"
    assert "incomplete: 2 dailies" in row["error"]


def test_resolve_no_source_clears_only_when_owned(rollup_store):
    # A claimed period with no source content is cleared iff still owned, so it
    # cannot linger stale consuming a build slot (maintainer #388 no-source).
    scope = "session-empty"
    token = rollup_store.upsert_building("day", "2026-07-15", scope)
    rollup_store.mark_stale_for_day("2026-07-15", scope)  # supersede

    assert rollup_store.resolve_no_source(token) is False
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "stale"

    token2 = rollup_store.upsert_building("day", "2026-07-15", scope)
    rollup_store.connection.execute(
        "INSERT INTO lcm_rollup_sources(rollup_id, node_id) VALUES(?, ?)",
        (token2.rollup_id, 9),
    )
    rollup_store.connection.commit()

    assert rollup_store.resolve_no_source(token2) is True
    assert rollup_store.get_rollup("day", "2026-07-15", scope) is None
    orphan_sources = rollup_store.connection.execute(
        "SELECT COUNT(*) FROM lcm_rollup_sources WHERE rollup_id = ?",
        (token2.rollup_id,),
    ).fetchone()[0]
    assert orphan_sources == 0


def test_rollup_indexes_are_scope_leading_and_cover_source_node(rollup_store):
    # Reads/stale-scan filter by scope, and purge looks up by node_id
    # (maintainer #387 perf indexes 18/19).
    conn = rollup_store.connection

    def index_columns(name):
        return [row["name"] for row in conn.execute(f"PRAGMA index_info({name})").fetchall()]

    assert index_columns("idx_lcm_rollups_ready_period")[0] == "scope"
    assert index_columns("idx_lcm_rollups_pending")[0] == "scope"
    assert index_columns("idx_lcm_rollup_sources_node") == ["node_id"]

    plan = conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT DISTINCT rollup_id FROM lcm_rollup_sources WHERE node_id IN (1, 2)"
    ).fetchall()
    assert any("idx_lcm_rollup_sources_node" in str(row[3]) for row in plan)


# --- FIXSPEC4 A1/A2/A3 store additions -----------------------------------------


def test_late_mark_failed_cannot_flip_a_published_ready_row(rollup_store):
    # Maintainer #387 A1 repro: mark_ready() succeeds (row -> ready), then a late
    # mark_failed() with the SAME token still flipped row -> failed while keeping
    # the summary, because mark_ready does not advance generation and the guard
    # matched only (rollup_id, generation). The terminal transition must ALSO
    # require status='building': once a row leaves 'building', no token-holder
    # transitions it.
    scope = "conversation:conv-1"
    token = rollup_store.upsert_building("day", "2026-07-15", scope)
    assert rollup_store.mark_ready(token, "published summary", 7, [1], "fp") is True

    assert (
        rollup_store.mark_failed(
            token.rollup_id, "late boom", generation=token.generation
        )
        is False
    )
    row = rollup_store.get_rollup("day", "2026-07-15", scope)
    assert row["status"] == "ready"
    assert row["summary"] == "published summary"


def test_resolve_no_source_cannot_delete_a_published_row_at_same_generation(rollup_store):
    # Same A1 invariant for the token-guarded DELETE: mark_ready leaves the row
    # 'ready' at the captured generation, so a late resolve_no_source with the
    # same token must not delete the published row.
    scope = "conversation:conv-1"
    token = rollup_store.upsert_building("day", "2026-07-15", scope)
    assert rollup_store.mark_ready(token, "published", 3, [1], "fp") is True

    assert rollup_store.resolve_no_source(token) is False
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "ready"


def test_rebuild_claim_preserves_lineage_and_purge_still_restales(rollup_store):
    # Maintainer #387 A2 repro: upsert_building cleared lcm_rollup_sources at claim
    # time, so a concurrent purge-by-node found ZERO affected rollups and did not
    # re-stale, letting the in-flight build publish deleted-node content. The
    # claim must keep the prior lineage queryable until mark_ready swaps it.
    scope = "conversation:conv-1"
    token = rollup_store.upsert_building("day", "2026-07-15", scope)
    assert rollup_store.mark_ready(token, "v1", 3, [10, 11], "fp1") is True

    rebuild = rollup_store.upsert_building("day", "2026-07-15", scope)
    # The last-known-good lineage survives the claim (queryable).
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["source_node_ids"] == [10, 11]

    # A concurrent purge of node 11 still sees the rollup via its retained lineage
    # and re-stales it (advancing generation).
    assert rollup_store.purge_rollups_for_sources([11]) == 1
    restaled = rollup_store.get_rollup("day", "2026-07-15", scope)
    assert restaled["status"] == "stale"

    # The superseded rebuild can no longer publish the now-purged content.
    assert rollup_store.mark_ready(rebuild, "v2 with purged node", 3, [10, 11], "fp2") is False
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "stale"


def test_mark_ready_atomically_swaps_sources_and_status(rollup_store):
    # The sources-replace + status='ready' write is one transaction, so a fresh
    # rebuild atomically supersedes the old lineage (A2 atomic swap).
    scope = "conversation:conv-1"
    first = rollup_store.upsert_building("day", "2026-07-15", scope)
    rollup_store.mark_ready(first, "v1", 3, [1, 2], "fp1")
    second = rollup_store.upsert_building("day", "2026-07-15", scope)
    assert rollup_store.mark_ready(second, "v2", 4, [3, 4], "fp2") is True
    row = rollup_store.get_rollup("day", "2026-07-15", scope)
    assert row["status"] == "ready"
    assert row["summary"] == "v2"
    assert row["source_node_ids"] == [3, 4]


def test_upsert_stale_many_is_atomic_all_or_nothing(rollup_store):
    # Maintainer #391 D2: seeding many targets must be all-or-nothing. A failure
    # on any target (here a period_kind that violates the CHECK constraint) rolls
    # the whole batch back so no target is left half-seeded.
    scope = "conversation:conv-1"
    with pytest.raises(sqlite3.Error):
        rollup_store.upsert_stale_many(
            [
                ("day", "2026-07-15", scope),
                ("bogus", "2026-07-13", scope),  # violates period_kind CHECK
            ]
        )
    assert rollup_store.get_rollup("day", "2026-07-15", scope) is None

    # A clean batch seeds every target and leaves a currently-building row alone.
    rollup_store.upsert_building("month", "2026-07-01", scope)
    assert rollup_store.upsert_stale_many(
        [
            ("day", "2026-07-15", scope),
            ("week", "2026-07-13", scope),
            ("month", "2026-07-01", scope),  # building -> untouched
        ]
    ) == 2
    assert rollup_store.get_rollup("day", "2026-07-15", scope)["status"] == "stale"
    assert rollup_store.get_rollup("week", "2026-07-13", scope)["status"] == "stale"
    assert rollup_store.get_rollup("month", "2026-07-01", scope)["status"] == "building"


def test_init_repairs_dropped_rollup_table_despite_marker(tmp_path):
    # Maintainer #387 A3: the temporal_rollups_v1 marker can outlive its tables
    # (a table dropped from under it). A fresh RollupStore init must not trust the
    # marker; it verifies and repairs the missing table.
    db_path = tmp_path / "marker-repair.db"
    RollupStore(db_path).close()

    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE lcm_rollup_sources")
    marker = conn.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name = 'temporal_rollups_v1'"
    ).fetchone()
    assert marker is not None  # marker present, but a table is now missing
    conn.commit()
    conn.close()

    store = RollupStore(db_path)
    try:
        assert ROLLUP_TABLES <= _table_names(store.connection)
        assert db_bootstrap.verify_temporal_rollup_schema(store.connection) == []
    finally:
        store.close()


def test_verify_temporal_rollup_schema_flags_missing_objects(rollup_store):
    # No marker/tables missing on a healthy store; dropping an index is detected.
    assert db_bootstrap.verify_temporal_rollup_schema(rollup_store.connection) == []
    rollup_store.connection.execute("DROP INDEX idx_lcm_rollups_pending")
    missing = db_bootstrap.verify_temporal_rollup_schema(rollup_store.connection)
    assert "index:idx_lcm_rollups_pending" in missing
