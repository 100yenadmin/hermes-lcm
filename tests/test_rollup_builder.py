from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.rollup_builder as builder_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.rollup_builder import (
    _PENDING_ROLLUPS_SQL,
    build_day,
    build_month,
    build_week,
    run_rollup_maintenance,
)
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.tokens import count_tokens


@pytest.fixture
def rollup_parts(tmp_path):
    db_path = tmp_path / "rollup-builder.db"
    dag = SummaryDAG(db_path)
    store = RollupStore(db_path)
    config = LCMConfig(
        database_path=str(db_path),
        rollup_daily_target_tokens=12,
        rollup_daily_max_tokens=20,
        rollup_aggregate_max_tokens=30,
    )
    try:
        yield store, dag, config
    finally:
        store.close()
        dag.close()


def _timestamp(day: date, hour: int = 12) -> float:
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp()


def _add_node(
    dag: SummaryDAG,
    scope: str,
    day: date,
    summary: str,
    *,
    depth: int = 0,
    latest_day: date | None = None,
) -> int:
    latest = latest_day or day
    return dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=depth,
            summary=summary,
            token_count=count_tokens(summary),
            source_token_count=count_tokens(summary) * 2,
            source_ids=[depth + 1],
            source_type="messages" if depth == 0 else "nodes",
            created_at=_timestamp(day, 18),
            earliest_at=_timestamp(day, 8),
            latest_at=_timestamp(latest, 22),
        )
    )


def _ready(
    store: RollupStore,
    kind: str,
    start: str,
    scope: str,
    *,
    summary: str,
    source_ids: list[int],
    fingerprint: str,
) -> int:
    token = store.upsert_building(kind, start, scope)
    store.mark_ready(
        token,
        summary,
        count_tokens(summary),
        source_ids,
        fingerprint,
    )
    return token.rollup_id


def test_build_day_uses_newest_source_day_and_mocked_summarizer(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-a"
    target_day = date(2026, 7, 15)
    first = _add_node(dag, scope, target_day, "leaf fixture summary")
    second = _add_node(
        dag,
        scope,
        target_day - timedelta(days=1),
        "condensed fixture summary",
        depth=1,
        latest_day=target_day,
    )
    _add_node(dag, scope, target_day - timedelta(days=1), "older summary")
    calls = []

    def summarize(text, **kwargs):
        calls.append((text, kwargs))
        return "deterministic daily rollup", 1

    result = build_day(store, dag, config, scope, target_day, summarizer=summarize)

    assert result is not None
    assert result["summary"] == "deterministic daily rollup"
    assert result["token_count"] == count_tokens("deterministic daily rollup")
    assert result["source_node_ids"] == [first, second]
    assert "leaf fixture summary" in calls[0][0]
    assert "condensed fixture summary" in calls[0][0]
    assert "older summary" not in calls[0][0]
    assert calls[0][1]["token_budget"] == config.rollup_daily_target_tokens
    assert calls[0][1]["l3_truncate_tokens"] == config.rollup_daily_max_tokens


def test_build_day_honors_target_and_hard_cap_after_oversize_result(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-budget"
    target_day = date(2026, 7, 15)
    config.rollup_daily_target_tokens = 8
    config.rollup_daily_max_tokens = 10
    _add_node(dag, scope, target_day, "source material " * 100)
    calls = []

    def summarize(text, **kwargs):
        calls.append((text, kwargs))
        if len(calls) == 1:
            return "oversize " * 30, 1
        return "bounded fallback", 3

    result = build_day(store, dag, config, scope, target_day, summarizer=summarize)

    assert result is not None
    assert result["summary"] == "bounded fallback"
    assert result["token_count"] <= config.rollup_daily_max_tokens
    assert len(calls) == 2
    assert all(call[1]["token_budget"] == config.rollup_daily_target_tokens for call in calls)
    assert all(call[1]["l3_truncate_tokens"] == config.rollup_daily_max_tokens for call in calls)


def test_week_requires_all_content_dailies_ready_before_publishing(rollup_parts):
    # Both Monday and Tuesday have summary-node content, so a week must not
    # publish while Tuesday's daily is stale; it publishes only once every
    # content day has a ready daily (maintainer #388 blocker 5).
    store, dag, config = rollup_parts
    scope = "session-aggregate"
    monday = date(2026, 7, 13)
    tuesday = monday + timedelta(days=1)
    monday_node = _add_node(dag, scope, monday, "monday node")
    tuesday_node = _add_node(dag, scope, tuesday, "tuesday node")
    _ready(
        store, "day", monday.isoformat(), scope,
        summary="ready monday daily", source_ids=[monday_node], fingerprint="monday-v1",
    )
    _ready(
        store, "day", tuesday.isoformat(), scope,
        summary="stale tuesday daily", source_ids=[tuesday_node], fingerprint="tuesday-v1",
    )
    store.mark_stale_for_day(tuesday, scope)
    seen_text = []

    def summarize(text, **_kwargs):
        seen_text.append(text)
        return f"aggregate version {len(seen_text)}", 1

    # Tuesday is a content day but not ready -> the week is left incomplete.
    blocked = build_week(store, dag, config, scope, monday, summarizer=summarize)
    assert blocked is None
    assert seen_text == []
    incomplete_row = store.get_rollup("week", monday.isoformat(), scope)
    assert incomplete_row["status"] == "stale"
    assert "incomplete" in (incomplete_row["error"] or "")

    # Rebuild Tuesday's daily; now every content day is ready and the week builds.
    rebuilt_daily = build_day(
        store, dag, config, scope, tuesday,
        summarizer=lambda _text, **_kwargs: ("ready tuesday rebuilt", 1),
    )
    assert rebuilt_daily is not None
    built = build_week(store, dag, config, scope, monday, summarizer=summarize)

    assert built is not None
    assert "ready monday daily" in seen_text[0]
    assert "ready tuesday rebuilt" in seen_text[0]
    assert built["source_node_ids"] == [monday_node, tuesday_node]
    assert built["status"] == "ready"


def test_rebuilding_a_daily_stales_its_containing_week_and_month(rollup_parts):
    # A daily rebuild must invalidate any already-published week/month so they
    # never stay ready against an outdated day (maintainer #388 blocker 5).
    store, dag, config = rollup_parts
    scope = "session-cascade"
    monday = date(2026, 7, 13)
    monday_node = _add_node(dag, scope, monday, "monday node")
    _ready(
        store, "week", monday.isoformat(), scope,
        summary="published week", source_ids=[monday_node], fingerprint="week-v1",
    )
    _ready(
        store, "month", date(2026, 7, 1).isoformat(), scope,
        summary="published month", source_ids=[monday_node], fingerprint="month-v1",
    )

    rebuilt = build_day(
        store, dag, config, scope, monday,
        summarizer=lambda _text, **_kwargs: ("fresh monday daily", 1),
    )
    assert rebuilt is not None
    assert store.get_rollup("week", monday.isoformat(), scope)["status"] == "stale"
    assert store.get_rollup("month", date(2026, 7, 1).isoformat(), scope)["status"] == "stale"


def test_month_aggregate_never_queries_dag_when_ready_dailies_exist(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-month"
    month_start = date(2026, 7, 1)
    _ready(
        store,
        "day",
        month_start.isoformat(),
        scope,
        summary="first daily",
        source_ids=[101],
        fingerprint="day-one",
    )
    monkeypatch.setattr(
        dag,
        "get_session_nodes",
        lambda *_args, **_kwargs: pytest.fail("aggregate queried DAG nodes"),
    )

    result = build_month(
        store,
        dag,
        config,
        scope,
        month_start,
        summarizer=lambda _text, **_kwargs: ("monthly", 1),
    )

    assert result is not None
    assert result["source_node_ids"] == [101]


def test_builder_failure_is_marked_failed_and_does_not_raise(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-failure"
    target_day = date(2026, 7, 15)
    _add_node(dag, scope, target_day, "source summary")

    def fail(_text, **_kwargs):
        raise RuntimeError("summarizer unavailable")

    result = build_day(store, dag, config, scope, target_day, summarizer=fail)

    assert result is None
    failed = store.get_rollup("day", target_day.isoformat(), scope)
    assert failed is not None
    assert failed["status"] == "failed"
    assert "summarizer unavailable" in failed["error"]


def test_empty_builds_have_zero_store_side_effects(rollup_parts):
    store, dag, config = rollup_parts

    assert build_day(store, dag, config, "empty", date(2026, 7, 15)) is None
    assert build_week(store, dag, config, "empty", date(2026, 7, 13)) is None
    assert store.connection.execute("SELECT COUNT(*) FROM lcm_rollups").fetchone()[0] == 0


def test_publication_staleness_and_bounded_bind_maintenance(tmp_path, monkeypatch):
    # Raw ingest ALONE must not stale rollups (maintainer #388 P1): a period is
    # staled only when a covering summary node is PUBLISHED. Then bind-time
    # maintenance rebuilds up to rollup_builds_per_pass targets, leaving the rest
    # durably stale for the next pass.
    db_path = tmp_path / "engine-rollups.db"
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        rollup_builds_per_pass=2,
    )
    engine = LCMEngine(config=config)
    scope = "temporal-session"
    today = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    try:
        engine.on_session_start(scope, conversation_id="temporal-conversation")
        node_id = _add_node(engine._dag, scope, today, "today source node")
        store = RollupStore(db_path)
        try:
            for kind, start in (
                ("day", today),
                ("week", week_start),
                ("month", month_start),
            ):
                _ready(
                    store,
                    kind,
                    start.isoformat(),
                    scope,
                    summary=f"old {kind}",
                    source_ids=[node_id],
                    fingerprint=f"old-{kind}",
                )

            # Raw ingest does not stale rollups: no covering summary was published.
            engine.ingest([{"role": "user", "content": "raw ingest alone must not stale"}])
            assert [
                store.get_rollup(kind, start.isoformat(), scope)["status"]
                for kind, start in (("day", today), ("week", week_start), ("month", month_start))
            ] == ["ready", "ready", "ready"]

            # Publishing a summary covering today is the staleness signal.
            engine._invalidate_rollups_for_published_node(engine._dag.get_node(node_id))
            assert [
                store.get_rollup(kind, start.isoformat(), scope)["status"]
                for kind, start in (("day", today), ("week", week_start), ("month", month_start))
            ] == ["stale", "stale", "stale"]

            monkeypatch.setattr(
                builder_module,
                "summarize_with_escalation",
                lambda _text, **_kwargs: ("rebuilt rollup", 1),
            )
            engine._bind_lifecycle_state(scope, conversation_id="temporal-conversation")

            statuses = [
                store.get_rollup(kind, start.isoformat(), scope)["status"]
                for kind, start in (("day", today), ("week", week_start), ("month", month_start))
            ]
            assert statuses.count("ready") == 2
            assert statuses.count("stale") == 1
            assert statuses[0] == "ready"
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_never_built_stale_day_is_automatically_built(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-never-built"
    target_day = date(2026, 7, 15)
    _add_node(dag, scope, target_day, "source for first automatic build")
    store.mark_stale_for_day(target_day, scope)
    config.rollup_builds_per_pass = 1
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("first daily rollup", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "ready"


def test_failed_rollup_retry_honors_backoff(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-retry"
    old_day = date(2026, 7, 14)
    recent_day = date(2026, 7, 15)
    _add_node(dag, scope, old_day, "old failed source")
    _add_node(dag, scope, recent_day, "recent failed source")
    old_id = store.upsert_building("day", old_day.isoformat(), scope).rollup_id
    recent_id = store.upsert_building("day", recent_day.isoformat(), scope).rollup_id
    store.mark_failed(old_id, "old failure")
    store.mark_failed(recent_id, "recent failure")
    store.connection.execute(
        "UPDATE lcm_rollups SET built_at = ? WHERE rollup_id = ?",
        ("2026-07-15T00:00:00+00:00", old_id),
    )
    store.connection.commit()
    config.rollup_builds_per_pass = 5
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("retried daily", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    assert store.get_rollup("day", old_day.isoformat(), scope)["status"] == "ready"
    assert store.get_rollup("day", recent_day.isoformat(), scope)["status"] == "failed"


def test_maintenance_budget_stops_before_starting_next_build(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-budget-stop"
    for offset in range(2):
        target_day = date(2026, 7, 14) + timedelta(days=offset)
        _add_node(dag, scope, target_day, f"source {offset}")
        store.mark_stale_for_day(target_day, scope)
    config.rollup_builds_per_pass = 2
    config.rollup_maintenance_budget_ms = 5
    times = iter((0.0, 0.001, 0.006))
    monkeypatch.setattr(builder_module, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("budgeted daily", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    statuses = [
        store.get_rollup("day", (date(2026, 7, 14) + timedelta(days=offset)).isoformat(), scope)["status"]
        for offset in range(2)
    ]
    assert statuses == ["ready", "stale"]


def test_pending_maintenance_query_uses_partial_index(rollup_parts):
    store, _dag, _config = rollup_parts
    store.mark_stale_for_day("2026-07-15", "query-plan")

    plan = store.connection.execute(
        "EXPLAIN QUERY PLAN " + _PENDING_ROLLUPS_SQL,
        ("query-plan", "2026-07-15T00:00:00+00:00", 2),
    ).fetchall()

    assert any("idx_lcm_rollups_pending" in str(row[3]) for row in plan)


def test_session_reset_stales_rollups_referencing_deleted_nodes(tmp_path):
    db_path = tmp_path / "reset-rollups.db"
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        new_session_retain_depth=0,
    )
    engine = LCMEngine(config=config)
    scope = "reset-session"
    try:
        engine.on_session_start(scope, conversation_id="reset-conversation")
        node_id = _add_node(engine._dag, scope, date(2026, 7, 15), "deleted source")
        store = RollupStore(db_path)
        try:
            _ready(
                store,
                "day",
                "2026-07-15",
                scope,
                summary="summary referencing deleted node",
                source_ids=[node_id],
                fingerprint="deleted-node",
            )

            engine.on_session_reset()

            assert engine._dag.get_session_nodes(scope) == []
            assert store.get_rollup("day", "2026-07-15", scope)["status"] == "stale"
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_flag_off_skips_rollup_maintenance(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "flag-off.db"),
        temporal_rollups_enabled=False,
    )
    calls = []
    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        lambda *_args, **_kwargs: calls.append("maintenance"),
    )
    engine = LCMEngine(config=config)
    try:
        engine.on_session_start("flag-off-session", conversation_id="flag-off-conversation")
        engine.ingest([{"role": "user", "content": "stored without rollup queries"}])
        engine._bind_lifecycle_state("flag-off-session", conversation_id="flag-off-conversation")
        assert calls == []
    finally:
        engine.shutdown()


def test_publishing_summary_for_ready_day_marks_it_stale(rollup_parts):
    # Publication of a summary covering an already-ready day is the load-bearing
    # staleness signal (maintainer #388 blocker 1): the day (and its week/month)
    # go stale so a later summary cannot leave an older rollup apparently current.
    store, dag, config = rollup_parts
    scope = "session-publish"
    target_day = date(2026, 7, 15)
    node_id = _add_node(dag, scope, target_day, "already summarized")
    for kind, start in (("day", target_day), ("week", date(2026, 7, 13)), ("month", date(2026, 7, 1))):
        _ready(
            store, kind, start.isoformat(), scope,
            summary=f"ready {kind}", source_ids=[node_id], fingerprint=f"{kind}-v1",
        )

    later_node = _add_node(dag, scope, target_day, "a newer summary covering the same day")
    latest_at = _timestamp(target_day, 22)
    from hermes_lcm.rollup_builder import mark_stale_for_published_summary

    assert mark_stale_for_published_summary(dag, scope, latest_at, latest_at) == 3
    assert later_node  # published node exists
    assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "stale"
    assert store.get_rollup("week", "2026-07-13", scope)["status"] == "stale"
    assert store.get_rollup("month", "2026-07-01", scope)["status"] == "stale"


def test_engine_invalidates_rollups_when_a_node_is_published(tmp_path):
    db_path = tmp_path / "publish-hook.db"
    config = LCMConfig(database_path=str(db_path), temporal_rollups_enabled=True)
    engine = LCMEngine(config=config)
    scope = "publish-hook-session"
    target_day = date(2026, 7, 15)
    try:
        engine.on_session_start(scope, conversation_id="publish-hook-conversation")
        node_id = _add_node(engine._dag, scope, target_day, "published node")
        store = RollupStore(db_path)
        try:
            _ready(
                store, "day", target_day.isoformat(), scope,
                summary="ready day", source_ids=[node_id], fingerprint="day-v1",
            )
            node = engine._dag.get_node(node_id)
            engine._invalidate_rollups_for_published_node(node)
            assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "stale"
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_maintenance_reclaims_crashed_building_row_and_rebuilds(rollup_parts, monkeypatch):
    # A build that crashed leaves a 'building' row forever; maintenance reclaims
    # it once its lease has expired and rebuilds it (maintainer #388 blocker 2).
    store, dag, config = rollup_parts
    scope = "session-reclaim"
    target_day = date(2026, 7, 15)
    _add_node(dag, scope, target_day, "source for reclaimed build")
    store.upsert_building("day", target_day.isoformat(), scope)
    store.connection.execute(
        "UPDATE lcm_rollups SET lease_expires_at = ? WHERE period_kind = 'day'",
        ("2000-01-01T00:00:00+00:00",),
    )
    store.connection.commit()
    config.rollup_builds_per_pass = 1
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("reclaimed rebuild", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "ready"


def test_rollup_builds_per_pass_config_default_and_environment(monkeypatch):
    assert LCMConfig().rollup_builds_per_pass == 2
    assert LCMConfig().rollup_maintenance_budget_ms == 5_000

    monkeypatch.setenv("LCM_ROLLUP_BUILDS_PER_PASS", "5")
    monkeypatch.setenv("LCM_ROLLUP_MAINTENANCE_BUDGET_MS", "750")

    assert LCMConfig.from_env().rollup_builds_per_pass == 5
    assert LCMConfig.from_env().rollup_maintenance_budget_ms == 750


# --- FIXSPEC3 generation-model + staleness + dedup + scope additions -----------


def test_build_day_captures_token_before_reading_sources(rollup_parts, monkeypatch):
    # The build lease must be claimed BEFORE the source snapshot is read, so an
    # invalidation between snapshot and claim cannot escape the generation CAS
    # (maintainer #388 capture-token-first).
    store, dag, config = rollup_parts
    scope = "session-order"
    day = date(2026, 7, 15)
    _add_node(dag, scope, day, "ordered content")
    order: list[str] = []
    real_claim = store.upsert_building
    real_sources = builder_module._daily_sources

    def spy_claim(*args, **kwargs):
        order.append("claim")
        return real_claim(*args, **kwargs)

    def spy_sources(*args, **kwargs):
        order.append("sources")
        return real_sources(*args, **kwargs)

    monkeypatch.setattr(store, "upsert_building", spy_claim)
    monkeypatch.setattr(builder_module, "_daily_sources", spy_sources)

    build_day(store, dag, config, scope, day, summarizer=lambda _t, **_k: ("daily", 1))
    assert order[:2] == ["claim", "sources"]


def test_build_day_supersedes_invalidation_arriving_during_summarize(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-race"
    day = date(2026, 7, 15)
    _add_node(dag, scope, day, "racy content")

    def summarize(_text, **_kwargs):
        # An invalidation lands mid-build; the pre-invalidation token must not
        # publish stale content over it.
        store.mark_stale_for_day(day, scope)
        return "would-be daily", 1

    build_day(store, dag, config, scope, day, summarizer=summarize)
    row = store.get_rollup("day", day.isoformat(), scope)
    assert row["status"] == "stale"
    assert row["summary"] is None


def test_deletion_staleness_bumps_generation_and_supersedes_inflight(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-del"
    day = date(2026, 7, 15)
    node_id = _add_node(dag, scope, day, "to be deleted")
    token = store.upsert_building("day", day.isoformat(), scope)
    store.connection.execute(
        "INSERT INTO lcm_rollup_sources(rollup_id, node_id) VALUES(?, ?)",
        (token.rollup_id, node_id),
    )
    store.connection.commit()

    assert builder_module.mark_stale_for_deleted_nodes(dag, [node_id]) == 1
    # The in-flight build cannot publish deleted-node content over the stale row.
    assert store.mark_ready(token, "deleted content", 1, [node_id], "fp") is False
    row = store.get_rollup("day", day.isoformat(), scope)
    assert row["status"] == "stale"
    assert row["generation"] == token.generation + 1


def test_no_source_stale_day_is_resolved_not_left_lingering(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-nosource"
    day = date(2026, 7, 15)
    # A stale day whose only sources were deleted: no summary node remains.
    store.mark_stale_for_day(day, scope)

    assert build_day(store, dag, config, scope, day) is None
    # The day row is cleared, not left stale forever consuming a build slot.
    assert store.get_rollup("day", day.isoformat(), scope) is None


def test_daily_sources_excludes_condensed_children_present_same_day(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-dedup"
    day = date(2026, 7, 15)
    child = _add_node(dag, scope, day, "child leaf summary")
    parent = dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=1,
            summary="parent condensed summary",
            token_count=count_tokens("parent condensed summary"),
            source_token_count=10,
            source_ids=[child],
            source_type="nodes",
            created_at=_timestamp(day, 18),
            earliest_at=_timestamp(day, 8),
            latest_at=_timestamp(day, 22),
        )
    )
    captured: dict[str, str] = {}

    def summarize(text, **_kwargs):
        captured["text"] = text
        return "daily", 1

    result = build_day(store, dag, config, scope, day, summarizer=summarize)
    assert result["source_node_ids"] == [parent]
    assert "parent condensed summary" in captured["text"]
    assert "child leaf summary" not in captured["text"]


def test_raw_ingest_does_not_prebuild_then_publication_drives_stale(tmp_path, monkeypatch):
    # Item 6 P1: raw ingest must not build/omit a rollup before its summary
    # exists; publication of the covering leaf is the sole staleness signal.
    db_path = tmp_path / "p1.db"
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        rollup_builds_per_pass=4,
    )
    engine = LCMEngine(config=config)
    scope = "p1-session"
    day = datetime.now(timezone.utc).date()
    try:
        engine.on_session_start(scope, conversation_id="p1-conv")
        store = RollupStore(db_path)
        try:
            engine.ingest([{"role": "user", "content": "raw only, no summary yet"}])
            engine._bind_lifecycle_state(scope, conversation_id="p1-conv")
            assert store.get_rollup("day", day.isoformat(), scope) is None

            node_id = _add_node(engine._dag, scope, day, "published leaf summary")
            monkeypatch.setattr(
                builder_module,
                "summarize_with_escalation",
                lambda _t, **_k: ("rebuilt with leaf", 1),
            )
            engine._invalidate_rollups_for_published_node(engine._dag.get_node(node_id))
            assert store.get_rollup("day", day.isoformat(), scope)["status"] == "stale"

            engine._bind_lifecycle_state(scope, conversation_id="p1-conv")
            built = store.get_rollup("day", day.isoformat(), scope)
            assert built["status"] == "ready"
            assert node_id in built["source_node_ids"]
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_bypassed_session_skips_rollup_maintenance(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "bypass.db"),
        temporal_rollups_enabled=True,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        lambda *_args, **_kwargs: calls.append("maintenance"),
    )
    engine = LCMEngine(config=config)
    try:
        engine.on_session_start("bypass-session", conversation_id="bypass-conv")
        calls.clear()

        engine._session_stateless = True
        engine._bind_lifecycle_state("bypass-session", conversation_id="bypass-conv")
        assert calls == []

        engine._session_stateless = False
        engine._session_ignored = True
        engine._bind_lifecycle_state("bypass-session", conversation_id="bypass-conv")
        assert calls == []

        engine._session_ignored = False
        engine._bind_lifecycle_state("bypass-session", conversation_id="bypass-conv")
        assert calls == ["maintenance"]
    finally:
        engine.shutdown()


def test_deleted_node_staleness_covers_more_than_get_session_nodes_limit(rollup_parts):
    # get_session_nodes caps at 1000; the unbounded id capture must return every
    # deleted node so rollups past the cap are still staled (maintainer #388).
    store, dag, _config = rollup_parts
    scope = "session-over-1000"
    for i in range(1001):
        _add_node(dag, scope, date(2026, 7, 15), f"node {i}")

    assert len(dag.get_session_nodes(scope)) == 1000
    all_ids = dag.get_session_node_ids_below_depth(scope, None)
    assert len(all_ids) == 1001
    # Depth filtering also returns the complete set unbounded (all are depth 0).
    assert len(dag.get_session_node_ids_below_depth(scope, 1)) == 1001
