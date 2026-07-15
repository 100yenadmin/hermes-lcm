from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.rollup_builder as builder_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.rollup_builder import build_day, build_month, build_week
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
    rollup_id = store.upsert_building(kind, start, scope)
    store.mark_ready(
        rollup_id,
        summary,
        count_tokens(summary),
        source_ids,
        fingerprint,
    )
    return rollup_id


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


def test_aggregates_use_only_ready_dailies_and_refingerprint_when_stale_daily_recovers(
    rollup_parts,
):
    store, dag, config = rollup_parts
    scope = "session-aggregate"
    monday = date(2026, 7, 13)
    monday_node = _add_node(dag, scope, monday, "monday node")
    tuesday_node = _add_node(dag, scope, monday + timedelta(days=1), "tuesday node")
    _ready(
        store,
        "day",
        monday.isoformat(),
        scope,
        summary="ready monday daily",
        source_ids=[monday_node],
        fingerprint="monday-v1",
    )
    _ready(
        store,
        "day",
        (monday + timedelta(days=1)).isoformat(),
        scope,
        summary="stale tuesday daily",
        source_ids=[tuesday_node],
        fingerprint="tuesday-v1",
    )
    store.mark_stale_for_day(monday + timedelta(days=1), scope)
    seen_text = []

    def summarize(text, **_kwargs):
        seen_text.append(text)
        return f"aggregate version {len(seen_text)}", 1

    first = build_week(store, dag, config, scope, monday, summarizer=summarize)

    assert first is not None
    assert "ready monday daily" in seen_text[0]
    assert "stale tuesday daily" not in seen_text[0]
    assert first["source_node_ids"] == [monday_node]
    first_fingerprint = first["source_fingerprint"]

    rebuilt_daily = build_day(
        store,
        dag,
        config,
        scope,
        monday + timedelta(days=1),
        summarizer=lambda _text, **_kwargs: ("ready tuesday rebuilt", 1),
    )
    assert rebuilt_daily is not None
    second = build_week(store, dag, config, scope, monday, summarizer=summarize)

    assert second is not None
    assert "ready tuesday rebuilt" in seen_text[1]
    assert second["source_node_ids"] == [monday_node, tuesday_node]
    assert second["source_fingerprint"] != first_fingerprint


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


def test_ingest_staleness_cascade_and_bounded_bind_maintenance(tmp_path, monkeypatch):
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

            engine.ingest([{"role": "user", "content": "new message makes rollups stale"}])
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


def test_flag_off_skips_both_engine_hook_helpers(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "flag-off.db"),
        temporal_rollups_enabled=False,
    )
    calls = []
    monkeypatch.setattr(
        engine_module,
        "mark_stale_after_ingest",
        lambda *_args, **_kwargs: calls.append("ingest"),
    )
    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        lambda *_args, **_kwargs: calls.append("maintenance"),
    )
    engine = LCMEngine(config=config)
    try:
        engine.on_session_start("flag-off-session", conversation_id="flag-off-conversation")
        engine.ingest([{"role": "user", "content": "stored without rollup queries"}])
        assert calls == []
    finally:
        engine.shutdown()


def test_rollup_builds_per_pass_config_default_and_environment(monkeypatch):
    assert LCMConfig().rollup_builds_per_pass == 2

    monkeypatch.setenv("LCM_ROLLUP_BUILDS_PER_PASS", "5")

    assert LCMConfig.from_env().rollup_builds_per_pass == 5
