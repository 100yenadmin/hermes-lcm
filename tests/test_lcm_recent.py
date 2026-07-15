from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.rollup_periods import parse_recent_period
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.tokens import count_tokens
from hermes_lcm.tools import lcm_recent


NOW = datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("period", "start", "end", "kind", "subday"),
    [
        ("today", "2026-07-15T00:00:00+00:00", "2026-07-16T00:00:00+00:00", "day", False),
        ("yesterday", "2026-07-14T00:00:00+00:00", "2026-07-15T00:00:00+00:00", "day", False),
        ("7d", "2026-07-09T00:00:00+00:00", "2026-07-16T00:00:00+00:00", "day", False),
        ("week", "2026-07-13T00:00:00+00:00", "2026-07-20T00:00:00+00:00", "week", False),
        ("month", "2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00", "month", False),
        ("date:2026-02-28", "2026-02-28T00:00:00+00:00", "2026-03-01T00:00:00+00:00", "day", False),
        ("last 6h", "2026-07-15T09:30:00+00:00", "2026-07-15T15:30:00+00:00", "day", True),
    ],
)
def test_parse_recent_period_table(period, start, end, kind, subday):
    parsed = parse_recent_period(period, now=NOW)

    assert parsed.start.isoformat() == start
    assert parsed.end.isoformat() == end
    assert parsed.rollup_kind == kind
    assert parsed.subday is subday


@pytest.mark.parametrize(
    "period",
    [
        None,
        "",
        "0d",
        "last 0h",
        "date:2026-02-30",
        "7 days",
        "tomorrow",
        f"{10**30}d",
        f"last {10**30}h",
    ],
)
def test_parse_recent_period_invalid_values_are_clean_errors(period):
    with pytest.raises(ValueError, match="period|day|hour"):
        parse_recent_period(period, now=NOW)


@pytest.fixture
def recent_parts(tmp_path):
    db_path = tmp_path / "recent.db"
    dag = SummaryDAG(db_path)
    store = RollupStore(db_path)
    config = LCMConfig(database_path=str(db_path), temporal_rollups_enabled=True)
    engine = SimpleNamespace(
        _dag=dag,
        _config=config,
        current_session_id="conversation-a",
    )
    try:
        yield engine, store
    finally:
        store.close()
        dag.close()


def _timestamp(day: date, hour: int = 12) -> float:
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp()


def _add_leaf(dag, session_id, day, summary, *, timestamp=None):
    content_time = timestamp if timestamp is not None else _timestamp(day)
    return dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary=summary,
            token_count=count_tokens(summary),
            source_token_count=count_tokens(summary),
            source_ids=[1],
            source_type="messages",
            created_at=content_time,
            earliest_at=content_time,
            latest_at=content_time,
        )
    )


def _ready(store, kind, period_start, scope, summary="ready rollup", token_count=17):
    rollup_id = store.upsert_building(kind, period_start, scope)
    store.mark_ready(rollup_id, summary, token_count, [], "fingerprint")
    return rollup_id


def test_lcm_recent_serves_ready_rollup_with_provenance(recent_parts):
    engine, store = recent_parts
    rollup_id = _ready(store, "day", "2026-07-15", engine.current_session_id)

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["mode"] == "rollup"
    assert result["provenance"] == {
        "fallback": False,
        "rollups": [{"rollup_id": rollup_id, "status": "ready"}],
    }
    assert result["sections"][0]["content"].startswith("Tokens: 17\n")


def test_lcm_recent_stale_rollup_falls_back_to_leaf_summaries(recent_parts):
    engine, store = recent_parts
    _add_leaf(engine._dag, engine.current_session_id, date(2026, 7, 15), "leaf fallback")
    _ready(store, "day", "2026-07-15", engine.current_session_id)
    store.mark_stale_for_day("2026-07-15", engine.current_session_id)

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["mode"] == "leaf_summary_fallback"
    assert result["fallback_reason"] == "rollups_unavailable"
    assert result["provenance"] == {"fallback": True, "rollups": []}
    assert [section["content"] for section in result["sections"]] == ["leaf fallback"]


def test_lcm_recent_disabled_flag_falls_back_even_when_ready(recent_parts):
    engine, store = recent_parts
    engine._config.temporal_rollups_enabled = False
    _add_leaf(engine._dag, engine.current_session_id, date(2026, 7, 15), "flag-off leaf")
    _ready(store, "day", "2026-07-15", engine.current_session_id)

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["fallback_reason"] == "temporal_rollups_disabled"
    assert result["provenance"]["fallback"] is True
    assert result["sections"][0]["kind"] == "leaf_summary"


def test_lcm_recent_subday_window_always_falls_back(recent_parts):
    engine, _store = recent_parts
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    _add_leaf(
        engine._dag,
        engine.current_session_id,
        recent_time.date(),
        "subday leaf",
        timestamp=recent_time.timestamp(),
    )

    result = json.loads(lcm_recent({"period": "last 2h"}, engine=engine))

    assert result["fallback_reason"] == "subday_window"
    assert result["provenance"]["fallback"] is True
    assert result["sections"][0]["content"] == "subday leaf"


def test_lcm_recent_empty_window_is_a_successful_empty_fallback(recent_parts):
    engine, _store = recent_parts

    result = json.loads(lcm_recent({"period": "date:1999-01-01"}, engine=engine))

    assert "error" not in result
    assert result["provenance"]["fallback"] is True
    assert result["sections"] == []
    assert result["returned_sections"] == 0


def test_lcm_recent_limit_order_and_response_char_bound(recent_parts):
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False
    target_day = date(2026, 7, 15)
    _add_leaf(engine._dag, engine.current_session_id, target_day, "older " * 6000, timestamp=_timestamp(target_day, 8))
    newest_id = _add_leaf(
        engine._dag,
        engine.current_session_id,
        target_day,
        "newer " * 6000,
        timestamp=_timestamp(target_day, 20),
    )
    _add_leaf(engine._dag, engine.current_session_id, target_day, "middle", timestamp=_timestamp(target_day, 12))

    raw = lcm_recent({"period": "date:2026-07-15", "limit": 2}, engine=engine)
    result = json.loads(raw)

    assert len(raw) <= 20_000
    assert result["total_sections"] == 2
    assert len(result["sections"]) <= 2
    assert result["sections"][0]["node_id"] == newest_id
    assert result["truncated"] is True


def test_lcm_recent_global_scope_uses_global_rollup_and_reports_clamped_limit(recent_parts):
    engine, store = recent_parts
    rollup_id = _ready(store, "day", "2026-07-15", "global")

    result = json.loads(
        lcm_recent(
            {"period": "date:2026-07-15", "scope": "global", "limit": 500},
            engine=engine,
        )
    )

    assert result["scope"] == "global"
    assert result["limit"] == 200
    assert result["limit_clamped_from"] == 500
    assert result["provenance"]["rollups"][0]["rollup_id"] == rollup_id


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({}, "period is required"),
        ({"period": "today", "scope": "workspace"}, "scope must be one of"),
        ({"period": "today", "limit": 0}, "limit must be a positive integer"),
        ({"period": "today", "limit": True}, "limit must be an integer"),
    ],
)
def test_lcm_recent_argument_validation(recent_parts, args, message):
    engine, _store = recent_parts
    result = json.loads(lcm_recent(args, engine=engine))
    assert message in result["error"]
