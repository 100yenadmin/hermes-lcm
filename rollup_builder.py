"""Flag-gated temporal rollup construction and engine wiring helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from calendar import monthrange
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Callable, Sequence

from .config import LCMConfig
from .dag import SummaryDAG
from .escalation import _deterministic_truncate, summarize_with_escalation
from .rollup_store import RollupStore
from .tokens import count_tokens

logger = logging.getLogger(__name__)

Summarizer = Callable[..., tuple[str, int]]


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _utc_bounds(day: date) -> tuple[float, float]:
    start = datetime.combine(day, datetime_time.min, tzinfo=timezone.utc)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _daily_sources(dag: SummaryDAG, scope: str, day: date) -> list[dict[str, object]]:
    """Return summaries whose newest covered source message falls on ``day``."""
    connection = dag.connection
    if connection is None:
        return []
    start, end = _utc_bounds(day)
    rows = connection.execute(
        """
        SELECT node_id, summary
        FROM summary_nodes
        WHERE session_id = ?
          AND COALESCE(latest_at, created_at) >= ?
          AND COALESCE(latest_at, created_at) < ?
        ORDER BY COALESCE(latest_at, created_at), node_id
        """,
        (scope, start, end),
    ).fetchall()
    return [
        {"node_id": int(row[0]), "summary": str(row[1] or "")}
        for row in rows
    ]


def _summary_controls(config: LCMConfig) -> dict[str, object]:
    return {
        "model": config.summary_model,
        "timeout": config.summary_timeout_ms / 1000.0,
        "l2_budget_ratio": config.l2_budget_ratio,
        "custom_instructions": config.custom_instructions,
        "fallback_models": config.summary_fallback_models,
    }


def _summarize_capped(
    text: str,
    *,
    target_tokens: int,
    max_tokens: int,
    config: LCMConfig,
    summarizer: Summarizer,
    circuit_breaker: object | None,
    spend_guard: object | None,
) -> tuple[str, int]:
    """Use escalation until the result is within the configured hard cap."""
    target = max(1, min(int(target_tokens), int(max_tokens)))
    hard_max = max(1, int(max_tokens))
    candidate = text
    previous_tokens = count_tokens(candidate)

    while True:
        summary, _level = summarizer(
            candidate,
            source_tokens=max(1, previous_tokens),
            token_budget=target,
            l3_truncate_tokens=hard_max,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
            **_summary_controls(config),
        )
        summary = str(summary)
        summary_tokens = count_tokens(summary)
        if summary_tokens <= hard_max:
            return summary, summary_tokens
        if summary_tokens >= previous_tokens:
            truncated = _deterministic_truncate(summary, hard_max)
            return truncated, count_tokens(truncated)
        candidate = summary
        previous_tokens = summary_tokens


def _mark_failed(store: RollupStore, rollup_id: int | None, exc: Exception) -> None:
    if rollup_id is not None:
        try:
            store.mark_failed(rollup_id, f"{type(exc).__name__}: {exc}")
        except Exception:
            logger.debug("LCM temporal rollup failure state could not be persisted", exc_info=True)
    logger.debug("LCM temporal rollup build failed", exc_info=True)


def build_day(
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_date: date | str,
    *,
    summarizer: Summarizer | None = None,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> dict[str, object] | None:
    """Build one UTC daily rollup without allowing failures into the caller."""
    rollup_id: int | None = None
    try:
        summarizer = summarizer or summarize_with_escalation
        day = _as_date(period_date)
        sources = _daily_sources(dag, scope, day)
        if not sources:
            return None

        source_ids = sorted(int(source["node_id"]) for source in sources)
        fingerprint = _stable_hash(
            [
                [int(source["node_id"]), hashlib.sha256(str(source["summary"]).encode("utf-8")).hexdigest()]
                for source in sorted(sources, key=lambda source: int(source["node_id"]))
            ]
        )
        text = "\n\n".join(
            f"[Summary node {source['node_id']}]\n{source['summary']}"
            for source in sources
        )
        rollup_id = store.upsert_building("day", day.isoformat(), scope)
        summary, token_count = _summarize_capped(
            text,
            target_tokens=config.rollup_daily_target_tokens,
            max_tokens=config.rollup_daily_max_tokens,
            config=config,
            summarizer=summarizer,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
        )
        store.mark_ready(rollup_id, summary, token_count, source_ids, fingerprint)
        return store.get_rollup("day", day.isoformat(), scope)
    except Exception as exc:
        _mark_failed(store, rollup_id, exc)
        return None


def _period_window(period_kind: str, period_start: date | str) -> tuple[date, date]:
    start = _as_date(period_start)
    if period_kind == "week":
        start -= timedelta(days=start.weekday())
        return start, start + timedelta(days=6)
    if period_kind == "month":
        start = start.replace(day=1)
        return start, start.replace(day=monthrange(start.year, start.month)[1])
    raise ValueError(f"unsupported aggregate period: {period_kind}")


def _daily_statuses(
    store: RollupStore,
    start: date,
    end: date,
    scope: str,
) -> dict[str, dict[str, object]]:
    connection = store.connection
    if connection is None:
        return {}
    rows = connection.execute(
        """
        SELECT period_start, status, source_fingerprint, summary, token_count, rollup_id
        FROM lcm_rollups
        WHERE period_kind = 'day'
          AND period_start >= ?
          AND period_start <= ?
          AND scope = ?
        ORDER BY period_start
        """,
        (start.isoformat(), end.isoformat(), scope),
    ).fetchall()
    return {
        str(row["period_start"]): {
            "status": str(row["status"]),
            "source_fingerprint": row["source_fingerprint"],
            "summary": row["summary"],
            "token_count": row["token_count"],
            "rollup_id": int(row["rollup_id"]),
        }
        for row in rows
    }


def _rollup_source_ids(store: RollupStore, rollup_ids: Sequence[int]) -> list[int]:
    if not rollup_ids or store.connection is None:
        return []
    placeholders = ",".join("?" for _ in rollup_ids)
    rows = store.connection.execute(
        f"""
        SELECT DISTINCT node_id
        FROM lcm_rollup_sources
        WHERE rollup_id IN ({placeholders})
        ORDER BY node_id
        """,
        [int(rollup_id) for rollup_id in rollup_ids],
    ).fetchall()
    return [int(row[0]) for row in rows]


def _build_aggregate(
    period_kind: str,
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_start: date | str,
    *,
    summarizer: Summarizer | None,
    circuit_breaker: object | None,
    spend_guard: object | None,
) -> dict[str, object] | None:
    del dag  # Aggregates deliberately depend only on ready daily rollups.
    rollup_id: int | None = None
    try:
        summarizer = summarizer or summarize_with_escalation
        start, end = _period_window(period_kind, period_start)
        statuses = _daily_statuses(store, start, end, scope)
        days: list[dict[str, object]] = []
        ready: list[tuple[str, dict[str, object]]] = []
        current = start
        while current <= end:
            day_key = current.isoformat()
            row = statuses.get(day_key)
            status = str(row["status"]) if row else "missing"
            fingerprint_value: object = status
            if row and status == "ready":
                fingerprint_value = row.get("source_fingerprint") or _stable_hash(row.get("summary") or "")
                ready.append((day_key, row))
            days.append({"day": day_key, "status": status, "fingerprint": fingerprint_value})
            current += timedelta(days=1)

        if not ready:
            return None

        fingerprint = _stable_hash(days)
        text = "\n\n".join(
            f"[Daily rollup {day_key}]\n{row.get('summary') or ''}"
            for day_key, row in ready
        )
        source_ids = _rollup_source_ids(
            store,
            [int(row["rollup_id"]) for _day_key, row in ready],
        )
        rollup_id = store.upsert_building(period_kind, start.isoformat(), scope)
        summary, token_count = _summarize_capped(
            text,
            target_tokens=config.rollup_aggregate_max_tokens,
            max_tokens=config.rollup_aggregate_max_tokens,
            config=config,
            summarizer=summarizer,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
        )
        store.mark_ready(rollup_id, summary, token_count, source_ids, fingerprint)
        return store.get_rollup(period_kind, start.isoformat(), scope)
    except Exception as exc:
        _mark_failed(store, rollup_id, exc)
        return None


def build_week(
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_start: date | str,
    *,
    summarizer: Summarizer | None = None,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> dict[str, object] | None:
    return _build_aggregate(
        "week", store, dag, config, scope, period_start,
        summarizer=summarizer,
        circuit_breaker=circuit_breaker,
        spend_guard=spend_guard,
    )


def build_month(
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_start: date | str,
    *,
    summarizer: Summarizer | None = None,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> dict[str, object] | None:
    return _build_aggregate(
        "month", store, dag, config, scope, period_start,
        summarizer=summarizer,
        circuit_breaker=circuit_breaker,
        spend_guard=spend_guard,
    )


def mark_stale_after_ingest(
    dag: SummaryDAG,
    scope: str,
    store_ids: Sequence[int],
) -> int:
    """Mark rollups stale for the UTC days of newly persisted messages."""
    store: RollupStore | None = None
    try:
        connection = dag.connection
        if not store_ids or connection is None:
            return 0
        placeholders = ",".join("?" for _ in store_ids)
        rows = connection.execute(
            f"""
            SELECT DISTINCT date(timestamp, 'unixepoch')
            FROM messages
            WHERE store_id IN ({placeholders})
            ORDER BY 1
            """,
            [int(store_id) for store_id in store_ids],
        ).fetchall()
        store = RollupStore(dag.db_path)
        return sum(store.mark_stale_for_day(str(row[0]), scope) for row in rows if row[0])
    except Exception:
        logger.debug("LCM temporal rollup staleness update failed", exc_info=True)
        return 0
    finally:
        if store is not None:
            store.close()


def run_rollup_maintenance(
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    *,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> int:
    """Rebuild a bounded stale batch: all dailies before any aggregates."""
    store: RollupStore | None = None
    try:
        limit = max(0, int(config.rollup_builds_per_pass))
        connection = dag.connection
        if limit <= 0 or connection is None:
            return 0
        rows = connection.execute(
            """
            WITH stale AS (
                SELECT period_kind, period_start
                FROM lcm_rollups INDEXED BY sqlite_autoindex_lcm_rollups_1
                WHERE period_kind = 'day' AND scope = ? AND status = 'stale'
                UNION ALL
                SELECT period_kind, period_start
                FROM lcm_rollups INDEXED BY sqlite_autoindex_lcm_rollups_1
                WHERE period_kind = 'week' AND scope = ? AND status = 'stale'
                UNION ALL
                SELECT period_kind, period_start
                FROM lcm_rollups INDEXED BY sqlite_autoindex_lcm_rollups_1
                WHERE period_kind = 'month' AND scope = ? AND status = 'stale'
            )
            SELECT period_kind, period_start
            FROM stale
            ORDER BY CASE WHEN period_kind = 'day' THEN 0 ELSE 1 END,
                     period_start,
                     period_kind
            LIMIT ?
            """,
            (scope, scope, scope, limit),
        ).fetchall()
        if not rows:
            return 0
        store = RollupStore(dag.db_path)
        builders: dict[str, Callable[..., dict[str, object] | None]] = {
            "day": build_day,
            "week": build_week,
            "month": build_month,
        }
        for row in rows:
            builder = builders[str(row[0])]
            builder(
                store,
                dag,
                config,
                scope,
                str(row[1]),
                circuit_breaker=circuit_breaker,
                spend_guard=spend_guard,
            )
        return len(rows)
    except Exception:
        logger.debug("LCM temporal rollup maintenance failed", exc_info=True)
        return 0
    finally:
        if store is not None:
            store.close()
