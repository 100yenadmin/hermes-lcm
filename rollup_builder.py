"""Flag-gated temporal rollup construction and engine wiring helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from time import monotonic
from typing import Callable, Sequence

from .config import LCMConfig
from .dag import SummaryDAG
from .escalation import _deterministic_truncate, summarize_with_escalation
from .rollup_periods import CoverageNode, canonical_frontier, covered_days
from .rollup_store import RollupBuildToken, RollupStore
from .tokens import count_tokens

logger = logging.getLogger(__name__)

Summarizer = Callable[..., tuple[str, int]]
_FAILED_ROLLUP_BACKOFF = timedelta(seconds=30)

_PENDING_ROLLUPS_SQL = """
    SELECT period_kind, period_start
    FROM lcm_rollups
    WHERE scope = ?
      AND status IN ('stale', 'failed')
      AND (
        status = 'stale'
        OR built_at IS NULL
        OR built_at <= ?
      )
    ORDER BY CASE WHEN period_kind = 'day' THEN 0 ELSE 1 END,
             period_start,
             period_kind
    LIMIT ?
"""


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scope_frontier(dag: SummaryDAG, scope: str) -> list[dict[str, object]]:
    """The scope's canonical frontier nodes, each tagged with a representative day.

    Loads every summary node for ``scope`` and applies the shared interval-aware
    :func:`canonical_frontier`: a node condensed by a higher-depth parent anywhere
    in the scope is suppressed, so a parent that spans several days stands in for
    the children it covers even when those children land on adjacent days. Each
    surviving frontier node is assigned to exactly ONE representative day (the UTC
    day of its newest covered timestamp) so its lineage appears in a single daily
    rollup — never duplicated across adjacent dailies (maintainer #388 B1).
    """
    connection = dag.connection
    if connection is None:
        return []
    rows = connection.execute(
        """
        SELECT node_id, depth, summary, source_ids, source_type,
               COALESCE(earliest_at, created_at) AS earliest_at,
               COALESCE(latest_at, created_at) AS latest_at
        FROM summary_nodes
        WHERE session_id = ?
        ORDER BY COALESCE(latest_at, created_at), node_id
        """,
        (scope,),
    ).fetchall()
    candidates: list[CoverageNode] = []
    meta: dict[int, dict[str, object]] = {}
    for row in rows:
        node_id = int(row[0])
        source_type = str(row[4] or "")
        source_node_ids: tuple[int, ...] = ()
        if source_type == "nodes" and row[3]:
            try:
                source_node_ids = tuple(int(value) for value in json.loads(row[3]))
            except (TypeError, ValueError):
                source_node_ids = ()
        latest_at = row[6]
        candidates.append(
            CoverageNode(
                node_id=node_id,
                depth=int(row[1] or 0),
                source_node_ids=source_node_ids,
                earliest_at=row[5],
                latest_at=latest_at,
            )
        )
        representative_day = (
            datetime.fromtimestamp(float(latest_at), tz=timezone.utc).date().isoformat()
            if latest_at is not None
            else None
        )
        meta[node_id] = {
            "summary": str(row[2] or ""),
            "representative_day": representative_day,
        }
    frontier: list[dict[str, object]] = []
    for node in canonical_frontier(candidates):
        info = meta[node.node_id]
        frontier.append(
            {
                "node_id": node.node_id,
                "summary": info["summary"],
                "representative_day": info["representative_day"],
            }
        )
    return frontier


def _daily_sources(dag: SummaryDAG, scope: str, day: date) -> list[dict[str, object]]:
    """Return the canonical frontier nodes whose representative day is ``day``.

    A condensed child and its condensing parent must not both feed rollups: the
    parent already covers the child's lineage. Crucially the parent may land on a
    DIFFERENT day than the child (its span crosses midnight), so the suppression
    is computed over the whole scope's frontier, not just this day's rows — a
    child on Jul15 covered by a parent whose representative day is Jul16 is
    suppressed from Jul15, and the parent feeds only Jul16, so adjacent dailies
    never duplicate the same covered leaf lineage (maintainer #388 B1 /
    #389 source-dedup).
    """
    day_key = day.isoformat()
    return [
        {"node_id": node["node_id"], "summary": node["summary"]}
        for node in _scope_frontier(dag, scope)
        if node["representative_day"] == day_key
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


def _mark_failed(
    store: RollupStore, token: "RollupBuildToken | None", exc: Exception
) -> None:
    if token is not None:
        try:
            # Generation-guarded: a superseded builder's late exception must not
            # flip a newer ready/stale row to failed (maintainer #387 blocker 2).
            store.mark_failed(
                token.rollup_id,
                f"{type(exc).__name__}: {exc}",
                generation=token.generation,
            )
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
    token: RollupBuildToken | None = None
    try:
        summarizer = summarizer or summarize_with_escalation
        day = _as_date(period_date)
        # Capture the build token (advancing the generation) BEFORE reading the
        # source snapshot, so an invalidation that arrives while we build is
        # guaranteed to supersede this token's mark_ready — a snapshot read
        # before the claim could otherwise be published stale (maintainer #388
        # blocker: capture-token-first).
        token = store.upsert_building("day", day.isoformat(), scope)
        sources = _daily_sources(dag, scope, day)
        if not sources:
            # A stale day with no summary node to build from must resolve, not
            # linger stale forever consuming a per-pass build slot (maintainer
            # #388 blocker: no-source jobs).
            store.resolve_no_source(token)
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
        summary, token_count = _summarize_capped(
            text,
            target_tokens=config.rollup_daily_target_tokens,
            max_tokens=config.rollup_daily_max_tokens,
            config=config,
            summarizer=summarizer,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
        )
        published = store.mark_ready(token, summary, token_count, source_ids, fingerprint)
        if published:
            # A (re)built daily makes any already-published week/month that
            # covered the previous daily outdated: stale them so they rebuild
            # from the new daily (maintainer #388 blocker 5 — aggregate rebuild).
            store.stale_aggregates_for_day(day, scope)
        return store.get_rollup("day", day.isoformat(), scope)
    except Exception as exc:
        _mark_failed(store, token, exc)
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


def _days_with_content(
    dag: SummaryDAG,
    scope: str,
    start: date,
    end: date,
) -> set[str]:
    """UTC days in ``[start, end]`` that have canonical-frontier content for ``scope``.

    A rollup consumes the scope's canonical frontier; a day counts as having
    content when a frontier node's representative day falls on it. This MUST use
    the same frontier as :func:`_daily_sources`: a day whose only node is a child
    suppressed by a multi-day parent has NO daily to build, so it must not count
    as content and block the aggregate's completeness gate (maintainer #388 B1).
    A day with no frontier content legitimately has no daily rollup.
    """
    start_key = start.isoformat()
    end_key = end.isoformat()
    return {
        str(node["representative_day"])
        for node in _scope_frontier(dag, scope)
        if node["representative_day"] is not None
        and start_key <= str(node["representative_day"]) <= end_key
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
    token: RollupBuildToken | None = None
    try:
        summarizer = summarizer or summarize_with_escalation
        start, end = _period_window(period_kind, period_start)
        # Capture the build token (advancing the generation) BEFORE reading the
        # daily-status snapshot, so a daily (re)build that stales this aggregate
        # while we build supersedes this token's mark_ready — reading dailies
        # before the claim could otherwise publish an aggregate that omits a
        # just-rebuilt daily (maintainer #388 blocker: capture-token-first).
        token = store.upsert_building(period_kind, start.isoformat(), scope)
        statuses = _daily_statuses(store, start, end, scope)

        # Completeness gate (maintainer #388 blocker 5): only publish a ready
        # aggregate when every day that HAS content in the window has a ready
        # daily rollup. A content day that is missing/stale/building blocks the
        # aggregate, which is released back to stale (token-guarded) with a
        # recorded reason so it rebuilds once the daily catches up. Days with no
        # content do not block.
        content_days = _days_with_content(dag, scope, start, end)
        pending_days = sorted(
            day_key
            for day_key in content_days
            if str((statuses.get(day_key) or {}).get("status")) != "ready"
        )
        if pending_days:
            preview = ", ".join(pending_days[:5])
            store.defer_incomplete(
                token,
                f"incomplete: {len(pending_days)} daily rollup(s) not ready ({preview})",
            )
            return None

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
            # No ready constituent daily to aggregate: clear the claimed row so
            # it does not linger stale consuming a build slot (a later daily
            # (re)build re-stales the aggregate via stale_aggregates_for_day).
            store.resolve_no_source(token)
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
        summary, token_count = _summarize_capped(
            text,
            target_tokens=config.rollup_aggregate_max_tokens,
            max_tokens=config.rollup_aggregate_max_tokens,
            config=config,
            summarizer=summarizer,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
        )
        store.mark_ready(token, summary, token_count, source_ids, fingerprint)
        return store.get_rollup(period_kind, start.isoformat(), scope)
    except Exception as exc:
        _mark_failed(store, token, exc)
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


# NOTE: there is deliberately no ``mark_stale_after_ingest`` raw-ingest hook.
# Staleness is driven SOLELY by summary-node publication
# (:func:`mark_stale_for_published_summary`, wired at every ``_dag.add_node``
# site). A raw-ingest bind-time mark would stale a period before its covering
# summary exists, letting a rebuild publish ``ready`` from the OLD sources and
# omit the not-yet-published leaf (maintainer #388 P1). Rollups consume
# published summary nodes, so publication is the only correct trigger.


def mark_stale_for_published_summary(
    dag: SummaryDAG,
    scope: str,
    latest_at: float | None,
    created_at: float | None = None,
    *,
    earliest_at: float | None = None,
) -> int:
    """Invalidate the rollups for EVERY UTC day a published summary node covers.

    Rollups consume PUBLISHED summary nodes, not raw messages, so publication is
    the load-bearing staleness signal (maintainer #388 blocker 1): when a summary
    covering days D..D' is published, each of those days and their containing
    week/month must go stale so a later summary cannot leave an older rollup
    ``ready`` and apparently current. A summary whose coverage spans past midnight
    covers more than one day; keying only on ``latest_at`` left the earlier day(s)
    ``ready`` (maintainer #388 B2). The covered span is
    ``[earliest_at, latest_at]`` (each falling back to ``created_at``); the shared
    :func:`covered_days` helper enumerates the intersected UTC days.
    """
    store: RollupStore | None = None
    try:
        span_end = latest_at if latest_at is not None else created_at
        span_start = earliest_at if earliest_at is not None else span_end
        if not scope or span_end is None or span_start is None:
            return 0
        store = RollupStore(dag.db_path)
        affected = 0
        for day_key in covered_days(span_start, span_end):
            affected += store.mark_stale_for_day(day_key, scope)
        return affected
    except Exception:
        logger.debug("LCM temporal rollup publication staleness update failed", exc_info=True)
        return 0
    finally:
        if store is not None:
            store.close()


def mark_stale_for_deleted_nodes(dag: SummaryDAG, node_ids: Sequence[int]) -> int:
    """Mark rollups that reference deleted summary nodes stale for rebuilding."""
    store: RollupStore | None = None
    try:
        unique_node_ids = list(dict.fromkeys(int(node_id) for node_id in node_ids))
        if not unique_node_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_node_ids)
        store = RollupStore(dag.db_path)
        with store.connection:
            # Advance generation on the invalidation so an in-flight build for
            # any affected rollup is superseded and cannot publish deleted-node
            # content over the stale row (maintainer #388 blocker: deletion must
            # bump generation, mirroring mark_stale_for_day).
            cur = store.connection.execute(
                f"""
                UPDATE lcm_rollups
                SET status = 'stale',
                    generation = generation + 1,
                    lease_expires_at = NULL
                WHERE status != 'stale'
                  AND rollup_id IN (
                    SELECT rollup_id
                    FROM lcm_rollup_sources
                    WHERE node_id IN ({placeholders})
                  )
                """,
                unique_node_ids,
            )
        return int(cur.rowcount or 0)
    except Exception:
        logger.debug("LCM temporal rollup deletion staleness update failed", exc_info=True)
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
    """Best-effort bounded maintenance; slow summarizers may leave rollups lagging."""
    store: RollupStore | None = None
    started_at = monotonic()
    try:
        limit = max(0, int(config.rollup_builds_per_pass))
        budget_ms = max(0, int(config.rollup_maintenance_budget_ms))
        connection = dag.connection
        if limit <= 0 or budget_ms <= 0 or connection is None:
            return 0
        store = RollupStore(dag.db_path)
        # Reclaim rows whose build lease expired (a crashed builder left them
        # 'building' forever) back to 'stale' so this pass can retry them
        # (maintainer #388 blocker 2).
        store.reclaim_stale_building()
        retry_before = (datetime.now(timezone.utc) - _FAILED_ROLLUP_BACKOFF).isoformat()
        rows = connection.execute(
            _PENDING_ROLLUPS_SQL,
            (scope, retry_before, limit),
        ).fetchall()
        if not rows:
            return 0
        builders: dict[str, Callable[..., dict[str, object] | None]] = {
            "day": build_day,
            "week": build_week,
            "month": build_month,
        }
        builds_started = 0
        for row in rows:
            if (monotonic() - started_at) * 1000 >= budget_ms:
                break
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
            builds_started += 1
        return builds_started
    except Exception:
        logger.debug("LCM temporal rollup maintenance failed", exc_info=True)
        return 0
    finally:
        if store is not None:
            store.close()
