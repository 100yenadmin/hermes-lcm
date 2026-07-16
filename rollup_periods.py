"""Natural-time parsing for temporal rollup retrieval."""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence


@dataclass(frozen=True)
class CoverageNode:
    """A summary node's coverage span, for interval-aware rollup selection.

    ``source_node_ids`` are the *node* ids this node condenses (empty for
    message-sourced leaves); ``depth`` is its DAG depth. ``earliest_at`` /
    ``latest_at`` are its covered-span bounds (newest/oldest covered message
    timestamps), used to reason about which UTC days the node covers.
    """

    node_id: int
    depth: int
    source_node_ids: tuple[int, ...] = ()
    earliest_at: float | None = None
    latest_at: float | None = None


def covered_days(covered_start: float, covered_end: float) -> list[str]:
    """The sorted UTC day (``YYYY-MM-DD``) set a coverage interval intersects.

    ``[covered_start, covered_end]`` is a summary node's covered span; a node
    that spans past midnight covers MORE than one UTC day. Publication
    invalidation and interval-aware frontier selection both need "which days
    does this node touch?", so the answer lives in one shared helper
    (maintainer #388 B2 / #389 C1). Bounds are epoch seconds; a reversed pair is
    tolerated (swapped) so callers need not pre-sort.
    """
    start = datetime.fromtimestamp(float(covered_start), tz=timezone.utc).date()
    end = datetime.fromtimestamp(float(covered_end), tz=timezone.utc).date()
    if end < start:
        start, end = end, start
    days: list[str] = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def canonical_frontier(nodes: Sequence[CoverageNode]) -> list[CoverageNode]:
    """Return the interval-aware canonical covering set of ``nodes``.

    Drop any node referenced as a direct source by a higher-depth node ALSO
    present in the candidate set: that higher-depth parent already covers the
    child's lineage, so keeping both double-counts the same covered content.
    Coverage is interval-aware because the candidate set spans day boundaries —
    a parent whose span crosses midnight still suppresses its children on
    adjacent days as long as both are candidates. Order is preserved.

    This is the one frontier used by the daily source selection (maintainer #388
    B1) AND the ``lcm_recent`` leaf fallback (maintainer #389 C1), so both agree
    on what "the canonical node for this coverage" means.
    """
    by_id = {node.node_id: node for node in nodes}
    suppressed: set[int] = set()
    for parent in nodes:
        for child_id in parent.source_node_ids:
            child = by_id.get(child_id)
            if child is not None and child.depth < parent.depth:
                suppressed.add(child_id)
    return [node for node in nodes if node.node_id not in suppressed]


@dataclass(frozen=True)
class RecentPeriodWindow:
    """A normalized UTC ``[start, end)`` retrieval window."""

    period: str
    start: datetime
    end: datetime
    rollup_kind: str
    subday: bool = False


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc)


def _day_start(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def parse_recent_period(period: str, *, now: datetime | None = None) -> RecentPeriodWindow:
    """Parse an ``lcm_recent`` period into a deterministic UTC window."""
    if not isinstance(period, str) or not period.strip():
        raise ValueError("period is required")

    requested = " ".join(period.strip().lower().split())
    current = _utc_now(now)
    today = current.date()

    if requested == "today":
        start = _day_start(today)
        return RecentPeriodWindow(requested, start, start + timedelta(days=1), "day")

    if requested == "yesterday":
        start = _day_start(today - timedelta(days=1))
        return RecentPeriodWindow(requested, start, start + timedelta(days=1), "day")

    if requested == "week":
        week_start = today - timedelta(days=today.weekday())
        start = _day_start(week_start)
        return RecentPeriodWindow(requested, start, start + timedelta(days=7), "week")

    if requested == "month":
        month_start = today.replace(day=1)
        start = _day_start(month_start)
        end = _day_start(month_start.replace(day=monthrange(today.year, today.month)[1])) + timedelta(days=1)
        return RecentPeriodWindow(requested, start, end, "month")

    date_match = re.fullmatch(r"date:(\d{4}-\d{2}-\d{2})", requested)
    if date_match:
        try:
            parsed_date = date.fromisoformat(date_match.group(1))
        except ValueError as exc:
            raise ValueError("period date must be a valid YYYY-MM-DD") from exc
        start = _day_start(parsed_date)
        return RecentPeriodWindow(requested, start, start + timedelta(days=1), "day")

    days_match = re.fullmatch(r"(\d+)d", requested)
    if days_match:
        days = int(days_match.group(1))
        if days <= 0:
            raise ValueError("day period must be at least 1d")
        try:
            start = _day_start(today - timedelta(days=days - 1))
        except OverflowError as exc:
            raise ValueError("day period is outside the supported date range") from exc
        return RecentPeriodWindow(requested, start, _day_start(today) + timedelta(days=1), "day")

    hours_match = re.fullmatch(r"last (\d+)h", requested)
    if hours_match:
        hours = int(hours_match.group(1))
        if hours <= 0:
            raise ValueError("hour period must be at least last 1h")
        try:
            start = current - timedelta(hours=hours)
        except OverflowError as exc:
            raise ValueError("hour period is outside the supported date range") from exc
        return RecentPeriodWindow(requested, start, current, "day", subday=True)

    raise ValueError(
        "period must be one of: today, yesterday, Nd, week, month, "
        "date:YYYY-MM-DD, last Nh"
    )
