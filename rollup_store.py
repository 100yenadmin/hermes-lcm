"""SQLite-backed storage for derived temporal summary rollups.

This module is intentionally not wired into the LCM engine yet. It provides
only the durable schema-facing operations used by later temporal-memory work.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Sequence

from .db_bootstrap import (
    configure_connection,
    ensure_temporal_rollup_tables,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .sqlite_util import _is_sqlite_locked_error

logger = logging.getLogger(__name__)

# How long a ``building`` row's lease is valid. A build that outlives its lease
# can be reclaimed to ``stale`` by a later maintenance pass; the generation
# compare-and-set in :meth:`RollupStore.mark_ready` still protects correctness
# if the original builder returns late, so a generous lease only costs a
# possible redundant rebuild, never a wrong publish.
_BUILD_LEASE_SECONDS = 900


class RollupBuildToken(NamedTuple):
    """A build lease returned by :meth:`RollupStore.upsert_building`.

    ``generation`` is the row's optimistic-concurrency counter captured at build
    start; :meth:`RollupStore.mark_ready` publishes only if the row's generation
    still equals this token (a compare-and-set), so an invalidation that arrives
    mid-build supersedes the stale builder.
    """

    rollup_id: int
    generation: int


class RollupStore:
    """SQLite-backed store for temporal rollups and their source nodes."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
        )
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        # The rollup tables are a lazy, opt-in feature: they are NOT part of the
        # core numeric schema_version (see db_bootstrap.run_versioned_migrations).
        # RollupStore is only constructed on the temporal_rollups_enabled path, so
        # creating them here keeps a disabled install at the base schema with no
        # rollup tables while still being idempotent under concurrent construction.
        ensure_temporal_rollup_tables(self._conn)
        mark_migration_step_complete(self._conn, "temporal_rollups_v1")
        self._conn.commit()

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        try:
            with self._write_lock, self._conn:
                yield
        except sqlite3.Error as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning("Temporal rollup write blocked by SQLite lock contention")
            raise

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live connection for read-only diagnostics, or ``None`` after close."""
        return getattr(self, "_conn", None)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _lease_deadline() -> str:
        return (
            datetime.now(timezone.utc) + timedelta(seconds=_BUILD_LEASE_SECONDS)
        ).isoformat()

    def _row_to_rollup(self, row: sqlite3.Row | None) -> dict[str, object] | None:
        if row is None:
            return None
        rollup_id = int(row["rollup_id"])
        source_rows = self._conn.execute(
            "SELECT node_id FROM lcm_rollup_sources WHERE rollup_id = ? ORDER BY node_id",
            (rollup_id,),
        ).fetchall()
        return {
            "rollup_id": rollup_id,
            "period_kind": row["period_kind"],
            "period_start": row["period_start"],
            "scope": row["scope"],
            "summary": row["summary"],
            "token_count": row["token_count"],
            "status": row["status"],
            "built_at": row["built_at"],
            "source_fingerprint": row["source_fingerprint"],
            "error": row["error"],
            "generation": int(row["generation"] or 0),
            "source_node_ids": [int(source_row["node_id"]) for source_row in source_rows],
        }

    def get_rollup(
        self,
        period_kind: str,
        period_start: str,
        scope: str,
    ) -> dict[str, object] | None:
        row = self._conn.execute(
            """
            SELECT *
            FROM lcm_rollups
            WHERE period_kind = ? AND period_start = ? AND scope = ?
            """,
            (period_kind, period_start, scope),
        ).fetchone()
        return self._row_to_rollup(row)

    def upsert_building(
        self, period_kind: str, period_start: str, scope: str
    ) -> RollupBuildToken:
        """Claim a build lease, returning the row id and its NEW generation.

        The claim is itself a compare-and-set: re-claiming an existing row
        advances ``generation`` so two builders racing the same period get
        DISTINCT tokens and only the latest claimant's :meth:`mark_ready` /
        :meth:`mark_failed` succeeds (an older claimant is superseded). A fresh
        row starts at generation 0 (no prior claimant to race). A
        ``lease_expires_at`` is stamped so a crashed builder's row can later be
        reclaimed by :meth:`reclaim_stale_building`.
        """
        with self._write_transaction():
            self._conn.execute(
                """
                INSERT INTO lcm_rollups(period_kind, period_start, scope, status, built_at, lease_expires_at)
                VALUES(?, ?, ?, 'building', ?, ?)
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    status = 'building',
                    generation = lcm_rollups.generation + 1,
                    built_at = excluded.built_at,
                    lease_expires_at = excluded.lease_expires_at,
                    source_fingerprint = NULL,
                    error = NULL
                """,
                (period_kind, period_start, scope, self._now(), self._lease_deadline()),
            )
            row = self._conn.execute(
                """
                SELECT rollup_id, generation
                FROM lcm_rollups
                WHERE period_kind = ? AND period_start = ? AND scope = ?
                """,
                (period_kind, period_start, scope),
            ).fetchone()
            rollup_id = int(row["rollup_id"])
            self._conn.execute(
                "DELETE FROM lcm_rollup_sources WHERE rollup_id = ?",
                (rollup_id,),
            )
        return RollupBuildToken(rollup_id, int(row["generation"] or 0))

    def mark_ready(
        self,
        token: RollupBuildToken,
        summary: str,
        token_count: int,
        source_node_ids: Sequence[int],
        fingerprint: str,
    ) -> bool:
        """Publish a completed build iff it was not superseded.

        Returns ``True`` when the row is published, or ``False`` when the build
        was superseded (an invalidation advanced the row's generation past the
        token, or the row was reclaimed) — in which case the newer state is left
        untouched. Raises ``ValueError`` only for a genuinely unknown rollup id.
        """
        unique_source_ids = list(dict.fromkeys(int(node_id) for node_id in source_node_ids))
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET summary = ?, token_count = ?, status = 'ready', built_at = ?,
                    source_fingerprint = ?, error = NULL, lease_expires_at = NULL
                WHERE rollup_id = ? AND generation = ?
                """,
                (
                    summary,
                    int(token_count),
                    self._now(),
                    fingerprint,
                    int(token.rollup_id),
                    int(token.generation),
                ),
            )
            if cur.rowcount == 0:
                exists = self._conn.execute(
                    "SELECT 1 FROM lcm_rollups WHERE rollup_id = ?",
                    (int(token.rollup_id),),
                ).fetchone()
                if exists is None:
                    raise ValueError(f"unknown rollup_id: {token.rollup_id}")
                # Superseded by a newer generation: discard this build's result.
                return False
            self._conn.execute(
                "DELETE FROM lcm_rollup_sources WHERE rollup_id = ?",
                (int(token.rollup_id),),
            )
            self._conn.executemany(
                "INSERT INTO lcm_rollup_sources(rollup_id, node_id) VALUES(?, ?)",
                ((int(token.rollup_id), node_id) for node_id in unique_source_ids),
            )
        return True

    def mark_failed(
        self, rollup_id: int, error: str, *, generation: int | None = None
    ) -> bool:
        """Record a build failure, optionally guarded by ``generation``.

        When ``generation`` is given the update is a compare-and-set (exactly
        like :meth:`mark_ready`): a superseded builder's late exception cannot
        flip a newer ready/stale row to ``failed``. Returns ``True`` when a row
        was updated, ``False`` when the build was superseded. An unguarded call
        (``generation is None``) always writes and is used only to seed a known
        failure state directly.
        """
        with self._write_transaction():
            if generation is None:
                cur = self._conn.execute(
                    """
                    UPDATE lcm_rollups
                    SET status = 'failed', error = ?
                    WHERE rollup_id = ?
                    """,
                    (error, int(rollup_id)),
                )
            else:
                cur = self._conn.execute(
                    """
                    UPDATE lcm_rollups
                    SET status = 'failed', error = ?
                    WHERE rollup_id = ? AND generation = ?
                    """,
                    (error, int(rollup_id), int(generation)),
                )
        return int(cur.rowcount or 0) > 0

    @staticmethod
    def _period_starts_for_day(day: date | str) -> tuple[str, str, str]:
        if isinstance(day, datetime):
            parsed = day.date()
        elif isinstance(day, date):
            parsed = day
        else:
            parsed = date.fromisoformat(str(day))
        week_start = parsed - timedelta(days=parsed.weekday())
        month_start = parsed.replace(day=1)
        return parsed.isoformat(), week_start.isoformat(), month_start.isoformat()

    def mark_stale_for_day(self, day: date | str, scope: str) -> int:
        """Invalidate a day and its containing week + month.

        Every invalidation advances ``generation`` (regardless of the prior
        status) so a build that is in flight for any of these periods is
        superseded and its late ``mark_ready`` becomes a no-op. Rows that do not
        yet exist are seeded as ``stale`` so maintenance builds them.
        """
        day_start, week_start, month_start = self._period_starts_for_day(day)
        with self._write_transaction():
            cur = self._conn.execute(
                """
                INSERT INTO lcm_rollups(period_kind, period_start, scope, status)
                VALUES
                    ('day', ?, ?, 'stale'),
                    ('week', ?, ?, 'stale'),
                    ('month', ?, ?, 'stale')
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    status = 'stale',
                    generation = lcm_rollups.generation + 1,
                    lease_expires_at = NULL
                """,
                (day_start, scope, week_start, scope, month_start, scope),
            )
        return int(cur.rowcount or 0)

    def stale_aggregates_for_day(self, day: date | str, scope: str) -> int:
        """Invalidate only the week + month containing ``day`` (not the day).

        Called after a daily rollup is (re)built so its containing aggregates,
        which may have published against the previous daily, are rebuilt from the
        new daily. Advances ``generation`` so an in-flight aggregate build is
        superseded.
        """
        _day_start, week_start, month_start = self._period_starts_for_day(day)
        with self._write_transaction():
            cur = self._conn.execute(
                """
                INSERT INTO lcm_rollups(period_kind, period_start, scope, status)
                VALUES
                    ('week', ?, ?, 'stale'),
                    ('month', ?, ?, 'stale')
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    status = 'stale',
                    generation = lcm_rollups.generation + 1,
                    lease_expires_at = NULL
                """,
                (week_start, scope, month_start, scope),
            )
        return int(cur.rowcount or 0)

    def upsert_stale(self, period_kind: str, period_start: str, scope: str) -> int:
        """Durably seed a single ``stale`` row for one period.

        Used by ``/lcm rollups rebuild`` to queue every requested target before
        the per-pass build budget is applied, so unattempted targets remain
        durably ``stale`` (not absent) and get built by later maintenance. Does
        not disturb a row that is currently ``building``.
        """
        with self._write_transaction():
            cur = self._conn.execute(
                """
                INSERT INTO lcm_rollups(period_kind, period_start, scope, status)
                VALUES(?, ?, ?, 'stale')
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    status = 'stale',
                    generation = lcm_rollups.generation + 1,
                    lease_expires_at = NULL
                WHERE lcm_rollups.status != 'building'
                """,
                (period_kind, period_start, scope),
            )
        return int(cur.rowcount or 0)

    def defer_incomplete(self, token: RollupBuildToken, reason: str) -> bool:
        """Release a claimed aggregate build back to ``stale`` with a reason.

        Called when a claimed aggregate cannot be published because a constituent
        daily is missing/stale/building. Guarded by the build token
        (compare-and-set on ``generation`` for a row this builder still owns), so
        a superseded builder's deferral cannot erase a newer ready aggregate that
        another builder published in the meantime. ``generation`` is left
        unchanged (a deferral is not an input change). Returns ``True`` when the
        owned ``building`` row was released, ``False`` when superseded.
        """
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET status = 'stale', error = ?, lease_expires_at = NULL
                WHERE rollup_id = ? AND generation = ? AND status = 'building'
                """,
                (reason, int(token.rollup_id), int(token.generation)),
            )
        return int(cur.rowcount or 0) > 0

    def resolve_no_source(self, token: RollupBuildToken) -> bool:
        """Clear a claimed period that turned out to have no source content.

        A period can go ``stale`` yet have no summary node to build from (all its
        sources were deleted, or a rebuild request seeded a contentless target).
        Such a period must not linger ``stale`` forever consuming a per-pass build
        slot: this deletes the claimed row (and its sources) iff this builder
        still owns it (compare-and-set on ``generation``). A later covering
        publication re-seeds a fresh ``stale`` row. Returns ``True`` when cleared,
        ``False`` when superseded (a newer invalidation is left to rebuild).
        """
        with self._write_transaction():
            cur = self._conn.execute(
                "DELETE FROM lcm_rollups WHERE rollup_id = ? AND generation = ?",
                (int(token.rollup_id), int(token.generation)),
            )
            if int(cur.rowcount or 0) > 0:
                self._conn.execute(
                    "DELETE FROM lcm_rollup_sources WHERE rollup_id = ?",
                    (int(token.rollup_id),),
                )
        return int(cur.rowcount or 0) > 0

    def reclaim_stale_building(self, now: str | None = None) -> int:
        """Flip expired ``building`` rows back to ``stale`` so a crashed build is
        retried. Advances ``generation`` so the crashed builder, if it ever
        returns, cannot publish over the reclaimed (and possibly re-superseded)
        row.
        """
        cutoff = now or self._now()
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET status = 'stale',
                    generation = generation + 1,
                    lease_expires_at = NULL
                WHERE status = 'building'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (cutoff,),
            )
        return int(cur.rowcount or 0)

    def ready_rollups_for_window(
        self,
        period_kind: str,
        start: str,
        end: str,
        scope: str,
    ) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM lcm_rollups
            WHERE period_kind = ?
              AND period_start >= ?
              AND period_start <= ?
              AND scope = ?
              AND status = 'ready'
            ORDER BY period_start
            """,
            (period_kind, start, end, scope),
        ).fetchall()
        return [self._row_to_rollup(row) for row in rows]

    def get_cursor(self, period_kind: str, scope: str = "") -> str | None:
        row = self._conn.execute(
            "SELECT last_build_cursor FROM lcm_rollup_state WHERE period_kind = ? AND scope = ?",
            (period_kind, scope),
        ).fetchone()
        return str(row["last_build_cursor"]) if row and row["last_build_cursor"] is not None else None

    def set_cursor(
        self,
        period_kind: str,
        cursor: str | None,
        scope: str = "",
        *,
        built_at: str | None = None,
    ) -> None:
        with self._write_transaction():
            self._conn.execute(
                """
                INSERT INTO lcm_rollup_state(period_kind, scope, last_build_cursor, last_built_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(period_kind, scope) DO UPDATE SET
                    last_build_cursor = excluded.last_build_cursor,
                    last_built_at = excluded.last_built_at
                """,
                (period_kind, scope, cursor, built_at or self._now()),
            )

    def purge_rollups_for_sources(self, node_ids: Sequence[int]) -> int:
        """Invalidate rollups built from now-purged source nodes.

        The affected periods are re-seeded ``stale`` (generation advanced, so an
        in-flight build cannot publish purged content) rather than hard-deleted:
        a deleted row at/before the build cursor would leave the cursor advanced
        with no pending row, so the window would never be rebuilt. Re-staling
        leaves a durable pending row that maintenance rebuilds from whatever
        sources remain (or clears via :meth:`resolve_no_source` if none do). The
        rollups' stale source rows are dropped and repopulated on rebuild.
        Returns the number of periods re-staled.
        """
        unique_node_ids = list(dict.fromkeys(int(node_id) for node_id in node_ids))
        if not unique_node_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_node_ids)
        with self._write_transaction():
            rows = self._conn.execute(
                f"""
                SELECT DISTINCT rollup_id
                FROM lcm_rollup_sources
                WHERE node_id IN ({placeholders})
                """,
                unique_node_ids,
            ).fetchall()
            rollup_ids = [int(row["rollup_id"]) for row in rows]
            if not rollup_ids:
                return 0
            rollup_placeholders = ",".join("?" for _ in rollup_ids)
            self._conn.execute(
                f"DELETE FROM lcm_rollup_sources WHERE rollup_id IN ({rollup_placeholders})",
                rollup_ids,
            )
            cur = self._conn.execute(
                f"""
                UPDATE lcm_rollups
                SET status = 'stale',
                    generation = generation + 1,
                    summary = NULL,
                    token_count = NULL,
                    source_fingerprint = NULL,
                    error = NULL,
                    lease_expires_at = NULL
                WHERE rollup_id IN ({rollup_placeholders})
                """,
                rollup_ids,
            )
        return int(cur.rowcount or 0)

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
