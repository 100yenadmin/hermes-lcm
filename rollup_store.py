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
from typing import Iterator, Optional, Sequence

from .db_bootstrap import (
    configure_connection,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .sqlite_util import _is_sqlite_locked_error

logger = logging.getLogger(__name__)


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

    def upsert_building(self, period_kind: str, period_start: str, scope: str) -> int:
        with self._write_transaction():
            self._conn.execute(
                """
                INSERT INTO lcm_rollups(period_kind, period_start, scope, status, built_at)
                VALUES(?, ?, ?, 'building', ?)
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    status = 'building',
                    built_at = excluded.built_at,
                    source_fingerprint = NULL,
                    error = NULL
                """,
                (period_kind, period_start, scope, self._now()),
            )
            row = self._conn.execute(
                """
                SELECT rollup_id
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
        return rollup_id

    def mark_ready(
        self,
        rollup_id: int,
        summary: str,
        token_count: int,
        source_node_ids: Sequence[int],
        fingerprint: str,
    ) -> None:
        unique_source_ids = list(dict.fromkeys(int(node_id) for node_id in source_node_ids))
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET summary = ?, token_count = ?, status = 'ready', built_at = ?,
                    source_fingerprint = ?, error = NULL
                WHERE rollup_id = ?
                """,
                (summary, int(token_count), self._now(), fingerprint, int(rollup_id)),
            )
            if cur.rowcount == 0:
                raise ValueError(f"unknown rollup_id: {rollup_id}")
            self._conn.execute(
                "DELETE FROM lcm_rollup_sources WHERE rollup_id = ?",
                (int(rollup_id),),
            )
            self._conn.executemany(
                "INSERT INTO lcm_rollup_sources(rollup_id, node_id) VALUES(?, ?)",
                ((int(rollup_id), node_id) for node_id in unique_source_ids),
            )

    def mark_failed(self, rollup_id: int, error: str) -> None:
        with self._write_transaction():
            self._conn.execute(
                """
                UPDATE lcm_rollups
                SET status = 'failed', error = ?
                WHERE rollup_id = ?
                """,
                (error, int(rollup_id)),
            )

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
                    status = 'stale'
                WHERE lcm_rollups.status = 'ready'
                """,
                (day_start, scope, week_start, scope, month_start, scope),
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

    def get_cursor(self, period_kind: str) -> str | None:
        row = self._conn.execute(
            "SELECT last_build_cursor FROM lcm_rollup_state WHERE period_kind = ?",
            (period_kind,),
        ).fetchone()
        return str(row["last_build_cursor"]) if row and row["last_build_cursor"] is not None else None

    def set_cursor(
        self,
        period_kind: str,
        cursor: str | None,
        *,
        built_at: str | None = None,
    ) -> None:
        with self._write_transaction():
            self._conn.execute(
                """
                INSERT INTO lcm_rollup_state(period_kind, last_build_cursor, last_built_at)
                VALUES(?, ?, ?)
                ON CONFLICT(period_kind) DO UPDATE SET
                    last_build_cursor = excluded.last_build_cursor,
                    last_built_at = excluded.last_built_at
                """,
                (period_kind, cursor, built_at or self._now()),
            )

    def purge_rollups_for_sources(self, node_ids: Sequence[int]) -> int:
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
                f"DELETE FROM lcm_rollups WHERE rollup_id IN ({rollup_placeholders})",
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
