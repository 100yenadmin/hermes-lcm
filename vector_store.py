"""SQLite-backed storage and brute-force KNN for summary embeddings.

This module is intentionally not wired into the LCM engine yet. It owns only
the durable embedding schema operations and the dependency-optional compute
ladder used by later semantic retrieval work.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from array import array
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from .config import LCMConfig
from .db_bootstrap import (
    configure_connection,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .sqlite_util import _is_sqlite_locked_error

logger = logging.getLogger(__name__)


def _load_numpy():
    """Import NumPy lazily. Kept isolated so fallback behavior is testable."""
    import numpy

    return numpy


class KNNResult(list[tuple[str, float, str]]):
    """Ranked KNN rows with result-level vector coverage metadata."""

    def __init__(
        self,
        rows: Sequence[tuple[str, float, str]] = (),
        *,
        coverage: str,
    ) -> None:
        super().__init__(rows)
        self.coverage = coverage


class VectorStore:
    """SQLite-backed store for normalized summary embedding vectors."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        config: LCMConfig | None = None,
        bounded_scan_rows: int | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config = config or LCMConfig.from_env()
        self.bounded_scan_rows = (
            resolved_config.embedding_bounded_scan_rows
            if bounded_scan_rows is None
            else int(bounded_scan_rows)
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._matrix_cache: dict[
            tuple[str, int, int],
            tuple[list[int], list[str], list[str], Any],
        ] = {}
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
            with self._write_lock, self._cache_lock:
                with self._conn:
                    yield
                self._matrix_cache.clear()
        except sqlite3.Error as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning("Embedding write blocked by SQLite lock contention")
            raise

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live connection for read-only diagnostics, or ``None`` after close."""
        return getattr(self, "_conn", None)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalized(values: Sequence[float], *, expected_dim: int) -> list[float]:
        try:
            floats = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vector must contain only numeric values") from exc
        if len(floats) != expected_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {expected_dim}, got {len(floats)}"
            )
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("embedding vector values must be finite")
        magnitude = math.sqrt(sum(value * value for value in floats))
        if magnitude == 0.0:
            raise ValueError("embedding vector must have non-zero magnitude")
        return [value / magnitude for value in floats]

    def _profile(self, model_name: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT model_name, provider, dim, registered_at, active, archived_at
            FROM lcm_embedding_profile
            WHERE model_name = ?
            """,
            (model_name,),
        ).fetchone()

    def _current_profile(self) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT model_name, provider, dim, registered_at, active, archived_at
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL
            ORDER BY registered_at DESC, model_name DESC
            LIMIT 1
            """
        ).fetchone()

    def register_profile(self, model_name: str, provider: str, dim: int) -> None:
        model_name = str(model_name)
        provider = str(provider)
        dim = int(dim)
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not provider:
            raise ValueError("provider must not be empty")
        if dim < 1 or dim > 4096:
            raise ValueError("embedding dimension must be between 1 and 4096")

        with self._write_transaction():
            existing = self._profile(model_name)
            if existing is not None:
                if int(existing["dim"]) != dim:
                    raise ValueError(
                        f"embedding dimension for {model_name!r} is locked at "
                        f"{existing['dim']}, not {dim}"
                    )
                return
            self._conn.execute(
                """
                INSERT INTO lcm_embedding_profile(
                    model_name, provider, dim, registered_at, active, archived_at
                )
                VALUES(?, ?, ?, ?, 1, NULL)
                """,
                (model_name, provider, dim, self._now()),
            )

    def record_embedding(
        self,
        embedded_id: str,
        kind: str,
        model: str,
        vec: Sequence[float],
    ) -> None:
        embedded_id = str(embedded_id)
        model = str(model)
        if kind != "summary":
            raise ValueError("embedded kind must be 'summary'")
        profile = self._profile(model)
        if profile is None:
            raise ValueError(f"embedding profile is not registered: {model}")
        normalized = self._normalized(vec, expected_dim=int(profile["dim"]))
        packed = array("f", normalized).tobytes()

        with self._write_transaction():
            summary = self._conn.execute(
                """
                SELECT source_token_count
                FROM summary_nodes
                WHERE CAST(node_id AS TEXT) = ?
                """,
                (embedded_id,),
            ).fetchone()
            if summary is None:
                raise ValueError(f"summary node does not exist: {embedded_id}")
            embedded_at = self._now()
            self._conn.execute(
                """
                DELETE FROM lcm_embedding_vectors
                WHERE embedded_id = ? AND embedding_model = ?
                """,
                (embedded_id, model),
            )
            self._conn.execute(
                """
                DELETE FROM lcm_embedding_meta
                WHERE embedded_id = ? AND embedded_kind = ? AND embedding_model = ?
                """,
                (embedded_id, kind, model),
            )
            self._conn.execute(
                """
                INSERT INTO lcm_embedding_vectors(embedded_id, embedding_model, vec)
                VALUES(?, ?, ?)
                """,
                (embedded_id, model, packed),
            )
            self._conn.execute(
                """
                INSERT INTO lcm_embedding_meta(
                    embedded_id, embedded_kind, embedding_model, embedded_at,
                    source_token_count, archived
                )
                VALUES(?, ?, ?, ?, ?, 0)
                """,
                (
                    embedded_id,
                    kind,
                    model,
                    embedded_at,
                    int(summary["source_token_count"] or 0),
                ),
            )

    def purge_embeddings_for_nodes(self, node_ids: Sequence[int | str]) -> int:
        unique_ids = list(dict.fromkeys(str(node_id) for node_id in node_ids))
        if not unique_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        with self._write_transaction():
            cur = self._conn.execute(
                f"DELETE FROM lcm_embedding_vectors WHERE embedded_id IN ({placeholders})",
                unique_ids,
            )
            self._conn.execute(
                f"DELETE FROM lcm_embedding_meta WHERE embedded_id IN ({placeholders})",
                unique_ids,
            )
        return int(cur.rowcount or 0)

    def _vector_state(self, model: str) -> tuple[int, int]:
        row = self._conn.execute(
            """
            SELECT COALESCE(MAX(v.rowid), 0) AS max_rowid, COUNT(*) AS row_count
            FROM lcm_embedding_vectors v
            JOIN lcm_embedding_meta m
              ON m.embedded_id = v.embedded_id
             AND m.embedding_model = v.embedding_model
            WHERE v.embedding_model = ? AND m.archived = 0
            """,
            (model,),
        ).fetchone()
        return int(row["max_rowid"] or 0), int(row["row_count"] or 0)

    def _numpy_rows(
        self,
        numpy: Any,
        model: str,
        dim: int,
    ) -> tuple[list[int], list[str], list[str], Any]:
        with self._cache_lock:
            max_rowid, row_count = self._vector_state(model)
            key = (model, max_rowid, row_count)
            cached = self._matrix_cache.get(key)
            if cached is not None:
                return cached
            rows = self._conn.execute(
                """
                SELECT v.rowid, v.embedded_id, m.embedded_kind, v.vec
                FROM lcm_embedding_vectors v
                JOIN lcm_embedding_meta m
                  ON m.embedded_id = v.embedded_id
                 AND m.embedding_model = v.embedding_model
                WHERE v.embedding_model = ? AND m.archived = 0
                ORDER BY v.rowid
                """,
                (model,),
            ).fetchall()
            rowids: list[int] = []
            embedded_ids: list[str] = []
            kinds: list[str] = []
            vectors: list[Any] = []
            for row in rows:
                vector = numpy.frombuffer(row["vec"], dtype=numpy.float32)
                if int(vector.size) != dim:
                    continue
                rowids.append(int(row["rowid"]))
                embedded_ids.append(str(row["embedded_id"]))
                kinds.append(str(row["embedded_kind"]))
                vectors.append(vector)
            matrix = (
                numpy.vstack(vectors)
                if vectors
                else numpy.empty((0, dim), dtype=numpy.float32)
            )
            loaded = (rowids, embedded_ids, kinds, matrix)
            self._matrix_cache.clear()
            self._matrix_cache[key] = loaded
            return loaded

    def _bounded_rows(
        self,
        model: str,
        dim: int,
    ) -> tuple[list[int], list[str], list[str], list[list[float]]]:
        limit = max(0, self.bounded_scan_rows)
        with self._cache_lock:
            rows = self._conn.execute(
                """
                SELECT v.rowid, v.embedded_id, m.embedded_kind, v.vec
                FROM lcm_embedding_vectors v
                JOIN lcm_embedding_meta m
                  ON m.embedded_id = v.embedded_id
                 AND m.embedding_model = v.embedding_model
                WHERE v.embedding_model = ? AND m.archived = 0
                ORDER BY v.rowid DESC
                LIMIT ?
                """,
                (model, limit),
            ).fetchall()
        rowids: list[int] = []
        embedded_ids: list[str] = []
        kinds: list[str] = []
        vectors: list[list[float]] = []
        for row in rows:
            vector = array("f")
            try:
                vector.frombytes(row["vec"])
            except (TypeError, ValueError):
                continue
            if len(vector) != dim:
                continue
            rowids.append(int(row["rowid"]))
            embedded_ids.append(str(row["embedded_id"]))
            kinds.append(str(row["embedded_kind"]))
            vectors.append(list(vector))
        return rowids, embedded_ids, kinds, vectors

    @staticmethod
    def _ranked(
        rowids: Sequence[int],
        embedded_ids: Sequence[str],
        kinds: Sequence[str],
        scores: Sequence[float],
        limit: int,
    ) -> list[tuple[str, float, str]]:
        ranked = sorted(
            zip(rowids, embedded_ids, scores, kinds),
            key=lambda row: (-float(row[2]), -int(row[0]), str(row[1])),
        )
        return [
            (str(embedded_id), float(score), str(kind))
            for _, embedded_id, score, kind in ranked[:limit]
        ]

    def _filter_candidates(
        self,
        candidates: Sequence[tuple[str, float, str]],
        *,
        since: float | None,
        conversation_ids: Sequence[str] | None,
        limit: int,
    ) -> list[tuple[str, float, str]]:
        if not candidates:
            return []
        embedded_ids = list(dict.fromkeys(row[0] for row in candidates))
        placeholders = ",".join("?" for _ in embedded_ids)
        columns = {
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(summary_nodes)").fetchall()
        }
        where = [f"CAST(node_id AS TEXT) IN ({placeholders})"]
        args: list[object] = list(embedded_ids)
        if "suppressed_at" in columns:
            where.append("suppressed_at IS NULL")
        if since is not None:
            where.append("COALESCE(latest_at, created_at) >= ?")
            args.append(float(since))
        if conversation_ids is not None:
            normalized_ids = [str(value) for value in conversation_ids]
            if not normalized_ids:
                return []
            conversation_placeholders = ",".join("?" for _ in normalized_ids)
            where.append(f"session_id IN ({conversation_placeholders})")
            args.extend(normalized_ids)
        rows = self._conn.execute(
            f"SELECT CAST(node_id AS TEXT) FROM summary_nodes WHERE {' AND '.join(where)}",
            args,
        ).fetchall()
        allowed = {str(row[0]) for row in rows}
        return [row for row in candidates if row[0] in allowed][:limit]

    def knn(
        self,
        query_vec: Sequence[float],
        k: int = 50,
        model: str | None = None,
        since: float | None = None,
        conversation_ids: Sequence[str] | None = None,
    ) -> KNNResult:
        k = int(k)
        if k <= 0:
            return KNNResult(coverage="none")
        profile = self._profile(str(model)) if model is not None else self._current_profile()
        if profile is None:
            return KNNResult(coverage="none")
        selected_model = str(profile["model_name"])
        dim = int(profile["dim"])
        query = self._normalized(query_vec, expected_dim=dim)
        _, row_count = self._vector_state(selected_model)
        if row_count == 0:
            return KNNResult(coverage="none")

        filters_present = since is not None or conversation_ids is not None
        candidate_limit = min(k * 10, 500) if filters_present else k
        try:
            numpy = _load_numpy()
        except ImportError:
            numpy = None

        if numpy is not None:
            rowids, embedded_ids, kinds, matrix = self._numpy_rows(
                numpy,
                selected_model,
                dim,
            )
            query_array = numpy.asarray(query, dtype=numpy.float32)
            scores = matrix @ query_array
            coverage = "full"
        else:
            rowids, embedded_ids, kinds, vectors = self._bounded_rows(
                selected_model,
                dim,
            )
            scores = [
                sum(value * query_value for value, query_value in zip(vector, query))
                for vector in vectors
            ]
            coverage = "bounded"

        candidates = self._ranked(
            rowids,
            embedded_ids,
            kinds,
            scores,
            candidate_limit,
        )
        filtered = self._filter_candidates(
            candidates,
            since=since,
            conversation_ids=conversation_ids,
            limit=k,
        )
        return KNNResult(filtered, coverage=coverage)

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None
        with self._cache_lock:
            self._matrix_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
