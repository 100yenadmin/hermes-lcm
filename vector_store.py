"""SQLite-backed storage and brute-force KNN for summary embeddings.

This module is intentionally not wired into the LCM engine yet. It owns only
the durable embedding schema operations and the dependency-optional compute
ladder used by later semantic retrieval work.
"""

from __future__ import annotations

import hashlib
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
    ensure_embedding_tables,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .sqlite_util import _is_sqlite_locked_error

logger = logging.getLogger(__name__)

# Vectors are float32 in native little-endian order today. These are recorded as
# part of the canonical profile identity so a future dtype/byteorder change is
# detectable rather than silently reinterpreting stored bytes.
_VECTOR_DTYPE = "float32"
_VECTOR_BYTEORDER = "little"
_DEFAULT_TASK = "summary"

# SQLite caps host parameters per statement (SQLITE_MAX_VARIABLE_NUMBER); a
# single ``WHERE id IN (?, ?, ...)`` over tens of thousands of ids overflows it
# (observed failure at ~33k ids). Candidate id resolution loads ids into a temp
# table in bounded chunks and JOINs instead, so it scales past that limit.
_ID_INSERT_CHUNK = 500


def _identity_hash(
    provider: str,
    model_name: str,
    revision: str,
    dim: int,
    dtype: str,
    byteorder: str,
    task: str,
) -> str:
    """Return the stable canonical-identity hash for an embedding profile."""
    canonical = "\x1f".join(
        [
            str(provider).strip().lower(),
            str(model_name).strip(),
            str(revision).strip(),
            str(int(dim)),
            str(dtype).strip().lower(),
            str(byteorder).strip().lower(),
            str(task).strip().lower(),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        # Cache key: (identity_hash, data_version, row_count). The durable
        # per-identity ``data_version`` counter is bumped inside every vector
        # write/delete transaction, so a cross-process write invalidates this
        # cache even when max_rowid and row_count are unchanged.
        self._matrix_cache: dict[
            tuple[str, int, int],
            tuple[list[int], list[str], list[str], Any],
        ] = {}
        self._init_db()

    def _init_db(self) -> None:
        # isolation_level=None (autocommit) so every read runs as its own
        # short read transaction rather than pinning a WAL snapshot for the
        # life of the connection. Without this, a long-lived VectorStore never
        # observes another process's committed vector writes (or the bumped
        # data_version), defeating the cross-process cache invalidation. Write
        # atomicity is preserved via the explicit BEGIN IMMEDIATE in
        # _write_transaction.
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        self._ensure_embedding_schema()
        self._conn.commit()

    def _ensure_embedding_schema(self) -> None:
        """Materialize the opt-in embedding tables lazily on first VectorStore use.

        Constructing a VectorStore means embeddings are in use; the core
        migration path leaves these tables uncreated so a disabled install
        stays at schema_version 5 with none of them. Creation is idempotent
        (CREATE TABLE IF NOT EXISTS) and recorded via a named migration marker.
        """
        with self._write_lock:
            ensure_embedding_tables(self._conn)
            mark_migration_step_complete(self._conn, "embeddings_v1")
            self._conn.commit()

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        # The connection is in autocommit mode (isolation_level=None), so a
        # multi-statement write must open its own explicit transaction to stay
        # atomic — the data_version bump has to commit with the vector write it
        # accompanies, never separately.
        try:
            with self._write_lock, self._cache_lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    yield
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
                finally:
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

    def _profile_by_identity(self, identity_hash: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, dim, registered_at,
                   active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE identity_hash = ?
            """,
            (str(identity_hash),),
        ).fetchone()

    def _resolve_profile(self, model_name: str) -> sqlite3.Row | None:
        """Resolve a model name to its profile, preferring the current/active one.

        Multiple identities can share a ``model_name`` (e.g. the same model
        served by two providers). Vector operations that take a bare model name
        target the active identity for that name; recording and querying are
        thus consistent with the operator's current provider selection.
        """
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, dim, registered_at,
                   active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE model_name = ?
            ORDER BY (active = 1 AND archived_at IS NULL) DESC,
                     registered_at DESC, identity_hash DESC
            LIMIT 1
            """,
            (str(model_name),),
        ).fetchone()

    def _current_profile(self) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, dim, registered_at,
                   active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """
        ).fetchone()

    def register_profile(
        self,
        model_name: str,
        provider: str,
        dim: int,
        *,
        revision: str = "",
        dtype: str = _VECTOR_DTYPE,
        byteorder: str = _VECTOR_BYTEORDER,
        task: str = _DEFAULT_TASK,
    ) -> str:
        """Register (or reactivate) the profile for a canonical identity.

        The identity is ``(provider, model_name, revision, dim, dtype,
        byteorder, task)``. Registering a *different* identity for the same
        model_name creates a new profile row rather than clobbering the old
        provider's metadata; re-registering an already-known identity
        reactivates it (its vectors remain valid, so switching config back
        needs no re-backfill). Exactly one profile is left active. Returns the
        identity hash.
        """
        model_name = str(model_name)
        provider = str(provider)
        dim = int(dim)
        revision = str(revision)
        dtype = str(dtype)
        byteorder = str(byteorder)
        task = str(task)
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not provider:
            raise ValueError("provider must not be empty")
        if dim < 1 or dim > 4096:
            raise ValueError("embedding dimension must be between 1 and 4096")

        identity = _identity_hash(
            provider, model_name, revision, dim, dtype, byteorder, task
        )
        with self._write_transaction():
            existing = self._profile_by_identity(identity)
            if existing is None:
                self._conn.execute(
                    """
                    INSERT INTO lcm_embedding_profile(
                        identity_hash, provider, model_name, revision, dim,
                        dtype, byteorder, task, registered_at, active,
                        archived_at, data_version
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, 0)
                    """,
                    (
                        identity, provider, model_name, revision, dim,
                        dtype, byteorder, task, self._now(),
                    ),
                )
            else:
                # Reactivate a previously-registered identity in place; its
                # vectors are still valid so no re-backfill is required.
                self._conn.execute(
                    """
                    UPDATE lcm_embedding_profile
                    SET active = 1, archived_at = NULL
                    WHERE identity_hash = ?
                    """,
                    (identity,),
                )
            # Selection of the "current" profile is by identity: only the
            # just-registered identity stays active; every other profile is
            # deactivated but retained so its vectors survive a later switch back.
            self._conn.execute(
                "UPDATE lcm_embedding_profile SET active = 0 WHERE identity_hash != ?",
                (identity,),
            )
        return identity

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
        profile = self._resolve_profile(model)
        if profile is None:
            raise ValueError(f"embedding profile is not registered: {model}")
        identity = str(profile["identity_hash"])
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
                WHERE embedded_id = ? AND identity_hash = ?
                """,
                (embedded_id, identity),
            )
            self._conn.execute(
                """
                DELETE FROM lcm_embedding_meta
                WHERE embedded_id = ? AND embedded_kind = ? AND identity_hash = ?
                """,
                (embedded_id, kind, identity),
            )
            self._conn.execute(
                """
                INSERT INTO lcm_embedding_vectors(embedded_id, identity_hash, vec)
                VALUES(?, ?, ?)
                """,
                (embedded_id, identity, packed),
            )
            self._conn.execute(
                """
                INSERT INTO lcm_embedding_meta(
                    embedded_id, embedded_kind, identity_hash, embedded_at,
                    source_token_count, archived
                )
                VALUES(?, ?, ?, ?, ?, 0)
                """,
                (
                    embedded_id,
                    kind,
                    identity,
                    embedded_at,
                    int(summary["source_token_count"] or 0),
                ),
            )
            self._bump_data_version(identity)

    def _bump_data_version(self, identity_hash: str) -> None:
        """Bump one identity's durable data-version counter.

        Must run inside the same write transaction as the vector write/delete
        it accompanies so a cross-process reader observes the new counter and
        invalidates its NumPy matrix cache.
        """
        self._conn.execute(
            "UPDATE lcm_embedding_profile SET data_version = data_version + 1 "
            "WHERE identity_hash = ?",
            (str(identity_hash),),
        )

    def _bump_all_data_versions(self) -> None:
        self._conn.execute(
            "UPDATE lcm_embedding_profile SET data_version = data_version + 1"
        )

    @contextmanager
    def _temp_id_table(self, ids: Sequence[str]) -> Iterator[str]:
        """Load ids into a session-temp table in bounded chunks for JOINs.

        Avoids the ``WHERE id IN (?, ?, ...)`` host-parameter cap
        (SQLITE_MAX_VARIABLE_NUMBER) that failed at ~33k ids: ``executemany``
        binds one row at a time, so an arbitrary number of ids can be staged.
        """
        table = "_lcm_id_scratch"
        self._conn.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {table}(id TEXT PRIMARY KEY)"
        )
        self._conn.execute(f"DELETE FROM {table}")
        rows = [(str(value),) for value in ids]
        for offset in range(0, len(rows), _ID_INSERT_CHUNK):
            self._conn.executemany(
                f"INSERT OR IGNORE INTO {table}(id) VALUES(?)",
                rows[offset:offset + _ID_INSERT_CHUNK],
            )
        try:
            yield table
        finally:
            self._conn.execute(f"DELETE FROM {table}")

    def purge_embeddings_for_nodes(self, node_ids: Sequence[int | str]) -> int:
        unique_ids = list(dict.fromkeys(str(node_id) for node_id in node_ids))
        if not unique_ids:
            return 0
        with self._write_transaction():
            with self._temp_id_table(unique_ids) as table:
                cur = self._conn.execute(
                    f"DELETE FROM lcm_embedding_vectors "
                    f"WHERE embedded_id IN (SELECT id FROM {table})"
                )
                self._conn.execute(
                    f"DELETE FROM lcm_embedding_meta "
                    f"WHERE embedded_id IN (SELECT id FROM {table})"
                )
            # Purge spans every identity keyed by these ids; bump all data
            # versions so any cached matrix that referenced a removed vector is
            # invalidated. Over-invalidation across identities is safe.
            if cur.rowcount:
                self._bump_all_data_versions()
        return int(cur.rowcount or 0)

    def _vector_state(self, identity_hash: str) -> tuple[int, int]:
        row = self._conn.execute(
            """
            SELECT COALESCE(MAX(v.rowid), 0) AS max_rowid, COUNT(*) AS row_count
            FROM lcm_embedding_vectors v
            JOIN lcm_embedding_meta m
              ON m.embedded_id = v.embedded_id
             AND m.identity_hash = v.identity_hash
            WHERE v.identity_hash = ? AND m.archived = 0
            """,
            (str(identity_hash),),
        ).fetchone()
        return int(row["max_rowid"] or 0), int(row["row_count"] or 0)

    def _data_version(self, identity_hash: str) -> int:
        row = self._conn.execute(
            "SELECT data_version FROM lcm_embedding_profile WHERE identity_hash = ?",
            (str(identity_hash),),
        ).fetchone()
        return int(row["data_version"]) if row is not None else 0

    def _numpy_rows(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
    ) -> tuple[list[int], list[str], list[str], Any]:
        with self._cache_lock:
            _, row_count = self._vector_state(identity_hash)
            data_version = self._data_version(identity_hash)
            key = (identity_hash, data_version, row_count)
            cached = self._matrix_cache.get(key)
            if cached is not None:
                return cached
            rows = self._conn.execute(
                """
                SELECT v.rowid, v.embedded_id, m.embedded_kind, v.vec
                FROM lcm_embedding_vectors v
                JOIN lcm_embedding_meta m
                  ON m.embedded_id = v.embedded_id
                 AND m.identity_hash = v.identity_hash
                WHERE v.identity_hash = ? AND m.archived = 0
                ORDER BY v.rowid
                """,
                (identity_hash,),
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
        identity_hash: str,
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
                 AND m.identity_hash = v.identity_hash
                WHERE v.identity_hash = ? AND m.archived = 0
                ORDER BY m.embedded_at DESC, v.rowid DESC
                LIMIT ?
                """,
                (identity_hash, limit),
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

    def _source_allowed_ids(self, table: str, source: str) -> set[str]:
        """Root candidate ids whose source subtree reaches a message with ``source``.

        Enforced at the store layer (before the top-k cap) by walking each
        candidate summary's source tree down to its raw messages — the same
        recursion the DAG uses — so an ineligible high-scoring vector cannot
        consume a slot ahead of an eligible lower one.
        """
        from .store import _UNKNOWN_SOURCE, _legacy_blank_source_clause, _normalize_source_value

        normalized_source = _normalize_source_value(source)
        legacy_blank_clause = _legacy_blank_source_clause("m.source")
        rows = self._conn.execute(
            f"""
            WITH RECURSIVE walk(root_id, source_type, source_id) AS (
                SELECT t.id, sn.source_type, CAST(j.value AS INTEGER)
                FROM {table} t
                JOIN summary_nodes sn ON CAST(sn.node_id AS TEXT) = t.id
                JOIN json_each(sn.source_ids) j

                UNION ALL

                SELECT w.root_id, child.source_type, CAST(j.value AS INTEGER)
                FROM walk w
                JOIN summary_nodes child
                  ON w.source_type = 'nodes' AND child.node_id = w.source_id
                JOIN json_each(child.source_ids) j
            )
            SELECT DISTINCT w.root_id
            FROM walk w
            JOIN messages m
              ON w.source_type = 'messages' AND m.store_id = w.source_id
            WHERE CASE
                    WHEN ? = ? THEN (m.source = ? OR {legacy_blank_clause})
                    ELSE m.source = ?
                  END
            """,
            (normalized_source, _UNKNOWN_SOURCE, normalized_source, normalized_source),
        ).fetchall()
        return {str(row[0]) for row in rows}

    def _filtered_candidate_indexes(
        self,
        embedded_ids: Sequence[str],
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
        source: str | None,
    ) -> list[int]:
        if not embedded_ids:
            return []
        if conversation_ids is not None and not list(conversation_ids):
            return []
        unique_ids = list(dict.fromkeys(str(value) for value in embedded_ids))
        columns = {
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(summary_nodes)").fetchall()
        }
        with self._temp_id_table(unique_ids) as table:
            where = ["1 = 1"]
            args: list[object] = []
            if "suppressed_at" in columns:
                where.append("sn.suppressed_at IS NULL")
            if since is not None:
                where.append("COALESCE(sn.latest_at, sn.created_at) >= ?")
                args.append(float(since))
            if until is not None:
                where.append("COALESCE(sn.latest_at, sn.created_at) <= ?")
                args.append(float(until))
            if conversation_ids is not None:
                normalized_ids = [str(value) for value in conversation_ids]
                conversation_placeholders = ",".join("?" for _ in normalized_ids)
                where.append(f"sn.session_id IN ({conversation_placeholders})")
                args.extend(normalized_ids)
            rows = self._conn.execute(
                f"""
                SELECT CAST(sn.node_id AS TEXT)
                FROM {table} t
                JOIN summary_nodes sn ON CAST(sn.node_id AS TEXT) = t.id
                WHERE {' AND '.join(where)}
                """,
                args,
            ).fetchall()
            allowed = {str(row[0]) for row in rows}
            if source:
                allowed &= self._source_allowed_ids(table, source)
        return [
            index
            for index, embedded_id in enumerate(embedded_ids)
            if str(embedded_id) in allowed
        ]

    def knn(
        self,
        query_vec: Sequence[float],
        k: int = 50,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        conversation_ids: Sequence[str] | None = None,
        source: str | None = None,
    ) -> KNNResult:
        k = int(k)
        if k <= 0:
            return KNNResult(coverage="none")
        profile = (
            self._resolve_profile(str(model))
            if model is not None
            else self._current_profile()
        )
        if profile is None:
            return KNNResult(coverage="none")
        identity = str(profile["identity_hash"])
        dim = int(profile["dim"])
        query = self._normalized(query_vec, expected_dim=dim)
        _, row_count = self._vector_state(identity)
        if row_count == 0:
            return KNNResult(coverage="none")

        try:
            numpy = _load_numpy()
        except ImportError:
            numpy = None

        if numpy is not None:
            rowids, embedded_ids, kinds, matrix = self._numpy_rows(
                numpy,
                identity,
                dim,
            )
            filtered_indexes = self._filtered_candidate_indexes(
                embedded_ids,
                since=since,
                until=until,
                conversation_ids=conversation_ids,
                source=source,
            )
            rowids = [rowids[index] for index in filtered_indexes]
            embedded_ids = [embedded_ids[index] for index in filtered_indexes]
            kinds = [kinds[index] for index in filtered_indexes]
            matrix = matrix[filtered_indexes]
            query_array = numpy.asarray(query, dtype=numpy.float32)
            scores = matrix @ query_array
            coverage = "full"
        else:
            rowids, embedded_ids, kinds, vectors = self._bounded_rows(
                identity,
                dim,
            )
            filtered_indexes = self._filtered_candidate_indexes(
                embedded_ids,
                since=since,
                until=until,
                conversation_ids=conversation_ids,
                source=source,
            )
            rowids = [rowids[index] for index in filtered_indexes]
            embedded_ids = [embedded_ids[index] for index in filtered_indexes]
            kinds = [kinds[index] for index in filtered_indexes]
            vectors = [vectors[index] for index in filtered_indexes]
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
            k,
        )
        return KNNResult(candidates, coverage=coverage)

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
