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
import struct
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from .config import LCMConfig
from .db_bootstrap import (
    configure_connection,
    ensure_embedding_tables,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
    verify_embedding_schema,
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
_SOURCE_LINEAGE_WORK_LIMIT = 4096


class _UnverifiableProvenance(RuntimeError):
    """A requested provenance filter could not be checked completely."""


def _require_supported_identity(identity: "EmbeddingIdentity") -> None:
    """Reject profile representations this implementation cannot encode."""
    unsupported = []
    if identity.dtype != _VECTOR_DTYPE:
        unsupported.append(f"dtype={identity.dtype!r}")
    if identity.byteorder != _VECTOR_BYTEORDER:
        unsupported.append(f"byteorder={identity.byteorder!r}")
    if identity.task != _DEFAULT_TASK:
        unsupported.append(f"task={identity.task!r}")
    if unsupported:
        raise ValueError(
            "unsupported embedding representation: "
            + ", ".join(unsupported)
            + "; supported representation is float32/little/summary"
        )


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Immutable canonical identity captured before provider execution."""

    provider: str
    model_name: str
    revision: str
    dim: int
    dtype: str
    byteorder: str
    task: str

    @classmethod
    def canonical(
        cls,
        provider: str,
        model_name: str,
        revision: str,
        dim: int,
        dtype: str,
        byteorder: str,
        task: str,
    ) -> "EmbeddingIdentity":
        return cls(
            provider=str(provider).strip().lower(),
            model_name=str(model_name).strip(),
            revision=str(revision).strip(),
            dim=int(dim),
            dtype=str(dtype).strip().lower(),
            byteorder=str(byteorder).strip().lower(),
            task=str(task).strip().lower(),
        )

    @property
    def identity_hash(self) -> str:
        canonical = "\x1f".join(
            [
                self.provider,
                self.model_name,
                self.revision,
                str(self.dim),
                self.dtype,
                self.byteorder,
                self.task,
            ]
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    identity = EmbeddingIdentity.canonical(
        provider, model_name, revision, dim, dtype, byteorder, task
    )
    return identity.identity_hash


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
        reason: str | None = None,
    ) -> None:
        super().__init__(rows)
        self.coverage = coverage
        self.reason = reason


class EmbeddingPublishOutcome(str, Enum):
    """Typed result for the post-provider publication CAS."""

    PUBLISHED = "published"
    OWNERSHIP_LOST = "ownership_lost"
    IDENTITY_SUPERSEDED = "identity_superseded"


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
        # Cache key: (identity_hash, data_version, candidate_ids). The durable
        # per-identity ``data_version`` counter is bumped inside every vector
        # write/delete transaction, so a cross-process write invalidates this
        # cache even when max_rowid and row_count are unchanged.
        self._matrix_cache: dict[
            tuple[str, int, tuple[str, ...]],
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
        """Materialize (and verify) the opt-in embedding tables on VectorStore use.

        Constructing a VectorStore means embeddings are in use; the core
        migration path leaves these tables uncreated so a disabled install
        stays at schema_version 5 with none of them. Creation is idempotent
        (CREATE TABLE/INDEX IF NOT EXISTS).

        The ``embeddings_v1`` marker is NOT trusted on its own: it can be set
        while a table or index is absent (e.g. one was dropped after the marker
        was written). So the required tables/indexes are re-ensured and then
        VERIFIED every init — a set marker over a missing table is repaired
        rather than believed. Only a fully-materialized schema keeps the marker.
        """
        with self._write_lock:
            ensure_embedding_tables(self._conn)
            errors = verify_embedding_schema(self._conn)
            if errors:
                # Should be unreachable: the ensure above is a superset of the
                # verified set. Re-ensure once more and fail loudly if a
                # required object still cannot be created, rather than marking a
                # broken schema complete.
                ensure_embedding_tables(self._conn)
                errors = verify_embedding_schema(self._conn)
                if errors:
                    raise sqlite3.OperationalError(
                        "embedding schema incompatible after ensure: "
                        + "; ".join(errors)
                    )
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
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE identity_hash = ?
            """,
            (str(identity_hash),),
        ).fetchone()

    def _resolve_profile(
        self, model_name: str, provider: str | None = None
    ) -> sqlite3.Row | None:
        """Resolve a model name to its profile, preferring the current/active one.

        Multiple identities can share a ``model_name`` (e.g. the same model
        served by two providers). When ``provider`` is given the resolution is
        constrained to that provider too, so the read path selects vectors that
        match the *configured* provider identity rather than whichever provider
        happens to be active for the bare model name — switching provider A→B
        for the same model no longer scores a B-embedded query against A's
        vectors. With no ``provider`` the bare-name active identity is used
        (recording targets the operator's current selection).
        """
        model_name = str(model_name).strip()
        if provider is not None:
            provider = str(provider).strip().lower()
            return self._conn.execute(
                """
                SELECT identity_hash, model_name, provider, revision, dim, dtype,
                       byteorder, task, registered_at, active, archived_at, data_version
                FROM lcm_embedding_profile
                WHERE model_name = ? AND provider = ?
                ORDER BY (active = 1 AND archived_at IS NULL) DESC,
                         registered_at DESC, identity_hash DESC
                LIMIT 1
                """,
                (model_name, provider),
            ).fetchone()
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE model_name = ?
            ORDER BY (active = 1 AND archived_at IS NULL) DESC,
                     registered_at DESC, identity_hash DESC
            LIMIT 1
            """,
            (model_name,),
        ).fetchone()

    @staticmethod
    def _as_node_id(embedded_id: str) -> int | None:
        """Coerce an embedded id to the integer ``summary_nodes.node_id`` PK.

        ``node_id`` is an ``INTEGER PRIMARY KEY``; binding the integer lets the
        lookup use the PK/rowid index instead of a ``CAST(node_id AS TEXT)``
        full scan. A non-integer id can never match a node, so it maps to None.
        """
        try:
            return int(str(embedded_id))
        except (TypeError, ValueError):
            return None

    def _table_columns(self, table: str) -> set[str]:
        return {
            str(row[1])
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _current_profile(self) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """
        ).fetchone()

    @staticmethod
    def _identity_from_profile(profile: sqlite3.Row) -> EmbeddingIdentity:
        identity = EmbeddingIdentity.canonical(
            profile["provider"],
            profile["model_name"],
            profile["revision"],
            profile["dim"],
            profile["dtype"],
            profile["byteorder"],
            profile["task"],
        )
        if identity.identity_hash != str(profile["identity_hash"]):
            raise ValueError("embedding profile identity hash does not match its fields")
        _require_supported_identity(identity)
        return identity

    def capture_identity(
        self, model_name: str | None = None, *, provider: str | None = None
    ) -> EmbeddingIdentity:
        """Capture the full immutable identity before provider execution."""
        profile = (
            self._resolve_profile(model_name, provider=provider)
            if model_name is not None
            else self._current_profile()
        )
        if profile is None:
            target = model_name if model_name is not None else "active"
            raise ValueError(f"embedding profile is not registered: {target}")
        return self._identity_from_profile(profile)

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
        canonical = EmbeddingIdentity.canonical(
            provider, model_name, revision, dim, dtype, byteorder, task
        )
        if not canonical.model_name:
            raise ValueError("model_name must not be empty")
        if not canonical.provider:
            raise ValueError("provider must not be empty")
        if canonical.dim < 1 or canonical.dim > 4096:
            raise ValueError("embedding dimension must be between 1 and 4096")
        _require_supported_identity(canonical)

        identity = canonical.identity_hash
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
                        identity,
                        canonical.provider,
                        canonical.model_name,
                        canonical.revision,
                        canonical.dim,
                        canonical.dtype,
                        canonical.byteorder,
                        canonical.task,
                        self._now(),
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
        *,
        identity: EmbeddingIdentity,
    ) -> None:
        with self._write_transaction():
            profile = self._validate_embedding_write(identity, model=model)
            self._write_embedding_row(
                str(embedded_id), kind, vec, identity=identity, profile=profile
            )

    def publish_embedding_under_lease(
        self,
        embedded_id: str,
        kind: str,
        model: str,
        vec: Sequence[float],
        *,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
        request_id: str,
    ) -> EmbeddingPublishOutcome:
        """Atomically revalidate ownership and publish one accepted vector.

        The metadata owner/generation check, active captured identity check,
        matching dispatched-row check, vector/meta write, data-version bump,
        and in-flight clear all share one ``BEGIN IMMEDIATE`` transaction. A
        superseded worker therefore performs none of these mutations.
        """
        embedded_id = str(embedded_id)
        with self._write_transaction():
            owner = self._conn.execute(
                """
                SELECT 1
                FROM metadata
                WHERE key = ?
                  AND json_extract(value, '$.owner') = ?
                  AND CAST(json_extract(value, '$.generation') AS INTEGER) = ?
                """,
                (claim_key, str(lease_id), int(generation)),
            ).fetchone()
            inflight = self._conn.execute(
                """
                SELECT 1
                FROM lcm_embedding_backfill_inflight
                WHERE embedded_id = ? AND identity_hash = ?
                  AND lease_id = ? AND generation = ?
                  AND request_id = ? AND state = 'dispatched'
                """,
                (
                    embedded_id,
                    identity.identity_hash,
                    str(lease_id),
                    int(generation),
                    str(request_id),
                ),
            ).fetchone()
            if owner is None or inflight is None:
                return EmbeddingPublishOutcome.OWNERSHIP_LOST
            profile = self._validate_embedding_write(identity, model=model)
            if int(profile["active"] or 0) != 1 or profile["archived_at"] is not None:
                transitioned = self._conn.execute(
                    """
                    UPDATE lcm_embedding_backfill_inflight
                    SET state = 'uncertain', updated_at = ?,
                        last_error = 'embedding identity superseded after remote acceptance'
                    WHERE identity_hash = ? AND lease_id = ? AND generation = ?
                      AND request_id = ? AND state = 'dispatched'
                    """,
                    (
                        datetime.now(timezone.utc).timestamp(),
                        identity.identity_hash,
                        str(lease_id),
                        int(generation),
                        str(request_id),
                    ),
                )
                if int(transitioned.rowcount or 0) < 1:
                    return EmbeddingPublishOutcome.OWNERSHIP_LOST
                released = self._conn.execute(
                    """
                    DELETE FROM metadata
                    WHERE key = ?
                      AND json_extract(value, '$.owner') = ?
                      AND CAST(json_extract(value, '$.generation') AS INTEGER) = ?
                    """,
                    (claim_key, str(lease_id), int(generation)),
                )
                if int(released.rowcount or 0) != 1:
                    raise sqlite3.OperationalError(
                        "identity supersession lost lease release ownership"
                    )
                return EmbeddingPublishOutcome.IDENTITY_SUPERSEDED
            self._write_embedding_row(
                embedded_id, kind, vec, identity=identity, profile=profile
            )
            cleared = self._conn.execute(
                """
                DELETE FROM lcm_embedding_backfill_inflight
                WHERE embedded_id = ? AND identity_hash = ?
                  AND lease_id = ? AND generation = ?
                  AND request_id = ? AND state = 'dispatched'
                """,
                (
                    embedded_id,
                    identity.identity_hash,
                    str(lease_id),
                    int(generation),
                    str(request_id),
                ),
            )
            if cleared.rowcount != 1:
                raise sqlite3.OperationalError(
                    "embedding publication lost its in-flight ownership"
                )
            return EmbeddingPublishOutcome.PUBLISHED

    def _validate_embedding_write(
        self, identity: EmbeddingIdentity, *, model: str
    ) -> sqlite3.Row:
        model = str(model)
        if not isinstance(identity, EmbeddingIdentity):
            raise TypeError("identity must be an EmbeddingIdentity captured before provider work")
        canonical_identity = EmbeddingIdentity.canonical(
            identity.provider,
            identity.model_name,
            identity.revision,
            identity.dim,
            identity.dtype,
            identity.byteorder,
            identity.task,
        )
        if identity != canonical_identity:
            raise ValueError("embedding identity fields are not canonical")
        _require_supported_identity(identity)
        expected_identity = identity.identity_hash
        profile = self._profile_by_identity(expected_identity)
        if profile is None:
            raise ValueError(f"embedding profile is not registered: {expected_identity}")
        registered_identity = self._identity_from_profile(profile)
        if registered_identity != identity:
            raise ValueError(
                "captured embedding identity does not match registered profile"
            )
        if str(profile["model_name"]) != model.strip():
            raise ValueError(
                "embedding model does not match captured identity: "
                f"{model} != {profile['model_name']}"
            )
        return profile

    def _write_embedding_row(
        self,
        embedded_id: str,
        kind: str,
        vec: Sequence[float],
        *,
        identity: EmbeddingIdentity,
        profile: sqlite3.Row,
    ) -> None:
        if kind != "summary":
            raise ValueError("embedded kind must be 'summary'")
        identity_hash = identity.identity_hash
        normalized = self._normalized(vec, expected_dim=int(profile["dim"]))
        # Persist an explicit little-endian float32 wire format. Native
        # ``array('f')`` bytes would be mislabeled on a big-endian host.
        packed = struct.pack(f"<{len(normalized)}f", *normalized)
        numeric_id = self._as_node_id(embedded_id)
        summary = (
            self._conn.execute(
                """
                SELECT source_token_count
                FROM summary_nodes
                WHERE node_id = ?
                """,
                (numeric_id,),
            ).fetchone()
            if numeric_id is not None
            else None
        )
        if summary is None:
            raise ValueError(f"summary node does not exist: {embedded_id}")
        embedded_at = self._now()
        self._conn.execute(
            "DELETE FROM lcm_embedding_vectors "
            "WHERE embedded_id = ? AND identity_hash = ?",
            (embedded_id, identity_hash),
        )
        self._conn.execute(
            "DELETE FROM lcm_embedding_meta WHERE embedded_id = ? "
            "AND embedded_kind = ? AND identity_hash = ?",
            (embedded_id, kind, identity_hash),
        )
        self._conn.execute(
            "INSERT INTO lcm_embedding_vectors(embedded_id, identity_hash, vec) "
            "VALUES(?, ?, ?)",
            (embedded_id, identity_hash, packed),
        )
        self._conn.execute(
            """
            INSERT INTO lcm_embedding_meta(
                embedded_id, embedded_kind, identity_hash, embedded_at,
                source_token_count, archived
            ) VALUES(?, ?, ?, ?, ?, 0)
            """,
            (
                embedded_id,
                kind,
                identity_hash,
                embedded_at,
                int(summary["source_token_count"] or 0),
            ),
        )
        self._bump_data_version(identity_hash)

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
        """Load ids into a unique per-call temp table in bounded chunks for JOINs.

        Avoids the ``WHERE id IN (?, ?, ...)`` host-parameter cap
        (SQLITE_MAX_VARIABLE_NUMBER) that failed at ~33k ids: ``executemany``
        binds one row at a time, so an arbitrary number of ids can be staged.

        The table name is unique per call (the store's connection is shared
        across threads, ``check_same_thread=False``); a single reused name meant
        two concurrent ``knn()`` calls scribbled over each other's candidate set.
        The table is dropped in ``finally`` so it never outlives the call.
        """
        table = f"_lcm_id_scratch_{uuid.uuid4().hex}"
        self._conn.execute(f"CREATE TEMP TABLE {table}(id TEXT PRIMARY KEY)")
        try:
            for offset in range(0, len(ids), _ID_INSERT_CHUNK):
                rows = [
                    (str(value),)
                    for value in ids[offset:offset + _ID_INSERT_CHUNK]
                ]
                self._conn.executemany(
                    f"INSERT OR IGNORE INTO {table}(id) VALUES(?)",
                    rows,
                )
            yield table
        finally:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")

    @contextmanager
    def _optional_temp_id_table(
        self, ids: Sequence[str] | None
    ) -> Iterator[str | None]:
        if ids is None:
            yield None
            return
        normalized = [str(value) for value in ids]
        with self._temp_id_table(normalized) as table:
            yield table

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
                if self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='lcm_embedding_backfill_inflight'"
                ).fetchone() is not None:
                    self._conn.execute(
                        f"DELETE FROM lcm_embedding_backfill_inflight "
                        f"WHERE embedded_id IN (SELECT id FROM {table})"
                    )
            # Purge spans every identity keyed by these ids; bump all data
            # versions so any cached matrix that referenced a removed vector is
            # invalidated. Over-invalidation across identities is safe.
            if cur.rowcount:
                self._bump_all_data_versions()
        return int(cur.rowcount or 0)

    @staticmethod
    def purge_embedding_batch_on_connection(
        conn: sqlite3.Connection, node_ids: Sequence[int | str]
    ) -> int:
        """Purge one already-bounded node batch in the caller's transaction."""
        unique_ids = list(dict.fromkeys(str(node_id) for node_id in node_ids))
        if not unique_ids:
            return 0
        if len(unique_ids) > 256:
            raise ValueError("embedding purge batch exceeds 256 rows")
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ("
                "'lcm_embedding_vectors','lcm_embedding_meta',"
                "'lcm_embedding_profile','lcm_embedding_backfill_inflight')"
            ).fetchall()
        }
        if not {
            "lcm_embedding_vectors",
            "lcm_embedding_meta",
            "lcm_embedding_profile",
        } <= tables:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        cur = conn.execute(
            f"DELETE FROM lcm_embedding_vectors "
            f"WHERE embedded_id IN ({placeholders})",
            unique_ids,
        )
        conn.execute(
            f"DELETE FROM lcm_embedding_meta "
            f"WHERE embedded_id IN ({placeholders})",
            unique_ids,
        )
        if "lcm_embedding_backfill_inflight" in tables:
            conn.execute(
                f"DELETE FROM lcm_embedding_backfill_inflight "
                f"WHERE embedded_id IN ({placeholders})",
                unique_ids,
            )
        if cur.rowcount:
            conn.execute(
                "UPDATE lcm_embedding_profile "
                "SET data_version = data_version + 1"
            )
        return int(cur.rowcount or 0)

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
        embedded_ids: Sequence[str],
    ) -> tuple[list[int], list[str], list[str], Any]:
        """Load only the SQL-bounded candidate set into a NumPy matrix."""
        with self._cache_lock:
            data_version = self._data_version(identity_hash)
            key = (identity_hash, data_version, tuple(str(value) for value in embedded_ids))
            cached = self._matrix_cache.get(key)
            if cached is not None:
                return cached
            rowids, loaded_ids, kinds, raw_vectors = self._load_vectors_for_ids(
                identity_hash, dim, embedded_ids
            )
            matrix = (
                numpy.asarray(raw_vectors, dtype=numpy.float32)
                if raw_vectors
                else numpy.empty((0, dim), dtype=numpy.float32)
            )
            loaded = (rowids, loaded_ids, kinds, matrix)
            self._matrix_cache.clear()
            self._matrix_cache[key] = loaded
            return loaded

    def _bounded_candidate_ids(
        self,
        identity_hash: str,
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
        source: str | None,
        limit: int,
    ) -> list[str]:
        """Enumerate at most ``limit`` live candidate ids, most-recent first.

        The candidate enumeration itself is bounded at the SQL layer: the
        column filters (recency window, conversation, suppression) are applied
        in the ``WHERE`` clause and a hard ``LIMIT`` caps the result, so neither
        the SQL result set nor host memory materializes the whole corpus (a
        100-row corpus with bound 10 loads ~10 ids, not 100). Ordering is by
        ``embedded_at DESC`` — served by
        ``idx_lcm_embedding_meta_identity_embedded_at`` — matching the vector
        write order rather than rowid.

        Column filters are enforced BEFORE the bound (they are in ``WHERE``), so
        a filtered match inside the most-recent window is never dropped; only
        matches beyond the bounded window are out of scope, which is exactly
        what ``coverage='bounded'`` signals. The source-lineage filter is a
        recursive descendant walk, so it is applied to the already-bounded id
        set afterward (and fails closed when provenance is unverifiable).
        """
        if limit <= 0:
            return []
        if conversation_ids is not None and not list(conversation_ids):
            return []
        columns = self._table_columns("summary_nodes")
        if not columns:
            return []
        if (since is not None or until is not None) and "latest_at" not in columns:
            return []
        recency_expr = "sn.latest_at"
        where = ["m.identity_hash = ?", "m.archived = 0"]
        args: list[object] = [str(identity_hash)]
        if "suppressed_at" in columns:
            where.append("sn.suppressed_at IS NULL")
        if since is not None:
            where.append(f"{recency_expr} >= ?")
            args.append(float(since))
        if until is not None:
            where.append(f"{recency_expr} <= ?")
            args.append(float(until))
        args.append(int(limit))
        with self._optional_temp_id_table(conversation_ids) as conversation_table:
            conversation_join = (
                f"JOIN {conversation_table} c ON c.id = sn.session_id"
                if conversation_table is not None
                else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT m.embedded_id
                FROM lcm_embedding_meta m
                JOIN summary_nodes sn ON sn.node_id = CAST(m.embedded_id AS INTEGER)
                {conversation_join}
                WHERE {' AND '.join(where)}
                ORDER BY m.embedded_at DESC, m.embedded_id DESC
                LIMIT ?
                """,
                args,
            ).fetchall()
        bounded_ids = [str(row[0]) for row in rows]
        if source and bounded_ids:
            with self._temp_id_table(bounded_ids) as table:
                allowed = self._source_allowed_ids(table, source)
            bounded_ids = [
                embedded_id for embedded_id in bounded_ids if embedded_id in allowed
            ]
        return bounded_ids

    def _load_vectors_for_ids(
        self,
        identity_hash: str,
        dim: int,
        embedded_ids: Sequence[str],
    ) -> tuple[list[int], list[str], list[str], list[list[float]]]:
        """Load vectors (pure-Python) for a bounded, already-filtered id set.

        Ordering is irrelevant here — the caller has already applied the recency
        bound and ``_ranked`` re-sorts by score — so a temp-table JOIN is used.
        """
        rowids: list[int] = []
        out_ids: list[str] = []
        kinds: list[str] = []
        vectors: list[list[float]] = []
        if not embedded_ids:
            return rowids, out_ids, kinds, vectors
        with self._temp_id_table(embedded_ids) as table:
            rows = self._conn.execute(
                f"""
                SELECT v.rowid, v.embedded_id, m.embedded_kind, v.vec
                FROM {table} t
                JOIN lcm_embedding_vectors v
                  ON v.embedded_id = t.id AND v.identity_hash = ?
                JOIN lcm_embedding_meta m
                  ON m.embedded_id = v.embedded_id
                 AND m.identity_hash = v.identity_hash
                WHERE m.archived = 0
                """,
                (identity_hash,),
            ).fetchall()
        for row in rows:
            try:
                blob = bytes(row["vec"])
                if len(blob) != dim * 4:
                    continue
                vector = struct.unpack(f"<{dim}f", blob)
            except (TypeError, ValueError, struct.error):
                continue
            rowids.append(int(row["rowid"]))
            out_ids.append(str(row["embedded_id"]))
            kinds.append(str(row["embedded_kind"]))
            vectors.append(list(vector))
        return rowids, out_ids, kinds, vectors

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

        # A VectorStore-only worker DB may predate MessageStore's source repair
        # (``_ensure_source_column``) and lack ``messages.source`` entirely.
        # Provenance is then unverifiable, so the source filter must FAIL CLOSED:
        # return no allowed ids rather than treating "can't check" as "all
        # allowed". Failing open let a source-filtered query surface a legacy
        # summary whose source could not be confirmed; an empty allowed set
        # instead yields no false-positive hit (and lets the caller degrade to
        # full_text), which is the safe direction for an unverifiable filter.
        message_columns = {
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "source" not in message_columns:
            return set()

        normalized_source = _normalize_source_value(source)
        legacy_blank_clause = _legacy_blank_source_clause("m.source")
        rows = self._conn.execute(
            f"""
            WITH RECURSIVE walk(root_id, source_type, source_id) AS (
                SELECT t.id, sn.source_type, CAST(j.value AS INTEGER)
                FROM {table} t
                JOIN summary_nodes sn ON sn.node_id = CAST(t.id AS INTEGER)
                JOIN json_each(sn.source_ids) j

                UNION ALL

                SELECT w.root_id, child.source_type, CAST(j.value AS INTEGER)
                FROM walk w
                JOIN summary_nodes child
                  ON w.source_type = 'nodes' AND child.node_id = w.source_id
                JOIN json_each(child.source_ids) j
                LIMIT ?
            ), matched AS (
                SELECT DISTINCT w.root_id
                FROM walk w
                JOIN messages m
                  ON w.source_type = 'messages' AND m.store_id = w.source_id
                WHERE CASE
                        WHEN ? = ? THEN (m.source = ? OR {legacy_blank_clause})
                        ELSE m.source = ?
                      END
            ), work_count AS (
                SELECT COUNT(*) AS visited FROM walk
            )
            SELECT root_id, 0 AS overflow FROM matched
            UNION ALL
            SELECT NULL, 1 FROM work_count WHERE visited > ?
            """,
            (
                _SOURCE_LINEAGE_WORK_LIMIT + 1,
                normalized_source,
                _UNKNOWN_SOURCE,
                normalized_source,
                normalized_source,
                _SOURCE_LINEAGE_WORK_LIMIT,
            ),
        ).fetchall()
        if any(int(row[1]) for row in rows):
            raise _UnverifiableProvenance(
                "source lineage exceeded the bounded verification budget"
            )
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
        columns = self._table_columns("summary_nodes")
        if (since is not None or until is not None) and "latest_at" not in columns:
            return []
        recency_expr = "sn.latest_at"
        with self._temp_id_table(unique_ids) as table:
            where = ["1 = 1"]
            args: list[object] = []
            if "suppressed_at" in columns:
                where.append("sn.suppressed_at IS NULL")
            if since is not None:
                where.append(f"{recency_expr} >= ?")
                args.append(float(since))
            if until is not None:
                where.append(f"{recency_expr} <= ?")
                args.append(float(until))
            with self._optional_temp_id_table(conversation_ids) as conversation_table:
                conversation_join = (
                    f"JOIN {conversation_table} c ON c.id = sn.session_id"
                    if conversation_table is not None
                    else ""
                )
                rows = self._conn.execute(
                    f"""
                    SELECT t.id
                    FROM {table} t
                    JOIN summary_nodes sn ON sn.node_id = CAST(t.id AS INTEGER)
                    {conversation_join}
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
        provider: str | None = None,
    ) -> KNNResult:
        k = int(k)
        if k <= 0:
            return KNNResult(coverage="none")
        summary_columns = self._table_columns("summary_nodes")
        if conversation_ids is not None and "session_id" not in summary_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if (since is not None or until is not None) and "latest_at" not in summary_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if source is not None and "source" not in self._table_columns("messages"):
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        # Resolve by the FULL configured identity (provider + model) when the
        # caller knows its provider, so a config switch A→B for the same model
        # name scores the query against B's vectors, not whichever provider is
        # active. An unbackfilled identity resolves to None → coverage "none" →
        # the caller degrades to full_text.
        profile = (
            self._resolve_profile(str(model), provider=provider)
            if model is not None
            else self._current_profile()
        )
        if profile is None:
            return KNNResult(coverage="none")
        identity = str(profile["identity_hash"])
        dim = int(profile["dim"])
        query = self._normalized(query_vec, expected_dim=dim)
        try:
            numpy = _load_numpy()
        except ImportError:
            numpy = None

        limit = max(0, self.bounded_scan_rows)
        # Probe at most bound+1 through the indexed candidate query. This
        # determines full-vs-bounded coverage without COUNT(*) scanning the
        # entire identity on every request.
        try:
            probed_ids = self._bounded_candidate_ids(
                identity,
                since=since,
                until=until,
                conversation_ids=conversation_ids,
                source=source,
                limit=limit + 1,
            )
        except _UnverifiableProvenance:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if not probed_ids:
            return KNNResult(coverage="none")
        bounded_ids = probed_ids[:limit]
        candidate_coverage = (
            "bounded"
            if source is not None or len(probed_ids) > limit
            else "full"
        )

        if numpy is not None:
            rowids, embedded_ids, kinds, matrix = self._numpy_rows(
                numpy,
                identity,
                dim,
                bounded_ids,
            )
            query_array = numpy.asarray(query, dtype=numpy.float32)
            scores = matrix @ query_array
            coverage = candidate_coverage
        else:
            # Bound the candidate enumeration at the SQL layer: the column
            # filters + ORDER BY embedded_at DESC + LIMIT run inside SQLite, so
            # neither the result set nor host memory enumerates the whole
            # corpus. Filters live in the WHERE clause (applied before the
            # bound), so a filtered match inside the bounded window is not lost;
            # only the source-lineage walk runs on the already-bounded set.
            rowids, embedded_ids, kinds, vectors = self._load_vectors_for_ids(
                identity,
                dim,
                bounded_ids,
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
