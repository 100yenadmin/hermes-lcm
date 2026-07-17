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
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from .config import LCMConfig
from .db_bootstrap import (
    configure_connection,
    ensure_chunk_tables,
    ensure_embedding_tables,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
    verify_chunk_schema,
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
_CHUNK_TASK = "chunk"
# Tasks this store can encode. Summary and chunk corpora coexist in the shared
# profile table, each keyed by its own canonical identity (task is part of the
# identity hash), so their vectors never collide.
_SUPPORTED_TASKS = frozenset({_DEFAULT_TASK, _CHUNK_TASK})

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
    if identity.task not in _SUPPORTED_TASKS:
        unsupported.append(f"task={identity.task!r}")
    if unsupported:
        raise ValueError(
            "unsupported embedding representation: "
            + ", ".join(unsupported)
            + "; supported representation is float32/little/summary|chunk"
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
        scanned: int | None = None,
        total: int | None = None,
    ) -> None:
        super().__init__(rows)
        self.coverage = coverage
        self.reason = reason
        # Bounded-coverage provenance: how many of the corpus's live vectors were
        # actually scored (``scanned``) out of the total live for the identity
        # (``total``), so a caller can surface partial-archive coverage (SCAN-1).
        self.scanned = scanned
        self.total = total


class EmbeddingPublishOutcome(str, Enum):
    """Typed result for the post-provider publication CAS."""

    PUBLISHED = "published"
    OWNERSHIP_LOST = "ownership_lost"
    IDENTITY_SUPERSEDED = "identity_superseded"


@dataclass(frozen=True)
class BatchPublishResult:
    """Per-row result from a batched publication.

    ``outcome`` carries the same CAS verdict a single-row publish would return;
    ``error`` is set (and ``outcome`` is ``None``) when that row's publish raised
    — its savepoint was rolled back, so the row stayed in-flight while its
    batch siblings that already succeeded remain committed. The caller reproduces
    exactly the accounting it applied on the single-row path.
    """

    embedded_id: str
    outcome: EmbeddingPublishOutcome | None
    error: Exception | None = None


class VectorStore:
    """SQLite-backed store for normalized summary embedding vectors."""

    # Opt-in marker read by the retrieval-core pool: only the genuine store is
    # long-lived/poolable (test fakes injected as ``vector_store_cls`` are not).
    _supports_pooling = True
    # Per-store matrix-cache ceiling. A bounded LRU (not clear-on-every-miss) so
    # back-to-back recalls over an unchanged corpus reuse the loaded candidate
    # matrix across BOTH arms and across calls once the store is pooled, without
    # letting distinct candidate sets grow the cache without bound (sprint-opt-6).
    _MATRIX_CACHE_MAX_ENTRIES = 4

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
        # Nesting depth for _write_transaction. The outermost entry owns the
        # BEGIN IMMEDIATE/COMMIT (one fsync); a re-entrant inner entry (e.g. a
        # single-row publish invoked from publish_*_batch_under_lease) rides the
        # open transaction under a SAVEPOINT so its failure rolls back only its
        # own row, never the sibling rows already written in the same batch.
        self._txn_depth = 0
        # Cache key: (identity_hash, data_version, candidate_ids). The durable
        # per-identity ``data_version`` counter is bumped inside every vector
        # write/delete transaction, so a cross-process write invalidates this
        # cache even when max_rowid and row_count are unchanged.
        self._matrix_cache: "OrderedDict[tuple[str, int, tuple[str, ...]], tuple[list[int], list[str], list[str], Any]]" = OrderedDict()
        # Separate cache for chunk-corpus matrices: chunk identities never
        # collide with summary identities (task is part of the hash), but the
        # loaders differ (chunk vectors join messages, summary vectors join
        # summary_nodes), so the two corpora keep independent matrix caches.
        self._chunk_matrix_cache: "OrderedDict[tuple[str, int, tuple[str, ...]], tuple[list[int], list[str], list[str], Any]]" = OrderedDict()
        self._chunk_schema_ready = False
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
            # Skip the marker write + commit when it is already stamped and the
            # schema verified clean -- re-writing it every construction is an
            # otherwise-needless write-transaction contending with real writers
            # (sprint-opt-1). CREATE-IF-NOT-EXISTS and verify above are read-only
            # no-ops on an already-materialized schema.
            if not self._migration_step_present("embeddings_v1"):
                mark_migration_step_complete(self._conn, "embeddings_v1")
                self._conn.commit()

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        # The connection is in autocommit mode (isolation_level=None), so a
        # multi-statement write must open its own explicit transaction to stay
        # atomic — the data_version bump has to commit with the vector write it
        # accompanies, never separately.
        #
        # Re-entrant: the outermost caller owns the BEGIN IMMEDIATE + COMMIT (one
        # fsync); a nested caller opens a SAVEPOINT instead so a batch of per-row
        # publications shares a single transaction. A nested failure rolls back
        # to its own savepoint and re-raises, leaving the earlier rows of the
        # batch intact for the outer COMMIT.
        try:
            with self._write_lock, self._cache_lock:
                if self._txn_depth > 0:
                    savepoint = f"lcm_pub_sp_{self._txn_depth}"
                    self._conn.execute(f"SAVEPOINT {savepoint}")
                    self._txn_depth += 1
                    try:
                        yield
                        self._conn.execute(f"RELEASE {savepoint}")
                    except BaseException:
                        self._conn.execute(f"ROLLBACK TO {savepoint}")
                        self._conn.execute(f"RELEASE {savepoint}")
                        raise
                    finally:
                        self._txn_depth -= 1
                    return
                self._conn.execute("BEGIN IMMEDIATE")
                self._txn_depth = 1
                try:
                    yield
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
                finally:
                    self._txn_depth = 0
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
        # The current SUMMARY profile. Task-scoped so a coexisting active chunk
        # profile can never be mistaken for the summary corpus's current profile.
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL AND task = ?
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """,
            (_DEFAULT_TASK,),
        ).fetchone()

    def _resolve_chunk_profile(
        self, model_name: str, *, provider: str | None = None
    ) -> sqlite3.Row | None:
        """Resolve a profile by model (+optional provider) within task='chunk'."""
        base = (
            "SELECT identity_hash, model_name, provider, revision, dim, dtype, "
            "byteorder, task, registered_at, active, archived_at, data_version "
            "FROM lcm_embedding_profile WHERE model_name = ? AND task = ?"
        )
        order = (
            " ORDER BY (active = 1 AND archived_at IS NULL) DESC, "
            "registered_at DESC, identity_hash DESC LIMIT 1"
        )
        if provider is not None:
            return self._conn.execute(
                base + " AND provider = ?" + order,
                (str(model_name), _CHUNK_TASK, str(provider).strip().lower()),
            ).fetchone()
        return self._conn.execute(
            base + order, (str(model_name), _CHUNK_TASK)
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
            # Selection of the "current" profile is by identity WITHIN A TASK:
            # only the just-registered identity stays active for its task; other
            # profiles of the SAME task are deactivated (retained so their
            # vectors survive a later switch back). Profiles of a DIFFERENT task
            # are untouched, so the summary and chunk corpora coexist — each has
            # its own active profile.
            self._conn.execute(
                "UPDATE lcm_embedding_profile SET active = 0 "
                "WHERE identity_hash != ? AND task = ?",
                (identity, canonical.task),
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
        """Atomically revalidate ownership and publish one accepted summary vector."""
        embedded_id = str(embedded_id)
        return self._publish_under_lease(
            embedded_id,
            model=model,
            identity=identity,
            claim_key=claim_key,
            lease_id=lease_id,
            generation=generation,
            request_id=request_id,
            write_row=lambda profile: self._write_embedding_row(
                embedded_id, kind, vec, identity=identity, profile=profile
            ),
        )

    def publish_embedding_batch_under_lease(
        self,
        rows: Sequence[tuple[str, str, str, Sequence[float]]],
        *,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
    ) -> list[BatchPublishResult]:
        """Publish an accepted network batch of summary vectors in ONE transaction.

        ``rows`` is a sequence of ``(embedded_id, kind, model, vec)`` in provider
        order. All rows share one ``BEGIN IMMEDIATE`` (a single fsync per network
        batch instead of one per row), yet each still runs the full per-row
        ownership/identity CAS under its own savepoint via the single-row
        ``publish_embedding_under_lease``. Publication stops at the first row that
        is not ``PUBLISHED`` (or that raises), mirroring the single-row loop it
        replaces; the rows already published in that transaction are kept.
        """
        return self._run_publish_batch(
            [
                (
                    str(embedded_id),
                    lambda embedded_id=embedded_id, kind=kind, model=model, vec=vec, request_id=request_id: self.publish_embedding_under_lease(
                        embedded_id,
                        kind,
                        model,
                        vec,
                        identity=identity,
                        claim_key=claim_key,
                        lease_id=lease_id,
                        generation=generation,
                        request_id=request_id,
                    ),
                )
                for embedded_id, kind, model, vec, request_id in rows
            ]
        )

    def _run_publish_batch(
        self, publishers: Sequence[tuple[str, "Any"]]
    ) -> list[BatchPublishResult]:
        """Run per-row publishers inside one write transaction.

        Each entry is ``(embedded_id, publish)`` where ``publish`` is a zero-arg
        call into the corpus-specific single-row publish method (kept as the
        per-row worker so its behaviour — and any test monkeypatch of it — is
        exercised unchanged, now under a savepoint). A raised publisher rolls its
        own savepoint back and is reported as an errored row; the loop then stops
        so the caller can quarantine the request exactly as the single-row path
        did, while the committed siblings survive.
        """
        results: list[BatchPublishResult] = []
        with self._write_transaction():
            for embedded_id, publish in publishers:
                try:
                    outcome = publish()
                except Exception as exc:  # noqa: BLE001 — reported to the caller
                    results.append(
                        BatchPublishResult(embedded_id, outcome=None, error=exc)
                    )
                    break
                results.append(BatchPublishResult(embedded_id, outcome=outcome))
                if outcome is not EmbeddingPublishOutcome.PUBLISHED:
                    break
        return results

    def _publish_under_lease(
        self,
        embedded_id: str,
        *,
        model: str,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
        request_id: str,
        write_row,
    ) -> EmbeddingPublishOutcome:
        """Corpus-agnostic lease publication CAS shared by summary and chunk.

        The metadata owner/generation check, active captured identity check,
        matching dispatched-row check, corpus-specific vector/meta write (via
        ``write_row``), data-version bump, and in-flight clear all share one
        ``BEGIN IMMEDIATE`` transaction. A superseded worker therefore performs
        none of these mutations. Summary and chunk publications share the one
        ``lcm_embedding_backfill_inflight`` table because their embedded_ids are
        disjoint (numeric node ids vs ``store_id:chunk_index``) and their
        identity hashes differ (task is part of the hash).
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
            write_row(profile)
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

    def _migration_step_present(self, step_name: str) -> bool:
        """Read-before-write probe: is this migration marker already stamped?

        Lets schema-ensure skip an otherwise-needless marker re-write + commit on
        every construction (sprint-opt-1); returns False if the state table is
        absent (nothing stamped yet).
        """
        try:
            row = self._conn.execute(
                "SELECT 1 FROM lcm_migration_state WHERE step_name = ? LIMIT 1",
                (str(step_name),),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        return row is not None

    def _count_embedded_vectors(self, identity_hash: str, *, chunk: bool) -> int | None:
        """Cheap single-table COUNT of embedded vectors for one identity.

        Used only to annotate a ``coverage='bounded'`` result with the total
        corpus size (SCAN-1); returns ``None`` if the table is absent. Archival
        is tracked in the meta tables, so this is a total-embedded hint (the
        scanned/total ratio signals partial-archive coverage, not an exact live
        count).
        """
        table = "lcm_chunk_vectors" if chunk else "lcm_embedding_vectors"
        try:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE identity_hash = ?",
                (str(identity_hash),),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return int(row[0]) if row is not None else None

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
                self._matrix_cache.move_to_end(key)  # mark most-recently used
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
            self._matrix_cache[key] = loaded
            while len(self._matrix_cache) > self._MATRIX_CACHE_MAX_ENTRIES:
                self._matrix_cache.popitem(last=False)  # evict oldest
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
        100-row corpus with bound 10 loads ~10 ids, not 100). Ordering is by the
        source summary's ``latest_at DESC`` so a newest-first backfill cannot
        invert the bounded retrieval window when older summaries are embedded
        later.

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
        # Per-row fallback matters, not just column existence: the DAG
        # migration adds ``latest_at`` without backfilling legacy rows, and a
        # bare ``latest_at DESC`` sorts those NULLs last — silently dropping
        # legacy summaries out of the bounded window on upgraded databases.
        recency_expr = (
            "COALESCE(sn.latest_at, sn.created_at)"
            if "latest_at" in columns
            else "sn.created_at"
        )
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
                ORDER BY {recency_expr} DESC, sn.node_id DESC
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
            # filters + ORDER BY source latest_at DESC + LIMIT run inside SQLite, so
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
        scanned = total = None
        if coverage == "bounded":
            scanned = len(bounded_ids)
            total = self._count_embedded_vectors(identity, chunk=False)
        return KNNResult(candidates, coverage=coverage, scanned=scanned, total=total)

    # -- Chunk corpus ------------------------------------------------------
    #
    # The chunk corpus mirrors the summary corpus exactly: identity-hashed
    # profiles (task='chunk') in the shared lcm_embedding_profile table, the
    # same bounded-candidate KNN + coverage contract, the same lease publication
    # CAS. It differs only in its source table (raw messages, keyed by store_id)
    # and its own meta/vectors tables. Chunk schema is materialized lazily on
    # first chunk-corpus use so a summary-only install never creates it.

    def ensure_chunk_schema(self) -> None:
        """Materialize (and verify) the opt-in chunk tables on first chunk use.

        Same discipline as ``_ensure_embedding_schema``: the ``chunk_vectors_v1``
        marker is not trusted on its own; the tables/indexes are re-ensured and
        VERIFIED, and only a fully-materialized schema keeps the marker.
        """
        if self._chunk_schema_ready:
            return
        with self._write_lock:
            ensure_embedding_tables(self._conn)
            ensure_chunk_tables(self._conn)
            errors = verify_chunk_schema(self._conn)
            if errors:
                ensure_chunk_tables(self._conn)
                errors = verify_chunk_schema(self._conn)
                if errors:
                    raise sqlite3.OperationalError(
                        "chunk schema incompatible after ensure: " + "; ".join(errors)
                    )
            # Read-before-write: skip the needless marker re-write + commit when
            # already stamped and verified clean (sprint-opt-1).
            if not self._migration_step_present("chunk_vectors_v1"):
                mark_migration_step_complete(self._conn, "chunk_vectors_v1")
                self._conn.commit()
        self._chunk_schema_ready = True

    def _write_chunk_row(
        self,
        chunk_id: str,
        *,
        store_id: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_estimate: int,
        vec: Sequence[float],
        identity: EmbeddingIdentity,
        profile: sqlite3.Row,
    ) -> None:
        identity_hash = identity.identity_hash
        normalized = self._normalized(vec, expected_dim=int(profile["dim"]))
        packed = struct.pack(f"<{len(normalized)}f", *normalized)
        embedded_at = self._now()
        self._conn.execute(
            "DELETE FROM lcm_chunk_vectors WHERE chunk_id = ? AND identity_hash = ?",
            (chunk_id, identity_hash),
        )
        self._conn.execute(
            "DELETE FROM lcm_chunk_meta WHERE chunk_id = ? AND identity_hash = ?",
            (chunk_id, identity_hash),
        )
        self._conn.execute(
            "INSERT INTO lcm_chunk_vectors(chunk_id, identity_hash, vec) VALUES(?, ?, ?)",
            (chunk_id, identity_hash, packed),
        )
        self._conn.execute(
            """
            INSERT INTO lcm_chunk_meta(
                chunk_id, identity_hash, store_id, chunk_index, char_start,
                char_end, token_estimate, embedded_at, archived
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                chunk_id,
                identity_hash,
                int(store_id),
                int(chunk_index),
                int(char_start),
                int(char_end),
                int(token_estimate),
                embedded_at,
            ),
        )
        self._bump_data_version(identity_hash)

    def record_chunk_embedding(
        self,
        chunk_id: str,
        model: str,
        vec: Sequence[float],
        *,
        store_id: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_estimate: int,
        identity: EmbeddingIdentity,
    ) -> None:
        self.ensure_chunk_schema()
        with self._write_transaction():
            profile = self._validate_embedding_write(identity, model=model)
            self._write_chunk_row(
                str(chunk_id),
                store_id=store_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                token_estimate=token_estimate,
                vec=vec,
                identity=identity,
                profile=profile,
            )

    def publish_chunk_embedding_under_lease(
        self,
        chunk_id: str,
        model: str,
        vec: Sequence[float],
        *,
        store_id: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_estimate: int,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
        request_id: str,
    ) -> EmbeddingPublishOutcome:
        """Atomically revalidate ownership and publish one accepted chunk vector."""
        chunk_id = str(chunk_id)
        return self._publish_under_lease(
            chunk_id,
            model=model,
            identity=identity,
            claim_key=claim_key,
            lease_id=lease_id,
            generation=generation,
            request_id=request_id,
            write_row=lambda profile: self._write_chunk_row(
                chunk_id,
                store_id=store_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                token_estimate=token_estimate,
                vec=vec,
                identity=identity,
                profile=profile,
            ),
        )

    def publish_chunk_embedding_batch_under_lease(
        self,
        rows: Sequence[dict],
        *,
        model: str,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
    ) -> list[BatchPublishResult]:
        """Publish an accepted network batch of chunk vectors in ONE transaction.

        Chunk analogue of :meth:`publish_embedding_batch_under_lease`. Each row
        dict carries ``chunk_id``, ``vec``, ``request_id`` and the chunk-meta
        fields (``store_id``/``chunk_index``/``char_start``/``char_end``/
        ``token_estimate``); each publishes under its own savepoint via the
        single-row ``publish_chunk_embedding_under_lease``.
        """
        return self._run_publish_batch(
            [
                (
                    str(row["chunk_id"]),
                    lambda row=row: self.publish_chunk_embedding_under_lease(
                        row["chunk_id"],
                        model,
                        row["vec"],
                        store_id=row["store_id"],
                        chunk_index=row["chunk_index"],
                        char_start=row["char_start"],
                        char_end=row["char_end"],
                        token_estimate=row["token_estimate"],
                        identity=identity,
                        claim_key=claim_key,
                        lease_id=lease_id,
                        generation=generation,
                        request_id=row["request_id"],
                    ),
                )
                for row in rows
            ]
        )

    def archive_chunks_for_messages(self, store_ids: Sequence[int | str]) -> int:
        """Soft-archive chunks whose source messages were purged/GC'd.

        Mirrors ``purge_embeddings_for_nodes`` but ARCHIVES (archived=1) rather
        than deletes, matching the meta.archived contract KNN filters on: an
        archived chunk is dropped from ranking but its row survives for bounded
        recovery. No-op (0) when the chunk schema was never materialized.
        """
        if not self._chunk_tables_exist():
            return 0
        unique_ids = list(dict.fromkeys(int(store_id) for store_id in store_ids))
        if not unique_ids:
            return 0
        with self._write_transaction():
            with self._temp_id_table([str(value) for value in unique_ids]) as table:
                cur = self._conn.execute(
                    f"UPDATE lcm_chunk_meta SET archived = 1 "
                    f"WHERE archived = 0 AND store_id IN "
                    f"(SELECT CAST(id AS INTEGER) FROM {table})"
                )
            if cur.rowcount:
                self._bump_all_data_versions()
        return int(cur.rowcount or 0)

    @staticmethod
    def archive_chunks_for_messages_on_connection(
        conn: sqlite3.Connection, store_ids: Sequence[int | str]
    ) -> int:
        """Archive one already-bounded message batch in the caller's transaction."""
        unique_ids = list(dict.fromkeys(int(store_id) for store_id in store_ids))
        if not unique_ids:
            return 0
        if len(unique_ids) > 256:
            raise ValueError("chunk archive batch exceeds 256 rows")
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ("
                "'lcm_chunk_meta','lcm_embedding_profile')"
            ).fetchall()
        }
        if not {"lcm_chunk_meta", "lcm_embedding_profile"} <= tables:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        cur = conn.execute(
            f"UPDATE lcm_chunk_meta SET archived = 1 "
            f"WHERE archived = 0 AND store_id IN ({placeholders})",
            unique_ids,
        )
        if cur.rowcount:
            conn.execute(
                "UPDATE lcm_embedding_profile SET data_version = data_version + 1"
            )
        return int(cur.rowcount or 0)

    def _chunk_tables_exist(self) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_chunk_meta'"
        ).fetchone() is not None

    def _bounded_chunk_candidate_ids(
        self,
        identity_hash: str,
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
        source: str | None,
        limit: int,
    ) -> list[str]:
        """Enumerate at most ``limit`` live chunk ids, most-recent-message first.

        The candidate enumeration is bounded at the SQL layer exactly like the
        summary path: recency (message timestamp), conversation (message
        session_id), and source (message source) filters live in the WHERE
        clause and a hard LIMIT caps the result. Unlike the summary corpus a
        chunk maps DIRECTLY to one raw message, so the source filter is a plain
        indexed column match (no recursive lineage walk) and needs no
        fail-open/closed budget.
        """
        if limit <= 0:
            return []
        if conversation_ids is not None and not list(conversation_ids):
            return []
        message_columns = self._table_columns("messages")
        if not message_columns:
            return []
        where = ["cm.identity_hash = ?", "cm.archived = 0"]
        args: list[object] = [str(identity_hash)]
        if since is not None:
            where.append("m.timestamp >= ?")
            args.append(float(since))
        if until is not None:
            where.append("m.timestamp <= ?")
            args.append(float(until))
        if source is not None:
            where.append("m.source = ?")
            args.append(str(source))
        args.append(int(limit))
        with self._optional_temp_id_table(conversation_ids) as conversation_table:
            conversation_join = (
                f"JOIN {conversation_table} c ON c.id = m.session_id"
                if conversation_table is not None
                else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT cm.chunk_id
                FROM lcm_chunk_meta cm
                JOIN messages m ON m.store_id = cm.store_id
                {conversation_join}
                WHERE {' AND '.join(where)}
                ORDER BY m.timestamp DESC, cm.chunk_id DESC
                LIMIT ?
                """,
                args,
            ).fetchall()
        return [str(row[0]) for row in rows]

    def _load_chunk_vectors_for_ids(
        self,
        identity_hash: str,
        dim: int,
        chunk_ids: Sequence[str],
    ) -> tuple[list[int], list[str], list[str], list[list[float]]]:
        rowids: list[int] = []
        out_ids: list[str] = []
        kinds: list[str] = []
        vectors: list[list[float]] = []
        if not chunk_ids:
            return rowids, out_ids, kinds, vectors
        with self._temp_id_table(chunk_ids) as table:
            rows = self._conn.execute(
                f"""
                SELECT v.rowid, v.chunk_id, v.vec
                FROM {table} t
                JOIN lcm_chunk_vectors v
                  ON v.chunk_id = t.id AND v.identity_hash = ?
                JOIN lcm_chunk_meta m
                  ON m.chunk_id = v.chunk_id AND m.identity_hash = v.identity_hash
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
            out_ids.append(str(row["chunk_id"]))
            kinds.append("chunk")
            vectors.append(list(vector))
        return rowids, out_ids, kinds, vectors

    def _numpy_chunk_rows(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        chunk_ids: Sequence[str],
    ) -> tuple[list[int], list[str], list[str], Any]:
        with self._cache_lock:
            data_version = self._data_version(identity_hash)
            key = (identity_hash, data_version, tuple(str(value) for value in chunk_ids))
            cached = self._chunk_matrix_cache.get(key)
            if cached is not None:
                self._chunk_matrix_cache.move_to_end(key)  # mark most-recently used
                return cached
            rowids, loaded_ids, kinds, raw_vectors = self._load_chunk_vectors_for_ids(
                identity_hash, dim, chunk_ids
            )
            matrix = (
                numpy.asarray(raw_vectors, dtype=numpy.float32)
                if raw_vectors
                else numpy.empty((0, dim), dtype=numpy.float32)
            )
            loaded = (rowids, loaded_ids, kinds, matrix)
            self._chunk_matrix_cache[key] = loaded
            while len(self._chunk_matrix_cache) > self._MATRIX_CACHE_MAX_ENTRIES:
                self._chunk_matrix_cache.popitem(last=False)  # evict oldest
            return loaded

    def knn_chunks(
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
        """Bounded-candidate chunk KNN with the summary coverage contract.

        Coverage is full|bounded|none exactly as for summaries: ``none`` when
        the corpus/identity is unbackfilled or a requested filter is
        unverifiable (missing message column), ``bounded`` when more candidates
        exist than the scan bound, ``full`` otherwise.
        """
        k = int(k)
        if k <= 0:
            return KNNResult(coverage="none")
        if not self._chunk_tables_exist():
            return KNNResult(coverage="none")
        message_columns = self._table_columns("messages")
        if conversation_ids is not None and "session_id" not in message_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if (since is not None or until is not None) and "timestamp" not in message_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if source is not None and "source" not in message_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        profile = (
            self._resolve_chunk_profile(str(model), provider=provider)
            if model is not None
            else self._current_chunk_profile()
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
        probed_ids = self._bounded_chunk_candidate_ids(
            identity,
            since=since,
            until=until,
            conversation_ids=conversation_ids,
            source=source,
            limit=limit + 1,
        )
        if not probed_ids:
            return KNNResult(coverage="none")
        bounded_ids = probed_ids[:limit]
        candidate_coverage = "bounded" if len(probed_ids) > limit else "full"

        if numpy is not None:
            rowids, chunk_ids, kinds, matrix = self._numpy_chunk_rows(
                numpy, identity, dim, bounded_ids
            )
            query_array = numpy.asarray(query, dtype=numpy.float32)
            scores = matrix @ query_array
            coverage = candidate_coverage
        else:
            rowids, chunk_ids, kinds, vectors = self._load_chunk_vectors_for_ids(
                identity, dim, bounded_ids
            )
            scores = [
                sum(value * query_value for value, query_value in zip(vector, query))
                for vector in vectors
            ]
            coverage = "bounded"

        candidates = self._ranked(rowids, chunk_ids, kinds, scores, k)
        scanned = total = None
        if coverage == "bounded":
            scanned = len(bounded_ids)
            total = self._count_embedded_vectors(identity, chunk=True)
        return KNNResult(candidates, coverage=coverage, scanned=scanned, total=total)

    def _current_chunk_profile(self) -> sqlite3.Row | None:
        """The active profile registered under task='chunk' (most recent)."""
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL AND task = ?
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """,
            (_CHUNK_TASK,),
        ).fetchone()

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
            self._chunk_matrix_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
