"""Immutable trajectory sources and bounded exact retrieval in one ``lcm.db``.

The store is intentionally provider-free.  It models agent trajectories as
first-class source material instead of flattening them into chat messages.
One database owns one corpus identity; normalized states and image manifests
remain traceable to protected canonical source JSON.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import struct
import threading
import time
from typing import Any, Iterable, Protocol, Sequence
from urllib.parse import quote, unquote

from .db_bootstrap import (
    configure_connection,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .ingest_protection import redact_sensitive_value
from .search_query import extract_search_terms


TRAJECTORY_MIGRATION_STEP = "trajectory_store_v1"
TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_SEMANTIC_DOCUMENT_VERSION = "trajectory-semantic-document-v1"
_MAX_CANDIDATES = 128
_MAX_RESULTS = 24
_MAX_IMAGES = 8
_MAX_QUERY_TEXT_CHARS = 8_000
_MAX_ADJACENCY_RADIUS = 8
_MAX_SOURCE_JSON_CHARS = 16_000_000
_MAX_TEXT_CHARS = 2_000_000
_MAX_SEMANTIC_DOCUMENT_CHARS = 48_000
_MAX_SEMANTIC_STATE_CHARS = 900
_MAX_SEMANTIC_STATES = 64
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "before", "did", "do",
    "does", "for", "from", "happen", "happened", "how", "i", "in", "is",
    "it", "of", "on", "or", "should", "the", "then", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
})
_EXACT_REF_RE = re.compile(
    r"^trajectory://(?P<corpus>[0-9a-f]{64})/"
    r"(?P<trajectory>[^/]+)/state/(?P<state>[0-9]+)$"
)


class TrajectoryStoreError(RuntimeError):
    """Base class for trajectory-store failures."""


class CorpusIdentityError(TrajectoryStoreError):
    """Raised when a database belongs to another immutable corpus."""


class TrajectoryAssetError(TrajectoryStoreError):
    """Raised when an image asset is missing, changed, or escapes its root."""


class ExactTrajectoryRefError(TrajectoryStoreError):
    """Raised when an exact trajectory reference is invalid or unresolved."""


class TrajectorySchemaUnavailableError(TrajectoryStoreError):
    """Raised when a read-only database lacks the trajectory schema."""


class TrajectoryEmbeddingProvider(Protocol):
    """Provider-neutral surface needed by the trajectory semantic index."""

    provider_id: str
    model_id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class CorpusIdentity:
    dataset_name: str
    dataset_revision: str
    harness_commit: str
    tier: str
    domain: str
    ingest_config_digest: str = ""

    def to_dict(self) -> dict[str, str]:
        values = {
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "harness_commit": self.harness_commit,
            "tier": self.tier,
            "domain": self.domain,
            "ingest_config_digest": self.ingest_config_digest,
        }
        normalized = {key: str(value or "").strip() for key, value in values.items()}
        missing = [key for key, value in normalized.items() if key != "ingest_config_digest" and not value]
        if missing:
            raise ValueError(f"corpus identity fields must not be empty: {missing}")
        return normalized

    @property
    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class TrajectoryState:
    state_index: int
    step: int
    url: str
    incoming_action: str | None
    thoughts: str | None
    text: str
    screenshot_path: str | Path
    observed_at: float | None = None
    observed_at_source: str | None = None
    occurred_at: float | None = None
    occurred_at_source: str | None = None


@dataclass(frozen=True)
class TrajectorySource:
    trajectory_id: str
    ordinal: int
    goal: str
    start_url: str
    outcome: str | None
    states: tuple[TrajectoryState, ...]
    source_payload: Any


@dataclass(frozen=True)
class TrajectoryInsertResult:
    trajectory_id: str
    source_sha256: str
    state_count: int
    already_current: bool


@dataclass(frozen=True)
class TrajectorySemanticAttempt:
    """Typed record of one semantic-retrieval attempt on the query path.

    Replaces the historical bare ``except Exception: fallbacks += 1`` that
    discarded the failure class/status entirely (the defect that hid the
    client-side spend-guard rate-limit behind an undifferentiated counter in
    the frozen V2 run). ``outcome`` is ``"success"`` or ``"fallback"``; failure
    fields are populated by best-effort duck-typing off the raised exception so
    this record stays decoupled from the embedding_provider exception classes.
    """

    provider: str
    model: str
    outcome: str
    exception_class: str | None = None
    http_status: int | None = None
    retry_after: float | None = None
    latency_ms: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class TrajectoryHit:
    exact_ref: str
    trajectory_id: str
    goal: str
    outcome: str | None
    state_index: int
    sequence_ordinal: int
    step: int
    url: str
    incoming_action: str | None
    thoughts: str | None
    text: str
    text_offset: int
    text_truncated: bool
    observed_at: float | None
    observed_at_source: str | None
    occurred_at: float | None
    occurred_at_source: str | None
    screenshot_path: str | None
    screenshot_sha256: str | None
    score: float
    match_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_ref": self.exact_ref,
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "outcome": self.outcome,
            "state_index": self.state_index,
            "sequence_ordinal": self.sequence_ordinal,
            "step": self.step,
            "url": self.url,
            "incoming_action": self.incoming_action,
            "thoughts": self.thoughts,
            "text": self.text,
            "text_offset": self.text_offset,
            "text_truncated": self.text_truncated,
            "observed_at": self.observed_at,
            "observed_at_source": self.observed_at_source,
            "occurred_at": self.occurred_at,
            "occurred_at_source": self.occurred_at_source,
            "screenshot_path": self.screenshot_path,
            "screenshot_sha256": self.screenshot_sha256,
            "score": round(float(self.score), 8),
            "match_kind": self.match_kind,
        }


class _ProtectionConfig:
    sensitive_patterns_enabled = True
    sensitive_patterns = (
        "api_key",
        "bearer_token",
        "password_assignment",
        "private_key",
    )
    sensitive_patterns_source = "trajectory_store"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory source must be finite JSON") from exc
    if len(encoded) > _MAX_SOURCE_JSON_CHARS:
        raise ValueError(
            f"trajectory source exceeds {_MAX_SOURCE_JSON_CHARS} characters"
        )
    return encoded


def _normalized_vector(values: Sequence[float], *, expected_dim: int | None = None) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory embedding vector must be numeric") from exc
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("trajectory embedding vector must be finite and non-empty")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError("trajectory embedding vector dimension changed")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("trajectory embedding vector must have a finite nonzero norm")
    return tuple(value / norm for value in vector)


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(payload: bytes, dim: int) -> tuple[float, ...]:
    expected_bytes = dim * 4
    if len(payload) != expected_bytes:
        raise TrajectoryStoreError("stored trajectory embedding dimension is invalid")
    return tuple(struct.unpack(f"<{dim}f", payload))


def create_trajectory_embedding_provider(
    provider_name: str,
    model_name: str,
    *,
    timeout_seconds: float,
    for_backfill: bool = False,
) -> TrajectoryEmbeddingProvider:
    """Resolve an existing LCM provider without persisting its credential.

    The provider reads credentials from its normal environment-backed secret
    seam. Only the provider/model identifiers belong in saved memory config.
    """
    from .config import LCMConfig
    from .embedding_provider import resolve_provider

    timeout = max(0.1, float(timeout_seconds))
    config = LCMConfig(
        embedding_provider=str(provider_name).strip(),
        embedding_model=str(model_name).strip(),
        embedding_query_timeout_s=timeout,
        embedding_backfill_timeout_s=timeout,
    )
    provider = resolve_provider(config, for_backfill=for_backfill)
    if provider is None:
        raise TrajectoryStoreError("trajectory embedding provider is not configured")
    return provider


def _finite_optional(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _time_with_source(
    value: float | None,
    source: str | None,
    field: str,
) -> tuple[float | None, str | None]:
    timestamp = _finite_optional(value, field)
    normalized_source = (
        _bounded_text(source, f"{field}_source", allow_empty=False).strip()
        if source is not None
        else None
    )
    if timestamp is None and normalized_source is not None:
        raise ValueError(f"{field}_source requires {field}")
    if timestamp is not None and normalized_source is None:
        raise ValueError(f"{field} requires explicit {field}_source provenance")
    return timestamp, normalized_source


def _bounded_text(value: Any, field: str, *, allow_empty: bool = True) -> str:
    text = str(value if value is not None else "")
    if not allow_empty and not text.strip():
        raise ValueError(f"{field} must not be empty")
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(f"{field} exceeds {_MAX_TEXT_CHARS} characters")
    return text


class TrajectoryStore:
    """SQLite trajectory store bound to the same physical file as LCM core."""

    def __init__(
        self,
        db_path: str | Path,
        identity: CorpusIdentity,
        *,
        asset_root: str | Path,
        read_only: bool = False,
        protect_sensitive: bool = True,
        embedding_provider: TrajectoryEmbeddingProvider | None = None,
        semantic_top_trajectories: int = 12,
    ) -> None:
        self.db_path = Path(db_path)
        self.identity = identity
        self.identity_payload = identity.to_dict()
        self.identity_digest = identity.digest
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.read_only = bool(read_only)
        self.protect_sensitive = bool(protect_sensitive)
        self.embedding_provider = embedding_provider
        self.semantic_top_trajectories = min(
            max(1, int(semantic_top_trajectories)),
            32,
        )
        self._semantic_usage: dict[str, int] = {
            "document_calls": 0,
            "document_tokens": 0,
            "query_calls": 0,
            "query_tokens": 0,
            "fallbacks": 0,
        }
        # Typed per-run instrument state (additive; the existing
        # ``_semantic_usage`` counters above are unchanged for back-compat).
        # The attempt log is BOUNDED (recent-window ring) so a long run cannot
        # grow it without bound; the funnel counters are tracked SEPARATELY and
        # stay cumulative regardless of the ring's cap.
        self._semantic_attempts: deque[TrajectorySemanticAttempt] = deque(maxlen=1024)
        self._semantic_attempt_totals: dict[str, Any] = {
            "attempts": 0,
            "successes": 0,
            "fallbacks": 0,
            "fallbacks_by_reason": {},
        }
        self._last_semantic_attempt: TrajectorySemanticAttempt | None = None
        self._last_query_telemetry: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._conn = self._open_connection()
        try:
            self._init_schema()
            self._bind_identity()
        except Exception:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
            raise

    def _open_connection(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{quote(str(self.db_path), safe='/')}?mode=ro"
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=30000")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,
            )
            refuse_schema_version_too_new(conn)
            configure_connection(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        required = {
            "lcm_trajectory_corpora",
            "lcm_trajectory_sources",
            "lcm_trajectory_states",
            "lcm_trajectory_assets",
            "lcm_trajectory_ingest_receipts",
            "lcm_trajectory_transitions",
            "lcm_trajectory_states_fts",
        }
        if self.read_only:
            existing = {
                str(row[0])
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE 'lcm_trajectory%'"
                )
            }
            missing = sorted(required - existing)
            if missing:
                raise TrajectorySchemaUnavailableError(
                    f"trajectory schema unavailable for read-only query: {missing}"
                )
            return

        run_versioned_migrations(self._conn)
        fts_preexisting = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'lcm_trajectory_states_fts'"
        ).fetchone() is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lcm_trajectory_corpora (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                identity_digest TEXT NOT NULL UNIQUE,
                identity_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                corpus_uid TEXT,
                haystack_digest TEXT,
                source_manifest_digest TEXT,
                trajectory_count INTEGER,
                ingest_cursor INTEGER NOT NULL DEFAULT 0 CHECK(ingest_cursor >= 0),
                status TEXT NOT NULL CHECK(status IN ('building', 'complete', 'invalid')),
                created_at REAL NOT NULL,
                completed_at REAL
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trajectory_id TEXT NOT NULL UNIQUE,
                ordinal INTEGER NOT NULL UNIQUE,
                source_json TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                goal TEXT NOT NULL,
                start_url TEXT NOT NULL,
                outcome TEXT,
                state_count INTEGER NOT NULL CHECK(state_count > 0),
                inserted_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_states (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES lcm_trajectory_sources(source_id) ON DELETE CASCADE,
                state_index INTEGER NOT NULL CHECK(state_index >= 0),
                sequence_ordinal INTEGER NOT NULL CHECK(sequence_ordinal >= 0),
                step INTEGER NOT NULL CHECK(step >= 0),
                url TEXT NOT NULL,
                incoming_action TEXT,
                thoughts TEXT,
                text TEXT NOT NULL,
                search_text TEXT NOT NULL,
                observed_at REAL,
                observed_at_source TEXT,
                occurred_at REAL,
                occurred_at_source TEXT,
                ingested_at REAL NOT NULL,
                UNIQUE(source_id, state_index),
                UNIQUE(source_id, sequence_ordinal)
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_assets (
                asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_id INTEGER NOT NULL UNIQUE REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                byte_size INTEGER NOT NULL CHECK(byte_size >= 0)
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_ingest_receipts (
                ordinal INTEGER PRIMARY KEY CHECK(ordinal >= 0),
                trajectory_id TEXT NOT NULL UNIQUE,
                source_sha256 TEXT NOT NULL,
                committed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_transitions (
                transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES lcm_trajectory_sources(source_id) ON DELETE CASCADE,
                sequence_ordinal INTEGER NOT NULL CHECK(sequence_ordinal >= 1),
                pre_state_id INTEGER NOT NULL REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                post_state_id INTEGER NOT NULL REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                incoming_action TEXT,
                UNIQUE(source_id, sequence_ordinal),
                UNIQUE(source_id, post_state_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS lcm_trajectory_states_fts
            USING fts5(search_text, content='lcm_trajectory_states', content_rowid='state_id');

            CREATE TRIGGER IF NOT EXISTS lcm_trajectory_fts_insert
            AFTER INSERT ON lcm_trajectory_states BEGIN
                INSERT INTO lcm_trajectory_states_fts(rowid, search_text)
                VALUES (new.state_id, new.search_text);
            END;

            CREATE TRIGGER IF NOT EXISTS lcm_trajectory_fts_delete
            AFTER DELETE ON lcm_trajectory_states BEGIN
                INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts, rowid, search_text)
                VALUES ('delete', old.state_id, old.search_text);
            END;

            CREATE TRIGGER IF NOT EXISTS lcm_trajectory_fts_update
            AFTER UPDATE OF search_text ON lcm_trajectory_states BEGIN
                INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts, rowid, search_text)
                VALUES ('delete', old.state_id, old.search_text);
                INSERT INTO lcm_trajectory_states_fts(rowid, search_text)
                VALUES (new.state_id, new.search_text);
            END;

            CREATE INDEX IF NOT EXISTS lcm_trajectory_states_source_sequence
            ON lcm_trajectory_states(source_id, sequence_ordinal);
            """
        )
        state_count = int(
            self._conn.execute("SELECT COUNT(*) FROM lcm_trajectory_states").fetchone()[0]
        )
        fts_count = int(
            self._conn.execute("SELECT COUNT(*) FROM lcm_trajectory_states_fts").fetchone()[0]
        )
        fts_needs_rebuild = not fts_preexisting or state_count != fts_count
        if not fts_needs_rebuild:
            try:
                self._conn.execute(
                    "INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts, rank) "
                    "VALUES ('integrity-check', 1)"
                )
            except sqlite3.DatabaseError:
                fts_needs_rebuild = True
        if fts_needs_rebuild:
            self._conn.execute(
                "INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts) VALUES ('rebuild')"
            )
        marker = self._conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = ?",
            (TRAJECTORY_MIGRATION_STEP,),
        ).fetchone()
        if marker is None:
            mark_migration_step_complete(self._conn, TRAJECTORY_MIGRATION_STEP)
        self._conn.commit()

    def _bind_identity(self) -> None:
        row = self._conn.execute(
            "SELECT * FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        if row is None:
            if self.read_only:
                raise CorpusIdentityError("trajectory database has no corpus identity")
            self._conn.execute(
                """
                INSERT INTO lcm_trajectory_corpora(
                    singleton, identity_digest, identity_json, schema_version,
                    status, created_at
                ) VALUES (1, ?, ?, ?, 'building', ?)
                """,
                (
                    self.identity_digest,
                    _canonical_json(self.identity_payload),
                    TRAJECTORY_SCHEMA_VERSION,
                    time.time(),
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM lcm_trajectory_corpora WHERE singleton = 1"
            ).fetchone()
        if (
            str(row["identity_digest"]) != self.identity_digest
            or int(row["schema_version"]) != TRAJECTORY_SCHEMA_VERSION
        ):
            raise CorpusIdentityError(
                "trajectory database corpus identity does not match requested identity"
            )

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def status(self) -> str:
        row = self._conn.execute(
            "SELECT status FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        return str(row[0])

    @property
    def corpus_uid(self) -> str | None:
        row = self._conn.execute(
            "SELECT corpus_uid FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        return str(row[0]) if row and row[0] else None

    def _require_writable(self) -> None:
        if self.read_only:
            raise TrajectoryStoreError("read-only TrajectoryStore cannot write")

    def _protect(self, value: Any) -> Any:
        if not self.protect_sensitive:
            return value
        return redact_sensitive_value(
            value,
            _ProtectionConfig(),
            parse_json_strings=True,
        )

    def _validated_asset(self, path_value: str | Path) -> tuple[str, str, int]:
        try:
            candidate = Path(path_value).expanduser().resolve(strict=True)
            root = self.asset_root.resolve(strict=True)
            relative = candidate.relative_to(root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise TrajectoryAssetError(
                "trajectory screenshot must exist inside the configured asset root"
            ) from exc
        if not candidate.is_file():
            raise TrajectoryAssetError(f"trajectory screenshot is not a file: {candidate}")
        return relative.as_posix(), _sha256_file(candidate), candidate.stat().st_size

    def _protected_source(
        self, source: TrajectorySource
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[tuple[str, str, int], ...]]:
        trajectory_id = _bounded_text(
            source.trajectory_id, "trajectory_id", allow_empty=False
        ).strip()
        if source.ordinal < 0:
            raise ValueError("trajectory ordinal must be non-negative")
        if not source.states:
            raise ValueError("trajectory must contain at least one state")
        state_indexes = [state.state_index for state in source.states]
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in state_indexes
        ):
            raise ValueError("state indexes must be non-negative integers")
        if len(state_indexes) != len(set(state_indexes)):
            raise ValueError("trajectory state indexes must be unique")

        protected_states: list[dict[str, Any]] = []
        assets: list[tuple[str, str, int]] = []
        for sequence_ordinal, state in enumerate(source.states):
            if (
                not isinstance(state.step, int)
                or isinstance(state.step, bool)
                or state.step < 0
            ):
                raise ValueError("trajectory state step must be a non-negative integer")
            observed_at, observed_at_source = _time_with_source(
                state.observed_at,
                state.observed_at_source,
                "observed_at",
            )
            occurred_at, occurred_at_source = _time_with_source(
                state.occurred_at,
                state.occurred_at_source,
                "occurred_at",
            )
            relative_path, asset_sha, asset_size = self._validated_asset(
                state.screenshot_path
            )
            assets.append((relative_path, asset_sha, asset_size))
            protected = self._protect({
                "state_index": int(state.state_index),
                "sequence_ordinal": sequence_ordinal,
                "step": int(state.step),
                "url": _bounded_text(state.url, "url", allow_empty=False),
                "incoming_action": (
                    _bounded_text(state.incoming_action, "incoming_action")
                    if state.incoming_action is not None
                    else None
                ),
                "thoughts": (
                    _bounded_text(state.thoughts, "thoughts")
                    if state.thoughts is not None
                    else None
                ),
                "text": _bounded_text(state.text, "text"),
                "screenshot": relative_path,
                "screenshot_sha256": asset_sha,
                "observed_at": observed_at,
                "observed_at_source": observed_at_source,
                "occurred_at": occurred_at,
                "occurred_at_source": occurred_at_source,
            })
            protected_states.append(dict(protected))

        protected_source = self._protect({
            "source_payload": source.source_payload,
            "normalized": {
                "trajectory_id": trajectory_id,
                "ordinal": int(source.ordinal),
                "goal": _bounded_text(source.goal, "goal"),
                "start_url": _bounded_text(
                    source.start_url, "start_url", allow_empty=False
                ),
                "outcome": (
                    _bounded_text(source.outcome, "outcome")
                    if source.outcome is not None
                    else None
                ),
                "states": protected_states,
            },
        })
        return dict(protected_source), tuple(protected_states), tuple(assets)

    @staticmethod
    def _search_text(
        *,
        goal: str,
        outcome: str | None,
        url: str,
        incoming_action: str | None,
        thoughts: str | None,
        text: str,
    ) -> str:
        return "\n".join((
            f"Goal: {goal}",
            f"Outcome: {outcome or ''}",
            f"URL: {url}",
            f"Incoming action: {incoming_action or ''}",
            f"Thought: {thoughts or ''}",
            f"Visible state: {text}",
        ))

    def insert(self, source: TrajectorySource) -> TrajectoryInsertResult:
        self._require_writable()
        protected_source, protected_states, assets = self._protected_source(source)
        source_json = _canonical_json(protected_source)
        source_sha = _sha256_text(source_json)
        normalized = protected_source["normalized"]
        trajectory_id = str(normalized["trajectory_id"])
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._conn.execute(
                    """
                    SELECT source_sha256, state_count
                    FROM lcm_trajectory_sources
                    WHERE trajectory_id = ?
                    """,
                    (trajectory_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["source_sha256"]) != source_sha:
                        raise TrajectoryStoreError(
                            f"trajectory {trajectory_id} already exists with a different digest"
                        )
                    self._conn.commit()
                    return TrajectoryInsertResult(
                        trajectory_id=trajectory_id,
                        source_sha256=source_sha,
                        state_count=int(existing["state_count"]),
                        already_current=True,
                    )
                corpus_row = self._conn.execute(
                    """
                    SELECT status, ingest_cursor
                    FROM lcm_trajectory_corpora WHERE singleton = 1
                    """
                ).fetchone()
                if str(corpus_row["status"]) == "complete":
                    raise TrajectoryStoreError("complete trajectory corpus is immutable")
                expected_ordinal = int(corpus_row["ingest_cursor"])
                if int(normalized["ordinal"]) != expected_ordinal:
                    raise TrajectoryStoreError(
                        f"trajectory ordinal {normalized['ordinal']} does not match contiguous ingest cursor {expected_ordinal}"
                    )

                now = time.time()
                cursor = self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_sources(
                        trajectory_id, ordinal, source_json, source_sha256,
                        goal, start_url, outcome, state_count, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trajectory_id,
                        int(normalized["ordinal"]),
                        source_json,
                        source_sha,
                        str(normalized["goal"]),
                        str(normalized["start_url"]),
                        normalized["outcome"],
                        len(protected_states),
                        now,
                    ),
                )
                source_id = int(cursor.lastrowid)
                inserted_state_ids: list[int] = []
                for state, asset in zip(protected_states, assets):
                    search_text = self._search_text(
                        goal=str(normalized["goal"]),
                        outcome=normalized["outcome"],
                        url=str(state["url"]),
                        incoming_action=state["incoming_action"],
                        thoughts=state["thoughts"],
                        text=str(state["text"]),
                    )
                    state_cursor = self._conn.execute(
                        """
                        INSERT INTO lcm_trajectory_states(
                            source_id, state_index, sequence_ordinal, step, url,
                            incoming_action, thoughts, text, search_text,
                            observed_at, observed_at_source,
                            occurred_at, occurred_at_source, ingested_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            int(state["state_index"]),
                            int(state["sequence_ordinal"]),
                            int(state["step"]),
                            str(state["url"]),
                            state["incoming_action"],
                            state["thoughts"],
                            str(state["text"]),
                            search_text,
                            state["observed_at"],
                            state["observed_at_source"],
                            state["occurred_at"],
                            state["occurred_at_source"],
                            now,
                        ),
                    )
                    inserted_state_ids.append(int(state_cursor.lastrowid))
                    relative_path, asset_sha, asset_size = asset
                    self._conn.execute(
                        """
                        INSERT INTO lcm_trajectory_assets(
                            state_id, relative_path, sha256, byte_size
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            int(state_cursor.lastrowid),
                            relative_path,
                            asset_sha,
                            asset_size,
                        ),
                    )
                for sequence_ordinal in range(1, len(inserted_state_ids)):
                    state = protected_states[sequence_ordinal]
                    self._conn.execute(
                        """
                        INSERT INTO lcm_trajectory_transitions(
                            source_id, sequence_ordinal, pre_state_id,
                            post_state_id, incoming_action
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            sequence_ordinal,
                            inserted_state_ids[sequence_ordinal - 1],
                            inserted_state_ids[sequence_ordinal],
                            state["incoming_action"],
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_ingest_receipts(
                        ordinal, trajectory_id, source_sha256, committed_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (int(normalized["ordinal"]), trajectory_id, source_sha, now),
                )
                self._conn.execute(
                    """
                    UPDATE lcm_trajectory_corpora
                    SET ingest_cursor = ingest_cursor + 1
                    WHERE singleton = 1
                    """
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return TrajectoryInsertResult(
            trajectory_id=trajectory_id,
            source_sha256=source_sha,
            state_count=len(protected_states),
            already_current=False,
        )

    def finalize(self, ordered_trajectory_ids: Sequence[str]) -> str:
        self._require_writable()
        ordered = tuple(str(value or "").strip() for value in ordered_trajectory_ids)
        if not ordered or any(not value for value in ordered):
            raise ValueError("ordered trajectory ids must be non-empty")
        if len(ordered) != len(set(ordered)):
            raise ValueError("ordered trajectory ids must be unique")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    """
                    SELECT trajectory_id, source_sha256
                    FROM lcm_trajectory_sources
                    ORDER BY ordinal, trajectory_id
                    """
                ).fetchall()
                actual = tuple(str(row["trajectory_id"]) for row in rows)
                if actual != ordered:
                    raise CorpusIdentityError(
                        "inserted trajectory order does not match final ordered haystack"
                    )
                corpus_row = self._conn.execute(
                    """
                    SELECT status, corpus_uid, ingest_cursor
                    FROM lcm_trajectory_corpora WHERE singleton = 1
                    """
                ).fetchone()
                if int(corpus_row["ingest_cursor"]) != len(ordered):
                    raise CorpusIdentityError(
                        "trajectory ingest cursor does not cover the final ordered haystack"
                    )
                haystack_digest = _sha256_text(_canonical_json(list(ordered)))
                source_manifest_digest = _sha256_text(_canonical_json([
                    [str(row["trajectory_id"]), str(row["source_sha256"])]
                    for row in rows
                ]))
                corpus_uid = _sha256_text(
                    f"{self.identity_digest}\0{haystack_digest}\0{source_manifest_digest}"
                )
                if str(corpus_row["status"]) == "complete":
                    if str(corpus_row["corpus_uid"]) != corpus_uid:
                        raise CorpusIdentityError(
                            "complete corpus source or haystack identity mismatch"
                        )
                    self._conn.commit()
                    return corpus_uid
                self._conn.execute(
                    """
                    UPDATE lcm_trajectory_corpora
                    SET corpus_uid = ?, haystack_digest = ?, source_manifest_digest = ?,
                        trajectory_count = ?, status = 'complete', completed_at = ?
                    WHERE singleton = 1
                    """,
                    (
                        corpus_uid,
                        haystack_digest,
                        source_manifest_digest,
                        len(ordered),
                        time.time(),
                    ),
                )
                self._conn.commit()
                return corpus_uid
            except Exception:
                self._conn.rollback()
                raise

    def set_embedding_provider(
        self,
        provider: TrajectoryEmbeddingProvider | None,
    ) -> None:
        """Replace the ephemeral provider without changing saved corpus state."""
        self.embedding_provider = provider

    def _ensure_semantic_schema(self) -> None:
        self._require_writable()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lcm_trajectory_embedding_profiles (
                profile_digest TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL CHECK(dim > 0),
                document_version TEXT NOT NULL,
                source_manifest_digest TEXT NOT NULL,
                document_count INTEGER NOT NULL CHECK(document_count >= 0),
                index_digest TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0, 1)),
                created_at REAL NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS lcm_trajectory_embedding_one_active
            ON lcm_trajectory_embedding_profiles(active) WHERE active = 1;

            CREATE TABLE IF NOT EXISTS lcm_trajectory_embeddings (
                source_id INTEGER PRIMARY KEY
                    REFERENCES lcm_trajectory_sources(source_id) ON DELETE CASCADE,
                profile_digest TEXT NOT NULL
                    REFERENCES lcm_trajectory_embedding_profiles(profile_digest)
                    ON DELETE CASCADE,
                document_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                embedded_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS lcm_trajectory_embeddings_profile
            ON lcm_trajectory_embeddings(profile_digest, source_id);
            """
        )
        self._conn.commit()

    def _semantic_profile(self) -> sqlite3.Row | None:
        exists = self._conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'lcm_trajectory_embedding_profiles'
            """
        ).fetchone()
        if exists is None:
            return None
        return self._conn.execute(
            """
            SELECT * FROM lcm_trajectory_embedding_profiles
            WHERE active = 1
            """
        ).fetchone()

    @staticmethod
    def _sample_state_rows(rows: Sequence[sqlite3.Row]) -> list[sqlite3.Row]:
        if len(rows) <= _MAX_SEMANTIC_STATES:
            return list(rows)
        last = len(rows) - 1
        indexes = {
            round(position * last / (_MAX_SEMANTIC_STATES - 1))
            for position in range(_MAX_SEMANTIC_STATES)
        }
        return [rows[index] for index in sorted(indexes)]

    def _semantic_documents(self) -> list[tuple[int, str, str]]:
        sources = self._conn.execute(
            """
            SELECT source_id, trajectory_id, goal, start_url, outcome
            FROM lcm_trajectory_sources
            ORDER BY ordinal, trajectory_id
            """
        ).fetchall()
        documents: list[tuple[int, str, str]] = []
        for source in sources:
            states = self._conn.execute(
                """
                SELECT state_index, sequence_ordinal, step, url,
                       incoming_action, thoughts, text
                FROM lcm_trajectory_states
                WHERE source_id = ?
                ORDER BY sequence_ordinal
                """,
                (int(source["source_id"]),),
            ).fetchall()
            lines = [
                f"Trajectory: {source['trajectory_id']}",
                f"Goal: {source['goal']}",
                f"Start URL: {source['start_url']}",
                f"Outcome: {source['outcome'] or '<unknown>'}",
            ]
            for state in self._sample_state_rows(states):
                state_text = " | ".join(
                    part
                    for part in (
                        f"State {state['state_index']} sequence {state['sequence_ordinal']} step {state['step']}",
                        f"URL {state['url']}",
                        (
                            f"Action {state['incoming_action']}"
                            if state["incoming_action"] is not None
                            else ""
                        ),
                        (
                            f"Thought {state['thoughts']}"
                            if state["thoughts"] is not None
                            else ""
                        ),
                        f"Visible {state['text']}",
                    )
                    if part
                )
                lines.append(state_text[:_MAX_SEMANTIC_STATE_CHARS])
            document = "\n".join(lines)[:_MAX_SEMANTIC_DOCUMENT_CHARS]
            documents.append(
                (int(source["source_id"]), document, _sha256_text(document))
            )
        return documents

    def build_semantic_index(
        self,
        provider: TrajectoryEmbeddingProvider | None = None,
    ) -> dict[str, Any]:
        """Build one deterministic source-derived vector per trajectory."""
        if self.status != "complete":
            raise CorpusIdentityError("trajectory corpus must be finalized before indexing")
        active_provider = provider or self.embedding_provider
        if active_provider is None:
            raise TrajectoryStoreError("trajectory embedding provider is not configured")
        self._ensure_semantic_schema()
        corpus = self._conn.execute(
            """
            SELECT source_manifest_digest, trajectory_count
            FROM lcm_trajectory_corpora WHERE singleton = 1
            """
        ).fetchone()
        provider_name = str(getattr(active_provider, "provider_id", "unknown"))
        model_name = str(getattr(active_provider, "model_id", "")).strip()
        if not model_name:
            raise ValueError("trajectory embedding model_id must not be empty")
        current = self._semantic_profile()
        expected_count = int(corpus["trajectory_count"])
        if current is not None and (
            str(current["provider"]) == provider_name
            and str(current["model_name"]) == model_name
            and str(current["document_version"]) == TRAJECTORY_SEMANTIC_DOCUMENT_VERSION
            and str(current["source_manifest_digest"]) == str(corpus["source_manifest_digest"])
            and int(current["document_count"]) == expected_count
        ):
            actual_count = int(self._conn.execute(
                "SELECT COUNT(*) FROM lcm_trajectory_embeddings WHERE profile_digest = ?",
                (str(current["profile_digest"]),),
            ).fetchone()[0])
            if actual_count == expected_count:
                return {
                    "status": "current",
                    "profile_digest": str(current["profile_digest"]),
                    "document_count": actual_count,
                    "dim": int(current["dim"]),
                    "index_digest": str(current["index_digest"]),
                }

        documents = self._semantic_documents()
        texts = [document for _source_id, document, _digest in documents]
        embed_batches = getattr(active_provider, "embed_document_batches", None)
        vectors: list[list[float]] = []
        if callable(embed_batches):
            indexed_vectors: dict[int, list[float]] = {}
            for batch in embed_batches(texts):
                self._semantic_usage["document_calls"] += 1
                self._semantic_usage["document_tokens"] += max(
                    0,
                    int(getattr(active_provider, "last_usage_tokens", 0) or 0),
                )
                for index, vector in zip(batch.indexes, batch.vectors):
                    indexed_vectors[int(index)] = list(vector)
            vectors = [indexed_vectors[index] for index in range(len(texts))]
        else:
            self._semantic_usage["document_calls"] += 1
            vectors = active_provider.embed_documents(texts)
            self._semantic_usage["document_tokens"] += max(
                0,
                int(getattr(active_provider, "last_usage_tokens", 0) or 0),
            )
        if len(vectors) != len(documents):
            raise ValueError("trajectory embedding count does not match source count")
        normalized: list[tuple[float, ...]] = []
        dim: int | None = None
        for vector in vectors:
            normalized_vector = _normalized_vector(vector, expected_dim=dim)
            if dim is None:
                dim = len(normalized_vector)
            normalized.append(normalized_vector)
        if dim is None:
            raise ValueError("trajectory semantic index cannot be empty")
        profile_digest = _sha256_text(_canonical_json({
            "provider": provider_name,
            "model": model_name,
            "dim": dim,
            "document_version": TRAJECTORY_SEMANTIC_DOCUMENT_VERSION,
            "source_manifest_digest": str(corpus["source_manifest_digest"]),
        }))
        packed = [_pack_vector(vector) for vector in normalized]
        index_digest = _sha256_text(_canonical_json([
            [source_id, document_sha, hashlib.sha256(vector).hexdigest()]
            for (source_id, _document, document_sha), vector in zip(documents, packed)
        ]))
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE lcm_trajectory_embedding_profiles SET active = 0 WHERE active = 1"
                )
                self._conn.execute("DELETE FROM lcm_trajectory_embeddings")
                self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_embedding_profiles(
                        profile_digest, provider, model_name, dim,
                        document_version, source_manifest_digest,
                        document_count, index_digest, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(profile_digest) DO UPDATE SET
                        document_count = excluded.document_count,
                        index_digest = excluded.index_digest,
                        active = 1,
                        created_at = excluded.created_at
                    """,
                    (
                        profile_digest,
                        provider_name,
                        model_name,
                        dim,
                        TRAJECTORY_SEMANTIC_DOCUMENT_VERSION,
                        str(corpus["source_manifest_digest"]),
                        len(documents),
                        index_digest,
                        now,
                    ),
                )
                self._conn.executemany(
                    """
                    INSERT INTO lcm_trajectory_embeddings(
                        source_id, profile_digest, document_sha256, vector, embedded_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (source_id, profile_digest, document_sha, vector, now)
                        for (source_id, _document, document_sha), vector
                        in zip(documents, packed)
                    ],
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "status": "built",
            "profile_digest": profile_digest,
            "document_count": len(documents),
            "dim": dim,
            "index_digest": index_digest,
        }

    def semantic_metrics(self) -> dict[str, int]:
        return dict(self._semantic_usage)

    @staticmethod
    def _safe_getattr(obj: Any, name: str) -> Any:
        """getattr that swallows EVERY exception, not just AttributeError.

        A hostile/exotic exception can expose ``kind``/``status_code``/
        ``retry_after`` as *properties that raise*; plain getattr(obj, name,
        default) only absorbs AttributeError and would let that propagate out
        of telemetry recording and crash the query path. Introspection must
        never be able to turn a semantic fallback into a total failure.
        """
        try:
            return getattr(obj, name, None)
        except Exception:
            return None

    @staticmethod
    def _classify_semantic_reason(exception: BaseException) -> str:
        """Best-effort failure category for the fallbacks-by-reason counter.

        Duck-typed so trajectory_store need not import the provider exception
        classes: client-side spend guard, circuit breaker, and Voyage's
        classified ``kind`` (auth/rate_limit/bad_request/server_error/...) all
        surface distinctly instead of collapsing to one opaque bucket.
        """
        name = type(exception).__name__
        kind = str(TrajectoryStore._safe_getattr(exception, "kind") or "")
        if name == "ProviderRateLimited":
            return "client_rate_guard"
        if name == "ProviderCircuitOpen":
            return "circuit_open"
        if isinstance(exception, TimeoutError) or kind == "timeout":
            return "timeout"
        if kind:
            return kind
        return "other"

    def _bump_attempt_totals(self, attempt: TrajectorySemanticAttempt) -> None:
        totals = self._semantic_attempt_totals
        totals["attempts"] += 1
        if attempt.outcome == "success":
            totals["successes"] += 1
        elif attempt.outcome == "fallback":
            totals["fallbacks"] += 1
            key = attempt.reason or "other"
            by_reason = totals["fallbacks_by_reason"]
            by_reason[key] = by_reason.get(key, 0) + 1

    def _store_attempt(self, attempt: TrajectorySemanticAttempt) -> TrajectorySemanticAttempt:
        self._semantic_attempts.append(attempt)
        self._last_semantic_attempt = attempt
        self._bump_attempt_totals(attempt)
        return attempt

    def _record_semantic_attempt(
        self,
        *,
        outcome: str,
        latency_ms: float,
        exception: BaseException | None,
    ) -> TrajectorySemanticAttempt:
        provider = self.embedding_provider
        provider_id = str(self._safe_getattr(provider, "provider_id") or "unknown") if provider else "none"
        model_id = str(self._safe_getattr(provider, "model_id") or "") if provider else ""
        exception_class: str | None = None
        http_status: int | None = None
        retry_after: float | None = None
        reason: str | None = None
        if exception is not None:
            exception_class = type(exception).__name__
            raw_status = self._safe_getattr(exception, "status_code")
            http_status = (
                int(raw_status)
                if isinstance(raw_status, int) and not isinstance(raw_status, bool)
                else None
            )
            raw_retry = self._safe_getattr(exception, "retry_after")
            retry_after = (
                float(raw_retry)
                if isinstance(raw_retry, (int, float)) and not isinstance(raw_retry, bool)
                else None
            )
            reason = self._classify_semantic_reason(exception)
        attempt = TrajectorySemanticAttempt(
            provider=provider_id,
            model=model_id,
            outcome=str(outcome),
            exception_class=exception_class,
            http_status=http_status,
            retry_after=retry_after,
            latency_ms=round(float(latency_ms), 3),
            reason=reason,
        )
        return self._store_attempt(attempt)

    def _record_minimal_fallback_attempt(
        self, *, latency_ms: float
    ) -> TrajectorySemanticAttempt:
        """Last-resort record when even best-effort introspection failed. Keeps
        the funnel honest (a fallback still counts) without touching the
        offending exception again."""
        provider = self.embedding_provider
        attempt = TrajectorySemanticAttempt(
            provider="unknown" if provider is not None else "none",
            model="",
            outcome="fallback",
            reason="attempt_record_introspection_failed",
            latency_ms=round(float(latency_ms), 3),
        )
        return self._store_attempt(attempt)

    def last_semantic_attempt(self) -> dict[str, Any] | None:
        """The typed attempt record from the most recent ``query()`` call."""
        if self._last_semantic_attempt is None:
            return None
        return asdict(self._last_semantic_attempt)

    def semantic_attempt_counters(self) -> dict[str, Any]:
        """Per-run semantic funnel counters (attempts / successes / fallbacks by
        reason). Tracked cumulatively and independently of the bounded attempt
        ring, so counts stay exact across an arbitrarily long run."""
        totals = self._semantic_attempt_totals
        return {
            "attempts": totals["attempts"],
            "successes": totals["successes"],
            "fallbacks": totals["fallbacks"],
            "fallbacks_by_reason": dict(totals["fallbacks_by_reason"]),
        }

    def last_query_telemetry(self) -> dict[str, Any] | None:
        """Side-channel telemetry for the most recent ``query()``: the semantic
        attempt record, the ranked source-candidate list, the pre-selection
        state-candidate pool, and the delivered-evidence refs. Written after the
        call returns and never affects the returned hits (byte-identical
        evidence)."""
        if self._last_query_telemetry is None:
            return None
        return dict(self._last_query_telemetry)

    def _semantic_source_ranks(self, query: str) -> list[tuple[int, float]]:
        provider = self.embedding_provider
        profile = self._semantic_profile()
        if provider is None or profile is None:
            return []
        if (
            str(profile["provider"]) != str(getattr(provider, "provider_id", "unknown"))
            or str(profile["model_name"]) != str(getattr(provider, "model_id", ""))
        ):
            return []
        self._semantic_usage["query_calls"] += 1
        query_vector = _normalized_vector(
            provider.embed_query(query),
            expected_dim=int(profile["dim"]),
        )
        self._semantic_usage["query_tokens"] += max(
            0,
            int(getattr(provider, "last_usage_tokens", 0) or 0),
        )
        rows = self._conn.execute(
            """
            SELECT source_id, vector
            FROM lcm_trajectory_embeddings
            WHERE profile_digest = ?
            """,
            (str(profile["profile_digest"]),),
        ).fetchall()
        ranked = []
        for row in rows:
            vector = _unpack_vector(bytes(row["vector"]), int(profile["dim"]))
            similarity = sum(left * right for left, right in zip(query_vector, vector))
            ranked.append((int(row["source_id"]), float(similarity)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: self.semantic_top_trajectories]

    @staticmethod
    def _fts_expression(query: str) -> str:
        terms: list[str] = []
        seen: set[str] = set()
        for raw in extract_search_terms(query):
            normalized = raw.casefold().strip()
            if len(normalized) < 2 or normalized in _STOPWORDS or normalized in seen:
                continue
            safe = normalized.replace('"', '""')
            if not any(character.isalnum() for character in safe):
                continue
            seen.add(normalized)
            terms.append(f'"{safe}"')
            if len(terms) >= 16:
                break
        return " OR ".join(terms)

    def _asset_from_row(self, row: sqlite3.Row) -> tuple[str, str] | tuple[None, None]:
        relative = row["relative_path"]
        expected_sha = row["asset_sha256"]
        if relative is None or expected_sha is None:
            return None, None
        candidate = (self.asset_root / str(relative)).resolve()
        try:
            candidate.relative_to(self.asset_root.resolve())
        except ValueError as exc:
            raise TrajectoryAssetError("stored screenshot path escapes asset root") from exc
        if not candidate.is_file() or _sha256_file(candidate) != str(expected_sha):
            raise TrajectoryAssetError(
                f"stored screenshot is missing or changed: {relative}"
            )
        return str(candidate), str(expected_sha)

    def _row_to_hit(
        self,
        row: sqlite3.Row,
        *,
        score: float,
        match_kind: str,
        include_image: bool,
        query: str | None = None,
        text_char_limit: int | None = None,
    ) -> TrajectoryHit:
        corpus_uid = self.corpus_uid
        if not corpus_uid:
            raise CorpusIdentityError("trajectory corpus is not complete")
        trajectory_id = str(row["trajectory_id"])
        state_index = int(row["state_index"])
        encoded_id = quote(trajectory_id, safe="")
        exact_ref = f"trajectory://{corpus_uid}/{encoded_id}/state/{state_index}"
        screenshot_path: str | None = None
        screenshot_sha: str | None = None
        if include_image:
            screenshot_path, screenshot_sha = self._asset_from_row(row)
        full_text = str(row["text"])
        if text_char_limit is None or len(full_text) <= text_char_limit:
            text = full_text
            text_offset = 0
            text_truncated = False
        else:
            text, text_offset = self._exact_excerpt(
                full_text,
                query or "",
                text_char_limit,
            )
            text_truncated = True
        return TrajectoryHit(
            exact_ref=exact_ref,
            trajectory_id=trajectory_id,
            goal=str(row["goal"]),
            outcome=str(row["outcome"]) if row["outcome"] is not None else None,
            state_index=state_index,
            sequence_ordinal=int(row["sequence_ordinal"]),
            step=int(row["step"]),
            url=str(row["url"]),
            incoming_action=(
                str(row["incoming_action"]) if row["incoming_action"] is not None else None
            ),
            thoughts=str(row["thoughts"]) if row["thoughts"] is not None else None,
            text=text,
            text_offset=text_offset,
            text_truncated=text_truncated,
            observed_at=(
                float(row["observed_at"]) if row["observed_at"] is not None else None
            ),
            observed_at_source=(
                str(row["observed_at_source"])
                if row["observed_at_source"] is not None
                else None
            ),
            occurred_at=(
                float(row["occurred_at"]) if row["occurred_at"] is not None else None
            ),
            occurred_at_source=(
                str(row["occurred_at_source"])
                if row["occurred_at_source"] is not None
                else None
            ),
            screenshot_path=screenshot_path,
            screenshot_sha256=screenshot_sha,
            score=float(score),
            match_kind=match_kind,
        )

    @staticmethod
    def _exact_excerpt(text: str, query: str, limit: int) -> tuple[str, int]:
        """Return a bounded verbatim substring centered near the first query match."""
        if len(text) <= limit:
            return text, 0
        folded = text.casefold()
        positions = [
            folded.find(term.casefold())
            for term in extract_search_terms(query)
            if len(term.strip()) >= 2 and folded.find(term.casefold()) >= 0
        ]
        first_match = min(positions) if positions else 0
        start = max(0, first_match - (limit // 3))
        end = min(len(text), start + limit)
        start = max(0, end - limit)
        return text[start:end], start

    @staticmethod
    def _select_diverse(rows: Iterable[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
        selected: list[sqlite3.Row] = []
        per_trajectory: dict[str, int] = {}
        for row in rows:
            trajectory_id = str(row["trajectory_id"])
            if per_trajectory.get(trajectory_id, 0) >= 5:
                continue
            selected.append(row)
            per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _select_with_floor(
        fused_rows: Sequence[sqlite3.Row],
        global_rows: Sequence[sqlite3.Row],
        limit: int,
        floor_k: int,
    ) -> list[sqlite3.Row]:
        """Policy A -- reserve ``floor_k`` nucleus slots for the top pure-BM25
        states, then fill the remainder from the fused order.

        The lexical floor guarantees the strongest lexical winners a slot even
        when the semantic boost would otherwise let a few semantic-top
        trajectories monopolise the nucleus. Both the floor and the fill honour
        the same 5-per-trajectory diversity cap as ``_select_diverse``.
        """
        selected = list(TrajectoryStore._select_diverse(global_rows, floor_k))
        selected_ids = {int(row["state_id"]) for row in selected}
        per_trajectory: dict[str, int] = {}
        for row in selected:
            trajectory_id = str(row["trajectory_id"])
            per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
        for row in fused_rows:
            if len(selected) >= limit:
                break
            state_id = int(row["state_id"])
            if state_id in selected_ids:
                continue
            trajectory_id = str(row["trajectory_id"])
            if per_trajectory.get(trajectory_id, 0) >= 5:
                continue
            selected.append(row)
            selected_ids.add(state_id)
            per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
        return selected[:limit]

    @staticmethod
    def _merge_arms(
        arm_lex: Sequence[sqlite3.Row],
        arm_sem: Sequence[sqlite3.Row],
        limit: int,
        q_lex: int,
        q_sem: int,
        floor_k: int = 0,
    ) -> list[sqlite3.Row]:
        """Policy D -- round-robin a pure-lexical arm and the semantic/fused arm
        into the nucleus by a ``q_lex:q_sem`` quota.

        Strictly generalises Policy A: the lexical arm is guaranteed its quota
        (serving the SOURCE_MISS bucket) while the semantic arm keeps its own
        quota (preserving the semantic gains). Deduped by ``state_id``; a short
        arm is backfilled by the other (a skipped duplicate does not consume a
        quota slot). Arm order is the deterministic tie-break.

        ``floor_k`` composes Policy A on top (the A+D hybrid, issue #127): the
        top ``floor_k`` pure-lexical (BM25) states are reserved a nucleus slot
        FIRST -- protecting the strongest lexical incumbents as Policy A does --
        and the quota round-robin then fills the remaining slots. ``floor_k == 0``
        (default) is byte-identical to the pure Policy D round-robin. The floor
        is drawn from the head of ``arm_lex`` (already 5-per-trajectory
        diversity-capped), matching ``_select_with_floor``'s guaranteed slots.
        """
        selected: list[sqlite3.Row] = []
        seen: set[int] = set()

        def _pull(arm: Sequence[sqlite3.Row], start: int, quota: int) -> int:
            added = 0
            index = start
            while index < len(arm) and added < quota and len(selected) < limit:
                row = arm[index]
                index += 1
                state_id = int(row["state_id"])
                if state_id in seen:
                    continue
                selected.append(row)
                seen.add(state_id)
                added += 1
            return index

        lex_i = sem_i = 0
        floor_k = min(max(0, int(floor_k)), limit)
        if floor_k:
            lex_i = _pull(arm_lex, lex_i, floor_k)
        while len(selected) < limit and (lex_i < len(arm_lex) or sem_i < len(arm_sem)):
            next_lex = _pull(arm_lex, lex_i, q_lex)
            next_sem = _pull(arm_sem, sem_i, q_sem)
            if next_lex == lex_i and next_sem == sem_i:
                break
            lex_i, sem_i = next_lex, next_sem
        return selected[:limit]

    def _fts_rows(
        self,
        expression: str,
        candidate_limit: int,
        *,
        source_ids: Sequence[int] = (),
    ) -> list[sqlite3.Row]:
        source_clause = ""
        params: list[Any] = [expression]
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            source_clause = f" AND s.source_id IN ({placeholders})"
            params.extend(int(source_id) for source_id in source_ids)
        params.append(int(candidate_limit))
        return self._conn.execute(
            f"""
            SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
                   a.relative_path, a.sha256 AS asset_sha256,
                   bm25(lcm_trajectory_states_fts) AS rank
            FROM lcm_trajectory_states_fts
            JOIN lcm_trajectory_states s
              ON s.state_id = lcm_trajectory_states_fts.rowid
            JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
            LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
            WHERE lcm_trajectory_states_fts MATCH ?{source_clause}
            ORDER BY rank ASC, src.ordinal ASC, s.sequence_ordinal ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    def _adjacency_expansion_arm(
        self,
        seed_rows: Sequence[sqlite3.Row],
        radius: int,
    ) -> list[tuple[sqlite3.Row, int, int]]:
        """H5(b) pool-expansion arm (issue #135): sequence neighbors of the
        lexical seed hits, as ``(row, seed_state_id, distance)`` triples.

        Every pool row IS a lexical seed (it entered via ``global_rows`` /
        ``scoped_rows`` FTS), so the seeds are exactly ``seed_rows`` in pool
        (fused-rank) order. For each seed, the states at ``sequence_ordinal
        +/- 1..radius`` WITHIN THE SAME SOURCE are candidates, giving a
        non-lexical recall path: a target state with no query-term match of
        its own is reachable when any state of its trajectory seeds.

        Deterministic arm order is DISTANCE-major, then seed pool rank, then
        ordinal ascending (``-d`` before ``+d``): a +/-1 neighbor of any seed
        outranks a +/-2 neighbor of a stronger seed, mirroring the
        ``ORDER BY ABS(...)`` discipline of the delivery-stage adjacency
        backfill. States already in the pool are excluded; expanded neighbors
        earn NO semantic boost and no BM25 rank of their own (anti-magnet
        control -- they are admitted by the caller only through the
        quota-capped ``_merge_arms`` tail).
        """
        pool_ids = {int(row["state_id"]) for row in seed_rows}
        positions: list[tuple[int, int]] = []
        wanted: set[tuple[int, int]] = set()
        occupied = {
            (int(row["source_id"]), int(row["sequence_ordinal"]))
            for row in seed_rows
        }
        for row in seed_rows:
            source_id = int(row["source_id"])
            ordinal = int(row["sequence_ordinal"])
            for distance in range(1, radius + 1):
                for neighbor in (ordinal - distance, ordinal + distance):
                    key = (source_id, neighbor)
                    if neighbor < 0 or key in occupied or key in wanted:
                        continue
                    wanted.add(key)
                    positions.append(key)
        row_by_position: dict[tuple[int, int], sqlite3.Row] = {}
        chunk = 400  # 2 bound params per pair; stay far below SQLite limits
        for start in range(0, len(positions), chunk):
            batch = positions[start : start + chunk]
            values = ",".join("(?,?)" for _ in batch)
            params: list[int] = []
            for source_id, neighbor in batch:
                params.extend((source_id, neighbor))
            fetched = self._conn.execute(
                f"""
                SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
                       a.relative_path, a.sha256 AS asset_sha256, 0.0 AS rank
                FROM lcm_trajectory_states s
                JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
                LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
                WHERE (s.source_id, s.sequence_ordinal) IN (VALUES {values})
                """,
                params,
            ).fetchall()
            for fetched_row in fetched:
                row_by_position[
                    (int(fetched_row["source_id"]), int(fetched_row["sequence_ordinal"]))
                ] = fetched_row
        arm: list[tuple[sqlite3.Row, int, int]] = []
        emitted: set[int] = set(pool_ids)
        for distance in range(1, radius + 1):
            for row in seed_rows:
                source_id = int(row["source_id"])
                ordinal = int(row["sequence_ordinal"])
                for neighbor in (ordinal - distance, ordinal + distance):
                    neighbor_row = row_by_position.get((source_id, neighbor))
                    if neighbor_row is None:
                        continue
                    state_id = int(neighbor_row["state_id"])
                    if state_id in emitted:
                        continue
                    emitted.add(state_id)
                    arm.append((neighbor_row, int(row["state_id"]), distance))
        return arm

    def query(
        self,
        query: str,
        *,
        candidate_limit: int = 128,
        limit: int = 16,
        image_limit: int = 8,
        include_adjacent: bool = True,
        text_char_limit: int = 2_000,
        lexical_floor: int = 0,
        arm_quota: tuple[int, int] | None = None,
        adjacency_radius: int = 0,
        adjacency_quota: int = 0,
    ) -> tuple[TrajectoryHit, ...]:
        if self.status != "complete":
            raise CorpusIdentityError("trajectory corpus must be finalized before query")
        candidate_limit = min(max(1, int(candidate_limit)), _MAX_CANDIDATES)
        limit = min(max(1, int(limit)), _MAX_RESULTS)
        image_limit = min(max(0, int(image_limit)), _MAX_IMAGES)
        lexical_floor = min(max(0, int(lexical_floor)), _MAX_RESULTS)
        adjacency_radius = min(max(0, int(adjacency_radius)), _MAX_ADJACENCY_RADIUS)
        adjacency_quota = min(max(0, int(adjacency_quota)), _MAX_CANDIDATES)
        text_char_limit = min(
            max(256, int(text_char_limit)),
            _MAX_QUERY_TEXT_CHARS,
        )
        expression = self._fts_expression(query)
        if not expression:
            self._last_query_telemetry = {
                "semantic_attempt": None,
                "source_candidate_ranks": [],
                "state_candidate_pool": [],
                "delivered_evidence_refs": [],
            }
            return ()
        global_rows = self._fts_rows(expression, candidate_limit)
        semantic_ranks: list[tuple[int, float]] = []
        semantic_attempt: TrajectorySemanticAttempt | None = None
        attempt_started = time.monotonic()
        calls_before = self._semantic_usage["query_calls"]
        try:
            semantic_ranks = self._semantic_source_ranks(query)
        except Exception as exc:
            # Restore historical semantics FIRST, unconditionally, before any
            # introspection can fail: the fallback counter must bump even if the
            # (hostile) exception explodes during telemetry recording.
            self._semantic_usage["fallbacks"] += 1
            fallback_latency_ms = (time.monotonic() - attempt_started) * 1000.0
            try:
                # Was a bare ``except Exception: fallbacks += 1`` that discarded
                # the failure class/status. Now the typed reason survives -- and
                # the whole record step is itself fenced so an exotic exception
                # (kind/status_code/retry_after as raising properties) degrades
                # to a clean FTS fallback instead of failing the query.
                semantic_attempt = self._record_semantic_attempt(
                    outcome="fallback",
                    latency_ms=fallback_latency_ms,
                    exception=exc,
                )
            except Exception:
                semantic_attempt = self._record_minimal_fallback_attempt(
                    latency_ms=fallback_latency_ms,
                )
        else:
            # Only record a success when an embed was actually dispatched; an
            # early return (no provider / profile mismatch) is a skip, not an
            # attempt, and must not inflate the success count.
            if self._semantic_usage["query_calls"] > calls_before:
                semantic_attempt = self._record_semantic_attempt(
                    outcome="success",
                    latency_ms=(time.monotonic() - attempt_started) * 1000.0,
                    exception=None,
                )
        scoped_rows = self._fts_rows(
            expression,
            candidate_limit,
            source_ids=[source_id for source_id, _score in semantic_ranks],
        ) if semantic_ranks else []

        if semantic_ranks and scoped_rows:
            row_by_id: dict[int, sqlite3.Row] = {}
            score_by_candidate: dict[int, float] = {}
            semantic_position = {
                source_id: position
                for position, (source_id, _score) in enumerate(semantic_ranks, start=1)
            }
            for position, row in enumerate(global_rows, start=1):
                state_id = int(row["state_id"])
                row_by_id[state_id] = row
                score_by_candidate[state_id] = score_by_candidate.get(state_id, 0.0) + (
                    1.0 / (60.0 + position)
                )
            for position, row in enumerate(scoped_rows, start=1):
                state_id = int(row["state_id"])
                row_by_id[state_id] = row
                score_by_candidate[state_id] = score_by_candidate.get(state_id, 0.0) + (
                    1.0 / (60.0 + position)
                )
                trajectory_position = semantic_position.get(int(row["source_id"]), 32)
                score_by_candidate[state_id] += 1.0 / (60.0 + trajectory_position)
            rows = sorted(
                row_by_id.values(),
                key=lambda row: (
                    -score_by_candidate[int(row["state_id"])],
                    int(row["ordinal"]),
                    int(row["sequence_ordinal"]),
                ),
            )
            candidate_kind = {
                int(row["state_id"]): "semantic_fts"
                for row in scoped_rows
            }
            candidate_score = {
                state_id: -score
                for state_id, score in score_by_candidate.items()
            }
        else:
            rows = global_rows
            candidate_kind = {}
            candidate_score = {
                int(row["state_id"]): float(row["rank"])
                for row in rows
            }

        # H5(b) lexical-seed adjacency pool-expansion (issue #135): pull the
        # sequence neighbors of the lexical seed hits INTO the state pool,
        # pre-selection, as a QUOTA-CAPPED ADDITIVE arm through the existing
        # ``_merge_arms`` machinery. The arm is appended strictly AFTER the
        # ranked pool (no semantic boost, no BM25 rank of its own), so the
        # nucleus selection -- and therefore delivery -- only changes when the
        # ranked pool alone cannot fill the nucleus; the 5-per-trajectory
        # diversity cap at selection is untouched. ``adjacency_radius == 0``
        # or ``adjacency_quota == 0`` (defaults) skip this path entirely and
        # reproduce current bytes.
        adjacency_admitted: list[dict[str, int]] = []
        if adjacency_radius > 0 and adjacency_quota > 0 and rows:
            arm_triples = self._adjacency_expansion_arm(rows, adjacency_radius)
            if arm_triples:
                arm_adjacent = [triple[0] for triple in arm_triples]
                seed_by_state = {
                    int(triple[0]["state_id"]): (triple[1], triple[2])
                    for triple in arm_triples
                }
                expanded = self._merge_arms(
                    rows,
                    arm_adjacent,
                    len(rows) + adjacency_quota,
                    len(rows),
                    adjacency_quota,
                )
                for row in expanded[len(rows):]:
                    state_id = int(row["state_id"])
                    seed_state_id, distance = seed_by_state[state_id]
                    candidate_kind[state_id] = "adjacent"
                    candidate_score[state_id] = (
                        candidate_score[seed_state_id] + 0.000001 * distance
                    )
                    adjacency_admitted.append({
                        "state_id": state_id,
                        "seed_state_id": seed_state_id,
                        "distance": distance,
                    })
                rows = expanded

        adjacent_reserve = min(6, limit // 3) if include_adjacent else 0
        nucleus_limit = max(1, limit - adjacent_reserve)
        if arm_quota is not None:
            # Policy D (candidate-composition repair, issue #127): round-robin a
            # pure-lexical arm and the semantic/fused arm into the nucleus by the
            # requested quota. Superset of Policy A; ``arm_quota is None``
            # (default) is byte-identical to the historical selection below.
            # When ``lexical_floor > 0`` is ALSO supplied this becomes the A+D
            # hybrid: the top ``lexical_floor`` pure-BM25 incumbents are reserved
            # a slot first, then the quota round-robin fills the rest
            # (``lexical_floor == 0`` reproduces the pure Policy D bytes).
            q_lex = max(0, int(arm_quota[0]))
            q_sem = max(0, int(arm_quota[1]))
            arm_lex = self._select_diverse(global_rows, nucleus_limit)
            arm_sem = self._select_diverse(rows, nucleus_limit)
            selected = self._merge_arms(
                arm_lex, arm_sem, nucleus_limit, q_lex, q_sem,
                floor_k=lexical_floor,
            )
        elif lexical_floor > 0:
            # Policy A (candidate-composition repair, issue #127): guarantee the
            # top pure-BM25 states a nucleus slot before the fused order fills
            # the rest. ``lexical_floor == 0`` (default) is byte-identical to the
            # historical fused-only selection below.
            selected = self._select_with_floor(
                rows, global_rows, nucleus_limit, lexical_floor
            )
        else:
            selected = self._select_diverse(rows, nucleus_limit)
        selected_ids = {int(row["state_id"]) for row in selected}
        match_kind_by_id = {
            int(row["state_id"]): candidate_kind.get(int(row["state_id"]), "fts")
            for row in selected
        }
        score_by_id = {
            int(row["state_id"]): candidate_score[int(row["state_id"])]
            for row in selected
        }

        if include_adjacent and selected and len(selected) < limit:
            nucleus_rows = list(selected)
            adjacent_by_nucleus: list[list[sqlite3.Row]] = []
            for nucleus in nucleus_rows:
                adjacent_rows = self._conn.execute(
                    """
                    SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
                           a.relative_path, a.sha256 AS asset_sha256, 0.0 AS rank
                    FROM lcm_trajectory_states s
                    JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
                    LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
                    WHERE s.source_id = ? AND s.sequence_ordinal IN (?, ?)
                    ORDER BY ABS(s.sequence_ordinal - ?), s.sequence_ordinal
                    """,
                    (
                        int(nucleus["source_id"]),
                        int(nucleus["sequence_ordinal"]) - 1,
                        int(nucleus["sequence_ordinal"]) + 1,
                        int(nucleus["sequence_ordinal"]),
                    ),
                ).fetchall()
                adjacent_by_nucleus.append(list(adjacent_rows))
            while len(selected) < limit and any(adjacent_by_nucleus):
                made_progress = False
                for nucleus, adjacent_rows in zip(nucleus_rows, adjacent_by_nucleus):
                    while adjacent_rows:
                        row = adjacent_rows.pop(0)
                        state_id = int(row["state_id"])
                        if state_id in selected_ids:
                            continue
                        selected.append(row)
                        selected_ids.add(state_id)
                        match_kind_by_id[state_id] = "adjacent"
                        score_by_id[state_id] = score_by_id[int(nucleus["state_id"])] + 0.000001
                        made_progress = True
                        break
                    if len(selected) >= limit:
                        break
                if not made_progress:
                    break

        hits: list[TrajectoryHit] = []
        for index, row in enumerate(selected[:limit]):
            state_id = int(row["state_id"])
            hits.append(self._row_to_hit(
                row,
                score=score_by_id[state_id],
                match_kind=match_kind_by_id[state_id],
                include_image=index < image_limit,
                query=query,
                text_char_limit=text_char_limit,
            ))

        # Side-channel per-query telemetry (does not affect the returned hits).
        self._last_query_telemetry = {
            "semantic_attempt": (
                asdict(semantic_attempt) if semantic_attempt is not None else None
            ),
            "source_candidate_ranks": [
                {"source_id": int(source_id), "rank": position, "score": float(score)}
                for position, (source_id, score) in enumerate(
                    semantic_ranks[:64], start=1
                )
            ],
            "state_candidate_pool": [
                {
                    "state_id": int(row["state_id"]),
                    "rank": position,
                    "score": float(candidate_score.get(int(row["state_id"]), 0.0)),
                }
                for position, row in enumerate(rows[:64], start=1)
            ],
            "delivered_evidence_refs": [hit.exact_ref for hit in hits],
        }
        if adjacency_radius > 0 and adjacency_quota > 0:
            # Present only when the H5(b) knob is active so the default
            # telemetry payload stays byte-identical (golden 451/451).
            self._last_query_telemetry["adjacency_expansion"] = {
                "radius": adjacency_radius,
                "quota": adjacency_quota,
                "admitted": adjacency_admitted,
            }
        return tuple(hits)

    def resolve_exact_ref(self, exact_ref: str) -> TrajectoryHit:
        match = _EXACT_REF_RE.fullmatch(str(exact_ref or "").strip())
        if not match or match.group("corpus") != self.corpus_uid:
            raise ExactTrajectoryRefError("exact trajectory ref does not match this corpus")
        trajectory_id = unquote(match.group("trajectory"))
        state_index = int(match.group("state"))
        row = self._conn.execute(
            """
            SELECT s.*, src.trajectory_id, src.goal, src.outcome, a.relative_path,
                   a.sha256 AS asset_sha256
            FROM lcm_trajectory_states s
            JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
            LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
            WHERE src.trajectory_id = ? AND s.state_index = ?
            """,
            (trajectory_id, state_index),
        ).fetchone()
        if row is None:
            raise ExactTrajectoryRefError("exact trajectory ref does not resolve")
        return self._row_to_hit(
            row,
            score=0.0,
            match_kind="exact_ref",
            include_image=True,
        )

    @staticmethod
    def query_digest(hits: Sequence[TrajectoryHit]) -> str:
        portable_hits: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.to_dict()
            payload.pop("screenshot_path", None)
            portable_hits.append(payload)
        return _sha256_text(_canonical_json(portable_hits))

    def manifest(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        semantic = self._semantic_profile()
        return {
            "identity": dict(self.identity_payload),
            "identity_digest": str(row["identity_digest"]),
            "schema_version": int(row["schema_version"]),
            "corpus_uid": row["corpus_uid"],
            "haystack_digest": row["haystack_digest"],
            "source_manifest_digest": row["source_manifest_digest"],
            "trajectory_count": row["trajectory_count"],
            "ingest_cursor": int(row["ingest_cursor"]),
            "status": str(row["status"]),
            "semantic_index": (
                {
                    "profile_digest": str(semantic["profile_digest"]),
                    "provider": str(semantic["provider"]),
                    "model_name": str(semantic["model_name"]),
                    "dim": int(semantic["dim"]),
                    "document_version": str(semantic["document_version"]),
                    "document_count": int(semantic["document_count"]),
                    "index_digest": str(semantic["index_digest"]),
                }
                if semantic is not None
                else None
            ),
        }

    def backup_to(self, destination: str | Path) -> Path:
        self._require_writable()
        destination_path = Path(destination)
        if destination_path.resolve() == self.db_path.resolve():
            raise ValueError("trajectory backup destination must differ from source")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            raise FileExistsError(destination_path)
        with self._lock:
            self._conn.commit()
            target = sqlite3.connect(str(destination_path))
            try:
                self._conn.backup(target)
                target.commit()
            finally:
                target.close()
        return destination_path

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            try:
                if not self.read_only:
                    conn.commit()
            finally:
                conn.close()
                self._conn = None  # type: ignore[assignment]
