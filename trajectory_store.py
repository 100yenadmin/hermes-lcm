"""Immutable trajectory sources and bounded exact retrieval in one ``lcm.db``.

The store is intentionally provider-free.  It models agent trajectories as
first-class source material instead of flattening them into chat messages.
One database owns one corpus identity; normalized states and image manifests
remain traceable to protected canonical source JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Iterable, Sequence
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
_MAX_CANDIDATES = 128
_MAX_RESULTS = 24
_MAX_IMAGES = 8
_MAX_QUERY_TEXT_CHARS = 8_000
_MAX_SOURCE_JSON_CHARS = 16_000_000
_MAX_TEXT_CHARS = 2_000_000
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
    ) -> None:
        self.db_path = Path(db_path)
        self.identity = identity
        self.identity_payload = identity.to_dict()
        self.identity_digest = identity.digest
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.read_only = bool(read_only)
        self.protect_sensitive = bool(protect_sensitive)
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

    def query(
        self,
        query: str,
        *,
        candidate_limit: int = 128,
        limit: int = 16,
        image_limit: int = 8,
        include_adjacent: bool = True,
        text_char_limit: int = 2_000,
    ) -> tuple[TrajectoryHit, ...]:
        if self.status != "complete":
            raise CorpusIdentityError("trajectory corpus must be finalized before query")
        candidate_limit = min(max(1, int(candidate_limit)), _MAX_CANDIDATES)
        limit = min(max(1, int(limit)), _MAX_RESULTS)
        image_limit = min(max(0, int(image_limit)), _MAX_IMAGES)
        text_char_limit = min(
            max(256, int(text_char_limit)),
            _MAX_QUERY_TEXT_CHARS,
        )
        expression = self._fts_expression(query)
        if not expression:
            return ()
        rows = self._conn.execute(
            """
            SELECT s.*, src.trajectory_id, src.goal, src.outcome, a.relative_path,
                   a.sha256 AS asset_sha256, bm25(lcm_trajectory_states_fts) AS rank
            FROM lcm_trajectory_states_fts
            JOIN lcm_trajectory_states s ON s.state_id = lcm_trajectory_states_fts.rowid
            JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
            LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
            WHERE lcm_trajectory_states_fts MATCH ?
            ORDER BY rank ASC, src.ordinal ASC, s.sequence_ordinal ASC
            LIMIT ?
            """,
            (expression, candidate_limit),
        ).fetchall()
        selected = self._select_diverse(rows, limit)
        selected_ids = {int(row["state_id"]) for row in selected}
        match_kind_by_id = {int(row["state_id"]): "fts" for row in selected}
        score_by_id = {int(row["state_id"]): float(row["rank"]) for row in selected}

        if include_adjacent and len(selected) < limit:
            nucleus_rows = list(selected)
            for nucleus in nucleus_rows:
                adjacent_rows = self._conn.execute(
                    """
                    SELECT s.*, src.trajectory_id, src.goal, src.outcome, a.relative_path,
                           a.sha256 AS asset_sha256, 0.0 AS rank
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
                for row in adjacent_rows:
                    state_id = int(row["state_id"])
                    if state_id in selected_ids:
                        continue
                    selected.append(row)
                    selected_ids.add(state_id)
                    match_kind_by_id[state_id] = "adjacent"
                    score_by_id[state_id] = float(nucleus["rank"]) + 0.000001
                    if len(selected) >= limit:
                        break
                if len(selected) >= limit:
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
