"""Shared retrieval plumbing for LCM search tools.

This module factors the retrieval/fusion core out of ``tools.py`` so that
``lcm_grep`` (and the forthcoming ``lcm_recall``) call one engine instead of
duplicating ranking logic. Everything here is a pure move from ``tools.py`` —
callers keep their existing contracts, guards, and error strings; only the
plumbing relocated.

Guards (role/time/conversation/scope/content_scope degrade matrix) intentionally
stay in ``tools.py``: they are contract, not plumbing. So do the bounded worker /
deadline machinery (``_run_within_deadline`` and its semaphores) and the
provider-resolution / query-embedding steps, because their module-level names are
monkeypatched through the ``tools`` namespace and the semaphore default binding is
lexical to that module.
"""

from __future__ import annotations

import copy
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import LCMEngine


def _lcm_grep_confidence(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.35:
        return "low"
    return "noise"


def _lcm_grep_deadline_error(mode: str, stage: str) -> dict[str, Any]:
    return {
        "error": "lcm_grep request deadline exceeded",
        "mode": mode,
        "timeout": True,
        "timeout_stage": stage,
    }


def _shape_message_hit(
    hit: Dict[str, Any],
    *,
    current_session_id: str | None,
    has_current_session: bool,
) -> dict[str, Any]:
    """Shape a raw MessageStore hit into an lcm_grep result row."""
    timestamp_value = hit.get("timestamp", 0) or 0
    return {
        "type": "message",
        "depth": "raw",
        "store_id": hit["store_id"],
        "session_id": hit["session_id"],
        "source": hit.get("source") or "",
        "conversation_id": hit.get("conversation_id") or "",
        "role": hit["role"],
        "timestamp": timestamp_value,
        "snippet": hit.get("snippet", hit.get("content", "")[:200]),
        "from_current_session": has_current_session
        and hit["session_id"] == current_session_id,
        "_sort_ts": timestamp_value,
        "_sort_rank": hit.get("search_rank"),
        "_sort_directness": hit.get("_directness_score") or 0.0,
    }


def _shape_summary_hit(node: Any) -> dict[str, Any]:
    """Shape a SummaryDAG node hit into an lcm_grep result row."""
    return {
        "type": "summary",
        "depth": f"d{node.depth}",
        "node_id": node.node_id,
        "session_id": node.session_id,
        "snippet": node.summary[:300],
        "token_count": node.token_count,
        "expand_hint": node.expand_hint,
        "earliest_at": node.earliest_at,
        "latest_at": node.latest_at,
        "from_current_session": True,
        "_sort_ts": node.latest_at or node.created_at,
        "_sort_rank": node.search_rank,
        "_sort_directness": node.search_directness or 0.0,
    }


def _resolve_semantic_conversation_scope(
    engine: "LCMEngine",
    *,
    search_session_id: str | None,
    conversation_id: str | None,
) -> list[str] | None:
    """Resolve the conversation filter to the session_ids KNN should allow.

    Summaries are keyed by session_id, so a message-level ``conversation_id`` is
    enforced by resolving it to the sessions that carry it (intersected with the
    active scope). Returns ``None`` for "no session constraint", or a possibly
    empty list when a conversation matches no sessions (which then degrades).
    """
    if not conversation_id:
        return [search_session_id] if search_session_id is not None else None
    try:
        rows = engine._store.connection.execute(
            "SELECT DISTINCT session_id FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        conv_sessions = {str(row[0]) for row in rows if row and row[0] is not None}
    except Exception:  # pragma: no cover - defensive; degrade on any store error
        conv_sessions = set()
    if search_session_id is not None:
        return sorted(conv_sessions & {search_session_id})
    return sorted(conv_sessions)


def run_knn(
    engine: "LCMEngine",
    *,
    query_vector: list[float],
    provider: Any,
    knn_limit: int,
    deadline: float,
    since: float | None,
    until: float | None,
    conversation_ids: list[str] | None,
    source: str | None,
    vector_store_cls: Any,
) -> Any:
    """Run the vector KNN query inside the operation's absolute deadline.

    ``vector_store_cls`` is injected so callers (and their tests) keep resolving
    the VectorStore binding through the ``tools`` module namespace.
    """
    if time.monotonic() >= deadline:
        raise TimeoutError("semantic vector search deadline exhausted")
    vector_store = vector_store_cls(engine._store.db_path, config=engine._config)
    try:
        vector_conn = getattr(vector_store, "_conn", None)
        if vector_conn is not None:
            vector_conn.set_progress_handler(
                lambda: 1 if time.monotonic() >= deadline else 0,
                1000,
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("semantic vector search deadline exhausted")
        return vector_store.knn(
            query_vector,
            k=knn_limit,
            model=provider.model_id,
            provider=provider.provider_id,
            since=since,
            until=until,
            conversation_ids=conversation_ids,
            source=source,
        )
    finally:
        vector_store.close()


def hydrate_semantic_nodes(
    engine: "LCMEngine",
    *,
    ranked_rows: list[Any],
    knn_limit: int,
    deadline: float,
) -> list[tuple[Any, float]]:
    """Hydrate ranked vector hits into summary nodes on a read-only connection."""
    conn: sqlite3.Connection | None = None

    def require_remaining(stage: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"semantic result hydration deadline exhausted before {stage}"
            )
        return remaining

    try:
        require_remaining("database path resolution")
        db_path = Path(engine._store.db_path).resolve()
        require_remaining("database path resolution")
        uri = f"{db_path.as_uri()}?mode=ro"
        require_remaining("database URI construction")
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=max(0.001, require_remaining("database connection")),
        )
        require_remaining("database connection")
        conn.row_factory = sqlite3.Row
        require_remaining("connection setup")
        conn.execute("PRAGMA query_only=ON")
        require_remaining("connection setup")
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() >= deadline else 0,
            1000,
        )
        require_remaining("DAG setup")
        read_dag = copy.copy(engine._dag)
        read_dag._conn = conn
        read_dag._db_lock = threading.RLock()
        require_remaining("DAG setup")
        hydrated: list[tuple[Any, float]] = []
        for embedded_id, score, kind in ranked_rows:
            require_remaining("node lookup")
            if kind != "summary":
                continue
            try:
                node_id = int(embedded_id)
            except (TypeError, ValueError):
                continue
            require_remaining("node lookup")
            node = read_dag.get_node(node_id)
            require_remaining("node lookup")
            if node is not None:
                hydrated.append((node, float(score)))
            if len(hydrated) >= knn_limit:
                break
        return hydrated
    finally:
        if conn is not None:
            conn.close()


def _hit_identity(hit: dict[str, Any]) -> tuple[str, Any]:
    if hit.get("node_id") is not None:
        return ("node", hit.get("node_id"))
    return ("message", hit.get("store_id"))


def rrf_fuse(arms: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion over ranked hit arms, keyed by hit identity.

    Each entry accumulates ``1 / (k + rank)`` across the arms it appears in.
    Returns entries ordered by descending RRF score with the source hit under
    ``hit``, the fused score under ``rrf_score``, and each arm's 1-based rank in
    ``ranks`` (keyed by arm index). Callers own arm-specific metadata (which arm
    is FTS vs semantic, confidence, snippet provenance).
    """
    fused: dict[tuple[str, Any], dict[str, Any]] = {}
    for arm_index, arm in enumerate(arms):
        for rank, hit in enumerate(arm, start=1):
            key = _hit_identity(hit)
            entry = fused.setdefault(
                key, {"hit": dict(hit), "rrf_score": 0.0, "ranks": {}}
            )
            entry["ranks"][arm_index] = rank
            entry["rrf_score"] += 1.0 / (k + rank)
    return sorted(
        fused.values(),
        key=lambda entry: (
            -float(entry["rrf_score"]),
            *(int(entry["ranks"].get(i, 10**9)) for i in range(len(arms))),
            _hit_identity(entry["hit"]),
        ),
    )
