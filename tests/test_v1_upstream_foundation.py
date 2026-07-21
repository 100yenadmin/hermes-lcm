"""Focused contracts for the reconstructed V1 upstream foundation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from types import SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore


@pytest.fixture
def recall_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "recall.db"),
        embeddings_enabled=False,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(tmp_path),
        current_session_id="current-session",
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


def test_answer_ready_is_opt_in_and_default_bytes_are_unchanged(recall_engine):
    recall_engine._store.append(
        "session-a",
        {"role": "user", "content": "The kanban dashboard sprint is ready."},
    )
    args = {"query": "kanban dashboard sprint", "include": "verbatim", "limit": 1}

    implicit = lcm_tools.lcm_recall(args, engine=recall_engine)
    explicit = lcm_tools.lcm_recall({**args, "detail": "snippets"}, engine=recall_engine)

    assert implicit == explicit
    payload = json.loads(implicit)
    assert "detail" not in payload
    assert "content" not in payload["hits"][0]


def test_answer_ready_returns_exact_grounding_and_separate_times(recall_engine):
    observed_at = datetime(2024, 3, 20, tzinfo=timezone.utc).timestamp()
    content = "I finished the kanban dashboard sprint 5 days ago."
    store_id = recall_engine._store.append(
        "session-a",
        {
            "role": "user",
            "content": content,
            "timestamp": observed_at,
        },
        source="chat",
    )

    payload = json.loads(
        lcm_tools.lcm_recall(
            {
                "query": "kanban dashboard sprint",
                "include": "verbatim",
                "limit": 1,
                "detail": "answer_ready",
                "include_occurrence_time": True,
            },
            engine=recall_engine,
        )
    )

    hit = payload["hits"][0]
    assert hit["store_id"] == store_id
    assert hit["exact_ref"].startswith(f"lcm:{store_id}:")
    assert hit["content"] == content
    assert hit["source_provenance"] == {
        "store_id": store_id,
        "session_id": "session-a",
        "source": "chat",
        "role": "user",
    }
    assert hit["observation_time"]["observed_at"] == observed_at
    assert hit["observation_time"]["ingested_at"] == hit["timestamp"]
    assert hit["observation_time"]["observed_at"] != hit["observation_time"]["ingested_at"]
    assert hit["occurrence_time"]["occurred_at"] != observed_at
    assert hit["occurrence_time"]["event_date"] == "2024-03-15"


def test_answer_ready_rejects_invalid_detail(recall_engine):
    payload = json.loads(
        lcm_tools.lcm_recall(
            {"query": "kanban", "detail": "full-transcript"},
            engine=recall_engine,
        )
    )
    assert payload["error"] == "detail must be one of: snippets, answer_ready"


def test_exact_ref_resolution_is_content_bound_and_fails_after_source_deletion(tmp_path):
    from hermes_lcm.exact_refs import resolve_exact_ref

    store = MessageStore(tmp_path / "exact.db")
    try:
        content = "The first draft was 12 pages."
        store_id = store.append("session-a", {"role": "user", "content": content})
        start = content.index("12 pages")
        exact_ref = f"lcm:{store_id}:{start}-{start + len('12 pages')}"

        resolved, error = resolve_exact_ref(
            store,
            {"exact_ref": exact_ref, "quote": "12 pages"},
        )
        assert error is None
        assert resolved is not None
        assert resolved.exact_ref == exact_ref
        assert resolved.quote == "12 pages"

        stale, error = resolve_exact_ref(
            store,
            {"exact_ref": exact_ref, "quote": "13 pages"},
        )
        assert stale is None
        assert error == "quote does not match the exact source span"

        store._conn.execute("DELETE FROM messages WHERE store_id = ?", (store_id,))
        store._conn.commit()
        missing, error = resolve_exact_ref(
            store,
            {"exact_ref": exact_ref, "quote": "12 pages"},
        )
        assert missing is None
        assert error == "exact source row does not exist"
    finally:
        store.close()


def test_exact_ref_recovery_is_opt_in_and_tracks_page_offsets(recall_engine):
    content = "alpha beta gamma delta"
    store_id = recall_engine._store.append(
        "session-a", {"role": "user", "content": content}
    )

    load_args = {"session_id": "session-a", "max_content_chars": 10}
    implicit = lcm_tools.lcm_load_session(load_args, engine=recall_engine)
    explicit_default = lcm_tools.lcm_load_session(
        {**load_args, "include_exact_ref": False}, engine=recall_engine
    )
    assert implicit == explicit_default
    loaded = json.loads(
        lcm_tools.lcm_load_session(
            {**load_args, "include_exact_ref": True}, engine=recall_engine
        )
    )["messages"][0]
    assert loaded["content"] == content[:10]
    assert loaded["exact_ref"] == f"lcm:{store_id}:0-10"

    expand_args = {"store_id": store_id, "content_offset": 6, "max_tokens": 2}
    implicit_expand = lcm_tools.lcm_expand(expand_args, engine=recall_engine)
    explicit_expand = lcm_tools.lcm_expand(
        {**expand_args, "include_exact_ref": False}, engine=recall_engine
    )
    assert implicit_expand == explicit_expand
    expanded = json.loads(
        lcm_tools.lcm_expand(
            {**expand_args, "include_exact_ref": True}, engine=recall_engine
        )
    )
    assert expanded["content"] == content[6 : 6 + expanded["content_returned_chars"]]
    assert expanded["exact_ref"] == (
        f"lcm:{store_id}:6-{6 + expanded['content_returned_chars']}"
    )


def test_exact_ref_paths_never_restore_secrets_redacted_before_storage(tmp_path):
    secret = "sk-test-secret-value-123456"
    config = LCMConfig(
        database_path=str(tmp_path / "secret.db"),
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key"],
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    engine = SimpleNamespace(_store=store, current_session_id="session-a")
    try:
        store_id = store.append(
            "session-a",
            {"role": "user", "content": f"api_key={secret} for the dashboard"},
        )
        loaded = json.loads(
            lcm_tools.lcm_load_session(
                {"session_id": "session-a", "include_exact_ref": True},
                engine=engine,
            )
        )["messages"][0]
        assert secret not in loaded["content"]
        assert "[LCM sensitive redaction:" in loaded["content"]
        assert loaded["exact_ref"] == f"lcm:{store_id}:0-{len(loaded['content'])}"
    finally:
        store.close()


def test_pure_computation_uses_exact_refs_and_fails_closed(tmp_path):
    store = MessageStore(tmp_path / "compute.db")
    engine = SimpleNamespace(_store=store)
    try:
        first = "The first draft was 12 pages."
        second = "The appendix was 8 pages."
        first_id = store.append("session-a", {"role": "user", "content": first})
        second_id = store.append("session-a", {"role": "assistant", "content": second})
        first_start = first.index("12 pages")
        second_start = second.index("8 pages")
        args = {
            "question": "What is the total number of pages?",
            "evidence_complete": True,
            "operands": [
                {
                    "exact_ref": f"lcm:{first_id}:{first_start}-{first_start + 8}",
                    "quote": "12 pages",
                    "value": 12,
                    "unit": "pages",
                },
                {
                    "exact_ref": f"lcm:{second_id}:{second_start}-{second_start + 7}",
                    "quote": "8 pages",
                    "value": 8,
                    "unit": "pages",
                },
            ],
        }

        computed = json.loads(lcm_tools.lcm_compute(args, engine=engine))
        assert computed["status"] == "computed"
        assert computed["trace"]["result"] == "20 pages"
        assert computed["provenance"]["runtime_inputs"] == [
            "question",
            "question_date",
            "exact_retrieved_evidence",
        ]

        ambiguous = json.loads(
            lcm_tools.lcm_compute(
                {
                    **args,
                    "operands": [{**args["operands"][0], "value": 13}],
                },
                engine=engine,
            )
        )
        assert ambiguous["status"] == "fallback"
        assert "not explicit" in ambiguous["reason"]
    finally:
        store.close()


def test_question_as_of_is_distinct_and_excludes_later_observations(tmp_path):
    store = MessageStore(tmp_path / "as-of.db")
    engine = SimpleNamespace(_store=store)
    try:
        content = "The report was 4 pages."
        store_id = store.append(
            "session-a",
            {
                "role": "user",
                "content": content,
                "timestamp": "2024-03-20T10:00:00Z",
            },
        )
        start = content.index("4 pages")
        payload = json.loads(
            lcm_tools.lcm_compute(
                {
                    "question": "What is the total number of pages?",
                    "question_date": "2024-03-01",
                    "evidence_complete": True,
                    "operands": [
                        {
                            "exact_ref": f"lcm:{store_id}:{start}-{start + 7}",
                            "quote": "4 pages",
                            "value": 4,
                            "unit": "pages",
                        }
                    ],
                },
                engine=engine,
            )
        )
        assert payload["status"] == "fallback"
        assert "after the question-date boundary" in payload["reason"]
    finally:
        store.close()


def test_time_sidecars_migrate_legacy_rows_and_survive_sqlite_backup(tmp_path):
    legacy_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy_path)
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO messages(session_id, role, content, timestamp)
        VALUES ('legacy', 'user', 'old row', 123.5);
        """
    )
    conn.commit()
    conn.close()

    migrated = MessageStore(legacy_path)
    backup_path = tmp_path / "backup.db"
    try:
        row = migrated.get(1)
        assert row["timestamp"] == 123.5
        assert row["ingested_at"] == 123.5
        assert row["observed_at"] is None
        destination = sqlite3.connect(backup_path)
        try:
            migrated._conn.backup(destination)
        finally:
            destination.close()
    finally:
        migrated.close()

    restored = MessageStore(backup_path)
    try:
        restored_row = restored.get(1)
        assert restored_row["ingested_at"] == 123.5
        assert restored_row["observed_at"] is None
    finally:
        restored.close()


def test_occurrence_time_ambiguity_never_relabels_observation_time():
    from hermes_lcm.occurrence_time import resolve_occurrence_time

    observed_at = datetime(2024, 3, 20, 17, 45, tzinfo=timezone.utc).timestamp()
    result = resolve_occurrence_time(
        "We met on 2024-03-01 and 2024-03-02.",
        observed_at=observed_at,
        session_date="2024-03-20",
    )
    assert result["observed_at"] == observed_at
    assert result["stored_at"] is None
    assert result["occurred_at"] is None
    assert result["event_time_source"] == "unknown"
    assert result["reason"] == "ambiguous_multiple_explicit_dates"
