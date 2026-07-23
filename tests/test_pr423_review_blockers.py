"""Regressions for the current-head PR #423 fail-closed review blockers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.store import MessageStore


_IDENTITY_FIELDS = (
    "store_id",
    "session_id",
    "source",
    "conversation_id",
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "ingested_at",
    "observed_at",
    "observed_at_source",
)


def _evidence_identity(row):
    payload = {field: row.get(field) for field in _IDENTITY_FIELDS}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _operand(store, store_id, content, *, value=None, unit=None, label=None):
    operand = {
        "exact_ref": f"lcm:{store_id}:0-{len(content)}",
        "evidence_identity": _evidence_identity(store.get(store_id)),
    }
    if value is not None:
        operand["value"] = value
    if unit is not None:
        operand["unit"] = unit
    if label is not None:
        operand["label"] = label
    return operand


@pytest.fixture
def compute_store(tmp_path):
    store = MessageStore(tmp_path / "compute.db")
    try:
        yield store, SimpleNamespace(_store=store)
    finally:
        store.close()


def test_compute_rejects_identity_after_content_rewrite(compute_store):
    store, engine = compute_store
    first = "Alice amount $12"
    second = "Bob amount $8"
    first_id = store.append("session-a", {"role": "tool", "content": first})
    second_id = store.append("session-a", {"role": "tool", "content": second})
    first_identity = _evidence_identity(store.get(first_id))
    second_identity = _evidence_identity(store.get(second_id))

    assert store.gc_externalized_tool_result(first_id, "Alice amount $99") is True
    assert store.gc_externalized_tool_result(second_id, "Bob amount $1") is True

    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": "What is the total amount?",
                "evidence_complete": True,
                "operands": [
                    {
                        "exact_ref": f"lcm:{first_id}:0-{len(first)}",
                        "evidence_identity": first_identity,
                        "value": 99,
                        "unit": "usd",
                    },
                    {
                        "exact_ref": f"lcm:{second_id}:0-{len(second)}",
                        "evidence_identity": second_identity,
                        "value": 1,
                        "unit": "usd",
                    },
                ],
            },
            engine=engine,
        )
    )

    assert payload["status"] == "fallback"
    assert "evidence_identity" in payload["reason"]


def test_compute_rejects_identity_after_provenance_rewrite(compute_store):
    store, engine = compute_store
    first = "Alice amount $12"
    second = "Bob amount $8"
    first_id = store.append("session-a", {"role": "user", "content": first})
    second_id = store.append("session-a", {"role": "user", "content": second})
    operands = [
        _operand(store, first_id, first, value=12, unit="usd"),
        _operand(store, second_id, second, value=8, unit="usd"),
    ]

    assert store.reassign_session_messages("session-a", "session-b") == 2
    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": "What is the total amount?",
                "evidence_complete": True,
                "operands": operands,
            },
            engine=engine,
        )
    )

    assert payload["status"] == "fallback"
    assert "evidence_identity" in payload["reason"]


def test_compute_rejects_incomplete_named_evidence(compute_store):
    store, engine = compute_store
    rows = {}
    for name, amount in (("Alice", 10), ("Bob", 20), ("Carol", 30)):
        content = f"{name} amount ${amount}"
        store_id = store.append("session-a", {"role": "user", "content": content})
        rows[name] = (store_id, content, amount)

    operands = [
        _operand(store, rows[name][0], rows[name][1], value=rows[name][2], unit="usd", label=name)
        for name in ("Alice", "Bob")
    ]
    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": "What is the total amount for Alice, Bob, and Carol?",
                "evidence_complete": True,
                "operands": operands,
            },
            engine=engine,
        )
    )

    assert payload["status"] == "fallback"
    assert "Carol" in payload["reason"]


@pytest.mark.parametrize(
    "selector",
    ["for", "by", "among", "spent by", "paid to"],
)
def test_compute_rejects_incomplete_named_evidence_across_selectors(compute_store, selector):
    """The finite-set coverage check must not be dodged by rephrasing the selector."""
    store, engine = compute_store
    rows = {}
    for name, amount in (("Alice", 10), ("Bob", 20), ("Carol", 30)):
        content = f"{name} amount ${amount}"
        store_id = store.append("session-a", {"role": "user", "content": content})
        rows[name] = (store_id, content, amount)

    operands = [
        _operand(store, rows[name][0], rows[name][1], value=rows[name][2], unit="usd", label=name)
        for name in ("Alice", "Bob")
    ]
    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": f"What is the total amount {selector} Alice, Bob, and Carol?",
                "evidence_complete": True,
                "operands": operands,
            },
            engine=engine,
        )
    )

    assert payload["status"] == "fallback"
    assert "Carol" in payload["reason"]


def test_compute_accepts_complete_named_evidence_with_nonstandard_selector(compute_store):
    """Broadened selector detection must still let a fully-covered set compute."""
    store, engine = compute_store
    rows = {}
    for name, amount in (("Alice", 10), ("Bob", 20), ("Carol", 30)):
        content = f"{name} amount ${amount}"
        store_id = store.append("session-a", {"role": "user", "content": content})
        rows[name] = (store_id, content, amount)

    operands = [
        _operand(store, rows[name][0], rows[name][1], value=rows[name][2], unit="usd", label=name)
        for name in ("Alice", "Bob", "Carol")
    ]
    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": "What is the total amount by Alice, Bob, and Carol?",
                "evidence_complete": True,
                "operands": operands,
            },
            engine=engine,
        )
    )

    assert payload["status"] == "computed"
    assert payload["trace"]["result"] == "$60"


def test_compute_rejects_overlapping_spans_from_one_source(compute_store):
    store, engine = compute_store
    content = "Alice amount $12"
    store_id = store.append("session-a", {"role": "user", "content": content})
    identity = _evidence_identity(store.get(store_id))

    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": "What is the total amount?",
                "evidence_complete": True,
                "operands": [
                    {
                        "exact_ref": f"lcm:{store_id}:0-{len(content)}",
                        "evidence_identity": identity,
                        "value": 12,
                        "unit": "usd",
                    },
                    {
                        "exact_ref": f"lcm:{store_id}:1-{len(content)}",
                        "evidence_identity": identity,
                        "value": 12,
                        "unit": "usd",
                    },
                ],
            },
            engine=engine,
        )
    )

    assert payload["status"] == "fallback"
    assert "overlap" in payload["reason"]


def test_compute_falls_back_for_mixed_temporal_expressions(compute_store):
    store, engine = compute_store
    observed = datetime(2024, 3, 20, tzinfo=timezone.utc).timestamp()
    dashboard = "I finished the dashboard 5 days ago. The contract was signed on 2024-03-01."
    kickoff = "The kickoff was on 2024-03-10."
    dashboard_id = store.append(
        "session-a", {"role": "user", "content": dashboard, "timestamp": observed}
    )
    kickoff_id = store.append(
        "session-a", {"role": "user", "content": kickoff, "timestamp": observed}
    )

    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": "Put the dashboard and kickoff in chronological order.",
                "evidence_complete": True,
                "operands": [
                    {
                        **_operand(store, dashboard_id, dashboard, label="dashboard"),
                        "occurrence_time": {
                            "event_time_source": "explicit",
                            "event_date": "2024-03-01",
                            "session_date": "2024-03-20",
                        },
                    },
                    {
                        **_operand(store, kickoff_id, kickoff, label="kickoff"),
                        "occurrence_time": {
                            "event_time_source": "explicit",
                            "event_date": "2024-03-10",
                            "session_date": "2024-03-20",
                        },
                    },
                ],
            },
            engine=engine,
        )
    )

    assert payload["status"] == "fallback"
    assert "occurrence_time" in payload["reason"]


def test_compute_enforces_explicit_as_of_date_without_optional_argument(compute_store):
    store, engine = compute_store
    observed = datetime(2024, 3, 10, tzinfo=timezone.utc).timestamp()
    first = "The report was 4 pages."
    second = "The appendix was 6 pages."
    first_id = store.append(
        "session-a", {"role": "user", "content": first, "timestamp": observed}
    )
    second_id = store.append(
        "session-a", {"role": "user", "content": second, "timestamp": observed}
    )

    payload = json.loads(
        lcm_tools.lcm_compute(
            {
                "question": "As of 2024-03-01, what is the total page count?",
                "evidence_complete": True,
                "operands": [
                    _operand(store, first_id, first, value=4, unit="pages"),
                    _operand(store, second_id, second, value=6, unit="pages"),
                ],
            },
            engine=engine,
        )
    )

    assert payload["status"] == "fallback"
    assert "after the question-date boundary" in payload["reason"]
