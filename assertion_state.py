"""Typed, conflict-preserving state reads over source-valid V4 assertions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from .assertion_store import AssertionStore


_INACTIVE_RELATION_STATUS = {
    "cancels": "cancelled",
    "fulfills": "fulfilled",
    "reverses": "reversed",
    "supersedes": "superseded",
}


@dataclass(frozen=True)
class AssertionStateResult:
    subject_key: str
    predicate_key: str | None
    as_of: float | None
    assertions: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    active_assertion_ids: tuple[str, ...]
    conflict_assertion_ids: tuple[str, ...]
    assertions_truncated: bool
    relations_truncated: bool


def _object_identity(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "object": row.get("object_value"),
            "polarity": row.get("polarity"),
            "value_text": row.get("value_text"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def query_assertion_state(
    store: AssertionStore,
    *,
    subject_key: str,
    predicate_key: str | None = None,
    kinds: Iterable[str] | None = None,
    scope_key: str | None = None,
    speaker_role: str | None = None,
    as_of: float | None = None,
    limit: int = 100,
) -> AssertionStateResult:
    """Return explicit lifecycle state without recency-based conflict erasure."""

    normalized_subject = str(subject_key or "").strip()
    if not normalized_subject:
        raise ValueError("subject_key is required for a bounded state query")
    rows = store.query_assertions(
        subject_key=normalized_subject,
        predicate_key=predicate_key,
        kinds=kinds,
        scope_key=scope_key,
        speaker_role=speaker_role,
        as_of=as_of,
        limit=limit,
    )
    ids = [str(row["assertion_id"]) for row in rows]
    raw_relations = (
        store.query_relations(assertion_ids=ids, as_of=as_of, limit=500)
        if ids
        else []
    )
    relations_truncated = len(raw_relations) == 500
    id_set = set(ids)
    relations = [
        relation
        for relation in raw_relations
        if relation["from_assertion_id"] in id_set
        and relation["to_assertion_id"] in id_set
    ]

    lifecycle: dict[str, set[str]] = {assertion_id: set() for assertion_id in ids}
    explicit_conflicts: set[str] = set()
    for relation in relations:
        relation_type = str(relation["relation_type"])
        from_id = str(relation["from_assertion_id"])
        to_id = str(relation["to_assertion_id"])
        if relation_type in _INACTIVE_RELATION_STATUS:
            lifecycle[to_id].add(_INACTIVE_RELATION_STATUS[relation_type])
        if relation_type == "contradicts":
            explicit_conflicts.update((from_id, to_id))

    active_ids = {
        assertion_id
        for assertion_id, statuses in lifecycle.items()
        if not statuses
    }
    unresolved_conflicts = set(explicit_conflicts) & active_ids
    groups: dict[tuple[str, str, str, str], dict[str, list[str]]] = {}
    for row in rows:
        assertion_id = str(row["assertion_id"])
        if assertion_id not in active_ids:
            continue
        key = (
            str(row["subject_key"]),
            str(row["predicate_key"]),
            str(row["scope_key"]),
            str(row["kind"]),
        )
        groups.setdefault(key, {}).setdefault(_object_identity(row), []).append(assertion_id)
    for variants in groups.values():
        if len(variants) > 1:
            for assertion_ids in variants.values():
                unresolved_conflicts.update(assertion_ids)

    annotated: list[dict[str, Any]] = []
    for row in rows:
        assertion_id = str(row["assertion_id"])
        item = dict(row)
        item["active"] = assertion_id in active_ids
        item["lifecycle_status"] = tuple(sorted(lifecycle[assertion_id]))
        item["unresolved_conflict"] = assertion_id in unresolved_conflicts
        annotated.append(item)
    return AssertionStateResult(
        subject_key=normalized_subject,
        predicate_key=predicate_key,
        as_of=as_of,
        assertions=tuple(annotated),
        relations=tuple(relations),
        active_assertion_ids=tuple(
            assertion_id for assertion_id in ids if assertion_id in active_ids
        ),
        conflict_assertion_ids=tuple(
            assertion_id for assertion_id in ids if assertion_id in unresolved_conflicts
        ),
        assertions_truncated=len(rows) == int(limit),
        relations_truncated=relations_truncated,
    )
