"""Exact message-span references and immutable source provenance.

An exact ref identifies one character span in one persisted message row.  Every
compute operand must also carry the content-and-provenance identity emitted with
the ref.  The resolver rereads the authoritative row and rejects any identity or
optional quote drift, so missing, rewritten, or deleted sources fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
from typing import Any, Mapping


_EXACT_REF_RE = re.compile(
    r"^lcm:(?P<store_id>[1-9]\d*):(?P<start>\d+)-(?P<end>\d+)$"
)
_EVIDENCE_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVIDENCE_IDENTITY_FIELDS = (
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


@dataclass(frozen=True)
class ExactRef:
    store_id: int
    span_start: int
    span_end: int

    @property
    def value(self) -> str:
        return f"lcm:{self.store_id}:{self.span_start}-{self.span_end}"


@dataclass(frozen=True)
class ResolvedExactRef:
    exact_ref: str
    store_id: int
    span_start: int
    span_end: int
    quote: str
    session_id: str
    source: str
    role: str
    observed_at: float | None
    ingested_at: float | None
    observed_at_source: str | None
    evidence_identity: str

    @property
    def source_provenance(self) -> dict[str, Any]:
        return {
            "store_id": self.store_id,
            "session_id": self.session_id,
            "source": self.source,
            "role": self.role,
        }

    @property
    def observation_time(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "ingested_at": self.ingested_at,
            "source": self.observed_at_source or "unavailable",
        }


def _valid_epoch(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def evidence_identity_for_row(row: Mapping[str, Any]) -> str:
    """Return a stable identity for the full content and source provenance."""
    payload = {field: row.get(field) for field in _EVIDENCE_IDENTITY_FIELDS}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def parse_exact_ref(value: Any) -> ExactRef | None:
    """Parse either ``lcm:<row>:<start>-<end>`` or explicit span fields."""
    if isinstance(value, str):
        raw = value.strip()
    elif isinstance(value, Mapping):
        raw = str(value.get("exact_ref") or "").strip()
        if not raw:
            try:
                parsed = ExactRef(
                    int(value["store_id"]),
                    int(value["span_start"]),
                    int(value["span_end"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if (
                parsed.store_id <= 0
                or parsed.span_start < 0
                or parsed.span_end <= parsed.span_start
            ):
                return None
            return parsed
    else:
        return None

    match = _EXACT_REF_RE.fullmatch(raw)
    if match is None:
        return None
    parsed = ExactRef(
        int(match.group("store_id")),
        int(match.group("start")),
        int(match.group("end")),
    )
    if parsed.span_end <= parsed.span_start:
        return None
    return parsed


def resolve_exact_ref(
    messages: Any,
    candidate: Any,
    *,
    question_as_of: float | None = None,
) -> tuple[ResolvedExactRef | None, str | None]:
    """Resolve and validate one exact ref against the current source row."""
    if not isinstance(candidate, Mapping):
        return None, "exact evidence operand must be an object"
    parsed = parse_exact_ref(candidate)
    if parsed is None:
        return None, "invalid exact_ref"
    row = messages.get(parsed.store_id)
    if row is None:
        return None, "exact source row does not exist"
    supplied_identity = str(candidate.get("evidence_identity") or "").strip()
    if not _EVIDENCE_IDENTITY_RE.fullmatch(supplied_identity):
        return None, "evidence_identity is required for every exact ref"
    current_identity = evidence_identity_for_row(row)
    if not hmac.compare_digest(supplied_identity, current_identity):
        return None, "evidence_identity does not match current content and provenance"
    content = str(row.get("content") or "")
    if (
        parsed.span_start < 0
        or parsed.span_end <= parsed.span_start
        or parsed.span_end > len(content)
    ):
        return None, "exact_ref span is outside the source row"
    exact_quote = content[parsed.span_start : parsed.span_end]
    supplied_quote = candidate.get("quote")
    if supplied_quote is not None and str(supplied_quote) != exact_quote:
        return None, "quote does not match the exact source span"
    if not exact_quote:
        return None, "exact source span is empty"

    observed_at = _valid_epoch(row.get("observed_at"))
    ingested_at = _valid_epoch(row.get("ingested_at")) or _valid_epoch(
        row.get("timestamp")
    )
    if question_as_of is not None:
        if observed_at is not None and observed_at > question_as_of:
            return None, "source was observed after the question-date boundary"
        if observed_at is None and ingested_at is not None and ingested_at > question_as_of:
            return None, "source was ingested after the question-date boundary"

    return (
        ResolvedExactRef(
            exact_ref=parsed.value,
            store_id=parsed.store_id,
            span_start=parsed.span_start,
            span_end=parsed.span_end,
            quote=exact_quote,
            session_id=str(row.get("session_id") or ""),
            source=str(row.get("source") or ""),
            role=str(row.get("role") or "unknown"),
            observed_at=observed_at,
            ingested_at=ingested_at,
            observed_at_source=(
                str(row.get("observed_at_source"))
                if row.get("observed_at_source")
                else None
            ),
            evidence_identity=current_identity,
        ),
        None,
    )
