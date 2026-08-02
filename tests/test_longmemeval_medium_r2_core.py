"""Round-2 guards for LongMemEval streaming and prepared iteration."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

import benchmarking.longmemeval as lme


class _FakeJSONError(Exception):
    pass


class _OffsetValueError(ValueError):
    pass


@pytest.mark.parametrize(
    ("backend_error", "detail"),
    [
        (_OffsetValueError("builder failed at byte offset 41"), "byte offset 41"),
        (KeyError("map_key"), "map_key"),
        (TypeError("builder type mismatch"), "builder type mismatch"),
    ],
)
def test_streaming_backend_builder_errors_fail_closed(
    monkeypatch, backend_error, detail
):
    def _items(*_args, **_kwargs):
        raise backend_error
        yield  # pragma: no cover - makes this a lazy backend iterator

    monkeypatch.setitem(
        sys.modules,
        "ijson",
        SimpleNamespace(JSONError=_FakeJSONError, items=_items),
    )

    with pytest.raises(ValueError, match="invalid LongMemEval dataset JSON") as caught:
        list(lme._iter_dataset_rows(object()))

    assert detail in str(caught.value)
    assert caught.value.__cause__ is backend_error


def _question(question_id: str) -> lme.Question:
    return lme.Question(
        question_id=question_id,
        question_type="single-session-user",
        question="question",
        haystack_session_ids=[],
        haystack_sessions=[],
        answer_session_ids=[],
    )


def _prepared_dataset(tmp_path, question_ids: tuple[str, ...]) -> lme.PreparedDataset:
    return lme.PreparedDataset(
        directory=tmp_path,
        dataset_label="m",
        source_sha256="0" * 64,
        manifest_sha256="1" * 64,
        question_count=len(question_ids),
        questions=tuple(
            {
                "question_id": question_id,
                "file": f"{question_id}.json",
                "sha256": hashlib.sha256(question_id.encode()).hexdigest(),
            }
            for question_id in question_ids
        ),
    )


def test_prepared_qid_preflight_rejects_short_iterator_before_scoring(
    tmp_path, monkeypatch
):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))
    scoring_started = False

    def _short_iterator(_self, *, limit=None):
        assert limit is None
        yield _question("q0")

    monkeypatch.setattr(lme.PreparedDataset, "iter_questions", _short_iterator)

    with pytest.raises(ValueError, match="ended early.*'q1'"):
        prepared.validate_question_ids()
        scoring_started = True

    assert scoring_started is False


def test_prepared_qid_preflight_rejects_mismatched_sequence(tmp_path, monkeypatch):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))

    def _mismatched_iterator(_self, *, limit=None):
        assert limit is None
        yield _question("q0")
        yield _question("wrong-qid")

    monkeypatch.setattr(lme.PreparedDataset, "iter_questions", _mismatched_iterator)

    with pytest.raises(
        ValueError,
        match="sequence id mismatch: expected 'q1', got 'wrong-qid'",
    ):
        prepared.validate_question_ids()
