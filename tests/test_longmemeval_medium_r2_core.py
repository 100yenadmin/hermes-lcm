"""Round-2 guards for LongMemEval streaming and prepared iteration."""

from __future__ import annotations

import hashlib
import json
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


def _scored(question_id: str) -> dict:
    value = float(int(question_id.removeprefix("q")) + 1)
    scored = {}
    for arm in lme.ARMS:
        scored[arm] = {
            "recall@1": value,
            "recall@5": value + 0.1,
            "recall@10": value + 0.2,
            "ndcg@10": value + 0.3,
            "latency_ms": value + 0.4,
            "turn": {
                "recall@1": value + 0.5,
                "recall@5": value + 0.6,
                "recall@10": value + 0.7,
                "ndcg@10": value + 0.8,
                "session_granularity": arm == "summary_vectors",
            },
        }
    scored["hybrid_rerank"]["rerank_mode"] = lme.RERANK_MODE_PLACEHOLDER
    scored["ingest_ms"] = value + 0.9
    return scored


def _run_with_checkpoint(tmp_path, questions, checkpoint, **kwargs):
    return lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path,
        reuse_db_template=False,
        checkpoint_path=checkpoint,
        **kwargs,
    )


def _prepared_dataset(tmp_path, question_ids: tuple[str, ...]) -> lme.PreparedDataset:
    for question_id in question_ids:
        (tmp_path / f"{question_id}.json").write_text("{}", encoding="utf-8")
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


def test_prepared_qid_preflight_rejects_missing_file_before_scoring(tmp_path):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))
    (tmp_path / "q1.json").unlink()

    with pytest.raises(ValueError, match="prepared question file not found"):
        prepared.validate_question_ids()


def test_prepared_qid_preflight_rejects_extra_file_before_scoring(tmp_path):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))
    (tmp_path / "extra.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="file set does not match manifest"):
        prepared.validate_question_ids()


def test_prepared_qid_preflight_does_not_hash_question_bytes(tmp_path, monkeypatch):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))

    def _unexpected_hash(_path):
        raise AssertionError("preflight hashed bytes")

    monkeypatch.setattr(lme, "sha256_file", _unexpected_hash)
    prepared.validate_question_ids()


def test_prepared_iterator_verifies_checksum_at_consumption(tmp_path):
    prepared = _prepared_dataset(tmp_path, ("q0",))

    with pytest.raises(ValueError, match="prepared question checksum mismatch: q0.json"):
        list(prepared.iter_questions())


def test_prepared_iterator_rejects_question_id_mismatch(tmp_path):
    # Checksum-valid file whose embedded id differs from the manifest entry —
    # only reachable via manifest corruption, still fails closed at consumption.
    payload = b'{"question_id": "q-other"}'
    (tmp_path / "q0.json").write_bytes(payload)
    prepared = lme.PreparedDataset(
        directory=tmp_path,
        dataset_label="m",
        source_sha256="0" * 64,
        manifest_sha256="1" * 64,
        question_count=1,
        questions=(
            {
                "question_id": "q0",
                "file": "q0.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        ),
    )

    with pytest.raises(ValueError, match="prepared question id mismatch: q0.json"):
        list(prepared.iter_questions())


def test_question_filename_reserves_template():
    with pytest.raises(ValueError, match="unsafe question_id"):
        lme._question_filename("_TEMPLATE")


@pytest.mark.parametrize("question_id", [".hidden", "...", ".q1"])
def test_question_filename_rejects_leading_dot_ids(question_id):
    # glob("*.json") skips dotfiles on POSIX, so a hidden prepared file would
    # spuriously fail the manifest file-set check at load — reject at prepare.
    with pytest.raises(ValueError, match="unsafe question_id"):
        lme._question_filename(question_id)


@pytest.mark.parametrize(
    "question_id",
    [
        "CON",
        "prn.txt",
        "AUX",
        "nul",
        "COM1",
        "com9.json",
        "LPT1",
        "lpt9.txt",
        "question:name",
        "question*name",
        "question?name",
        'question"name',
        "question<name",
        "question>name",
        "question|name",
    ],
)
def test_question_filename_rejects_cross_platform_unsafe_names(question_id):
    with pytest.raises(ValueError, match="unsafe question_id"):
        lme._question_filename(question_id)


def test_run_harness_cleans_question_db_when_evaluation_raises(tmp_path, monkeypatch):
    question = _question("q0")
    error = RuntimeError("evaluation failed")
    cleanup_calls = []

    def _evaluate(*_args, **_kwargs):
        raise error

    def _cleanup(tmp_dir, question_id):
        cleanup_calls.append((tmp_dir, question_id))

    monkeypatch.setattr(lme, "evaluate_question", _evaluate)
    monkeypatch.setattr(lme, "_cleanup_question_db", _cleanup)

    with pytest.raises(RuntimeError) as caught:
        lme.run_harness(
            [question],
            provider_name="stub",
            model="",
            tmp_dir=tmp_path,
            reuse_db_template=False,
        )

    assert caught.value is error
    assert cleanup_calls == [(tmp_path, "q0")]


def test_checkpoint_is_flushed_after_each_completed_question(tmp_path, monkeypatch):
    questions = [_question(f"q{index}") for index in range(3)]
    checkpoint = tmp_path / lme.PER_QUESTION_CHECKPOINT_FILENAME

    def _evaluate(question, *_args, **_kwargs):
        if question.question_id == "q2":
            raise RuntimeError("simulated crash")
        return _scored(question.question_id)

    monkeypatch.setattr(lme, "evaluate_question", _evaluate)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_with_checkpoint(tmp_path, questions, checkpoint)

    records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert [record["question_id"] for record in records] == ["q0", "q1"]
    assert records[0]["arms"]["fts"]["recall@1"] == 1.0


def test_resume_report_is_identical_to_uninterrupted_report(tmp_path, monkeypatch):
    questions = [_question(f"q{index}") for index in range(3)]
    full_checkpoint = tmp_path / "full.jsonl"
    resumed_checkpoint = tmp_path / "resumed.jsonl"
    monkeypatch.setattr(lme.time, "strftime", lambda *_args, **_kwargs: "fixed-time")
    monkeypatch.setattr(
        lme, "evaluate_question", lambda question, *_args, **_kwargs: _scored(question.question_id)
    )
    uninterrupted = _run_with_checkpoint(tmp_path, questions, full_checkpoint)

    def _crash_on_q2(question, *_args, **_kwargs):
        if question.question_id == "q2":
            raise RuntimeError("simulated crash")
        return _scored(question.question_id)

    monkeypatch.setattr(lme, "evaluate_question", _crash_on_q2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_with_checkpoint(tmp_path, questions, resumed_checkpoint)

    evaluated = []

    def _record_evaluation(question, *_args, **_kwargs):
        evaluated.append(question.question_id)
        return _scored(question.question_id)

    monkeypatch.setattr(lme, "evaluate_question", _record_evaluation)
    resumed = _run_with_checkpoint(
        tmp_path,
        questions,
        resumed_checkpoint,
        resume=True,
        selected_question_ids=[question.question_id for question in questions],
    )

    assert evaluated == ["q2"]
    assert resumed == uninterrupted
    assert json.dumps(resumed, indent=2, sort_keys=True).encode() == json.dumps(
        uninterrupted, indent=2, sort_keys=True
    ).encode()


def test_resume_drops_torn_final_line_and_reruns_that_question(
    tmp_path, monkeypatch, caplog
):
    questions = [_question("q0"), _question("q1")]
    checkpoint = tmp_path / lme.PER_QUESTION_CHECKPOINT_FILENAME
    monkeypatch.setattr(
        lme, "evaluate_question", lambda question, *_args, **_kwargs: _scored(question.question_id)
    )
    _run_with_checkpoint(tmp_path, questions[:1], checkpoint)
    with checkpoint.open("ab") as checkpoint_file:
        checkpoint_file.write(b'{"question_id":"q1"')

    evaluated = []

    def _record_evaluation(question, *_args, **_kwargs):
        evaluated.append(question.question_id)
        return _scored(question.question_id)

    monkeypatch.setattr(lme, "evaluate_question", _record_evaluation)
    with caplog.at_level("WARNING"):
        report = _run_with_checkpoint(
            tmp_path,
            questions,
            checkpoint,
            resume=True,
            selected_question_ids=["q0", "q1"],
        )

    assert evaluated == ["q1"]
    assert report["question_count"] == 2
    assert "dropping torn final checkpoint line" in caplog.text
    assert [
        json.loads(line)["question_id"] for line in checkpoint.read_text().splitlines()
    ] == ["q0", "q1"]


def test_resume_rejects_checkpoint_from_wrong_question_selection(tmp_path, monkeypatch):
    checkpoint = tmp_path / lme.PER_QUESTION_CHECKPOINT_FILENAME
    checkpoint.write_text('{"question_id":"q-other"}\n', encoding="utf-8")

    def _unexpected_evaluation(*_args, **_kwargs):
        raise AssertionError("wrong-directory checkpoint must fail before scoring")

    monkeypatch.setattr(lme, "evaluate_question", _unexpected_evaluation)
    with pytest.raises(ValueError, match="wrong output directory"):
        _run_with_checkpoint(
            tmp_path,
            [_question("q0")],
            checkpoint,
            resume=True,
            selected_question_ids=["q0"],
        )
