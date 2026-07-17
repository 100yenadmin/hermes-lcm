"""Tests for the LongMemEval retrieval harness.

Covers the evidence-matching scorer, the metric math (recall@k / NDCG@10 /
percentiles), CLI argument validation, and an end-to-end offline stub run that
proves the ingest -> retrieve -> score plumbing over a real temp LCM store.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from benchmarking.longmemeval import (
    ARMS,
    DATASET_REVISION,
    Question,
    chunk_sessions,
    deterministic_session_summary,
    evaluate_question,
    evidence_sessions,
    evidence_turns,
    load_questions,
    ndcg_at_k,
    parse_question,
    percentiles,
    recall_at_k,
    rrf_fuse,
    run_harness,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lcm_longmemeval.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("lcm_longmemeval_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_raw(
    question_id: str,
    question_type: str,
    *,
    sessions: dict[str, list[dict]],
    answer_session_ids: list[str],
    question: str = "what did we decide",
) -> dict:
    session_ids = list(sessions)
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": "irrelevant",
        "question_date": "2023-01-01",
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2023-01-01"] * len(session_ids),
        "haystack_sessions": [sessions[sid] for sid in session_ids],
        "answer_session_ids": answer_session_ids,
    }


# --------------------------------------------------------------------------- #
# Evidence-matching scorer.
# --------------------------------------------------------------------------- #


def test_evidence_sessions_uses_answer_session_ids():
    raw = _make_raw(
        "q1",
        "multi-session",
        sessions={
            "s1": [{"role": "user", "content": "hi"}],
            "s2": [{"role": "user", "content": "budget is 500", "has_answer": True}],
        },
        answer_session_ids=["s2"],
    )
    question = parse_question(raw)
    assert evidence_sessions(question) == {"s2"}


def test_abstention_questions_have_no_evidence_and_are_flagged():
    raw = _make_raw(
        "q9_abs",
        "single-session-user",
        sessions={"s1": [{"role": "user", "content": "hi"}]},
        answer_session_ids=[],
    )
    question = parse_question(raw)
    assert question.is_abstention is True
    assert evidence_sessions(question) == set()
    assert evidence_turns(question) == set()


def test_evidence_turns_reads_has_answer_markers():
    raw = _make_raw(
        "q2",
        "single-session-assistant",
        sessions={
            "s1": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "the code is X7", "has_answer": True},
            ],
        },
        answer_session_ids=["s1"],
    )
    question = parse_question(raw)
    assert evidence_turns(question) == {("s1", 1)}


def test_category_maps_temporal_reasoning_label():
    raw = _make_raw(
        "q3",
        "temporal-reasoning",
        sessions={"s1": [{"role": "user", "content": "a"}]},
        answer_session_ids=["s1"],
    )
    assert parse_question(raw).category == "temporal"


# --------------------------------------------------------------------------- #
# Metric math.
# --------------------------------------------------------------------------- #


def test_recall_at_k_counts_relevant_within_top_k():
    retrieved = ["s3", "s1", "s2", "s9"]
    assert recall_at_k(retrieved, {"s1", "s2"}, 1) == 0.0
    assert recall_at_k(retrieved, {"s1", "s2"}, 2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, {"s1", "s2"}, 3) == pytest.approx(1.0)


def test_recall_at_k_dedups_and_handles_empty_relevant():
    assert recall_at_k(["s1", "s1", "s1"], set(), 5) == 0.0
    assert recall_at_k(["s1", "s1", "s2"], {"s2"}, 2) == pytest.approx(1.0)


def test_ndcg_at_k_perfect_and_ranked():
    # Single relevant item at rank 1 -> perfect NDCG.
    assert ndcg_at_k(["s1", "s2"], {"s1"}, 10) == pytest.approx(1.0)
    # Relevant item at rank 2: DCG = 1/log2(3); IDCG (1 relevant) = 1/log2(2)=1.
    import math

    assert ndcg_at_k(["s2", "s1"], {"s1"}, 10) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(["s2", "s3"], {"s1"}, 10) == 0.0


def test_percentiles_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    result = percentiles(values, points=(50, 90, 99))
    assert result["p50"] == 30.0
    assert result["p90"] == 100.0
    assert result["p99"] == 100.0
    assert percentiles([], points=(50,)) == {"p50": 0.0}


def test_rrf_fuse_rewards_agreement_across_arms():
    fts = ["s1", "s2", "s3"]
    vectors = ["s3", "s1", "s9"]
    fused = rrf_fuse(fts, vectors)
    # s1 (ranks 1 and 2) and s3 (ranks 3 and 1) outrank single-arm-only items.
    assert set(fused[:2]) == {"s1", "s3"}
    assert set(fused) == {"s1", "s2", "s3", "s9"}


def test_deterministic_summary_is_stable_and_content_bearing():
    turns = [{"role": "user", "content": "the vault code is 4417"}]
    first = deterministic_session_summary(turns)
    second = deterministic_session_summary(turns)
    assert first == second
    assert "4417" in first


# --------------------------------------------------------------------------- #
# CLI argument validation.
# --------------------------------------------------------------------------- #


def test_cli_run_requires_model_for_non_stub_provider():
    cli = _load_cli()
    args = cli._parse_args(
        ["run", "--dataset", "x.json", "--output", "out", "--provider", "fastembed"]
    )
    with pytest.raises(SystemExit):
        cli._cmd_run(args)


def test_cli_run_rejects_nonpositive_limit(tmp_path):
    cli = _load_cli()
    dataset = tmp_path / "d.json"
    dataset.write_text("[]", encoding="utf-8")
    args = cli._parse_args(
        ["run", "--dataset", str(dataset), "--output", str(tmp_path / "o"), "--limit", "0"]
    )
    with pytest.raises(SystemExit):
        cli._cmd_run(args)


def test_cli_run_rejects_missing_dataset(tmp_path):
    cli = _load_cli()
    args = cli._parse_args(
        ["run", "--dataset", str(tmp_path / "missing.json"), "--output", str(tmp_path / "o")]
    )
    with pytest.raises(SystemExit):
        cli._cmd_run(args)


def test_cli_rejects_unknown_provider():
    cli = _load_cli()
    with pytest.raises(SystemExit):
        cli._parse_args(["run", "--dataset", "x", "--output", "o", "--provider", "nope"])


def test_load_questions_limit_validation(tmp_path):
    dataset = tmp_path / "d.json"
    dataset.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(dataset, limit=-1)


# --------------------------------------------------------------------------- #
# End-to-end offline stub run (plumbing proof).
# --------------------------------------------------------------------------- #


def _synthetic_dataset() -> list[Question]:
    questions: list[Question] = []
    for index in range(3):
        evidence_id = f"q{index}-s-evidence"
        sessions = {
            f"q{index}-s-noise": [
                {"role": "user", "content": "unrelated small talk about the weather"},
                {"role": "assistant", "content": "yes it is sunny today"},
            ],
            evidence_id: [
                {"role": "user", "content": f"remember my locker passcode is ZEBRA{index}"},
                {
                    "role": "assistant",
                    "content": f"noted, locker passcode ZEBRA{index}",
                    "has_answer": True,
                },
            ],
        }
        questions.append(
            parse_question(
                _make_raw(
                    f"q{index}",
                    "single-session-user",
                    sessions=sessions,
                    answer_session_ids=[evidence_id],
                    question=f"what is my locker passcode ZEBRA{index}",
                )
            )
        )
    return questions


class _FakeChunkStore:
    """Minimal stand-in exposing only ``knn_chunks`` for the chunk arm."""

    def __init__(self, hits):
        self._hits = hits

    def knn_chunks(self, query_vec, k, model, provider):
        return list(self._hits)


def test_chunk_sessions_maps_hits_to_sessions_and_dedups():
    # Chunk ids are ``store_id:chunk_index``; each votes for its owning session,
    # first-seen order wins, and an unmapped store_id is dropped.
    hits = [("10:0", 0.9, "chunk"), ("11:2", 0.8, "chunk"),
            ("10:1", 0.7, "chunk"), ("99:0", 0.6, "chunk")]
    store_id_to_session = {10: "sess-a", 11: "sess-b"}
    ranked = chunk_sessions(
        _FakeChunkStore(hits), [1.0, 0.0], "model", "provider", 10, store_id_to_session
    )
    assert ranked == ["sess-a", "sess-b"]


class _KeyedEmbedder:
    """Deterministic embedder: text mentioning the passcode maps to one axis.

    This makes the chunk KNN arm resolvable offline — the evidence chunk and the
    query share the ``passcode`` axis, so the evidence session ranks first.
    """

    model_id = "keyed"
    dim = 2

    def _vec(self, text: str) -> list[float]:
        return [1.0, 0.0] if "passcode" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vec(str(text)) for text in texts]

    def embed_query(self, text):
        return self._vec(str(text))


def test_evaluate_question_chunk_arm_recovers_evidence(tmp_path):
    evidence_id = "s-evidence"
    long_evidence = (
        "please remember for later reference that my personal locker passcode "
        "phrase is the northern lighthouse keeper, and that this detail matters "
        "quite a lot to me because I keep forgetting it every single time I try "
        "to open the locker at the gym after my afternoon workout session"
    )
    sessions = {
        "s-noise": [
            {"role": "user", "content": "unrelated small talk about the sunny weather today"},
            {"role": "assistant", "content": "yes it certainly is a pleasant afternoon outside"},
        ],
        evidence_id: [
            {"role": "user", "content": long_evidence},
            {"role": "assistant", "content": "noted, I will keep that safe", "has_answer": True},
        ],
    }
    question = parse_question(
        _make_raw(
            "q-chunk", "single-session-user", sessions=sessions,
            answer_session_ids=[evidence_id],
            question="what is my locker passcode phrase",
        )
    )

    scored = evaluate_question(
        question, _KeyedEmbedder(), provider_name="stub",
        tmp_dir=tmp_path, embeddings_enabled=True,
    )

    assert "chunk_vectors" in scored and "hybrid_rrf3" in scored
    # The chunk arm recovers the evidence session via the shared passcode axis.
    assert scored["chunk_vectors"]["recall@1"] == pytest.approx(1.0)
    # The three-arm fusion keeps the evidence session in its top-k.
    assert scored["hybrid_rrf3"]["recall@10"] == pytest.approx(1.0)
    for arm in ARMS:
        assert scored[arm]["latency_ms"] >= 0.0


def test_stub_run_end_to_end_produces_report_and_fts_recovers_evidence(tmp_path):
    report = run_harness(
        _synthetic_dataset(),
        provider_name="stub",
        model="",
        tmp_dir=tmp_path,
    )
    assert report["scored_count"] == 3
    assert report["dataset"]["revision"] == DATASET_REVISION
    assert report["transcript_contents_included"] is False
    assert set(report["arms"]) == set(ARMS)
    # FTS is lexical and provider-independent: the passcode query must recover
    # its evidence session even under the meaningless stub embedder.
    assert report["arms"]["fts"]["recall@10"] == pytest.approx(1.0)
    for arm in ARMS:
        assert report["arms"][arm]["latency_ms"]["p50"] >= 0.0
