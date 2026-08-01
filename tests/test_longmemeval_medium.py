"""Medium-tier preparation, provenance, and batching regression tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

import benchmarking.longmemeval as lme

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lcm_longmemeval.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("lcm_longmemeval_medium_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_question(index: int) -> dict:
    evidence_id = f"q{index}-evidence"
    session_ids = [f"q{index}-first", evidence_id, f"q{index}-last"]
    sessions = [
        [{"role": "user", "content": f"ordinary note {index}"}],
        [
            {
                "role": "user",
                "content": f"locker passcode is MEDIUM{index}",
                "has_answer": True,
            }
        ],
        [{"role": "assistant", "content": f"closing note {index}"}],
    ]
    return {
        "question_id": f"q{index}",
        "question_type": "single-session-user",
        "question": f"what is the locker passcode MEDIUM{index}",
        "answer": f"MEDIUM{index}",
        "question_date": "2023-01-01",
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2023-01-01"] * len(session_ids),
        "haystack_sessions": sessions,
        "answer_session_ids": [evidence_id],
    }


def _write_dataset(directory: Path, label: str = "m", count: int = 3) -> tuple[Path, list[dict]]:
    rows = [_raw_question(index) for index in range(count)]
    path = directory / lme.DATASET_COORDS[label]["file"]
    path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return path, rows


def test_prepare_streams_and_writes_checksum_manifest(tmp_path, monkeypatch):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path)
    prepared_dir = tmp_path / "prepared"

    def _full_parse_forbidden(*_args, **_kwargs):
        raise AssertionError("prepare must not call json.loads on the corpus")

    monkeypatch.setattr(lme.json, "loads", _full_parse_forbidden)
    manifest = lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert manifest["dataset_label"] == "m"
    assert manifest["source_file"] == "longmemeval_m"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["question_count"] == len(rows)
    assert [entry["question_id"] for entry in manifest["questions"]] == ["q0", "q1", "q2"]
    for entry, row in zip(manifest["questions"], rows):
        payload = (prepared_dir / entry["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert payload == lme._canonical_json_bytes(row)
    assert (prepared_dir / "manifest.json").is_file()


def test_prepare_rejects_malformed_json_without_publishing_partial_output(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source = tmp_path / lme.DATASET_COORDS["m"]["file"]
    source.write_text(json.dumps([_raw_question(0)])[:-1], encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match="invalid LongMemEval dataset JSON"):
        lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert not prepared_dir.exists()
    assert not list(tmp_path.glob(".prepared.prepare-*"))


def test_prepare_rejects_casefolded_reserved_question_id_atomically(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path, count=2)
    rows[1]["question_id"] = "Manifest"
    source.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match="unsafe question_id"):
        lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert not prepared_dir.exists()


def test_prepare_atomically_replaces_an_existing_empty_directory(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, _rows = _write_dataset(tmp_path, count=1)
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert sorted(path.name for path in prepared_dir.iterdir()) == ["manifest.json", "q0.json"]


def test_prepared_manifest_fails_closed_on_label_count_and_content_mismatch(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, _rows = _write_dataset(tmp_path)
    prepared_dir = tmp_path / "prepared"
    lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    with pytest.raises(ValueError, match="dataset label mismatch"):
        lme.load_prepared_dataset(prepared_dir, dataset_label="s")

    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["question_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="question_count mismatch"):
        lme.load_prepared_dataset(prepared_dir, dataset_label="m")

    manifest["question_count"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (prepared_dir / "q1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        lme.load_prepared_dataset(prepared_dir, dataset_label="m")


def test_direct_dataset_label_must_match_filename(tmp_path):
    source, _rows = _write_dataset(tmp_path, label="m")
    with pytest.raises(ValueError, match="requires filename 'longmemeval_s'"):
        lme.validate_dataset_path_label(source, "s")
    lme.validate_dataset_path_label(source, "m")


def test_cli_requires_exactly_one_run_source_and_exposes_prepare():
    cli = _load_cli()
    prepared = cli._parse_args(
        [
            "prepare", "--dataset", "longmemeval_m", "--prepared-dir", "prepared",
            "--dataset-label", "m",
        ]
    )
    assert prepared.command == "prepare"
    assert prepared.dataset_label == "m"
    with pytest.raises(SystemExit):
        cli._parse_args(["run", "--output", "out"])
    with pytest.raises(SystemExit):
        cli._parse_args(
            [
                "run", "--dataset", "longmemeval_s", "--prepared-dir", "prepared",
                "--output", "out",
            ]
        )


def _zero_timing(monkeypatch):
    monkeypatch.setattr(lme.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(lme.time, "strftime", lambda *_args, **_kwargs: "2000-01-01T00:00:00Z")
    monkeypatch.setattr(lme, "_timed", lambda fn: (fn(), 0.0))


def test_prepared_and_dataset_runs_have_identical_metrics(tmp_path, monkeypatch):
    pytest.importorskip("ijson", reason="prepared-run equivalence requires the prepare path (ijson); run env installs it explicitly")
    monkeypatch.delenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", raising=False)
    _zero_timing(monkeypatch)
    source, _rows = _write_dataset(tmp_path)
    prepared_dir = tmp_path / "prepared"
    lme.prepare_dataset(source, prepared_dir, dataset_label="m")
    prepared = lme.load_prepared_dataset(prepared_dir, dataset_label="m")

    direct_tmp = tmp_path / "direct-run"
    prepared_tmp = tmp_path / "prepared-run"
    direct_tmp.mkdir()
    prepared_tmp.mkdir()
    direct = lme.run_harness(
        lme.load_questions(source),
        provider_name="stub",
        model="",
        tmp_dir=direct_tmp,
        dataset_label="m",
        source_sha256=lme.sha256_file(source),
    )
    from_prepared = lme.run_harness(
        prepared.iter_questions(),
        provider_name="stub",
        model="",
        tmp_dir=prepared_tmp,
        question_count=prepared.question_count,
        dataset_label="m",
        source_sha256=prepared.source_sha256,
        manifest_sha256=prepared.manifest_sha256,
    )

    assert from_prepared["dataset"].pop("manifest_sha256") == prepared.manifest_sha256
    assert json.dumps(from_prepared, sort_keys=True) == json.dumps(direct, sort_keys=True)


class _RecordingStub(lme.StubEmbedder):
    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        return super().embed_documents(batch)


def test_batched_embedding_preserves_order_and_single_call_values(monkeypatch):
    texts = ["alpha", "bravo", "charlie", "delta", "echo"]
    expected = [lme.StubEmbedder().embed_documents([text])[0] for text in texts]
    recorder = _RecordingStub()
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", "2")

    actual = lme._embed_in_batches(recorder, texts)

    assert recorder.calls == [["alpha", "bravo"], ["charlie", "delta"], ["echo"]]
    assert actual == expected


def test_evaluate_question_keeps_session_insertion_order_when_summary_embedding_is_batched(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", "2")
    raw = _raw_question(7)
    question = lme.parse_question(raw)
    recorder = _RecordingStub()

    lme.evaluate_question(
        question,
        recorder,
        provider_name="stub",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    expected_summaries = [
        lme.deterministic_session_summary(session) for session in raw["haystack_sessions"]
    ]
    assert recorder.calls[-2:] == [expected_summaries[:2], expected_summaries[2:]]
    with sqlite3.connect(tmp_path / "q7.db") as conn:
        inserted = conn.execute(
            "SELECT session_id, created_at FROM summary_nodes ORDER BY node_id"
        ).fetchall()
    assert inserted == list(zip(raw["haystack_session_ids"], [1.0, 2.0, 3.0]))


def test_small_default_report_is_byte_identical_to_golden(tmp_path, monkeypatch):
    """Freeze the complete deterministic default report used by banked small runs."""
    monkeypatch.delenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", raising=False)
    _zero_timing(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = lme.run_harness(
        [lme.parse_question(_raw_question(0))],
        provider_name="stub",
        model="",
        tmp_dir=run_dir,
    )
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == (
        "01a62d4e045ed608d102c128a67e6cbfb5929f0427104507323a5d2099202f7e"
    )
    assert report["dataset"] == {
        "name": "LongMemEval_S",
        "repo_id": lme.DATASET_REPO_ID,
        "revision": lme.DATASET_REVISION,
        "file": lme.DATASET_FILENAME,
    }
