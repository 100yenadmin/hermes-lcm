"""Operator backfill safety and idempotency tests."""

import importlib.util
import json
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.externalize import get_large_output_storage_dir
from hermes_lcm.store import MessageStore


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backfill_externalized_tool_outputs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("historical_externalization_backfill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(tmp_path, *, content="large historical output " * 100, session_id="session-private"):
    home = tmp_path / "hermes"
    database = home / "lcm.db"
    config = LCMConfig(
        database_path=str(database),
        large_output_externalization_enabled=False,
        large_output_externalization_threshold_chars=100,
    )
    store = MessageStore(database, ingest_protection_config=config, hermes_home=str(home))
    store.append(
        session_id,
        {"role": "tool", "tool_call_id": "call-private", "content": content},
    )
    store.close()
    return home, database, config, content


def _run_backfill(module, home, database, config, manifest, *, apply):
    return module.run_backfill(
        database_path=database,
        hermes_home=home,
        manifest_path=manifest,
        threshold_chars=100,
        apply=apply,
        config=config,
    )


def test_dry_run_writes_scrubbed_manifest_without_sidecars_or_db_rewrite(tmp_path):
    module = _load_script()
    home, database, config, content = _seed(tmp_path)
    manifest_path = tmp_path / "dry-run.json"

    result = _run_backfill(module, home, database, config, manifest_path, apply=False)

    assert result["applied"] is False
    assert result["counts"]["eligible"] == 1
    assert result["counts"]["created"] == 0
    assert len(result["items"]) == 1
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "session-private" not in manifest_text
    assert "call-private" not in manifest_text
    assert content not in manifest_text
    storage_dir = get_large_output_storage_dir(config, hermes_home=str(home), create=False)
    assert not storage_dir.exists()
    with module._read_only_connection(database) as connection:
        assert connection.execute("SELECT content FROM messages").fetchone()[0] == content


def test_apply_is_idempotent_and_raw_rows_remain_unchanged(tmp_path):
    module = _load_script()
    home, database, config, content = _seed(tmp_path)

    first = _run_backfill(module, home, database, config, tmp_path / "first.json", apply=True)
    second = _run_backfill(module, home, database, config, tmp_path / "second.json", apply=True)

    assert first["counts"]["created"] == 1
    assert len(first["items"]) == 1
    assert first["items"][0]["created"] is True
    assert second["counts"]["created"] == 0
    assert second["counts"]["existing"] == 1
    assert second["items"] == []
    storage_dir = get_large_output_storage_dir(config, hermes_home=str(home), create=False)
    assert len(list(storage_dir.glob("*.json"))) == 1
    with module._read_only_connection(database) as connection:
        assert connection.execute("SELECT content FROM messages").fetchone()[0] == content


def test_media_shaped_tool_rows_are_skipped(tmp_path):
    module = _load_script()
    media = json.dumps([
        {"type": "input_image", "image_url": "https://example.invalid/image.png"},
        {"type": "text", "text": "x" * 500},
    ])
    home, database, config, _ = _seed(tmp_path, content=media)

    result = _run_backfill(module, home, database, config, tmp_path / "media.json", apply=True)

    assert result["counts"]["skipped_media"] == 1
    assert result["counts"]["created"] == 0


def test_dry_run_counts_already_externalized_rows(tmp_path):
    module = _load_script()
    placeholder = (
        "[Externalized tool output: tool_call_id=call-private-with-a-long-identifier; "
        "chars=1200; bytes=1200; ref=tool-result.json]"
    )
    home, database, config, _ = _seed(tmp_path, content=placeholder)

    result = _run_backfill(module, home, database, config, tmp_path / "externalized.json", apply=False)

    assert result["counts"]["skipped_externalized"] == 1
    assert result["counts"]["eligible"] == 0
    assert result["items"] == []


def test_rollback_dry_run_then_apply_deletes_only_manifest_owned_sidecar(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "apply.json"
    _run_backfill(module, home, database, config, manifest, apply=True)
    ref = json.loads(manifest.read_text(encoding="utf-8"))["items"][0]["ref"]
    sidecar = get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref

    dry_run = module.run_rollback(
        database_path=database,
        hermes_home=home,
        source_manifest_path=manifest,
        apply=False,
        config=config,
    )
    applied = module.run_rollback(
        database_path=database,
        hermes_home=home,
        source_manifest_path=manifest,
        apply=True,
        config=config,
    )

    assert dry_run["counts"]["eligible"] == 1
    assert dry_run["counts"]["deleted"] == 0
    assert applied["counts"]["deleted"] == 1
    assert applied["counts"]["succeeded"] == 1
    assert applied["counts"]["failed"] == 0
    assert applied["counts"]["skipped"] == 0
    assert applied["failed_paths"] == []
    assert not sidecar.exists()


def test_rollback_skips_invalid_ref(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "apply.json"
    _run_backfill(module, home, database, config, manifest, apply=True)
    source = json.loads(manifest.read_text(encoding="utf-8"))
    source["items"][0]["ref"] = "../outside.json"
    manifest.write_text(json.dumps(source), encoding="utf-8")

    result = module.run_rollback(
        database_path=database,
        hermes_home=home,
        source_manifest_path=manifest,
        apply=True,
        config=config,
    )

    assert result["counts"]["skipped_invalid_ref"] == 1
    assert result["counts"]["skipped"] == 1


def test_rollback_skips_missing_sidecar(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "apply.json"
    _run_backfill(module, home, database, config, manifest, apply=True)
    ref = json.loads(manifest.read_text(encoding="utf-8"))["items"][0]["ref"]
    sidecar = get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref
    sidecar.unlink()

    result = module.run_rollback(
        database_path=database,
        hermes_home=home,
        source_manifest_path=manifest,
        apply=True,
        config=config,
    )

    assert result["counts"]["skipped_missing"] == 1
    assert result["counts"]["skipped"] == 1


def test_rollback_skips_symlink(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "apply.json"
    _run_backfill(module, home, database, config, manifest, apply=True)
    ref = json.loads(manifest.read_text(encoding="utf-8"))["items"][0]["ref"]
    sidecar = get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref
    target = tmp_path / "sidecar-target.json"
    sidecar.replace(target)
    sidecar.symlink_to(target)

    result = module.run_rollback(
        database_path=database,
        hermes_home=home,
        source_manifest_path=manifest,
        apply=True,
        config=config,
    )

    assert result["counts"]["skipped_symlink"] == 1
    assert result["counts"]["skipped"] == 1
    assert sidecar.is_symlink()


def test_rollback_rejects_non_schema_v1_manifest(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "invalid-schema.json"
    manifest.write_text(json.dumps({"schema_version": 2, "applied": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="applied schema-v1"):
        module.run_rollback(
            database_path=database,
            hermes_home=home,
            source_manifest_path=manifest,
            apply=True,
            config=config,
        )


def test_rollback_rejects_non_applied_source_manifest(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "dry-run.json"
    _run_backfill(module, home, database, config, manifest, apply=False)

    with pytest.raises(ValueError, match="applied schema-v1"):
        module.run_rollback(
            database_path=database,
            hermes_home=home,
            source_manifest_path=manifest,
            apply=True,
            config=config,
        )


def test_rollback_reports_partial_failure_and_continues(tmp_path, monkeypatch, capsys):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path, content="first historical output " * 100)
    store = MessageStore(database, ingest_protection_config=config, hermes_home=str(home))
    store.append(
        "session-private",
        {"role": "tool", "tool_call_id": "call-second", "content": "second historical output " * 100},
    )
    store.close()
    manifest = tmp_path / "apply.json"
    _run_backfill(module, home, database, config, manifest, apply=True)
    refs = [item["ref"] for item in json.loads(manifest.read_text(encoding="utf-8"))["items"]]
    storage_dir = get_large_output_storage_dir(config, hermes_home=str(home), create=False)
    failed_path = storage_dir / refs[0]
    succeeded_path = storage_dir / refs[1]
    original_unlink = Path.unlink

    def fail_one_unlink(path, *args, **kwargs):
        if path == failed_path:
            raise OSError("simulated unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one_unlink)

    exit_code = module.main(
        [
            "--database",
            str(database),
            "--hermes-home",
            str(home),
            "--rollback",
            str(manifest),
            "--apply",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert result["counts"]["eligible"] == 2
    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["failed"] == 1
    assert result["counts"]["skipped"] == 0
    assert result["failed_paths"] == [str(failed_path)]
    assert failed_path.exists()
    assert not succeeded_path.exists()


def test_rollback_refuses_referenced_sidecar(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "apply.json"
    _run_backfill(module, home, database, config, manifest, apply=True)
    ref = json.loads(manifest.read_text(encoding="utf-8"))["items"][0]["ref"]
    connection = __import__("sqlite3").connect(database)
    connection.execute("UPDATE messages SET content = ?", (f"[Externalized tool output: ref={ref}]",))
    connection.commit()
    connection.close()

    result = module.run_rollback(
        database_path=database,
        hermes_home=home,
        source_manifest_path=manifest,
        apply=True,
        config=config,
    )

    assert result["counts"]["skipped_referenced"] == 1
    sidecar = get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref
    assert sidecar.exists()


def test_rollback_refuses_digest_mismatch(tmp_path):
    module = _load_script()
    home, database, config, _ = _seed(tmp_path)
    manifest = tmp_path / "apply.json"
    _run_backfill(module, home, database, config, manifest, apply=True)
    ref = json.loads(manifest.read_text(encoding="utf-8"))["items"][0]["ref"]
    sidecar = get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["content"] = "changed"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    result = module.run_rollback(
        database_path=database,
        hermes_home=home,
        source_manifest_path=manifest,
        apply=True,
        config=config,
    )

    assert result["counts"]["skipped_digest_mismatch"] == 1
    assert sidecar.exists()
