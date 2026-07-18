#!/usr/bin/env python3
"""Pre-create Hermes-native sidecars for historical textual tool results.

The SQLite database is always opened read-only. Dry-run is the default; pass
``--apply`` to create sidecars. Raw message rows and summary nodes are never
rewritten. Rollback also defaults to dry-run and deletes only manifest-owned,
digest-matching sidecars that no message or summary references.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import sqlite3
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "hermes_lcm"
BACKFILL_OPERATION = "historical_tool_output_externalization"
BACKFILL_PROVENANCE_KEY = "historical_backfill_provenance"


def _ensure_local_package_importable() -> None:
    if PACKAGE_NAME in sys.modules:
        return
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PLUGIN_DIR)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package


_ensure_local_package_importable()

from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.externalize import (  # noqa: E402
    _replace_externalized_payload,
    find_externalized_payload_for_message,
    get_large_output_storage_dir,
    is_externalized_placeholder,
    maybe_externalize_payload,
)
from hermes_lcm.ingest_protection import _contains_media_payload  # noqa: E402
from hermes_lcm.tokens import count_tokens  # noqa: E402


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _ownership_proof(*, manifest_id: str, ref: str, content_sha256: str) -> str:
    identity = json.dumps(
        {
            "operation": BACKFILL_OPERATION,
            "manifest_id": manifest_id,
            "ref": ref,
            "sha256": content_sha256,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256(identity)


def _is_hex_digest(value: Any, *, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def _is_safe_ref(ref: str) -> bool:
    return bool(ref) and Path(ref).name == ref and "/" not in ref and "\\" not in ref


def _remove_unowned_sidecar(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _validate_rollback_manifest(source: Any) -> str:
    if not isinstance(source, dict):
        raise ValueError("rollback requires a JSON object manifest")
    if source.get("schema_version") != 1 or source.get("applied") is not True:
        raise ValueError("rollback requires an applied schema-v1 backfill manifest")
    if source.get("operation") != BACKFILL_OPERATION:
        raise ValueError(f"rollback requires operation {BACKFILL_OPERATION!r}")
    manifest_id = str(source.get("manifest_id") or "")
    if not _is_hex_digest(manifest_id, length=32):
        raise ValueError("rollback manifest lacks valid backfill provenance")
    items = source.get("items")
    if not isinstance(items, list):
        raise ValueError("rollback manifest items must be a list")

    seen_refs: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("created") is not True:
            raise ValueError("rollback manifest contains an entry without backfill provenance")
        ref = str(item.get("ref") or "")
        if not _is_safe_ref(ref):
            continue
        content_sha256 = str(item.get("sha256") or "")
        proof = str(item.get("ownership_proof") or "")
        if not _is_hex_digest(content_sha256, length=64) or proof != _ownership_proof(
            manifest_id=manifest_id,
            ref=ref,
            content_sha256=content_sha256,
        ):
            raise ValueError("rollback manifest contains an entry without valid backfill provenance")
        if ref in seen_refs:
            raise ValueError("rollback manifest contains duplicate sidecar references")
        seen_refs.add(ref)
    return manifest_id


def _sidecar_matches_provenance(
    payload: dict[str, Any],
    *,
    manifest_id: str,
    ownership_proof: str,
) -> bool:
    return payload.get(BACKFILL_PROVENANCE_KEY) == {
        "operation": BACKFILL_OPERATION,
        "manifest_id": manifest_id,
        "ownership_proof": ownership_proof,
    }


def _contains_serialized_media(content: str) -> bool:
    if _contains_media_payload(content):
        return True
    stripped = content.lstrip()
    if not stripped.startswith(("[", "{")):
        return False
    try:
        return _contains_media_payload(json.loads(content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _read_only_connection(database_path: Path) -> sqlite3.Connection:
    uri = f"{database_path.expanduser().resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.is_symlink():
        raise ValueError("manifest path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_item(*, content: str, ref: str = "", created: bool = False) -> dict[str, Any]:
    return {
        "ref": ref,
        "sha256": _sha256(content),
        "content_chars": len(content),
        "token_estimate": count_tokens(content),
        "created": created,
    }


def run_backfill(
    *,
    database_path: Path,
    hermes_home: Path,
    manifest_path: Path,
    threshold_chars: int,
    apply: bool,
    max_rows: int = 0,
    config: LCMConfig | None = None,
) -> dict[str, Any]:
    """Scan historical rows and optionally create idempotent native sidecars."""
    manifest_id = secrets.token_hex(16)
    runtime_config = copy.copy(config or LCMConfig.from_env())
    runtime_config.large_output_externalization_enabled = True
    runtime_config.large_output_externalization_threshold_chars = max(1, threshold_chars)

    counts = {
        "scanned": 0,
        "eligible": 0,
        "created": 0,
        "existing": 0,
        "skipped_below_threshold": 0,
        "skipped_externalized": 0,
        "skipped_media": 0,
        "failed": 0,
    }
    items: list[dict[str, Any]] = []
    token_estimate_total = 0

    with _read_only_connection(database_path) as connection:
        cursor = connection.execute(
            """
            SELECT session_id, content, tool_call_id
            FROM messages
            WHERE role = 'tool'
            ORDER BY store_id
            """
        )
        for row in cursor:
            if max_rows > 0 and counts["scanned"] >= max_rows:
                break
            counts["scanned"] += 1
            content = row["content"]
            if not isinstance(content, str) or len(content) <= threshold_chars:
                counts["skipped_below_threshold"] += 1
                continue
            if is_externalized_placeholder(content):
                counts["skipped_externalized"] += 1
                continue
            if _contains_serialized_media(content):
                counts["skipped_media"] += 1
                continue

            counts["eligible"] += 1
            item = _manifest_item(content=content)
            token_estimate_total += int(item["token_estimate"])
            existing = find_externalized_payload_for_message(
                content,
                tool_call_id=str(row["tool_call_id"] or ""),
                session_id=str(row["session_id"] or ""),
                kind="tool_result",
                role="tool",
                config=runtime_config,
                hermes_home=str(hermes_home),
            )
            if existing is not None:
                counts["existing"] += 1
                continue
            if not apply:
                items.append(item)
                continue

            created = maybe_externalize_payload(
                content,
                kind="tool_result",
                tool_call_id=str(row["tool_call_id"] or ""),
                session_id=str(row["session_id"] or ""),
                role="tool",
                config=runtime_config,
                hermes_home=str(hermes_home),
                force=True,
            )
            ref = str((created or {}).get("path", ""))
            if not created or not ref:
                counts["failed"] += 1
                continue
            content_sha256 = _sha256(content)
            ownership_proof = _ownership_proof(
                manifest_id=manifest_id,
                ref=Path(ref).name,
                content_sha256=content_sha256,
            )
            payload = created.get("payload")
            path = Path(ref)
            if not isinstance(payload, dict) or payload.get("content") != content:
                # A concurrent writer may have created the same sidecar between
                # our preflight lookup and maybe_externalize_payload(). Never
                # claim or rewrite a sidecar this operation did not create.
                counts["existing"] += 1
                continue
            payload[BACKFILL_PROVENANCE_KEY] = {
                "operation": BACKFILL_OPERATION,
                "manifest_id": manifest_id,
                "ownership_proof": ownership_proof,
            }
            try:
                _replace_externalized_payload(path, payload)
            except OSError:
                counts["failed"] += 1
                _remove_unowned_sidecar(path)
                continue
            counts["created"] += 1
            item = _manifest_item(content=content, ref=path.name, created=True)
            item["ownership_proof"] = ownership_proof
            items.append(item)

    manifest = {
        "schema_version": 1,
        "operation": BACKFILL_OPERATION,
        "manifest_id": manifest_id,
        "applied": apply,
        "threshold_chars": max(1, threshold_chars),
        "counts": counts,
        "token_estimate_total": token_estimate_total,
        "items": items,
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def _ref_is_referenced(connection: sqlite3.Connection, ref: str) -> bool:
    pattern = f"%{ref}%"
    if connection.execute(
        "SELECT 1 FROM messages WHERE content LIKE ? LIMIT 1",
        (pattern,),
    ).fetchone():
        return True
    has_summary_nodes = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'summary_nodes'"
    ).fetchone()
    if not has_summary_nodes:
        return False
    return bool(
        connection.execute(
            "SELECT 1 FROM summary_nodes WHERE summary LIKE ? LIMIT 1",
            (pattern,),
        ).fetchone()
    )


def run_rollback(
    *,
    database_path: Path,
    hermes_home: Path,
    source_manifest_path: Path,
    apply: bool,
    config: LCMConfig | None = None,
) -> dict[str, Any]:
    """Delete only safe, unreferenced sidecars owned by an apply manifest."""
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_id = _validate_rollback_manifest(source)
    runtime_config = copy.copy(config or LCMConfig.from_env())
    storage_dir = get_large_output_storage_dir(
        runtime_config,
        hermes_home=str(hermes_home),
        create=False,
    )
    counts = {
        "manifest_items": 0,
        "eligible": 0,
        "deleted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "skipped_invalid_ref": 0,
        "skipped_missing": 0,
        "skipped_symlink": 0,
        "skipped_provenance_mismatch": 0,
        "skipped_digest_mismatch": 0,
        "skipped_referenced": 0,
    }
    failed_paths: list[str] = []
    eligible_paths: list[Path] = []

    with _read_only_connection(database_path) as connection:
        for item in source["items"]:
            counts["manifest_items"] += 1
            ref = str(item.get("ref") or "")
            if not _is_safe_ref(ref):
                counts["skipped_invalid_ref"] += 1
                continue
            path = storage_dir / ref
            if path.is_symlink():
                counts["skipped_symlink"] += 1
                continue
            if not path.exists() or not path.is_file():
                counts["skipped_missing"] += 1
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if not isinstance(payload, dict) or not _sidecar_matches_provenance(
                payload,
                manifest_id=manifest_id,
                ownership_proof=str(item["ownership_proof"]),
            ):
                counts["skipped_provenance_mismatch"] += 1
                continue
            content = payload.get("content")
            if not isinstance(content, str) or _sha256(content) != str(item.get("sha256") or ""):
                counts["skipped_digest_mismatch"] += 1
                continue
            if _ref_is_referenced(connection, ref):
                counts["skipped_referenced"] += 1
                continue
            counts["eligible"] += 1
            eligible_paths.append(path)

    if apply:
        for path in eligible_paths:
            try:
                path.unlink()
            except OSError:
                counts["failed"] += 1
                failed_paths.append(str(path))
                continue
            counts["deleted"] += 1
            counts["succeeded"] += 1

    counts["skipped"] = sum(
        counts[key]
        for key in (
            "skipped_invalid_ref",
            "skipped_missing",
            "skipped_symlink",
            "skipped_provenance_mismatch",
            "skipped_digest_mismatch",
            "skipped_referenced",
        )
    )

    return {
        "schema_version": 1,
        "operation": "historical_tool_output_externalization_rollback",
        "applied": apply,
        "counts": counts,
        "failed_paths": failed_paths,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", help="LCM SQLite path; defaults to LCM_DATABASE_PATH or HERMES_HOME/lcm.db")
    parser.add_argument("--hermes-home", help="Hermes profile home; defaults to HERMES_HOME or ~/.hermes")
    parser.add_argument("--manifest", default="lcm-externalization-backfill-manifest.json")
    parser.add_argument("--threshold-chars", type=int)
    parser.add_argument("--max-rows", type=int, default=0, help="0 scans all historical tool rows")
    parser.add_argument("--rollback", help="Applied manifest to roll back safely")
    parser.add_argument("--apply", action="store_true", help="Create or delete sidecars; otherwise dry-run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config = LCMConfig.from_env()
    hermes_home = Path(args.hermes_home or os.environ.get("HERMES_HOME") or "~/.hermes").expanduser().resolve()
    database_path = Path(args.database or config.database_path or hermes_home / "lcm.db").expanduser().resolve()
    if not database_path.is_file():
        raise SystemExit("LCM database does not exist")

    if args.rollback:
        result = run_rollback(
            database_path=database_path,
            hermes_home=hermes_home,
            source_manifest_path=Path(args.rollback).expanduser().resolve(),
            apply=args.apply,
            config=config,
        )
    else:
        threshold = args.threshold_chars
        if threshold is None:
            threshold = max(1, int(config.large_output_externalization_threshold_chars or 12_000))
        result = run_backfill(
            database_path=database_path,
            hermes_home=hermes_home,
            manifest_path=Path(args.manifest),
            threshold_chars=threshold,
            apply=args.apply,
            max_rows=max(0, args.max_rows),
            config=config,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.rollback and result["counts"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
