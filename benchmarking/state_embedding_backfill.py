#!/usr/bin/env python3
"""Resumable per-state embedding backfill CLI (issue #142, Lane S / W3a).

Embeds one vector per trajectory state (``states.text``) into the additive
``lcm_trajectory_state_embeddings`` table via ``TrajectoryStore.
build_state_semantic_index`` -- packed 32 items / 72K tokens per request, a
chunked path for states over Voyage's per-document cap, resumable (a re-run
embeds only the remainder), and metered. A JSONL spend ledger is appended after
every dispatched request so an interrupted run leaves an auditable trail, and the
run ABORTS if the projected total cost would exceed the cap.

Usage (backfill a WORKING COPY -- never the frozen originals):

    VOYAGE_API_KEY=... python3 -m benchmarking.state_embedding_backfill \
        --db /path/to/copy/lcm.db \
        --provider voyage --model voyage-4 \
        --ledger /path/to/artifacts/W3A-backfill-web-ledger.jsonl \
        --cost-cap 10.0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

# voyage-4 document-embedding price ($/1M tokens); mirrors command.py's
# _VOYAGE_USD_PER_MILLION_TOKENS (the table the #141 sizing used).
_VOYAGE_USD_PER_MILLION_TOKENS = {
    "voyage-4-large": 0.12,
    "voyage-4": 0.06,
    "voyage-4-lite": 0.02,
    "voyage-3": 0.06,
    "voyage-3.5": 0.06,
    "voyage-3-large": 0.18,
}


def _bootstrap_package(repo_root: Path) -> Any:
    pkg = "hermes_lcm"
    if pkg in sys.modules:
        return sys.modules[pkg]
    parent = str(repo_root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(
        pkg, str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(repo_root)]
    mod.__package__ = pkg
    sys.modules[pkg] = mod
    for py_file in repo_root.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"{pkg}.{py_file.stem}"
        if sub_name in sys.modules:
            continue
        sub_spec = importlib.util.spec_from_file_location(
            sub_name, str(py_file), submodule_search_locations=[]
        )
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sub_mod.__package__ = pkg
        sys.modules[sub_name] = sub_mod
        setattr(mod, py_file.stem, sub_mod)
        try:
            sub_spec.loader.exec_module(sub_mod)
        except Exception:
            pass
    return mod


def _open_store(db_path: Path, asset_root: Path):
    ts = sys.modules["hermes_lcm.trajectory_store"]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    identity_json = json.loads(
        conn.execute(
            "SELECT identity_json FROM lcm_trajectory_corpora WHERE singleton=1"
        ).fetchone()[0]
    )
    conn.close()
    identity = ts.CorpusIdentity(
        dataset_name=identity_json["dataset_name"],
        dataset_revision=identity_json["dataset_revision"],
        harness_commit=identity_json["harness_commit"],
        tier=identity_json["tier"],
        domain=identity_json["domain"],
        ingest_config_digest=identity_json.get("ingest_config_digest", ""),
    )
    return ts.TrajectoryStore(db_path, identity, asset_root=asset_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True,
                        help="WORKING-COPY lcm.db to backfill (never a frozen original)")
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--provider", default="voyage")
    parser.add_argument("--model", default="voyage-4")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--batch-items", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cost-cap", type=float, default=10.0,
                        help="abort if projected total cost exceeds this (USD)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    _bootstrap_package(_REPO_ROOT)
    ts = sys.modules["hermes_lcm.trajectory_store"]
    asset_root = args.asset_root or args.db.parent
    store = _open_store(args.db, asset_root)

    provider = ts.create_trajectory_embedding_provider(
        args.provider, args.model, timeout_seconds=args.timeout, for_backfill=True,
    )
    rate = _VOYAGE_USD_PER_MILLION_TOKENS.get(args.model, 0.06)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint_state = {"logged_50pct": False}

    def _projected(stats: dict[str, Any]) -> tuple[float, float]:
        cost = stats["billed_tokens"] / 1e6 * rate
        embedded = max(1, stats["states_embedded"])
        pending = max(1, stats["pending"])
        projected_tokens = stats["billed_tokens"] / embedded * pending
        projected_cost = projected_tokens / 1e6 * rate
        return cost, projected_cost

    class _AbortCostCap(RuntimeError):
        pass

    def _callback(stats: dict[str, Any]) -> None:
        cost, projected_cost = _projected(stats)
        elapsed = time.perf_counter() - started
        record = {
            "ts": time.time(),
            "db": str(args.db),
            "model": args.model,
            "states_embedded": stats["states_embedded"],
            "pending": stats["pending"],
            "chunked_states": stats["chunked_states"],
            "provider_calls": stats["provider_calls"],
            "billed_tokens": stats["billed_tokens"],
            "cost_usd": round(cost, 6),
            "projected_total_usd": round(projected_cost, 6),
            "elapsed_s": round(elapsed, 1),
        }
        with args.ledger.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        done = stats["states_embedded"]
        pending = stats["pending"]
        if pending and not checkpoint_state["logged_50pct"] and done >= pending / 2:
            checkpoint_state["logged_50pct"] = True
            print(f"[CHECKPOINT 50%] {done}/{pending} states, spent ${cost:.4f}, "
                  f"projected total ${projected_cost:.4f} (cap ${args.cost_cap})",
                  flush=True)
        if projected_cost > args.cost_cap:
            raise _AbortCostCap(
                f"projected total ${projected_cost:.2f} exceeds cap ${args.cost_cap:.2f} "
                f"after {done} states (${cost:.4f} spent)"
            )
        if stats["provider_calls"] % 25 == 0:
            print(f"  {done}/{pending} embedded, {stats['provider_calls']} calls, "
                  f"${cost:.4f} spent, ~${projected_cost:.4f} projected "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"== state backfill: {args.db} (model={args.model}, rate=${rate}/M) ==",
          flush=True)
    try:
        stats = store.build_state_semantic_index(
            provider,
            resume=not args.no_resume,
            batch_max_items=args.batch_items,
            progress_callback=_callback,
        )
    except _AbortCostCap as exc:
        print(f"ABORTED (cost cap): {exc}", flush=True)
        store.close()
        return 2

    elapsed = time.perf_counter() - started
    cost = stats["billed_tokens"] / 1e6 * rate
    summary = {
        "db": str(args.db),
        "provider": args.provider,
        "model": args.model,
        "rate_usd_per_million": rate,
        "profile_digest": stats["profile_digest"],
        "dim": stats["dim"],
        "total_states": stats["total_states"],
        "already_embedded_at_start": stats["already_embedded"],
        "pending_at_start": stats["pending"],
        "states_embedded_this_run": stats["states_embedded"],
        "chunked_states": stats["chunked_states"],
        "provider_calls": stats["provider_calls"],
        "billed_tokens": stats["billed_tokens"],
        "cost_usd": round(cost, 6),
        "runtime_s": round(elapsed, 1),
        "status": stats["status"],
    }
    store.close()
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2))
    print("== DONE ==", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
