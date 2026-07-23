"""State-level semantic pool-expansion + backfill (issue #142, Lane S / W3a).

The per-SOURCE semantic index carries one coarse vector per trajectory and so
cannot surface a lexically-invisible answer STATE. These exercise the additive
per-STATE index that can: a target state carries NO query term of its own
(lexically invisible) but is the semantic nearest neighbour of the query, so the
``state_semantic_quota`` knob must pull it INTO the state pool as a quota-capped
additive tail -- while the defaults reproduce the pre-expansion pool and delivery
byte-for-byte, delivery stays unchanged when the ranked pool already fills the
nucleus (additive-only proof), the 5-per-trajectory diversity cap at selection is
preserved, and the backfill itself is resumable/idempotent (skip embedded
states) with a chunked path for over-cap documents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hermes_lcm.trajectory_store import (
    CorpusIdentity,
    TrajectorySource,
    TrajectoryState,
    TrajectoryStore,
)


class StateVectorProvider:
    """A tiny 3-D embedder used to steer per-STATE ranking deterministically.

    A state whose text mentions ``alpha-answer`` embeds onto the +x axis (the
    query direction), ``beta-answer`` onto +y, and everything else onto +z, so
    the query (``embed_query`` -> +x) ranks exactly the alpha states first
    regardless of their lexical (BM25) visibility.
    """

    provider_id = "fake"
    model_id = "fake-state-v1"

    def __init__(self) -> None:
        self.last_usage_tokens = 0
        self.document_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        folded = str(text).casefold()
        if "alpha-answer" in folded:
            return [1.0, 0.0, 0.0]
        if "beta-answer" in folded:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_calls += 1
        self.last_usage_tokens = sum(max(1, len(str(t)) // 4) for t in texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text):  # noqa: ARG002
        self.last_usage_tokens = 1
        return [1.0, 0.0, 0.0]


def _identity() -> CorpusIdentity:
    return CorpusIdentity(
        dataset_name="example/state-semantic",
        dataset_revision="rev-state-semantic",
        harness_commit="harness-state-semantic-1",
        tier="small",
        domain="web",
        ingest_config_digest="state-semantic-test-v1",
    )


def _source(asset_root, *, trajectory_id, ordinal, goal, texts) -> TrajectorySource:
    states = []
    for index, text in enumerate(texts):
        screenshot = asset_root / f"{trajectory_id}-{index}.png"
        screenshot.write_bytes(b"png" + hashlib.sha256(text.encode()).digest())
        states.append(TrajectoryState(
            state_index=index,
            step=index,
            url=f"https://example.test/{trajectory_id}/{index}",
            incoming_action=None if index == 0 else f"advance {index}",
            thoughts=f"inspect state {index}",
            text=text,
            screenshot_path=screenshot,
        ))
    return TrajectorySource(
        trajectory_id=trajectory_id,
        ordinal=ordinal,
        goal=goal,
        start_url=f"https://example.test/{trajectory_id}",
        outcome="completed",
        states=tuple(states),
        source_payload={"id": trajectory_id, "goal": goal},
    )


_QUERY = "widget configuration export"


def _state_id(store, trajectory_id: str, state_index: int) -> int:
    row = store._conn.execute(
        """
        SELECT s.state_id FROM lcm_trajectory_states s
        JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
        WHERE src.trajectory_id = ? AND s.state_index = ?
        """,
        (trajectory_id, state_index),
    ).fetchone()
    assert row is not None, f"unknown state {trajectory_id}/{state_index}"
    return int(row[0])


def _pool_state_ids(store) -> set[int]:
    telemetry = store.last_query_telemetry()
    return {int(item["state_id"]) for item in telemetry["state_candidate_pool"]}


def _admitted(store) -> list[dict[str, int]]:
    telemetry = store.last_query_telemetry()
    expansion = telemetry.get("state_semantic_expansion")
    return list(expansion["admitted"]) if expansion else []


def _build_invisible_semantic_store(tmp_path: Path, *, provider=None):
    """One trajectory whose answer state is lexically INVISIBLE (no query term)
    but is the semantic nearest neighbour (alpha-answer); plus a second lexical
    trajectory so the query pool is non-empty."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = provider or StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "navigate homepage dashboard overview",         # 0: invisible filler
            "widget configuration export panel form",       # 1: the lexical seed
            "alpha-answer success banner shows view link",  # 2: invisible ANSWER
            "logout footer copyright notice",               # 3: invisible filler
        ),
    ))
    store.insert(_source(
        asset_root,
        trajectory_id="othertask",
        ordinal=1,
        goal="Export the report",
        texts=("report export toolbar button",),
    ))
    store.finalize(["answerpath", "othertask"])
    store.build_state_semantic_index(provider)
    return store


# --- default-off byte-identity ----------------------------------------------

def test_defaults_reproduce_current_bytes(tmp_path):
    store = _build_invisible_semantic_store(tmp_path)
    baseline = store.query(_QUERY, image_limit=0)
    baseline_telemetry = store.last_query_telemetry()
    explicit_off = store.query(_QUERY, image_limit=0, state_semantic_quota=0)
    off_telemetry = store.last_query_telemetry()
    assert [h.exact_ref for h in explicit_off] == [h.exact_ref for h in baseline]
    assert off_telemetry == baseline_telemetry
    # No telemetry key leaks into the default payload (frozen-run byte parity).
    assert "state_semantic_expansion" not in baseline_telemetry


# --- the core recall mechanism ----------------------------------------------

def test_expansion_pulls_lexically_invisible_state_into_pool(tmp_path):
    store = _build_invisible_semantic_store(tmp_path)
    answer = _state_id(store, "answerpath", 2)
    store.query(_QUERY, image_limit=0)
    assert answer not in _pool_state_ids(store), "answer state must start invisible"
    store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    assert answer in _pool_state_ids(store)
    by_state = {entry["state_id"]: entry for entry in _admitted(store)}
    assert answer in by_state
    assert by_state[answer]["rank"] == 1  # the alpha state is the top-ranked


def test_quota_caps_admissions(tmp_path):
    """Two lexically-invisible alpha states; a quota of 1 admits exactly one."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",   # lexical seed
            "alpha-answer first invisible banner",       # invisible alpha
            "alpha-answer second invisible banner",      # invisible alpha
        ),
    ))
    store.finalize(["answerpath"])
    store.build_state_semantic_index(provider)
    store.query(_QUERY, image_limit=0, state_semantic_quota=1)
    admitted = _admitted(store)
    assert len(admitted) == 1


def test_pool_incumbents_are_not_readmitted(tmp_path):
    """A state that already entered the pool lexically is never duplicated, even
    when it is also the semantic top."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="allmatch",
        ordinal=0,
        goal="Finish the setup flow",
        # Both matching states are ALSO alpha (lexical + semantic top); only the
        # third is a non-matching invisible alpha the arm can add.
        texts=(
            "widget configuration export alpha-answer intro",
            "widget configuration export alpha-answer detail",
            "alpha-answer plain closing remark",
        ),
    ))
    store.finalize(["allmatch"])
    store.build_state_semantic_index(provider)
    store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    admitted = _admitted(store)
    only = _state_id(store, "allmatch", 2)
    assert [entry["state_id"] for entry in admitted] == [only]
    pool = [
        int(item["state_id"])
        for item in store.last_query_telemetry()["state_candidate_pool"]
    ]
    assert len(pool) == len(set(pool)), "pool must stay duplicate-free"


# --- anti-filler / additive-only controls -----------------------------------

def test_delivery_unchanged_when_ranked_pool_fills_nucleus(tmp_path):
    """Additive-only proof on a full pool: the semantic state may enter the POOL
    but must not displace any delivered nucleus/backfill incumbent."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    order = []
    # Enough lexical trajectories to fill the delivered nucleus + backfill.
    for index in range(12):
        trajectory_id = f"lexical-{index:02d}"
        store.insert(_source(
            asset_root,
            trajectory_id=trajectory_id,
            ordinal=index,
            goal="Configure the widget",
            texts=(f"widget configuration export row {index}",),
        ))
        order.append(trajectory_id)
    # One trajectory carrying the lexically-invisible semantic target.
    store.insert(_source(
        asset_root,
        trajectory_id="target",
        ordinal=12,
        goal="Set up the dashboard gadget",
        texts=(
            "widget configuration export panel with all terms",
            "alpha-answer invisible aftermath confirmation",
        ),
    ))
    order.append("target")
    store.finalize(order)
    store.build_state_semantic_index(provider)

    baseline = [h.exact_ref for h in store.query(_QUERY, image_limit=0)]
    expanded = [
        h.exact_ref
        for h in store.query(_QUERY, image_limit=0, state_semantic_quota=16)
    ]
    assert expanded == baseline
    admitted = _admitted(store)
    assert admitted, "the pool itself must still gain a semantic entry"
    invisible = _state_id(store, "target", 1)
    assert invisible in {entry["state_id"] for entry in admitted}


def test_five_per_trajectory_cap_preserved_at_selection(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    # A long trajectory: one lexical seed + seven invisible alpha states, so the
    # arm can admit MORE than five states from a single trajectory to the pool.
    texts = ["widget configuration export summary step"]
    texts += [f"alpha-answer invisible step {index}" for index in range(7)]
    store.insert(_source(
        asset_root,
        trajectory_id="longtask",
        ordinal=0,
        goal="Complete the long procedure",
        texts=tuple(texts),
    ))
    store.finalize(["longtask"])
    store.build_state_semantic_index(provider)
    hits = store.query(
        _QUERY,
        image_limit=0,
        include_adjacent=False,  # nucleus only: isolate the selection cap
        state_semantic_quota=32,
    )
    per_trajectory: dict[str, int] = {}
    for hit in hits:
        per_trajectory[hit.trajectory_id] = per_trajectory.get(hit.trajectory_id, 0) + 1
    assert per_trajectory.get("longtask", 0) <= 5
    # ...even though MORE than five longtask states were admitted to the pool.
    assert len(_admitted(store)) == 7


def test_expanded_states_selected_only_from_the_tail(tmp_path):
    """Semantic-admitted states earn no BM25 rank: every ranked pool row still
    precedes every admitted state-semantic row in the candidate pool."""
    store = _build_invisible_semantic_store(tmp_path)
    store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    telemetry = store.last_query_telemetry()
    pool = [int(item["state_id"]) for item in telemetry["state_candidate_pool"]]
    admitted_ids = {entry["state_id"] for entry in _admitted(store)}
    ranked_positions = [
        index for index, state_id in enumerate(pool) if state_id not in admitted_ids
    ]
    admitted_positions = [
        index for index, state_id in enumerate(pool) if state_id in admitted_ids
    ]
    assert admitted_positions, "admitted states must be visible in the pool"
    assert max(ranked_positions) < min(admitted_positions)


# --- backfill: resumability + inert-without-index ---------------------------

def test_backfill_is_idempotent_and_resumable(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",
            "alpha-answer success banner",
            "logout footer copyright notice",
        ),
    ))
    store.finalize(["answerpath"])
    first = store.build_state_semantic_index(provider)
    assert first["states_embedded"] == 3
    assert first["total_states"] == 3
    assert first["provider_calls"] >= 1
    # A re-run embeds nothing (every state already carries a row).
    provider.document_calls = 0
    second = store.build_state_semantic_index(provider)
    assert second["states_embedded"] == 0
    assert second["already_embedded"] == 3
    assert provider.document_calls == 0


def test_chunked_path_pools_oversize_documents(tmp_path):
    """A document over the (test-lowered) per-document token budget takes the
    chunked path and still yields exactly one usable per-state vector."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    long_text = "alpha-answer " + ("token " * 400)  # well over a 5-token budget
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",
            long_text,
        ),
    ))
    store.finalize(["answerpath"])
    stats = store.build_state_semantic_index(
        provider, document_token_budget=5, batch_token_budget=50, batch_max_items=4
    )
    assert stats["chunked_states"] == 1
    assert stats["states_embedded"] == 2
    oversize = _state_id(store, "answerpath", 1)
    row = store._conn.execute(
        "SELECT vector FROM lcm_trajectory_state_embeddings WHERE state_id = ?",
        (oversize,),
    ).fetchone()
    assert row is not None and len(bytes(row["vector"])) == stats["dim"] * 4


def test_arm_inert_without_provider_or_index(tmp_path):
    """Knob-on but no state index (or no provider) is a no-op, not an error."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    # No provider attached and no state backfill performed.
    store = TrajectoryStore(tmp_path / "lcm.db", _identity(), asset_root=asset_root)
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=("widget configuration export panel form", "plain closing remark"),
    ))
    store.finalize(["answerpath"])
    baseline = [h.exact_ref for h in store.query(_QUERY, image_limit=0)]
    with_knob = [
        h.exact_ref
        for h in store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    ]
    assert with_knob == baseline
    assert _admitted(store) == []
