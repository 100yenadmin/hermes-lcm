"""W3b compact-delivery components: C1 diversity, C2 excerpts, C3 compilation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from hermes_lcm.tokens import count_tokens
from hermes_lcm.trajectory_store import (
    CorpusIdentity,
    TrajectorySource,
    TrajectoryState,
    TrajectoryStore,
)


class StateProvider:
    provider_id = "fake"
    model_id = "fake-state-v1"
    dim = 3

    def __init__(self) -> None:
        self.last_usage_tokens = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        folded = str(text).casefold()
        if "alpha-answer" in folded:
            return [1.0, 0.0, 0.0]
        if "beta-answer" in folded:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.last_usage_tokens = len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):  # noqa: ARG002
        self.last_usage_tokens = 1
        return [1.0, 0.0, 0.0]


def _identity(domain: str = "web") -> CorpusIdentity:
    return CorpusIdentity(
        dataset_name="example/w3b",
        dataset_revision="rev-w3b",
        harness_commit="harness-w3b-1",
        tier="small",
        domain=domain,
        ingest_config_digest="w3b-test-v1",
    )


def _source(
    asset_root: Path,
    *,
    trajectory_id: str,
    ordinal: int,
    goal: str,
    texts: tuple[str, ...],
    urls: tuple[str, ...] | None = None,
) -> TrajectorySource:
    states = []
    for index, text in enumerate(texts):
        screenshot = asset_root / f"{trajectory_id}-{index}.png"
        screenshot.write_bytes(b"png" + hashlib.sha256(text.encode()).digest())
        states.append(TrajectoryState(
            state_index=index,
            step=index,
            url=(
                urls[index]
                if urls is not None
                else f"https://example.test/{trajectory_id}/{index}"
            ),
            incoming_action=None if index == 0 else f"click step {index}",
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


def _build_store(tmp_path: Path, *, provider=None) -> TrajectoryStore:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="hub",
        ordinal=0,
        goal="Generic export dashboard",
        texts=tuple(
            f"widget configuration export repeated dashboard panel {index}"
            for index in range(7)
        ),
    ))
    store.insert(_source(
        asset_root,
        trajectory_id="target",
        ordinal=1,
        goal="Problem List Cleanup",
        texts=(
            "widget export procedure overview",
            "alpha-answer duplicate description field exact instruction",
        ),
    ))
    store.insert(_source(
        asset_root,
        trajectory_id="other",
        ordinal=2,
        goal="Open another report",
        texts=("widget export report toolbar",),
    ))
    store.finalize(["hub", "target", "other"])
    return store


def test_all_w3b_knobs_default_off_preserve_bytes_and_telemetry(tmp_path: Path):
    store = _build_store(tmp_path)
    query = "widget configuration export"
    baseline = store.query(query, image_limit=0)
    baseline_payload = [hit.to_dict() for hit in baseline]
    baseline_telemetry = store.last_query_telemetry()

    explicit_off = store.query(
        query,
        image_limit=0,
        diversity_cap=0,
        adaptive_excerpt=False,
        sharp_token_budget=0,
    )
    assert [hit.to_dict() for hit in explicit_off] == baseline_payload
    assert store.last_query_telemetry() == baseline_telemetry
    assert "diversity_cap" not in baseline_telemetry
    assert "adaptive_excerpt" not in baseline_telemetry
    assert "sharp_compilation" not in baseline_telemetry


def test_c1_caps_after_state_arm_composition_and_reports_trajectory(tmp_path: Path):
    provider = StateProvider()
    store = _build_store(tmp_path, provider=provider)
    store.build_state_semantic_index(provider)

    hits = store.query(
        "widget configuration export",
        image_limit=0,
        include_adjacent=False,
        state_semantic_quota=8,
        diversity_cap=2,
    )
    per_trajectory: dict[str, int] = {}
    for hit in hits:
        per_trajectory[hit.trajectory_id] = (
            per_trajectory.get(hit.trajectory_id, 0) + 1
        )
    assert max(per_trajectory.values()) <= 2
    assert any("alpha-answer" in hit.text for hit in hits)

    telemetry = store.last_query_telemetry()
    assert telemetry["state_semantic_expansion"]["admitted"]
    cap = telemetry["diversity_cap"]
    assert cap["cap"] == 2
    hub = next(
        entry for entry in cap["trajectories"]
        if entry["trajectory_id"] == "hub"
    )
    assert hub["before"] == 7
    assert hub["after"] == 2
    assert hub["capped_out"] == 5


def test_c2_densest_window_anchors_rare_query_term_and_shifts_budget():
    prefix = ("Low Stock Report dashboard navigation report " * 100)
    needle = "TABLE HEADER Quantity Source Code Scope"
    suffix = (" report footer " * 100)
    text = prefix + needle + suffix
    excerpt, offset = TrajectoryStore._densest_exact_excerpt(
        text,
        "Which column is next to Quantity in the Low Stock Report?",
        500,
    )
    assert "Quantity Source Code" in excerpt
    assert offset > 0

    rows = [
        {
            "state_id": 1,
            "trajectory_id": "repeat",
            "text": "x" * 4_000,
        },
        {
            "state_id": 2,
            "trajectory_id": "repeat",
            "text": "y" * 4_000,
        },
        {
            "state_id": 3,
            "trajectory_id": "sole",
            "text": "z" * 4_000,
        },
    ]
    limits, telemetry = TrajectoryStore._adaptive_excerpt_limits(
        rows, 1_000, True
    )
    assert limits[1] == limits[2] == 750
    assert limits[3] == 1_500
    assert sum(limits.values()) == 3_000
    assert telemetry["raised_only_hits"] == [{"state_id": 3, "chars": 1500}]


def test_c3_typed_queries_exact_title_template_and_budget(tmp_path: Path):
    store = _build_store(tmp_path)
    budget = 900
    hits = store.query(
        (
            "According to company protocol `Problem List Cleanup`, what should "
            "we change in the duplicate description field before deletion?"
        ),
        image_limit=0,
        include_adjacent=False,
        sharp_token_budget=budget,
    )
    assert hits
    assert hits[0].trajectory_id == "target"
    rendered_tokens = sum(
        count_tokens(TrajectoryStore._rendered_hit_text(hit)) for hit in hits
    )
    assert rendered_tokens <= budget

    sharp = store.last_query_telemetry()["sharp_compilation"]
    assert sharp["question_template"] == "procedure"
    assert {entry["pool_type"] for entry in sharp["subqueries"]} >= {
        "raw_state",
        "entity",
        "action",
    }
    assert sharp["exact_title_boosts"]
    assert sharp["rendered_text_tokens_after"] <= budget
