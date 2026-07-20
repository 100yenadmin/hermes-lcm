"""Provider-free V4.6.1 host-supplied evidence envelope fixtures."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

from hermes_lcm.config import LCMConfig
from hermes_lcm.evidence_compiler import SELECTOR_SCHEMA_VERSION
from hermes_lcm.host_evidence import (
    build_host_supplied_evidence,
    call_auxiliary_selector,
    prepare_host_evidence_selector,
)
from hermes_lcm.store import MessageStore


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    return SimpleNamespace(
        _config=config,
        _store=MessageStore(config.database_path, ingest_protection_config=config),
        _assertions=None,
        _session_occurrence_dates={},
    )


def test_code_owns_host_envelope_and_selector_proposes_semantics_only(tmp_path):
    engine = _engine(tmp_path)
    owner_text = "Maya owns the Atlas rollout."
    deadline_text = "The Atlas rollout was due on 2026-07-15."
    owner_id = engine._store.append("session-a", {"role": "user", "content": owner_text})
    deadline_id = engine._store.append(
        "session-a", {"role": "user", "content": deadline_text}
    )
    owner = {
        "exact_ref": f"lcm:{owner_id}:0-{len(owner_text)}",
        "quote": owner_text,
    }
    deadline = {
        "exact_ref": f"lcm:{deadline_id}:0-{len(deadline_text)}",
        "quote": deadline_text,
    }
    calls = []

    def selector(request):
        calls.append(request)
        return {
            "version": SELECTOR_SCHEMA_VERSION,
            "requested_facets": [],
            "selections": [
                {
                    "claim_id": "atlas-owner",
                    "facet": "owner",
                    "exact_ref": owner["exact_ref"],
                    "quote": owner["quote"],
                    "entity": "Maya",
                    "value": "Maya",
                },
                {
                    "claim_id": "atlas-deadline",
                    "facet": "deadline",
                    "exact_ref": deadline["exact_ref"],
                    "quote": deadline["quote"],
                    "date": "2026-07-15",
                    "value": "2026-07-15",
                },
            ],
            "missing_facets": [],
            "usage": {
                "provider": "fixture",
                "model": "none",
                "effort": "none",
                "calls": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
                "cost_usd": 0,
            },
        }

    try:
        result = build_host_supplied_evidence(
            "Who owns the Atlas rollout and when is it due?",
            engine=engine,
            baseline_refs=[owner, deadline],
            question_date="2026-07-20",
            selector=selector,
            enabled=True,
            budgets={"max_input_refs": 8, "max_selections": 8},
        )
    finally:
        engine._store.close()

    assert len(calls) == 1
    envelope = calls[0]
    assert envelope["question"] == "Who owns the Atlas rollout and when is it due?"
    assert envelope["as_of"] == "2026-07-20"
    assert envelope["operation"] == "none"
    assert envelope["baseline_evidence"] == [owner, deadline]
    assert envelope["budgets"] == {"max_selections": 8, "max_quote_chars": 2400}
    prepared = prepare_host_evidence_selector(
        "Who owns the Atlas rollout and when is it due?",
        baseline_refs=[owner, deadline],
        question_date="2026-07-20",
        budgets={"max_input_refs": 8, "max_selections": 8},
    )
    assert prepared["baseline_exact_ref_count"] == 2
    assert prepared["request"]["operation"] == "none"
    assert "Do not return the question" in prepared["prompt"]
    assert len(prepared["envelope_sha256"]) == 64
    assert result["state"] == "answer_sufficient", json.dumps(result, sort_keys=True)
    assert result["baseline_retained"] is True
    assert result["context"].startswith("<lcm-compiled-evidence")
    assert owner["exact_ref"] in result["context"]
    assert deadline["exact_ref"] in result["context"]
    assert result["provenance"]["envelope_owner"] == "hermes_lcm_product_code"
    assert result["provenance"]["registered_tool_transport_used"] is False


def test_selector_cannot_override_host_fields_and_failure_retains_baseline(tmp_path):
    engine = _engine(tmp_path)
    content = "Maya owns the Atlas rollout."
    store_id = engine._store.append("session-a", {"role": "user", "content": content})
    source = {
        "exact_ref": f"lcm:{store_id}:0-{len(content)}",
        "quote": content,
    }

    def hostile_selector(_request):
        return {
            "version": SELECTOR_SCHEMA_VERSION,
            "question": "benchmark-authored replacement",
            "question_date": "2099-01-01",
            "baseline_refs": [],
            "budgets": {"max_input_refs": 0},
            "selections": [],
            "missing_facets": [],
        }

    try:
        result = build_host_supplied_evidence(
            "Who owns the Atlas rollout?",
            engine=engine,
            baseline_refs=[source],
            question_date="2026-07-20",
            selector=hostile_selector,
            enabled=True,
        )
    finally:
        engine._store.close()

    assert result["status"] == "fallback"
    assert result["reason_code"] == "selector_schema_invalid"
    assert result["baseline_retained"] is True
    assert result["context"] is None
    assert result["evidence"] == []
    assert result["computation"] is None


def test_compiled_brief_over_context_budget_is_not_injected(tmp_path):
    engine = _engine(tmp_path)
    content = "Maya owns the Atlas rollout."
    store_id = engine._store.append("session-a", {"role": "user", "content": content})
    source = {
        "exact_ref": f"lcm:{store_id}:0-{len(content)}",
        "quote": content,
    }

    def selector(_request):
        return {
            "version": SELECTOR_SCHEMA_VERSION,
            "requested_facets": [],
            "selections": [
                {
                    "claim_id": "atlas-owner",
                    "facet": "owner",
                    "exact_ref": source["exact_ref"],
                    "quote": content,
                    "entity": "Maya",
                    "value": "Maya",
                }
            ],
            "missing_facets": [],
        }

    try:
        result = build_host_supplied_evidence(
            "Who owns the Atlas rollout?",
            engine=engine,
            baseline_refs=[source],
            selector=selector,
            enabled=True,
            max_context_chars=256,
        )
    finally:
        engine._store.close()

    assert result["status"] == "compiled"
    assert result["state"] == "answer_sufficient"
    assert result["context"] is None
    assert result["context_status"] == "context_budget_exhausted"
    assert result["baseline_retained"] is True
    assert result["trace"]["state"] == result["state"]


def test_auxiliary_selector_uses_existing_structured_model_seam_and_actual_usage(
    monkeypatch,
):
    calls = []
    response_payload = {
        "version": SELECTOR_SCHEMA_VERSION,
        "requested_facets": [],
        "selections": [],
        "missing_facets": ["owner"],
        "usage": {"provider": "model-authored-value-must-not-win"},
    }
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(response_payload))
            )
        ],
        usage=SimpleNamespace(input_tokens=123, output_tokens=45),
        provider="fixture-provider",
        model="fixture-model",
    )
    auxiliary = types.ModuleType("agent.auxiliary_client")

    def call_llm(**kwargs):
        calls.append(kwargs)
        return response

    auxiliary.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)
    request = {
        "version": SELECTOR_SCHEMA_VERSION,
        "question": "Who owns Atlas?",
        "as_of": "2026-07-20",
        "facets": [{"name": "owner", "required": True}],
        "operation": "none",
        "exhaustive": False,
        "expected_cardinality": None,
        "baseline_evidence": [],
        "budgets": {"max_selections": 8, "max_quote_chars": 2400},
    }

    proposal = call_auxiliary_selector(
        request, model="fixture-model", timeout_seconds=7.0
    )

    assert len(calls) == 1
    assert calls[0]["task"] == "evidence_selector"
    assert calls[0]["temperature"] == 0
    assert calls[0]["timeout"] == 7.0
    assert "Do not return the question" in calls[0]["messages"][0]["content"]
    assert proposal["usage"] == {
        "provider": "fixture-provider",
        "model": "fixture-model",
        "effort": "host_default",
        "calls": 1,
        "input_tokens": 123,
        "output_tokens": 45,
        "latency_ms": proposal["usage"]["latency_ms"],
    }
