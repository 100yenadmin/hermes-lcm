from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.embedding_provider as provider_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.embedding_provider import (
    EmbeddingCircuitBreaker,
    EmbeddingProviderError,
    EmbeddingSpendGuard,
    FastembedProvider,
    HttpResponse,
    OllamaProvider,
    ProviderCircuitOpen,
    ProviderNotWarmedUp,
    ProviderRateLimited,
    ProviderUnavailable,
    VoyageError,
    VoyageProvider,
    resolve_provider,
)
from hermes_lcm.vector_store import VectorStore


def _response(status: int, payload, headers=None) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers=headers or {},
        body=json.dumps(payload).encode(),
    )


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _voyage_success(count: int, dim: int = 3) -> HttpResponse:
    return _response(
        200,
        {
            "data": [
                {"index": index, "embedding": [float(index + 1)] * dim}
                for index in range(count)
            ]
        },
    )


def test_voyage_batch_bin_packing_boundary(monkeypatch):
    token_counts = {"a": 24_000, "b": 24_000, "c": 24_000, "d": 1}
    monkeypatch.setattr(provider_mod, "count_tokens", token_counts.__getitem__)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(3), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    vectors = provider.embed_documents(["a", "b", "c", "d"])

    assert len(vectors) == 4
    assert [call["payload"]["input"] for call in transport.calls] == [
        ["a", "b", "c"],
        ["d"],
    ]
    assert all(call["payload"]["truncation"] is False for call in transport.calls)


def test_voyage_batch_splits_on_item_count_cap(monkeypatch):
    # Many tiny documents stay well under the token budget but exceed Voyage's
    # 1000-item per-request cap, so the batch must split.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(1000), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    docs = [f"doc-{index}" for index in range(1001)]
    vectors = provider.embed_documents(docs)

    assert len(vectors) == 1001
    assert [len(call["payload"]["input"]) for call in transport.calls] == [1000, 1]


def test_voyage_item_cap_is_configurable(monkeypatch):
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(2), _voyage_success(2), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, max_batch_items=2)

    provider.embed_documents([f"doc-{index}" for index in range(5)])
    assert [len(call["payload"]["input"]) for call in transport.calls] == [2, 2, 1]


def test_voyage_absolute_deadline_bounds_total_retry_time(monkeypatch):
    # A tiny budget must return in ~that budget: the backoff between attempts
    # would blow the deadline, so no retry stacks up despite max_attempts>1.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    slept: list[float] = []
    # Every attempt returns a retryable 500; without an absolute deadline this
    # would sleep 0.5 + 1.0s across three attempts.
    transport = FakeTransport(
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
    )
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=slept.append)

    with pytest.raises(VoyageError):
        provider.embed_query_interactive("q", timeout=0.02)

    # One attempt, zero backoff sleeps (the 0.5s backoff would exceed 0.02s).
    assert len(transport.calls) == 1
    assert slept == []


def test_voyage_normal_embed_query_bounds_total_retry_time(monkeypatch):
    # Maintainer repro (B1): normal embed_query() with timeout=0.02 + 3 retryable
    # failures made three attempts and took ~1.501s (the 0.5s + 1.0s backoffs sat
    # outside any total budget). The NORMAL path must get ONE absolute deadline
    # too, just like the interactive path: a tiny budget returns in ~that budget.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    slept: list[float] = []
    transport = FakeTransport(
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
    )
    provider = VoyageProvider(
        "voyage-test", transport=transport, timeout=0.02, sleeper=slept.append
    )

    with pytest.raises(VoyageError):
        provider.embed_query("q")

    # One attempt, zero backoff sleeps (the 0.5s backoff would exceed 0.02s).
    assert len(transport.calls) == 1
    assert slept == []


def test_ollama_normal_embed_query_bounds_total_retry_time():
    # B1 for Ollama's normal path: a tiny timeout budget bounds all attempts +
    # backoff so retryable network failures cannot stack backoff sleeps.
    slept: list[float] = []
    transport = FakeTransport(OSError("boom"), OSError("boom"), OSError("boom"))
    provider = OllamaProvider(
        "model", transport=transport, timeout=0.02, sleeper=slept.append
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("q")

    assert len(transport.calls) == 1
    assert slept == []


def test_voyage_item_cap_clamped_to_hard_limit(monkeypatch):
    # Maintainer repro (B2): configuring max_batch_items=2000 emitted a request
    # with 1,001 inputs. Voyage's hard 1,000 cap is authoritative regardless of
    # config: clamp down so no request ever exceeds 1000 inputs.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(1000), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, max_batch_items=2000)

    assert provider.max_batch_items == 1000
    provider.embed_documents([f"doc-{index}" for index in range(1001)])

    sizes = [len(call["payload"]["input"]) for call in transport.calls]
    assert sizes == [1000, 1]
    assert all(size <= 1000 for size in sizes)


def test_voyage_over_cap_document_is_skipped_and_reported(monkeypatch, caplog):
    monkeypatch.setattr(
        provider_mod,
        "count_tokens",
        lambda text: 24_301 if text == "too large" else 2,
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    with caplog.at_level(logging.WARNING):
        vectors = provider.embed_documents(["keep", "too large"])

    assert len(vectors) == 1
    assert provider.last_skipped_documents == [1]
    assert transport.calls[0]["payload"]["input"] == ["keep"]
    assert "index=1" in caplog.text


def test_voyage_uses_asymmetric_input_types(monkeypatch):
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(2), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    provider.embed_documents(["first", "second"])
    provider.embed_query("question")

    assert [call["payload"]["input_type"] for call in transport.calls] == [
        "document",
        "query",
    ]


def test_voyage_429_honors_retry_after_and_caps_budget(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    sleeps = []
    retry = _response(429, {"error": "slow down"}, {"Retry-After": "2.5"})
    transport = FakeTransport(retry, _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=sleeps.append)

    assert provider.embed_query("question")
    assert sleeps == [2.5]

    capped_transport = FakeTransport(
        _response(429, {"error": "slow down"}, {"Retry-After": "61"})
    )
    capped = VoyageProvider("voyage-test", transport=capped_transport, sleeper=sleeps.append)
    with pytest.raises(VoyageError, match="429") as exc_info:
        capped.embed_query("question")
    assert exc_info.value.kind == "rate_limit"
    assert sleeps == [2.5]


def test_voyage_5xx_retries_then_raises(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    server_error = _response(503, {"error": "unavailable"})
    transport = FakeTransport(server_error, server_error, server_error)
    sleeps = []
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=sleeps.append)

    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("question")

    assert exc_info.value.kind == "server_error"
    assert len(transport.calls) == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "bad_request"), (401, "auth"), (403, "auth")],
)
def test_voyage_4xx_is_classified_without_retry(monkeypatch, status, kind):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_response(status, {"error": "rejected"}))
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=lambda _delay: None)

    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("question")

    assert exc_info.value.kind == kind
    assert len(transport.calls) == 1


def test_voyage_error_logging_scrubs_echoed_inputs(monkeypatch, caplog):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    body = {
        "error": {"kind": "invalid_request", "message": "secret error content"},
        "detail": "private detail content",
        "msg": "private message content",
        "input": "private prompt",
        "nested": {"texts": ["secret text"], "documents": ["secret doc"]},
    }
    provider = VoyageProvider("voyage-test", transport=FakeTransport(_response(400, body)))

    with caplog.at_level(logging.WARNING), pytest.raises(VoyageError):
        provider.embed_query("question")

    for private_text in (
        "secret error content",
        "private detail content",
        "private message content",
        "private prompt",
        "secret text",
        "secret doc",
    ):
        assert private_text not in caplog.text
    assert "status=400" in caplog.text
    assert "REDACTED" in caplog.text


def test_interactive_http_calls_use_one_attempt_and_explicit_timeout(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    voyage_transport = FakeTransport(OSError("offline"), _voyage_success(1))
    voyage = VoyageProvider(
        "voyage-test",
        transport=voyage_transport,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VoyageError, match="network"):
        voyage.embed_query_interactive("question", timeout=0.125)

    # The absolute deadline equals the budget, so the backoff after the first
    # failure would blow it and no retry is attempted: exactly one HTTP call,
    # each attempt bounded by (approximately) the remaining budget.
    assert len(voyage_transport.calls) == 1
    assert voyage_transport.calls[0]["timeout"] == pytest.approx(0.125, abs=0.02)

    ollama_transport = FakeTransport(OSError("offline"), _response(200, {}))
    ollama = OllamaProvider(
        "model",
        transport=ollama_transport,
        sleeper=lambda _delay: None,
    )
    with pytest.raises(EmbeddingProviderError, match="network"):
        ollama.embed_query_interactive("question", timeout=0.25)

    assert len(ollama_transport.calls) == 1
    assert ollama_transport.calls[0]["timeout"] == pytest.approx(0.25, abs=0.02)


def test_voyage_network_errors_are_classified(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(OSError("offline"), OSError("offline"), OSError("offline"))
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=lambda _delay: None)

    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("question")

    assert exc_info.value.kind == "network"


def test_ollama_request_shape_base_url_and_network_only_retry():
    transport = FakeTransport(OSError("starting"), _response(200, {"embeddings": [[1, 2]]}))
    sleeps = []
    provider = OllamaProvider(
        "nomic-embed-text",
        base_url="http://ollama.internal:11434/",
        transport=transport,
        sleeper=sleeps.append,
    )

    assert provider.embed_query("hello") == [1.0, 2.0]
    assert len(transport.calls) == 2
    assert transport.calls[1]["url"] == "http://ollama.internal:11434/api/embed"
    assert transport.calls[1]["payload"] == {
        "model": "nomic-embed-text",
        "input": ["hello"],
        "truncate": False,
    }
    assert sleeps == [0.5]

    http_error = FakeTransport(_response(500, {"error": "broken"}))
    provider = OllamaProvider("model", transport=http_error, sleeper=sleeps.append)
    with pytest.raises(EmbeddingProviderError, match="500"):
        provider.embed_query("hello")
    assert len(http_error.calls) == 1


class FakeFastembedModel:
    constructions = []

    def __init__(self, **kwargs):
        self.constructions.append(kwargs)
        self.embed_calls = []
        self.query_calls = []
        if kwargs["local_files_only"]:
            raise RuntimeError("not cached")

    def embed(self, texts):
        texts = list(texts)
        self.embed_calls.append(texts)
        return ([1.0, float(index)] for index, _text in enumerate(texts))

    def query_embed(self, texts):
        # Distinct marker vector so tests can prove the query API is used for
        # queries instead of the generic passage embed().
        texts = list(texts)
        self.query_calls.append(texts)
        return ([2.0, float(index)] for index, _text in enumerate(texts))


def test_fastembed_not_warmed_never_downloads(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path)

    with pytest.raises(ProviderNotWarmedUp, match="/lcm embed warmup"):
        provider.embed_query("hello")

    assert FakeFastembedModel.constructions == [{
        "model_name": "local-model",
        "cache_dir": str(tmp_path),
        "local_files_only": True,
    }]


def test_fastembed_warmup_is_only_download_path(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path)

    # warmup() embeds a query, so it goes through the query-specific API.
    assert provider.warmup() == [2.0, 0.0]
    assert provider.dim == 2
    assert FakeFastembedModel.constructions[0]["local_files_only"] is False


def test_fastembed_documents_and_queries_use_distinct_apis(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path)
    provider.warmup()  # force download-path construction
    model = provider._model

    docs = provider.embed_documents(["a", "b"])
    query = provider.embed_query("q")

    # Documents go through embed(); the query goes through query_embed().
    assert docs == [[1.0, 0.0], [1.0, 1.0]]
    assert query == [2.0, 0.0]
    assert model.embed_calls[-1] == ["a", "b"]
    assert model.query_calls[-1] == ["q"]


def test_fastembed_absent_dependency_is_clean(monkeypatch):
    def missing():
        raise ImportError("no fastembed")

    monkeypatch.setattr(provider_mod, "_load_fastembed", missing)
    provider = FastembedProvider("local-model")

    with pytest.raises(ProviderUnavailable, match="not installed"):
        provider.embed_query("hello")


def test_resolve_provider_strings_and_dormant_defaults(monkeypatch):
    defaults = LCMConfig()
    assert resolve_provider(defaults) is None

    config = LCMConfig(
        embedding_provider="  VOYAGEAI ",
        embedding_model="voyage-3-lite",
        embedding_query_timeout_s=1.25,
    )
    assert isinstance(resolve_provider(config), VoyageProvider)

    config.embedding_provider = "ollama"
    config.ollama_base_url = "http://custom:1234"
    ollama = resolve_provider(config)
    assert isinstance(ollama, OllamaProvider)
    assert ollama.base_url == "http://custom:1234"

    config.embedding_provider = "fast-embed"
    assert isinstance(resolve_provider(config), FastembedProvider)

    config.embedding_provider = "unsupported"
    with pytest.raises(ProviderUnavailable, match="Unsupported"):
        resolve_provider(config)

    config.embedding_provider = ""
    with pytest.raises(ProviderUnavailable, match="must both be set"):
        resolve_provider(config)


def test_embedding_config_defaults_and_environment(monkeypatch):
    defaults = LCMConfig()
    assert defaults.embedding_provider == ""
    assert defaults.embedding_model == ""
    assert defaults.ollama_base_url == "http://localhost:11434"
    assert defaults.embedding_query_timeout_s == 3.0
    assert defaults.embedding_max_batch_items == 1000

    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "model-a")
    monkeypatch.setenv("LCM_OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("LCM_EMBEDDING_QUERY_TIMEOUT_S", "4.5")
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", "500")
    configured = LCMConfig.from_env()
    assert configured.embedding_provider == "ollama"
    assert configured.embedding_model == "model-a"
    assert configured.ollama_base_url == "http://ollama:11434"
    assert configured.embedding_query_timeout_s == 4.5
    assert configured.embedding_max_batch_items == 500


class FakeWarmupProvider:
    provider_id = "ollama"
    model_id = "model-a"

    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        return self.vector


def _command_engine(tmp_path):
    return SimpleNamespace(
        _config=LCMConfig(database_path=str(tmp_path / "warmup.db")),
        _store=SimpleNamespace(db_path=tmp_path / "warmup.db"),
    )


def test_warmup_command_probes_and_registers_profile(monkeypatch, tmp_path):
    provider = FakeWarmupProvider([0.1, 0.2, 0.3])
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config: provider)
    engine = _command_engine(tmp_path)

    result = handle_lcm_command("embed warmup", engine)

    assert "status: ready" in result
    assert "provider: ollama" in result
    assert "model: model-a" in result
    assert "dim: 3" in result
    assert provider.calls == ["warmup"]
    store = VectorStore(engine._store.db_path)
    try:
        row = store.connection.execute(
            "SELECT provider, dim FROM lcm_embedding_profile WHERE model_name = ?",
            ("model-a",),
        ).fetchone()
        assert tuple(row) == ("ollama", 3)
    finally:
        store.close()


def test_warmup_command_new_dim_is_a_distinct_identity_no_clobber(monkeypatch, tmp_path):
    # A different dim is a different canonical identity, so warmup registers a
    # new active profile rather than clobbering (or erroring against) the old
    # one — the dim-2 profile and its vectors survive for a future switch back.
    engine = _command_engine(tmp_path)
    store = VectorStore(engine._store.db_path)
    store.register_profile("model-a", "ollama", 2)
    store.close()
    monkeypatch.setattr(
        command_mod,
        "resolve_provider",
        lambda _config: FakeWarmupProvider([0.1, 0.2, 0.3]),
    )

    result = handle_lcm_command("embed warmup", engine)

    assert "status: ready" in result
    assert "dim: 3" in result
    assert "Traceback" not in result
    store = VectorStore(engine._store.db_path)
    try:
        rows = store.connection.execute(
            "SELECT dim, active FROM lcm_embedding_profile "
            "WHERE model_name = 'model-a' ORDER BY dim"
        ).fetchall()
        assert [tuple(r) for r in rows] == [(2, 0), (3, 1)]
    finally:
        store.close()


def test_warmup_command_fastembed_uses_explicit_download(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config: provider)
    engine = _command_engine(tmp_path)

    result = handle_lcm_command("embed warmup", engine)

    assert "status: ready" in result
    assert "download: ready" in result
    assert "provider: fastembed" in result
    assert "no per-call API charge" in result
    assert FakeFastembedModel.constructions[0]["local_files_only"] is False


def test_circuit_breaker_opens_then_cools_down():
    breaker = EmbeddingCircuitBreaker(failure_threshold=2, cooldown_seconds=10)
    transport = FakeTransport(
        _response(400, {"error": "bad"}),
        _response(400, {"error": "bad"}),
    )
    provider = OllamaProvider("model", transport=transport, breaker=breaker)

    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("one")
    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("two")
    with pytest.raises(ProviderCircuitOpen):
        provider.embed_query("three")
    assert len(transport.calls) == 2
    assert breaker.allows(now=breaker._open_until + 0.1) is True


def test_spend_guard_rate_limits_provider_calls():
    guard = EmbeddingSpendGuard(max_calls=1, window_seconds=100, backoff_seconds=50)
    transport = FakeTransport(_response(200, {"embeddings": [[1, 2]]}))
    provider = OllamaProvider("model", transport=transport, spend_guard=guard)

    assert provider.embed_query("one") == [1.0, 2.0]
    with pytest.raises(ProviderRateLimited):
        provider.embed_query("two")
    assert len(transport.calls) == 1
