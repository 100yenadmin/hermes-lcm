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
    token_counts = {"a": 27_000, "b": 27_000, "c": 26_000, "d": 1}
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


def test_voyage_over_cap_document_is_skipped_and_reported(monkeypatch, caplog):
    monkeypatch.setattr(
        provider_mod,
        "count_tokens",
        lambda text: 27_001 if text == "too large" else 2,
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
        "error": "invalid request",
        "input": "private prompt",
        "nested": {"texts": ["secret text"], "documents": ["secret doc"]},
    }
    provider = VoyageProvider("voyage-test", transport=FakeTransport(_response(400, body)))

    with caplog.at_level(logging.WARNING), pytest.raises(VoyageError):
        provider.embed_query("question")

    assert "invalid request" in caplog.text
    assert "private prompt" not in caplog.text
    assert "secret text" not in caplog.text
    assert "secret doc" not in caplog.text
    assert "REDACTED" in caplog.text


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
        if kwargs["local_files_only"]:
            raise RuntimeError("not cached")

    def embed(self, texts):
        return ([1.0, float(index)] for index, _text in enumerate(texts))


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

    assert provider.warmup() == [1.0, 0.0]
    assert provider.dim == 2
    assert FakeFastembedModel.constructions[0]["local_files_only"] is False


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

    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "model-a")
    monkeypatch.setenv("LCM_OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("LCM_EMBEDDING_QUERY_TIMEOUT_S", "4.5")
    configured = LCMConfig.from_env()
    assert configured.embedding_provider == "ollama"
    assert configured.embedding_model == "model-a"
    assert configured.ollama_base_url == "http://ollama:11434"
    assert configured.embedding_query_timeout_s == 4.5


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


def test_warmup_command_surfaces_dimension_lock_readably(monkeypatch, tmp_path):
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

    assert "status: error" in result
    assert "locked at 2, not 3" in result
    assert "Traceback" not in result


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
