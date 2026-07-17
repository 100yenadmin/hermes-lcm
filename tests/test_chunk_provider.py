from __future__ import annotations

import json

import pytest

import hermes_lcm.embedding_provider as provider_mod
from hermes_lcm.embedding_provider import (
    HttpResponse,
    VoyageError,
    VoyageProvider,
    default_chunk_model,
    embed_contextualized,
)

_CONTEXT_URL = "https://api.voyageai.com/v1/contextualizedembeddings"
_FLAT_URL = "https://api.voyageai.com/v1/embeddings"


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _context_response(group_sizes, *, usage=100, order=None):
    """Build a nested contextualized response.

    Each embedding encodes its own ``(outer_index, inner_index)`` so a caller
    can assert order preservation even when the wire order is shuffled via
    ``order`` (a permutation of the outer-document objects).
    """
    data = []
    for outer_index, count in enumerate(group_sizes):
        inner = [
            {"index": i, "embedding": [float(outer_index), float(i), 0.0], "text": "x"}
            for i in range(count)
        ]
        # Shuffle the inner list too, to prove inner sorting by index.
        inner = list(reversed(inner))
        data.append({"index": outer_index, "data": inner})
    if order is not None:
        data = [data[i] for i in order]
    body = {"data": data, "model": "voyage-context-3", "usage": {"total_tokens": usage}}
    return HttpResponse(status=200, headers={}, body=json.dumps(body).encode())


class FlatProvider:
    """A provider exposing only the flat embed_documents contract."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        # Deterministic 1-d vector per document.
        return [[float(len(text))] for text in texts]


class ContextualProvider(FlatProvider):
    def embed_contextualized(self, chunks_by_doc):
        return [[[1.0] for _ in chunks] for chunks in chunks_by_doc]


class TestDefaultChunkModel:
    def test_voyage_maps_to_context_4(self):
        assert default_chunk_model("voyage", "voyage-3") == "voyage-context-4"
        assert default_chunk_model("VoyageAI", "x") == "voyage-context-4"

    def test_local_providers_reuse_configured_model(self):
        assert default_chunk_model("ollama", "nomic-embed") == "nomic-embed"
        assert default_chunk_model("fastembed", "bge-small") == "bge-small"


class TestEmbedContextualized:
    def test_plain_fallback_regroups_by_document(self):
        provider = FlatProvider()
        result = embed_contextualized(provider, [["a", "bb"], ["ccc"]])
        # One vector list per document, aligned to that doc's chunks.
        assert len(result) == 2
        assert len(result[0]) == 2 and len(result[1]) == 1
        assert result[0][0] == [1.0] and result[0][1] == [2.0]
        assert result[1][0] == [3.0]
        # Flattened into a single embed_documents call.
        assert provider.calls == [["a", "bb", "ccc"]]

    def test_uses_provider_contextualized_when_available(self):
        provider = ContextualProvider()
        result = embed_contextualized(provider, [["a"], ["b", "c"]])
        assert result == [[[1.0]], [[1.0], [1.0]]]
        # The flat path was not used.
        assert provider.calls == []

    def test_empty_input(self):
        assert embed_contextualized(FlatProvider(), []) == []


class TestVoyageContextualizedWireShape:
    def test_embed_contextualized_request_body_shape(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        transport = FakeTransport(_context_response([2, 1]))
        provider = VoyageProvider("voyage-context-3", transport=transport)

        provider.embed_contextualized([["a", "bb"], ["ccc"]])

        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert call["url"] == _CONTEXT_URL
        payload = call["payload"]
        # inputs is a list of lists (one inner list per document); no flat "input".
        assert payload["inputs"] == [["a", "bb"], ["ccc"]]
        assert payload["input_type"] == "document"
        assert "input" not in payload

    def test_embed_contextualized_parses_nested_and_preserves_order(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        # Wire order shuffles the outer documents; parsing must restore index order.
        transport = FakeTransport(_context_response([2, 1, 3], order=[2, 0, 1]))
        provider = VoyageProvider("voyage-context-3", transport=transport)

        result = provider.embed_contextualized([["a", "b"], ["c"], ["d", "e", "f"]])

        # Nested shape mirrors the mixed batch sizes (2, 1, 3).
        assert [len(group) for group in result] == [2, 1, 3]
        # Each embedding encodes (outer_index, inner_index): order is preserved.
        assert result[0] == [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        assert result[1] == [[1.0, 0.0, 0.0]]
        assert result[2] == [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
        assert provider.last_usage_tokens == 100

    def test_context_model_documents_never_hit_flat_endpoint(self, monkeypatch):
        # 400-on-flat regression guard: a context model routed through the
        # document batch path must POST to the contextualized endpoint only.
        monkeypatch.setattr(provider_mod, "count_tokens", lambda _t: 1)
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        transport = FakeTransport(_context_response([1, 1]))
        provider = VoyageProvider("voyage-context-4", transport=transport)

        vectors = provider.embed_documents(["doc-a", "doc-b"])

        assert len(vectors) == 2
        urls = [call["url"] for call in transport.calls]
        assert urls == [_CONTEXT_URL]
        assert _FLAT_URL not in urls
        assert transport.calls[0]["payload"]["inputs"] == [["doc-a"], ["doc-b"]]

    def test_context_model_query_uses_context_endpoint(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        transport = FakeTransport(_context_response([1]))
        provider = VoyageProvider("voyage-context-3", transport=transport)

        provider.embed_query("what is the answer")

        call = transport.calls[0]
        assert call["url"] == _CONTEXT_URL
        assert call["payload"]["inputs"] == [["what is the answer"]]
        assert call["payload"]["input_type"] == "query"

    def test_flat_request_refuses_context_model(self, monkeypatch):
        # The flat payload path structurally refuses a context model, so the
        # 400-producing flat request can never be emitted.
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        provider = VoyageProvider("voyage-context-3", transport=FakeTransport())
        with pytest.raises(VoyageError) as excinfo:
            provider._request(("x",), input_type="document")
        assert excinfo.value.kind == "bad_request"

    def test_non_context_model_keeps_flat_endpoint(self, monkeypatch):
        monkeypatch.setattr(provider_mod, "count_tokens", lambda _t: 1)
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        flat_body = {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
        transport = FakeTransport(
            HttpResponse(status=200, headers={}, body=json.dumps(flat_body).encode())
        )
        provider = VoyageProvider("voyage-3", transport=transport)

        provider.embed_documents(["only-doc"])

        assert transport.calls[0]["url"] == _FLAT_URL
        assert transport.calls[0]["payload"]["input"] == ["only-doc"]
