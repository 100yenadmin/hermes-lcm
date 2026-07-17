from __future__ import annotations

from hermes_lcm.embedding_provider import default_chunk_model, embed_contextualized


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
