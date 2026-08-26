"""Local HF embeddings must fail closed when vector width != EMBEDDING_DIM.

A silent shape mismatch would write the wrong dimensionality into LightRAG
storage and break later retrieval. Existing async tests only cover the
matching-width happy path.
"""

import types

import pytest

from raganything.local_hf_embedding import make_local_hf_embedding_func


@pytest.mark.asyncio
async def test_hf_embed_raises_when_model_width_does_not_match_config(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            return [[1.0, 2.0, 3.0] for _ in texts]

    class FakeEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(
        __import__("sys").modules, "sentence_transformers", fake_sentence_transformers
    )

    fake_lightrag = types.ModuleType("lightrag")
    fake_lightrag_utils = types.ModuleType("lightrag.utils")
    fake_lightrag_utils.EmbeddingFunc = FakeEmbeddingFunc
    monkeypatch.setitem(__import__("sys").modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(
        __import__("sys").modules, "lightrag.utils", fake_lightrag_utils
    )

    embedding = make_local_hf_embedding_func(
        embedding_dim=1024, embedding_model="fake/model"
    )

    with pytest.raises(ValueError, match="EMBEDDING_DIM=1024"):
        await embedding.func(["hello"])
