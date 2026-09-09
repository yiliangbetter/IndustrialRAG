"""Local HF embedding factory must reuse a single SentenceTransformer instance.

make_local_hf_embedding_func lazy-loads the model on first encode. Reloading
on every batch would change HF_HOME/cache_folder application and stall ingest.
"""

import sys
import types

import numpy as np
import pytest

from raganything.local_hf_embedding import make_local_hf_embedding_func


def _install_fake_embedding_stack(monkeypatch, inits):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            inits.append({"model_id": model_id, "kwargs": kwargs, "self": self})

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            return [[0.25, 0.75] for _ in texts]

    class FakeEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    fake_lightrag = types.ModuleType("lightrag")
    fake_utils = types.ModuleType("lightrag.utils")
    fake_utils.EmbeddingFunc = FakeEmbeddingFunc
    monkeypatch.setitem(sys.modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(sys.modules, "lightrag.utils", fake_utils)


@pytest.mark.asyncio
async def test_sentence_transformer_is_constructed_once(monkeypatch):
    inits = []
    _install_fake_embedding_stack(monkeypatch, inits)
    monkeypatch.delenv("HF_HOME", raising=False)

    embedding = make_local_hf_embedding_func(
        embedding_dim=2, embedding_model="fake/bge"
    )
    first = await embedding.func(["alpha"])
    second = await embedding.func(["beta", "gamma"])

    assert len(inits) == 1
    assert inits[0]["model_id"] == "fake/bge"
    assert "cache_folder" not in inits[0]["kwargs"]
    assert first.shape == (1, 2)
    assert second.shape == (2, 2)
    assert first.dtype == np.float32
    assert second.dtype == np.float32
