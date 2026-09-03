"""HF_EMBED_MAX_TOKEN must fail closed on invalid values.

The factory reads the env var with int() at construction. A typo would
otherwise crash later inside LightRAG token accounting, or worse, be
silently ignored. Valid numeric values are covered by #122; this locks
the invalid-input contract and the 8192 default.
"""

import sys
import types

import pytest

from raganything.local_hf_embedding import make_local_hf_embedding_func


def _install_fake_embedding_stack(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            return [[1.0, 0.0, 0.0] for _ in texts]

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


def test_hf_embed_max_token_defaults_to_8192(monkeypatch):
    _install_fake_embedding_stack(monkeypatch)
    monkeypatch.delenv("HF_EMBED_MAX_TOKEN", raising=False)

    embedding = make_local_hf_embedding_func(embedding_dim=3)
    assert embedding.max_token_size == 8192


def test_hf_embed_max_token_invalid_raises(monkeypatch):
    _install_fake_embedding_stack(monkeypatch)
    monkeypatch.setenv("HF_EMBED_MAX_TOKEN", "abc")

    with pytest.raises(ValueError):
        make_local_hf_embedding_func(embedding_dim=3)


def test_hf_embed_max_token_empty_raises(monkeypatch):
    _install_fake_embedding_stack(monkeypatch)
    monkeypatch.setenv("HF_EMBED_MAX_TOKEN", "")

    with pytest.raises(ValueError):
        make_local_hf_embedding_func(embedding_dim=3)
