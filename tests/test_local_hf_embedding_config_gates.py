"""Config and fail-closed gates for local Hugging Face embeddings.

Open PRs #58–#63 cover dimension mismatch / concurrency on
``tests/test_local_hf_embedding_async.py``. This module covers setup contracts
that still lack dedicated assertions on ``main``.
"""

from __future__ import annotations

import types

import pytest

from raganything import local_hf_embedding as local_hf


def _install_fake_embedding_stack(monkeypatch, *, transformer_cls):
    """Stub sentence-transformers + LightRAG EmbeddingFunc for unit tests."""

    class FakeEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = transformer_cls
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_st)

    fake_lightrag = types.ModuleType("lightrag")
    fake_utils = types.ModuleType("lightrag.utils")
    fake_utils.EmbeddingFunc = FakeEmbeddingFunc
    monkeypatch.setitem(__import__("sys").modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(__import__("sys").modules, "lightrag.utils", fake_utils)


def test_make_local_hf_embedding_func_requires_sentence_transformers(monkeypatch):
    """Missing optional dependency must fail closed with an install hint."""

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(__import__("sys").modules, "sentence_transformers", raising=False)

    with pytest.raises(ImportError, match="raganything\\[local-embed\\]") as exc_info:
        local_hf.make_local_hf_embedding_func(embedding_dim=1024)

    assert "sentence-transformers" in str(exc_info.value)


def test_make_local_hf_embedding_func_reads_max_token_env(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            return [[0.0] * 4 for _ in texts]

    _install_fake_embedding_stack(monkeypatch, transformer_cls=FakeSentenceTransformer)
    monkeypatch.setenv("HF_EMBED_MAX_TOKEN", "2048")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    embedding = local_hf.make_local_hf_embedding_func(
        embedding_dim=4, embedding_model="fake/model"
    )

    assert embedding.max_token_size == 2048
    assert embedding.embedding_dim == 4


@pytest.mark.asyncio
async def test_make_local_hf_embedding_func_passes_hf_home_cache_folder(monkeypatch):
    constructed = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            constructed["model_id"] = model_id
            constructed["kwargs"] = kwargs

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            return [[1.0, 2.0, 3.0] for _ in texts]

    _install_fake_embedding_stack(monkeypatch, transformer_cls=FakeSentenceTransformer)
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache-under-test")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    embedding = local_hf.make_local_hf_embedding_func(
        embedding_dim=3, embedding_model="org/model-a"
    )

    # Lazy model construction happens on first encode.
    await embedding.func(["hello"])

    assert constructed["model_id"] == "org/model-a"
    assert constructed["kwargs"]["cache_folder"] == "/tmp/hf-cache-under-test"


@pytest.mark.asyncio
async def test_make_local_hf_embedding_func_uses_embedding_model_env(monkeypatch):
    constructed = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            constructed["model_id"] = model_id
            constructed["kwargs"] = kwargs

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            return [[0.5, 0.5] for _ in texts]

    _install_fake_embedding_stack(monkeypatch, transformer_cls=FakeSentenceTransformer)
    monkeypatch.setenv("EMBEDDING_MODEL", "env/default-model")
    monkeypatch.delenv("HF_HOME", raising=False)

    embedding = local_hf.make_local_hf_embedding_func(embedding_dim=2)
    await embedding.func(["x"])

    assert constructed["model_id"] == "env/default-model"
    assert "cache_folder" not in constructed["kwargs"]
