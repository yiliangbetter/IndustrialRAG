"""Blank HF_HOME must follow the unset-cache contract for local embeddings.

A `.env` line `HF_HOME=` is common on offline machines. Treating that blank
value as a real cache directory skips `<repo>/.hf_cache` or passes an empty
`cache_folder` into sentence-transformers.
"""

import os
import sys
import types

import pytest

from raganything.local_hf_embedding import (
    ensure_hf_home_from_repo_fallback,
    make_local_hf_embedding_func,
)


def test_blank_hf_home_uses_existing_repo_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", "")
    cache = tmp_path / ".hf_cache"
    cache.mkdir()

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert os.environ["HF_HOME"] == str(cache)


def test_blank_hf_home_stays_blank_when_repo_cache_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", "")

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert os.environ["HF_HOME"] == ""


def _install_fake_embedding_stack(monkeypatch, init_kwargs):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            init_kwargs["model_id"] = model_id
            init_kwargs["kwargs"] = kwargs

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            return [[1.0, 0.0] for _ in texts]

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
async def test_blank_hf_home_does_not_set_cache_folder(monkeypatch):
    init_kwargs = {}
    _install_fake_embedding_stack(monkeypatch, init_kwargs)
    monkeypatch.setenv("HF_HOME", "")
    monkeypatch.setenv("HF_EMBED_MAX_TOKEN", "8192")

    embedding = make_local_hf_embedding_func(
        embedding_dim=2, embedding_model="fake/bge"
    )
    await embedding.func(["torque spec"])

    assert "cache_folder" not in init_kwargs["kwargs"]


@pytest.mark.asyncio
async def test_cache_folder_is_fixed_at_factory_time(monkeypatch):
    init_kwargs = {}
    _install_fake_embedding_stack(monkeypatch, init_kwargs)
    monkeypatch.setenv("HF_HOME", "/models/first")
    monkeypatch.setenv("HF_EMBED_MAX_TOKEN", "8192")

    embedding = make_local_hf_embedding_func(
        embedding_dim=2, embedding_model="fake/bge"
    )
    monkeypatch.setenv("HF_HOME", "/models/later")
    await embedding.func(["torque spec"])

    assert init_kwargs["kwargs"]["cache_folder"] == "/models/first"
    assert init_kwargs["model_id"] == "fake/bge"
