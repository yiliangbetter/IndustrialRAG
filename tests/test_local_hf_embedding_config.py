"""Local HF embedding factory gates: cache dir, max tokens, import, dim check.

Offline ingest scripts depend on HF_HOME / HF_EMBED_MAX_TOKEN and a clear
InstallError when sentence-transformers is missing. Wrong cache_folder or
silent dim mismatch corrupts the vector store.
"""

import os
import sys
import types

import numpy as np
import pytest

from raganything.local_hf_embedding import (
    ensure_hf_home_from_repo_fallback,
    make_local_hf_embedding_func,
)


def test_ensure_hf_home_sets_existing_repo_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    cache = tmp_path / ".hf_cache"
    cache.mkdir()
    ensure_hf_home_from_repo_fallback(tmp_path)
    assert os.environ["HF_HOME"] == str(cache)


def test_ensure_hf_home_noop_when_already_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/already/set")
    (tmp_path / ".hf_cache").mkdir()
    ensure_hf_home_from_repo_fallback(tmp_path)
    assert os.environ["HF_HOME"] == "/already/set"


def test_ensure_hf_home_noop_when_cache_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    ensure_hf_home_from_repo_fallback(tmp_path)
    assert "HF_HOME" not in os.environ


def test_ensure_hf_home_noop_when_repo_root_is_none(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    ensure_hf_home_from_repo_fallback(None)
    assert "HF_HOME" not in os.environ


def _install_fake_embedding_stack(
    monkeypatch, encode_vectors=None, init_kwargs_holder=None
):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            if init_kwargs_holder is not None:
                init_kwargs_holder.update(kwargs)

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            if encode_vectors is not None:
                return encode_vectors
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


def test_missing_sentence_transformers_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="raganything\\[local-embed\\]"):
        make_local_hf_embedding_func(embedding_dim=1024)


def test_hf_embed_max_token_and_cache_folder(monkeypatch):
    init_kwargs = {}
    _install_fake_embedding_stack(monkeypatch, init_kwargs_holder=init_kwargs)
    monkeypatch.setenv("HF_EMBED_MAX_TOKEN", "4096")
    monkeypatch.setenv("HF_HOME", "/models/hf")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    embedding = make_local_hf_embedding_func(embedding_dim=3)
    assert embedding.max_token_size == 4096
    assert embedding.embedding_dim == 3


@pytest.mark.asyncio
async def test_cache_folder_passed_on_first_encode(monkeypatch):
    init_kwargs = {}
    _install_fake_embedding_stack(monkeypatch, init_kwargs_holder=init_kwargs)
    monkeypatch.setenv("HF_HOME", "/opt/hf-cache")

    embedding = make_local_hf_embedding_func(
        embedding_dim=3, embedding_model="fake/model"
    )
    result = await embedding.func(["hello"])
    assert init_kwargs["cache_folder"] == "/opt/hf-cache"
    assert result.shape == (1, 3)
    assert result.dtype == np.float32


@pytest.mark.asyncio
async def test_dim_mismatch_raises_value_error(monkeypatch):
    _install_fake_embedding_stack(monkeypatch, encode_vectors=[[0.1, 0.2, 0.3, 0.4]])
    embedding = make_local_hf_embedding_func(
        embedding_dim=3, embedding_model="fake/model"
    )
    with pytest.raises(ValueError, match="EMBEDDING_DIM=3"):
        await embedding.func(["hello"])
