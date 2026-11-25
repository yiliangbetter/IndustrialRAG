"""The local HF embedder must refuse a matrix whose width is not EMBEDDING_DIM."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_embedding_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "raganything" / "local_hf_embedding.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_hf_embedding_dimension_guard_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_encoder(monkeypatch, vectors):
    class FakeSentenceTransformer:
        built = []

        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs
            self.calls = []
            FakeSentenceTransformer.built.append(self)

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            self.calls.append(
                {
                    "texts": list(texts),
                    "normalize_embeddings": normalize_embeddings,
                    "convert_to_numpy": convert_to_numpy,
                }
            )
            current = vectors["value"]
            return current

    class FakeEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            self.embedding_dim = embedding_dim
            self.max_token_size = max_token_size
            self.func = func

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", fake_sentence_transformers
    )

    fake_lightrag = types.ModuleType("lightrag")
    fake_lightrag.__path__ = []
    fake_lightrag_utils = types.ModuleType("lightrag.utils")
    fake_lightrag_utils.EmbeddingFunc = FakeEmbeddingFunc
    monkeypatch.setitem(sys.modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(sys.modules, "lightrag.utils", fake_lightrag_utils)
    return FakeSentenceTransformer


def test_matching_bge_width_is_float32_and_reuses_one_model(monkeypatch, tmp_path):
    module = _load_embedding_module()
    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_dir))
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    vectors = {"value": np.ones((2, 1024), dtype=np.float64)}
    encoder = _install_encoder(monkeypatch, vectors)

    embedding = module.make_local_hf_embedding_func(1024, embedding_model=None)
    first = asyncio.run(embedding.func(["bearing clearance", "torque 40 Nm"]))
    vectors["value"] = np.full((1, 1024), 0.5, dtype=np.float64)
    second = asyncio.run(embedding.func(["seal kit"]))

    assert embedding.embedding_dim == 1024
    assert first.dtype == np.float32
    assert first.shape == (2, 1024)
    assert second.dtype == np.float32
    assert second.shape == (1, 1024)
    assert np.allclose(second, 0.5)
    assert len(encoder.built) == 1
    assert encoder.built[0].model_id == "BAAI/bge-m3"
    assert encoder.built[0].kwargs == {"cache_folder": str(cache_dir)}
    assert encoder.built[0].calls == [
        {
            "texts": ["bearing clearance", "torque 40 Nm"],
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        },
        {
            "texts": ["seal kit"],
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        },
    ]


def test_openai_width_is_rejected_for_bge_dimension_then_a_later_batch_works(
    monkeypatch,
):
    module = _load_embedding_module()
    monkeypatch.delenv("HF_HOME", raising=False)
    vectors = {"value": np.zeros((2, 1536), dtype=np.float64)}
    encoder = _install_encoder(monkeypatch, vectors)

    embedding = module.make_local_hf_embedding_func(1024, embedding_model="BAAI/bge-m3")

    with pytest.raises(ValueError, match=r"Embedding shape 1536 != EMBEDDING_DIM=1024"):
        asyncio.run(embedding.func(["housing", "gasket"]))

    assert encoder.built[0].model_id == "BAAI/bge-m3"
    assert encoder.built[0].kwargs == {}
    assert encoder.built[0].calls[0]["texts"] == ["housing", "gasket"]

    vectors["value"] = np.arange(1024, dtype=np.float64).reshape(1, 1024)
    recovered = asyncio.run(embedding.func(["housing"]))

    assert recovered.dtype == np.float32
    assert recovered.shape == (1, 1024)
    assert len(encoder.built) == 1
