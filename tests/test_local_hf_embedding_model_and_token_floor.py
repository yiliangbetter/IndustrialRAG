"""Local HF embedding factory keeps a zero token limit and resolves the model id.

``HF_EMBED_MAX_TOKEN=0`` is an explicit limit. Treating that zero as "unset"
restores the 8192 default and changes how every local-embedding ingest chunks
text. The model id follows the same falsy-or-env rule: an omitted or blank
argument uses ``EMBEDDING_MODEL`` (or ``BAAI/bge-m3``), and only a non-empty
argument overrides it. A blank argument must not be handed to
SentenceTransformer, and the factory must not construct the model until the
first encode.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "raganything" / "local_hf_embedding.py"


def _load_embedding_module():
    name = "local_hf_embedding_model_token_floor_1023"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


embedding_module = _load_embedding_module()


def _install_fakes(monkeypatch, constructed):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            constructed.append({"model_id": model_id, "kwargs": dict(kwargs)})

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            del normalize_embeddings, convert_to_numpy
            return np.ones((len(list(texts)), 4), dtype=np.float32)

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

    if "lightrag" not in sys.modules:
        monkeypatch.setitem(sys.modules, "lightrag", types.ModuleType("lightrag"))
    fake_utils = types.ModuleType("lightrag.utils")
    fake_utils.EmbeddingFunc = FakeEmbeddingFunc
    monkeypatch.setitem(sys.modules, "lightrag.utils", fake_utils)


def _make(monkeypatch, constructed, **kwargs):
    monkeypatch.delenv("HF_HOME", raising=False)
    _install_fakes(monkeypatch, constructed)
    return embedding_module.make_local_hf_embedding_func(embedding_dim=4, **kwargs)


def test_zero_embed_max_token_is_kept(monkeypatch):
    monkeypatch.setenv("HF_EMBED_MAX_TOKEN", "0")
    embedding = _make(monkeypatch, [])

    assert embedding.max_token_size == 0
    assert embedding.embedding_dim == 4


def test_model_id_follows_env_until_the_caller_passes_one(monkeypatch):
    constructed = []

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("HF_EMBED_MAX_TOKEN", raising=False)
    default_embedding = _make(monkeypatch, constructed)
    assert constructed == []
    default_vectors = asyncio.run(default_embedding.func(["bolt"]))
    assert constructed[0]["model_id"] == "BAAI/bge-m3"
    assert default_vectors.shape == (1, 4)

    monkeypatch.setenv("EMBEDDING_MODEL", "org/from-env")
    omitted = _make(monkeypatch, constructed)
    asyncio.run(omitted.func(["bolt"]))
    blank = _make(monkeypatch, constructed, embedding_model="")
    asyncio.run(blank.func(["bolt"]))
    explicit = _make(monkeypatch, constructed, embedding_model="org/explicit")
    asyncio.run(explicit.func(["bolt"]))

    assert [item["model_id"] for item in constructed] == [
        "BAAI/bge-m3",
        "org/from-env",
        "org/from-env",
        "org/explicit",
    ]
