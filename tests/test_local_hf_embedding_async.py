import asyncio
import threading
import time
import types

import pytest

from raganything.local_hf_embedding import make_local_hf_embedding_func


@pytest.mark.asyncio
async def test_hf_embed_does_not_block_event_loop(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            # Simulate CPU/GPU-bound synchronous work.
            time.sleep(0.2)
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
        embedding_dim=3, embedding_model="fake/model"
    )

    tick_count = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal tick_count
        while not stop.is_set():
            tick_count += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    try:
        result = await embedding.func(["a", "b"])
    finally:
        stop.set()
        await ticker_task

    # This reproduces the prior issue: without to_thread, encode() blocks the loop
    # and ticker does not advance during embedding.
    assert tick_count >= 5
    assert result.shape == (2, 3)


@pytest.mark.asyncio
async def test_hf_embed_serializes_model_initialization_and_encode(monkeypatch):
    state = {"constructed": 0, "active_encodes": 0, "max_active_encodes": 0}
    state_lock = threading.Lock()

    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            with state_lock:
                state["constructed"] += 1
            time.sleep(0.05)

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            with state_lock:
                state["active_encodes"] += 1
                state["max_active_encodes"] = max(
                    state["max_active_encodes"], state["active_encodes"]
                )
            time.sleep(0.05)
            with state_lock:
                state["active_encodes"] -= 1
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
        embedding_dim=3, embedding_model="fake/model"
    )
    results = await asyncio.gather(
        *(embedding.func([str(index)]) for index in range(4))
    )

    assert state["constructed"] == 1
    assert state["max_active_encodes"] == 1
    assert all(result.shape == (1, 3) for result in results)
