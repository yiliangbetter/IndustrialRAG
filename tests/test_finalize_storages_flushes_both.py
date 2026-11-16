"""RAGAnything.finalize_storages must flush every live storage.

tests/test_close_event_loop.py copies the close() control flow, and the close()
coverage calls a replaced finalize_storages. Neither one runs this method, so a
rewrite can skip the parse cache, skip LightRAG, or swallow a flush error
while those tests still pass.
"""

import asyncio
import logging

import pytest

from raganything.raganything import RAGAnything


def _bare_rag():
    """Skip __post_init__ so finalizing storages does not start LightRAG."""
    rag = object.__new__(RAGAnything)
    rag.logger = logging.getLogger("raganything.test_finalize_storages")
    return rag


@pytest.mark.asyncio
async def test_finalize_storages_overlaps_cache_and_lightrag(caplog):
    rag = _bare_rag()
    release_cache = asyncio.Event()
    order = []

    class Cache:
        async def finalize(self):
            order.append("cache-start")
            await release_cache.wait()
            order.append("cache-end")

    class LightRAG:
        async def finalize_storages(self):
            order.append("lightrag-start")
            release_cache.set()
            order.append("lightrag-end")

    rag.parse_cache = Cache()
    rag.lightrag = LightRAG()

    with caplog.at_level(logging.INFO, logger=rag.logger.name):
        await asyncio.wait_for(rag.finalize_storages(), timeout=1)

    assert order == [
        "cache-start",
        "lightrag-start",
        "lightrag-end",
        "cache-end",
    ]
    assert "Successfully finalized all RAGAnything storages" in caplog.text


@pytest.mark.asyncio
async def test_cache_flush_error_still_finalizes_lightrag(caplog):
    rag = _bare_rag()
    started = []

    class Cache:
        async def finalize(self):
            started.append("cache")
            raise RuntimeError("cache flush failed")

    class LightRAG:
        async def finalize_storages(self):
            started.append("lightrag")

    rag.parse_cache = Cache()
    rag.lightrag = LightRAG()

    with caplog.at_level(logging.INFO, logger=rag.logger.name):
        with pytest.raises(RuntimeError, match="cache flush failed"):
            await rag.finalize_storages()

    assert started == ["cache", "lightrag"]
    assert "Error during storage finalization: cache flush failed" in caplog.text
    assert "Successfully finalized all RAGAnything storages" not in caplog.text


@pytest.mark.asyncio
async def test_lightrag_flush_error_propagates_after_cache_succeeds(caplog):
    rag = _bare_rag()
    started = []

    class Cache:
        async def finalize(self):
            started.append("cache")

    class LightRAG:
        async def finalize_storages(self):
            started.append("lightrag")
            raise RuntimeError("index flush failed")

    rag.parse_cache = Cache()
    rag.lightrag = LightRAG()

    with caplog.at_level(logging.INFO, logger=rag.logger.name):
        with pytest.raises(RuntimeError, match="index flush failed"):
            await rag.finalize_storages()

    assert started == ["cache", "lightrag"]
    assert "Error during storage finalization: index flush failed" in caplog.text
    assert "Successfully finalized all RAGAnything storages" not in caplog.text


@pytest.mark.asyncio
async def test_finalize_storages_with_nothing_initialized_returns(caplog):
    rag = _bare_rag()
    rag.parse_cache = None
    rag.lightrag = None

    with caplog.at_level(logging.DEBUG, logger=rag.logger.name):
        await rag.finalize_storages()

    assert "No storages to finalize" in caplog.text
    assert "Successfully finalized all RAGAnything storages" not in caplog.text
