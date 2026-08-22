"""Regression tests for QueryMixin.aquery_with_multimodal orchestration.

Cache *key* identity is covered by open #125. This file covers the remaining
fail-closed and cache-runtime contract: init failure, empty-content fallback,
cache hit short-circuit, write+persist on miss, disabled cache, and soft
failure of cache get/upsert/persist.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("lightrag")

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.debugs = []
        self.errors = []
        self.warnings = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg) % args if args else str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg) % args if args else str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))


class FakeLLMCache:
    def __init__(self, *, enabled=True, hit=None, fail_get=False, fail_upsert=False):
        self.global_config = {"enable_llm_cache": enabled}
        self._hit = hit
        self.fail_get = fail_get
        self.fail_upsert = fail_upsert
        self.get_calls = []
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        self.get_calls.append(key)
        if self.fail_get:
            raise RuntimeError("cache get failed")
        return self._hit

    async def upsert(self, data):
        if self.fail_upsert:
            raise RuntimeError("cache upsert failed")
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_query(*, cache=None):
    query = object.__new__(QueryMixin)
    query.logger = FakeLogger()
    query.modal_processors = {}
    query.lightrag = SimpleNamespace(llm_response_cache=cache)
    return query


@pytest.mark.asyncio
async def test_aquery_with_multimodal_fails_closed_when_init_fails():
    query = _make_query()

    async def bad_init():
        return {"success": False, "error": "no models"}

    query._ensure_lightrag_initialized = bad_init
    query.aquery = AsyncMock(return_value="should-not-run")

    with pytest.raises(RuntimeError, match="LightRAG initialization failed: no models"):
        await query.aquery_with_multimodal(
            "question",
            multimodal_content=[{"type": "table", "table_data": "a,b"}],
        )

    query.aquery.assert_not_awaited()


@pytest.mark.asyncio
async def test_aquery_with_multimodal_empty_content_falls_back_to_text():
    query = _make_query()

    async def ok_init():
        return {"success": True}

    query._ensure_lightrag_initialized = ok_init
    query.aquery = AsyncMock(return_value="text-only")

    result = await query.aquery_with_multimodal(
        "plain question",
        multimodal_content=None,
        mode="local",
        system_prompt="sys",
        top_k=3,
    )

    assert result == "text-only"
    query.aquery.assert_awaited_once_with(
        "plain question",
        mode="local",
        system_prompt="sys",
        top_k=3,
    )


@pytest.mark.asyncio
async def test_aquery_with_multimodal_returns_cache_hit_without_processing():
    cache = FakeLLMCache(
        hit={"return": "cached-answer", "cache_type": "multimodal_query"}
    )
    query = _make_query(cache=cache)

    async def ok_init():
        return {"success": True}

    query._ensure_lightrag_initialized = ok_init
    query.aquery = AsyncMock(return_value="fresh")
    query._process_multimodal_query_content = AsyncMock(return_value="enhanced")
    query._generate_multimodal_cache_key = lambda *a, **k: "mm-cache-key"

    result = await query.aquery_with_multimodal(
        "analyze table",
        multimodal_content=[{"type": "table", "table_data": "a,b\n1,2"}],
        mode="mix",
        system_prompt="be precise",
    )

    assert result == "cached-answer"
    assert cache.get_calls == ["mm-cache-key"]
    query._process_multimodal_query_content.assert_not_awaited()
    query.aquery.assert_not_awaited()
    assert cache.upserts == []
    assert cache.index_done_calls == 0


@pytest.mark.asyncio
async def test_aquery_with_multimodal_writes_and_persists_cache_on_miss():
    cache = FakeLLMCache(hit=None)
    query = _make_query(cache=cache)

    async def ok_init():
        return {"success": True}

    query._ensure_lightrag_initialized = ok_init
    query.aquery = AsyncMock(return_value="fresh-answer")
    query._process_multimodal_query_content = AsyncMock(
        return_value="User query: analyze\nRelated table content: dims"
    )
    query._generate_multimodal_cache_key = lambda *a, **k: "mm-key-42"

    result = await query.aquery_with_multimodal(
        "analyze",
        multimodal_content=[{"type": "table", "table_data": "x,y"}],
        mode="hybrid",
        system_prompt="sys",
        top_k=5,
    )

    assert result == "fresh-answer"
    query._process_multimodal_query_content.assert_awaited_once_with(
        "analyze",
        [{"type": "table", "table_data": "x,y"}],
    )
    query.aquery.assert_awaited_once_with(
        "User query: analyze\nRelated table content: dims",
        mode="hybrid",
        system_prompt="sys",
        top_k=5,
    )

    assert len(cache.upserts) == 1
    entry = cache.upserts[0]["mm-key-42"]
    assert entry == {
        "return": "fresh-answer",
        "cache_type": "multimodal_query",
        "original_query": "analyze",
        "multimodal_content_count": 1,
        "mode": "hybrid",
    }
    assert cache.index_done_calls == 1


@pytest.mark.asyncio
async def test_aquery_with_multimodal_continues_when_cache_ops_fail():
    cache = FakeLLMCache(hit=None, fail_get=True, fail_upsert=True)
    query = _make_query(cache=cache)

    async def ok_init():
        return {"success": True}

    query._ensure_lightrag_initialized = ok_init
    query.aquery = AsyncMock(return_value="still-ok")
    query._process_multimodal_query_content = AsyncMock(return_value="enhanced-q")
    query._generate_multimodal_cache_key = lambda *a, **k: "mm-soft-fail"

    async def boom_persist():
        raise RuntimeError("persist failed")

    cache.index_done_callback = boom_persist

    result = await query.aquery_with_multimodal(
        "q",
        multimodal_content=[{"type": "equation", "latex": "E=mc^2"}],
        mode="mix",
    )

    assert result == "still-ok"
    query.aquery.assert_awaited_once()
    assert any(
        "Error accessing multimodal query cache" in d for d in query.logger.debugs
    )
    assert any("Error saving multimodal query to cache" in d for d in query.logger.debugs)
    assert any(
        "Error persisting multimodal query cache" in d for d in query.logger.debugs
    )


@pytest.mark.asyncio
async def test_aquery_with_multimodal_skips_cache_when_disabled():
    cache = FakeLLMCache(enabled=False, hit={"return": "should-not-use"})
    query = _make_query(cache=cache)

    async def ok_init():
        return {"success": True}

    query._ensure_lightrag_initialized = ok_init
    query.aquery = AsyncMock(return_value="live")
    query._process_multimodal_query_content = AsyncMock(return_value="enhanced")
    query._generate_multimodal_cache_key = lambda *a, **k: "unused-key"

    result = await query.aquery_with_multimodal(
        "q",
        multimodal_content=[{"type": "table", "table_data": "a"}],
    )

    assert result == "live"
    assert cache.get_calls == []
    assert cache.upserts == []
    # Persist still attempted when cache object exists (current contract)
    assert cache.index_done_calls == 1
