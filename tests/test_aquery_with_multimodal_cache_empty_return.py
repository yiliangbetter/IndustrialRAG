"""Empty or malformed multimodal-query cache hits must not short-circuit.

#130 covers a successful cache hit and fail-open cache I/O. The live path
only returns a hit when cached_result is a dict with a truthy "return".
Treating "" / missing / non-dict as a hit would serve blank answers forever.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class FakeLLMCache:
    def __init__(self, hit):
        self.global_config = {"enable_llm_cache": True}
        self._hit = hit
        self.get_calls = []
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        self.get_calls.append(key)
        return self._hit

    async def upsert(self, data):
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_query(hit):
    cache = FakeLLMCache(hit)
    query = object.__new__(QueryMixin)
    query.logger = FakeLogger()
    query.modal_processors = {}
    query.lightrag = SimpleNamespace(llm_response_cache=cache)
    query._cache = cache

    async def ok_init():
        return {"success": True}

    query._ensure_lightrag_initialized = ok_init
    query.aquery = AsyncMock(return_value="live-answer")
    query._process_multimodal_query_content = AsyncMock(return_value="enhanced-q")
    query._generate_multimodal_cache_key = lambda *a, **k: "mm-empty-return"
    return query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hit",
    [
        {"return": ""},
        {"cache_type": "multimodal_query"},
        "not-a-dict",
        None,
    ],
    ids=["empty_return", "missing_return", "non_dict", "none"],
)
async def test_falsy_or_malformed_cache_hit_falls_through_to_live_query(hit):
    query = _make_query(hit)
    content = [{"type": "table", "table_data": "a,b"}]

    result = await query.aquery_with_multimodal(
        "analyze table",
        multimodal_content=content,
        mode="mix",
        system_prompt="sys",
    )

    assert result == "live-answer"
    query._process_multimodal_query_content.assert_awaited_once_with(
        "analyze table", content
    )
    query.aquery.assert_awaited_once_with(
        "enhanced-q",
        mode="mix",
        system_prompt="sys",
    )
    assert query._cache.get_calls == ["mm-empty-return"]
    assert query._cache.upserts == [
        {
            "mm-empty-return": {
                "return": "live-answer",
                "cache_type": "multimodal_query",
                "original_query": "analyze table",
                "multimodal_content_count": 1,
                "mode": "mix",
            }
        }
    ]
    assert query._cache.index_done_calls == 1


@pytest.mark.asyncio
async def test_empty_multimodal_list_falls_back_to_text_query():
    """[] is falsy and must skip cache/enhancement, same as omitted content."""
    query = _make_query({"return": "should-not-use"})

    result = await query.aquery_with_multimodal(
        "plain question",
        multimodal_content=[],
        mode="local",
        system_prompt="sys",
        top_k=3,
    )

    assert result == "live-answer"
    query._process_multimodal_query_content.assert_not_awaited()
    query.aquery.assert_awaited_once_with(
        "plain question",
        mode="local",
        system_prompt="sys",
        top_k=3,
    )
    assert query._cache.get_calls == []
    assert query._cache.upserts == []
