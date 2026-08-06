"""Regression tests for multimodal query cache hit/miss and soft-fail paths.

Open coverage PRs exercise cache *key* shape (#67/#68/#94) but not runtime
behavior: early return on hit, soft-continue when cache I/O fails, and the
enable_llm_cache gate. Regressions here either serve stale/wrong answers or
turn cache outages into hard query failures.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lightrag")

from raganything.query import QueryMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeLightRAG:
    def __init__(self):
        self.calls = []
        self.llm_response_cache = None

    async def aquery(self, query, param, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "mode": param.mode,
                "system_prompt": system_prompt,
            }
        )
        return "fresh-answer"


class FakeCache:
    def __init__(
        self,
        *,
        enable: bool = True,
        get_result=None,
        get_error: Exception | None = None,
        upsert_error: Exception | None = None,
        persist_error: Exception | None = None,
    ):
        self.global_config = {"enable_llm_cache": enable}
        self.get_result = get_result
        self.get_error = get_error
        self.upsert_error = upsert_error
        self.persist_error = persist_error
        self.requested_keys = []
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, cache_key):
        self.requested_keys.append(cache_key)
        if self.get_error is not None:
            raise self.get_error
        return self.get_result

    async def upsert(self, data):
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1
        if self.persist_error is not None:
            raise self.persist_error


class DummyQuery(QueryMixin):
    def __init__(self, *, init_ok: bool = True, init_error: str = "boom"):
        self.lightrag = FakeLightRAG()
        self.logger = FakeLogger()
        self.vision_model_func = None
        self.modal_processors = {}
        self._init_ok = init_ok
        self._init_error = init_error

    async def _ensure_lightrag_initialized(self):
        if self._init_ok:
            return {"success": True}
        return {"success": False, "error": self._init_error}


CONTENT = [
    {
        "type": "table",
        "table_data": "Name,Age\nAlice,25",
        "table_caption": ["ages"],
    }
]


@pytest.mark.asyncio
async def test_multimodal_cache_hit_returns_without_query():
    dummy = DummyQuery()
    cache = FakeCache(get_result={"return": "cached-answer", "cache_type": "multimodal_query"})
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="mix",
        vlm_enhanced=False,
    )

    assert result == "cached-answer"
    assert dummy.lightrag.calls == []
    assert cache.upserts == []
    assert cache.index_done_calls == 0
    assert len(cache.requested_keys) == 1


@pytest.mark.asyncio
async def test_multimodal_cache_empty_return_falls_through_to_query():
    """A cache entry without a usable 'return' must not short-circuit the query."""
    dummy = DummyQuery()
    cache = FakeCache(get_result={"return": "", "cache_type": "multimodal_query"})
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="mix",
        vlm_enhanced=False,
    )

    assert result == "fresh-answer"
    assert len(dummy.lightrag.calls) == 1
    assert len(cache.upserts) == 1
    assert cache.index_done_calls == 1


@pytest.mark.asyncio
async def test_multimodal_cache_nondict_entry_falls_through_to_query():
    dummy = DummyQuery()
    cache = FakeCache(get_result="not-a-dict")
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="mix",
        vlm_enhanced=False,
    )

    assert result == "fresh-answer"
    assert len(dummy.lightrag.calls) == 1


@pytest.mark.asyncio
async def test_multimodal_cache_miss_upserts_and_persists():
    dummy = DummyQuery()
    cache = FakeCache(get_result=None)
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="hybrid",
        vlm_enhanced=False,
    )

    # Production always forwards system_prompt (even None) into the key kwargs.
    expected_key = dummy._generate_multimodal_cache_key(
        "What is the age?", CONTENT, "hybrid", system_prompt=None, vlm_enhanced=False
    )
    assert result == "fresh-answer"
    assert cache.requested_keys == [expected_key]
    assert list(cache.upserts[0].keys()) == [expected_key]
    entry = cache.upserts[0][expected_key]
    assert entry["return"] == "fresh-answer"
    assert entry["cache_type"] == "multimodal_query"
    assert entry["original_query"] == "What is the age?"
    assert entry["multimodal_content_count"] == 1
    assert entry["mode"] == "hybrid"
    assert cache.index_done_calls == 1
    assert dummy.lightrag.calls[0]["query"].startswith("User query: What is the age?")


@pytest.mark.asyncio
async def test_multimodal_cache_get_error_soft_continues():
    dummy = DummyQuery()
    cache = FakeCache(get_error=RuntimeError("cache read failed"))
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="mix",
        vlm_enhanced=False,
    )

    assert result == "fresh-answer"
    assert len(dummy.lightrag.calls) == 1
    assert len(cache.upserts) == 1


@pytest.mark.asyncio
async def test_multimodal_cache_upsert_error_soft_continues():
    dummy = DummyQuery()
    cache = FakeCache(upsert_error=RuntimeError("cache write failed"))
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="mix",
        vlm_enhanced=False,
    )

    assert result == "fresh-answer"
    assert cache.upserts == []
    # Persist still attempted after a failed upsert.
    assert cache.index_done_calls == 1


@pytest.mark.asyncio
async def test_multimodal_cache_persist_error_soft_continues():
    dummy = DummyQuery()
    cache = FakeCache(persist_error=RuntimeError("flush failed"))
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="mix",
        vlm_enhanced=False,
    )

    assert result == "fresh-answer"
    assert len(cache.upserts) == 1
    assert cache.index_done_calls == 1


@pytest.mark.asyncio
async def test_multimodal_cache_disabled_skips_lookup_and_upsert():
    dummy = DummyQuery()
    cache = FakeCache(enable=False)
    dummy.lightrag.llm_response_cache = cache

    result = await dummy.aquery_with_multimodal(
        "What is the age?",
        multimodal_content=CONTENT,
        mode="mix",
        vlm_enhanced=False,
    )

    assert result == "fresh-answer"
    assert cache.requested_keys == []
    assert cache.upserts == []
    # Persistence is outside the enable_llm_cache gate in production.
    assert cache.index_done_calls == 1


@pytest.mark.asyncio
async def test_aquery_with_multimodal_init_failure_raises():
    dummy = DummyQuery(init_ok=False, init_error="missing llm_model_func")

    with pytest.raises(RuntimeError, match="missing llm_model_func"):
        await dummy.aquery_with_multimodal(
            "What is the age?",
            multimodal_content=CONTENT,
            mode="mix",
            vlm_enhanced=False,
        )

    assert dummy.lightrag.calls == []
