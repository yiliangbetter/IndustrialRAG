"""Blank multimodal query cache entries must not be served as answers.

``aquery_with_multimodal`` returns a cached ``return`` only when that value
is non-empty. An empty string and a missing ``return`` are incomplete
entries: the table or figure still has to be described and the live query
has to run, and the fresh answer has to replace the blank entry.

A whitespace-only cached answer is stored text, so it is still a hit.
When ``enable_llm_cache`` is omitted, the cache stays on.
"""

import pytest

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


class FakeLLMCache:
    def __init__(self, hit, global_config=None):
        self.hit = hit
        self.global_config = {} if global_config is None else global_config
        self.lookups = []
        self.upserts = []
        self.persisted = 0

    async def get_by_id(self, key):
        self.lookups.append(key)
        return self.hit

    async def upsert(self, data):
        self.upserts.append(data)

    async def index_done_callback(self):
        self.persisted += 1


class FakeLightRAG:
    def __init__(self, cache):
        self.llm_response_cache = cache


def _query(cache):
    query = QueryMixin()
    query.logger = FakeLogger()
    query.lightrag = FakeLightRAG(cache)
    return query


TABLE = [{"type": "table", "table_data": "bearing load 12 kN"}]


def _arm(query, answer="fresh-answer"):
    calls = {"processed": 0, "aquery": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_process(base_query, multimodal_content):
        calls["processed"] += 1
        calls["content"] = multimodal_content
        return f"User query: {base_query}\nRelated table content: bearing load"

    async def fake_aquery(enhanced, mode="mix", system_prompt=None, **kwargs):
        calls["aquery"].append(
            {
                "enhanced": enhanced,
                "mode": mode,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        return answer

    query._ensure_lightrag_initialized = fake_ensure
    query._process_multimodal_query_content = fake_process
    query.aquery = fake_aquery
    return calls


async def _ask(query):
    return await query.aquery_with_multimodal(
        "What is the bearing load?",
        multimodal_content=TABLE,
        mode="hybrid",
        system_prompt="Answer from the manual.",
        top_k=5,
    )


@pytest.mark.asyncio
async def test_blank_cached_return_reruns_query_and_replaces_entry():
    cache = FakeLLMCache(
        {"return": "", "cache_type": "multimodal_query"},
        global_config={"enable_llm_cache": True},
    )
    query = _query(cache)
    calls = _arm(query)

    result = await _ask(query)

    assert result == "fresh-answer"
    assert calls["processed"] == 1
    assert calls["content"] == TABLE
    assert calls["aquery"] == [
        {
            "enhanced": (
                "User query: What is the bearing load?\n"
                "Related table content: bearing load"
            ),
            "mode": "hybrid",
            "system_prompt": "Answer from the manual.",
            "kwargs": {"top_k": 5},
        }
    ]
    assert len(cache.lookups) == 1
    assert cache.lookups[0].startswith("multimodal_query:")
    assert len(cache.upserts) == 1
    stored = cache.upserts[0]
    assert list(stored) == cache.lookups
    entry = stored[cache.lookups[0]]
    assert entry["return"] == "fresh-answer"
    assert entry["cache_type"] == "multimodal_query"
    assert entry["original_query"] == "What is the bearing load?"
    assert entry["multimodal_content_count"] == 1
    assert entry["mode"] == "hybrid"
    assert cache.persisted == 1


@pytest.mark.asyncio
async def test_missing_return_field_is_not_a_cache_hit():
    cache = FakeLLMCache(
        {"cache_type": "multimodal_query"},
        global_config={"enable_llm_cache": True},
    )
    query = _query(cache)
    calls = _arm(query)

    result = await _ask(query)

    assert result == "fresh-answer"
    assert calls["processed"] == 1
    assert cache.upserts[0][cache.lookups[0]]["return"] == "fresh-answer"


@pytest.mark.asyncio
async def test_whitespace_cached_return_is_served_without_requery():
    cache = FakeLLMCache(
        {"return": "   ", "cache_type": "multimodal_query"},
        global_config={"enable_llm_cache": True},
    )
    query = _query(cache)
    calls = _arm(query)

    result = await _ask(query)

    assert result == "   "
    assert calls["processed"] == 0
    assert calls["aquery"] == []
    assert cache.upserts == []
    assert cache.persisted == 0


@pytest.mark.asyncio
async def test_non_dict_cache_payload_is_not_returned_as_the_answer():
    cache = FakeLLMCache(
        "cached-answer",
        global_config={"enable_llm_cache": True},
    )
    query = _query(cache)
    calls = _arm(query)

    result = await _ask(query)

    assert result == "fresh-answer"
    assert result != "cached-answer"
    assert calls["processed"] == 1
    assert cache.upserts[0][cache.lookups[0]]["return"] == "fresh-answer"


@pytest.mark.asyncio
async def test_omitted_enable_llm_cache_still_serves_a_stored_answer():
    cache = FakeLLMCache(
        {"return": "stored bearing answer", "cache_type": "multimodal_query"},
        global_config={},
    )
    query = _query(cache)
    calls = _arm(query)

    result = await _ask(query)

    assert result == "stored bearing answer"
    assert calls["processed"] == 0
    assert calls["aquery"] == []
    assert cache.upserts == []
