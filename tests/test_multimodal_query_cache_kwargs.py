"""Regression: multimodal query cache must isolate QueryParam-affecting kwargs."""

import logging
from types import SimpleNamespace

import pytest

from raganything.query import QueryMixin


class _CacheKeyHost(QueryMixin):
    """Minimal host for exercising QueryMixin._generate_multimodal_cache_key."""

    pass


def _key(**kwargs) -> str:
    host = _CacheKeyHost()
    return host._generate_multimodal_cache_key(
        "What does this show?",
        [{"type": "table", "table_body": "| A | 1 |\n| B | 2 |"}],
        "mix",
        **kwargs,
    )


def test_conversation_history_changes_cache_key():
    base = _key(conversation_history=[])
    follow_up = _key(
        conversation_history=[
            {"role": "user", "content": "Focus on column B"},
            {"role": "assistant", "content": "Understood."},
        ]
    )
    assert base != follow_up


def test_only_need_context_changes_cache_key():
    assert _key(only_need_context=False) != _key(only_need_context=True)


def test_enable_rerank_and_chunk_top_k_change_cache_key():
    base = _key(enable_rerank=True, chunk_top_k=20)
    assert base != _key(enable_rerank=False, chunk_top_k=20)
    assert base != _key(enable_rerank=True, chunk_top_k=5)


def test_user_prompt_and_keywords_change_cache_key():
    base = _key(user_prompt="Answer from manuals only", hl_keywords=["pump"])
    assert base != _key(user_prompt="Ignore manuals", hl_keywords=["pump"])
    assert base != _key(user_prompt="Answer from manuals only", hl_keywords=["valve"])


def test_identical_kwargs_keep_same_cache_key():
    history = [{"role": "user", "content": "hi"}]
    assert _key(conversation_history=history, only_need_context=True) == _key(
        conversation_history=history, only_need_context=True
    )


class _FakeLLMCache:
    def __init__(self):
        self.store = {}
        self.global_config = {"enable_llm_cache": True}

    async def get_by_id(self, cache_key):
        return self.store.get(cache_key)

    async def upsert(self, data):
        self.store.update(data)

    async def index_done_callback(self):
        return None


class _MultimodalQueryHost(QueryMixin):
    def __init__(self):
        self.logger = logging.getLogger("test_multimodal_query_cache_kwargs")
        self.lightrag = SimpleNamespace(llm_response_cache=_FakeLLMCache())
        self.aquery_kwargs = []

    async def _ensure_lightrag_initialized(self):
        return {"success": True}

    async def _process_multimodal_query_content(self, query, multimodal_content):
        return f"enhanced:{query}"

    async def aquery(self, query, mode="mix", system_prompt=None, **kwargs):
        self.aquery_kwargs.append(kwargs)
        history = kwargs.get("conversation_history") or []
        if history:
            return "answer-about-pump-p100"
        if kwargs.get("only_need_context"):
            return "retrieved-context-only"
        return "generic-table-answer"


@pytest.mark.asyncio
async def test_cache_does_not_reuse_answer_across_conversation_history():
    host = _MultimodalQueryHost()
    table = [{"type": "table", "table_body": "| Pump | Flow |\n| P-100 | 50 |"}]
    first = await host.aquery_with_multimodal(
        "What does this show?",
        multimodal_content=table,
        conversation_history=[],
    )
    second = await host.aquery_with_multimodal(
        "What does this show?",
        multimodal_content=table,
        conversation_history=[
            {"role": "user", "content": "Focus on pump P-100"},
            {"role": "assistant", "content": "Understood."},
        ],
    )
    assert first == "generic-table-answer"
    assert second == "answer-about-pump-p100"
    assert len(host.aquery_kwargs) == 2


@pytest.mark.asyncio
async def test_cache_does_not_reuse_answer_when_only_need_context_differs():
    host = _MultimodalQueryHost()
    table = [{"type": "table", "table_body": "| Pump | Flow |\n| P-100 | 50 |"}]
    full = await host.aquery_with_multimodal(
        "What does this show?",
        multimodal_content=table,
        only_need_context=False,
    )
    context_only = await host.aquery_with_multimodal(
        "What does this show?",
        multimodal_content=table,
        only_need_context=True,
    )
    assert full == "generic-table-answer"
    assert context_only == "retrieved-context-only"
