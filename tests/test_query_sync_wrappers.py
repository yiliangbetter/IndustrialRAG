"""Synchronous QueryMixin wrappers must forward mode and kwargs to async methods.

Library callers use query() / query_with_multimodal() outside async context.
Dropping vlm_enhanced, system_prompt, or multimodal_content here silently
changes retrieval vs the async API.
"""

import asyncio
from unittest.mock import AsyncMock

from raganything.query import QueryMixin


class DummyQuery(QueryMixin):
    def __init__(self):
        self.aquery = AsyncMock(return_value="text-answer")
        self.aquery_with_multimodal = AsyncMock(return_value="mm-answer")


def _patch_event_loop(monkeypatch):
    class ImmediateLoop:
        def run_until_complete(self, coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    monkeypatch.setattr(
        "raganything.query.always_get_an_event_loop", lambda: ImmediateLoop()
    )


def test_query_forwards_mode_and_kwargs(monkeypatch):
    _patch_event_loop(monkeypatch)
    q = DummyQuery()

    result = q.query(
        "what is the rating?",
        mode="local",
        vlm_enhanced=False,
        system_prompt="Be brief.",
        top_k=5,
    )

    assert result == "text-answer"
    q.aquery.assert_awaited_once_with(
        "what is the rating?",
        mode="local",
        vlm_enhanced=False,
        system_prompt="Be brief.",
        top_k=5,
    )
    q.aquery_with_multimodal.assert_not_awaited()


def test_query_with_multimodal_forwards_content_and_kwargs(monkeypatch):
    _patch_event_loop(monkeypatch)
    q = DummyQuery()
    content = [{"type": "table", "table_data": "A|B\n1|2"}]

    result = q.query_with_multimodal(
        "analyze the table",
        multimodal_content=content,
        mode="hybrid",
        system_prompt="Use SI units.",
        temperature=0.0,
    )

    assert result == "mm-answer"
    q.aquery_with_multimodal.assert_awaited_once_with(
        "analyze the table",
        content,
        mode="hybrid",
        system_prompt="Use SI units.",
        temperature=0.0,
    )
    q.aquery.assert_not_awaited()
