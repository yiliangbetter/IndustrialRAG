"""aquery must not call len() on non-string LightRAG results.

Streaming or structured answers are not always ``str``. Completing the query
callback with ``len(result)`` would raise and look like a failed query after
retrieval already succeeded.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raganything.callbacks import CallbackManager, ProcessingCallback
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


class RecordingQueryCallback(ProcessingCallback):
    def __init__(self):
        self.completes = []

    def on_query_complete(self, query, mode="", result_length=0, **kwargs):
        self.completes.append((query, mode, result_length))


class DummyQuery(QueryMixin):
    def __init__(self, lightrag, callback_manager=None):
        self.lightrag = lightrag
        self.vision_model_func = None
        self.logger = FakeLogger()
        self.callback_manager = callback_manager


class BoomLenResult:
    def __len__(self):
        raise AssertionError("len() must not be called on non-string query results")


@pytest.mark.asyncio
async def test_non_string_result_reports_zero_length_and_returns_payload():
    payload = BoomLenResult()
    lightrag = SimpleNamespace(aquery=AsyncMock(return_value=payload))
    cb = RecordingQueryCallback()
    mgr = CallbackManager()
    mgr.register(cb)
    query = DummyQuery(lightrag=lightrag, callback_manager=mgr)

    result = await query.aquery("what is the rating?", mode="mix", vlm_enhanced=False)

    assert result is payload
    assert cb.completes == [("what is the rating?", "mix", 0)]


@pytest.mark.asyncio
async def test_string_result_reports_character_length():
    lightrag = SimpleNamespace(aquery=AsyncMock(return_value="ok"))
    cb = RecordingQueryCallback()
    mgr = CallbackManager()
    mgr.register(cb)
    query = DummyQuery(lightrag=lightrag, callback_manager=mgr)

    result = await query.aquery("hello", mode="naive", vlm_enhanced=False)

    assert result == "ok"
    assert cb.completes == [("hello", "naive", 2)]
