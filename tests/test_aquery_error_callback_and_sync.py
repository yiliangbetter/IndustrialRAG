"""aquery must re-raise failures after on_query_error, and sync wrappers must
forward to the async methods without swallowing results.
"""

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


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_query_start(self, query, mode="", **kwargs):
        self.events.append(("start", query, mode))

    def on_query_complete(self, query, mode="", **kwargs):
        self.events.append(("complete", query, mode))

    def on_query_error(self, query, error, mode="", **kwargs):
        self.events.append(("error", query, mode, type(error).__name__, str(error)))


def _query_mixin(lightrag):
    mixin = QueryMixin.__new__(QueryMixin)
    mixin.lightrag = lightrag
    mixin.logger = FakeLogger()
    mixin.vision_model_func = None
    mixin.callback_manager = CallbackManager()
    return mixin


@pytest.mark.asyncio
async def test_aquery_dispatches_error_and_reraises():
    class FakeLightRAG:
        async def aquery(self, query, param, system_prompt=None):
            raise RuntimeError("retriever down")

    mixin = _query_mixin(FakeLightRAG())
    cb = RecordingCallback()
    mixin.callback_manager.register(cb)

    with pytest.raises(RuntimeError, match="retriever down"):
        await mixin.aquery("what failed?", mode="mix")

    kinds = [e[0] for e in cb.events]
    assert kinds == ["start", "error"]
    assert cb.events[1][3] == "RuntimeError"
    assert cb.events[1][4] == "retriever down"


def test_query_sync_wrapper_forwards_mode():
    mixin = _query_mixin(object())

    async def fake_aquery(query, mode="mix", system_prompt=None, **kwargs):
        return f"{query}:{mode}:{kwargs.get('top_k')}"

    mixin.aquery = fake_aquery
    assert mixin.query("hello", mode="naive", top_k=5) == "hello:naive:5"


def test_query_with_multimodal_sync_wrapper_forwards_payload():
    mixin = _query_mixin(object())
    seen = {}

    async def fake_aquery_with_multimodal(
        query, multimodal_content=None, mode="mix", **kwargs
    ):
        seen["query"] = query
        seen["content"] = multimodal_content
        seen["mode"] = mode
        return "mm-answer"

    mixin.aquery_with_multimodal = fake_aquery_with_multimodal
    payload = [{"type": "table", "table_data": "a|b"}]
    assert (
        mixin.query_with_multimodal("describe", payload, mode="hybrid") == "mm-answer"
    )
    assert seen == {"query": "describe", "content": payload, "mode": "hybrid"}
