"""Regression tests for query error callbacks and sync query wrappers.

test_callbacks.py covers on_query_start/complete happy paths. Failures must
still emit on_query_error (then re-raise) so MetricsCallback / ops hooks see
query failures. Sync wrappers are the common non-async entrypoints and must
forward kwargs into the async implementations.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lightrag")

from raganything.callbacks import CallbackManager, MetricsCallback, ProcessingCallback
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
        self.events = []

    def on_query_start(self, query, mode="", **kw):
        self.events.append(("start", query, mode))

    def on_query_complete(self, query, mode="", **kw):
        self.events.append(("complete", query, mode))

    def on_query_error(self, query, error="", mode="", **kw):
        self.events.append(("error", query, mode, error))


class FailingLightRAG:
    async def aquery(self, query, param, system_prompt=None):
        raise RuntimeError("retrieval backend down")


class OkLightRAG:
    def __init__(self):
        self.calls = []

    async def aquery(self, query, param, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "mode": param.mode,
                "system_prompt": system_prompt,
            }
        )
        return "ok-answer"


class DummyQuery(QueryMixin):
    def __init__(self, lightrag=None):
        self.lightrag = lightrag if lightrag is not None else OkLightRAG()
        self.logger = FakeLogger()
        self.vision_model_func = None
        self.callback_manager = CallbackManager()
        self.mm_calls = []

    async def aquery_with_multimodal(
        self,
        query,
        multimodal_content=None,
        mode="mix",
        system_prompt=None,
        **kwargs,
    ):
        self.mm_calls.append(
            {
                "query": query,
                "multimodal_content": multimodal_content,
                "mode": mode,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        return "mm-answer"


@pytest.mark.asyncio
async def test_aquery_dispatches_on_query_error_then_reraises():
    cb = RecordingQueryCallback()
    metrics = MetricsCallback()
    query = DummyQuery(lightrag=FailingLightRAG())
    query.callback_manager.register(cb)
    query.callback_manager.register(metrics)

    with pytest.raises(RuntimeError, match="retrieval backend down"):
        await query.aquery("hello", mode="mix", vlm_enhanced=False)

    assert ("start", "hello", "mix") in cb.events
    error_events = [e for e in cb.events if e[0] == "error"]
    assert len(error_events) == 1
    assert error_events[0][1] == "hello"
    assert error_events[0][2] == "mix"
    assert isinstance(error_events[0][3], RuntimeError)
    assert str(error_events[0][3]) == "retrieval backend down"
    assert not any(e[0] == "complete" for e in cb.events)

    assert metrics.metrics["queries_executed"] == 0
    assert len(metrics.metrics["errors"]) == 1
    assert metrics.metrics["errors"][0]["stage"] == "query"
    assert "retrieval backend down" in metrics.metrics["errors"][0]["error"]


def test_sync_query_forwards_to_aquery():
    query = DummyQuery()

    result = query.query("sync question", mode="hybrid", vlm_enhanced=False)

    assert result == "ok-answer"
    assert query.lightrag.calls == [
        {
            "query": "sync question",
            "mode": "hybrid",
            "system_prompt": None,
        }
    ]


def test_sync_query_with_multimodal_forwards_to_async():
    query = DummyQuery()
    multimodal = [{"type": "table", "table_data": "a|b"}]

    result = query.query_with_multimodal(
        "compare",
        multimodal_content=multimodal,
        mode="local",
        top_k=5,
    )

    assert result == "mm-answer"
    assert query.mm_calls == [
        {
            "query": "compare",
            "multimodal_content": multimodal,
            "mode": "local",
            "system_prompt": None,
            "kwargs": {"top_k": 5},
        }
    ]
