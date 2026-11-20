"""Regression tests for query entry guards before retrieval or vision calls."""

import pytest

from raganything.query import QueryMixin


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class _Callbacks:
    def __init__(self):
        self.events = []

    def dispatch(self, event_name, **kwargs):
        self.events.append((event_name, kwargs))


def _mixin():
    mixin = QueryMixin()
    mixin.logger = _Logger()
    mixin.lightrag = None
    mixin.callback_manager = _Callbacks()
    mixin.vision_calls = []

    async def vision(*args, **kwargs):
        mixin.vision_calls.append((args, kwargs))
        return "vision-answer"

    mixin.vision_model_func = vision
    return mixin


@pytest.mark.asyncio
@pytest.mark.parametrize("vlm_enhanced", [None, True])
async def test_aquery_without_lightrag_stops_before_vision(vlm_enhanced):
    mixin = _mixin()
    kwargs = {}
    if vlm_enhanced is not None:
        kwargs["vlm_enhanced"] = vlm_enhanced

    with pytest.raises(ValueError, match="No LightRAG instance available"):
        await mixin.aquery(
            "What is the bearing load?",
            mode="hybrid",
            system_prompt="Answer from the bearing manual.",
            **kwargs,
        )

    assert mixin.vision_calls == []
    assert mixin.callback_manager.events == []


def test_sync_query_without_lightrag_propagates_guard():
    mixin = _mixin()

    with pytest.raises(ValueError, match="No LightRAG instance available"):
        mixin.query(
            "What is the bearing load?",
            mode="hybrid",
            system_prompt="Answer from the bearing manual.",
        )

    assert mixin.vision_calls == []
    assert mixin.callback_manager.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("init_result", "message"),
    [
        (None, "unknown error"),
        ({"success": False, "error": "index down"}, "index down"),
    ],
)
async def test_multimodal_query_init_failure_skips_description(init_result, message):
    mixin = _mixin()
    mixin.lightrag = object()
    calls = []

    async def init():
        calls.append("init")
        return init_result

    async def aquery(*args, **kwargs):
        calls.append("aquery")
        return "should-not-run"

    async def describe(query, multimodal_content):
        calls.append(("describe", query, multimodal_content))
        return query

    mixin._ensure_lightrag_initialized = init
    mixin.aquery = aquery
    mixin._process_multimodal_query_content = describe

    with pytest.raises(RuntimeError, match=message):
        await mixin.aquery_with_multimodal(
            "What is the bearing load?",
            multimodal_content=[{"type": "table", "table_body": "bearing load 12 kN"}],
            mode="hybrid",
            system_prompt="Answer from the bearing manual.",
        )

    assert calls == ["init"]
