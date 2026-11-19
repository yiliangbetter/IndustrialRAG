"""Regression tests for aquery's VLM routing switch.

aquery_vlm_enhanced is covered separately. These tests lock the decision
that happens before that method: an explicit text query must stay on
LightRAG, and a requested vision query without a vision model must still
answer instead of failing closed.
"""

import pytest

from raganything.callbacks import CallbackManager, ProcessingCallback
from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        if args:
            self.warnings.append(args[0])

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_query_start(self, query, mode="", **kwargs):
        self.events.append(("start", query, mode))

    def on_query_complete(
        self,
        query,
        mode="",
        duration_seconds=0.0,
        result_length=0,
        **kwargs,
    ):
        self.events.append(("complete", query, mode, result_length))

    def on_query_error(self, query, mode="", error=None, **kwargs):
        self.events.append(("error", query, mode, error))


class FakeLightRAG:
    def __init__(self, result="12 kN radial"):
        self.result = result
        self.calls = []

    async def aquery(self, query, param, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "mode": param.mode,
                "top_k": param.top_k,
                "only_need_prompt": param.only_need_prompt,
                "system_prompt": system_prompt,
            }
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def _vision_must_not_run(*args, **kwargs):
    raise AssertionError("vision model should not run")


def _query_mixin(lightrag, vision=_vision_must_not_run, callback=None):
    query = QueryMixin()
    query.logger = FakeLogger()
    query.lightrag = lightrag
    query.callback_manager = callback
    if vision is _MISSING:
        if hasattr(query, "vision_model_func"):
            delattr(query, "vision_model_func")
    else:
        query.vision_model_func = vision

    async def vlm_must_not_run(*args, **kwargs):
        raise AssertionError("aquery_vlm_enhanced should not run")

    query.aquery_vlm_enhanced = vlm_must_not_run
    return query


_MISSING = object()
QUESTION = "What is the bearing load?"
SYSTEM_PROMPT = "Answer from the bearing manual."


@pytest.mark.asyncio
async def test_explicit_text_query_ignores_vision_model():
    lightrag = FakeLightRAG()
    callback = RecordingCallback()
    manager = CallbackManager()
    manager.register(callback)
    query = _query_mixin(lightrag, callback=manager)

    result = await query.aquery(
        QUESTION,
        mode="hybrid",
        system_prompt=SYSTEM_PROMPT,
        vlm_enhanced=False,
        top_k=5,
    )

    assert result == "12 kN radial"
    assert lightrag.calls == [
        {
            "query": QUESTION,
            "mode": "hybrid",
            "top_k": 5,
            "only_need_prompt": False,
            "system_prompt": SYSTEM_PROMPT,
        }
    ]
    assert query.logger.warnings == []
    assert callback.events[0] == ("start", QUESTION, "hybrid")
    assert callback.events[1] == ("complete", QUESTION, "hybrid", len(result))


@pytest.mark.asyncio
@pytest.mark.parametrize("vision", [None, _MISSING])
async def test_requested_vlm_without_model_falls_back_to_text(vision):
    lightrag = FakeLightRAG()
    query = _query_mixin(lightrag, vision=vision)

    result = await query.aquery(
        QUESTION,
        mode="hybrid",
        system_prompt=SYSTEM_PROMPT,
        vlm_enhanced=True,
        top_k=5,
    )

    assert result == "12 kN radial"
    assert lightrag.calls == [
        {
            "query": QUESTION,
            "mode": "hybrid",
            "top_k": 5,
            "only_need_prompt": False,
            "system_prompt": SYSTEM_PROMPT,
        }
    ]
    assert query.logger.warnings == [
        "VLM enhanced query requested but vision_model_func is not available, "
        "falling back to normal query"
    ]


@pytest.mark.asyncio
async def test_omitted_flag_with_vision_delegates_before_text_query():
    lightrag = FakeLightRAG()
    query = _query_mixin(lightrag, vision=_vision_must_not_run)
    delegated = []

    async def fake_vlm(query_text, mode="mix", system_prompt=None, **kwargs):
        delegated.append(
            {
                "query": query_text,
                "mode": mode,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        return "vlm-answer"

    query.aquery_vlm_enhanced = fake_vlm

    result = await query.aquery(
        QUESTION,
        mode="hybrid",
        system_prompt=SYSTEM_PROMPT,
        top_k=5,
    )

    assert result == "vlm-answer"
    assert lightrag.calls == []
    assert delegated == [
        {
            "query": QUESTION,
            "mode": "hybrid",
            "system_prompt": SYSTEM_PROMPT,
            "kwargs": {"top_k": 5},
        }
    ]


@pytest.mark.asyncio
async def test_text_query_error_is_reported_and_not_completed():
    failure = RuntimeError("retriever down")
    lightrag = FakeLightRAG(result=failure)
    callback = RecordingCallback()
    manager = CallbackManager()
    manager.register(callback)
    query = _query_mixin(lightrag, vision=None, callback=manager)

    with pytest.raises(RuntimeError, match="retriever down"):
        await query.aquery(QUESTION, mode="hybrid", vlm_enhanced=False)

    assert callback.events == [
        ("start", QUESTION, "hybrid"),
        ("error", QUESTION, "hybrid", failure),
    ]
    assert lightrag.calls[0]["only_need_prompt"] is False


@pytest.mark.asyncio
async def test_non_string_answer_reports_zero_result_length():
    payload = {"answer": "bearing load 12 kN"}
    lightrag = FakeLightRAG(result=payload)
    callback = RecordingCallback()
    manager = CallbackManager()
    manager.register(callback)
    query = _query_mixin(lightrag, vision=None, callback=manager)

    result = await query.aquery(QUESTION, mode="hybrid", vlm_enhanced=False)

    assert result is payload
    assert callback.events[1] == ("complete", QUESTION, "hybrid", 0)
