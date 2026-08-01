"""Regression tests for aquery VLM auto-routing.

aquery chooses between normal LightRAG query and aquery_vlm_enhanced based on
vlm_enhanced (explicit or auto from vision_model_func). Wrong routing either
skips figure-aware answers when a vision model exists, or crashes/warns when
VLM is forced without one.
"""

import pytest

pytest.importorskip("lightrag")

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeLightRAG:
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
        return "text-answer"


class DummyQuery(QueryMixin):
    def __init__(self, vision_model_func=None):
        self.lightrag = FakeLightRAG()
        self.logger = FakeLogger()
        self.vision_model_func = vision_model_func
        self.vlm_calls = []

    async def aquery_vlm_enhanced(self, query, mode="mix", system_prompt=None, **kwargs):
        self.vlm_calls.append(
            {
                "query": query,
                "mode": mode,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        return "vlm-answer"


@pytest.mark.asyncio
async def test_aquery_auto_routes_to_vlm_when_vision_available():
    dummy = DummyQuery(vision_model_func=lambda *a, **k: "vision")

    result = await dummy.aquery("what is in the figure?", mode="mix")

    assert result == "vlm-answer"
    assert dummy.vlm_calls == [
        {
            "query": "what is in the figure?",
            "mode": "mix",
            "system_prompt": None,
            "kwargs": {},
        }
    ]
    assert dummy.lightrag.calls == []


@pytest.mark.asyncio
async def test_aquery_auto_stays_text_when_vision_missing():
    dummy = DummyQuery(vision_model_func=None)

    result = await dummy.aquery("summarize", mode="hybrid", system_prompt="Be brief.")

    assert result == "text-answer"
    assert dummy.vlm_calls == []
    assert dummy.lightrag.calls == [
        {
            "query": "summarize",
            "mode": "hybrid",
            "system_prompt": "Be brief.",
        }
    ]


@pytest.mark.asyncio
async def test_aquery_explicit_false_skips_vlm_even_with_vision():
    dummy = DummyQuery(vision_model_func=lambda *a, **k: "vision")

    result = await dummy.aquery("summarize", mode="mix", vlm_enhanced=False)

    assert result == "text-answer"
    assert dummy.vlm_calls == []
    assert len(dummy.lightrag.calls) == 1


@pytest.mark.asyncio
async def test_aquery_explicit_true_without_vision_falls_back_with_warning():
    dummy = DummyQuery(vision_model_func=None)

    result = await dummy.aquery("summarize", mode="mix", vlm_enhanced=True)

    assert result == "text-answer"
    assert dummy.vlm_calls == []
    assert len(dummy.lightrag.calls) == 1
    assert any("vision_model_func is not available" in w for w in dummy.logger.warnings)


@pytest.mark.asyncio
async def test_aquery_requires_lightrag():
    dummy = DummyQuery()
    dummy.lightrag = None

    with pytest.raises(ValueError, match="No LightRAG instance available"):
        await dummy.aquery("hello")
