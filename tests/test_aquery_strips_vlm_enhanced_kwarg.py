"""aquery must not forward vlm_enhanced into QueryParam or VLM kwargs.

vlm_enhanced is a RAGAnything routing flag. If it leaks into LightRAG's
QueryParam it can TypeError the query or silently change retrieval. Distinct
from #125 (routing / fallback / callbacks) which does not inspect constructor
kwargs after the pop.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

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


class DummyQuery(QueryMixin):
    def __init__(self, lightrag=None, vision_model_func=None):
        self.lightrag = lightrag
        self.vision_model_func = vision_model_func
        self.logger = FakeLogger()
        self.callback_manager = None


class TestAqueryStripsVlmEnhanced:
    @pytest.mark.asyncio
    async def test_text_path_does_not_put_vlm_enhanced_on_query_param(
        self, monkeypatch
    ):
        captured = {}

        def fake_query_param(*, mode, **kwargs):
            captured["mode"] = mode
            captured["kwargs"] = dict(kwargs)
            return SimpleNamespace(mode=mode)

        monkeypatch.setattr("raganything.query.QueryParam", fake_query_param)

        lightrag = SimpleNamespace(aquery=AsyncMock(return_value="text-answer"))
        q = DummyQuery(lightrag=lightrag, vision_model_func=object())

        async def should_not_run(*args, **kwargs):
            raise AssertionError("aquery_vlm_enhanced should not be called")

        q.aquery_vlm_enhanced = should_not_run

        result = await q.aquery(
            "hello", mode="mix", vlm_enhanced=False, top_k=5, temperature=0.1
        )

        assert result == "text-answer"
        assert captured["mode"] == "mix"
        assert "vlm_enhanced" not in captured["kwargs"]
        assert captured["kwargs"]["top_k"] == 5
        assert captured["kwargs"]["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_explicit_vlm_true_does_not_forward_flag_to_vlm_path(self):
        lightrag = SimpleNamespace(aquery=AsyncMock(return_value="should-not-use"))
        q = DummyQuery(lightrag=lightrag, vision_model_func=object())
        q.aquery_vlm_enhanced = AsyncMock(return_value="vlm-answer")

        result = await q.aquery(
            "hello", mode="hybrid", vlm_enhanced=True, top_k=8, system_prompt="sys"
        )

        assert result == "vlm-answer"
        lightrag.aquery.assert_not_called()
        kwargs = q.aquery_vlm_enhanced.await_args.kwargs
        assert "vlm_enhanced" not in kwargs
        assert kwargs["mode"] == "hybrid"
        assert kwargs["top_k"] == 8
        assert kwargs["system_prompt"] == "sys"

    @pytest.mark.asyncio
    async def test_default_vision_route_does_not_invent_vlm_enhanced_kwarg(self):
        lightrag = SimpleNamespace(aquery=AsyncMock(return_value="should-not-use"))
        q = DummyQuery(lightrag=lightrag, vision_model_func=object())
        q.aquery_vlm_enhanced = AsyncMock(return_value="vlm-answer")

        result = await q.aquery("hello", mode="local")

        assert result == "vlm-answer"
        kwargs = q.aquery_vlm_enhanced.await_args.kwargs
        assert "vlm_enhanced" not in kwargs
        assert kwargs["mode"] == "local"
