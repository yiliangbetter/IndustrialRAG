"""Regression tests for QueryMixin text query routing and cache kwargs.

aquery silently routes to VLM when vision_model_func is set. Cache keys must
include retrieval knobs (temperature/top_k) or answers can be served from the
wrong cached completion. Path identity (D18) is covered by open #121 — this
file does not assert basename vs full path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class RecordingCallbackManager:
    def __init__(self):
        self.events = []

    def dispatch(self, event_name, **kwargs):
        self.events.append((event_name, kwargs))


class DummyQuery(QueryMixin):
    def __init__(self, lightrag=None, vision_model_func=None, callback_manager=None):
        self.lightrag = lightrag
        self.vision_model_func = vision_model_func
        self.logger = FakeLogger()
        self.callback_manager = callback_manager


class TestAqueryTextPath:
    @pytest.mark.asyncio
    async def test_missing_lightrag_raises(self):
        q = DummyQuery(lightrag=None)
        with pytest.raises(ValueError, match="No LightRAG instance"):
            await q.aquery("what is the spec?")

    @pytest.mark.asyncio
    async def test_vlm_enhanced_false_skips_vlm_even_when_vision_exists(self):
        lightrag = SimpleNamespace(aquery=AsyncMock(return_value="text-answer"))
        q = DummyQuery(lightrag=lightrag, vision_model_func=object())

        async def should_not_run(*args, **kwargs):
            raise AssertionError("aquery_vlm_enhanced should not be called")

        q.aquery_vlm_enhanced = should_not_run

        result = await q.aquery("hello", mode="mix", vlm_enhanced=False)
        assert result == "text-answer"
        lightrag.aquery.assert_awaited_once()
        args, kwargs = lightrag.aquery.call_args
        assert args[0] == "hello"
        assert kwargs["param"].mode == "mix"

    @pytest.mark.asyncio
    async def test_vlm_requested_without_vision_falls_back_to_text(self):
        lightrag = SimpleNamespace(aquery=AsyncMock(return_value="fallback"))
        q = DummyQuery(lightrag=lightrag, vision_model_func=None)

        result = await q.aquery("hello", mode="naive", vlm_enhanced=True)

        assert result == "fallback"
        assert any("vision_model_func is not available" in w for w in q.logger.warnings)
        lightrag.aquery.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_default_routes_to_vlm_when_vision_func_present(self):
        lightrag = SimpleNamespace(aquery=AsyncMock(return_value="should-not-use"))
        q = DummyQuery(lightrag=lightrag, vision_model_func=object())
        q.aquery_vlm_enhanced = AsyncMock(return_value="vlm-answer")

        result = await q.aquery("hello", mode="hybrid", system_prompt="sys")

        assert result == "vlm-answer"
        q.aquery_vlm_enhanced.assert_awaited_once()
        lightrag.aquery.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_error_dispatches_callback_and_reraises(self):
        boom = RuntimeError("index missing")
        lightrag = SimpleNamespace(aquery=AsyncMock(side_effect=boom))
        callbacks = RecordingCallbackManager()
        q = DummyQuery(lightrag=lightrag, callback_manager=callbacks)

        with pytest.raises(RuntimeError, match="index missing"):
            await q.aquery("hello", mode="local", vlm_enhanced=False)

        names = [name for name, _ in callbacks.events]
        assert "on_query_start" in names
        assert "on_query_error" in names
        assert "on_query_complete" not in names
        error_event = next(
            kw for name, kw in callbacks.events if name == "on_query_error"
        )
        assert error_event["error"] is boom
        assert error_event["query"] == "hello"
        assert error_event["mode"] == "local"


class TestMultimodalQueryCacheRelevantKwargs:
    def test_temperature_and_top_k_change_cache_key(self):
        q = DummyQuery()
        content = [{"type": "table", "table_body": "a|b"}]
        base = q._generate_multimodal_cache_key("q", content, "mix", temperature=0.0)
        hotter = q._generate_multimodal_cache_key("q", content, "mix", temperature=0.7)
        fewer = q._generate_multimodal_cache_key(
            "q", content, "mix", temperature=0.0, top_k=5
        )
        assert base != hotter
        assert base != fewer
        assert hotter != fewer

    def test_irrelevant_kwargs_do_not_change_cache_key(self):
        q = DummyQuery()
        content = [{"type": "equation", "latex": "E=mc^2"}]
        without = q._generate_multimodal_cache_key("q", content, "mix")
        with_noise = q._generate_multimodal_cache_key(
            "q",
            content,
            "mix",
            conversation_history=[{"role": "user", "content": "hi"}],
            extra_safe_dirs=["/tmp"],
        )
        assert without == with_noise

    def test_long_table_body_is_hashed_not_stored_verbatim(self):
        q = DummyQuery()
        long_body = "x" * 250
        key_a = q._generate_multimodal_cache_key(
            "q", [{"type": "table", "table_body": long_body}], "mix"
        )
        key_b = q._generate_multimodal_cache_key(
            "q", [{"type": "table", "table_body": "y" * 250}], "mix"
        )
        assert key_a != key_b
        assert key_a.startswith("multimodal_query:")
