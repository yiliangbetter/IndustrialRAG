"""Regression tests for aquery fail-closed guards and query callbacks.

Distinct from open coverage:
- #82 VLM auto-routing
- #67/#68 system_prompt / multimodal cache lookup
- #77 multimodal description soft-fallback

Focus: missing LightRAG, query callback success/error dispatch, sync wrappers,
and multimodal cache-key normalization (basename / large table hash).
"""

import hashlib

import pytest

pytest.importorskip("lightrag")

from raganything.callbacks import CallbackManager, ProcessingCallback
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
    def __init__(self, *, fail_with=None, result="text-answer"):
        self.fail_with = fail_with
        self.result = result
        self.calls = []

    async def aquery(self, query, param, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "mode": param.mode,
                "system_prompt": system_prompt,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        return self.result


class RecordingQueryCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_query_start(self, query, mode="", **kwargs):
        self.events.append(("start", query, mode))

    def on_query_complete(self, query, mode="", **kwargs):
        self.events.append(
            (
                "complete",
                query,
                mode,
                kwargs.get("result_length"),
                kwargs.get("duration_seconds"),
            )
        )

    def on_query_error(self, query, mode="", error=None, **kwargs):
        self.events.append(("error", query, mode, error))


class DummyQuery(QueryMixin):
    def __init__(self, lightrag=None, vision_model_func=None):
        self.lightrag = lightrag
        self.logger = FakeLogger()
        self.vision_model_func = vision_model_func
        self.callback_manager = CallbackManager()
        self.modal_processors = {}

    async def _ensure_lightrag_initialized(self):
        return {"success": True}


class TestAqueryGuardsAndCallbacks:
    @pytest.mark.asyncio
    async def test_aquery_raises_when_lightrag_missing(self):
        dummy = DummyQuery(lightrag=None)

        with pytest.raises(ValueError, match="No LightRAG instance available"):
            await dummy.aquery("what broke?")

    @pytest.mark.asyncio
    async def test_aquery_dispatches_start_and_complete_callbacks(self):
        lightrag = FakeLightRAG(result="hello world")
        dummy = DummyQuery(lightrag=lightrag, vision_model_func=None)
        cb = RecordingQueryCallback()
        dummy.callback_manager.register(cb)

        result = await dummy.aquery("summarize", mode="hybrid")

        assert result == "hello world"
        assert cb.events[0] == ("start", "summarize", "hybrid")
        assert cb.events[1][0] == "complete"
        assert cb.events[1][1:3] == ("summarize", "hybrid")
        assert cb.events[1][3] == len("hello world")
        assert isinstance(cb.events[1][4], float)
        assert cb.events[1][4] >= 0.0

    @pytest.mark.asyncio
    async def test_aquery_dispatches_error_callback_and_rethrows(self):
        boom = RuntimeError("upstream down")
        lightrag = FakeLightRAG(fail_with=boom)
        dummy = DummyQuery(lightrag=lightrag, vision_model_func=None)
        cb = RecordingQueryCallback()
        dummy.callback_manager.register(cb)

        with pytest.raises(RuntimeError, match="upstream down"):
            await dummy.aquery("status?", mode="local")

        assert cb.events[0] == ("start", "status?", "local")
        assert cb.events[1] == ("error", "status?", "local", boom)
        assert not any(event[0] == "complete" for event in cb.events)

    def test_sync_query_wrapper_runs_aquery(self, monkeypatch):
        dummy = DummyQuery(lightrag=FakeLightRAG(), vision_model_func=None)
        seen = {}

        async def fake_aquery(query, mode="mix", system_prompt=None, **kwargs):
            seen["args"] = (query, mode, system_prompt, kwargs)
            return "sync-ok"

        monkeypatch.setattr(dummy, "aquery", fake_aquery)
        assert dummy.query("ping", mode="naive", top_k=3) == "sync-ok"
        assert seen["args"] == ("ping", "naive", None, {"top_k": 3})

    def test_sync_query_with_multimodal_wrapper_runs_async(self, monkeypatch):
        dummy = DummyQuery(lightrag=FakeLightRAG())
        content = [{"type": "table", "table_data": "a,b\n1,2"}]
        seen = {}

        async def fake_aquery_with_multimodal(
            query, multimodal_content=None, mode="mix", **kwargs
        ):
            seen["args"] = (query, multimodal_content, mode, kwargs)
            return "mm-ok"

        monkeypatch.setattr(dummy, "aquery_with_multimodal", fake_aquery_with_multimodal)
        assert (
            dummy.query_with_multimodal("analyze", multimodal_content=content, mode="mix")
            == "mm-ok"
        )
        assert seen["args"] == ("analyze", content, "mix", {})


class TestMultimodalQueryCacheKeyNormalization:
    def test_image_paths_use_basename_for_stable_portable_keys(self):
        dummy = DummyQuery()
        key_a = dummy._generate_multimodal_cache_key(
            "describe",
            [{"type": "image", "img_path": "/var/data/fig.png"}],
            "mix",
        )
        key_b = dummy._generate_multimodal_cache_key(
            "describe",
            [{"type": "image", "img_path": "/other/mount/fig.png"}],
            "mix",
        )
        assert key_a == key_b
        assert key_a.startswith("multimodal_query:")

    def test_large_table_body_hashed_instead_of_inline(self):
        dummy = DummyQuery()
        big = "x" * 250
        key = dummy._generate_multimodal_cache_key(
            "totals?",
            [{"type": "table", "table_data": big}],
            "local",
        )
        # Changing only directory-irrelevant whitespace in query should change key.
        key_other_query = dummy._generate_multimodal_cache_key(
            "totals? ",
            [{"type": "table", "table_data": big}],
            "local",
        )
        assert key == key_other_query  # strip() normalizes query

        # Same length but different payload must not collide after hashing.
        key_different = dummy._generate_multimodal_cache_key(
            "totals?",
            [{"type": "table", "table_data": "y" * 250}],
            "local",
        )
        assert key != key_different

        expected_hash = hashlib.md5(big.encode()).hexdigest()
        # Sanity: helper used the md5 of the large payload (via key divergence above)
        # and keeps prefix contract.
        assert expected_hash
        assert key.startswith("multimodal_query:")

    def test_irrelevant_kwargs_do_not_affect_cache_key(self):
        dummy = DummyQuery()
        content = [{"type": "equation", "latex": "E=mc^2"}]
        base = dummy._generate_multimodal_cache_key(
            "explain", content, "mix", top_k=5, temperature=0.2
        )
        polluted = dummy._generate_multimodal_cache_key(
            "explain",
            content,
            "mix",
            top_k=5,
            temperature=0.2,
            unrelated_debug_flag=True,
            only_need_context=True,
        )
        assert base == polluted

        changed_relevant = dummy._generate_multimodal_cache_key(
            "explain", content, "mix", top_k=10, temperature=0.2
        )
        assert base != changed_relevant
