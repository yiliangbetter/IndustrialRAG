"""LightRAG init: inherit model funcs from a pre-provided instance and merge kwargs.

A regression here silently drops custom storage/workspace settings or leaves
processors without the LLM/embedding functions that the caller already
configured on LightRAG.
"""

from types import SimpleNamespace

import pytest

from raganything.raganything import RAGAnything


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _bind_ensure(target):
    bound = RAGAnything._ensure_lightrag_initialized.__get__(target, type(target))
    target._ensure_lightrag_initialized = bound
    target.logger = FakeLogger()
    return target


@pytest.mark.asyncio
async def test_inherits_model_funcs_from_preprovided_lightrag():
    async def llm_fn(prompt, **kwargs):
        return "llm"

    def embed_fn(texts):
        return texts

    holder = SimpleNamespace()
    holder.config = SimpleNamespace(
        allow_embedding_only_ingestion=False, parser="mineru"
    )
    holder._parser_installation_checked = True
    holder.llm_model_func = None
    holder.embedding_func = None
    holder.parse_cache = object()
    holder.modal_processors = {"generic": object()}
    holder.lightrag = SimpleNamespace(
        llm_model_func=llm_fn,
        embedding_func=embed_fn,
        _storages_status=SimpleNamespace(name="INITIALIZED"),
    )
    _bind_ensure(holder)

    result = await holder._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert holder.llm_model_func is llm_fn
    assert holder.embedding_func is embed_fn


@pytest.mark.asyncio
async def test_lightrag_kwargs_merge_and_override_working_dir(monkeypatch):
    captured = {}

    class FakeParseCache:
        async def initialize(self):
            return None

    class FakeLightRAG:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.workspace = kwargs.get("workspace", "default")

            def _make_parse_cache(**_kw):
                return FakeParseCache()

            self.key_string_value_json_storage_cls = _make_parse_cache

        async def initialize_storages(self):
            return None

    async def fake_pipeline_status():
        return None

    async def llm_fn(prompt, **kwargs):
        return ""

    def embed_fn(texts):
        return texts

    holder = SimpleNamespace()
    holder.config = SimpleNamespace(
        allow_embedding_only_ingestion=False, parser="mineru"
    )
    holder._parser_installation_checked = True
    holder.lightrag = None
    holder.llm_model_func = llm_fn
    holder.embedding_func = embed_fn
    holder.parse_cache = None
    holder.modal_processors = {}
    holder.working_dir = "/tmp/rag-default-wd"
    holder.lightrag_kwargs = {
        "workspace": "plant-manuals",
        "chunk_token_size": 777,
        "working_dir": "/tmp/rag-override-wd",
    }
    holder._initialize_processors = lambda: None
    _bind_ensure(holder)

    import lightrag.kg.shared_storage as shared_storage

    monkeypatch.setattr("raganything.raganything.LightRAG", FakeLightRAG)
    monkeypatch.setattr(
        shared_storage, "initialize_pipeline_status", fake_pipeline_status
    )

    result = await holder._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert captured["workspace"] == "plant-manuals"
    assert captured["chunk_token_size"] == 777
    assert captured["working_dir"] == "/tmp/rag-override-wd"
    assert captured["llm_model_func"] is llm_fn
    assert captured["embedding_func"] is embed_fn
    assert holder.parse_cache is not None
