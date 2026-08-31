"""Pre-provided LightRAG must inherit model funcs and finish lazy init.

Callers often construct RAGAnything(lightrag=existing). If llm/embedding
funcs are not copied, or parse cache / processors never start, ingest and
query look initialized but write nowhere. Init failures on that path must
fail closed instead of returning a half-wired instance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class StubParser:
    def check_installation(self):
        return True


def _make_rag(monkeypatch, tmp_path, **kwargs):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **k: None)
    monkeypatch.setattr(rag_module.atexit, "unregister", lambda *args, **k: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
    )
    rag = rag_module.RAGAnything(config=config, **kwargs)
    rag.doc_parser = StubParser()
    rag._parser_installation_checked = True
    return rag, rag_module


class FakeParseCache:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initialized = False

    async def initialize(self):
        self.initialized = True


@pytest.mark.asyncio
async def test_inherits_llm_and_embedding_from_preprovided_lightrag(
    monkeypatch, tmp_path
):
    rag, _ = _make_rag(monkeypatch, tmp_path)

    llm = object()
    embed = object()
    processor_calls = {"n": 0}

    def fake_init_processors():
        processor_calls["n"] += 1
        rag.modal_processors = {"generic": object()}

    rag._initialize_processors = fake_init_processors
    rag.llm_model_func = None
    rag.embedding_func = None
    rag.parse_cache = None
    rag.modal_processors = {}
    rag.lightrag = SimpleNamespace(
        _storages_status=SimpleNamespace(name="INITIALIZED"),
        llm_model_func=llm,
        embedding_func=embed,
        key_string_value_json_storage_cls=FakeParseCache,
        workspace="ws-1",
        tokenizer=object(),
        initialize_storages=AsyncMock(),
    )

    result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert rag.llm_model_func is llm
    assert rag.embedding_func is embed
    assert isinstance(rag.parse_cache, FakeParseCache)
    assert rag.parse_cache.initialized is True
    assert rag.parse_cache.kwargs["namespace"] == "parse_cache"
    assert rag.parse_cache.kwargs["workspace"] == "ws-1"
    assert rag.parse_cache.kwargs["embedding_func"] is embed
    assert processor_calls["n"] == 1
    rag.lightrag.initialize_storages.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_overwrite_explicit_model_funcs(monkeypatch, tmp_path):
    own_llm = object()
    own_embed = object()
    rag, _ = _make_rag(
        monkeypatch,
        tmp_path,
        llm_model_func=own_llm,
        embedding_func=own_embed,
    )

    rag._initialize_processors = lambda: None
    rag.parse_cache = object()
    rag.modal_processors = {"already": True}
    rag.lightrag = SimpleNamespace(
        _storages_status=SimpleNamespace(name="INITIALIZED"),
        llm_model_func=object(),
        embedding_func=object(),
        initialize_storages=AsyncMock(),
    )

    result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert rag.llm_model_func is own_llm
    assert rag.embedding_func is own_embed


@pytest.mark.asyncio
async def test_uninitialized_storages_are_initialized(monkeypatch, tmp_path):
    rag, _ = _make_rag(monkeypatch, tmp_path)

    pipeline_calls = {"n": 0}

    async def fake_pipeline_status():
        pipeline_calls["n"] += 1

    monkeypatch.setattr(
        "lightrag.kg.shared_storage.initialize_pipeline_status",
        fake_pipeline_status,
    )

    rag._initialize_processors = lambda: None
    rag.parse_cache = object()
    rag.modal_processors = {"already": True}
    rag.lightrag = SimpleNamespace(
        _storages_status=SimpleNamespace(name="CREATED"),
        llm_model_func=object(),
        embedding_func=object(),
        initialize_storages=AsyncMock(),
    )

    result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    rag.lightrag.initialize_storages.assert_awaited_once()
    assert pipeline_calls["n"] == 1


@pytest.mark.asyncio
async def test_preprovided_init_failure_is_fail_closed(monkeypatch, tmp_path):
    rag, _ = _make_rag(monkeypatch, tmp_path)

    processor_calls = {"n": 0}

    def fake_init_processors():
        processor_calls["n"] += 1

    rag._initialize_processors = fake_init_processors
    rag.parse_cache = None
    rag.modal_processors = {}
    rag.lightrag = SimpleNamespace(
        llm_model_func=object(),
        embedding_func=object(),
        initialize_storages=AsyncMock(side_effect=RuntimeError("disk full")),
    )

    result = await rag._ensure_lightrag_initialized()

    assert result["success"] is False
    assert "disk full" in result["error"]
    assert rag.parse_cache is None
    assert processor_calls["n"] == 0
    assert rag.modal_processors == {}
