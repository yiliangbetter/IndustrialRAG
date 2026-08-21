"""Tests for embedding-only LightRAG initialization gates.

Embedding-only ingest must skip parser installation and tolerate a missing
LLM function via a placeholder, while still failing closed when embeddings
or a required LLM are absent.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class StubParser:
    def __init__(self, installed=True):
        self.installed = installed
        self.checks = 0

    def check_installation(self):
        self.checks += 1
        return self.installed


def _make_rag(monkeypatch, tmp_path, *, embedding_only, parser=None, **kwargs):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    stub = parser or StubParser(installed=True)

    monkeypatch.setattr(rag_module, "get_parser", lambda name: stub)
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **k: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
        allow_embedding_only_ingestion=embedding_only,
    )
    rag = rag_module.RAGAnything(config=config, **kwargs)
    rag.doc_parser = stub
    return rag, stub, rag_module


def test_embedding_only_disables_multimodal_processor_flags(monkeypatch, tmp_path):
    rag, _, _ = _make_rag(monkeypatch, tmp_path, embedding_only=True)
    assert rag.config.enable_image_processing is False
    assert rag.config.enable_table_processing is False
    assert rag.config.enable_equation_processing is False


@pytest.mark.asyncio
async def test_embedding_only_skips_failed_parser_installation(
    monkeypatch, tmp_path
):
    parser = StubParser(installed=False)
    rag, stub, _ = _make_rag(
        monkeypatch, tmp_path, embedding_only=True, parser=parser
    )

    class FakeStatus:
        name = "INITIALIZED"

    rag.lightrag = SimpleNamespace(
        _storages_status=FakeStatus(),
        llm_model_func=None,
        embedding_func=MagicMock(),
    )
    rag.parse_cache = object()
    rag.modal_processors = {"already": True}

    result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert stub.checks == 0
    assert rag._parser_installation_checked is True


@pytest.mark.asyncio
async def test_parser_failure_is_fail_closed_without_embedding_only(
    monkeypatch, tmp_path
):
    parser = StubParser(installed=False)
    rag, stub, _ = _make_rag(
        monkeypatch, tmp_path, embedding_only=False, parser=parser
    )

    result = await rag._ensure_lightrag_initialized()

    assert result["success"] is False
    assert "not properly installed" in result["error"]
    assert stub.checks == 1
    assert rag._parser_installation_checked is False


@pytest.mark.asyncio
async def test_missing_llm_fails_closed_without_embedding_only(
    monkeypatch, tmp_path
):
    parser = StubParser(installed=True)
    rag, _, _ = _make_rag(
        monkeypatch, tmp_path, embedding_only=False, parser=parser
    )
    rag.lightrag = None
    rag.llm_model_func = None
    rag.embedding_func = MagicMock()
    rag._parser_installation_checked = True

    result = await rag._ensure_lightrag_initialized()

    assert result["success"] is False
    assert "llm_model_func must be provided" in result["error"]


@pytest.mark.asyncio
async def test_missing_embedding_func_fails_closed_even_when_embedding_only(
    monkeypatch, tmp_path
):
    rag, _, _ = _make_rag(monkeypatch, tmp_path, embedding_only=True)
    rag.lightrag = None
    rag.llm_model_func = None
    rag.embedding_func = None
    rag._parser_installation_checked = True

    result = await rag._ensure_lightrag_initialized()

    assert result["success"] is False
    assert "embedding_func must be provided" in result["error"]


@pytest.mark.asyncio
async def test_embedding_only_installs_llm_placeholder_and_creates_lightrag(
    monkeypatch, tmp_path
):
    rag, _, rag_module = _make_rag(monkeypatch, tmp_path, embedding_only=True)
    rag.lightrag = None
    rag.llm_model_func = None
    rag.embedding_func = MagicMock(name="embed")
    rag.parse_cache = None
    rag.modal_processors = {}
    rag.lightrag_kwargs = {}
    rag._parser_installation_checked = True

    created = {}

    class FakeParseCache:
        async def initialize(self):
            created["cache_init"] = True

    class FakeLightRAG:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            self.workspace = "ws"
            self.__dict__.update(kwargs)
            self.key_string_value_json_storage_cls = (
                lambda **kw: FakeParseCache()
            )
            self.initialize_storages = AsyncMock()

    async def fake_pipeline_status():
        created["pipeline"] = True

    monkeypatch.setattr(rag_module, "LightRAG", FakeLightRAG)
    monkeypatch.setattr(rag, "_initialize_processors", lambda: None)

    import lightrag.kg.shared_storage as shared_storage

    monkeypatch.setattr(
        shared_storage, "initialize_pipeline_status", fake_pipeline_status
    )

    result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert rag.llm_model_func is not None
    assert await rag.llm_model_func("prompt") == ""
    assert created.get("pipeline") is True
    assert created["kwargs"]["embedding_func"] is rag.embedding_func
    assert created["kwargs"]["llm_model_func"] is rag.llm_model_func
