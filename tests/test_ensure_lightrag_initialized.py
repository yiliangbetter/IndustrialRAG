"""Regression tests for RAGAnything._ensure_lightrag_initialized.

Covers parser-gate failures, required-model validation, pre-provided LightRAG
bootstrap (inherit funcs / storage / parse cache / processors), and create-new
LightRAG wiring — core ingest/query entry paths with large blast radius.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_rag(monkeypatch, tmp_path, parser=None, **config_overrides):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    if parser is None:

        class StubParser:
            def check_installation(self):
                return True

        parser = StubParser()

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: parser)
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    defaults = {
        "working_dir": str(tmp_path / "rag_workdir"),
        "parser": "mineru",
        "allow_embedding_only_ingestion": False,
    }
    defaults.update(config_overrides)
    config = RAGAnythingConfig(**defaults)
    return rag_module.RAGAnything(config=config), rag_module


class _InitializedStatus:
    name = "INITIALIZED"


class _UninitializedStatus:
    name = "CREATED"


class TestCheckParserInstallation:
    def test_delegates_to_doc_parser(self, monkeypatch, tmp_path):
        class StubParser:
            def __init__(self):
                self.calls = 0

            def check_installation(self):
                self.calls += 1
                return False

        parser = StubParser()
        rag, _ = _make_rag(monkeypatch, tmp_path, parser=parser)
        assert rag.check_parser_installation() is False
        assert parser.calls == 1


class TestEnsureLightragInitializedParserGate:
    @pytest.mark.asyncio
    async def test_parser_install_failure_returns_error(self, monkeypatch, tmp_path):
        class BrokenParser:
            def check_installation(self):
                return False

        rag, _ = _make_rag(monkeypatch, tmp_path, parser=BrokenParser())
        result = await rag._ensure_lightrag_initialized()

        assert result["success"] is False
        assert "not properly installed" in result["error"]
        assert rag._parser_installation_checked is False
        assert rag.lightrag is None

    @pytest.mark.asyncio
    async def test_parser_install_success_caches_check(self, monkeypatch, tmp_path):
        class CountingParser:
            def __init__(self):
                self.calls = 0

            def check_installation(self):
                self.calls += 1
                return True

        parser = CountingParser()
        rag, rag_module = _make_rag(monkeypatch, tmp_path, parser=parser)

        # Fail create-path validation after the parser gate so we can assert caching.
        result = await rag._ensure_lightrag_initialized()
        assert result["success"] is False
        assert "llm_model_func" in result["error"]
        assert parser.calls == 1
        assert rag._parser_installation_checked is True

        # Second call must not re-check installation.
        result2 = await rag._ensure_lightrag_initialized()
        assert result2["success"] is False
        assert parser.calls == 1


class TestEnsureLightragInitializedCreatePath:
    @pytest.mark.asyncio
    async def test_missing_llm_returns_error_when_not_embedding_only(
        self, monkeypatch, tmp_path
    ):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.embedding_func = object()
        result = await rag._ensure_lightrag_initialized()
        assert result == {
            "success": False,
            "error": "llm_model_func must be provided when LightRAG is not pre-initialized",
        }

    @pytest.mark.asyncio
    async def test_missing_embedding_returns_error(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.llm_model_func = object()
        result = await rag._ensure_lightrag_initialized()
        assert result == {
            "success": False,
            "error": "embedding_func must be provided when LightRAG is not pre-initialized",
        }

    @pytest.mark.asyncio
    async def test_create_new_lightrag_initializes_cache_and_processors(
        self, monkeypatch, tmp_path
    ):
        rag, rag_module = _make_rag(monkeypatch, tmp_path)

        class StubParseCache:
            def __init__(self):
                self.initialized = False

            async def initialize(self):
                self.initialized = True

        captured = {}
        parse_cache = StubParseCache()

        class StubLightRAG:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.workspace = "ws"
                self._storages_status = _UninitializedStatus()

            async def initialize_storages(self):
                captured["storages_initialized"] = True

            def key_string_value_json_storage_cls(self, **kwargs):
                captured["parse_cache_kwargs"] = kwargs
                return parse_cache

        async def fake_pipeline_status():
            captured["pipeline_status"] = True

        import lightrag.kg.shared_storage as shared_storage

        monkeypatch.setattr(rag_module, "LightRAG", StubLightRAG)
        monkeypatch.setattr(
            shared_storage, "initialize_pipeline_status", fake_pipeline_status
        )

        processors_called = {"n": 0}

        def fake_init_processors():
            processors_called["n"] += 1
            rag.modal_processors = {"generic": object()}

        monkeypatch.setattr(rag, "_initialize_processors", fake_init_processors)

        llm = object()
        emb = object()
        rag.llm_model_func = llm
        rag.embedding_func = emb
        rag.lightrag_kwargs = {"top_k": 7}

        result = await rag._ensure_lightrag_initialized()

        assert result == {"success": True}
        assert isinstance(rag.lightrag, StubLightRAG)
        assert captured["llm_model_func"] is llm
        assert captured["embedding_func"] is emb
        assert captured["working_dir"] == rag.working_dir
        assert captured["top_k"] == 7
        assert captured["storages_initialized"] is True
        assert captured["pipeline_status"] is True
        assert parse_cache.initialized is True
        assert rag.parse_cache is parse_cache
        assert processors_called["n"] == 1
        assert captured["parse_cache_kwargs"]["namespace"] == "parse_cache"
        assert captured["parse_cache_kwargs"]["embedding_func"] is emb

    @pytest.mark.asyncio
    async def test_create_new_lightrag_construction_failure(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(monkeypatch, tmp_path)

        class BoomLightRAG:
            def __init__(self, **kwargs):
                raise RuntimeError("boom-create")

        monkeypatch.setattr(rag_module, "LightRAG", BoomLightRAG)
        rag.llm_model_func = object()
        rag.embedding_func = object()

        result = await rag._ensure_lightrag_initialized()
        assert result["success"] is False
        assert "Failed to initialize LightRAG instance" in result["error"]
        assert "boom-create" in result["error"]


class TestEnsureLightragInitializedPreProvided:
    @pytest.mark.asyncio
    async def test_inherits_model_funcs_and_inits_when_not_ready(
        self, monkeypatch, tmp_path
    ):
        rag, _ = _make_rag(monkeypatch, tmp_path)

        class StubParseCache:
            async def initialize(self):
                self.ready = True

        parse_cache = StubParseCache()
        llm = object()
        emb = object()
        init_calls = {"storages": 0, "pipeline": 0, "processors": 0}

        class PreProvidedLightRAG:
            def __init__(self):
                self.llm_model_func = llm
                self.embedding_func = emb
                self.workspace = "pre"
                self._storages_status = _UninitializedStatus()

            async def initialize_storages(self):
                init_calls["storages"] += 1
                self._storages_status = _InitializedStatus()

            def key_string_value_json_storage_cls(self, **kwargs):
                return parse_cache

        async def fake_pipeline_status():
            init_calls["pipeline"] += 1

        import lightrag.kg.shared_storage as shared_storage

        monkeypatch.setattr(
            shared_storage, "initialize_pipeline_status", fake_pipeline_status
        )

        def fake_init_processors():
            init_calls["processors"] += 1
            rag.modal_processors = {"generic": object()}

        monkeypatch.setattr(rag, "_initialize_processors", fake_init_processors)

        rag.lightrag = PreProvidedLightRAG()
        rag.llm_model_func = None
        rag.embedding_func = None
        rag.parse_cache = None
        rag.modal_processors = {}

        result = await rag._ensure_lightrag_initialized()

        assert result == {"success": True}
        assert rag.llm_model_func is llm
        assert rag.embedding_func is emb
        assert init_calls == {"storages": 1, "pipeline": 1, "processors": 1}
        assert rag.parse_cache is parse_cache
        assert getattr(parse_cache, "ready", False) is True

    @pytest.mark.asyncio
    async def test_skips_storage_init_when_already_initialized(
        self, monkeypatch, tmp_path
    ):
        rag, _ = _make_rag(monkeypatch, tmp_path)

        class StubParseCache:
            async def initialize(self):
                pass

        storages_calls = {"n": 0}
        pipeline_calls = {"n": 0}

        class ReadyLightRAG:
            def __init__(self):
                self.workspace = "ready"
                self._storages_status = _InitializedStatus()

            async def initialize_storages(self):
                storages_calls["n"] += 1

            def key_string_value_json_storage_cls(self, **kwargs):
                return StubParseCache()

        async def fake_pipeline_status():
            pipeline_calls["n"] += 1

        import lightrag.kg.shared_storage as shared_storage

        monkeypatch.setattr(
            shared_storage, "initialize_pipeline_status", fake_pipeline_status
        )
        monkeypatch.setattr(rag, "_initialize_processors", lambda: None)

        rag.lightrag = ReadyLightRAG()
        rag.llm_model_func = object()
        rag.embedding_func = object()
        rag.parse_cache = None
        rag.modal_processors = {}

        result = await rag._ensure_lightrag_initialized()

        assert result == {"success": True}
        assert storages_calls["n"] == 0
        assert pipeline_calls["n"] == 0
        assert rag.parse_cache is not None

    @pytest.mark.asyncio
    async def test_preprovided_init_failure_returns_error(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)

        class BrokenLightRAG:
            def __init__(self):
                self._storages_status = _UninitializedStatus()

            async def initialize_storages(self):
                raise RuntimeError("storage-boom")

        rag.lightrag = BrokenLightRAG()
        rag.llm_model_func = object()
        rag.embedding_func = object()

        result = await rag._ensure_lightrag_initialized()
        assert result["success"] is False
        assert "Failed to initialize pre-provided LightRAG instance" in result["error"]
        assert "storage-boom" in result["error"]

    @pytest.mark.asyncio
    async def test_already_ready_preprovided_noop_path(self, monkeypatch, tmp_path):
        """When storages, parse_cache, and processors exist, return success without re-init."""
        rag, _ = _make_rag(monkeypatch, tmp_path)

        class ReadyLightRAG:
            def __init__(self):
                self._storages_status = _InitializedStatus()

            async def initialize_storages(self):
                raise AssertionError("should not re-initialize storages")

            def key_string_value_json_storage_cls(self, **kwargs):
                raise AssertionError("should not create parse cache")

        processors_called = {"n": 0}
        monkeypatch.setattr(
            rag,
            "_initialize_processors",
            lambda: processors_called.__setitem__("n", processors_called["n"] + 1),
        )

        rag.lightrag = ReadyLightRAG()
        rag.llm_model_func = object()
        rag.embedding_func = object()
        rag.parse_cache = AsyncMock()
        rag.modal_processors = {"generic": object()}

        result = await rag._ensure_lightrag_initialized()
        assert result == {"success": True}
        assert processors_called["n"] == 0
