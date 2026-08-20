"""Regression tests for RAGAnything lifecycle, processor gating, and finalize.

These paths decide whether images/tables are processed, whether parser install
failures raise, and whether secrets leak through get_config_info.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_rag(monkeypatch, tmp_path, **config_overrides):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    defaults = {
        "working_dir": str(tmp_path / "rag_workdir"),
        "parser": "mineru",
        "enable_image_processing": True,
        "enable_table_processing": True,
        "enable_equation_processing": True,
    }
    defaults.update(config_overrides)
    config = RAGAnythingConfig(**defaults)
    return rag_module.RAGAnything(config=config), rag_module


def _patch_modal_processors(monkeypatch, rag_module):
    created = {"image": [], "table": [], "equation": [], "generic": []}

    class StubProcessor:
        kind = "generic"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created[self.kind].append(self)

    class StubImage(StubProcessor):
        kind = "image"

    class StubTable(StubProcessor):
        kind = "table"

    class StubEquation(StubProcessor):
        kind = "equation"

    class StubGeneric(StubProcessor):
        kind = "generic"

    monkeypatch.setattr(rag_module, "ImageModalProcessor", StubImage)
    monkeypatch.setattr(rag_module, "TableModalProcessor", StubTable)
    monkeypatch.setattr(rag_module, "EquationModalProcessor", StubEquation)
    monkeypatch.setattr(rag_module, "GenericModalProcessor", StubGeneric)

    def fake_context_extractor(**kwargs):
        return SimpleNamespace(
            config=kwargs.get("config"),
            tokenizer=kwargs.get("tokenizer"),
        )

    monkeypatch.setattr(rag_module, "ContextExtractor", fake_context_extractor)
    return created


class TestEmbeddingOnlyDisablesModalFlags:
    def test_embedding_only_turns_off_image_table_equation(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(
            monkeypatch,
            tmp_path,
            allow_embedding_only_ingestion=True,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        assert rag.config.enable_image_processing is False
        assert rag.config.enable_table_processing is False
        assert rag.config.enable_equation_processing is False


class TestInitializeProcessors:
    def test_requires_lightrag(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.lightrag = None
        with pytest.raises(ValueError, match="LightRAG instance must be initialized"):
            rag._initialize_processors()

    def test_all_enabled_registers_typed_and_generic(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(monkeypatch, tmp_path)
        created = _patch_modal_processors(monkeypatch, rag_module)

        vision = object()
        llm = object()
        rag.vision_model_func = vision
        rag.llm_model_func = llm
        rag.lightrag = SimpleNamespace(tokenizer=object())

        rag._initialize_processors()

        assert set(rag.modal_processors) == {"image", "table", "equation", "generic"}
        assert created["image"][0].kwargs["modal_caption_func"] is vision
        assert created["table"][0].kwargs["modal_caption_func"] is llm
        assert created["generic"][0].kwargs["modal_caption_func"] is llm

    def test_disabled_flags_keep_only_generic(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(
            monkeypatch,
            tmp_path,
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
        )
        created = _patch_modal_processors(monkeypatch, rag_module)
        rag.llm_model_func = object()
        rag.lightrag = SimpleNamespace(tokenizer=object())

        rag._initialize_processors()

        assert set(rag.modal_processors) == {"generic"}
        assert created["image"] == []
        assert created["table"] == []
        assert created["equation"] == []

    def test_image_falls_back_to_llm_without_vision(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(
            monkeypatch,
            tmp_path,
            enable_image_processing=True,
            enable_table_processing=False,
            enable_equation_processing=False,
        )
        _patch_modal_processors(monkeypatch, rag_module)
        rag.llm_model_func = object()
        rag.vision_model_func = None
        rag.lightrag = SimpleNamespace(tokenizer=object())

        rag._initialize_processors()

        assert set(rag.modal_processors) == {"image", "generic"}
        assert (
            rag.modal_processors["image"].kwargs["modal_caption_func"]
            is rag.llm_model_func
        )


class TestUpdateConfigAndParserVerify:
    def test_updates_known_keys_and_warns_on_unknown(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        warnings = []
        monkeypatch.setattr(rag.logger, "warning", lambda msg: warnings.append(msg))

        rag.update_config(parse_method="ocr", not_a_real_key=123)

        assert rag.config.parse_method == "ocr"
        assert any("Unknown config parameter: not_a_real_key" in w for w in warnings)

    def test_verify_parser_success_is_idempotent(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        calls = {"n": 0}

        def check():
            calls["n"] += 1
            return True

        rag.doc_parser.check_installation = check
        assert rag.verify_parser_installation_once() is True
        assert rag.verify_parser_installation_once() is True
        assert calls["n"] == 1

    def test_verify_parser_failure_raises(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.doc_parser.check_installation = lambda: False

        with pytest.raises(RuntimeError, match="not properly installed"):
            rag.verify_parser_installation_once()

        assert rag._parser_installation_checked is False


class TestFinalizeStorages:
    @pytest.mark.asyncio
    async def test_finalizes_parse_cache_and_lightrag(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        parse_cache = SimpleNamespace(finalize=AsyncMock())
        lightrag = SimpleNamespace(finalize_storages=AsyncMock())
        rag.parse_cache = parse_cache
        rag.lightrag = lightrag

        await rag.finalize_storages()

        parse_cache.finalize.assert_awaited_once()
        lightrag.finalize_storages.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_propagates_finalization_errors(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.parse_cache = SimpleNamespace(
            finalize=AsyncMock(side_effect=RuntimeError("cache boom"))
        )
        rag.lightrag = None

        with pytest.raises(RuntimeError, match="cache boom"):
            await rag.finalize_storages()


class TestGetConfigInfo:
    def test_filters_sensitive_and_callable_lightrag_kwargs(
        self, monkeypatch, tmp_path
    ):
        rag, _ = _make_rag(monkeypatch, tmp_path, enable_image_processing=False)
        rag.lightrag_kwargs = {
            "top_k": 10,
            "llm_model_kwargs": {"api_key": "secret"},
            "vector_db_storage_cls_kwargs": {"password": "x"},
            "embedding_func": lambda x: x,
        }

        info = rag.get_config_info()

        assert info["parsing"]["parser"] == "mineru"
        assert info["multimodal_processing"]["enable_image_processing"] is False
        assert info["lightrag_config"]["custom_parameters"] == {"top_k": 10}
        assert "llm_model_kwargs" not in info["lightrag_config"]["custom_parameters"]
        assert "embedding_func" not in info["lightrag_config"]["custom_parameters"]
