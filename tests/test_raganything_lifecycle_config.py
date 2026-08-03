"""Regression tests for RAGAnything lifecycle and config introspection.

Covers processor enable-flag gating, config/update helpers, parser verification,
and storage finalization — high blast-radius paths with no coverage on main.
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
    """Replace modal processor classes with lightweight call-capturing stubs."""
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


class TestInitializeProcessors:
    def test_requires_lightrag(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.lightrag = None
        with pytest.raises(ValueError, match="LightRAG instance must be initialized"):
            rag._initialize_processors()

    def test_all_enabled_registers_image_table_equation_and_generic(
        self, monkeypatch, tmp_path
    ):
        rag, rag_module = _make_rag(monkeypatch, tmp_path)
        created = _patch_modal_processors(monkeypatch, rag_module)

        vision = object()
        llm = object()
        rag.vision_model_func = vision
        rag.llm_model_func = llm
        rag.lightrag = SimpleNamespace(tokenizer=object())

        rag._initialize_processors()

        assert set(rag.modal_processors) == {"image", "table", "equation", "generic"}
        assert len(created["image"]) == 1
        assert created["image"][0].kwargs["modal_caption_func"] is vision
        assert created["table"][0].kwargs["modal_caption_func"] is llm
        assert created["equation"][0].kwargs["modal_caption_func"] is llm
        assert created["generic"][0].kwargs["modal_caption_func"] is llm

    def test_disabled_flags_omit_typed_processors_but_keep_generic(
        self, monkeypatch, tmp_path
    ):
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
        assert len(created["generic"]) == 1

    def test_partial_enable_flags(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(
            monkeypatch,
            tmp_path,
            enable_image_processing=True,
            enable_table_processing=False,
            enable_equation_processing=True,
        )
        _patch_modal_processors(monkeypatch, rag_module)
        rag.llm_model_func = object()
        rag.vision_model_func = None
        rag.lightrag = SimpleNamespace(tokenizer=object())

        rag._initialize_processors()

        assert set(rag.modal_processors) == {"image", "equation", "generic"}
        # Without vision_model_func, image processor falls back to llm_model_func
        assert (
            rag.modal_processors["image"].kwargs["modal_caption_func"]
            is rag.llm_model_func
        )


class TestUpdateConfig:
    def test_updates_known_keys_and_warns_on_unknown(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        warnings = []
        monkeypatch.setattr(rag.logger, "warning", lambda msg: warnings.append(msg))

        rag.update_config(parse_method="ocr", not_a_real_key=123)

        assert rag.config.parse_method == "ocr"
        assert any("Unknown config parameter: not_a_real_key" in w for w in warnings)


class TestVerifyParserInstallationOnce:
    def test_success_is_idempotent(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        calls = {"n": 0}

        def check():
            calls["n"] += 1
            return True

        rag.doc_parser.check_installation = check
        assert rag.verify_parser_installation_once() is True
        assert rag.verify_parser_installation_once() is True
        assert calls["n"] == 1
        assert rag._parser_installation_checked is True

    def test_failure_raises_and_does_not_set_checked(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.doc_parser.check_installation = lambda: False

        with pytest.raises(RuntimeError, match="not properly installed"):
            rag.verify_parser_installation_once()

        assert rag._parser_installation_checked is False


class TestFinalizeStorages:
    @pytest.mark.asyncio
    async def test_finalizes_parse_cache_and_lightrag_concurrently(
        self, monkeypatch, tmp_path
    ):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        parse_cache = SimpleNamespace(finalize=AsyncMock())
        lightrag = SimpleNamespace(finalize_storages=AsyncMock())
        rag.parse_cache = parse_cache
        rag.lightrag = lightrag

        await rag.finalize_storages()

        parse_cache.finalize.assert_awaited_once()
        lightrag.finalize_storages.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_no_storages(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.parse_cache = None
        rag.lightrag = None

        # Should not raise
        await rag.finalize_storages()

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
    def test_reports_core_sections_and_filters_sensitive_kwargs(
        self, monkeypatch, tmp_path
    ):
        rag, _ = _make_rag(
            monkeypatch,
            tmp_path,
            enable_image_processing=False,
            context_window=3,
        )
        rag.lightrag_kwargs = {
            "top_k": 10,
            "llm_model_kwargs": {"api_key": "secret"},
            "vector_db_storage_cls_kwargs": {"password": "x"},
            "embedding_func": lambda x: x,
        }

        info = rag.get_config_info()

        assert info["parsing"]["parser"] == "mineru"
        assert info["multimodal_processing"]["enable_image_processing"] is False
        assert info["context_extraction"]["context_window"] == 3
        assert info["lightrag_config"]["custom_parameters"] == {"top_k": 10}
        assert "llm_model_kwargs" not in info["lightrag_config"]["custom_parameters"]
        assert "embedding_func" not in info["lightrag_config"]["custom_parameters"]

    def test_default_lightrag_note_when_no_kwargs(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.lightrag_kwargs = {}
        info = rag.get_config_info()
        assert info["lightrag_config"]["custom_parameters"] == {}
        assert "default" in info["lightrag_config"]["note"].lower()


class TestGetProcessorInfo:
    def test_uninitialized_status(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(monkeypatch, tmp_path)
        monkeypatch.setattr(
            rag_module.MineruParser,
            "check_installation",
            lambda self: True,
        )
        monkeypatch.setattr(
            rag_module,
            "get_parser",
            lambda name: SimpleNamespace(check_installation=lambda: name == "mineru"),
        )
        monkeypatch.setattr(rag_module, "SUPPORTED_PARSERS", ["mineru", "docling"])

        rag.modal_processors = {}
        info = rag.get_processor_info()

        assert info["status"] == "Not initialized"
        assert info["processors"] == {}
        assert info["models"]["llm_model"] == "Not provided"
        assert info["parser_installation"]["mineru"] is True
        assert info["parser_installation"]["docling"] is False

    def test_initialized_lists_processors_and_models(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(monkeypatch, tmp_path)
        monkeypatch.setattr(
            rag_module.MineruParser,
            "check_installation",
            lambda self: False,
        )
        monkeypatch.setattr(
            rag_module,
            "get_parser",
            lambda name: SimpleNamespace(check_installation=lambda: True),
        )
        monkeypatch.setattr(rag_module, "SUPPORTED_PARSERS", ["mineru"])
        monkeypatch.setattr(
            rag_module,
            "get_processor_supports",
            lambda proc_type: [f"support-{proc_type}"],
        )

        class FakeProc:
            pass

        rag.llm_model_func = lambda *a, **k: None
        rag.vision_model_func = None
        rag.embedding_func = lambda *a, **k: None
        rag.modal_processors = {"image": FakeProc(), "generic": FakeProc()}

        info = rag.get_processor_info()

        assert info["status"] == "Initialized"
        assert info["mineru_installed"] is False
        assert info["models"]["llm_model"] == "External function"
        assert info["models"]["vision_model"] == "Not provided"
        assert info["models"]["embedding_model"] == "External function"
        assert info["processors"]["image"]["class"] == "FakeProc"
        assert info["processors"]["image"]["supports"] == ["support-image"]
        assert info["processors"]["image"]["enabled"] is True
