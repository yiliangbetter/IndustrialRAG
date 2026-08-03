"""Tests for RAGAnything parser gate, context rewire, and introspection helpers.

Covers fail-closed installation checks, context config updates that must
propagate into modal processors, and safe config/processor info shapes.
"""

import atexit
import logging

import pytest

pytest.importorskip("lightrag")

import raganything.raganything as rag_module
from raganything.config import RAGAnythingConfig


class StubParser:
    def __init__(self, installed=True, counter=None):
        self._installed = installed
        self._counter = counter

    def check_installation(self):
        if self._counter is not None:
            self._counter["n"] += 1
        return self._installed


class StubProcessor:
    def __init__(self):
        self.context_extractor = "old-extractor"


def _make_rag(monkeypatch, tmp_path, parser=None, **config_kwargs):
    stub = parser or StubParser(installed=True)
    monkeypatch.setattr(rag_module, "get_parser", lambda _name: stub)
    monkeypatch.setattr(rag_module.atexit, "register", lambda *a, **k: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
        **config_kwargs,
    )
    rag = rag_module.RAGAnything(config=config)
    atexit.unregister(rag.close)
    rag.doc_parser = stub
    return rag


class TestVerifyParserInstallationOnce:
    def test_fail_closed_when_parser_missing(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path, parser=StubParser(installed=False))

        with pytest.raises(RuntimeError, match="not properly installed"):
            rag.verify_parser_installation_once()

        assert rag._parser_installation_checked is False

    def test_success_is_cached_and_skips_recheck(self, monkeypatch, tmp_path):
        counter = {"n": 0}
        rag = _make_rag(
            monkeypatch,
            tmp_path,
            parser=StubParser(installed=True, counter=counter),
        )

        assert rag.verify_parser_installation_once() is True
        assert rag.verify_parser_installation_once() is True
        assert counter["n"] == 1
        assert rag._parser_installation_checked is True


class TestUpdateContextConfig:
    def test_unknown_key_logs_warning_without_changing_known_fields(
        self, monkeypatch, tmp_path, caplog
    ):
        rag = _make_rag(monkeypatch, tmp_path, context_window=1)

        with caplog.at_level(logging.WARNING):
            rag.update_context_config(not_a_real_key=99, context_window=4)

        assert rag.config.context_window == 4
        assert not hasattr(rag.config, "not_a_real_key")
        assert any(
            "Unknown context config parameter: not_a_real_key" in r.message
            for r in caplog.records
        )

    def test_rewires_processors_when_initialized(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path, context_window=1)
        rag.lightrag = object()
        rag.modal_processors = {
            "image": StubProcessor(),
            "table": StubProcessor(),
        }
        new_extractor = object()
        monkeypatch.setattr(rag, "_create_context_extractor", lambda: new_extractor)
        monkeypatch.setattr(
            rag,
            "_create_context_config",
            lambda: {"context_window": rag.config.context_window},
        )

        rag.update_context_config(context_window=7, context_mode="chunk")

        assert rag.config.context_window == 7
        assert rag.config.context_mode == "chunk"
        assert rag.context_extractor is new_extractor
        for processor in rag.modal_processors.values():
            assert processor.context_extractor is new_extractor

    def test_skips_processor_rewire_when_not_initialized(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path, context_window=1)
        rag.lightrag = None
        rag.modal_processors = {}
        created = {"n": 0}

        def boom():
            created["n"] += 1
            raise AssertionError("should not create extractor without processors")

        monkeypatch.setattr(rag, "_create_context_extractor", boom)

        rag.update_context_config(context_window=3)
        assert rag.config.context_window == 3
        assert created["n"] == 0


class TestConfigAndProcessorInfo:
    def test_get_config_info_shape_and_filters_sensitive_kwargs(
        self, monkeypatch, tmp_path
    ):
        rag = _make_rag(monkeypatch, tmp_path)
        rag.lightrag_kwargs = {
            "top_k": 12,
            "llm_model_kwargs": {"api_key": "secret"},
            "vector_db_storage_cls_kwargs": {"password": "nope"},
            "callable_param": lambda: None,
        }

        info = rag.get_config_info()

        assert info["directory"]["working_dir"] == rag.config.working_dir
        assert info["parsing"]["parser"] == "mineru"
        assert "enable_image_processing" in info["multimodal_processing"]
        assert "filter_content_types" in info["context_extraction"]
        assert "supported_file_extensions" in info["batch_processing"]

        custom = info["lightrag_config"]["custom_parameters"]
        assert custom["top_k"] == 12
        assert "llm_model_kwargs" not in custom
        assert "vector_db_storage_cls_kwargs" not in custom
        assert "callable_param" not in custom

    def test_get_processor_info_not_initialized(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path)
        monkeypatch.setattr(
            rag_module.MineruParser,
            "check_installation",
            lambda self: True,
        )
        monkeypatch.setattr(
            rag_module,
            "get_parser",
            lambda _name: StubParser(installed=True),
        )

        info = rag.get_processor_info()

        assert info["status"] == "Not initialized"
        assert info["processors"] == {}
        assert info["models"]["llm_model"] == "Not provided"
        assert "config" in info
        assert "parser_installation" in info

    def test_get_processor_info_initialized_lists_processors(
        self, monkeypatch, tmp_path
    ):
        rag = _make_rag(monkeypatch, tmp_path)
        rag.llm_model_func = lambda *a, **k: None
        rag.modal_processors = {"image": StubProcessor(), "table": StubProcessor()}
        monkeypatch.setattr(
            rag_module.MineruParser,
            "check_installation",
            lambda self: False,
        )
        monkeypatch.setattr(
            rag_module,
            "get_parser",
            lambda _name: StubParser(installed=True),
        )
        monkeypatch.setattr(
            rag_module,
            "get_processor_supports",
            lambda proc_type: [f"{proc_type}-support"],
        )

        info = rag.get_processor_info()

        assert info["status"] == "Initialized"
        assert info["mineru_installed"] is False
        assert info["models"]["llm_model"] == "External function"
        assert set(info["processors"]) == {"image", "table"}
        assert info["processors"]["image"]["class"] == "StubProcessor"
        assert info["processors"]["image"]["supports"] == ["image-support"]
        assert info["processors"]["image"]["enabled"] is True
