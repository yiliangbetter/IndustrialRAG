"""Regression tests for context-extractor construction and parser-install probe.

_initialize_processors is already claimed elsewhere. These tests lock the
standalone _create_context_extractor fail-closed gate (missing LightRAG) and
tokenizer wiring, plus check_parser_installation delegation and empty
finalize_storages.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_rag(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
    )
    return rag_module.RAGAnything(config=config), rag_module


class TestCreateContextExtractor:
    def test_requires_initialized_lightrag(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.lightrag = None
        with pytest.raises(ValueError, match="LightRAG must be initialized"):
            rag._create_context_extractor()

    def test_wires_tokenizer_from_lightrag(self, monkeypatch, tmp_path):
        rag, rag_module = _make_rag(monkeypatch, tmp_path)
        tokenizer = object()
        rag.lightrag = SimpleNamespace(tokenizer=tokenizer)
        rag.config.context_window = 3
        rag.config.context_mode = "chunk"
        rag.config.max_context_tokens = 512

        created = {}

        def fake_context_extractor(**kwargs):
            created.update(kwargs)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(rag_module, "ContextExtractor", fake_context_extractor)

        extractor = rag._create_context_extractor()
        assert extractor.tokenizer is tokenizer
        assert created["tokenizer"] is tokenizer
        assert created["config"].context_window == 3
        assert created["config"].context_mode == "chunk"
        assert created["config"].max_context_tokens == 512


class TestCheckParserInstallation:
    def test_delegates_to_doc_parser(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        calls = {"n": 0}

        def check():
            calls["n"] += 1
            return False

        rag.doc_parser.check_installation = check
        assert rag.check_parser_installation() is False
        assert calls["n"] == 1
        assert rag._parser_installation_checked is False


class TestFinalizeStoragesEmpty:
    @pytest.mark.asyncio
    async def test_no_storages_does_not_raise(self, monkeypatch, tmp_path):
        rag, _ = _make_rag(monkeypatch, tmp_path)
        rag.parse_cache = None
        rag.lightrag = None
        await rag.finalize_storages()
