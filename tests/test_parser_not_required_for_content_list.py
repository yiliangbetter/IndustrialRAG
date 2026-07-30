"""Parser CLI must not be required for parse-free insert/query paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from raganything.config import RAGAnythingConfig
from raganything.raganything import RAGAnything


class _FakeParser:
    def check_installation(self) -> bool:
        return False


def _rag_with_missing_parser(**config_kwargs) -> RAGAnything:
    kwargs = {
        "working_dir": "/tmp/raganything_parser_check_test",
        "allow_embedding_only_ingestion": False,
    }
    kwargs.update(config_kwargs)
    config = RAGAnythingConfig(**kwargs)
    rag = RAGAnything(
        config=config,
        llm_model_func=AsyncMock(return_value="ok"),
        embedding_func=AsyncMock(return_value=[[0.1, 0.2]]),
    )
    rag.doc_parser = _FakeParser()
    rag._parser_installation_checked = False
    # Avoid atexit finalize noise against test stubs.
    rag.finalize_storages = AsyncMock()
    return rag


def _stub_lightrag(rag: RAGAnything) -> SimpleNamespace:
    return SimpleNamespace(
        _storages_status=SimpleNamespace(name="INITIALIZED"),
        llm_model_func=rag.llm_model_func,
        embedding_func=rag.embedding_func,
        workspace="default",
        tokenizer=MagicMock(),
        __dict__={},
        ainsert=AsyncMock(),
        finalize_storages=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_ensure_lightrag_requires_parser_by_default(tmp_path):
    rag = _rag_with_missing_parser(working_dir=str(tmp_path / "rag"))
    result = await rag._ensure_lightrag_initialized()
    assert result["success"] is False
    assert "not properly installed" in result["error"]
    assert rag._parser_installation_checked is False


@pytest.mark.asyncio
async def test_ensure_lightrag_skips_parser_when_not_required(tmp_path):
    rag = _rag_with_missing_parser(working_dir=str(tmp_path / "rag"))
    # Pre-provide a minimal LightRAG stub so init does not construct a real one.
    rag.lightrag = _stub_lightrag(rag)
    rag.parse_cache = MagicMock()
    rag.modal_processors = {"generic": MagicMock()}

    result = await rag._ensure_lightrag_initialized(require_parser=False)
    assert result["success"] is True
    # Later parse paths must still be able to enforce the check.
    assert rag._parser_installation_checked is False
    result_parse = await rag._ensure_lightrag_initialized(require_parser=True)
    assert result_parse["success"] is False


@pytest.mark.asyncio
async def test_insert_content_list_does_not_require_parser(tmp_path, monkeypatch):
    rag = _rag_with_missing_parser(working_dir=str(tmp_path / "rag"))
    rag.lightrag = _stub_lightrag(rag)
    rag.parse_cache = MagicMock()
    rag.modal_processors = {"generic": MagicMock()}

    called = {}

    async def _fake_insert_text_content(lightrag, input, **kwargs):
        called["text"] = input

    async def _fake_mark(doc_id):
        called["marked"] = doc_id

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", _fake_insert_text_content
    )
    monkeypatch.setattr(rag, "_mark_multimodal_processing_complete", _fake_mark)

    await rag.insert_content_list(
        [{"type": "text", "text": "hello from pre-parsed list", "page_idx": 0}],
        file_path="doc.json",
        skip_multimodal_processing=True,
    )

    assert "hello from pre-parsed list" in called["text"]
    assert called["marked"].startswith("doc-")
