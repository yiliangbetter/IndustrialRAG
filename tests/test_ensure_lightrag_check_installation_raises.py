"""Parser installation probes that raise must fail closed, not crash ingest.

MinerU/Docling/PaddleOCR `check_installation` can throw (ImportError, OSError
from a broken subprocess) instead of returning False. `_ensure_lightrag_initialized`
must catch that and return `{success: False}` so process_document_complete can
surface a RuntimeError instead of leaving scanners with an unhandled crash.
"""

from types import SimpleNamespace

import pytest


class RaisingParser:
    def __init__(self, exc):
        self.exc = exc
        self.checks = 0

    def check_installation(self):
        self.checks += 1
        raise self.exc


def _make_rag(monkeypatch, tmp_path, parser):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: parser)
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
        allow_embedding_only_ingestion=False,
    )
    rag = rag_module.RAGAnything(config=config)
    rag.doc_parser = parser
    rag.lightrag = None
    rag.llm_model_func = SimpleNamespace()
    rag.embedding_func = SimpleNamespace()
    return rag


@pytest.mark.asyncio
async def test_check_installation_exception_returns_fail_closed_dict(
    monkeypatch, tmp_path
):
    parser = RaisingParser(OSError("mineru binary crashed"))
    rag = _make_rag(monkeypatch, tmp_path, parser)

    result = await rag._ensure_lightrag_initialized()

    assert result["success"] is False
    assert "Unexpected error during LightRAG initialization" in result["error"]
    assert "mineru binary crashed" in result["error"]
    assert parser.checks == 1
    # Probe did not succeed, so a later retry can still run check_installation.
    assert rag._parser_installation_checked is False
    assert rag.lightrag is None


@pytest.mark.asyncio
async def test_check_installation_importerror_is_not_leaked(monkeypatch, tmp_path):
    parser = RaisingParser(ImportError("No module named 'magic_pdf'"))
    rag = _make_rag(monkeypatch, tmp_path, parser)

    result = await rag._ensure_lightrag_initialized()

    assert result["success"] is False
    assert "No module named 'magic_pdf'" in result["error"]
    assert rag._parser_installation_checked is False
