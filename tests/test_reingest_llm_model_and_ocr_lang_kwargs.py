"""OCR re-ingest must honor LLM_MODEL and OCR_LANG parse kwargs.

``scripts/reingest_uploaded_documents_ocr.py`` is the production OCR
re-parse path. Entity/caption LLM calls must use ``LLM_MODEL`` (default
``gpt-4o-mini``). ``OCR_LANG`` must reach MinerU when ``MINERU_LANG`` is
unset, and a non-MinerU ``PARSER`` must not inherit the MinerU
``backend=pipeline`` default.

Distinct from #130 (pipeline copy of ``_mineru_parse_kwargs`` /
``MINERU_LANG``), #179 (OCR glob / ``backend=pipeline`` on MinerU),
#184 (VISION_MODEL), and #185 (PARSER vs hardcoded ``parse_method=ocr``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_KEYS = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BACKEND",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "WORKING_DIR",
    "OUTPUT_DIR",
    "PARSER",
    "MAX_CONCURRENT_FILES",
    "HF_HOME",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "MINERU_LANG",
    "OCR_LANG",
    "MINERU_BACKEND",
    "MINERU_SOURCE",
    "MINERU_DEVICE",
)


@pytest.fixture(scope="module")
def reingest():
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_llm_model_parse_kwargs_065a",
        REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _CapturingLightRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def initialize_storages(self):
        return None


class _RecordingEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        pass


class _CapturingRAG:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.process_document_complete = AsyncMock()
        type(self).last_instance = self


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _enter_reingest_stack(*, complete_mock):
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", _CapturingLightRAG))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", complete_mock)
    )
    stack.enter_context(patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()))
    stack.enter_context(patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc))
    stack.enter_context(
        patch("lightrag.utils.logger", SimpleNamespace(info=lambda *a, **k: None))
    )
    stack.enter_context(patch("raganything.RAGAnything", _CapturingRAG))
    stack.enter_context(
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback")
    )
    return stack


def _argv(folder: Path, output_dir: Path):
    return [
        "reingest_uploaded_documents_ocr",
        "--folder",
        str(folder),
        "--output-dir",
        str(output_dir),
        "--skip-model-download",
    ]


async def _run_reingest_with_pdf(reingest, monkeypatch, tmp_path, *, complete_mock):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", _argv(folder, tmp_path / "out"))
    _CapturingLightRAG.last_kwargs = None
    _CapturingRAG.last_instance = None
    with _enter_reingest_stack(complete_mock=complete_mock):
        await reingest.main()
    assert _CapturingLightRAG.last_kwargs is not None
    assert _CapturingRAG.last_instance is not None
    return _CapturingLightRAG.last_kwargs, _CapturingRAG.last_instance


def test_reingest_ocr_lang_fills_lang_when_mineru_lang_unset(reingest, monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = reingest._mineru_parse_kwargs("mineru")
    assert kwargs["lang"] == "ch"
    assert kwargs["backend"] == "pipeline"


def test_reingest_mineru_lang_wins_over_ocr_lang(reingest, monkeypatch):
    monkeypatch.setenv("MINERU_LANG", "en")
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = reingest._mineru_parse_kwargs("mineru")
    assert kwargs["lang"] == "en"


def test_reingest_paddleocr_does_not_inject_mineru_pipeline_backend(
    reingest, monkeypatch
):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    assert reingest._mineru_parse_kwargs("paddleocr") == {}


@pytest.mark.asyncio
async def test_reingest_llm_model_defaults_to_gpt_4o_mini(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    complete_mock = MagicMock(return_value="ok")
    lightrag_kwargs, _rag = await _run_reingest_with_pdf(
        reingest, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    lightrag_kwargs["llm_model_func"]("extract entities")
    complete_mock.assert_called_once()
    assert complete_mock.call_args.args[0] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_reingest_llm_model_env_is_forwarded(reingest, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    complete_mock = MagicMock(return_value="ok")
    lightrag_kwargs, _rag = await _run_reingest_with_pdf(
        reingest, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    lightrag_kwargs["llm_model_func"]("extract entities")
    assert complete_mock.call_args.args[0] == "deepseek-chat"


@pytest.mark.asyncio
async def test_reingest_ocr_lang_reaches_process_document_complete(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("PARSER", raising=False)
    complete_mock = MagicMock(return_value="ok")
    _lightrag_kwargs, rag = await _run_reingest_with_pdf(
        reingest, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    rag.process_document_complete.assert_awaited_once()
    kwargs = rag.process_document_complete.await_args.kwargs
    assert kwargs["parse_method"] == "ocr"
    assert kwargs["lang"] == "ch"
    assert kwargs["backend"] == "pipeline"
