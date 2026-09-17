"""Graph ingest LLM_MODEL and OCR re-ingest PARSER env must reach construction.

``scripts/batch_ingest_content_lists_with_graph.py`` runs LightRAG entity
extraction; a missing LLM_MODEL default or ignored override writes the KG with
the wrong chat model. ``scripts/reingest_uploaded_documents_ocr.py`` still
honors PARSER while hardcoding ``parse_method=ocr`` — swapping MinerU for
Docling/PaddleOCR changes OCR quality even though the method stays ocr.
Distinct from #182 (gateway host), #179 (OCR glob / processor flags), and
#184 (VISION_MODEL routing on processor-on scripts).
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

_GRAPH_ENV = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BACKEND",
    "EMBEDDING_BINDING_HOST",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "PARSER",
    "RAG_DATA_REPO",
    "RAG_DATA_UPLOAD_SUBDIR",
    "HF_HOME",
)

_REINGEST_ENV = (
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
)


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_llm_model_d37f",
        REPO_ROOT / "scripts" / "batch_ingest_content_lists_with_graph.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def reingest():
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_parser_env_d37f",
        REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    keys = tuple(dict.fromkeys(_GRAPH_ENV + _REINGEST_ENV))
    before = {key: os.environ.get(key) for key in keys}
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
        self.initialize_storages = AsyncMock()

    async def initialize_storages(self):
        return None


class _RecordingEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        pass


class _CapturingRAG:
    last_config = None

    def __init__(self, **kwargs):
        type(self).last_config = kwargs.get("config")
        self.process_document_complete = AsyncMock()
        self.insert_content_list = AsyncMock()
        self.finalize_storages = AsyncMock()


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _empty_data_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    return repo


async def _run_graph_until_constructed(
    bigraph, monkeypatch, tmp_path, *, complete_mock
):
    repo = _empty_data_repo(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
        ],
    )

    _CapturingLightRAG.last_kwargs = None

    with (
        patch("lightrag.llm.openai.openai_complete_if_cache", complete_mock),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("raganything.RAGAnything", _CapturingRAG),
    ):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()

    assert _CapturingLightRAG.last_kwargs is not None
    return _CapturingLightRAG.last_kwargs


def _enter_reingest_stack():
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", _CapturingLightRAG))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock())
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


async def _run_reingest_until_config(reingest, monkeypatch, tmp_path):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(folder),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-model-download",
        ],
    )
    _CapturingRAG.last_config = None
    with _enter_reingest_stack():
        with pytest.raises(SystemExit, match="No supported files"):
            await reingest.main()
    assert _CapturingRAG.last_config is not None
    return _CapturingRAG.last_config


@pytest.mark.asyncio
async def test_graph_ingest_llm_model_defaults_to_gpt_4o_mini(
    bigraph, monkeypatch, tmp_path
):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _run_graph_until_constructed(
        bigraph, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await kwargs["llm_model_func"]("build graph")
    complete_mock.assert_awaited_once()
    assert complete_mock.await_args.args[0] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_graph_ingest_llm_model_env_is_forwarded(bigraph, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _run_graph_until_constructed(
        bigraph, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await kwargs["llm_model_func"]("build graph")
    assert complete_mock.await_args.args[0] == "deepseek-chat"


@pytest.mark.asyncio
async def test_reingest_parser_defaults_to_mineru_with_ocr_method(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.delenv("PARSER", raising=False)
    config = await _run_reingest_until_config(reingest, monkeypatch, tmp_path)
    assert config.parser == "mineru"
    assert config.parse_method == "ocr"


@pytest.mark.asyncio
async def test_reingest_parser_env_overrides_engine_but_keeps_ocr_method(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.setenv("PARSER", "paddleocr")
    config = await _run_reingest_until_config(reingest, monkeypatch, tmp_path)
    assert config.parser == "paddleocr"
    assert config.parse_method == "ocr"
