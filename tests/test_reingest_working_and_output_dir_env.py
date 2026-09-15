"""OCR re-ingest must persist into WORKING_DIR / OUTPUT_DIR when CLI omits dirs.

``scripts/reingest_uploaded_documents_ocr.py`` writes the graph into
``WORKING_DIR`` (default ``./rag_storage``) and parser artifacts into
``--output-dir`` or ``OUTPUT_DIR`` (default ``./output``). Pointing those
at the wrong tree overwrites an unrelated index or collides MinerU output.
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


@pytest.fixture(scope="module")
def reingest():
    path = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"
    spec = importlib.util.spec_from_file_location(
        "reingest_working_output_dir_env_be19", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env():
    keys = (
        "OPENAI_API_KEY",
        "LLM_BINDING_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_BINDING_HOST",
        "WORKING_DIR",
        "OUTPUT_DIR",
        "PARSER",
        "LLM_BINDING_HOST",
        "OPENAI_BASE_URL",
        "HF_HOME",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _FakeLightRAG:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeLightRAG.created.append(self)

    async def initialize_storages(self):
        return None


class _FakeEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size
        self.func = func


class _CapturingRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.process_document_complete = AsyncMock()


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _enter_reingest_stack(rag_ctor):
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", _FakeLightRAG))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock())
    )
    stack.enter_context(patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()))
    stack.enter_context(patch("lightrag.utils.EmbeddingFunc", _FakeEmbeddingFunc))
    stack.enter_context(
        patch("lightrag.utils.logger", SimpleNamespace(info=lambda *a, **k: None))
    )
    stack.enter_context(patch("raganything.RAGAnything", rag_ctor))
    stack.enter_context(
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback")
    )
    return stack


@pytest.mark.asyncio
async def test_reingest_uses_working_dir_and_output_dir_env_when_cli_omits_output(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    pdf = folder / "manual.pdf"
    pdf.write_bytes(b"%PDF")
    working = tmp_path / "rag_storage_ocr"
    parsed = tmp_path / "parser_out"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(working))
    monkeypatch.setenv("OUTPUT_DIR", str(parsed))
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(folder),
            "--skip-model-download",
        ],
    )

    _FakeLightRAG.created = []
    rag_holder = {}

    def rag_ctor(**kwargs):
        rag = _CapturingRAG(**kwargs)
        rag_holder["rag"] = rag
        return rag

    with _enter_reingest_stack(rag_ctor):
        await reingest.main()

    rag = rag_holder["rag"]
    config = rag.kwargs["config"]
    assert config.working_dir == str(working.resolve())
    assert config.parser_output_dir == str(parsed.resolve())
    assert config.parse_method == "ocr"

    assert len(_FakeLightRAG.created) == 1
    assert _FakeLightRAG.created[0].kwargs["working_dir"] == str(working.resolve())

    rag.process_document_complete.assert_awaited_once()
    call = rag.process_document_complete.await_args
    assert Path(call.args[0]) == pdf.resolve()
    assert call.kwargs["output_dir"] == str(parsed.resolve())
    assert call.kwargs["parse_method"] == "ocr"


@pytest.mark.asyncio
async def test_reingest_cli_output_dir_overrides_output_dir_env(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF")
    env_out = tmp_path / "env_out"
    cli_out = tmp_path / "cli_out"
    working = tmp_path / "wd"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(working))
    monkeypatch.setenv("OUTPUT_DIR", str(env_out))
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
            str(cli_out),
            "--skip-model-download",
        ],
    )

    _FakeLightRAG.created = []
    rag_holder = {}

    def rag_ctor(**kwargs):
        rag = _CapturingRAG(**kwargs)
        rag_holder["rag"] = rag
        return rag

    with _enter_reingest_stack(rag_ctor):
        await reingest.main()

    config = rag_holder["rag"].kwargs["config"]
    assert config.parser_output_dir == str(cli_out.resolve())
    assert config.working_dir == str(working.resolve())
    call = rag_holder["rag"].process_document_complete.await_args
    assert call.kwargs["output_dir"] == str(cli_out.resolve())
    assert call.kwargs["output_dir"] != str(env_out.resolve())
