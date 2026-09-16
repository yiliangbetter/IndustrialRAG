"""OCR re-ingest must pick OpenAI 1536-d vs HF 1024-d and keep OCR concurrency.

``scripts/reingest_uploaded_documents_ocr.py`` writes into the same vector
store used by demo Q&A. Defaulting the wrong dim/model poisons retrieval.
Unlike the parse→graph pipeline (async=1, batch=1), OCR re-ingest defaults
``EMBEDDING_FUNC_MAX_ASYNC=8`` and ``EMBEDDING_BATCH_NUM=10``.

#179 covers glob/limit/OCR method/processor flags, not embedding factory
args or LightRAG concurrency defaults. Distinct from #179 pipeline
``_build_rag`` dims and #180 graph-ingest dims.
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
        "reingest_ocr_embedding_defaults_afad", path
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
        "EMBEDDING_DIM",
        "EMBEDDING_MODEL",
        "EMBEDDING_FUNC_MAX_ASYNC",
        "EMBEDDING_BATCH_NUM",
        "WORKING_DIR",
        "OUTPUT_DIR",
        "PARSER",
        "MAX_CONCURRENT_FILES",
        "HF_HOME",
        "LLM_BINDING_HOST",
        "OPENAI_BASE_URL",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _FakeLightRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def initialize_storages(self):
        return None


class _RecordingEmbeddingFunc:
    calls = []

    def __init__(self, embedding_dim, max_token_size, func):
        type(self).calls.append(
            {
                "embedding_dim": embedding_dim,
                "max_token_size": max_token_size,
                "model": getattr(func, "keywords", {}).get("model"),
            }
        )


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


class _CapturingRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.process_document_complete = AsyncMock()


def _enter_reingest_stack(*, rag_ctor, make_hf=None):
    hf_target = make_hf if make_hf is not None else MagicMock()
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", _FakeLightRAG))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock())
    )
    stack.enter_context(patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()))
    stack.enter_context(patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc))
    stack.enter_context(
        patch("lightrag.utils.logger", SimpleNamespace(info=lambda *a, **k: None))
    )
    stack.enter_context(patch("raganything.RAGAnything", rag_ctor))
    stack.enter_context(
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback")
    )
    stack.enter_context(
        patch(
            "raganything.local_hf_embedding.make_local_hf_embedding_func",
            hf_target,
        )
    )
    return stack


def _argv(folder: Path, output_dir: Path) -> list[str]:
    return [
        "reingest_uploaded_documents_ocr",
        "--folder",
        str(folder),
        "--output-dir",
        str(output_dir),
        "--skip-model-download",
    ]


def _prepare_folder(monkeypatch, tmp_path) -> tuple[Path, Path]:
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF")
    out = tmp_path / "out"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_FUNC_MAX_ASYNC", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_NUM", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", _argv(folder, out))
    return folder, out


@pytest.mark.asyncio
async def test_reingest_openai_defaults_to_1536_and_ocr_concurrency(
    reingest, monkeypatch, tmp_path
):
    _prepare_folder(monkeypatch, tmp_path)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    _RecordingEmbeddingFunc.calls = []
    rag_holder = {}

    def rag_ctor(**kwargs):
        rag = _CapturingRAG(**kwargs)
        rag_holder["rag"] = rag
        return rag

    with _enter_reingest_stack(rag_ctor=rag_ctor):
        await reingest.main()

    assert _RecordingEmbeddingFunc.calls == [
        {
            "embedding_dim": 1536,
            "max_token_size": 8192,
            "model": "text-embedding-3-small",
        },
    ]
    assert _FakeLightRAG.last_kwargs["embedding_func_max_async"] == 8
    assert _FakeLightRAG.last_kwargs["embedding_batch_num"] == 10
    rag_holder["rag"].process_document_complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_reingest_hf_defaults_to_bge_m3_dim_1024(reingest, monkeypatch, tmp_path):
    _prepare_folder(monkeypatch, tmp_path)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://unused.example")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    _RecordingEmbeddingFunc.calls = []
    hf_calls = []

    def fake_hf(embedding_dim, embedding_model=None):
        hf_calls.append(
            {"embedding_dim": embedding_dim, "embedding_model": embedding_model}
        )
        return SimpleNamespace(kind="hf")

    def rag_ctor(**kwargs):
        return _CapturingRAG(**kwargs)

    with _enter_reingest_stack(rag_ctor=rag_ctor, make_hf=fake_hf):
        await reingest.main()

    assert _RecordingEmbeddingFunc.calls == []
    assert hf_calls == [
        {"embedding_dim": 1024, "embedding_model": "BAAI/bge-m3"},
    ]
