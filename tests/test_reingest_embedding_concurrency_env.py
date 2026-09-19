"""OCR re-ingest LightRAG concurrency must honor embedding env overrides.

``scripts/reingest_uploaded_documents_ocr.py`` defaults to
``EMBEDDING_FUNC_MAX_ASYNC=8`` / ``EMBEDDING_BATCH_NUM=10`` (#183). Operators
lower those env vars to avoid Ark/OpenAI rate limits on overnight OCR.
Ignoring the env keeps the aggressive defaults and stampede the host.

Distinct from #183 (default 8/10) and #188 (pipeline/graph default 1/1).
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
    "EMBEDDING_FUNC_MAX_ASYNC",
    "EMBEDDING_BATCH_NUM",
    "WORKING_DIR",
    "OUTPUT_DIR",
    "PARSER",
    "HF_HOME",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
)


@pytest.fixture(scope="module")
def reingest():
    path = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_embedding_concurrency_env_bd82", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _enter_reingest_stack(*, light_rag_cls, rag_ctor):
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", light_rag_cls))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock())
    )
    stack.enter_context(patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()))
    stack.enter_context(patch("lightrag.utils.EmbeddingFunc", MagicMock()))
    stack.enter_context(
        patch("lightrag.utils.logger", SimpleNamespace(info=lambda *a, **k: None))
    )
    stack.enter_context(patch("raganything.RAGAnything", rag_ctor))
    stack.enter_context(
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback")
    )
    stack.enter_context(
        patch(
            "raganything.local_hf_embedding.make_local_hf_embedding_func", MagicMock()
        )
    )
    return stack


def _prepare(monkeypatch, tmp_path):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF")
    out = tmp_path / "out"
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
            str(out),
            "--skip-model-download",
        ],
    )


@pytest.mark.asyncio
async def test_reingest_forwards_embedding_concurrency_env(
    reingest, monkeypatch, tmp_path
):
    _prepare(monkeypatch, tmp_path)
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "2")
    monkeypatch.setenv("EMBEDDING_BATCH_NUM", "3")

    captured = {}

    class RecordingLightRAG:
        def __init__(self, **kwargs):
            captured["lightrag_kwargs"] = kwargs

        async def initialize_storages(self):
            return None

    class CapturingRAG:
        def __init__(self, **kwargs):
            self.process_document_complete = AsyncMock()

    with _enter_reingest_stack(light_rag_cls=RecordingLightRAG, rag_ctor=CapturingRAG):
        await reingest.main()

    assert captured["lightrag_kwargs"]["embedding_func_max_async"] == 2
    assert captured["lightrag_kwargs"]["embedding_batch_num"] == 3
