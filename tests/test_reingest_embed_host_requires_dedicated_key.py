"""OCR re-ingest must not send the LLM key to a separate embedding host.

``scripts/reingest_uploaded_documents_ocr.py`` refuses non-HF embedding when
``EMBEDDING_BINDING_HOST`` is set without a dedicated ``EMBEDDING_API_KEY``.
A missing or whitespace-only key must exit before LightRAG is constructed.
When a dedicated key is present, that key and host are bound into the
embedding call, not the LLM credentials.

Distinct from pipeline ``_build_rag`` (#130), graph ingest (#127), and the
demo question bank (#133), which use different entrypoints and messages.
HF re-ingest skipping this gate is covered by #183.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"

_ENV_KEYS = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BACKEND",
    "EMBEDDING_BINDING_HOST",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EMBEDDING_FUNC_MAX_ASYNC",
    "EMBEDDING_BATCH_NUM",
    "HF_HOME",
    "WORKING_DIR",
    "OUTPUT_DIR",
    "HF_HUB_OFFLINE",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_FORCE_ONLINE",
    "NO_PROXY",
    "no_proxy",
    "PATH",
    "PARSER",
)


@pytest.fixture(scope="module")
def reingest():
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_embed_host_key_c393", SCRIPT_PATH
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


class _CapturingLightRAG:
    kwargs = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs

    async def initialize_storages(self):
        return None


class _FakeRAGAnything:
    def __init__(self, **kwargs):
        pass


class _RecordingEmbeddingFunc:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


def _prepare(monkeypatch, tmp_path, *, embedding_api_key):
    folder = tmp_path / "uploaded"
    folder.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "should-not-win")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embed.example/v1")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://llm.example/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    if embedding_api_key is None:
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_API_KEY", embedding_api_key)
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
    return folder


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding_api_key", [None, "   ", "\t"])
async def test_reingest_exits_when_embed_host_has_no_dedicated_key(
    reingest, monkeypatch, tmp_path, embedding_api_key
):
    _prepare(monkeypatch, tmp_path, embedding_api_key=embedding_api_key)
    _CapturingLightRAG.kwargs = None

    def _must_not_build(*args, **kwargs):
        raise AssertionError("LightRAG must not be built without an embedding key")

    monkeypatch.setattr("lightrag.LightRAG", _must_not_build)
    monkeypatch.setattr("raganything.RAGAnything", _FakeRAGAnything)
    monkeypatch.setattr("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc)
    openai_embed = MagicMock()
    openai_embed.func = MagicMock(name="openai_embed_func")
    monkeypatch.setattr("lightrag.llm.openai.openai_embed", openai_embed)
    monkeypatch.setattr(
        "lightrag.llm.openai.openai_complete_if_cache", MagicMock(name="complete")
    )

    with pytest.raises(SystemExit, match="EMBEDDING_API_KEY"):
        await reingest.main()

    assert _CapturingLightRAG.kwargs is None


@pytest.mark.asyncio
async def test_reingest_binds_dedicated_embed_key_and_host(
    reingest, monkeypatch, tmp_path
):
    _prepare(monkeypatch, tmp_path, embedding_api_key="embed-key")
    _CapturingLightRAG.kwargs = None
    monkeypatch.setattr("lightrag.LightRAG", _CapturingLightRAG)
    monkeypatch.setattr("raganything.RAGAnything", _FakeRAGAnything)
    monkeypatch.setattr("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc)
    openai_embed = MagicMock()
    openai_embed.func = MagicMock(name="openai_embed_func")
    monkeypatch.setattr("lightrag.llm.openai.openai_embed", openai_embed)
    monkeypatch.setattr(
        "lightrag.llm.openai.openai_complete_if_cache", MagicMock(name="complete")
    )

    with pytest.raises(SystemExit, match="No supported files"):
        await reingest.main()

    assert _CapturingLightRAG.kwargs is not None
    embedding_func = _CapturingLightRAG.kwargs["embedding_func"]
    bound = embedding_func.kwargs["func"]
    assert isinstance(bound, partial)
    assert bound.func is openai_embed.func
    assert bound.keywords["api_key"] == "embed-key"
    assert bound.keywords["base_url"] == "https://embed.example/v1"
    assert bound.keywords["model"] == "text-embedding-3-small"
    assert "llm-key" not in bound.keywords.values()
    assert "https://llm.example/v1" not in bound.keywords.values()
