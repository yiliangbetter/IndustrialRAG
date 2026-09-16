"""Pipeline RAG construction must honor OPENAI_BASE_URL when LLM_BINDING_HOST is blank.

scripts/rag_pipeline_parse_graph_chat.py is the production parse→graph path.
OpenAI-compatible clients set OPENAI_BASE_URL; LightRAG uses LLM_BINDING_HOST.
A missing fallback sends graph extraction to the SDK default instead of the
configured gateway. Distinct from #179 (processor flags / embed dims) and #130
(API-key / embedding-host fail-closed gates).
"""

from __future__ import annotations

import importlib.util
import os
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
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_openai_base_url_8bd3", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    last_func = None

    def __init__(self, embedding_dim, max_token_size, func):
        type(self).last_func = func


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


async def _build(pipeline, monkeypatch, tmp_path, *, complete_mock):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pipeline")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    _CapturingLightRAG.last_kwargs = None
    _RecordingEmbeddingFunc.last_func = None

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", complete_mock),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", lambda **kwargs: SimpleNamespace()),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        await pipeline._build_rag(tmp_path / "wd", tmp_path / "out")

    assert _CapturingLightRAG.last_kwargs is not None
    return _CapturingLightRAG.last_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("binding_host", [None, "", "   "])
async def test_openai_base_url_used_when_llm_binding_host_blank(
    pipeline, monkeypatch, tmp_path, binding_host
):
    gateway = "http://pipeline-gateway.internal/v1"
    if binding_host is None:
        monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    else:
        monkeypatch.setenv("LLM_BINDING_HOST", binding_host)
    monkeypatch.setenv("OPENAI_BASE_URL", gateway)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _build(pipeline, monkeypatch, tmp_path, complete_mock=complete_mock)

    await kwargs["llm_model_func"]("extract")
    complete_mock.assert_awaited_once()
    assert complete_mock.await_args.kwargs["base_url"] == gateway
    assert complete_mock.await_args.kwargs["api_key"] == "sk-pipeline"
    assert _RecordingEmbeddingFunc.last_func.keywords["base_url"] == gateway


@pytest.mark.asyncio
async def test_llm_binding_host_wins_over_openai_base_url(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_BINDING_HOST", "http://lightrag-host/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://should-not-win/v1")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _build(pipeline, monkeypatch, tmp_path, complete_mock=complete_mock)

    await kwargs["llm_model_func"]("prompt")
    assert complete_mock.await_args.kwargs["base_url"] == "http://lightrag-host/v1"
    assert _RecordingEmbeddingFunc.last_func.keywords["base_url"] == (
        "http://lightrag-host/v1"
    )


@pytest.mark.asyncio
async def test_embedding_binding_host_wins_for_embeddings_only(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm-gateway/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://embed-gateway/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-embed")

    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _build(pipeline, monkeypatch, tmp_path, complete_mock=complete_mock)

    await kwargs["llm_model_func"]("prompt")
    assert complete_mock.await_args.kwargs["base_url"] == "http://llm-gateway/v1"
    assert _RecordingEmbeddingFunc.last_func.keywords["base_url"] == (
        "http://embed-gateway/v1"
    )
    assert _RecordingEmbeddingFunc.last_func.keywords["api_key"] == "sk-embed"
