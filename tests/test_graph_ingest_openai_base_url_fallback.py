"""Graph ingest must honor OPENAI_BASE_URL when LLM_BINDING_HOST is blank.

Industrial OpenAI-compatible gateways typically set OPENAI_BASE_URL, while
LightRAG reads LLM_BINDING_HOST. scripts/batch_ingest_content_lists_with_graph.py
is the text-first KG path; dropping the fallback silently sends extraction
traffic to api.openai.com (or the SDK default) instead of the configured
gateway. EMBEDDING_BINDING_HOST must still win for embeddings.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "batch_ingest_content_lists_with_graph.py"

_ENV_KEYS = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BACKEND",
    "EMBEDDING_BINDING_HOST",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "RAG_DATA_REPO",
    "RAG_DATA_UPLOAD_SUBDIR",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_openai_base_url_8bd3", SCRIPT_PATH
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
        self.initialize_storages = AsyncMock()


class _RecordingEmbeddingFunc:
    last_func = None

    def __init__(self, embedding_dim, max_token_size, func):
        type(self).last_func = func


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _empty_data_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    return repo


async def _run_until_constructed(bigraph, monkeypatch, tmp_path, *, complete_mock):
    repo = _empty_data_repo(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
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
    _RecordingEmbeddingFunc.last_func = None

    rag = MagicMock()
    rag.insert_content_list = AsyncMock()
    rag.finalize_storages = AsyncMock()

    with (
        patch("lightrag.llm.openai.openai_complete_if_cache", complete_mock),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("raganything.RAGAnything", return_value=rag),
    ):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()

    assert _CapturingLightRAG.last_kwargs is not None
    return _CapturingLightRAG.last_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("binding_host", [None, "", "   "])
async def test_openai_base_url_used_when_llm_binding_host_blank(
    bigraph, monkeypatch, tmp_path, binding_host
):
    gateway = "http://llm-gateway.internal/v1"
    if binding_host is None:
        monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    else:
        monkeypatch.setenv("LLM_BINDING_HOST", binding_host)
    monkeypatch.setenv("OPENAI_BASE_URL", gateway)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _run_until_constructed(
        bigraph, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await kwargs["llm_model_func"]("extract entities")
    complete_mock.assert_awaited_once()
    assert complete_mock.await_args.kwargs["base_url"] == gateway
    assert complete_mock.await_args.kwargs["api_key"] == "sk-llm"

    embed_func = _RecordingEmbeddingFunc.last_func
    assert embed_func is not None
    assert embed_func.keywords["base_url"] == gateway


@pytest.mark.asyncio
async def test_llm_binding_host_wins_over_openai_base_url(
    bigraph, monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_BINDING_HOST", "http://lightrag-host/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://should-not-win/v1")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _run_until_constructed(
        bigraph, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await kwargs["llm_model_func"]("prompt")
    assert complete_mock.await_args.kwargs["base_url"] == "http://lightrag-host/v1"
    assert _RecordingEmbeddingFunc.last_func.keywords["base_url"] == (
        "http://lightrag-host/v1"
    )


@pytest.mark.asyncio
async def test_embedding_binding_host_wins_for_embeddings_only(
    bigraph, monkeypatch, tmp_path
):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm-gateway/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://embed-gateway/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-embed")

    complete_mock = AsyncMock(return_value="ok")
    kwargs = await _run_until_constructed(
        bigraph, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await kwargs["llm_model_func"]("prompt")
    assert complete_mock.await_args.kwargs["base_url"] == "http://llm-gateway/v1"
    assert _RecordingEmbeddingFunc.last_func.keywords["base_url"] == (
        "http://embed-gateway/v1"
    )
    assert _RecordingEmbeddingFunc.last_func.keywords["api_key"] == "sk-embed"
