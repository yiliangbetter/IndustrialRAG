"""Pipeline and graph ingest must keep LightRAG embedding concurrency at 1.

OCR re-ingest defaults async=8 / batch=10 (#183). The parse→graph and
JSON→KG CLIs default to 1/1 so entity extraction does not stampede the
embedding host. Copying the OCR defaults here rate-limits overnight jobs
and can interleave embeddings across manuals.

Distinct from #179/#180 (processor flags and embed dims) and #183
(reingest concurrency defaults).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "HF_HOME",
    "RAG_DATA_REPO",
    "RAG_DATA_UPLOAD_SUBDIR",
)


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_embedding_concurrency",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "graph_ingest_embedding_concurrency",
        REPO_ROOT / "scripts" / "batch_ingest_content_lists_with_graph.py",
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
    def __init__(self, embedding_dim, max_token_size, func):
        pass


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _clear_concurrency_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_FUNC_MAX_ASYNC", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_NUM", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _CapturingLightRAG.last_kwargs = None


@pytest.mark.asyncio
async def test_pipeline_defaults_embedding_concurrency_to_one(
    pipeline, monkeypatch, tmp_path
):
    _clear_concurrency_env(monkeypatch)

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock()),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", lambda **kwargs: SimpleNamespace()),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        await pipeline._build_rag(tmp_path / "wd", tmp_path / "out")

    assert _CapturingLightRAG.last_kwargs["embedding_func_max_async"] == 1
    assert _CapturingLightRAG.last_kwargs["embedding_batch_num"] == 1


@pytest.mark.asyncio
async def test_pipeline_forwards_embedding_concurrency_env(
    pipeline, monkeypatch, tmp_path
):
    _clear_concurrency_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "3")
    monkeypatch.setenv("EMBEDDING_BATCH_NUM", "4")

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock()),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", lambda **kwargs: SimpleNamespace()),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        await pipeline._build_rag(tmp_path / "wd", tmp_path / "out")

    assert _CapturingLightRAG.last_kwargs["embedding_func_max_async"] == 3
    assert _CapturingLightRAG.last_kwargs["embedding_batch_num"] == 4


@pytest.mark.asyncio
async def test_graph_ingest_defaults_embedding_concurrency_to_one(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    _clear_concurrency_env(monkeypatch)
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

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock()),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", lambda **kwargs: SimpleNamespace()),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()

    assert _CapturingLightRAG.last_kwargs["embedding_func_max_async"] == 1
    assert _CapturingLightRAG.last_kwargs["embedding_batch_num"] == 1


@pytest.mark.asyncio
async def test_graph_ingest_forwards_embedding_concurrency_env(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    _clear_concurrency_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "5")
    monkeypatch.setenv("EMBEDDING_BATCH_NUM", "6")
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

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock()),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", lambda **kwargs: SimpleNamespace()),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()

    assert _CapturingLightRAG.last_kwargs["embedding_func_max_async"] == 5
    assert _CapturingLightRAG.last_kwargs["embedding_batch_num"] == 6
