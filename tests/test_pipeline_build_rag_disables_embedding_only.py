"""Pipeline RAG construction must stay text-first and pick the right embed dims.

``scripts/rag_pipeline_parse_graph_chat.py`` is the production parse→graph
entrypoint. Enabling embedding-only ingestion or image/table/equation
processors would skip KG extraction or fire vision/table models for every
manual. OpenAI vs HF default dims (1536 vs 1024) must stay aligned with
the chosen backend so vectors are not written into a mismatched store.

Distinct from #130 (API-key / embedding-host fail-closed gates).
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_build_rag_config_44c5", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_embedding_env():
    keys = (
        "OPENAI_API_KEY",
        "LLM_BINDING_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_BINDING_HOST",
        "EMBEDDING_DIM",
        "EMBEDDING_MODEL",
        "PARSER",
        "PARSE_METHOD",
        "MAX_CONCURRENT_FILES",
        "LLM_MODEL",
        "VISION_MODEL",
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
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def initialize_storages(self):
        return None


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


@contextmanager
def _patched_build_stack(
    *,
    rag_ctor,
    embedding_func_cls,
    make_hf=None,
):
    hf_target = make_hf if make_hf is not None else MagicMock()
    with (
        patch("lightrag.LightRAG", _FakeLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock()),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", embedding_func_cls),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", rag_ctor),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
        patch(
            "raganything.local_hf_embedding.make_local_hf_embedding_func",
            hf_target,
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_build_rag_disables_embedding_only_and_modal_processors(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pipeline")
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    captured = {}

    class CapturingRAGAnything:
        def __init__(self, **kwargs):
            captured["rag_kwargs"] = kwargs

    class UnusedEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            captured["embedding_dim"] = embedding_dim

    with _patched_build_stack(
        rag_ctor=CapturingRAGAnything,
        embedding_func_cls=UnusedEmbeddingFunc,
    ):
        rag, config, _logger = await pipeline._build_rag(
            tmp_path / "wd", tmp_path / "out"
        )

    assert rag is not None
    assert config.allow_embedding_only_ingestion is False
    assert config.enable_image_processing is False
    assert config.enable_table_processing is False
    assert config.enable_equation_processing is False
    assert config.parser_output_dir == str(tmp_path / "out")
    assert captured["rag_kwargs"]["config"] is config


@pytest.mark.asyncio
async def test_build_rag_openai_embed_defaults_to_1536_text_embedding_3_small(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pipeline")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    seen_emb = {}
    hf_mock = MagicMock()

    class RecordingEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            seen_emb["embedding_dim"] = embedding_dim
            seen_emb["max_token_size"] = max_token_size

    with _patched_build_stack(
        rag_ctor=lambda **kwargs: SimpleNamespace(),
        embedding_func_cls=RecordingEmbeddingFunc,
        make_hf=hf_mock,
    ):
        await pipeline._build_rag(tmp_path / "wd", tmp_path / "out")

    hf_mock.assert_not_called()
    assert seen_emb["embedding_dim"] == 1536
    assert seen_emb["max_token_size"] == 8192


@pytest.mark.asyncio
async def test_build_rag_hf_backend_uses_bge_m3_dim_1024(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pipeline")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://unused.example")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    hf_calls = []
    openai_emb_calls = []

    class RecordingEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            openai_emb_calls.append(embedding_dim)

    def fake_hf_embed(embedding_dim, embedding_model=None):
        hf_calls.append(
            {"embedding_dim": embedding_dim, "embedding_model": embedding_model}
        )
        return SimpleNamespace(kind="hf")

    with _patched_build_stack(
        rag_ctor=lambda **kwargs: SimpleNamespace(),
        embedding_func_cls=RecordingEmbeddingFunc,
        make_hf=fake_hf_embed,
    ):
        await pipeline._build_rag(tmp_path / "wd", tmp_path / "out")

    assert openai_emb_calls == []
    assert hf_calls == [
        {"embedding_dim": 1024, "embedding_model": "BAAI/bge-m3"},
    ]
