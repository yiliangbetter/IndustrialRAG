"""Graph batch ingest must stay text-first and pick the right embed dims.

``scripts/batch_ingest_content_lists_with_graph.py`` builds the KG from MinerU
JSON. Enabling image/table/equation processors would fire vision models on a
text-first job (especially with ``--no-skip-multimodal``). OpenAI vs HF default
dims (1536 vs 1024) must stay aligned with the backend so vectors are not
written into a mismatched store.

Distinct from #127 (credential gates, skip-multimodal forwarding, embedding-only
False) and #179 (pipeline ``_build_rag`` flags).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "batch_ingest_content_lists_with_graph.py"


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_modal_processors_1c11", SCRIPT_PATH
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
        "LLM_MODEL",
        "VISION_MODEL",
        "LLM_BINDING_HOST",
        "OPENAI_BASE_URL",
        "HF_HOME",
        "RAG_DATA_REPO",
        "RAG_DATA_UPLOAD_SUBDIR",
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
def _patched_ingest_stack(*, rag_ctor, embedding_func_cls, make_hf=None):
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


def _argv(working_dir: Path, data_repo: Path):
    return [
        "batch_ingest_content_lists_with_graph",
        "-w",
        str(working_dir),
        "--data-repo-root",
        str(data_repo),
    ]


def _empty_upload_tree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    return repo


@pytest.mark.asyncio
async def test_graph_ingest_disables_modal_processors(bigraph, monkeypatch, tmp_path):
    repo = _empty_upload_tree(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path / "wd", repo))

    captured = {}

    class CapturingRAGAnything:
        def __init__(self, **kwargs):
            captured["rag_kwargs"] = kwargs

    class RecordingEmbeddingFunc:
        def __init__(self, embedding_dim, max_token_size, func):
            captured["embedding_dim"] = embedding_dim

    with _patched_ingest_stack(
        rag_ctor=CapturingRAGAnything,
        embedding_func_cls=RecordingEmbeddingFunc,
    ):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()

    config = captured["rag_kwargs"]["config"]
    assert config.allow_embedding_only_ingestion is False
    assert config.enable_image_processing is False
    assert config.enable_table_processing is False
    assert config.enable_equation_processing is False
    assert captured["embedding_dim"] == 1536


@pytest.mark.asyncio
async def test_graph_ingest_hf_backend_uses_bge_m3_dim_1024(
    bigraph, monkeypatch, tmp_path
):
    repo = _empty_upload_tree(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://unused.example")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path / "wd", repo))

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

    with _patched_ingest_stack(
        rag_ctor=lambda **kwargs: SimpleNamespace(),
        embedding_func_cls=RecordingEmbeddingFunc,
        make_hf=fake_hf_embed,
    ):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()

    assert openai_emb_calls == []
    assert hf_calls == [
        {"embedding_dim": 1024, "embedding_model": "BAAI/bge-m3"},
    ]
