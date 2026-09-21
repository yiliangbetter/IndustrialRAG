"""Graph JSON ingest omitted ``-w`` must keep the ``rag_storage_with_kg`` store.

``scripts/batch_ingest_content_lists_with_graph.py`` defaults LightRAG
persistence to ``<repo>/rag_storage_with_kg``. Changing that default to
``rag_storage`` would merge KG ingest into the demo Q&A index. Distinct from
#191 (PARSER env) and #192 (pipeline parse-output dir / local-HF PARSER).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    "LLM_MODEL",
    "VISION_MODEL",
    "PARSER",
    "PARSE_METHOD",
    "RAG_DATA_REPO",
    "RAG_DATA_UPLOAD_SUBDIR",
    "HF_HOME",
    "WORKING_DIR",
)


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_omitted_working_dir_8a56", SCRIPT_PATH
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


class _CapturingRAG:
    last_config = None

    def __init__(self, **kwargs):
        type(self).last_config = kwargs.get("config")

    async def finalize_storages(self):
        return None


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


@pytest.mark.asyncio
async def test_omitted_working_dir_uses_repo_rag_storage_with_kg(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("WORKING_DIR", raising=False)
    monkeypatch.setattr(bigraph, "_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "--data-repo-root",
            str(repo),
        ],
    )
    _CapturingRAG.last_config = None
    _CapturingLightRAG.last_kwargs = None

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock()),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("raganything.RAGAnything", _CapturingRAG),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
        patch(
            "raganything.local_hf_embedding.make_local_hf_embedding_func", MagicMock()
        ),
    ):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()

    expected = str(tmp_path / "rag_storage_with_kg")
    assert _CapturingLightRAG.last_kwargs["working_dir"] == expected
    assert _CapturingRAG.last_config is not None
    assert _CapturingRAG.last_config.working_dir == expected
