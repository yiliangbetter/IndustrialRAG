"""Graph JSON ingest must honor PARSER env at RAGAnything construction.

``scripts/batch_ingest_content_lists_with_graph.py`` does not parse files, but
``RAGAnything.__post_init__`` still calls ``get_parser(config.parser)``. Ignoring
``PARSER`` (or defaulting away from mineru) fails KG ingest at startup when the
operator swapped MinerU for Docling/PaddleOCR. Distinct from #185 (pipeline
PARSER / reingest PARSER / graph LLM_MODEL).
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
)


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_parser_env_8875", SCRIPT_PATH
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


class _FakeLightRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def initialize_storages(self):
        return None


class _RecordingEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        pass


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _empty_upload_tree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    return repo


async def _run_graph_until_constructed(bigraph, monkeypatch, tmp_path):
    repo = _empty_upload_tree(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
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
    _CapturingRAG.last_config = None
    with (
        patch("lightrag.LightRAG", _FakeLightRAG),
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
    assert _CapturingRAG.last_config is not None
    return _CapturingRAG.last_config


@pytest.mark.asyncio
async def test_graph_ingest_parser_defaults_to_mineru(bigraph, monkeypatch, tmp_path):
    monkeypatch.delenv("PARSER", raising=False)
    monkeypatch.delenv("PARSE_METHOD", raising=False)

    config = await _run_graph_until_constructed(bigraph, monkeypatch, tmp_path)

    assert config.parser == "mineru"
    assert config.parse_method == "auto"


@pytest.mark.asyncio
async def test_graph_ingest_parser_env_is_forwarded(bigraph, monkeypatch, tmp_path):
    monkeypatch.setenv("PARSER", "paddleocr")
    # PARSE_METHOD must not leak into this JSON path (parse_method is hardcoded).
    monkeypatch.setenv("PARSE_METHOD", "ocr")

    config = await _run_graph_until_constructed(bigraph, monkeypatch, tmp_path)

    assert config.parser == "paddleocr"
    assert config.parse_method == "auto"
