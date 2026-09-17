"""Pipeline graph extraction must honor LLM_MODEL, PARSER, and PARSE_METHOD env.

``scripts/rag_pipeline_parse_graph_chat.py`` is the production parse→graph path.
A wrong LLM_MODEL silently extracts entities with the default chat model. PARSER
selects MinerU vs Docling/PaddleOCR. Omitting ``--parse-method`` must still pick
up PARSE_METHOD so overnight jobs do not fall back to auto OCR. Distinct from
#179 (processor flags / embed dims), #182 (gateway host), and #164 (explicit
parse_method forwarding into ``_ingest_folder``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
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
    "LLM_MODEL",
    "VISION_MODEL",
    "PARSER",
    "PARSE_METHOD",
    "MAX_CONCURRENT_FILES",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_llm_model_parser_d37f", path
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


class _CapturingRAGAnything:
    last_config = None

    def __init__(self, **kwargs):
        type(self).last_config = kwargs.get("config")


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


async def _build(pipeline, monkeypatch, tmp_path, *, complete_mock):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pipeline")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _CapturingLightRAG.last_kwargs = None
    _CapturingRAGAnything.last_config = None

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", complete_mock),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", _CapturingRAGAnything),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        rag, config, _logger = await pipeline._build_rag(
            tmp_path / "wd", tmp_path / "out"
        )

    assert _CapturingLightRAG.last_kwargs is not None
    return rag, config, _CapturingLightRAG.last_kwargs


@pytest.mark.asyncio
async def test_llm_model_defaults_to_gpt_4o_mini(pipeline, monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    complete_mock = AsyncMock(return_value="ok")
    _rag, _config, kwargs = await _build(
        pipeline, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await kwargs["llm_model_func"]("extract entities")
    complete_mock.assert_awaited_once()
    assert complete_mock.await_args.args[0] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_llm_model_env_is_forwarded_to_openai_complete(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-32b")
    complete_mock = AsyncMock(return_value="ok")
    _rag, _config, kwargs = await _build(
        pipeline, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await kwargs["llm_model_func"]("extract entities")
    assert complete_mock.await_args.args[0] == "qwen2.5-32b"


@pytest.mark.asyncio
async def test_parser_and_parse_method_env_set_rag_config(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("PARSER", "docling")
    monkeypatch.setenv("PARSE_METHOD", "ocr")
    complete_mock = AsyncMock(return_value="ok")
    _rag, config, _kwargs = await _build(
        pipeline, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    assert config.parser == "docling"
    assert config.parse_method == "ocr"
    assert _CapturingRAGAnything.last_config is config


@pytest.mark.asyncio
async def test_omitted_parse_method_flag_uses_parse_method_env(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setenv("PARSE_METHOD", "txt")
    captured = {}

    class FakeRAG:
        async def finalize_storages(self):
            return None

    async def fake_build_rag(working_dir, parser_output_dir):
        return FakeRAG(), SimpleNamespace(parser="mineru"), SimpleNamespace()

    async def fake_ingest_folder(*args, **kwargs):
        captured.update(kwargs)
        return 1, 0

    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
    monkeypatch.setattr(pipeline, "_ingest_folder", fake_ingest_folder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_pipeline_parse_graph_chat",
            "--input-folder",
            str(docs),
            "-w",
            str(tmp_path / "wd"),
            "--parser-output-dir",
            str(tmp_path / "out"),
            "--ingest-only",
        ],
    )

    await pipeline.async_main()

    assert captured["parse_method"] == "txt"
