"""Pipeline ``--parser-output-dir`` must reach config and MinerU parse output.

``scripts/rag_pipeline_parse_graph_chat.py`` isolates parse artifacts from
LightRAG storage. Omitting the flag must use ``<repo>/output/pipeline_parse``.
A CLI path must be resolved, created, written onto ``RAGAnythingConfig``, and
forwarded to ``_ingest_folder`` so overnight jobs do not clobber a sibling
parse tree or mix MinerU JSON across plants.

Distinct from #164 (``_ingest_folder`` nested dirs given an explicit path),
#181 (OCR re-ingest ``WORKING_DIR`` / ``OUTPUT_DIR``), and #185 (PARSER /
PARSE_METHOD on the same ``_build_rag`` config).
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
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "PARSER",
    "PARSE_METHOD",
    "OUTPUT_DIR",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parser_output_dir_d920", path
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


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeRAG:
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


class _CapturingRAGAnything:
    last_config = None

    def __init__(self, **kwargs):
        type(self).last_config = kwargs.get("config")


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


async def _run_ingest_only(pipeline, monkeypatch, tmp_path, extra_argv):
    docs = tmp_path / "docs"
    docs.mkdir()
    rag = FakeRAG()
    built = []
    ingested = []

    async def fake_build_rag(working_dir, parser_output_dir):
        built.append(
            {"working_dir": working_dir, "parser_output_dir": parser_output_dir}
        )
        return rag, SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        ingested.append(kwargs)
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
            *extra_argv,
            "--ingest-only",
        ],
    )
    await pipeline.async_main()
    return built, ingested


@pytest.mark.asyncio
async def test_omitted_parser_output_dir_uses_repo_pipeline_parse(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setattr(pipeline, "_ROOT", tmp_path)
    built, ingested = await _run_ingest_only(
        pipeline, monkeypatch, tmp_path, extra_argv=[]
    )

    expected = (tmp_path / "output" / "pipeline_parse").resolve()
    assert built[0]["parser_output_dir"] == expected
    assert ingested[0]["parser_output_dir"] == expected
    assert expected.is_dir()


@pytest.mark.asyncio
async def test_parser_output_dir_cli_is_resolved_created_and_forwarded(
    pipeline, monkeypatch, tmp_path
):
    custom = tmp_path / "plants" / "parse-out"
    assert not custom.exists()

    built, ingested = await _run_ingest_only(
        pipeline,
        monkeypatch,
        tmp_path,
        extra_argv=["--parser-output-dir", str(custom)],
    )

    expected = custom.resolve()
    assert built[0]["parser_output_dir"] == expected
    assert ingested[0]["parser_output_dir"] == expected
    assert expected.is_dir()


@pytest.mark.asyncio
async def test_build_rag_writes_parser_output_dir_onto_config(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pipeline")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # OUTPUT_DIR must not replace the explicit _build_rag argument.
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "env-output"))
    parse_out = tmp_path / "explicit-parse"
    _CapturingLightRAG.last_kwargs = None
    _CapturingRAGAnything.last_config = None

    with (
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", AsyncMock()),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", _CapturingRAGAnything),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        _rag, config, _logger = await pipeline._build_rag(tmp_path / "wd", parse_out)

    assert config.parser_output_dir == str(parse_out)
    assert _CapturingRAGAnything.last_config is config
    assert config.parser_output_dir != str(tmp_path / "env-output")
