"""Demo question-bank must honor PARSER env at RAGAnything construction.

``scripts/run_demo_question_bank.py`` does not parse files, but
``RAGAnything.__post_init__`` still calls ``get_parser(config.parser)``. Ignoring
``PARSER`` fails the plant Q&A sheet at startup when operators swapped MinerU
for Docling/PaddleOCR. ``PARSE_METHOD`` must not leak (``parse_method`` stays
hardcoded ``auto``).

Distinct from #185 (pipeline PARSER / reingest PARSER), #186 (query mode /
LLM_MODEL), and #191 (graph-ingest PARSER).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_demo_question_bank.py"

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
    "HF_HOME",
    "PARSER",
    "PARSE_METHOD",
    "RAG_QUERY_MODE",
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_parser_env_d920", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env_and_openpyxl():
    before_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    before_openpyxl = sys.modules.get("openpyxl")
    yield
    for key, val in before_env.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    if before_openpyxl is None:
        sys.modules.pop("openpyxl", None)
    else:
        sys.modules["openpyxl"] = before_openpyxl


class _CapturingLightRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def initialize_storages(self):
        return None


class _FakeRAGAnything:
    last_config = None

    def __init__(self, **kwargs):
        type(self).last_config = kwargs.get("config")

    async def aquery(self, question, mode="mix", **kwargs):
        return f"ans:{question}"


class _RecordingEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        pass


class _Cell:
    def __init__(self, value, row=1):
        self.value = value
        self.row = row


class _Worksheet:
    def __init__(self, headers, data_rows):
        self.headers = list(headers)
        self.data_rows = list(data_rows)

    def iter_rows(self, min_row=1, max_row=None, values_only=False):
        if min_row == 1:
            yield [_Cell(h, row=1) for h in self.headers]
            return
        for offset, values in enumerate(self.data_rows):
            row_num = offset + 2
            yield [_Cell(v, row=row_num) for v in values]

    def cell(self, row, column, value=None):
        return _Cell(value, row=row)


class _Workbook:
    def __init__(self, headers, data_rows):
        self.active = _Worksheet(headers, data_rows)

    def save(self, path):
        self.saved_to = path


def _install_openpyxl(monkeypatch):
    fake = types.ModuleType("openpyxl")
    fake.load_workbook = lambda path: _Workbook(["问题"], [["bleed procedure"]])
    monkeypatch.setitem(sys.modules, "openpyxl", fake)


def _patch_demo_runtime(demo_script, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-demo")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _install_openpyxl(monkeypatch)
    _CapturingLightRAG.last_kwargs = None
    _FakeRAGAnything.last_config = None
    monkeypatch.setattr(demo_script, "LightRAG", _CapturingLightRAG)
    monkeypatch.setattr(demo_script, "RAGAnything", _FakeRAGAnything)
    monkeypatch.setattr(demo_script, "EmbeddingFunc", _RecordingEmbeddingFunc)
    monkeypatch.setattr(
        demo_script, "ensure_hf_home_from_repo_fallback", lambda *a, **k: None
    )
    monkeypatch.setattr(
        demo_script,
        "openai_embed",
        types.SimpleNamespace(func=MagicMock(name="openai_embed_func")),
    )
    monkeypatch.setattr(demo_script, "openai_complete_if_cache", MagicMock())


async def _run_demo(demo_script, monkeypatch, tmp_path):
    _patch_demo_runtime(demo_script, monkeypatch)
    await demo_script.run_batch(
        tmp_path / "in.xlsx",
        tmp_path / "out.xlsx",
        tmp_path / "out.jsonl",
        tmp_path / "wd",
        "mix",
        0.0,
        1,
        1,
        0,
        None,
    )
    assert _FakeRAGAnything.last_config is not None
    return _FakeRAGAnything.last_config


@pytest.mark.asyncio
async def test_demo_parser_defaults_to_mineru(demo_script, monkeypatch, tmp_path):
    monkeypatch.delenv("PARSER", raising=False)
    monkeypatch.delenv("PARSE_METHOD", raising=False)

    config = await _run_demo(demo_script, monkeypatch, tmp_path)

    assert config.parser == "mineru"
    assert config.parse_method == "auto"


@pytest.mark.asyncio
async def test_demo_parser_env_is_forwarded(demo_script, monkeypatch, tmp_path):
    monkeypatch.setenv("PARSER", "docling")
    # PARSE_METHOD must not leak into this Q&A path (parse_method is hardcoded).
    monkeypatch.setenv("PARSE_METHOD", "ocr")

    config = await _run_demo(demo_script, monkeypatch, tmp_path)

    assert config.parser == "docling"
    assert config.parse_method == "auto"
