"""Demo Q&A must honor OPENAI_BASE_URL when LLM_BINDING_HOST is blank.

Industrial OpenAI-compatible gateways typically set OPENAI_BASE_URL, while
LightRAG reads LLM_BINDING_HOST. scripts/run_demo_question_bank.py is the
operator Q&A path against the shared vector store; dropping the fallback
silently sends queries to api.openai.com (or the SDK default) instead of
the configured gateway. EMBEDDING_BINDING_HOST must still win for embeddings.

Complements #182 (graph ingest + pipeline ``_build_rag``). Distinct from
#133 (credentials / embed-host fail-closed) and #183 (dim/model defaults).
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
    "HF_HOME",
    "PARSER",
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_openai_base_url_b17b", SCRIPT_PATH
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


class _CapturingLightRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def initialize_storages(self):
        return None


class _FakeRAGAnything:
    def __init__(self, **kwargs):
        pass

    async def aquery(self, question, mode="mix", **kwargs):
        return f"ans:{question}"


class _RecordingEmbeddingFunc:
    last_func = None

    def __init__(self, embedding_dim, max_token_size, func):
        type(self).last_func = func


def _install_openpyxl(monkeypatch):
    fake = types.ModuleType("openpyxl")
    fake.load_workbook = lambda path: _Workbook(["问题"], [["q1"]])
    monkeypatch.setitem(sys.modules, "openpyxl", fake)


def _patch_demo_ctors(demo_script, monkeypatch, *, complete_mock):
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
    monkeypatch.setattr(demo_script, "openai_complete_if_cache", complete_mock)


async def _run_until_constructed(demo_script, monkeypatch, tmp_path, *, complete_mock):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-demo")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    _install_openpyxl(monkeypatch)
    _CapturingLightRAG.last_kwargs = None
    _RecordingEmbeddingFunc.last_func = None
    _patch_demo_ctors(demo_script, monkeypatch, complete_mock=complete_mock)

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

    assert _CapturingLightRAG.last_kwargs is not None
    return _CapturingLightRAG.last_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("binding_host", [None, "", "   "])
async def test_openai_base_url_used_when_llm_binding_host_blank(
    demo_script, monkeypatch, tmp_path, binding_host
):
    gateway = "http://demo-gateway.internal/v1"
    if binding_host is None:
        monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    else:
        monkeypatch.setenv("LLM_BINDING_HOST", binding_host)
    monkeypatch.setenv("OPENAI_BASE_URL", gateway)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    complete_mock = MagicMock(return_value="ok")
    kwargs = await _run_until_constructed(
        demo_script, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    kwargs["llm_model_func"]("answer the question")
    complete_mock.assert_called_once()
    assert complete_mock.call_args.kwargs["base_url"] == gateway
    assert complete_mock.call_args.kwargs["api_key"] == "sk-demo"

    embed_func = _RecordingEmbeddingFunc.last_func
    assert embed_func is not None
    assert embed_func.keywords["base_url"] == gateway


@pytest.mark.asyncio
async def test_llm_binding_host_wins_over_openai_base_url(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_BINDING_HOST", "http://lightrag-host/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://should-not-win/v1")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    complete_mock = MagicMock(return_value="ok")
    kwargs = await _run_until_constructed(
        demo_script, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    kwargs["llm_model_func"]("prompt")
    assert complete_mock.call_args.kwargs["base_url"] == "http://lightrag-host/v1"
    assert _RecordingEmbeddingFunc.last_func.keywords["base_url"] == (
        "http://lightrag-host/v1"
    )


@pytest.mark.asyncio
async def test_embedding_binding_host_wins_for_embeddings_only(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm-gateway/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://embed-gateway/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-embed")

    complete_mock = MagicMock(return_value="ok")
    kwargs = await _run_until_constructed(
        demo_script, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    kwargs["llm_model_func"]("prompt")
    assert complete_mock.call_args.kwargs["base_url"] == "http://llm-gateway/v1"
    assert _RecordingEmbeddingFunc.last_func.keywords["base_url"] == (
        "http://embed-gateway/v1"
    )
    assert _RecordingEmbeddingFunc.last_func.keywords["api_key"] == "sk-embed"
