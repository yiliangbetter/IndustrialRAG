"""Demo Q&A must pick OpenAI 1536-d vs HF 1024-d embedding defaults.

``scripts/run_demo_question_bank.py`` queries whatever store the ingest
scripts wrote. Defaulting OpenAI to 1024-d (or HF to 1536-d / the wrong
model id) poisons retrieval against an existing index.

#133/#178 cover credentials, empty rows, None answers, and timing logs —
not backend-specific dim/model wiring. Distinct from #179 pipeline
``_build_rag`` and #180 graph-ingest dim tests.
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
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "HF_HOME",
    "PARSER",
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_embedding_defaults_afad", SCRIPT_PATH
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


class _FakeLightRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def initialize_storages(self):
        return None


class _FakeRAGAnything:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def aquery(self, question, mode="mix", **kwargs):
        return f"ans:{question}"


class _RecordingEmbeddingFunc:
    calls = []

    def __init__(self, embedding_dim, max_token_size, func):
        type(self).calls.append(
            {
                "embedding_dim": embedding_dim,
                "max_token_size": max_token_size,
                "model": getattr(func, "keywords", {}).get("model"),
            }
        )


def _install_openpyxl(monkeypatch, workbook):
    fake = types.ModuleType("openpyxl")
    fake.load_workbook = lambda path: workbook
    monkeypatch.setitem(sys.modules, "openpyxl", fake)


def _patch_demo_ctors(demo_script, monkeypatch, *, embedding_func_cls, make_hf):
    monkeypatch.setattr(demo_script, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(demo_script, "RAGAnything", _FakeRAGAnything)
    monkeypatch.setattr(demo_script, "EmbeddingFunc", embedding_func_cls)
    monkeypatch.setattr(demo_script, "make_local_hf_embedding_func", make_hf)
    monkeypatch.setattr(
        demo_script, "ensure_hf_home_from_repo_fallback", lambda *a, **k: None
    )
    monkeypatch.setattr(
        demo_script,
        "openai_embed",
        types.SimpleNamespace(func=MagicMock(name="openai_embed_func")),
    )


async def _run_minimal_batch(demo_script, tmp_path):
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


@pytest.mark.asyncio
async def test_run_batch_openai_defaults_to_1536_text_embedding_3_small(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-demo")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    _install_openpyxl(monkeypatch, _Workbook(["问题"], [["q1"]]))
    _RecordingEmbeddingFunc.calls = []
    hf_mock = MagicMock()
    _patch_demo_ctors(
        demo_script,
        monkeypatch,
        embedding_func_cls=_RecordingEmbeddingFunc,
        make_hf=hf_mock,
    )

    await _run_minimal_batch(demo_script, tmp_path)

    hf_mock.assert_not_called()
    assert _RecordingEmbeddingFunc.calls == [
        {
            "embedding_dim": 1536,
            "max_token_size": 8192,
            "model": "text-embedding-3-small",
        },
    ]
    config = _FakeRAGAnything.last_kwargs["config"]
    assert config.enable_image_processing is True
    assert config.enable_table_processing is True
    assert config.enable_equation_processing is True


@pytest.mark.asyncio
async def test_run_batch_hf_defaults_to_bge_m3_dim_1024_and_skips_embed_host_key(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-demo")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://unused.example")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    _install_openpyxl(monkeypatch, _Workbook(["问题"], [["q1"]]))
    _RecordingEmbeddingFunc.calls = []
    hf_calls = []

    def fake_hf(embedding_dim, embedding_model=None):
        hf_calls.append(
            {"embedding_dim": embedding_dim, "embedding_model": embedding_model}
        )
        return types.SimpleNamespace(kind="hf")

    _patch_demo_ctors(
        demo_script,
        monkeypatch,
        embedding_func_cls=_RecordingEmbeddingFunc,
        make_hf=fake_hf,
    )

    await _run_minimal_batch(demo_script, tmp_path)

    assert _RecordingEmbeddingFunc.calls == []
    assert hf_calls == [
        {"embedding_dim": 1024, "embedding_model": "BAAI/bge-m3"},
    ]
