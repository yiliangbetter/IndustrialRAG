"""Demo question-bank CLI must honor RAG_QUERY_MODE and LLM_MODEL.

``scripts/run_demo_question_bank.py`` is the operator path for the plant
Q&A sheet. Omitting ``--mode`` must read ``RAG_QUERY_MODE`` (default
``mix``). Text LLM closures must use ``LLM_MODEL`` (default
``gpt-4o-mini``). Distinct from #133/#178 (row/credential gates), #183
(embed dims), and #184 (VISION_MODEL routing with mode hardcoded to mix).
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
    "RAG_QUERY_MODE",
    "DEMO_QA_QUERY_DELAY_SECONDS",
    "EMBEDDING_FUNC_MAX_ASYNC",
    "EMBEDDING_BATCH_NUM",
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_query_mode_llm_065a", SCRIPT_PATH
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
    last_kwargs = None
    aquery_calls = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        type(self).aquery_calls = []

    async def aquery(self, question, mode="mix", **kwargs):
        type(self).aquery_calls.append({"query": question, "mode": mode, **kwargs})
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


def _patch_demo_runtime(demo_script, monkeypatch, *, complete_mock):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-demo")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _install_openpyxl(monkeypatch)
    _CapturingLightRAG.last_kwargs = None
    _FakeRAGAnything.last_kwargs = None
    _FakeRAGAnything.aquery_calls = []
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


def test_demo_main_query_mode_defaults_to_mix(demo_script, monkeypatch, tmp_path):
    captured = {}

    async def fake_run_batch(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.delenv("RAG_QUERY_MODE", raising=False)
    monkeypatch.setattr(demo_script, "run_batch", fake_run_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_demo_question_bank",
            "--input",
            str(tmp_path / "in.xlsx"),
            "--out-xlsx",
            str(tmp_path / "out.xlsx"),
            "--out-jsonl",
            str(tmp_path / "out.jsonl"),
            "-w",
            str(tmp_path / "wd"),
        ],
    )

    demo_script.main()

    assert captured["args"][4] == "mix"


def test_demo_main_omitted_mode_reads_rag_query_mode_env(
    demo_script, monkeypatch, tmp_path
):
    captured = {}

    async def fake_run_batch(*args, **kwargs):
        captured["args"] = args

    monkeypatch.setenv("RAG_QUERY_MODE", "naive")
    monkeypatch.setattr(demo_script, "run_batch", fake_run_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_demo_question_bank",
            "--input",
            str(tmp_path / "in.xlsx"),
            "--out-xlsx",
            str(tmp_path / "out.xlsx"),
            "--out-jsonl",
            str(tmp_path / "out.jsonl"),
            "-w",
            str(tmp_path / "wd"),
        ],
    )

    demo_script.main()

    assert captured["args"][4] == "naive"


@pytest.mark.asyncio
async def test_demo_llm_model_defaults_to_gpt_4o_mini(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    complete_mock = MagicMock(return_value="ok")
    _patch_demo_runtime(demo_script, monkeypatch, complete_mock=complete_mock)

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
    _CapturingLightRAG.last_kwargs["llm_model_func"]("answer the question")
    complete_mock.assert_called_once()
    assert complete_mock.call_args.args[0] == "gpt-4o-mini"
    assert _FakeRAGAnything.aquery_calls[0]["mode"] == "mix"
    assert _FakeRAGAnything.aquery_calls[0]["vlm_enhanced"] is False


@pytest.mark.asyncio
async def test_demo_llm_model_env_is_forwarded(demo_script, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    complete_mock = MagicMock(return_value="ok")
    _patch_demo_runtime(demo_script, monkeypatch, complete_mock=complete_mock)

    await demo_script.run_batch(
        tmp_path / "in.xlsx",
        tmp_path / "out.xlsx",
        tmp_path / "out.jsonl",
        tmp_path / "wd",
        "hybrid",
        0.0,
        1,
        1,
        0,
        None,
    )

    _CapturingLightRAG.last_kwargs["llm_model_func"]("answer the question")
    assert complete_mock.call_args.args[0] == "deepseek-chat"
    assert _FakeRAGAnything.aquery_calls[0]["mode"] == "hybrid"
