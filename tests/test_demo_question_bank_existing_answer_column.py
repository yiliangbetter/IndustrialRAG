"""Question-bank reruns overwrite the existing answer column.

A sheet that already has ``RAG回答`` must not grow a second answer column.
A numeric ``0`` is a real question (falsy, but not blank text) and is sent
as ``"0"``. Distinct from the append-column row gates and from non-finite
JSONL export.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_demo_question_bank.py"
)
_STUBBED_MODULES = (
    "dotenv",
    "lightrag",
    "lightrag.llm",
    "lightrag.llm.openai",
    "lightrag.utils",
    "raganything",
    "raganything.local_hf_embedding",
)


def _install_import_stubs():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    lightrag = types.ModuleType("lightrag")
    lightrag.LightRAG = object
    sys.modules["lightrag"] = lightrag

    llm_pkg = types.ModuleType("lightrag.llm")
    sys.modules["lightrag.llm"] = llm_pkg
    llm_openai = types.ModuleType("lightrag.llm.openai")
    llm_openai.openai_complete_if_cache = lambda *args, **kwargs: None
    llm_openai.openai_embed = types.SimpleNamespace(func=lambda *args, **kwargs: None)
    sys.modules["lightrag.llm.openai"] = llm_openai

    utils = types.ModuleType("lightrag.utils")
    utils.EmbeddingFunc = lambda **kwargs: object()
    utils.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    sys.modules["lightrag.utils"] = utils

    raganything = types.ModuleType("raganything")
    raganything.RAGAnything = object
    raganything.RAGAnythingConfig = lambda **kwargs: object()
    sys.modules["raganything"] = raganything

    local_hf = types.ModuleType("raganything.local_hf_embedding")
    local_hf.ensure_hf_home_from_repo_fallback = lambda *args, **kwargs: None
    local_hf.make_local_hf_embedding_func = lambda *args, **kwargs: object()
    sys.modules["raganything.local_hf_embedding"] = local_hf


def _load_script():
    saved_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    saved_path = list(sys.path)
    try:
        _install_import_stubs()
        spec = importlib.util.spec_from_file_location(
            "run_demo_question_bank_existing_answer_767b", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        for name, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture(scope="module")
def demo_script():
    return _load_script()


@pytest.fixture(autouse=True)
def _restore_openpyxl_module():
    before = sys.modules.get("openpyxl")
    yield
    if before is None:
        sys.modules.pop("openpyxl", None)
    else:
        sys.modules["openpyxl"] = before


class _Cell:
    def __init__(self, value, row=1):
        self.value = value
        self.row = row


class _Worksheet:
    def __init__(self, headers, data_rows):
        self.headers = list(headers)
        self.data_rows = list(data_rows)
        self.written = {}

    def iter_rows(self, min_row=1, max_row=None, values_only=False):
        if min_row == 1:
            yield [_Cell(header, row=1) for header in self.headers]
            return
        for offset, values in enumerate(self.data_rows):
            row_num = offset + 2
            yield [_Cell(value, row=row_num) for value in values]

    def cell(self, row, column, value=None):
        self.written[(row, column)] = value
        if row == 1:
            while len(self.headers) < column:
                self.headers.append(None)
            self.headers[column - 1] = value
        return _Cell(value, row=row)


class _Workbook:
    def __init__(self, headers, data_rows):
        self.active = _Worksheet(headers, data_rows)
        self.saved_to = None

    def save(self, path):
        self.saved_to = path


class _FakeLightRAG:
    def __init__(self, **kwargs):
        pass

    async def initialize_storages(self):
        return None


class _FakeRAGAnything:
    def __init__(self, **kwargs):
        self.aquery_calls = []

    async def aquery(self, question, mode="mix", **kwargs):
        self.aquery_calls.append({"question": question, "mode": mode, **kwargs})
        return f"ans:{question}"


def _install_openpyxl(monkeypatch, workbook):
    fake = types.ModuleType("openpyxl")
    fake.load_workbook = lambda path: workbook
    monkeypatch.setitem(sys.modules, "openpyxl", fake)


def _stub_demo_runtime(demo_script, monkeypatch, rag):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(demo_script, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(demo_script, "RAGAnything", lambda **kwargs: rag)
    monkeypatch.setattr(demo_script, "RAGAnythingConfig", lambda **kwargs: object())
    monkeypatch.setattr(demo_script, "EmbeddingFunc", lambda **kwargs: object())
    monkeypatch.setattr(
        demo_script,
        "openai_embed",
        types.SimpleNamespace(func=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        demo_script, "ensure_hf_home_from_repo_fallback", lambda *args, **kwargs: None
    )


@pytest.mark.asyncio
async def test_existing_answer_column_is_overwritten_and_zero_is_asked(
    demo_script, monkeypatch, tmp_path
):
    headers = ["问题", "RAG回答", "备注"]
    workbook = _Workbook(
        headers,
        [
            [0, "stale-zero", "keep-zero"],
            ["   ", "stale-blank", "keep-blank"],
            ["Torque spec", "stale-text", "keep-text"],
        ],
    )
    _install_openpyxl(monkeypatch, workbook)
    rag = _FakeRAGAnything()
    _stub_demo_runtime(demo_script, monkeypatch, rag)

    out_xlsx = tmp_path / "out.xlsx"
    out_jsonl = tmp_path / "out.jsonl"
    await demo_script.run_batch(
        tmp_path / "in.xlsx",
        out_xlsx,
        out_jsonl,
        tmp_path / "wd",
        "mix",
        0.0,
        1,
        1,
        0,
        None,
    )

    assert [call["question"] for call in rag.aquery_calls] == ["0", "Torque spec"]
    assert all(call["vlm_enhanced"] is False for call in rag.aquery_calls)
    assert workbook.active.headers == ["问题", "RAG回答", "备注"]
    assert workbook.active.written == {
        (2, 2): "ans:0",
        (4, 2): "ans:Torque spec",
    }
    assert workbook.saved_to == out_xlsx

    records = [
        json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {"问题": 0, "RAG回答": "ans:0", "备注": "keep-zero"},
        {"问题": "Torque spec", "RAG回答": "ans:Torque spec", "备注": "keep-text"},
    ]
