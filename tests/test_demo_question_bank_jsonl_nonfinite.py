"""Demo question-bank JSONL must stay strict JSON when cells are non-finite.

Excel often stores blank numeric cells as NaN, and a formula error can surface
as Infinity. ``json.dumps`` emits those as ``NaN`` / ``Infinity``, which strict
parsers reject. ``run_batch`` must write ``null`` for those cells and leave
ordinary numbers and text unchanged.
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


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_jsonl_nonfinite", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
            yield [_Cell(h, row=1) for h in self.headers]
            return
        for offset, values in enumerate(self.data_rows):
            row_num = offset + 2
            yield [_Cell(v, row=row_num) for v in values]

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
        pass


class _FakeRAGAnything:
    def __init__(self, **kwargs):
        self.aquery_calls = []

    async def aquery(self, question, mode="mix", **kwargs):
        self.aquery_calls.append(question)
        return "grounded answer"


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
        demo_script, "ensure_hf_home_from_repo_fallback", lambda *a, **k: None
    )


def test_json_safe_nulls_only_nonfinite_floats(demo_script):
    assert demo_script._json_safe(float("nan")) is None
    assert demo_script._json_safe(float("inf")) is None
    assert demo_script._json_safe(float("-inf")) is None
    assert demo_script._json_safe(1.25) == 1.25
    assert demo_script._json_safe(2) == 2
    assert demo_script._json_safe("ok") == "ok"
    assert demo_script._json_safe(None) is None


@pytest.mark.asyncio
async def test_run_batch_jsonl_replaces_nonfinite_cells_with_null(
    demo_script, monkeypatch, tmp_path
):
    workbook = _Workbook(
        ["问题", "分数", "上限", "备注"],
        [["what is torque", float("nan"), float("-inf"), 1.25]],
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

    assert rag.aquery_calls == ["what is torque"]
    assert workbook.saved_to == out_xlsx
    # 问题 is column 1; RAG回答 is appended as column 5.
    assert workbook.active.written[(2, 5)] == "grounded answer"

    raw = out_jsonl.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    record = json.loads(raw)
    assert record["问题"] == "what is torque"
    assert record["分数"] is None
    assert record["上限"] is None
    assert record["备注"] == 1.25
    assert record["RAG回答"] == "grounded answer"
