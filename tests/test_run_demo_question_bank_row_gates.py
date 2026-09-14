"""Demo question-bank row loop: empty skip, --limit, None answers, timing log.

Extends #133 credential/schema gates. Empty or whitespace questions must not
call the LLM or consume --limit. A None aquery result must persist as "" so
downstream xlsx/jsonl rows stay aligned.
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
        "run_demo_question_bank_row_gates", SCRIPT_PATH
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
    def __init__(self, answers=None, **kwargs):
        self.aquery_calls = []
        self._answers = list(answers or [])

    async def aquery(self, question, mode="mix", **kwargs):
        self.aquery_calls.append({"question": question, "mode": mode, **kwargs})
        if self._answers:
            return self._answers.pop(0)
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
        demo_script, "ensure_hf_home_from_repo_fallback", lambda *a, **k: None
    )


@pytest.mark.asyncio
async def test_empty_and_whitespace_questions_skip_and_do_not_consume_limit(
    demo_script, monkeypatch, tmp_path
):
    workbook = _Workbook(
        ["问题", "机型"],
        [
            [None, "skip-none"],
            ["   ", "skip-ws"],
            ["first real", "m1"],
            ["second real", "m2"],
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
        1,
        None,
    )

    assert [c["question"] for c in rag.aquery_calls] == ["first real"]
    assert rag.aquery_calls[0]["vlm_enhanced"] is False
    assert workbook.saved_to == out_xlsx
    records = [
        json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["问题"] == "first real"
    assert records[0]["RAG回答"] == "ans:first real"
    # 问题 is column 1; RAG回答 is appended as column 3
    assert workbook.active.written[(4, 3)] == "ans:first real"


@pytest.mark.asyncio
async def test_none_aquery_result_persists_as_empty_string(
    demo_script, monkeypatch, tmp_path
):
    workbook = _Workbook(["问题"], [["needs empty answer"]])
    _install_openpyxl(monkeypatch, workbook)
    rag = _FakeRAGAnything(answers=[None])
    _stub_demo_runtime(demo_script, monkeypatch, rag)

    out_jsonl = tmp_path / "out.jsonl"
    await demo_script.run_batch(
        tmp_path / "in.xlsx",
        tmp_path / "out.xlsx",
        out_jsonl,
        tmp_path / "wd",
        "hybrid",
        0.0,
        1,
        1,
        0,
        None,
    )

    assert rag.aquery_calls[0]["mode"] == "hybrid"
    records = [
        json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["RAG回答"] == ""
    assert workbook.active.written[(2, 2)] == ""


@pytest.mark.asyncio
async def test_timing_log_appends_one_json_line_per_answered_question(
    demo_script, monkeypatch, tmp_path
):
    workbook = _Workbook(["问题"], [["timed q"]])
    _install_openpyxl(monkeypatch, workbook)
    rag = _FakeRAGAnything()
    _stub_demo_runtime(demo_script, monkeypatch, rag)

    timing_log = tmp_path / "logs" / "qa.jsonl"
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
        timing_log,
    )

    raw = timing_log.read_text(encoding="utf-8").splitlines()
    lines = [json.loads(line) for line in raw]
    assert len(lines) == 1
    assert lines[0]["row"] == 2
    assert lines[0]["question_chars"] == len("timed q")
    assert isinstance(lines[0]["query_seconds"], float)
    assert lines[0]["query_seconds"] >= 0
