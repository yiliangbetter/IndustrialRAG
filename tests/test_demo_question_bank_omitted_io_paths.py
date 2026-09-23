"""Demo Q&A omitted spreadsheet paths must not overwrite the source workbook.

``scripts/run_demo_question_bank.py`` reads ``question/Demo问题库.xlsx`` and
writes answers to ``Demo问题库_回答结果.xlsx`` / ``.jsonl``. Swapping those
defaults would answer the wrong sheet or overwrite the question bank. Distinct
from #194 (omitted ``-w`` → ``rag_storage``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_demo_question_bank.py"
)


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_omitted_io_paths", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_script():
    return _load_demo_module()


def _capture_run_batch(demo_script, monkeypatch):
    captured = {}

    async def fake_run_batch(
        input_xlsx,
        out_xlsx,
        out_jsonl,
        working_dir,
        mode,
        delay_seconds,
        embedding_func_max_async,
        embedding_batch_num,
        limit_questions,
        timing_log,
    ):
        captured["input"] = input_xlsx
        captured["out_xlsx"] = out_xlsx
        captured["out_jsonl"] = out_jsonl
        captured["working_dir"] = working_dir

    monkeypatch.setattr(demo_script, "run_batch", fake_run_batch)
    return captured


def test_omitted_io_paths_use_question_bank_defaults(
    demo_script, monkeypatch, tmp_path
):
    captured = _capture_run_batch(demo_script, monkeypatch)
    monkeypatch.setattr(demo_script, "_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_demo_question_bank"])

    demo_script.main()

    question_dir = tmp_path / "question"
    assert captured["input"] == question_dir / "Demo问题库.xlsx"
    assert captured["out_xlsx"] == question_dir / "Demo问题库_回答结果.xlsx"
    assert captured["out_jsonl"] == question_dir / "Demo问题库_回答结果.jsonl"
    assert captured["input"] != captured["out_xlsx"]


def test_explicit_io_paths_override_question_bank_defaults(
    demo_script, monkeypatch, tmp_path
):
    captured = _capture_run_batch(demo_script, monkeypatch)
    monkeypatch.setattr(demo_script, "_ROOT", tmp_path)
    src = tmp_path / "custom" / "questions.xlsx"
    out_xlsx = tmp_path / "custom" / "answers.xlsx"
    out_jsonl = tmp_path / "custom" / "answers.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_demo_question_bank",
            "--input",
            str(src),
            "--out-xlsx",
            str(out_xlsx),
            "--out-jsonl",
            str(out_jsonl),
        ],
    )

    demo_script.main()

    assert captured["input"] == src
    assert captured["out_xlsx"] == out_xlsx
    assert captured["out_jsonl"] == out_jsonl
    assert captured["working_dir"] == tmp_path / "rag_storage"
