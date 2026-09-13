"""Pipeline chat must survive a failed query, and model download is opt-in.

A single aquery exception must not abort the interactive loop. Empty /
quit input must end the session without calling the LLM. --mineru-download-models
must run before ingest; omitting it must not download weights.

Distinct from #174 (vlm_enhanced=False) and #176 (--ingest-only / source remap).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_chat_exit_and_download", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeRAG:
    def __init__(self, fail_first=False):
        self.aquery_calls = []
        self.fail_first = fail_first

    async def aquery(self, query, mode="mix", **kwargs):
        self.aquery_calls.append({"query": query, "mode": mode, **kwargs})
        if self.fail_first and len(self.aquery_calls) == 1:
            raise RuntimeError("llm timeout")
        return f"answer:{query}"

    async def finalize_storages(self):
        return None


@pytest.mark.asyncio
async def test_interactive_empty_line_exits_without_query(pipeline, monkeypatch):
    rag = FakeRAG()
    answers = iter(["   ", "should-not-run"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    await pipeline._interactive_loop(rag, query_mode="mix")

    assert rag.aquery_calls == []


@pytest.mark.asyncio
async def test_interactive_quit_exits_without_query(pipeline, monkeypatch):
    rag = FakeRAG()
    answers = iter(["QUIT"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    await pipeline._interactive_loop(rag, query_mode="hybrid")

    assert rag.aquery_calls == []


@pytest.mark.asyncio
async def test_interactive_query_error_continues_to_next_question(
    pipeline, monkeypatch
):
    rag = FakeRAG(fail_first=True)
    answers = iter(["first question", "second question", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    await pipeline._interactive_loop(rag, query_mode="mix")

    assert [c["query"] for c in rag.aquery_calls] == [
        "first question",
        "second question",
    ]


@pytest.mark.asyncio
async def test_async_main_download_flag_runs_before_ingest(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(b"%PDF")
    rag = FakeRAG()
    order = []

    def fake_download():
        order.append("download")

    async def fake_build_rag(working_dir, parser_output_dir):
        order.append("build")
        return rag, SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        order.append("ingest")
        return 1, 0

    monkeypatch.setattr(pipeline, "_download_mineru_pipeline_models", fake_download)
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
            "--mineru-download-models",
        ],
    )

    await pipeline.async_main()

    assert order == ["download", "build", "ingest"]


@pytest.mark.asyncio
async def test_async_main_skips_download_without_flag(pipeline, monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(b"%PDF")
    rag = FakeRAG()
    download_calls = []

    def fake_download():
        download_calls.append(True)

    async def fake_build_rag(working_dir, parser_output_dir):
        return rag, SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        return 1, 0

    monkeypatch.setattr(pipeline, "_download_mineru_pipeline_models", fake_download)
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

    assert download_calls == []
