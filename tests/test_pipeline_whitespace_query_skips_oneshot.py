"""Pipeline whitespace ``--query`` must not oneshot; it starts interactive chat.

``scripts/rag_pipeline_parse_graph_chat.py`` treats ``args.query.strip()`` as
the oneshot gate. A cron/CI invocation with ``--query '   '`` (or tabs) would
otherwise send blank text to LightRAG — or, if the strip check is dropped
from the ``if`` but not the call, hang in ``input()`` on unattended jobs.
#174 covers padded *non-empty* oneshot stripping; #176 covers ``--ingest-only``
skipping query entirely; #177 covers empty/quit *inside* the interactive loop.
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
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_whitespace_query_8875",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
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
    def __init__(self):
        self.aquery_calls = []

    async def aquery(self, query, mode="mix", **kwargs):
        self.aquery_calls.append({"query": query, "mode": mode, **kwargs})
        return "should-not-run"

    async def finalize_storages(self):
        return None


async def _run_after_ingest(pipeline, monkeypatch, tmp_path, extra_argv):
    docs = tmp_path / "docs"
    docs.mkdir()
    rag = FakeRAG()
    interactive = []

    async def fake_build_rag(working_dir, parser_output_dir):
        return rag, SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        return 1, 0

    async def fake_interactive(rag_arg, query_mode):
        interactive.append({"rag": rag_arg, "query_mode": query_mode})

    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
    monkeypatch.setattr(pipeline, "_ingest_folder", fake_ingest_folder)
    monkeypatch.setattr(pipeline, "_interactive_loop", fake_interactive)
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
            *extra_argv,
        ],
    )
    await pipeline.async_main()
    return rag, interactive


@pytest.mark.asyncio
async def test_whitespace_query_starts_interactive_instead_of_oneshot(
    pipeline, monkeypatch, tmp_path
):
    rag, interactive = await _run_after_ingest(
        pipeline,
        monkeypatch,
        tmp_path,
        extra_argv=["--query", " \t  ", "--query-mode", "hybrid"],
    )

    assert rag.aquery_calls == []
    assert interactive == [{"rag": rag, "query_mode": "hybrid"}]


@pytest.mark.asyncio
async def test_omitted_query_starts_interactive_loop(pipeline, monkeypatch, tmp_path):
    rag, interactive = await _run_after_ingest(
        pipeline, monkeypatch, tmp_path, extra_argv=["--query-mode", "naive"]
    )

    assert rag.aquery_calls == []
    assert interactive == [{"rag": rag, "query_mode": "naive"}]
