"""Pipeline CLI must honor --no-recursive and RAG_QUERY_MODE.

``scripts/rag_pipeline_parse_graph_chat.py`` is the parse→graph→Q&A path.
Nested plant manuals must stay out of ingest when ``--no-recursive`` is set.
Omitting ``--query-mode`` must read ``RAG_QUERY_MODE`` (default ``mix``) so
operators do not silently switch from mix to naive/local retrieval.

Distinct from #130 (``_collect_files`` helper), #164 (``_ingest_folder``
isolation with an explicit recursive flag), and #174 (explicit
``--query-mode hybrid`` plus ``vlm_enhanced=False``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_KEYS = ("RAG_QUERY_MODE", "PARSE_METHOD", "PARSER")


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_no_recursive_query_mode_065a", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeRAG:
    def __init__(self):
        self.parse_calls = []
        self.insert_calls = []
        self.aquery_calls = []

    async def parse_document(
        self,
        file_path,
        output_dir=None,
        parse_method=None,
        display_stats=None,
        **kwargs,
    ):
        self.parse_calls.append(Path(file_path).name)
        return [
            {"type": "text", "text": Path(file_path).name}
        ], f"doc-{Path(file_path).stem}"

    async def insert_content_list(self, content_list, file_path=None, **kwargs):
        self.insert_calls.append(file_path)

    async def aquery(self, query, mode="mix", **kwargs):
        self.aquery_calls.append({"query": query, "mode": mode, **kwargs})
        return "ok"

    async def finalize_storages(self):
        return None


def _config():
    return SimpleNamespace(
        parser="mineru",
        supported_file_extensions=[".pdf"],
        display_content_stats=False,
    )


@pytest.mark.asyncio
async def test_async_main_no_recursive_skips_nested_pdfs(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    nested = docs / "plant-a"
    nested.mkdir(parents=True)
    (docs / "top.pdf").write_bytes(b"%PDF")
    (nested / "nested.pdf").write_bytes(b"%PDF")
    rag = FakeRAG()

    async def fake_build_rag(working_dir, parser_output_dir):
        return rag, _config(), FakeLogger()

    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
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
            "--no-recursive",
            "--ingest-only",
        ],
    )

    await pipeline.async_main()

    assert rag.parse_calls == ["top.pdf"]
    assert rag.insert_calls == ["top.pdf"]


@pytest.mark.asyncio
async def test_async_main_query_mode_defaults_to_mix_when_env_unset(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    rag = FakeRAG()

    async def fake_build_rag(working_dir, parser_output_dir):
        return rag, _config(), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        return 1, 0

    monkeypatch.delenv("RAG_QUERY_MODE", raising=False)
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
            "--query",
            "What is the bleed procedure?",
        ],
    )

    await pipeline.async_main()

    assert len(rag.aquery_calls) == 1
    assert rag.aquery_calls[0]["mode"] == "mix"
    assert rag.aquery_calls[0]["vlm_enhanced"] is False


@pytest.mark.asyncio
async def test_async_main_omitted_query_mode_reads_rag_query_mode_env(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    rag = FakeRAG()

    async def fake_build_rag(working_dir, parser_output_dir):
        return rag, _config(), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        return 1, 0

    monkeypatch.setenv("RAG_QUERY_MODE", "naive")
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
            "--query",
            "torque spec",
        ],
    )

    await pipeline.async_main()

    assert rag.aquery_calls[0]["mode"] == "naive"
    assert rag.aquery_calls[0]["query"] == "torque spec"
