"""Pipeline ``--ingest-only`` must not start Q&A after graph ingest.

Overnight parse→graph jobs pass ``--ingest-only`` (sometimes with a leftover
``--query``). Querying anyway would call the LLM on every cron run, or fail
closed when no vision/LLM path is intended. Invalid ``MINERU_MODEL_SOURCE``
must also remap to huggingface so ``mineru-models-download`` is not pointed
at an unknown hub.

Distinct from #174 (``vlm_enhanced=False`` on chat / one-shot) and from
#130/#164 (folder/key gates and ingest isolation).
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
        "rag_pipeline_parse_graph_chat_ingest_only", path
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
        self.finalize_calls = 0

    async def aquery(self, query, mode="mix", **kwargs):
        self.aquery_calls.append({"query": query, "mode": mode, **kwargs})
        return "should-not-run"

    async def finalize_storages(self):
        self.finalize_calls += 1


@pytest.mark.asyncio
async def test_async_main_ingest_only_skips_query_and_chat(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    rag = FakeRAG()
    interactive_calls = []

    async def fake_build_rag(working_dir, parser_output_dir):
        return rag, SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        return 1, 0

    async def fake_interactive_loop(*args, **kwargs):
        interactive_calls.append(True)

    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
    monkeypatch.setattr(pipeline, "_ingest_folder", fake_ingest_folder)
    monkeypatch.setattr(pipeline, "_interactive_loop", fake_interactive_loop)
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
            "--query",
            "What is the torque spec?",
        ],
    )

    await pipeline.async_main()

    assert rag.finalize_calls == 1
    assert rag.aquery_calls == []
    assert interactive_calls == []


def test_download_models_unknown_source_remaps_to_huggingface(pipeline, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return None

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "not-a-hub")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline._download_mineru_pipeline_models()

    assert calls == [["mineru-models-download", "-s", "huggingface", "-m", "pipeline"]]


def test_download_models_modelscope_source_is_preserved(pipeline, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return None

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "ModelScope")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline._download_mineru_pipeline_models()

    assert calls[0][2] == "modelscope"
