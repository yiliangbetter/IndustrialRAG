"""Pipeline chat and one-shot query must not enable VLM by default.

``scripts/rag_pipeline_parse_graph_chat.py`` is the production parse→graph→Q&A
entrypoint. Interactive and ``--query`` paths must pass ``vlm_enhanced=False``
so retrieved ``Image Path:`` lines are not base64-encoded and sent to a vision
model. A regression here turns every plant-manual question into a VLM call
(or fails closed when no vision function is configured).
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
        "rag_pipeline_parse_graph_chat_vlm_gate", path
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
        return "ok"

    async def finalize_storages(self):
        return None


@pytest.mark.asyncio
async def test_interactive_loop_forwards_vlm_enhanced_false(pipeline, monkeypatch):
    rag = FakeRAG()
    answers = iter(["What is the torque spec?", "exit"])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    await pipeline._interactive_loop(rag, query_mode="mix")

    assert len(rag.aquery_calls) == 1
    call = rag.aquery_calls[0]
    assert call["query"] == "What is the torque spec?"
    assert call["mode"] == "mix"
    assert call["vlm_enhanced"] is False


@pytest.mark.asyncio
async def test_async_main_oneshot_query_disables_vlm(pipeline, monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    rag = FakeRAG()

    async def fake_build_rag(working_dir, parser_output_dir):
        return rag, SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        return 1, 0

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
            "  How do I bleed the brakes?  ",
            "--query-mode",
            "hybrid",
        ],
    )

    await pipeline.async_main()

    assert len(rag.aquery_calls) == 1
    call = rag.aquery_calls[0]
    assert call["query"] == "How do I bleed the brakes?"
    assert call["mode"] == "hybrid"
    assert call["vlm_enhanced"] is False
