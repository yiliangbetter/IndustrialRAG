"""Pipeline omitted ``-w`` must keep the ``rag_storage_pipeline`` store.

``scripts/rag_pipeline_parse_graph_chat.py`` defaults LightRAG persistence to
``<repo>/rag_storage_pipeline``. Changing that default to ``rag_storage`` would
clobber the demo Q&A index. Distinct from #192 (``--parser-output-dir`` defaults
and CLI isolation) and #181 (OCR re-ingest ``WORKING_DIR`` env).
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
        "rag_pipeline_omitted_working_dir_8a56",
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
    async def finalize_storages(self):
        return None


@pytest.mark.asyncio
async def test_omitted_working_dir_uses_repo_rag_storage_pipeline(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    captured = {}

    async def fake_build_rag(working_dir, parser_output_dir):
        captured["working_dir"] = working_dir
        captured["parser_output_dir"] = parser_output_dir
        return FakeRAG(), SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        return 1, 0

    monkeypatch.setattr(pipeline, "_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
    monkeypatch.setattr(pipeline, "_ingest_folder", fake_ingest_folder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_pipeline_parse_graph_chat",
            "--input-folder",
            str(docs),
            "--parser-output-dir",
            str(tmp_path / "out"),
            "--ingest-only",
        ],
    )

    await pipeline.async_main()

    expected = (tmp_path / "rag_storage_pipeline").resolve()
    assert captured["working_dir"] == expected
    assert captured["parser_output_dir"] == (tmp_path / "out").resolve()
